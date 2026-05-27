# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore[attr-defined]

    HAS_TLX = True
except ImportError:
    tlx = None
    HAS_TLX = False

from triton.language.extra.libdevice import (  # @manual=//triton:triton
    fast_dividef,
    fast_expf,
)

HAS_FAST_TANH_INSTRUCTION = (
    torch.version.cuda is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9  # >= H100
)
HAS_F32X2_INSTRUCTION = (
    torch.version.cuda is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 10  # >= B200
)

if HAS_FAST_TANH_INSTRUCTION:

    @triton.jit
    def tanh_approx_fp32(x):
        output = tl.inline_asm_elementwise(
            asm="""
            tanh.approx.f32 $0, $1;
            """,
            constraints="=r,r",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return output


if HAS_F32X2_INSTRUCTION:

    @triton.jit
    def _fma_f32x2(a, b, c):
        return tl.inline_asm_elementwise(
            """
            {
                .reg .b64 ra, rb, rc, rd;
                mov.b64 ra, { $2, $3 };
                mov.b64 rb, { $4, $5 };
                mov.b64 rc, { $6, $7 };
                fma.rn.f32x2 rd, ra, rb, rc;
                mov.b64 { $0, $1 }, rd;
            }
            """,
            "=r,=r,r,r,r,r,r,r",
            [a, b, c],
            dtype=tl.float32,
            is_pure=True,
            pack=2,
        )

    @triton.jit
    def fast_fma(a, b, c):
        return _fma_f32x2(a, b, c)

    @triton.jit
    def fast_silu(x, MULT_BY_X: tl.constexpr):
        x = x * 0.5
        if MULT_BY_X:
            return _fma_f32x2(x, tanh_approx_fp32(x), x)
        else:
            return _fma_f32x2(tanh_approx_fp32(x), 0.5, 0.5)

elif HAS_FAST_TANH_INSTRUCTION:

    @triton.jit
    def fast_fma(a, b, c):
        return a * b + c

    @triton.jit
    def fast_silu(x, MULT_BY_X: tl.constexpr):
        x = x * 0.5
        if MULT_BY_X:
            return x * (tanh_approx_fp32(x) + 1.0)
        else:
            return tanh_approx_fp32(x) * 0.5 + 0.5

else:

    @triton.jit
    def fast_fma(a, b, c):
        return a * b + c

    @triton.jit
    def fast_silu(x, MULT_BY_X: tl.constexpr):
        if MULT_BY_X:
            # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
            return fast_dividef(x, 1.0 + fast_expf(-x))
        else:
            # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
            return fast_dividef(1.0, 1.0 + fast_expf(-x))


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    buf_id = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return buf_id, phase


@triton.jit
def backward_d_silu_activation(
    dact_qk_trans,
    pT,  # pT represents sig_trans
    qk_trans,
    scale,
    valid_mask_trans,
):
    dqk_trans = dact_qk_trans * pT * (1 + qk_trans * (1 - pT)) * scale[None, :]
    dqk_trans = tl.where(valid_mask_trans, dqk_trans, 0)
    return dqk_trans


@triton.jit
def backward_d_softmax_activation(
    dact_qk_trans,
    Delta_off,
    offs_m,
    stride_mm,
    mask_m,
    pT,  # pT represents sig_trans
):
    Di = tl.load(Delta_off + offs_m * stride_mm, mask=mask_m)
    dqk_trans = pT * (dact_qk_trans - Di[None, :])
    return dqk_trans


@triton.jit
def backward_silu_activation(qk_trans, alpha, valid_mask_trans, dtype_k, scale):
    masked_alpha = tl.where(valid_mask_trans, alpha, 0.0)
    qk_trans = qk_trans * masked_alpha
    pT = fast_silu(qk_trans, MULT_BY_X=False)
    silu_trans = qk_trans * pT * scale[None, :]
    act_qk_trans = silu_trans.to(dtype_k)
    return qk_trans, act_qk_trans, pT


@triton.jit
def backward_softmax_activation(
    qk_trans, alpha, valid_mask_trans, M_off, offs_m, stride_mm, mask_m, k
):
    qk_trans = qk_trans * alpha * 1.44269504
    m = tl.load(M_off + offs_m * stride_mm, mask=mask_m)
    pT = tl.math.exp2(qk_trans - m[None, :])
    pT = tl.where(valid_mask_trans, pT, 0.0)
    act_qk_trans = pT.to(k.dtype)
    return qk_trans, act_qk_trans, pT


@triton.jit
def forward_softmax_activation_scaled_alpha(
    qk, scaled_alpha, valid_mask, m_i, acc, l_i
):
    qk = qk * scaled_alpha
    qk = tl.where(valid_mask, qk, -float("inf"))
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    act_qk = tl.math.exp2(qk)
    corr = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(act_qk, 1)
    acc = acc * corr[:, None]
    l_i = l_i * corr + l_ij
    m_i = m_ij
    return act_qk, acc, l_i, m_i


@triton.jit
def forward_softmax_activation_trans_scaled_alpha(
    qk_trans, scaled_alpha, valid_mask_trans, m_i, acc, l_i
):
    qk_trans = qk_trans * scaled_alpha
    qk_trans = tl.where(valid_mask_trans, qk_trans, -float("inf"))
    m_ij = tl.maximum(m_i, tl.max(qk_trans, 0))
    qk_trans -= m_ij[None, :]
    act_qk = tl.math.exp2(qk_trans)
    corr = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(act_qk, 0)
    acc = acc * corr[None, :]
    l_i = l_i * corr + l_ij
    m_i = m_ij
    return act_qk, acc, l_i, m_i


@triton.jit
def backward_softmax_activation_scaled_alpha(
    qk_trans, scaled_alpha, valid_mask_trans, M_off, offs_m, stride_mm, mask_m, k
):
    qk_trans = qk_trans * scaled_alpha
    m = tl.load(M_off + offs_m * stride_mm, mask=mask_m)
    pT = tl.math.exp2(qk_trans - m[None, :])
    pT = tl.where(valid_mask_trans, pT, 0.0)
    act_qk_trans = pT.to(k.dtype)
    return qk_trans, act_qk_trans, pT
