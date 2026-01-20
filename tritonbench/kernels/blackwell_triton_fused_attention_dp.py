"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from tritonbench.utils.env_utils import is_tile_enabled

from .attention_utils import WITH_MAXNREG
from .blackwell_attention_utils import (
    is_blackwell,
    is_cuda,
    is_hip,
    is_hopper,
    supports_host_descriptor,
)

FORCE_ON_DEVICE = os.getenv("FORCE_ON_DEVICE") == "1"


# Check if Triton version supports minRegAutoWS and maxRegAutoWS
# These parameters are only available in triton/tree/ws-3.5
def _supports_reg_auto_ws():
    """Check if the current Triton version supports minRegAutoWS/maxRegAutoWS"""
    try:
        # Try to create a Config with minRegAutoWS to test support
        test_config = triton.Config({}, minRegAutoWS=24, maxRegAutoWS=152)
        return True
    except (TypeError, AttributeError):
        # Parameter not supported in this Triton version
        return False


HAS_REG_AUTO_WS = _supports_reg_auto_ws()


# Check if Triton version supports pingpongAutoWS
# These parameters are only available in triton/tree/ws-3.5
def _supports_pingpong_auto_ws():
    """Check if the current Triton version supports pingpongAutoWS"""
    try:
        # Try to create a Config with minRegAutoWS to test support
        test_config = triton.Config({}, pingpongAutoWS=True)
        return True
    except (TypeError, AttributeError):
        # Parameter not supported in this Triton version
        return False


HAS_PINGPONG_AUTO_WS = _supports_pingpong_auto_ws()


@triton.jit
def _attn_fwd_subtile(
    q,
    k,
    offs_m,
    start_n,
    offs_n,
    qk_scale,
    l_i0,
    l_i1,  # used when FADD2_REDUCE is true
    m_i,
    acc,
    v,
    dtype: tl.constexpr,
    STAGE: tl.constexpr,
    SUBTILING: tl.constexpr,
    SUBTILING_P: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    qk = tl.dot(q, k)
    if STAGE == 2:
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
    else:
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        if VECT_MUL & 2:
            qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
        else:
            qk = qk * qk_scale - m_ij[:, None]

    PM: tl.constexpr = qk.shape[0]
    PN: tl.constexpr = qk.shape[1]

    if SUBTILING_P:
        qk0, qk1 = qk.reshape([PM, 2, PN // 2]).permute(0, 2, 1).split()

        p0 = tl.math.exp2(qk0)
        p0_bf16 = p0.to(dtype)
        p1 = tl.math.exp2(qk1)
        p1_bf16 = p1.to(dtype)

        p = tl.join(p0, p1).permute(0, 2, 1).reshape([PM, PN])
    else:
        p = tl.math.exp2(qk)

    # -- compute correction factor
    alpha = tl.math.exp2(m_i - m_ij)
    if not FADD2_REDUCE:
        l_ij = tl.sum(p, 1)

    # -- update output accumulator --
    BM: tl.constexpr = acc.shape[0]
    BN: tl.constexpr = acc.shape[1]

    if SUBTILING:
        acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
        if VECT_MUL & 1:
            acc0 = _mul_f32x2(acc0, alpha[:, None])
            acc1 = _mul_f32x2(acc1, alpha[:, None])
        else:
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
        acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
    else:
        acc = acc * alpha[:, None]

    # update m_i and l_i
    # place this at the end of the loop to reduce register pressure
    if FADD2_REDUCE:
        p0, p1 = p.reshape([PM, 2, PN // 2]).permute(0, 2, 1).split()
        l_ij0, l_ij1 = tl.reduce((p0, p1), axis=1, combine_fn=_reduce_fadd2)
        l_i0 = l_i0 * alpha + l_ij0
        l_i1 = l_i1 * alpha + l_ij1

    # We can potentially move these to be before updating l_ij, so the dot
    # is not blocked.
    # prepare p and v for the dot
    if not SUBTILING_P:
        p_bf16 = p.to(dtype)
    else:
        p_bf16 = tl.join(p0_bf16, p1_bf16).permute(0, 2, 1).reshape([PM, PN])
    # note that this non transposed v for FP8 is only supported on Blackwell
    acc = tl.dot(p_bf16, v, acc)
    if not FADD2_REDUCE:
        l_i0 = l_i0 * alpha + l_ij
    m_i = m_ij

    return l_i0, l_i1, m_i, acc


@triton.jit
def _attn_fwd_inner_oss_dp(
    acc0,
    acc1,
    l_i0,
    l_i0_1,
    l_i1,
    l_i1_1,
    m_i0,
    m_i1,
    q0,
    q1,  #
    desc_k,
    desc_v,  #
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m0: tl.constexpr,
    offs_m1: tl.constexpr,  #
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    SUBTILING: tl.constexpr,
    SUBTILING_P: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo

    # loop over k, v and update accumulator
    for start_n in tl.range(
        lo, hi, BLOCK_N, warp_specialize=warp_specialize, disallow_acc_multi_buffer=True
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = desc_k.load([offsetkv_y, 0]).T
        v = desc_v.load([offsetkv_y, 0])

        l_i0, l_i0_1, m_i0, acc0 = _attn_fwd_subtile(
            q0,
            k,
            offs_m0,
            start_n,
            offs_n,
            qk_scale,
            l_i0,
            l_i0_1,
            m_i0,
            acc0,
            v,
            dtype,
            STAGE,
            SUBTILING,
            SUBTILING_P,
            VECT_MUL,
            FADD2_REDUCE,
        )
        l_i1, l_i1_1, m_i1, acc1 = _attn_fwd_subtile(
            q1,
            k,
            offs_m1,
            start_n,
            offs_n,
            qk_scale,
            l_i1,
            l_i1_1,
            m_i1,
            acc1,
            v,
            dtype,
            STAGE,
            SUBTILING,
            SUBTILING_P,
            VECT_MUL,
            FADD2_REDUCE,
        )

        offsetkv_y += BLOCK_N

    return acc0, acc1, l_i0, l_i0_1, l_i1, l_i1_1, m_i0, m_i1


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M // 2, HEAD_DIM]  # due to data partitioning
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M // 2, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [3]
else:
    NUM_STAGES_OPTIONS = [3]

if is_tile_enabled():
    # Helper to build config with optional minRegAutoWS/maxRegAutoWS
    def make_tile_config(BM, BN, occ, subtile, subtile_p, vectmul, add2reduce):
        config_kwargs = {
            "BLOCK_M": BM,
            "BLOCK_N": BN,
            "occupancy": occ,
            "SUBTILING": subtile,
            "SUBTILING_P": subtile_p,
            "VECT_MUL": vectmul,
            "FADD2_REDUCE": add2reduce,
        }
        extra_kwargs = {"pre_hook": _host_descriptor_pre_hook}
        return triton.Config(config_kwargs, **extra_kwargs)

    configs = [
        make_tile_config(BM, BN, occ, subtile, subtile_p, vectmul, add2reduce)
        for BM in [256]
        for BN in [64, 128]
        for occ in [1, 2]
        for subtile in [True]
        for subtile_p in [False]
        for vectmul in [0]
        for add2reduce in [False]
    ]
else:
    # Helper to build config with optional minRegAutoWS/maxRegAutoWS
    def make_standard_config(
        BM, BN, s, w, subtile, subtile_p, vectmul, add2reduce, maxreg
    ):
        config_kwargs = {
            "BLOCK_M": BM,
            "BLOCK_N": BN,
            "SUBTILING": subtile,
            "SUBTILING_P": subtile_p,
            "VECT_MUL": vectmul,
            "FADD2_REDUCE": add2reduce,
        }
        extra_kwargs = {
            "num_stages": s,
            "num_warps": w,
            "pre_hook": _host_descriptor_pre_hook,
        }

        # Only add minRegAutoWS/maxRegAutoWS if supported (triton/tree/ws-3.5)
        if HAS_REG_AUTO_WS:
            extra_kwargs["minRegAutoWS"] = 24
            extra_kwargs["maxRegAutoWS"] = maxreg

        if HAS_PINGPONG_AUTO_WS:
            extra_kwargs["pingpongAutoWS"] = True

        return triton.Config(config_kwargs, **extra_kwargs)

    configs = [
        make_standard_config(
            BM, BN, s, w, subtile, subtile_p, vectmul, add2reduce, maxreg
        )
        for BM in [256]
        for BN in [64, 128]
        for s in NUM_STAGES_OPTIONS
        for w in [4]
        for subtile in [True]
        for subtile_p in [True, False]
        for vectmul in [1]
        for add2reduce in [False]
        for maxreg in [152, 192]
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        is_cuda()
        and torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M * BLOCK_N < 128 * 128
        and conf.num_warps == 8
    )


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]

    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


def prune_persistent_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    # Filter out configs based on desired BLOCK_n
    TARGET_BLOCK_N = 64 if N_CTX == 128 else 128
    return [conf for conf in configs if conf.kwargs.get("BLOCK_N", 0) == TARGET_BLOCK_N]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, triton.language.core.tensor_descriptor_base):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _mul_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


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
def _reduce_fadd2(p0a, p1a, p0b, p1b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 rc, ra, rb;
            mov.b64 ra, { $2, $4 };
            mov.b64 rb, { $3, $5 };
            add.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [p0a, p0b, p1a, p1b],
        dtype=[tl.float32, tl.float32],
        is_pure=True,
        pack=1,
    )


@triton.jit
def _attn_fwd_tma_dp(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    pid,
    off_hz,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    SUBTILING_P: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = pid  # tl.program_id(0)
    # off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m0 = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
    offs_m1 = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i0 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
    l_i0_0 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
    acc0 = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)

    m_i1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
    l_i1_0 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
    acc1 = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    q0 = desc_q.load([qo_offset_y, 0])
    q1 = desc_q.load([qo_offset_y + BLOCK_M // 2, 0])

    if FADD2_REDUCE:
        l_i0_1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
        l_i1_1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
    else:
        l_i0_1 = 0
        l_i1_1 = 0

    if STAGE & 1:
        acc0, acc1, l_i0_0, l_i0_1, l_i1_0, l_i1_1, m_i0, m_i1 = _attn_fwd_inner_oss_dp(
            acc0,
            acc1,
            l_i0_0,
            l_i0_1,
            l_i1_0,
            l_i1_1,
            m_i0,
            m_i1,
            q0,
            q1,  #
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m0,
            offs_m1,
            offs_n,
            N_CTX,  #
            warp_specialize,
            SUBTILING,
            SUBTILING_P,
            VECT_MUL,
            FADD2_REDUCE,
        )
    if STAGE & 2:
        acc0, acc1, l_i0_0, l_i0_1, l_i1_0, l_i1_1, m_i0, m_i1 = _attn_fwd_inner_oss_dp(
            acc0,
            acc1,
            l_i0_0,
            l_i0_1,
            l_i1_0,
            l_i1_1,
            m_i0,
            m_i1,
            q0,
            q1,  #
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m0,
            offs_m1,
            offs_n,
            N_CTX,  #
            warp_specialize,
            SUBTILING,
            SUBTILING_P,
            VECT_MUL,
            FADD2_REDUCE,
        )

    if FADD2_REDUCE:
        l_i0 = l_i0_0 + l_i0_1
        l_i1 = l_i1_0 + l_i1_1
    else:
        l_i0 = l_i0_0
        l_i1 = l_i1_0

    m_i0 += tl.math.log2(l_i0)
    acc0 = acc0 / l_i0[:, None]
    m_ptrs0 = M + off_hz * N_CTX + offs_m0
    tl.store(m_ptrs0, m_i0)
    desc_o.store([qo_offset_y, 0], acc0.to(dtype))

    m_i1 += tl.math.log2(l_i1)
    acc1 = acc1 / l_i1[:, None]
    m_ptrs1 = M + off_hz * N_CTX + offs_m1
    tl.store(m_ptrs1, m_i1)
    desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc1.to(dtype))


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    SUBTILING_P: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    pid = tl.program_id(0)
    off_hz = tl.program_id(1)
    _attn_fwd_tma_dp(
        sm_scale,
        M,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        pid,
        off_hz,
        N_CTX,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        FP8_OUTPUT,
        STAGE,
        warp_specialize,
        dtype,
        SUBTILING,
        SUBTILING_P,
        VECT_MUL,
        FADD2_REDUCE,
    )


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_persistent_configs},
)
@triton.jit
def _attn_fwd_persist(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    OUTER_LOOP: tl.constexpr,
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    SUBTILING_P: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M // 2, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M // 2, HEAD_DIM],
    )

    # inner loop warpspec vs. outer loop warpspec
    for _ in tl.range(0, tiles_per_sm, warp_specialize=warp_specialize and OUTER_LOOP):
        pid = tile_idx % n_tile_num
        off_hz = tile_idx // n_tile_num
        _attn_fwd_tma_dp(
            sm_scale,
            M,
            Z,
            H,
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            pid,
            off_hz,
            N_CTX,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            FP8_OUTPUT,
            STAGE,
            warp_specialize and not OUTER_LOOP,
            dtype,
            SUBTILING,
            SUBTILING_P,
            VECT_MUL,
            FADD2_REDUCE,
        )
        tile_idx += num_progs


def torch_dtype_to_triton(dtype):
    if dtype == torch.float8_e5m2:
        return tl.float8e5
    return getattr(tl, str(dtype).split(".")[1])


class _attention_opt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, baseVariant):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        warp_specialize = baseVariant == "ws" or baseVariant == "ws_persistent"
        # Use device_descriptor for Hopper + warpspec.
        if (
            not FORCE_ON_DEVICE
            and supports_host_descriptor()
            and not (is_hopper() and warp_specialize)
        ):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(
                q,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(
                    v,
                    shape=[HEAD_DIM_K, y_dim],
                    strides=[q.shape[2], 1],
                    block_shape=dummy_block,
                )
            else:
                desc_v = TensorDescriptor(
                    v,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=dummy_block,
                )
            desc_k = TensorDescriptor(
                k,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            desc_o = TensorDescriptor(
                o,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        def grid(META):
            return (
                triton.cdiv(q.shape[2], META["BLOCK_M"]),
                q.shape[0] * q.shape[1],
                1,
            )

        def grid_persist(META):
            return (
                min(
                    NUM_SMS,
                    triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1],
                ),
                1,
                1,
            )

        def grid_debug(META):
            return (
                1,
                1,
                1,
            )

        ctx.grid = grid
        persistent = baseVariant == "persistent" or baseVariant == "ws_persistent"
        if WITH_MAXNREG and is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and (
                q.dtype == torch.float16 or q.dtype == torch.bfloat16
            ):
                extra_kern_args["maxnreg"] = 128
            else:
                extra_kern_args["maxnreg"] = 80
        if persistent:
            _attn_fwd_persist[grid_persist](
                sm_scale,
                M,  #
                q.shape[0],
                q.shape[1],  #
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=stage,  #
                warp_specialize=warp_specialize,
                OUTER_LOOP=True,
                dtype=torch_dtype_to_triton(q.dtype),
                **extra_kern_args,
            )
        else:
            _attn_fwd[grid](
                sm_scale,
                M,  #
                q.shape[0],
                q.shape[1],  #
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=stage,  #
                warp_specialize=warp_specialize,
                dtype=torch_dtype_to_triton(q.dtype),
                **extra_kern_args,
            )

        ctx.save_for_backward(q, k, v, o, M)

        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


attention_opt = _attention_opt.apply
