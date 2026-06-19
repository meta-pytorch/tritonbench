# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attn_kv_step(
    q,
    desc_k,
    desc_v,
    offset_y,
    start_n,
    offs_m,
    offs_n_base,
    qk_scale,
    m_i,
    l_i,
    acc,
    STAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    offs_n = start_n + offs_n_base

    k = desc_k.load([offset_y + start_n, 0])
    # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
    qk = tl.dot(q, k.T, allow_tf32=True) * qk_scale

    if STAGE == 3:
        qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_ij[:, None])
    alpha = tl.math.exp2(m_i - m_ij)

    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]

    v = desc_v.load([offset_y + start_n, 0])
    # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
    acc = tl.dot(p.to(dtype), v, acc, allow_tf32=True)
    return m_ij, l_i, acc


# Triton TR001: this coverage backend intentionally keeps a fixed config while
# it is disabled by default pending proper runtime gates.
@triton.jit
def _autows_flex_attention_kernel(  # noqa: TR001
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    qk_scale,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    out_dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    offset_y = off_hz * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)

    q = desc_q.load([qo_offset_y, 0])

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if STAGE == 3 else N_CTX
    for start_n in tl.range(0, hi, BLOCK_N, warp_specialize=True):
        m_i, l_i, acc = _attn_kv_step(
            q,
            desc_k,
            desc_v,
            offset_y,
            start_n,
            offs_m,
            offs_n_base,
            qk_scale,
            m_i,
            l_i,
            acc,
            STAGE,
            BLOCK_N,
            out_dtype,
        )

    desc_o.store([qo_offset_y, 0], (acc / l_i[:, None]).to(out_dtype))


def _triton_dtype(tensor: torch.Tensor):
    return getattr(tl, str(tensor.dtype).split(".")[1])


def autows_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """AutoWS forward for the TritonBench flex_attention noop/causal cases.

    This re-expresses the compiled flex_attention forward as a standalone
    flash-attention forward so Meta's AutoWS path can see an explicit
    ``tl.range(..., warp_specialize=True)``.
    """
    if not hasattr(triton, "knobs"):
        raise NotImplementedError("autows_flex_attention requires Meta Triton")

    batch, heads, q_ctx, head_dim = q.shape
    k_batch, kv_heads, kv_ctx, k_head_dim = k.shape
    v_batch, v_heads, v_ctx, v_head_dim = v.shape
    if (
        batch != k_batch
        or batch != v_batch
        or heads != kv_heads
        or heads != v_heads
        or q_ctx != kv_ctx
        or q_ctx != v_ctx
        or head_dim != k_head_dim
        or head_dim != v_head_dim
    ):
        raise NotImplementedError("autows_flex_attention requires B/H/S/D match")
    if head_dim != 128:
        raise NotImplementedError("autows_flex_attention currently supports D=128")
    if q_ctx % block_m != 0:
        raise NotImplementedError(f"autows_flex_attention requires S % {block_m} == 0")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)

    q2 = q.view(batch * heads * q_ctx, head_dim)
    k2 = k.view(batch * heads * q_ctx, head_dim)
    v2 = v.view(batch * heads * q_ctx, head_dim)
    o2 = o.view(batch * heads * q_ctx, head_dim)

    desc_q = TensorDescriptor(
        q2,
        shape=[batch * heads * q_ctx, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim],
    )
    desc_k = TensorDescriptor(
        k2,
        shape=[batch * heads * q_ctx, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_v = TensorDescriptor(
        v2,
        shape=[batch * heads * q_ctx, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_o = TensorDescriptor(
        o2,
        shape=[batch * heads * q_ctx, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim],
    )

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=q.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = (triton.cdiv(q_ctx, block_m), batch * heads, 1)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504

    assert triton.knobs.nvidia.use_meta_ws, (
        "autows_flex_attention requires TRITON_USE_META_WS=1"
    )
    assert triton.knobs.nvidia.disable_wsbarrier_reorder, (
        "autows_flex_attention currently requires TRITON_DISABLE_WSBARRIER_REORDER=1"
    )
    # TODO: Tune max registers across partitions.
    _autows_flex_attention_kernel[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        qk_scale,
        N_CTX=q_ctx,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        STAGE=3 if causal else 1,
        out_dtype=_triton_dtype(q),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o
