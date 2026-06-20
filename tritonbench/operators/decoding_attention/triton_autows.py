# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _decode_kv_step(
    q,
    desc_k,
    desc_v,
    kv_base,
    start_n,
    offs_n_base,
    row_limit,
    qk_scale,
    m_i,
    l_i,
    acc,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    offs_n = start_n + offs_n_base

    k = desc_k.load([kv_base + start_n, 0])
    # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
    qk = tl.dot(q, k.T, allow_tf32=True) * qk_scale

    valid = offs_n[None, :] < row_limit[:, None]
    qk = tl.where(valid, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_ij[:, None])
    alpha = tl.math.exp2(m_i - m_ij)

    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]

    v = desc_v.load([kv_base + start_n, 0])
    # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
    acc = tl.dot(p.to(dtype), v, acc, allow_tf32=True)
    return m_ij, l_i, acc


# Triton TR001: this coverage backend intentionally keeps a fixed config while
# it is disabled by default pending proper runtime gates.
@triton.jit
def _autows_decode_fwd_kernel(  # noqa: TR001
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    cache_seqlens,
    qk_scale,
    MAX_LEN_KV,
    SEQ_LEN_Q: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    out_dtype: tl.constexpr,
):
    off_b = tl.program_id(1)
    kv_len = tl.load(cache_seqlens + off_b).to(tl.int32)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    seq_idx = offs_m // HEAD_Q
    row_limit = kv_len - SEQ_LEN_Q + seq_idx + 1

    q = desc_q.load([off_b * BLOCK_M, 0])

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    kv_base = off_b * MAX_LEN_KV
    for start_n in tl.range(0, kv_len, BLOCK_N, warp_specialize=True):
        m_i, l_i, acc = _decode_kv_step(
            q,
            desc_k,
            desc_v,
            kv_base,
            start_n,
            offs_n_base,
            row_limit,
            qk_scale,
            m_i,
            l_i,
            acc,
            BLOCK_N,
            out_dtype,
        )

    desc_o.store([off_b * BLOCK_M, 0], (acc / l_i[:, None]).to(out_dtype))


def _triton_dtype(tensor: torch.Tensor):
    return getattr(tl, str(tensor.dtype).split(".")[1])


def autows_decoding_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """AutoWS forward for TritonBench decoding_attention triton_splitk semantics.

    This re-expresses the split-K decode attention forward as a single-pass
    flash-decoding forward so Meta's AutoWS path can see an explicit
    ``tl.range(..., warp_specialize=True)``.
    """
    if not hasattr(triton, "knobs"):
        raise NotImplementedError("autows_decoding_attention requires Meta Triton")

    batch, seq_len_q, head_q, head_dim = q.shape
    k_batch, max_len_kv, head_kv, k_head_dim = k_cache.shape
    v_batch, v_max_len_kv, v_head_kv, v_head_dim = v_cache.shape
    if (
        batch != k_batch
        or batch != v_batch
        or max_len_kv != v_max_len_kv
        or head_kv != 1
        or v_head_kv != 1
        or head_dim != k_head_dim
        or head_dim != v_head_dim
    ):
        raise NotImplementedError(
            "autows_decoding_attention requires matching B/max_len/D and Hkv=1"
        )
    if head_dim != 128:
        raise NotImplementedError("autows_decoding_attention currently supports D=128")

    folded_m = seq_len_q * head_q
    if folded_m > block_m:
        raise NotImplementedError(
            f"autows_decoding_attention requires seq_len_q * head_q <= {block_m}"
        )

    cache_seqlens = cache_seqlens.to(torch.int32).contiguous()
    q_folded = q.contiguous().view(batch, folded_m, head_dim)
    q_pad = torch.zeros((batch, block_m, head_dim), dtype=q.dtype, device=q.device)
    q_pad[:, :folded_m, :] = q_folded
    o_pad = torch.empty((batch, block_m, head_dim), dtype=q.dtype, device=q.device)

    q2 = q_pad.view(batch * block_m, head_dim)
    o2 = o_pad.view(batch * block_m, head_dim)
    k2 = k_cache.contiguous().view(batch * max_len_kv, head_dim)
    v2 = v_cache.contiguous().view(batch * max_len_kv, head_dim)

    desc_q = TensorDescriptor(
        q2,
        shape=[batch * block_m, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim],
    )
    desc_o = TensorDescriptor(
        o2,
        shape=[batch * block_m, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim],
    )
    desc_k = TensorDescriptor(
        k2,
        shape=[batch * max_len_kv, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_v = TensorDescriptor(
        v2,
        shape=[batch * max_len_kv, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim],
    )

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=q.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = (1, batch, 1)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504

    assert triton.knobs.nvidia.use_meta_ws, (
        "autows_decoding_attention requires TRITON_USE_META_WS=1"
    )
    assert triton.knobs.nvidia.disable_wsbarrier_reorder, (
        "autows_decoding_attention currently requires "
        "TRITON_DISABLE_WSBARRIER_REORDER=1"
    )
    # TODO: Tune max registers across partitions.
    _autows_decode_fwd_kernel[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        cache_seqlens,
        qk_scale,
        max_len_kv,
        SEQ_LEN_Q=seq_len_q,
        HEAD_Q=head_q,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        out_dtype=_triton_dtype(q_pad),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o_pad[:, :folded_m, :].reshape(batch, seq_len_q, head_q, head_dim)
