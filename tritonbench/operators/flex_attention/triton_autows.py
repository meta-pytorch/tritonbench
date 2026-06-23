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


# Triton TR001: this coverage backend intentionally keeps a fixed config while
# it is disabled by default pending proper runtime gates.
@triton.jit
def _flex_attn_fwd_persistent_ws(  # noqa: TR001
    desc_q,  # [B*Hq*M, DIM] row-major (M-blocked tiles)
    desc_k,  # [B*Hkv*N, DIM]
    desc_v,  # [B*Hkv*N, DIM]
    desc_o,  # [B*Hq*M, DIM]
    scale,
    N,
    Hq,
    GROUP: tl.constexpr,  # Hq // Hkv (query heads per kv head)
    NUM_SMS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,  # M // BLOCK_M
    NUM_BHQ: tl.constexpr,  # B * Hq (flattened batch*q_head count)
    M: tl.constexpr,
):
    start_pid = tl.program_id(0)
    # One output tile per (batch, q_head, M-block). The (batch, q_head) index is
    # tile_id // NUM_M_BLOCKS; the M-block within that head is the remainder.
    total_tiles = NUM_BHQ * NUM_M_BLOCKS

    # Persistent tile loop: this is the loop Meta AutoWS specializes.
    for tile_id in tl.range(start_pid, total_tiles, NUM_SMS, warp_specialize=True):
        bhq = tile_id // NUM_M_BLOCKS  # flattened (batch, q_head)
        m_block = tile_id % NUM_M_BLOCKS
        start_m = m_block * BLOCK_M

        # GQA: kv head for this q head. bhq = b*Hq + hq.
        b = bhq // Hq
        hq = bhq % Hq
        hkv = hq // GROUP
        bhkv = b * (Hq // GROUP) + hkv

        q_row = bhq * M + start_m
        kv_row = bhkv * N

        # Q for this tile: [BLOCK_M, DIM]. Scale folded into Q (matches SDPA).
        q = desc_q.load([q_row, 0])
        q = (q.to(tl.float32) * scale).to(q.dtype)

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)

        # Full (non-causal) attention over all N keys (noop mask).
        for start_n in range(0, N, BLOCK_N):
            k = desc_k.load([kv_row + start_n, 0])  # [BLOCK_N, DIM]
            # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
            qk = tl.dot(q, k.T, allow_tf32=True)  # [BLOCK_M, BLOCK_N] (q pre-scaled)

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v = desc_v.load([kv_row + start_n, 0])  # [BLOCK_N, DIM]
            # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
            acc += tl.dot(p.to(v.dtype), v, allow_tf32=True)
            m_i = m_ij

        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / l_safe[:, None]

        desc_o.store([q_row, 0], acc.to(desc_o.dtype))


def autows_flex_attention_persistent(
    q: torch.Tensor,  # [B, Hq, M, D]
    k: torch.Tensor,  # [B, Hkv, N, D]
    v: torch.Tensor,  # [B, Hkv, N, D]
    *,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """PERSISTENT AutoWS FlexAttention (noop / full SDPA) forward (K-3).

    Same noop (non-causal SDPA) math as ``autows_flex_attention`` but expressed
    as a persistent kernel (``grid = #SMs`` with an outer warp-specialized tile
    loop over (batch, q_head, M-block)) so Meta AutoWS has a loop to specialize.
    """
    if not hasattr(triton, "knobs"):
        raise NotImplementedError(
            "autows_flex_attention_persistent requires Meta Triton"
        )

    batch, heads_q, q_ctx, head_dim = q.shape
    _, heads_kv, kv_ctx, _ = k.shape
    if heads_q % heads_kv != 0:
        raise NotImplementedError(
            "autows_flex_attention_persistent requires Hq % Hkv == 0"
        )
    if head_dim != 128:
        raise NotImplementedError(
            "autows_flex_attention_persistent currently supports D=128"
        )
    if q_ctx % block_m != 0:
        raise NotImplementedError(
            f"autows_flex_attention_persistent requires M % {block_m} == 0"
        )
    group = heads_q // heads_kv

    q2 = q.reshape(batch * heads_q * q_ctx, head_dim).contiguous()
    k2 = k.reshape(batch * heads_kv * kv_ctx, head_dim).contiguous()
    v2 = v.reshape(batch * heads_kv * kv_ctx, head_dim).contiguous()
    o2 = torch.empty(
        (batch * heads_q * q_ctx, head_dim), dtype=q.dtype, device=q.device
    )

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=q.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    desc_q = TensorDescriptor(
        q2,
        shape=[batch * heads_q * q_ctx, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim],
    )
    desc_k = TensorDescriptor(
        k2,
        shape=[batch * heads_kv * kv_ctx, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_v = TensorDescriptor(
        v2,
        shape=[batch * heads_kv * kv_ctx, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_o = TensorDescriptor(
        o2,
        shape=[batch * heads_q * q_ctx, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim],
    )

    scale = head_dim**-0.5
    num_m_blocks = q_ctx // block_m
    num_bhq = batch * heads_q
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (num_sms,)

    assert triton.knobs.nvidia.use_meta_ws, (
        "autows_flex_attention_persistent requires TRITON_USE_META_WS=1"
    )
    # TODO: Tune max registers across partitions.
    _flex_attn_fwd_persistent_ws[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        scale,
        kv_ctx,
        heads_q,
        GROUP=group,
        NUM_SMS=num_sms,
        DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_M_BLOCKS=num_m_blocks,
        NUM_BHQ=num_bhq,
        M=q_ctx,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o2.reshape(batch, heads_q, q_ctx, head_dim)
