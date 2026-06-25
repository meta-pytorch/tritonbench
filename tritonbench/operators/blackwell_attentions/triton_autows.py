# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _hstu_kv_step(
    q,
    desc_k,
    desc_v,
    seq_start,
    start_n,
    col,
    offs_m,
    alpha,
    scale,
    acc,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    k = desc_k.load([seq_start + start_n, col])
    # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
    qk = tl.dot(q, k.T, allow_tf32=True)

    qk = qk * alpha
    silu = (qk / (1.0 + tl.exp(-qk))) * scale
    p = tl.where(offs_m[:, None] >= offs_n[None, :], silu, 0.0)

    v = desc_v.load([seq_start + start_n, col])
    # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
    acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=True)
    return acc


# Triton TR001: this coverage backend intentionally keeps a fixed config while
# it is disabled by default pending compiler fixes and proper runtime gates.
@triton.jit
def _autows_hstu_fwd_kernel(  # noqa: TR001
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    seq_offsets,
    alpha,
    scale,
    H,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    out_dtype: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    seq_start = tl.load(seq_offsets + off_z).to(tl.int32)
    seq_end = tl.load(seq_offsets + off_z + 1).to(tl.int32)
    seq_len = seq_end - seq_start

    start_m = pid_m * BLOCK_M
    if start_m >= seq_len:
        return

    col = off_h * HEAD_DIM
    offs_m = start_m + tl.arange(0, BLOCK_M)
    q = desc_q.load([seq_start + start_m, col])
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    hi = tl.minimum(start_m + BLOCK_M, seq_len)
    for start_n in tl.range(0, hi, BLOCK_N, warp_specialize=True):
        acc = _hstu_kv_step(
            q,
            desc_k,
            desc_v,
            seq_start,
            start_n,
            col,
            offs_m,
            alpha,
            scale,
            acc,
            BLOCK_N,
        )

    desc_o.store([seq_start + start_m, col], acc.to(out_dtype))


def _triton_dtype(tensor: torch.Tensor):
    return getattr(tl, str(tensor.dtype).split(".")[1])


def autows_hstu_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_seq_len: int,
    *,
    alpha: float,
    block_m: int = 128,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """AutoWS HSTU forward for the blackwell_attentions tlx_hstu shape.

    This backend currently does not work: the Meta AutoWS compiler path crashes
    in NVGPUWarpSpecialization for this HSTU loop. The non-WS source math was
    validated separately, but the exported TritonBench backend is intentionally
    disabled until the compiler issue is fixed.
    """
    total_seq, heads, head_dim = q.shape
    if q.shape != k.shape or q.shape != v.shape:
        raise NotImplementedError("autows_hstu_attention requires matching Q/K/V")
    if head_dim != 128:
        raise NotImplementedError("autows_hstu_attention currently supports D=128")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    seq_offsets = seq_offsets.to(torch.int64).contiguous()
    o = torch.empty_like(q)

    q2 = q.view(total_seq, heads * head_dim)
    k2 = k.view(total_seq, heads * head_dim)
    v2 = v.view(total_seq, heads * head_dim)
    o2 = o.view(total_seq, heads * head_dim)

    desc_q = TensorDescriptor(
        q2,
        shape=[total_seq, heads * head_dim],
        strides=[heads * head_dim, 1],
        block_shape=[block_m, head_dim],
    )
    desc_k = TensorDescriptor(
        k2,
        shape=[total_seq, heads * head_dim],
        strides=[heads * head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_v = TensorDescriptor(
        v2,
        shape=[total_seq, heads * head_dim],
        strides=[heads * head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_o = TensorDescriptor(
        o2,
        shape=[total_seq, heads * head_dim],
        strides=[heads * head_dim, 1],
        block_shape=[block_m, head_dim],
    )

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=q.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = (triton.cdiv(max_seq_len, block_m), heads * (seq_offsets.numel() - 1), 1)
    scale = 1.0 / max_seq_len

    assert triton.knobs.nvidia.use_meta_ws, (
        "autows_hstu_attention requires TRITON_USE_META_WS=1"
    )
    assert triton.knobs.nvidia.disable_wsbarrier_reorder, (
        "autows_hstu_attention currently requires TRITON_DISABLE_WSBARRIER_REORDER=1"
    )
    # TODO: Tune max registers across partitions.
    _autows_hstu_fwd_kernel[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        seq_offsets,
        alpha,
        scale,
        heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        out_dtype=_triton_dtype(q),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


# Triton TR001: this coverage backend intentionally keeps a fixed config while
# it is disabled by default pending proper runtime gates.
@triton.jit
def _hstu_fwd_persistent_ws(  # noqa: TR001
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    alpha,
    attn_scale,
    seq_len,
    NUM_SMS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid is flat over #SMs (axis 0); (batch*head) is grid axis 1. Each program
    # walks a strided set of M-tiles within its (batch*head) row group.
    start_pid = tl.program_id(0)
    num_tiles = tl.cdiv(seq_len, BLOCK_M)  # M-tiles per (batch*head)

    bh = tl.program_id(1)
    q_row_off = bh * seq_len

    # Persistent tile loop: this is the loop Meta AutoWS specializes.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        start_m = tile_id * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q = desc_q.load([q_row_off + start_m, 0])  # [BLOCK_M, DIM]
        acc = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)

        # Causal: only need KV blocks up to and including the diagonal block.
        hi = start_m + BLOCK_M
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = desc_k.load([q_row_off + start_n, 0])  # [BLOCK_N, DIM]
            # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
            qk = tl.dot(q, k.T, allow_tf32=True)  # [BLOCK_M, BLOCK_N]

            qk = qk.to(tl.float32) * alpha
            # SiLU activation: x * sigmoid(x)
            silu = qk * tl.sigmoid(qk)
            act = silu * attn_scale
            # causal mask + seq_len bound
            causal = offs_m[:, None] >= offs_n[None, :]
            valid = causal & (offs_n[None, :] < seq_len)
            act = tl.where(valid, act, 0.0)

            v = desc_v.load([q_row_off + start_n, 0])  # [BLOCK_N, DIM]
            # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
            acc += tl.dot(act.to(v.dtype), v, allow_tf32=True)

        desc_o.store([q_row_off + start_m, 0], acc.to(desc_o.dtype))


def autows_hstu_attention_persistent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_seq_len: int,
    *,
    alpha: float,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """PERSISTENT AutoWS HSTU forward for the blackwell_attentions tlx_hstu shape (K-1).

    Same HSTU SiLU math as ``autows_hstu_attention`` but expressed as a persistent
    kernel (``grid = (#SMs, Z*H)`` with an outer warp-specialized tile loop) so
    Meta AutoWS has a loop to specialize. Inputs are packed with all sequences
    full length ``max_seq_len`` (the layout produced by ``preproc_hstu``):
    q, k, v : [Z * max_seq_len, H, DIM]; returns [Z * max_seq_len, H, DIM].
    """
    if not hasattr(triton, "knobs"):
        raise NotImplementedError(
            "autows_hstu_attention_persistent requires Meta Triton"
        )
    if q.shape != k.shape or q.shape != v.shape:
        raise NotImplementedError(
            "autows_hstu_attention_persistent requires matching Q/K/V"
        )

    total_tokens, heads, head_dim = q.shape
    if head_dim != 128:
        raise NotImplementedError(
            "autows_hstu_attention_persistent currently supports D=128"
        )
    if total_tokens % max_seq_len != 0:
        raise NotImplementedError(
            "autows_hstu_attention_persistent requires uniform full-length sequences"
        )
    Z = total_tokens // max_seq_len
    if seq_offsets.numel() != Z + 1:
        raise NotImplementedError(
            "autows_hstu_attention_persistent requires per-batch seq_offsets"
        )

    # View as [Z, H, max_seq_len, DIM] contiguous so TMA descriptors are 2D
    # row-major: row (z*H + h)*max_seq_len + s maps to q[z*max_seq_len + s, h, :].
    q4 = q.view(Z, max_seq_len, heads, head_dim).permute(0, 2, 1, 3).contiguous()
    k4 = k.view(Z, max_seq_len, heads, head_dim).permute(0, 2, 1, 3).contiguous()
    v4 = v.view(Z, max_seq_len, heads, head_dim).permute(0, 2, 1, 3).contiguous()
    o4 = torch.empty_like(q4)

    bh = Z * heads
    q2 = q4.view(bh * max_seq_len, head_dim)
    k2 = k4.view(bh * max_seq_len, head_dim)
    v2 = v4.view(bh * max_seq_len, head_dim)
    o2 = o4.view(bh * max_seq_len, head_dim)

    desc_q = TensorDescriptor(
        q2,
        shape=[bh * max_seq_len, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim],
    )
    desc_k = TensorDescriptor(
        k2,
        shape=[bh * max_seq_len, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_v = TensorDescriptor(
        v2,
        shape=[bh * max_seq_len, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_o = TensorDescriptor(
        o2,
        shape=[bh * max_seq_len, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim],
    )

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=q.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (num_sms, bh)
    attn_scale = 1.0 / max_seq_len

    assert triton.knobs.nvidia.use_meta_ws, (
        "autows_hstu_attention_persistent requires TRITON_USE_META_WS=1"
    )
    # TODO: Tune max registers across partitions.
    _hstu_fwd_persistent_ws[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        alpha,
        attn_scale,
        max_seq_len,
        NUM_SMS=num_sms,
        DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # repack o4 [Z, H, max_seq_len, DIM] -> [Z*max_seq_len, H, DIM]
    return o4.permute(0, 2, 1, 3).contiguous().view(total_tokens, heads, head_dim)
