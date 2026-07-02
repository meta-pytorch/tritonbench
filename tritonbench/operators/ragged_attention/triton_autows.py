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
    qk = tl.dot(q, k.T, allow_tf32=True) * alpha
    silu = (qk / (1.0 + tl.exp(-qk))) * scale
    p = tl.where(offs_m[:, None] >= offs_n[None, :], silu, 0.0)

    v = desc_v.load([seq_start + start_n, col])
    # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
    acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=True)
    return acc


# Triton TR001: this coverage backend intentionally keeps a fixed config while
# it is disabled by default pending compiler fixes and proper runtime gates.
@triton.jit
def _autows_ragged_hstu_fwd_kernel(  # noqa: TR001
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


def triton_autows_ragged_hstu(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    *,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """AutoWS ragged HSTU forward for the ragged_attention hammer_hstu shape.

    This backend currently does not work: the Meta AutoWS compiler path crashes
    in NVGPUWarpSpecialization for this HSTU SiLU -> PV-MMA loop. The non-WS
    source math was validated separately, but the exported TritonBench backend is
    intentionally disabled until the compiler issue is fixed.
    """
    if not hasattr(triton, "knobs"):
        raise NotImplementedError("triton_autows_ragged_hstu requires Meta Triton")
    if q.shape != k.shape or q.shape != v.shape:
        raise NotImplementedError("triton_autows_ragged_hstu requires matching Q/K/V")

    total_seq, heads, head_dim = q.shape
    if head_dim != 128:
        raise NotImplementedError("triton_autows_ragged_hstu currently supports D=128")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    seq_offsets = seq_offsets.to(torch.int64).contiguous()
    out = torch.empty_like(q)

    q2 = q.view(total_seq, heads * head_dim)
    k2 = k.view(total_seq, heads * head_dim)
    v2 = v.view(total_seq, heads * head_dim)
    o2 = out.view(total_seq, heads * head_dim)

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
        "triton_autows_ragged_hstu requires TRITON_USE_META_WS=1"
    )
    assert triton.knobs.nvidia.disable_wsbarrier_reorder, (
        "triton_autows_ragged_hstu currently requires TRITON_DISABLE_WSBARRIER_REORDER=1"
    )
    # TODO: Tune max registers across partitions.
    _autows_ragged_hstu_fwd_kernel[grid](
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
    return out


# Triton TR001: this coverage backend intentionally keeps a fixed config while
# it is disabled by default pending proper runtime gates.
@triton.jit
def _ragged_hstu_fwd_persistent_ws(  # noqa: TR001
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    seq_offsets,
    num_targets,
    alpha,
    scale,
    H,
    NUM_SMS: tl.constexpr,
    DIM_Q: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    # Grid axis 0 is the persistent dimension (#SMs); axis 1 is (Z * H). Each
    # program strides over the M-tiles of its (z, h) row group via the
    # warp-specialized tl.range; this is the loop Meta AutoWS specializes.
    off_zh = tl.program_id(1)

    # Loop-invariant per-(z, h) setup is hoisted OUT of the warp-specialized
    # loop (off_zh is constant per program), so the WS loop body has no
    # pointer-typed cross-partition channels.
    off_z = off_zh // H
    off_h = off_zh % H
    seq_start = tl.load(seq_offsets + off_z).to(tl.int32)
    seq_end = tl.load(seq_offsets + off_z + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = 0

    # Bound the persistent loop by the valid M-tile count for this jagged
    # sequence (instead of a data-dependent `if start_m < seq_len:` inside the
    # loop body) so the WS loop body stays straight-line and the partition
    # scheduler does not collapse into a single partition.
    num_m_blocks = (seq_len + BLOCK_M - 1) // BLOCK_M

    for tile_id in tl.range(start_pid, num_m_blocks, NUM_SMS, warp_specialize=True):
        start_m = tile_id * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q = desc_q.load([seq_start + start_m, off_h * DIM_Q])  # [BLOCK_M, DIM_Q]
        acc = tl.zeros([BLOCK_M, DIM_V], dtype=tl.float32)

        # Lower-triangular causal: only KV blocks up to the diagonal block.
        hi = start_m + BLOCK_M
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = desc_k.load([seq_start + start_n, off_h * DIM_Q])  # [BLOCK_N, DIM_Q]
            # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
            qk = tl.dot(q, k.T, allow_tf32=True) * alpha  # [BLOCK_M, BLOCK_N]

            # invalid_mask exactly per the reference SiLU / non-softmax path.
            cur_offs_m = offs_m
            cur_offs_n = offs_n
            max_ids = seq_len
            if HAS_MULTIPLE_TARGETS:
                max_ids = max_ids - n_targets
                cur_offs_m = tl.where(cur_offs_m < max_ids, cur_offs_m, max_ids)
                cur_offs_n = tl.where(cur_offs_n < max_ids, cur_offs_n, max_ids)

            offs_n_minus_m = cur_offs_n[None, :] - cur_offs_m[:, None]
            invalid_mask = (cur_offs_m[:, None] == cur_offs_n[None, :]) | (
                offs_n_minus_m < 0
            )
            invalid_mask = invalid_mask & (offs_n[None, :] < seq_len)

            qkf = qk.to(tl.float32)
            silu = (qkf * tl.sigmoid(qkf)) * scale
            act = tl.where(invalid_mask, silu, 0.0)

            v = desc_v.load([seq_start + start_n, off_h * DIM_V])  # [BLOCK_N, DIM_V]
            # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
            acc += tl.dot(act.to(v.dtype), v, allow_tf32=True)

        desc_o.store([seq_start + start_m, off_h * DIM_V], acc.to(desc_o.dtype))


def triton_autows_ragged_hstu_persistent(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
    *,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """PERSISTENT AutoWS ragged HSTU forward (K-6).

    Same ragged HSTU SiLU math (lower-triangular causal + num_targets clamping)
    as ``triton_autows_ragged_hstu`` but expressed as a persistent kernel
    (``grid = (#SMs, Z*H)`` with an outer warp-specialized tile loop bounded by
    each jagged sequence's valid M-tile count) so Meta AutoWS has a loop to
    specialize. Inputs are jagged/packed ``q, k : [L, H, DIM_Q]``,
    ``v : [L, H, DIM_V]``; returns ``[L, H, DIM_V]``.
    """
    if not hasattr(triton, "knobs"):
        raise NotImplementedError(
            "triton_autows_ragged_hstu_persistent requires Meta Triton"
        )

    total_seq, heads, dim_q = q.shape
    _, _, dim_v = v.shape
    batch = seq_offsets.shape[0] - 1
    scale = 1.0 / max_seq_len

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    seq_offsets = seq_offsets.to(torch.int64).contiguous()
    out = torch.empty((total_seq, heads, dim_v), dtype=v.dtype, device=q.device)

    q2 = q.view(total_seq, heads * dim_q)
    k2 = k.view(total_seq, heads * dim_q)
    v2 = v.view(total_seq, heads * dim_v)
    o2 = out.view(total_seq, heads * dim_v)

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=q.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    desc_q = TensorDescriptor(
        q2, shape=q2.shape, strides=q2.stride(), block_shape=[block_m, dim_q]
    )
    desc_k = TensorDescriptor(
        k2, shape=k2.shape, strides=k2.stride(), block_shape=[block_n, dim_q]
    )
    desc_v = TensorDescriptor(
        v2, shape=v2.shape, strides=v2.stride(), block_shape=[block_n, dim_v]
    )
    desc_o = TensorDescriptor(
        o2, shape=o2.shape, strides=o2.stride(), block_shape=[block_m, dim_v]
    )

    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (num_sms, batch * heads)

    has_multiple_targets = num_targets is not None
    if num_targets is None:
        # dummy tensor so the kernel signature stays uniform; size by batch so
        # indexing num_targets + off_z stays in bounds if targets are ever enabled
        num_targets = torch.zeros(batch, dtype=torch.int32, device=q.device)

    assert triton.knobs.nvidia.use_meta_ws, (
        "triton_autows_ragged_hstu_persistent requires TRITON_USE_META_WS=1"
    )
    # TODO: Tune max registers across partitions.
    _ragged_hstu_fwd_persistent_ws[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        seq_offsets,
        num_targets,
        alpha,
        scale,
        heads,
        NUM_SMS=num_sms,
        DIM_Q=dim_q,
        DIM_V=dim_v,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
