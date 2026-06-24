# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _fast_gelu(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    return 0.5 * x * (1.0 + tanh_inner)


@triton.jit
def _gdpa_kv_step(
    q,
    desc_k,
    desc_v,
    offset_y,
    start_n,
    qk_scale,
    acc,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)

    k = desc_k.load([offset_y + start_n, 0])
    # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
    qk = tl.dot(q, k.T, allow_tf32=True) * qk_scale
    p = _fast_gelu(qk)

    v = desc_v.load([offset_y + start_n, 0])
    # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
    acc = tl.dot(p.to(dtype), v, acc, allow_tf32=True)
    return acc


# Triton TR001: this coverage backend intentionally keeps a fixed config while
# follow-up reliability work covers the failed num_stages=3 and num_warps=8
# variants from the original AutoWS investigation.
@triton.jit
def _autows_gdpa_fwd_kernel(  # noqa: TR001
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    qk_scale,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    out_dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    offset_y = off_hz * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    q = desc_q.load([qo_offset_y, 0])
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N, warp_specialize=True):
        acc = _gdpa_kv_step(
            q,
            desc_k,
            desc_v,
            offset_y,
            start_n,
            qk_scale,
            acc,
            BLOCK_N,
            out_dtype,
        )

    desc_o.store([qo_offset_y, 0], acc.to(out_dtype))


def _triton_dtype(tensor: torch.Tensor):
    return getattr(tl, str(tensor.dtype).split(".")[1])


def _round_up(value: int, block: int) -> int:
    return ((value + block - 1) // block) * block


def _pack_jagged(
    values: torch.Tensor,
    offsets: torch.Tensor,
    batch: int,
    dense_seq_len: int,
) -> torch.Tensor:
    total_seq, heads, head_dim = values.shape
    packed = torch.zeros(
        (batch, heads, dense_seq_len, head_dim),
        dtype=values.dtype,
        device=values.device,
    )
    for batch_idx in range(batch):
        start = int(offsets[batch_idx].item())
        end = int(offsets[batch_idx + 1].item())
        seq_len = end - start
        if seq_len > 0:
            packed[batch_idx, :, :seq_len, :] = values[start:end].transpose(0, 1)
    return packed


def _unpack_jagged(
    values: torch.Tensor,
    offsets: torch.Tensor,
    total_seq: int,
) -> torch.Tensor:
    batch, heads, _dense_seq_len, head_dim = values.shape
    unpacked = torch.empty(
        (total_seq, heads, head_dim),
        dtype=values.dtype,
        device=values.device,
    )
    for batch_idx in range(batch):
        start = int(offsets[batch_idx].item())
        end = int(offsets[batch_idx + 1].item())
        seq_len = end - start
        if seq_len > 0:
            unpacked[start:end] = values[batch_idx, :, :seq_len, :].transpose(0, 1)
    return unpacked


def _autows_gdpa_dense_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    if not hasattr(triton, "knobs"):
        raise NotImplementedError("triton_autows_gdpa requires Meta Triton")

    batch, heads, seq_len, head_dim = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise NotImplementedError("triton_autows_gdpa requires matching Q/K/V")
    if head_dim != 128:
        raise NotImplementedError("triton_autows_gdpa currently supports D=128")
    if seq_len % block_m != 0 or seq_len % block_n != 0:
        raise NotImplementedError(
            "triton_autows_gdpa requires padded sequence length to be a multiple "
            f"of BLOCK_M={block_m} and BLOCK_N={block_n}"
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)

    q2 = q.view(batch * heads * seq_len, head_dim)
    k2 = k.view(batch * heads * seq_len, head_dim)
    v2 = v.view(batch * heads * seq_len, head_dim)
    o2 = o.view(batch * heads * seq_len, head_dim)

    desc_q = TensorDescriptor(
        q2,
        shape=[batch * heads * seq_len, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim],
    )
    desc_k = TensorDescriptor(
        k2,
        shape=[batch * heads * seq_len, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_v = TensorDescriptor(
        v2,
        shape=[batch * heads * seq_len, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim],
    )
    desc_o = TensorDescriptor(
        o2,
        shape=[batch * heads * seq_len, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim],
    )

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=q.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = (triton.cdiv(seq_len, block_m), batch * heads, 1)

    assert triton.knobs.nvidia.use_meta_ws, (
        "triton_autows_gdpa requires TRITON_USE_META_WS=1"
    )
    assert triton.knobs.nvidia.disable_wsbarrier_reorder, (
        "triton_autows_gdpa currently requires TRITON_DISABLE_WSBARRIER_REORDER=1"
    )
    # TODO: Tune max registers across partitions.
    _autows_gdpa_fwd_kernel[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        1.0,
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        out_dtype=_triton_dtype(q),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def triton_autows_gdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    *,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    activation: str,
    broadcast_q: bool,
    window_size: int | None,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Blackwell AutoWS GDPA forward for the default dense-core GDPA case."""
    if activation != "fast_gelu":
        raise NotImplementedError("triton_autows_gdpa currently supports fast_gelu")
    if broadcast_q:
        raise NotImplementedError("triton_autows_gdpa does not support broadcast_q")
    if window_size is not None:
        raise NotImplementedError("triton_autows_gdpa does not support window_size")
    if query.shape[1:] != key.shape[1:] or query.shape[1:] != value.shape[1:]:
        raise NotImplementedError("triton_autows_gdpa requires matching H/D")

    batch = query_offset.numel() - 1
    dense_seq_len = _round_up(max(max_seq_len_q, max_seq_len_kv), max(block_m, block_n))

    q = _pack_jagged(query, query_offset, batch, dense_seq_len)
    k = _pack_jagged(key, key_offset, batch, dense_seq_len)
    v = _pack_jagged(value, key_offset, batch, dense_seq_len)
    dense_out = _autows_gdpa_dense_fwd(
        q,
        k,
        v,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return _unpack_jagged(dense_out, query_offset, query.shape[0])


# Triton TR001: this coverage backend intentionally keeps a fixed config; it is
# disabled by default because it currently does not compile through Meta AutoWS.
@triton.jit
def _gdpa_fwd_persistent_ws(  # noqa: TR001
    Q,  # packed [total_q, H*DIM] (row-major), head h at cols [h*DIM, (h+1)*DIM)
    K,  # packed [total_k, Hkv*DIM]
    V,  # packed [total_k, Hkv*DIM]
    Out,  # packed [total_q, H*DIM]
    Q_offsets,  # [Z+1] int32 jagged row offsets for Q/Out
    K_offsets,  # [Z+1] int32 jagged row offsets for K/V
    qk_scale,
    H,  # number of q heads
    G: tl.constexpr,  # q heads per kv head (H // Hkv)
    N_CTX: tl.constexpr,  # max_seq_len_q (cap on qlen)
    DIM: tl.constexpr,  # head_dim
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILE_NUM: tl.constexpr,  # cdiv(N_CTX, BLOCK_M)
    N_KV_BLOCKS: tl.constexpr,  # cdiv(N_CTX_KV, BLOCK_N) -- STATIC inner trip count
    Z,  # batch count
):
    start_pid = tl.program_id(0)
    # One logical work item per (batch z, head h, M-block). Flattened tile index:
    #   tile = ((z * H) + h) * N_TILE_NUM + m_block
    total_tiles = Z * H * N_TILE_NUM

    # Persistent tile loop -- the loop Meta AutoWS specializes. On-device 2D TMA
    # descriptors are built per tile with the batch's [end_q]/[end_k] extent so the
    # TMA hardware bounds-checks the ragged tail.
    for tile_id in tl.range(start_pid, total_tiles, NUM_SMS, warp_specialize=True):
        m_block = tile_id % N_TILE_NUM
        off_hz = tile_id // N_TILE_NUM
        h = off_hz % H
        z = off_hz // H
        hkv = h // G
        Hkv = H // G  # number of kv heads (column width of packed K/V)

        begin_q = tl.load(Q_offsets + z)
        end_q = tl.load(Q_offsets + z + 1)
        qlen = tl.minimum(end_q - begin_q, N_CTX)

        begin_k = tl.load(K_offsets + z)
        end_k = tl.load(K_offsets + z + 1)
        klen = end_k - begin_k

        start_m = m_block * BLOCK_M

        # Per-tile descriptors bounded to this batch's row extent (end_q / end_k).
        q_desc = tl.make_tensor_descriptor(
            Q, shape=[end_q, H * DIM], strides=[H * DIM, 1], block_shape=[BLOCK_M, DIM]
        )
        k_desc = tl.make_tensor_descriptor(
            K,
            shape=[end_k, Hkv * DIM],
            strides=[Hkv * DIM, 1],
            block_shape=[BLOCK_N, DIM],
        )
        v_desc = tl.make_tensor_descriptor(
            V,
            shape=[end_k, Hkv * DIM],
            strides=[Hkv * DIM, 1],
            block_shape=[BLOCK_N, DIM],
        )
        o_desc = tl.make_tensor_descriptor(
            Out,
            shape=[end_q, H * DIM],
            strides=[H * DIM, 1],
            block_shape=[BLOCK_M, DIM],
        )

        q_row = begin_q + start_m
        q_col = h * DIM
        kv_col = hkv * DIM

        # Q tile: [BLOCK_M, DIM]. Scale folded into Q (qk_scale == 1.0 by default).
        q = q_desc.load([q_row, q_col])
        if qk_scale != 1.0:
            q = (q.to(tl.float32) * qk_scale).to(q.dtype)

        acc = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)

        offs_n = tl.arange(0, BLOCK_N)
        # STATIC inner trip count so the partition scheduler pipelines a
        # static-bound inner loop. The ragged KV tail is masked. No softmax --
        # GDPA applies an elementwise activation to the scores and accumulates
        # act @ v. P is kept register-resident (never TMEM-staged).
        for kv_blk in range(0, N_KV_BLOCKS):
            start_n = kv_blk * BLOCK_N
            k_row = begin_k + start_n
            k = k_desc.load([k_row, kv_col])  # [BLOCK_N, DIM] (TMA bounds-checked)
            # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
            qk = tl.dot(q, k.T, allow_tf32=True)  # [BLOCK_M, BLOCK_N], fp32

            n_mask = (start_n + offs_n) < klen
            act = _fast_gelu(qk)
            act = tl.where(n_mask[None, :], act, 0.0)

            v = v_desc.load([k_row, kv_col])  # [BLOCK_N, DIM]
            # Triton TR011: explicit TF32 policy keeps tensor-core behavior stable.
            acc += tl.dot(act.to(v.dtype), v, allow_tf32=True)

        # Store the output tile. TMA bounds-checks against end_q, so M-tail rows
        # past the batch are dropped (never written into the next batch).
        if klen > 0:
            o_desc.store([q_row, q_col], acc.to(Out.dtype.element_ty))


def triton_autows_gdpa_persistent(
    query: torch.Tensor,  # [total_q, H, DIM]
    key: torch.Tensor,  # [total_k, Hkv, DIM]
    value: torch.Tensor,  # [total_k, Hkv, DIM]
    query_offset: torch.Tensor,  # [Z+1] int32
    key_offset: torch.Tensor,  # [Z+1] int32
    *,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    activation: str,
    broadcast_q: bool,
    window_size: int | None,
    qk_scale: float = 1.0,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """PERSISTENT AutoWS GDPA forward for the default dense-core GDPA case (K-4).

    Jagged-native variant of ``triton_autows_gdpa`` (no dense pack/unpack): a
    persistent kernel whose outer tile loop is annotated
    ``tl.range(..., warp_specialize=True)`` with on-device per-tile TMA
    descriptors bounded to each batch's extent.

    NOTE: this candidate currently does NOT compile through Meta AutoWS -- the
    ``NVGPUWarpSpecialization`` pass crashes building the cross-partition TMEM
    MMA-operand channel for the QK -> activation -> PV two-MMA loop (tracked
    separately). The source math is correct (it matches ``eager_gdpa`` with
    ``warp_specialize=False``); the backend is carried disabled for coverage.
    """
    if not hasattr(triton, "knobs"):
        raise NotImplementedError("triton_autows_gdpa_persistent requires Meta Triton")
    if activation != "fast_gelu":
        raise NotImplementedError(
            "triton_autows_gdpa_persistent currently supports fast_gelu"
        )
    if broadcast_q:
        raise NotImplementedError(
            "triton_autows_gdpa_persistent does not support broadcast_q"
        )
    if window_size is not None:
        raise NotImplementedError(
            "triton_autows_gdpa_persistent does not support window_size"
        )

    total_q, heads, head_dim = query.shape
    total_k, heads_kv, head_dim_k = key.shape
    if head_dim != head_dim_k or value.shape != key.shape:
        raise NotImplementedError("triton_autows_gdpa_persistent requires matching K/V")
    if heads % heads_kv != 0:
        raise NotImplementedError("triton_autows_gdpa_persistent requires H % Hkv == 0")
    if head_dim != 128:
        raise NotImplementedError(
            "triton_autows_gdpa_persistent currently supports D=128"
        )

    group = heads // heads_kv
    batch = query_offset.numel() - 1

    # Pack to 2D [total, H*DIM] contiguous so the on-device TMA descriptors are
    # simple row-major 2D (head h selected by the h*DIM column offset).
    q2 = query.reshape(total_q, heads * head_dim).contiguous()
    k2 = key.reshape(total_k, heads_kv * head_dim).contiguous()
    v2 = value.reshape(total_k, heads_kv * head_dim).contiguous()
    o2 = torch.empty(
        (total_q, heads * head_dim), dtype=query.dtype, device=query.device
    )

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=query.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    n_tile_num = triton.cdiv(max_seq_len_q, block_m)
    n_kv_blocks = triton.cdiv(max_seq_len_kv, block_n)
    num_sms = torch.cuda.get_device_properties(query.device).multi_processor_count

    def grid(_meta):
        total_tiles = batch * heads * n_tile_num
        return (min(num_sms, total_tiles), 1, 1)

    assert triton.knobs.nvidia.use_meta_ws, (
        "triton_autows_gdpa_persistent requires TRITON_USE_META_WS=1"
    )
    # TODO: Tune max registers across partitions.
    _gdpa_fwd_persistent_ws[grid](
        q2,
        k2,
        v2,
        o2,
        query_offset,
        key_offset,
        qk_scale,
        heads,
        G=group,
        N_CTX=max_seq_len_q,
        DIM=head_dim,
        NUM_SMS=num_sms,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        N_TILE_NUM=n_tile_num,
        N_KV_BLOCKS=n_kv_blocks,
        Z=batch,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o2.reshape(total_q, heads, head_dim)
