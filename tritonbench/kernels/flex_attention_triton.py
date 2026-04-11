"""
Standalone Triton flex attention forward kernel with @triton.autotune.

This kernel implements the same algorithm as PyTorch Inductor's flex attention
Triton template, but as a self-contained file with native Triton autotuning.
It supports block-sparse attention via BlockMask from torch.nn.attention.flex_attention.

Currently implements causal mask only (mask_mod: m >= n).

Supports three modes:
- Standard: strided pointer loads (default)
- TMA: tensor descriptor loads via tl.make_tensor_descriptor (USE_TMA=True)
- Persistent: TMA + round-robin tile scheduling across SMs (persistent=True)
"""

import math

import torch
import triton
import triton.language as tl


# ─── Autotune configs ─────────────────────────────────────────────────────────


# Check if Triton version supports minRegAutoWS and maxRegAutoWS
def _supports_reg_auto_ws():
    try:
        triton.Config({}, minRegAutoWS=24, maxRegAutoWS=152)
        return True
    except (TypeError, AttributeError):
        return False


HAS_REG_AUTO_WS = _supports_reg_auto_ws()


def get_fwd_configs():
    configs = []
    for BLOCK_M, BLOCK_N, num_stages, num_warps in [
        (128, 64, 3, 4),
        (128, 128, 3, 4),
        (128, 128, 2, 8),
        (128, 128, 1, 8),
        (64, 128, 3, 4),
        (64, 64, 3, 4),
    ]:
        configs.append(
            triton.Config(
                {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
                num_stages=num_stages,
                num_warps=num_warps,
            )
        )
    return configs


def get_ws_configs():
    """Configs for warp-specialized kernels, following the Blackwell FA reference."""
    configs = []
    for BLOCK_M, BLOCK_N, num_stages, num_warps, maxreg in [
        # (128, 64, 2, 4, 152),
        # (128, 64, 3, 4, 152),
        # (128, 128, 2, 4, 152),
        # (128, 128, 3, 4, 152),
        # (128, 128, 2, 4, 192),
        # (128, 128, 3, 4, 192),
        # (64, 128, 2, 4, 152),
        # (64, 128, 3, 4, 152),
        # (64, 64, 2, 4, 152),
        # (64, 64, 3, 4, 152),
        # (256, 64, 2, 4, 152),
        # (256, 128, 2, 4, 192),  # Requires SPARSE_Q_BLOCK_SIZE >= 256
        (128, 128, 2, 4, 152),
    ]:
        extra_kwargs = dict(num_stages=num_stages, num_warps=num_warps)
        if HAS_REG_AUTO_WS:
            extra_kwargs["minRegAutoWS"] = 24
            extra_kwargs["maxRegAutoWS"] = maxreg
        configs.append(
            triton.Config(
                {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
                **extra_kwargs,
            )
        )
    return configs


def get_persistent_configs():
    configs = []
    for BLOCK_M, BLOCK_N, num_stages, num_warps in [
        (128, 64, 3, 4),
        (128, 128, 3, 4),
        (128, 128, 2, 8),
        (128, 128, 1, 8),
        (64, 128, 3, 4),
        (64, 64, 3, 4),
    ]:
        configs.append(
            triton.Config(
                {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
                num_stages=num_stages,
                num_warps=num_warps,
            )
        )
    return configs


# ─── Utility Triton JIT functions ─────────────────────────────────────────────


@triton.jit
def get_offset_for_next_block(
    loop_iter,
    col_indices,
    total_blocks,
    SPARSE_BLOCK,
    SPARSE_BLOCK_MULTIPLE,
    BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr,
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(
        col_indices + cur_block_idx + 1,
        eviction_policy="evict_last",
        mask=cur_block_idx + 1 < total_blocks,
    )
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block) * SPARSE_BLOCK - (
        SPARSE_BLOCK_MULTIPLE - 1
    ) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset


@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices


@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN,
    N_LEN,
):
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(
            ptr,
            mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_LEN),
            other=0.0,
        )
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_LEN), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:
        return tl.load(ptr)


# ─── Inner block compute ─────────────────────────────────────────────────────


@triton.jit
def forward_block_mn(
    q,
    K,
    V,
    desc_k,
    desc_v,
    Q_LEN,
    KV_LEN,
    acc,
    l_i,
    m_i,
    off_z,
    off_h,
    offs_m,
    offs_n,
    kv_start,
    kv_offset,
    MATMUL_PRECISION,
    RCP_LN2,
    stride_kk,
    stride_kn,
    stride_vn,
    stride_vk,
    SM_SCALE,
    IS_FULL_BLOCKS,
    CHECK_BLOCK_BOUNDARY=False,
    PRESCALE_QK: tl.constexpr = False,
    ROWS_GUARANTEED_SAFE: tl.constexpr = False,
    IS_DIVISIBLE: tl.constexpr = True,
    QK_HEAD_DIM: tl.constexpr = 128,
    QK_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    V_HEAD_DIM: tl.constexpr = 128,
    V_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    SAFE_HEAD_DIM: tl.constexpr = True,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    FLOAT32_PRECISION: tl.constexpr = "ieee",
    USE_TMA: tl.constexpr = False,
):
    kv_base_offset = kv_start + kv_offset

    if USE_TMA:
        k = tl.load_tensor_descriptor(desc_k, [kv_base_offset, 0])
    else:
        offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
        offs_n_load = kv_base_offset + tl.arange(0, BLOCK_N)
        k = load_checked_2d(
            K,
            offs_n_load,
            offs_k,
            stride_kn,
            stride_kk,
            IS_DIVISIBLE,
            SAFE_HEAD_DIM,
            KV_LEN,
            QK_HEAD_DIM,
        )

    k = tl.trans(k)
    k = k.to(q.dtype)

    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qk *= SM_SCALE

    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)
    post_mod_scores = qk

    if CHECK_BLOCK_BOUNDARY:
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    # Mask mod (causal: m >= n)
    # Use branchless logic to avoid scf.if with else blocks (unsupported by autoWS).
    # When IS_FULL_BLOCKS is true, mask_mod_output is all-true so tl.where is a no-op.
    mask_mod_output = IS_FULL_BLOCKS | (m >= n)
    if CHECK_BLOCK_BOUNDARY:
        mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, IS_FULL_BLOCKS)
    post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))

    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2

    # Online softmax
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = m_ij == float("-inf")
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp2(m_i - m_ij_masked)
    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])

    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]

    if USE_TMA:
        v = tl.load_tensor_descriptor(desc_v, [kv_base_offset, 0])
    else:
        if not USE_TMA:
            offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)
            offs_n_load_v = kv_base_offset + tl.arange(0, BLOCK_N)
            v = load_checked_2d(
                V,
                offs_n_load_v,
                offs_v,
                stride_vn,
                stride_vk,
                IS_DIVISIBLE,
                SAFE_HEAD_DIM,
                KV_LEN,
                V_HEAD_DIM,
            )

    acc = tl.dot(
        p.to(MATMUL_PRECISION), v.to(q.dtype), acc, input_precision=FLOAT32_PRECISION
    )
    m_i = m_ij

    return acc, l_i, m_i


# ─── Inner loop over KV blocks ───────────────────────────────────────────────


@triton.jit
def forward_inner(
    q,
    K,
    V,
    desc_k,
    desc_v,
    Q_LEN,
    KV_LEN,
    acc,
    l_i,
    m_i,
    off_z,
    off_h,
    offs_m,
    offs_n,
    kv_start,
    kv_indices,
    kv_num_blocks,
    block_n_start,
    block_n_end,
    MATMUL_PRECISION,
    stride_kk,
    stride_kn,
    stride_vn,
    stride_vk,
    SM_SCALE,
    IS_FULL_BLOCKS,
    PRESCALE_QK: tl.constexpr = False,
    ROWS_GUARANTEED_SAFE: tl.constexpr = False,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr = False,
    IS_DIVISIBLE: tl.constexpr = True,
    QK_HEAD_DIM: tl.constexpr = 128,
    QK_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    V_HEAD_DIM: tl.constexpr = 128,
    V_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    SAFE_HEAD_DIM: tl.constexpr = True,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr = 128,
    FLOAT32_PRECISION: tl.constexpr = "ieee",
    USE_TMA: tl.constexpr = False,
    WARP_SPECIALIZE: tl.constexpr = False,
    DP_FACTOR: tl.constexpr = 1,
):
    """Iterate over a single set of KV blocks (partial or full only)."""
    SPARSE_KV_MULTIPLE: tl.constexpr = SPARSE_KV_BLOCK_SIZE // BLOCK_N
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    kv_offset = 0

    for start_n in tl.range(
        block_n_start,
        block_n_end,
        warp_specialize=WARP_SPECIALIZE,
        data_partition_factor=DP_FACTOR,
        merge_epilogue=WARP_SPECIALIZE,
    ):
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                q,
                K,
                V,
                desc_k,
                desc_v,
                Q_LEN,
                KV_LEN,
                acc,
                l_i,
                m_i,
                off_z,
                off_h,
                offs_m,
                offs_n,
                kv_start,
                kv_offset,
                MATMUL_PRECISION,
                RCP_LN2,
                stride_kk,
                stride_kn,
                stride_vn,
                stride_vk,
                SM_SCALE,
                IS_FULL_BLOCKS,
                PRESCALE_QK=PRESCALE_QK,
                ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
                IS_DIVISIBLE=IS_DIVISIBLE,
                QK_HEAD_DIM=QK_HEAD_DIM,
                QK_HEAD_DIM_ROUNDED=QK_HEAD_DIM_ROUNDED,
                V_HEAD_DIM=V_HEAD_DIM,
                V_HEAD_DIM_ROUNDED=V_HEAD_DIM_ROUNDED,
                SAFE_HEAD_DIM=SAFE_HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                FLOAT32_PRECISION=FLOAT32_PRECISION,
                USE_TMA=USE_TMA,
            )
        else:
            acc, l_i, m_i = forward_block_mn(
                q,
                K,
                V,
                desc_k,
                desc_v,
                Q_LEN,
                KV_LEN,
                acc,
                l_i,
                m_i,
                off_z,
                off_h,
                offs_m,
                offs_n,
                kv_start,
                kv_offset,
                MATMUL_PRECISION,
                RCP_LN2,
                stride_kk,
                stride_kn,
                stride_vn,
                stride_vk,
                SM_SCALE,
                IS_FULL_BLOCKS,
                CHECK_BLOCK_BOUNDARY=True,
                PRESCALE_QK=PRESCALE_QK,
                ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
                IS_DIVISIBLE=IS_DIVISIBLE,
                QK_HEAD_DIM=QK_HEAD_DIM,
                QK_HEAD_DIM_ROUNDED=QK_HEAD_DIM_ROUNDED,
                V_HEAD_DIM=V_HEAD_DIM,
                V_HEAD_DIM_ROUNDED=V_HEAD_DIM_ROUNDED,
                SAFE_HEAD_DIM=SAFE_HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                FLOAT32_PRECISION=FLOAT32_PRECISION,
                USE_TMA=USE_TMA,
            )

        offset = get_offset_for_next_block(
            start_n,
            kv_indices,
            kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_KV_MULTIPLE,
            BLOCK_N,
            BLOCKS_ARE_CONTIGUOUS,
        )
        offs_n = offs_n + offset
        kv_offset += offset

    return acc, l_i, m_i


@triton.jit
def forward_inner_with_full_blocks(
    q,
    K,
    V,
    desc_k,
    desc_v,
    Q_LEN,
    KV_LEN,
    acc,
    l_i,
    m_i,
    off_z,
    off_h,
    offs_m,
    # Partial block data
    partial_offs_n,
    partial_kv_start,
    partial_kv_indices,
    partial_kv_num_blocks,
    partial_block_n_end,
    # Full block data
    full_offs_n,
    full_kv_start,
    full_kv_indices,
    full_kv_num_blocks,
    full_block_n_end,
    MATMUL_PRECISION,
    stride_kk,
    stride_kn,
    stride_vn,
    stride_vk,
    SM_SCALE,
    PRESCALE_QK: tl.constexpr = False,
    ROWS_GUARANTEED_SAFE: tl.constexpr = False,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr = False,
    IS_DIVISIBLE: tl.constexpr = True,
    QK_HEAD_DIM: tl.constexpr = 128,
    QK_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    V_HEAD_DIM: tl.constexpr = 128,
    V_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    SAFE_HEAD_DIM: tl.constexpr = True,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr = 128,
    FLOAT32_PRECISION: tl.constexpr = "ieee",
    USE_TMA: tl.constexpr = False,
    WARP_SPECIALIZE: tl.constexpr = False,
    DP_FACTOR: tl.constexpr = 1,
):
    """Iterate over both partial and full KV blocks in a single merged loop."""
    SPARSE_KV_MULTIPLE: tl.constexpr = SPARSE_KV_BLOCK_SIZE // BLOCK_N
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    total_iters = partial_block_n_end + full_block_n_end

    offs_n = partial_offs_n
    kv_start = partial_kv_start
    kv_indices = partial_kv_indices
    kv_num_blocks = partial_kv_num_blocks
    kv_offset = 0

    for start_n in tl.range(
        0,
        total_iters,
        warp_specialize=WARP_SPECIALIZE,
        data_partition_factor=DP_FACTOR,
        merge_epilogue=WARP_SPECIALIZE,
    ):
        is_full = start_n >= partial_block_n_end

        if start_n == partial_block_n_end:
            offs_n = full_offs_n
            kv_start = full_kv_start
            kv_indices = full_kv_indices
            kv_num_blocks = full_kv_num_blocks
            kv_offset = 0

        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                q,
                K,
                V,
                desc_k,
                desc_v,
                Q_LEN,
                KV_LEN,
                acc,
                l_i,
                m_i,
                off_z,
                off_h,
                offs_m,
                offs_n,
                kv_start,
                kv_offset,
                MATMUL_PRECISION,
                RCP_LN2,
                stride_kk,
                stride_kn,
                stride_vn,
                stride_vk,
                SM_SCALE,
                is_full,
                PRESCALE_QK=PRESCALE_QK,
                ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
                IS_DIVISIBLE=IS_DIVISIBLE,
                QK_HEAD_DIM=QK_HEAD_DIM,
                QK_HEAD_DIM_ROUNDED=QK_HEAD_DIM_ROUNDED,
                V_HEAD_DIM=V_HEAD_DIM,
                V_HEAD_DIM_ROUNDED=V_HEAD_DIM_ROUNDED,
                SAFE_HEAD_DIM=SAFE_HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                FLOAT32_PRECISION=FLOAT32_PRECISION,
                USE_TMA=USE_TMA,
            )
        else:
            acc, l_i, m_i = forward_block_mn(
                q,
                K,
                V,
                desc_k,
                desc_v,
                Q_LEN,
                KV_LEN,
                acc,
                l_i,
                m_i,
                off_z,
                off_h,
                offs_m,
                offs_n,
                kv_start,
                kv_offset,
                MATMUL_PRECISION,
                RCP_LN2,
                stride_kk,
                stride_kn,
                stride_vn,
                stride_vk,
                SM_SCALE,
                is_full,
                CHECK_BLOCK_BOUNDARY=True,
                PRESCALE_QK=PRESCALE_QK,
                ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
                IS_DIVISIBLE=IS_DIVISIBLE,
                QK_HEAD_DIM=QK_HEAD_DIM,
                QK_HEAD_DIM_ROUNDED=QK_HEAD_DIM_ROUNDED,
                V_HEAD_DIM=V_HEAD_DIM,
                V_HEAD_DIM_ROUNDED=V_HEAD_DIM_ROUNDED,
                SAFE_HEAD_DIM=SAFE_HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                FLOAT32_PRECISION=FLOAT32_PRECISION,
                USE_TMA=USE_TMA,
            )

        offset = get_offset_for_next_block(
            start_n - tl.where(is_full, partial_block_n_end, 0),
            kv_indices,
            kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_KV_MULTIPLE,
            BLOCK_N,
            BLOCKS_ARE_CONTIGUOUS,
        )
        offs_n = offs_n + offset
        kv_offset += offset

    return acc, l_i, m_i


# ─── Shared tile body (used by both standard and persistent kernels) ─────────


@triton.jit
def _flex_attention_fwd_tile(
    Q,
    K,
    V,
    Out,
    LSE,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_kv_num_blks_h,
    stride_kv_idx_h,
    stride_kv_idx_m,
    ZQ,
    HQ,
    Q_LEN,
    ZKV,
    KV_LEN,
    SM_SCALE,
    q_start,
    off_zq,
    off_hq,
    GQA_SHARED_HEADS: tl.constexpr = 1,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    QK_HEAD_DIM: tl.constexpr = 128,
    QK_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    V_HEAD_DIM: tl.constexpr = 128,
    V_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    SAFE_HEAD_DIM: tl.constexpr = True,
    PRESCALE_QK: tl.constexpr = False,
    ROWS_GUARANTEED_SAFE: tl.constexpr = False,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr = False,
    IS_DIVISIBLE: tl.constexpr = True,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr = 128,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    FLOAT32_PRECISION: tl.constexpr = "ieee",
    USE_TMA: tl.constexpr = False,
    WARP_SPECIALIZE: tl.constexpr = False,
    DP_FACTOR: tl.constexpr = 1,
):
    """Process one Q-tile. Shared by standard and persistent kernels."""
    INDEX_DTYPE: tl.constexpr = tl.int32

    MATMUL_PRECISION = Q.dtype.element_ty

    off_zkv = off_zq % ZKV
    off_hkv = off_hq // GQA_SHARED_HEADS

    q_offset = off_zq * stride_qz + off_hq * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    Q_ptr = Q + q_offset
    K_ptr = K + k_offset
    V_ptr = V + v_offset

    # TMA descriptors (None if not using TMA)
    desc_k = None
    desc_v = None
    if USE_TMA:
        desc_k = tl.make_tensor_descriptor(
            K_ptr,
            shape=[KV_LEN, QK_HEAD_DIM],
            strides=[stride_kn, 1],
            block_shape=[BLOCK_N, QK_HEAD_DIM_ROUNDED],
        )
        desc_v = tl.make_tensor_descriptor(
            V_ptr,
            shape=[KV_LEN, V_HEAD_DIM],
            strides=[stride_vn, 1],
            block_shape=[BLOCK_N, V_HEAD_DIM_ROUNDED],
        )

    SPARSE_Z: tl.constexpr = 1
    SPARSE_HQ: tl.constexpr = 1

    sparse_idx_z = off_zq % SPARSE_Z
    sparse_idx_hq = off_hq % SPARSE_HQ

    SPARSE_Q_MULTIPLE: tl.constexpr = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    SPARSE_KV_MULTIPLE: tl.constexpr = SPARSE_KV_BLOCK_SIZE // BLOCK_N

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)

    # Sparse block index offsets
    sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
    sparse_kv_num_blks_offset = (
        sparse_hz_offset * stride_kv_num_blks_h + q_start // SPARSE_Q_MULTIPLE
    )
    sparse_kv_idx_offset = (
        sparse_hz_offset * stride_kv_idx_h
        + (q_start // SPARSE_Q_MULTIPLE) * stride_kv_idx_m
    )

    # Load Q tile
    if USE_TMA:
        desc_q = tl.make_tensor_descriptor(
            Q_ptr,
            shape=[Q_LEN, QK_HEAD_DIM],
            strides=[stride_qm, 1],
            block_shape=[BLOCK_M, QK_HEAD_DIM_ROUNDED],
        )
        q = tl.load_tensor_descriptor(desc_q, [(q_start * BLOCK_M).to(tl.int32), 0])
    else:
        offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
        q = load_checked_2d(
            Q_ptr,
            offs_m,
            offs_k,
            stride_qm,
            stride_qk,
            IS_DIVISIBLE,
            SAFE_HEAD_DIM,
            Q_LEN,
            QK_HEAD_DIM,
        )

    # Partial blocks
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(
        kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1)
    )
    offs_n = kv_start + tl.arange(0, BLOCK_N)

    if HAS_FULL_BLOCKS:
        full_kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
        full_kv_start = tl.load(full_kv_indices) * SPARSE_KV_BLOCK_SIZE
        full_kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
        full_block_n_end = tl.minimum(
            full_kv_num_blocks * SPARSE_KV_MULTIPLE,
            tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1),
        )
        full_offs_n = full_kv_start + tl.arange(0, BLOCK_N)

        acc, l_i, m_i = forward_inner_with_full_blocks(
            q,
            K_ptr,
            V_ptr,
            desc_k,
            desc_v,
            Q_LEN,
            KV_LEN,
            acc,
            l_i,
            m_i,
            off_zq,
            off_hq,
            offs_m[:, None],
            offs_n[None, :],
            kv_start,
            kv_indices,
            kv_num_blocks,
            block_n_end,
            full_offs_n[None, :],
            full_kv_start,
            full_kv_indices,
            full_kv_num_blocks,
            full_block_n_end,
            MATMUL_PRECISION,
            stride_kk,
            stride_kn,
            stride_vn,
            stride_vk,
            SM_SCALE,
            PRESCALE_QK=PRESCALE_QK,
            ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
            BLOCKS_ARE_CONTIGUOUS=BLOCKS_ARE_CONTIGUOUS,
            IS_DIVISIBLE=IS_DIVISIBLE,
            QK_HEAD_DIM=QK_HEAD_DIM,
            QK_HEAD_DIM_ROUNDED=QK_HEAD_DIM_ROUNDED,
            V_HEAD_DIM=V_HEAD_DIM,
            V_HEAD_DIM_ROUNDED=V_HEAD_DIM_ROUNDED,
            SAFE_HEAD_DIM=SAFE_HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            FLOAT32_PRECISION=FLOAT32_PRECISION,
            USE_TMA=USE_TMA,
            WARP_SPECIALIZE=WARP_SPECIALIZE,
            DP_FACTOR=DP_FACTOR,
        )
    else:
        acc, l_i, m_i = forward_inner(
            q,
            K_ptr,
            V_ptr,
            desc_k,
            desc_v,
            Q_LEN,
            KV_LEN,
            acc,
            l_i,
            m_i,
            off_zq,
            off_hq,
            offs_m[:, None],
            offs_n[None, :],
            kv_start,
            kv_indices,
            kv_num_blocks,
            0,
            block_n_end,
            MATMUL_PRECISION,
            stride_kk,
            stride_kn,
            stride_vn,
            stride_vk,
            SM_SCALE,
            IS_FULL_BLOCKS=False,
            PRESCALE_QK=PRESCALE_QK,
            ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
            BLOCKS_ARE_CONTIGUOUS=BLOCKS_ARE_CONTIGUOUS,
            IS_DIVISIBLE=IS_DIVISIBLE,
            QK_HEAD_DIM=QK_HEAD_DIM,
            QK_HEAD_DIM_ROUNDED=QK_HEAD_DIM_ROUNDED,
            V_HEAD_DIM=V_HEAD_DIM,
            V_HEAD_DIM_ROUNDED=V_HEAD_DIM_ROUNDED,
            SAFE_HEAD_DIM=SAFE_HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            FLOAT32_PRECISION=FLOAT32_PRECISION,
            USE_TMA=USE_TMA,
            WARP_SPECIALIZE=WARP_SPECIALIZE,
            DP_FACTOR=DP_FACTOR,
        )

    # Handle fully masked out rows
    l_i = tl.where(l_i == 0.0, 1, l_i)
    acc = acc / l_i[:, None]

    # Store output
    idx_zq = off_zq.to(INDEX_DTYPE)
    idx_hq = off_hq.to(INDEX_DTYPE)
    idx_m = offs_m[:, None].to(INDEX_DTYPE)
    idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :].to(INDEX_DTYPE)

    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)
    out_ptrs = (
        Out
        + idx_zq * stride_oz
        + idx_hq * stride_oh
        + idx_m * stride_om
        + idx_d * stride_ok
    )
    tl.store(out_ptrs, acc, mask)

    # Store logsumexp
    off_hz = off_zq * HQ + off_hq
    l_ptrs = LSE + off_hz * Q_LEN + offs_m
    lse = m_i + tl.math.log2(l_i)
    if IS_DIVISIBLE:
        tl.store(l_ptrs, lse)
    else:
        tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)


# ─── Standard kernel entry point ─────────────────────────────────────────────


@triton.autotune(configs=get_fwd_configs(), key=["Q_LEN", "KV_LEN"])
@triton.jit
def flex_attention_fwd_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_kv_num_blks_h,
    stride_kv_idx_h,
    stride_kv_idx_m,
    ZQ,
    HQ,
    Q_LEN,
    ZKV,
    KV_LEN,
    SM_SCALE,
    GQA_SHARED_HEADS: tl.constexpr = 1,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    QK_HEAD_DIM: tl.constexpr = 128,
    QK_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    V_HEAD_DIM: tl.constexpr = 128,
    V_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    SAFE_HEAD_DIM: tl.constexpr = True,
    PRESCALE_QK: tl.constexpr = False,
    ROWS_GUARANTEED_SAFE: tl.constexpr = False,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr = False,
    IS_DIVISIBLE: tl.constexpr = True,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr = 128,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    FLOAT32_PRECISION: tl.constexpr = "ieee",
    USE_TMA: tl.constexpr = False,
    WARP_SPECIALIZE: tl.constexpr = False,
    DP_FACTOR: tl.constexpr = 2,
):
    INDEX_DTYPE: tl.constexpr = tl.int32
    q_start = tl.program_id(0)
    off_zq = tl.program_id(1).to(INDEX_DTYPE)
    off_hq = tl.program_id(2).to(INDEX_DTYPE)

    _flex_attention_fwd_tile(
        Q,
        K,
        V,
        Out,
        LSE,
        KV_NUM_BLKS,
        KV_IDX,
        FULL_KV_NUM_BLKS,
        FULL_KV_IDX,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_kv_num_blks_h,
        stride_kv_idx_h,
        stride_kv_idx_m,
        ZQ,
        HQ,
        Q_LEN,
        ZKV,
        KV_LEN,
        SM_SCALE,
        q_start,
        off_zq,
        off_hq,
        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
        HAS_FULL_BLOCKS=HAS_FULL_BLOCKS,
        QK_HEAD_DIM=QK_HEAD_DIM,
        QK_HEAD_DIM_ROUNDED=QK_HEAD_DIM_ROUNDED,
        V_HEAD_DIM=V_HEAD_DIM,
        V_HEAD_DIM_ROUNDED=V_HEAD_DIM_ROUNDED,
        SAFE_HEAD_DIM=SAFE_HEAD_DIM,
        PRESCALE_QK=PRESCALE_QK,
        ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
        BLOCKS_ARE_CONTIGUOUS=BLOCKS_ARE_CONTIGUOUS,
        IS_DIVISIBLE=IS_DIVISIBLE,
        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        FLOAT32_PRECISION=FLOAT32_PRECISION,
        USE_TMA=USE_TMA,
        WARP_SPECIALIZE=WARP_SPECIALIZE,
        DP_FACTOR=DP_FACTOR,
    )


# ─── WS kernel entry point (separate autotune configs for warp specialization)


@triton.autotune(configs=get_ws_configs(), key=["Q_LEN", "KV_LEN"])
@triton.jit
def flex_attention_fwd_kernel_ws(
    Q,
    K,
    V,
    Out,
    LSE,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_kv_num_blks_h,
    stride_kv_idx_h,
    stride_kv_idx_m,
    ZQ,
    HQ,
    Q_LEN,
    ZKV,
    KV_LEN,
    SM_SCALE,
    GQA_SHARED_HEADS: tl.constexpr = 1,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    QK_HEAD_DIM: tl.constexpr = 128,
    QK_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    V_HEAD_DIM: tl.constexpr = 128,
    V_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    SAFE_HEAD_DIM: tl.constexpr = True,
    PRESCALE_QK: tl.constexpr = False,
    ROWS_GUARANTEED_SAFE: tl.constexpr = False,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr = False,
    IS_DIVISIBLE: tl.constexpr = True,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr = 128,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    FLOAT32_PRECISION: tl.constexpr = "ieee",
    USE_TMA: tl.constexpr = True,
    WARP_SPECIALIZE: tl.constexpr = True,
    DP_FACTOR: tl.constexpr = 1,
):
    INDEX_DTYPE: tl.constexpr = tl.int32
    q_start = tl.program_id(0)
    off_zq = tl.program_id(1).to(INDEX_DTYPE)
    off_hq = tl.program_id(2).to(INDEX_DTYPE)

    _flex_attention_fwd_tile(
        Q,
        K,
        V,
        Out,
        LSE,
        KV_NUM_BLKS,
        KV_IDX,
        FULL_KV_NUM_BLKS,
        FULL_KV_IDX,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_kv_num_blks_h,
        stride_kv_idx_h,
        stride_kv_idx_m,
        ZQ,
        HQ,
        Q_LEN,
        ZKV,
        KV_LEN,
        SM_SCALE,
        q_start,
        off_zq,
        off_hq,
        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
        HAS_FULL_BLOCKS=HAS_FULL_BLOCKS,
        QK_HEAD_DIM=QK_HEAD_DIM,
        QK_HEAD_DIM_ROUNDED=QK_HEAD_DIM_ROUNDED,
        V_HEAD_DIM=V_HEAD_DIM,
        V_HEAD_DIM_ROUNDED=V_HEAD_DIM_ROUNDED,
        SAFE_HEAD_DIM=SAFE_HEAD_DIM,
        PRESCALE_QK=PRESCALE_QK,
        ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
        BLOCKS_ARE_CONTIGUOUS=BLOCKS_ARE_CONTIGUOUS,
        IS_DIVISIBLE=IS_DIVISIBLE,
        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        FLOAT32_PRECISION=FLOAT32_PRECISION,
        USE_TMA=USE_TMA,
        WARP_SPECIALIZE=WARP_SPECIALIZE,
        DP_FACTOR=DP_FACTOR,
    )


# ─── Persistent kernel entry point


@triton.autotune(configs=get_persistent_configs(), key=["Q_LEN", "KV_LEN"])
@triton.jit
def flex_attention_fwd_kernel_persistent(
    Q,
    K,
    V,
    Out,
    LSE,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_kv_num_blks_h,
    stride_kv_idx_h,
    stride_kv_idx_m,
    ZQ,
    HQ,
    Q_LEN,
    ZKV,
    KV_LEN,
    SM_SCALE,
    NUM_SMS,
    GQA_SHARED_HEADS: tl.constexpr = 1,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    QK_HEAD_DIM: tl.constexpr = 128,
    QK_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    V_HEAD_DIM: tl.constexpr = 128,
    V_HEAD_DIM_ROUNDED: tl.constexpr = 128,
    SAFE_HEAD_DIM: tl.constexpr = True,
    PRESCALE_QK: tl.constexpr = False,
    ROWS_GUARANTEED_SAFE: tl.constexpr = False,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr = False,
    IS_DIVISIBLE: tl.constexpr = True,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr = 128,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    FLOAT32_PRECISION: tl.constexpr = "ieee",
    WARP_SPECIALIZE: tl.constexpr = False,
    DP_FACTOR: tl.constexpr = 1,
):
    INDEX_DTYPE: tl.constexpr = tl.int32

    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0).to(INDEX_DTYPE)

    n_tile_num = tl.cdiv(Q_LEN, BLOCK_M)
    total_tiles = n_tile_num * ZQ * HQ

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    for _ in tl.range(
        0,
        tiles_per_sm,
        warp_specialize=WARP_SPECIALIZE,
        data_partition_factor=DP_FACTOR,
        merge_epilogue=WARP_SPECIALIZE,
    ):
        q_start = (tile_idx % n_tile_num).to(INDEX_DTYPE)
        off_hz = tile_idx // n_tile_num
        off_zq = (off_hz // HQ).to(INDEX_DTYPE)
        off_hq = (off_hz % HQ).to(INDEX_DTYPE)

        _flex_attention_fwd_tile(
            Q,
            K,
            V,
            Out,
            LSE,
            KV_NUM_BLKS,
            KV_IDX,
            FULL_KV_NUM_BLKS,
            FULL_KV_IDX,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            stride_oz,
            stride_oh,
            stride_om,
            stride_ok,
            stride_kv_num_blks_h,
            stride_kv_idx_h,
            stride_kv_idx_m,
            ZQ,
            HQ,
            Q_LEN,
            ZKV,
            KV_LEN,
            SM_SCALE,
            q_start,
            off_zq,
            off_hq,
            GQA_SHARED_HEADS=GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS=HAS_FULL_BLOCKS,
            QK_HEAD_DIM=QK_HEAD_DIM,
            QK_HEAD_DIM_ROUNDED=QK_HEAD_DIM_ROUNDED,
            V_HEAD_DIM=V_HEAD_DIM,
            V_HEAD_DIM_ROUNDED=V_HEAD_DIM_ROUNDED,
            SAFE_HEAD_DIM=SAFE_HEAD_DIM,
            PRESCALE_QK=PRESCALE_QK,
            ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
            BLOCKS_ARE_CONTIGUOUS=BLOCKS_ARE_CONTIGUOUS,
            IS_DIVISIBLE=IS_DIVISIBLE,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            FLOAT32_PRECISION=FLOAT32_PRECISION,
            USE_TMA=True,
            WARP_SPECIALIZE=WARP_SPECIALIZE,
            DP_FACTOR=DP_FACTOR,
        )

        tile_idx += num_progs


# ─── Python wrapper ──────────────────────────────────────────────────────────


def flex_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    sm_scale: float | None = None,
    persistent: bool = False,
    use_tma: bool = False,
    warp_specialize: bool = False,
    dp_factor: int = 1,
) -> torch.Tensor:
    """
    Standalone flex attention forward pass.

    Args:
        q: Query tensor [B, Hq, M, D]
        k: Key tensor [B, Hkv, N, D]
        v: Value tensor [B, Hkv, N, Dv]
        block_mask: BlockMask from torch.nn.attention.flex_attention.create_block_mask
        sm_scale: Softmax scale (default: 1/sqrt(D))
        persistent: Use persistent kernel with TMA (requires contiguous inner dim)
        use_tma: Use TMA descriptor loads (requires stride_*k == 1, i.e. contiguous inner dim)

    Returns:
        Output tensor [B, Hq, M, Dv]
    """
    B, Hq, M, D = q.shape
    _, Hkv, N, Dv = v.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty(B, Hq, M, Dv, device=q.device, dtype=q.dtype)
    lse = torch.empty(B, Hq, M, device=q.device, dtype=torch.float32)

    kv_num_blocks = block_mask.kv_num_blocks
    kv_indices = block_mask.kv_indices
    full_kv_num_blocks = block_mask.full_kv_num_blocks
    full_kv_indices = block_mask.full_kv_indices

    has_full_blocks = full_kv_num_blocks is not None
    if not has_full_blocks:
        full_kv_num_blocks = torch.zeros(1, device=q.device, dtype=torch.int32)
        full_kv_indices = torch.zeros(1, device=q.device, dtype=torch.int32)

    SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE = block_mask.BLOCK_SIZE
    GQA_SHARED_HEADS = Hq // Hkv
    IS_DIVISIBLE = (M % SPARSE_Q_BLOCK_SIZE == 0) and (N % SPARSE_KV_BLOCK_SIZE == 0)

    QK_HEAD_DIM = D
    V_HEAD_DIM = Dv
    QK_HEAD_DIM_ROUNDED = triton.next_power_of_2(QK_HEAD_DIM)
    V_HEAD_DIM_ROUNDED = triton.next_power_of_2(V_HEAD_DIM)
    SAFE_HEAD_DIM = (
        QK_HEAD_DIM == QK_HEAD_DIM_ROUNDED and V_HEAD_DIM == V_HEAD_DIM_ROUNDED
    )

    stride_kv_num_blks_h = kv_num_blocks.stride(-2)
    stride_kv_idx_h = kv_indices.stride(-3)
    stride_kv_idx_m = kv_indices.stride(-2)

    common_args = (
        q,
        k,
        v,
        out,
        lse,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        stride_kv_num_blks_h,
        stride_kv_idx_h,
        stride_kv_idx_m,
        B,
        Hq,
        M,
        B,
        N,
        sm_scale,
    )

    common_kwargs = dict(
        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
        HAS_FULL_BLOCKS=has_full_blocks,
        QK_HEAD_DIM=QK_HEAD_DIM,
        QK_HEAD_DIM_ROUNDED=QK_HEAD_DIM_ROUNDED,
        V_HEAD_DIM=V_HEAD_DIM,
        V_HEAD_DIM_ROUNDED=V_HEAD_DIM_ROUNDED,
        SAFE_HEAD_DIM=SAFE_HEAD_DIM,
        IS_DIVISIBLE=IS_DIVISIBLE,
        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
    )

    if persistent:
        NUM_SMS = torch.cuda.get_device_properties(q.device).multi_processor_count

        def grid_persist(META):
            return (
                min(NUM_SMS, triton.cdiv(M, META["BLOCK_M"]) * B * Hq),
                1,
                1,
            )

        flex_attention_fwd_kernel_persistent[grid_persist](
            *common_args,
            NUM_SMS,
            **common_kwargs,
            WARP_SPECIALIZE=warp_specialize,
            DP_FACTOR=dp_factor,
        )
    elif warp_specialize:

        def grid_ws(META):
            return (triton.cdiv(M, META["BLOCK_M"]), B, Hq)

        flex_attention_fwd_kernel_ws[grid_ws](
            *common_args,
            **common_kwargs,
            DP_FACTOR=dp_factor,
        )
    else:

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]), B, Hq)

        flex_attention_fwd_kernel[grid](
            *common_args,
            **common_kwargs,
            USE_TMA=use_tma,
            WARP_SPECIALIZE=warp_specialize,
            DP_FACTOR=dp_factor,
        )

    return out
