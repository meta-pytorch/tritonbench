"""
Standalone Triton flex attention forward kernel with @triton.autotune.

This kernel implements the same algorithm as PyTorch Inductor's flex attention
Triton template, but as a self-contained file with native Triton autotuning.
It supports block-sparse attention via BlockMask from torch.nn.attention.flex_attention.

Currently implements causal mask only (mask_mod: m >= n).
"""

import math

import torch
import triton
import triton.language as tl


# ─── Autotune configs (same as Inductor's max-autotune flex attention fwd) ────
def get_fwd_configs():
    configs = []
    for BLOCK_M, BLOCK_N, num_stages, num_warps in [
        # (128, 64, 3, 4),
        # (128, 128, 3, 4),
        # (128, 128, 2, 8),
        (128, 128, 1, 8),
        # (64, 128, 3, 4),
        # (64, 64, 3, 4),
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
):
    kv_base_offset = kv_start + kv_offset

    # Load K as [BLOCK_N, QK_HEAD_DIM_ROUNDED] then transpose
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

    # QK dot product
    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qk *= SM_SCALE

    # Score mod (identity for causal — score passes through unchanged)
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)
    post_mod_scores = qk

    if CHECK_BLOCK_BOUNDARY:
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    # Mask mod (causal: m >= n)
    if not IS_FULL_BLOCKS:
        mask_mod_output = m >= n
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, False)
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

    # Load V and accumulate
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)
    v = load_checked_2d(
        V,
        offs_n_load,
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
):
    """Iterate over a single set of KV blocks (partial or full only)."""
    SPARSE_KV_MULTIPLE: tl.constexpr = SPARSE_KV_BLOCK_SIZE // BLOCK_N
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    kv_offset = 0

    for start_n in range(block_n_start, block_n_end):
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                q,
                K,
                V,
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
            )
        else:
            acc, l_i, m_i = forward_block_mn(
                q,
                K,
                V,
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
):
    """Iterate over both partial and full KV blocks with shared accumulators."""
    SPARSE_KV_MULTIPLE: tl.constexpr = SPARSE_KV_BLOCK_SIZE // BLOCK_N
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    # Phase 1: partial blocks (need both score_mod and mask_mod)
    kv_offset = 0
    for start_n in range(0, partial_block_n_end):
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                q,
                K,
                V,
                Q_LEN,
                KV_LEN,
                acc,
                l_i,
                m_i,
                off_z,
                off_h,
                offs_m,
                partial_offs_n,
                partial_kv_start,
                kv_offset,
                MATMUL_PRECISION,
                RCP_LN2,
                stride_kk,
                stride_kn,
                stride_vn,
                stride_vk,
                SM_SCALE,
                IS_FULL_BLOCKS=False,
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
            )
        else:
            acc, l_i, m_i = forward_block_mn(
                q,
                K,
                V,
                Q_LEN,
                KV_LEN,
                acc,
                l_i,
                m_i,
                off_z,
                off_h,
                offs_m,
                partial_offs_n,
                partial_kv_start,
                kv_offset,
                MATMUL_PRECISION,
                RCP_LN2,
                stride_kk,
                stride_kn,
                stride_vn,
                stride_vk,
                SM_SCALE,
                IS_FULL_BLOCKS=False,
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
            )
        offset = get_offset_for_next_block(
            start_n,
            partial_kv_indices,
            partial_kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_KV_MULTIPLE,
            BLOCK_N,
            BLOCKS_ARE_CONTIGUOUS,
        )
        partial_offs_n = partial_offs_n + offset
        kv_offset += offset

    # Phase 2: full blocks (only score_mod, skip mask_mod)
    kv_offset = 0
    for start_n in range(0, full_block_n_end):
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                q,
                K,
                V,
                Q_LEN,
                KV_LEN,
                acc,
                l_i,
                m_i,
                off_z,
                off_h,
                offs_m,
                full_offs_n,
                full_kv_start,
                kv_offset,
                MATMUL_PRECISION,
                RCP_LN2,
                stride_kk,
                stride_kn,
                stride_vn,
                stride_vk,
                SM_SCALE,
                IS_FULL_BLOCKS=True,
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
            )
        else:
            acc, l_i, m_i = forward_block_mn(
                q,
                K,
                V,
                Q_LEN,
                KV_LEN,
                acc,
                l_i,
                m_i,
                off_z,
                off_h,
                offs_m,
                full_offs_n,
                full_kv_start,
                kv_offset,
                MATMUL_PRECISION,
                RCP_LN2,
                stride_kk,
                stride_kn,
                stride_vn,
                stride_vk,
                SM_SCALE,
                IS_FULL_BLOCKS=True,
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
            )
        offset = get_offset_for_next_block(
            start_n,
            full_kv_indices,
            full_kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_KV_MULTIPLE,
            BLOCK_N,
            BLOCKS_ARE_CONTIGUOUS,
        )
        full_offs_n = full_offs_n + offset
        kv_offset += offset

    return acc, l_i, m_i


# ─── Main kernel entry point ─────────────────────────────────────────────────


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
    # Strides
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
    # Block mask strides
    stride_kv_num_blks_h,
    stride_kv_idx_h,
    stride_kv_idx_m,
    # Dimensions
    ZQ,
    HQ,
    Q_LEN,
    ZKV,
    KV_LEN,
    SM_SCALE,
    # Constexpr
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
):
    INDEX_DTYPE: tl.constexpr = tl.int32

    tl.static_assert(
        SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0
    )
    tl.static_assert(
        SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0
    )

    MATMUL_PRECISION = Q.dtype.element_ty

    q_start = tl.program_id(0).to(INDEX_DTYPE)
    off_zq = tl.program_id(1).to(INDEX_DTYPE)
    off_hq = tl.program_id(2).to(INDEX_DTYPE)

    off_zkv = off_zq % ZKV
    off_hkv = off_hq // GQA_SHARED_HEADS

    q_offset = off_zq * stride_qz + off_hq * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    Q_ptr = Q + q_offset
    K_ptr = K + k_offset
    V_ptr = V + v_offset

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

    # ── Partial blocks (need both score_mod and mask_mod) ──
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(
        kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1)
    )
    offs_n = kv_start + tl.arange(0, BLOCK_N)

    if HAS_FULL_BLOCKS:
        # Full block setup
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
            Q_LEN,
            KV_LEN,
            acc,
            l_i,
            m_i,
            off_zq,
            off_hq,
            offs_m[:, None],
            # Partial block data
            offs_n[None, :],
            kv_start,
            kv_indices,
            kv_num_blocks,
            block_n_end,
            # Full block data
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
        )
    else:
        acc, l_i, m_i = forward_inner(
            q,
            K_ptr,
            V_ptr,
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
        )

    # Handle fully masked out rows
    l_i = tl.where(l_i == 0.0, 1, l_i)
    acc = acc / l_i[:, None]

    # Store output
    idx_zq = tl.program_id(1).to(INDEX_DTYPE)
    idx_hq = tl.program_id(2).to(INDEX_DTYPE)
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


# ─── Python wrapper ──────────────────────────────────────────────────────────


def flex_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """
    Standalone flex attention forward pass.

    Args:
        q: Query tensor [B, Hq, M, D]
        k: Key tensor [B, Hkv, N, D]
        v: Value tensor [B, Hkv, N, Dv]
        block_mask: BlockMask from torch.nn.attention.flex_attention.create_block_mask
        sm_scale: Softmax scale (default: 1/sqrt(D))

    Returns:
        Output tensor [B, Hq, M, Dv]
    """
    B, Hq, M, D = q.shape
    _, Hkv, N, Dv = v.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Allocate output and logsumexp
    out = torch.empty(B, Hq, M, Dv, device=q.device, dtype=q.dtype)
    lse = torch.empty(B, Hq, M, device=q.device, dtype=torch.float32)

    # Extract block mask tensors
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

    # Determine if seq lens are divisible by block sizes
    IS_DIVISIBLE = (M % SPARSE_Q_BLOCK_SIZE == 0) and (N % SPARSE_KV_BLOCK_SIZE == 0)

    # Head dim checks
    QK_HEAD_DIM = D
    V_HEAD_DIM = Dv
    QK_HEAD_DIM_ROUNDED = triton.next_power_of_2(QK_HEAD_DIM)
    V_HEAD_DIM_ROUNDED = triton.next_power_of_2(V_HEAD_DIM)
    SAFE_HEAD_DIM = (
        QK_HEAD_DIM == QK_HEAD_DIM_ROUNDED and V_HEAD_DIM == V_HEAD_DIM_ROUNDED
    )

    # Block mask strides
    stride_kv_num_blks_h = kv_num_blocks.stride(-2)
    stride_kv_idx_h = kv_indices.stride(-3)
    stride_kv_idx_m = kv_indices.stride(-2)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            B,
            Hq,
        )

    flex_attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        # Q strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        # K strides
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        # V strides
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        # Out strides
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        # Block mask strides
        stride_kv_num_blks_h,
        stride_kv_idx_h,
        stride_kv_idx_m,
        # Dimensions
        B,
        Hq,
        M,
        B,  # ZKV = ZQ = B
        N,
        sm_scale,
        # Constexpr
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

    return out
