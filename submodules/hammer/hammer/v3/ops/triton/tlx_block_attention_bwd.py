# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
TLX (Triton Language Extension) warp-specialized block attention BACKWARD
kernel for Blackwell (SM100+) - NON-PERSISTENT, first-draft variant.

Mirrors the 4-partition warp-specialization structure of
third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py:_attn_bwd_ws
(compute / reduction / mma / load) but adapts it to the blocked-MHA setting
used by hammer/v3:

  - Input format: jagged (q_list, k_list, v_list) with per-Q-tensor
    seq_offsets and attn_scale_list, plus a (num_q x num_kv) mask_matrix.
    Mirrors tlx_block_attention.py's forward signature.

  - Activation: SiLU (qk * sigmoid(qk)) scaled by attn_scale and the mask
    instead of softmax. There is no per-row max M and no
    Delta = sum(O * dO) preprocess.

  - Mask: per-pair CAUSAL / ALL / DIAGONAL / LOCAL / NULL dispatch with
    masked / unmasked region split (mirrors the Triton reference at
    hammer/v3/ops/triton/triton_attention.py:_mha_bwd_compute_list_varargs).

  - Cross-pair accumulation: one kernel launch per non-NULL (qi, ki) pair;
    dQ/dK/dV all use async TMA store_reduce='add' against host-pre-zeroed
    output buffers. Mirrors the Triton bwd's atomic_add for dQ and extends
    the same trick to dK/dV since multiple Q tensors can target the same
    KV tensor in the mask matrix.

This is the FIRST DRAFT of the kernel - it intentionally drops several
optimizations present in the FA bwd reference to keep the diff readable:

  - Non-persistent (no CLC scheduling): one CTA per (batch, kv_tile,
    head). Matches the existing tlx_block_attention.py forward layout so
    the persistent variant can be a follow-up (see PERSISTENT_GROUPING).

  - No REUSE_DP_FOR_DQ TMEM aliasing: dQ/dP/dS each get their own TMEM
    slot. Uses more TMEM but removes a class of phasing bugs from the
    first draft.

  - No alloc_warp_barrier variants.

  - Single autotune config (BLOCK_M1=64, BLOCK_N1=128, HEAD_DIM=128 only).
"""

from typing import List, Optional, Tuple

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

from generative_recommenders.common import switch_to_contiguous_if_needed
from hammer.v3.ops.pytorch.pt_attention import MaskType
from hammer.v3.ops.triton.triton_inline_asm_utils import _fma_f32x2, _mul_f32x2

# @manual=//triton:triton
from triton.tools.tensor_descriptor import TensorDescriptor


MASK_CAUSAL = MaskType.CAUSAL.value
MASK_ALL = MaskType.ALL.value
MASK_DIAGONAL = MaskType.DIAGONAL.value
MASK_NULL = MaskType.NULL.value
MASK_LOCAL = MaskType.LOCAL.value


# ---------------------------------------------------------------------------
# Helpers - copied from FA bwd reference for byte-level fidelity.
# ---------------------------------------------------------------------------


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


@triton.jit
def _sub_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            sub.f32x2 rc, ra, rb;
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
def _fast_sigmoid(x):
    # sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))
    half_x = x * 0.5
    t = tl.extra.cuda.libdevice.tanh(half_x)
    return 0.5 * (1.0 + t)


# ---------------------------------------------------------------------------
# Per-(qi, ki, start_n) loop bounds.
#
# Adapted directly from the Triton blocked bwd
# (triton_attention.py:_mha_bwd_compute_list_varargs, lines 1289..1320).
# A KV block at start_n iterates over Q blocks in [low_m, high_m), with the
# subrange [low_m, unmasked_start) requiring per-element masking and
# [unmasked_start, high_m) being fully unmasked.
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_mask_bounds_for_pair(
    cur_mask,
    start_n,
    q_t_seq_len,
    kv_t_seq_len,
    max_attn_len,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    delta = kv_t_seq_len - q_t_seq_len
    if cur_mask == MASK_CAUSAL:
        low_m = start_n - delta - BLOCK_M1 + 1
        low_m = tl.maximum(low_m, 0)
        high_m = q_t_seq_len
        unmasked_start = start_n + BLOCK_N1 - 1 - delta
        unmasked_start = ((unmasked_start + BLOCK_M1 - 1) // BLOCK_M1) * BLOCK_M1
        unmasked_start = tl.maximum(unmasked_start, low_m)
    elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
        low_m = start_n - delta - BLOCK_M1 + 1
        low_m = tl.maximum(low_m, 0)
        high_m = start_n + BLOCK_N1 - delta + max_attn_len
        high_m = tl.minimum(high_m, q_t_seq_len)
        unmasked_start = high_m
    elif cur_mask == MASK_DIAGONAL:
        low_m = tl.maximum(start_n, 0)
        high_m = tl.minimum(start_n + BLOCK_N1, q_t_seq_len)
        unmasked_start = high_m
    else:
        # MASK_ALL
        low_m = 0
        high_m = q_t_seq_len
        unmasked_start = low_m
    # Align loop bounds to BLOCK_M1 (low_m down, high_m up). Also align
    # unmasked_start UP to BLOCK_M1: the masked/unmasked split downstream
    # uses floor-division and an unaligned unmasked_start would lose the
    # trailing partial block. CAUSAL already aligns it; LOCAL / DIAGONAL
    # leave it == high_m (unaligned) so the ceil here brings it to
    # high_m_aligned, meaning num_unmasked_steps=0 (correct: those masks
    # apply to the full range, no unmasked region).
    low_m = (low_m // BLOCK_M1) * BLOCK_M1
    high_m_aligned = ((high_m + BLOCK_M1 - 1) // BLOCK_M1) * BLOCK_M1
    unmasked_start = ((unmasked_start + BLOCK_M1 - 1) // BLOCK_M1) * BLOCK_M1
    unmasked_start = tl.minimum(tl.maximum(unmasked_start, low_m), high_m_aligned)
    return low_m, unmasked_start, high_m_aligned, high_m


@triton.jit
def _bwd_per_cta_setup(
    tile_idx,
    Z,
    NEXT_POW2_BATCH: tl.constexpr,
    q_seq_offsets_tensor,
    kv_seq_offsets_tensor,
    stride_q_so_b,
    stride_kv_so_b,
    cur_mask,
    max_attn_len,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    """Compute per-CTA (batch, kv_tile, head) offsets + Q-block iteration plan.

    Called at the top of every warp-specialized partition because tensor-valued
    locals can't cross the async-task region isolation boundary.
    """
    batches = tl.arange(0, NEXT_POW2_BATCH)
    batch_mask = batches < Z
    kv_seq_b = tl.load(
        kv_seq_offsets_tensor + batches.to(tl.int64) * stride_kv_so_b,
        mask=batch_mask,
        other=0,
    )
    kv_seq_b1 = tl.load(
        kv_seq_offsets_tensor + (batches + 1).to(tl.int64) * stride_kv_so_b,
        mask=batch_mask,
        other=0,
    )
    kv_seqlens = (kv_seq_b1 - kv_seq_b).to(tl.int32)
    n_tiles = (kv_seqlens + BLOCK_N1 - 1) // BLOCK_N1
    n_tiles = tl.where(batch_mask, n_tiles, 0)

    cum_tiles = tl.cumsum(n_tiles, axis=0)
    le_mask = cum_tiles <= tile_idx
    off_z = tl.sum(le_mask.to(tl.int32))
    prev_cum = tl.sum(tl.where(le_mask, n_tiles, 0))
    pid_n = tile_idx - prev_cum

    off_z_i64 = off_z.to(tl.int64)
    q_seq_start = tl.load(q_seq_offsets_tensor + off_z_i64 * stride_q_so_b).to(tl.int64)
    q_seq_end = tl.load(q_seq_offsets_tensor + (off_z_i64 + 1) * stride_q_so_b).to(
        tl.int64
    )
    q_t_seq_len = (q_seq_end - q_seq_start).to(tl.int32)

    kv_seq_start = tl.load(kv_seq_offsets_tensor + off_z_i64 * stride_kv_so_b).to(
        tl.int64
    )
    kv_seq_end = tl.load(kv_seq_offsets_tensor + (off_z_i64 + 1) * stride_kv_so_b).to(
        tl.int64
    )
    kv_t_seq_len = (kv_seq_end - kv_seq_start).to(tl.int32)

    start_n = pid_n * BLOCK_N1

    low_m, unmasked_start, high_m_aligned, _ = _bwd_mask_bounds_for_pair(
        cur_mask,
        start_n,
        q_t_seq_len,
        kv_t_seq_len,
        max_attn_len,
        BLOCK_M1,
        BLOCK_N1,
        HAS_MAX_ATTN_LEN,
    )
    num_masked_steps = (unmasked_start - low_m) // BLOCK_M1
    num_unmasked_steps = (high_m_aligned - unmasked_start) // BLOCK_M1
    num_steps = num_masked_steps + num_unmasked_steps
    # Caller checks: skip CTA if off_z >= Z or q/kv_seq_len == 0 or
    # start_n >= kv_t_seq_len or high_m_aligned <= low_m.
    return (
        off_z,
        q_seq_start,
        q_t_seq_len,
        kv_seq_start,
        kv_t_seq_len,
        start_n,
        low_m,
        num_masked_steps,
        num_unmasked_steps,
        num_steps,
    )


@triton.jit
def _compute_valid_mask(
    cur_mask,
    offs_m,
    offs_n,
    q_t_seq_len,
    kv_t_seq_len,
    max_attn_len,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    """Build the per-element valid mask for SiLU attention.

    Returns a [BLOCK_N1, BLOCK_M1]-shaped bool tile in the same orientation
    as qkT / dpT (i.e., n on dim 0, m on dim 1) so it can be applied
    element-wise on those tiles.
    """
    rows_valid = offs_m < q_t_seq_len  # [M]
    cols_valid = offs_n < kv_t_seq_len  # [N]
    delta = kv_t_seq_len - q_t_seq_len
    if cur_mask == MASK_CAUSAL:
        q_shifted = offs_m + delta  # [M]
        causal_mask = q_shifted[None, :] >= offs_n[:, None]  # [N, M]
        valid_mask = rows_valid[None, :] & cols_valid[:, None] & causal_mask
    elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
        q_shifted = offs_m + delta  # [M]
        causal_mask = q_shifted[None, :] >= offs_n[:, None]  # [N, M]
        local_mask = (q_shifted[None, :] - offs_n[:, None]) < max_attn_len
        valid_mask = (
            rows_valid[None, :] & cols_valid[:, None] & causal_mask & local_mask
        )
    elif cur_mask == MASK_DIAGONAL:
        diag_mask = offs_m[None, :] == offs_n[:, None]  # [N, M]
        valid_mask = rows_valid[None, :] & cols_valid[:, None] & diag_mask
    else:
        # MASK_ALL
        valid_mask = rows_valid[None, :] & cols_valid[:, None]
    return valid_mask


# ---------------------------------------------------------------------------
# Pre-hook: zero dq/dk/dv staging-buffer base tensors before autotune warmup
# and patch TMA descriptor block shapes. Mirrors FA bwd pre-hook.
# ---------------------------------------------------------------------------


def _bwd_host_descriptor_pre_hook_tlx(nargs):
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    HEAD_DIM = nargs["HEAD_DIM"]

    nargs["desc_q"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_do"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_dq"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_dv"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_dk"].block_shape = [BLOCK_N1, HEAD_DIM]


def _get_bwd_configs() -> List[triton.Config]:
    return [
        triton.Config(
            {
                "BLOCK_M1": 64,
                "BLOCK_N1": 64,
                "NUM_BUFFERS_KV": 1,
                "NUM_BUFFERS_Q": 2,
                "NUM_BUFFERS_DO": 1,
                "NUM_BUFFERS_DS": 1,
                "NUM_BUFFERS_TMEM": 1,
            },
            num_warps=8,
            num_stages=1,
            pre_hook=_bwd_host_descriptor_pre_hook_tlx,
        )
    ]


# ---------------------------------------------------------------------------
# Inner compute loop: per CTA's KV-block, iterate over the relevant Q-blocks
# computing pT (silu_scaled) and dsT (dqk) and signalling MMA.
#
# Mirrors FA bwd's _bwd_compute_inner_loop but with:
#   - SiLU math instead of softmax (no M, no Delta)
#   - Per-Q-block attn_scale load (1D tensor of size [L_q])
#   - Per-pair mask dispatch on (offs_m, offs_n)
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_compute_inner_loop(  # noqa: C901
    start_n,
    qk_fulls,
    qk_tiles,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_empties,
    dp_fulls,
    dp_tiles,
    ds_tiles,
    ds_fulls,
    dsT_tmem_tiles,
    dsT_tmem_fulls,
    attn_scale_ptr,
    q_seq_start,
    curr_m,
    blk_idx,
    step_m,
    do_out_dtype,
    q_out_dtype,
    cur_mask,
    q_t_seq_len,
    kv_t_seq_len,
    max_attn_len,
    alpha,
    num_steps,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    APPLY_MASK: tl.constexpr,
):
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    for _ in range(num_steps):
        tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
        ds_buf_id, _ = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)

        # Wait for qkT = K @ Q^T to be produced by the MMA task.
        tlx.barrier_wait(qk_fulls[tmem_buf_id], tmem_phase)
        qkT = tlx.local_load(qk_tiles[tmem_buf_id])
        tlx.barrier_arrive(qk_empties[tmem_buf_id])

        # qkT is currently the raw dot product. Apply alpha scaling.
        qkT = _mul_f32x2(qkT, alpha)

        # Per-Q-row attention scale, shape [BLOCK_M1]. Loaded synchronously
        # (small, in L1 by the time we get here). Matches the TLX fwd
        # approach (tlx_block_attention.py:553).
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        rows_valid = offs_m < q_t_seq_len
        attn_scale_offs = q_seq_start + offs_m.to(tl.int64)
        row_scale = tl.load(
            attn_scale_ptr + attn_scale_offs,
            mask=rows_valid,
            other=0.0,
        ).to(tl.float32)

        # Build the per-element valid mask in qkT-layout [BLOCK_N1, BLOCK_M1].
        # scale_broadcast shape [BLOCK_N1, BLOCK_M1], = row_scale * valid_mask.
        if APPLY_MASK:
            valid_mask = _compute_valid_mask(
                cur_mask,
                offs_m,
                offs_n,
                q_t_seq_len,
                kv_t_seq_len,
                max_attn_len,
                HAS_MAX_ATTN_LEN,
            )
            scale_broadcast = tl.where(valid_mask, row_scale[None, :], 0.0)
        else:
            # Fully unmasked region: scale_broadcast = row_scale * cols_valid.
            cols_valid = offs_n < kv_t_seq_len
            valid_mask = rows_valid[None, :] & cols_valid[:, None]
            scale_broadcast = tl.where(valid_mask, row_scale[None, :], 0.0)

        # ----- forward recompute (silu) -----
        # silu_scaled = qkT * sigmoid(qkT) * scale_broadcast
        sig = _fast_sigmoid(qkT)
        silu_scaled = qkT * sig * scale_broadcast

        # Store ppT = silu_scaled^bf16 to TMEM (P tile) so MMA can do
        # dv += ppT @ do.
        ppT = silu_scaled.to(do_out_dtype)
        tlx.local_store(p_tiles[tmem_buf_id], ppT)
        tlx.barrier_arrive(p_fulls[tmem_buf_id])

        # ----- bwd math (dqkT) -----
        # Wait for dpT = v @ do^T from MMA.
        tlx.barrier_wait(dp_fulls[tmem_buf_id], tmem_phase)
        dpT = tlx.local_load(dp_tiles[tmem_buf_id])
        tlx.barrier_arrive(dp_empties[tmem_buf_id])

        # dqkT = dpT * sigmoid' * scale_broadcast
        # silu derivative wrt qk is sig * (1 + qk - qk * sig).
        d_silu = sig * (1.0 + qkT - qkT * sig)
        dqkT = dpT * d_silu * scale_broadcast
        # Final mask (defensive; scale_broadcast already zeros invalid).
        dqkT = tl.where(valid_mask, dqkT, 0.0)

        dsT = dqkT.to(q_out_dtype)
        tlx.local_store(ds_tiles[ds_buf_id], dsT)
        tlx.local_store(dsT_tmem_tiles[ds_buf_id], dsT)
        tlx.fence("async_shared")
        tlx.barrier_arrive(ds_fulls[ds_buf_id])
        tlx.barrier_arrive(dsT_tmem_fulls[ds_buf_id])

        curr_m += step_m
        blk_idx += 1
    return curr_m, blk_idx


# ---------------------------------------------------------------------------
# Main bwd kernel - 4 warp-specialized partitions (compute, reduction, mma,
# load). Non-persistent: one CTA per (batch, kv_tile, head).
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_bwd_configs(),
    key=["AUTOTUNE_MAX_KV_LEN", "HEAD_DIM", "CUR_MASK"],
)
@triton.jit
def _block_attn_bwd_ws(  # noqa: C901
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    desc_dq,
    desc_dk,
    desc_dv,
    alpha,
    Z,
    H,
    q_seq_offsets_tensor,
    kv_seq_offsets_tensor,
    attn_scale_ptr,
    stride_q_so_b,
    stride_kv_so_b,
    max_attn_len,
    CUR_MASK: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    AUTOTUNE_MAX_KV_LEN: tl.constexpr,
    NEXT_POW2_BATCH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
):
    tl.static_assert(NUM_BUFFERS_Q == 2)
    tl.static_assert(NUM_BUFFERS_DO == 1)

    # Per-element bytes for TMA expect_bytes math.
    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))
    DO_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_do))

    # ---------- CTA-scope dead-tile early-exit ----------
    # The varlen-aware grid axis 0 over-estimates by at most B*(BLOCK_N1-1)
    # tiles. Those overshoot CTAs would otherwise enter the partitions with
    # num_steps==0 and deadlock (MMA prolog waits p_fulls; COMPUTE has no
    # inner-loop iter to arrive it -- circular wait against COMPUTE's
    # end-of-kernel wait dv_fulls). Compute off_z via the same varlen scan
    # the partitions use and bail before async_tasks. Matches TLX fwd's
    # pattern at tlx_block_attention.py:468.
    tile_idx = tl.program_id(0)
    _batches = tl.arange(0, NEXT_POW2_BATCH)
    _batch_mask = _batches < Z
    _kv_seq_b = tl.load(
        kv_seq_offsets_tensor + _batches.to(tl.int64) * stride_kv_so_b,
        mask=_batch_mask,
        other=0,
    )
    _kv_seq_b1 = tl.load(
        kv_seq_offsets_tensor + (_batches + 1).to(tl.int64) * stride_kv_so_b,
        mask=_batch_mask,
        other=0,
    )
    _kv_seqlens = (_kv_seq_b1 - _kv_seq_b).to(tl.int32)
    _n_tiles = (_kv_seqlens + BLOCK_N1 - 1) // BLOCK_N1
    _n_tiles = tl.where(_batch_mask, _n_tiles, 0)
    _cum_tiles = tl.cumsum(_n_tiles, axis=0)
    _off_z_check = tl.sum((_cum_tiles <= tile_idx).to(tl.int32))
    if _off_z_check >= Z:
        return

    # Per-CTA tensor-valued setup (off_z, num_steps, ...) is recomputed
    # inside each partition because the TLX async_task regions are isolated
    # and tensor locals can't cross. CTA-scope keeps only the SMEM/TMEM
    # allocations and the SMEM/TMEM barriers, which are inherently shared.

    # ---------- SMEM allocations ----------
    k_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV
    )
    v_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS_KV
    )
    q_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    do_tiles = tlx.local_alloc(
        (BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_do), NUM_BUFFERS_DO
    )

    # SMEM for dsT (consumed by MMA's dq = ds^T @ k).
    ds_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1), tlx.dtype_of(desc_q), NUM_BUFFERS_DS
    )

    # SMEM staging buffers for async TMA reduce-add of dQ/dK/dV.
    dq_store_buf = tlx.local_alloc(
        (BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_dq), NUM_BUFFERS_TMEM
    )
    dv_store_buf = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_dv), NUM_BUFFERS_KV
    )
    dk_store_buf = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_dk), NUM_BUFFERS_KV
    )

    # ---------- SMEM barriers ----------
    k_mma_done = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    # k_empties pruned: arrive had no wait consumer (non-persistent kernel,
    # 1 KV tile per CTA, no slot reuse).
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    ds_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)
    dsT_tmem_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)

    # ---------- TMEM allocations ----------
    qk_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
    )
    p_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_do),
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    dp_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
    )
    dsT_tmem_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_q),
        NUM_BUFFERS_DS,
        tlx.storage_kind.tmem,
    )
    dv_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM),
        tl.float32,
        NUM_BUFFERS_KV,
        tlx.storage_kind.tmem,
    )
    dk_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM),
        tl.float32,
        NUM_BUFFERS_KV,
        tlx.storage_kind.tmem,
    )
    dq_tiles = tlx.local_alloc(
        (BLOCK_M1, HEAD_DIM),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
    )

    # ---------- TMEM barriers ----------
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dp_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    # dv_empties pruned: arrive had no wait consumer.
    dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    # dk_empties pruned: arrive had no wait consumer.

    with tlx.async_tasks():
        # =================================================================
        # COMPUTE partition (default): runs the silu / dsilu math, then
        # writes dV / dK to global via async TMA reduce-add.
        # =================================================================
        with tlx.async_task("default"):
            (
                off_z,
                q_seq_start,
                q_t_seq_len,
                kv_seq_start,
                kv_t_seq_len,
                start_n,
                low_m,
                num_masked_steps,
                num_unmasked_steps,
                num_steps,
            ) = _bwd_per_cta_setup(
                tl.program_id(0),
                Z,
                NEXT_POW2_BATCH,
                q_seq_offsets_tensor,
                kv_seq_offsets_tensor,
                stride_q_so_b,
                stride_kv_so_b,
                CUR_MASK,
                max_attn_len,
                BLOCK_M1,
                BLOCK_N1,
                HAS_MAX_ATTN_LEN,
            )
            off_h = tl.program_id(1)
            kv_off_y_base = kv_seq_start
            x_offset = off_h.to(tl.int32) * HEAD_DIM
            do_out_dtype = tlx.dtype_of(desc_do)
            q_out_dtype = tlx.dtype_of(desc_q)
            blk_idx = 0
            curr_m = low_m
            step_m = BLOCK_M1
            # Masked region. Inner loop's `for _ in range(num_steps):` is
            # a no-op when num_masked_steps==0, so no outer if-guard needed.
            curr_m, blk_idx = _bwd_compute_inner_loop(
                start_n,
                qk_fulls,
                qk_tiles,
                qk_empties,
                p_tiles,
                p_fulls,
                dp_empties,
                dp_fulls,
                dp_tiles,
                ds_tiles,
                ds_fulls,
                dsT_tmem_tiles,
                dsT_tmem_fulls,
                attn_scale_ptr,
                q_seq_start,
                curr_m,
                blk_idx,
                step_m,
                do_out_dtype,
                q_out_dtype,
                CUR_MASK,
                q_t_seq_len,
                kv_t_seq_len,
                max_attn_len,
                alpha,
                num_masked_steps,
                NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                BLOCK_M1=BLOCK_M1,
                BLOCK_N1=BLOCK_N1,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                APPLY_MASK=True,
            )
            # Unmasked region.
            curr_m, blk_idx = _bwd_compute_inner_loop(
                start_n,
                qk_fulls,
                qk_tiles,
                qk_empties,
                p_tiles,
                p_fulls,
                dp_empties,
                dp_fulls,
                dp_tiles,
                ds_tiles,
                ds_fulls,
                dsT_tmem_tiles,
                dsT_tmem_fulls,
                attn_scale_ptr,
                q_seq_start,
                curr_m,
                blk_idx,
                step_m,
                do_out_dtype,
                q_out_dtype,
                CUR_MASK,
                q_t_seq_len,
                kv_t_seq_len,
                max_attn_len,
                alpha,
                num_unmasked_steps,
                NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                BLOCK_M1=BLOCK_M1,
                BLOCK_N1=BLOCK_N1,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                APPLY_MASK=False,
            )

            # ----- dV write (reduce-add via TMA) -----
            kv_buf_id = 0  # NUM_BUFFERS_KV == 1 for first draft
            tlx.barrier_wait(dv_fulls[kv_buf_id], 0)
            dv = tlx.local_load(dv_tiles[kv_buf_id])
            tlx.async_descriptor_store_wait(0)
            tlx.local_store(dv_store_buf[kv_buf_id], dv.to(tlx.dtype_of(desc_dv)))
            tlx.fence("async_shared")
            tlx.async_descriptor_store(
                desc_dv,
                dv_store_buf[kv_buf_id],
                [
                    (kv_off_y_base + start_n).to(tl.int32),
                    x_offset,
                ],
                store_reduce="add",
            )
            # dv_empties arrive pruned (orphan, see alloc site).

            # ----- dK write (reduce-add via TMA), scaled by alpha -----
            tlx.barrier_wait(dk_fulls[kv_buf_id], 0)
            tlx.barrier_wait(k_mma_done[kv_buf_id], 0)
            dk = tlx.local_load(dk_tiles[kv_buf_id])
            dk = _mul_f32x2(dk, alpha)
            tlx.async_descriptor_store_wait(0)
            tlx.local_store(dk_store_buf[kv_buf_id], dk.to(tlx.dtype_of(desc_dk)))
            tlx.fence("async_shared")
            tlx.async_descriptor_store(
                desc_dk,
                dk_store_buf[kv_buf_id],
                [
                    (kv_off_y_base + start_n).to(tl.int32),
                    x_offset,
                ],
                store_reduce="add",
            )
            tlx.async_descriptor_store_wait(0)
            # k_empties and dk_empties arrives pruned (orphans).

        # =================================================================
        # REDUCTION partition: TMA reduce-add stores of dQ (per Q-block).
        # =================================================================
        with tlx.async_task(num_warps=4, registers=88):
            (
                _off_z,
                q_seq_start,
                _q_t_seq_len,
                _kv_seq_start,
                _kv_t_seq_len,
                _start_n,
                low_m,
                _num_masked_steps,
                _num_unmasked_steps,
                num_steps,
            ) = _bwd_per_cta_setup(
                tl.program_id(0),
                Z,
                NEXT_POW2_BATCH,
                q_seq_offsets_tensor,
                kv_seq_offsets_tensor,
                stride_q_so_b,
                stride_kv_so_b,
                CUR_MASK,
                max_attn_len,
                BLOCK_M1,
                BLOCK_N1,
                HAS_MAX_ATTN_LEN,
            )
            off_h = tl.program_id(1)
            q_off_y_base = q_seq_start
            x_offset = off_h.to(tl.int32) * HEAD_DIM
            blk_idx = 0
            curr_m = low_m
            step_m = BLOCK_M1
            for _ in range(num_steps):
                tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
                tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                dq = tlx.local_load(dq_tiles[tmem_buf_id])
                # alpha scaling for dq (matches Triton bwd: dq_contrib * alpha).
                dq = _mul_f32x2(dq, alpha)
                tlx.async_descriptor_store_wait(0)
                tlx.local_store(dq_store_buf[tmem_buf_id], dq.to(tlx.dtype_of(desc_dq)))
                tlx.fence("async_shared")
                tlx.async_descriptor_store(
                    desc_dq,
                    dq_store_buf[tmem_buf_id],
                    [
                        (q_off_y_base + curr_m).to(tl.int32),
                        x_offset,
                    ],
                    store_reduce="add",
                )
                tlx.barrier_arrive(dq_empties[tmem_buf_id])
                curr_m += step_m
                blk_idx += 1
            tlx.async_descriptor_store_wait(0)

        # =================================================================
        # MMA partition: drives the tensor-core dots.
        # =================================================================
        with tlx.async_task(num_warps=1, registers=24):
            tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
            (
                _off_z,
                _q_seq_start,
                _q_t_seq_len,
                _kv_seq_start,
                _kv_t_seq_len,
                _start_n,
                _low_m,
                _num_masked_steps,
                _num_unmasked_steps,
                num_steps,
            ) = _bwd_per_cta_setup(
                tl.program_id(0),
                Z,
                NEXT_POW2_BATCH,
                q_seq_offsets_tensor,
                kv_seq_offsets_tensor,
                stride_q_so_b,
                stride_kv_so_b,
                CUR_MASK,
                max_attn_len,
                BLOCK_M1,
                BLOCK_N1,
                HAS_MAX_ATTN_LEN,
            )
            # DEBUG: trace last-reached point in the MMA partition.
            blk_idx = 0
            kv_buf_id = 0  # NUM_BUFFERS_KV == 1

            # ----- Prolog (first Q-block) -----
            q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
            do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
            tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

            tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
            qT = tlx.local_trans(q_tiles[q_buf_id])
            tlx.async_dot(
                k_tiles[kv_buf_id],
                qT,
                qk_tiles[tmem_buf_id],
                use_acc=False,
                mBarriers=[qk_fulls[tmem_buf_id]],
            )

            tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
            doT = tlx.local_trans(do_tiles[do_buf_id])
            tlx.async_dot(
                v_tiles[kv_buf_id],
                doT,
                dp_tiles[tmem_buf_id],
                use_acc=False,
                mBarriers=[dp_fulls[tmem_buf_id]],
            )

            tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
            tlx.async_dot(
                p_tiles[tmem_buf_id],
                do_tiles[do_buf_id],
                dv_tiles[kv_buf_id],
                use_acc=False,
                mBarriers=[do_empties[do_buf_id]],
            )
            blk_idx += 1

            # ----- Main loop -----
            # Skip wait dk_empties: dk_tiles fresh; first dk dot uses
            # use_acc=False (gated by `(j - 1) > 0` in the main loop body).
            for j in range(1, num_steps):
                q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

                # qkT = K @ Q^T
                tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
                tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
                qT = tlx.local_trans(q_tiles[q_buf_id])
                tlx.async_dot(
                    k_tiles[kv_buf_id],
                    qT,
                    qk_tiles[tmem_buf_id],
                    use_acc=False,
                    mBarriers=[qk_fulls[tmem_buf_id]],
                )

                prev_blk_idx = blk_idx - 1
                q_buf_id_prev, _ = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                tmem_buf_id_prev, tmem_phase_prev = _get_bufidx_phase(
                    prev_blk_idx, NUM_BUFFERS_TMEM
                )
                ds_buf_id_prev, ds_phase_prev = _get_bufidx_phase(
                    prev_blk_idx, NUM_BUFFERS_DS
                )

                # dk += dsT @ q  (previous iter)
                tlx.barrier_wait(dsT_tmem_fulls[ds_buf_id_prev], ds_phase_prev)
                tlx.async_dot(
                    dsT_tmem_tiles[ds_buf_id_prev],
                    q_tiles[q_buf_id_prev],
                    dk_tiles[kv_buf_id],
                    use_acc=(j - 1) > 0,
                    mBarriers=[q_empties[q_buf_id_prev]],
                )

                # dq = ds^T @ k  (previous iter)
                tlx.barrier_wait(ds_fulls[ds_buf_id_prev], ds_phase_prev)
                tlx.barrier_wait(dq_empties[tmem_buf_id_prev], tmem_phase_prev ^ 1)
                dsT_view = tlx.local_trans(ds_tiles[ds_buf_id_prev])
                tlx.async_dot(
                    dsT_view,
                    k_tiles[kv_buf_id],
                    dq_tiles[tmem_buf_id_prev],
                    use_acc=False,
                    mBarriers=[dq_fulls[tmem_buf_id_prev]],
                )

                do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                tlx.barrier_wait(dp_empties[tmem_buf_id], tmem_phase ^ 1)
                doT = tlx.local_trans(do_tiles[do_buf_id])
                tlx.async_dot(
                    v_tiles[kv_buf_id],
                    doT,
                    dp_tiles[tmem_buf_id],
                    use_acc=False,
                    mBarriers=[dp_fulls[tmem_buf_id]],
                )

                tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
                tlx.async_dot(
                    p_tiles[tmem_buf_id],
                    do_tiles[do_buf_id],
                    dv_tiles[kv_buf_id],
                    use_acc=True,
                    mBarriers=[do_empties[do_buf_id]],
                )
                blk_idx += 1

            tlx.tcgen05_commit(dv_fulls[kv_buf_id])

            # ----- Epilog (last block's dk/dq) -----
            prev_blk_idx = blk_idx - 1
            q_buf_id, _ = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
            tmem_buf_id, tmem_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
            ds_buf_id, ds_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)

            tlx.barrier_wait(dsT_tmem_fulls[ds_buf_id], ds_phase)
            tlx.async_dot(
                dsT_tmem_tiles[ds_buf_id],
                q_tiles[q_buf_id],
                dk_tiles[kv_buf_id],
                use_acc=num_steps > 1,
                mBarriers=[q_empties[q_buf_id], dk_fulls[kv_buf_id]],
            )

            tlx.barrier_wait(ds_fulls[ds_buf_id], ds_phase)
            tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
            dsT_view = tlx.local_trans(ds_tiles[ds_buf_id])
            tlx.async_dot(
                dsT_view,
                k_tiles[kv_buf_id],
                dq_tiles[tmem_buf_id],
                use_acc=False,
                mBarriers=[dq_fulls[tmem_buf_id]],
            )
            tlx.tcgen05_commit(k_mma_done[kv_buf_id])

        # =================================================================
        # LOAD partition: TMA descriptor loads for K, V, Q, dO.
        # =================================================================
        with tlx.async_task(num_warps=1, registers=88):
            (
                _off_z,
                q_seq_start,
                _q_t_seq_len,
                kv_seq_start,
                _kv_t_seq_len,
                start_n,
                low_m,
                _num_masked_steps,
                _num_unmasked_steps,
                num_steps,
            ) = _bwd_per_cta_setup(
                tl.program_id(0),
                Z,
                NEXT_POW2_BATCH,
                q_seq_offsets_tensor,
                kv_seq_offsets_tensor,
                stride_q_so_b,
                stride_kv_so_b,
                CUR_MASK,
                max_attn_len,
                BLOCK_M1,
                BLOCK_N1,
                HAS_MAX_ATTN_LEN,
            )
            off_h = tl.program_id(1)
            q_off_y_base = q_seq_start
            kv_off_y_base = kv_seq_start
            x_offset = off_h.to(tl.int32) * HEAD_DIM
            kv_buf_id = 0  # NUM_BUFFERS_KV == 1

            curr_m = low_m
            step_m = BLOCK_M1

            # Prolog: K + Q bundled on q_fulls; V + dO bundled on do_fulls.
            q_buf_id, q_phase = _get_bufidx_phase(0, NUM_BUFFERS_Q)
            do_buf_id, do_phase = _get_bufidx_phase(0, NUM_BUFFERS_DO)
            tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
            tlx.barrier_expect_bytes(
                q_fulls[q_buf_id],
                K_BYTES_PER_ELEM * BLOCK_N1 * HEAD_DIM
                + Q_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM,
            )
            tlx.async_descriptor_load(
                desc_k,
                k_tiles[kv_buf_id],
                [
                    (kv_off_y_base + start_n).to(tl.int32),
                    x_offset,
                ],
                q_fulls[q_buf_id],
            )
            tlx.async_descriptor_load(
                desc_q,
                q_tiles[q_buf_id],
                [
                    (q_off_y_base + curr_m).to(tl.int32),
                    x_offset,
                ],
                q_fulls[q_buf_id],
            )

            tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
            tlx.barrier_expect_bytes(
                do_fulls[do_buf_id],
                V_BYTES_PER_ELEM * BLOCK_N1 * HEAD_DIM
                + DO_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM,
            )
            tlx.async_descriptor_load(
                desc_v,
                v_tiles[kv_buf_id],
                [
                    (kv_off_y_base + start_n).to(tl.int32),
                    x_offset,
                ],
                do_fulls[do_buf_id],
            )
            tlx.async_descriptor_load(
                desc_do,
                do_tiles[do_buf_id],
                [
                    (q_off_y_base + curr_m).to(tl.int32),
                    x_offset,
                ],
                do_fulls[do_buf_id],
            )

            curr_m += step_m

            # Main-loop loads (Q, dO per iter).
            for j in range(1, num_steps):
                q_buf_id, q_phase = _get_bufidx_phase(j, NUM_BUFFERS_Q)
                do_buf_id, do_phase = _get_bufidx_phase(j, NUM_BUFFERS_DO)
                tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                tlx.barrier_expect_bytes(
                    q_fulls[q_buf_id],
                    Q_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM,
                )
                tlx.async_descriptor_load(
                    desc_q,
                    q_tiles[q_buf_id],
                    [
                        (q_off_y_base + curr_m).to(tl.int32),
                        x_offset,
                    ],
                    q_fulls[q_buf_id],
                )

                tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                tlx.barrier_expect_bytes(
                    do_fulls[do_buf_id],
                    DO_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM,
                )
                tlx.async_descriptor_load(
                    desc_do,
                    do_tiles[do_buf_id],
                    [
                        (q_off_y_base + curr_m).to(tl.int32),
                        x_offset,
                    ],
                    do_fulls[do_buf_id],
                )
                curr_m += step_m


# ---------------------------------------------------------------------------
# Python host wrapper
# ---------------------------------------------------------------------------


def tlx_block_attention_bwd(
    alpha: float,
    q_list: List[torch.Tensor],
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
    do_list: List[torch.Tensor],
    q_seq_offsets_list: List[torch.Tensor],
    mask_matrix: List[List[MaskType]],
    attn_scale_list: List[torch.Tensor],
    kv_seq_offsets_list: Optional[List[torch.Tensor]] = None,
    max_attn_len: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Backward pass for blocked MHA using TLX (non-persistent).

    Launches one kernel per non-NULL (qi, ki) pair. dQ/dK/dV outputs are
    pre-zeroed; each launch writes via async TMA reduce-add so cross-pair
    contributions accumulate atomically.

    Returns (dq_list, dk_list, dv_list) with the same shapes/dtypes as
    (q_list, k_list, v_list).
    """
    if kv_seq_offsets_list is None:
        kv_seq_offsets_list = q_seq_offsets_list

    assert HAS_TLX, "TLX is not available"

    num_q_tensors = len(q_list)
    num_kv_tensors = len(k_list)

    device = q_list[0].device
    B = q_seq_offsets_list[0].numel() - 1
    H = q_list[0].shape[1]
    DimQ = q_list[0].shape[2]
    DimV = v_list[0].shape[2]
    # First-draft kernel constraints.
    assert DimQ == DimV, "First-draft tlx bwd requires DimQ == DimV"

    q_list = [switch_to_contiguous_if_needed(q) for q in q_list]
    k_list = [switch_to_contiguous_if_needed(k) for k in k_list]
    v_list = [switch_to_contiguous_if_needed(v) for v in v_list]
    do_list = [switch_to_contiguous_if_needed(d) for d in do_list]

    dq_list = [torch.zeros_like(q) for q in q_list]
    dk_list = [torch.zeros_like(k) for k in k_list]
    dv_list = [torch.zeros_like(v) for v in v_list]

    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    dummy_block = [1, 1]

    def make_desc(tensor, shape, strides):
        return TensorDescriptor(
            tensor,
            shape=shape,
            strides=strides,
            block_shape=dummy_block,
        )

    desc_q_list = [
        make_desc(q, [q.shape[0], H * DimQ], [q.stride(0), 1]) for q in q_list
    ]
    desc_k_list = [
        make_desc(k, [k.shape[0], H * DimQ], [k.stride(0), 1]) for k in k_list
    ]
    desc_v_list = [
        make_desc(v, [v.shape[0], H * DimV], [v.stride(0), 1]) for v in v_list
    ]
    desc_do_list = [
        make_desc(d, [d.shape[0], H * DimV], [d.stride(0), 1]) for d in do_list
    ]
    desc_dq_list = [
        make_desc(d, [d.shape[0], H * DimQ], [d.stride(0), 1]) for d in dq_list
    ]
    desc_dk_list = [
        make_desc(d, [d.shape[0], H * DimQ], [d.stride(0), 1]) for d in dk_list
    ]
    desc_dv_list = [
        make_desc(d, [d.shape[0], H * DimV], [d.stride(0), 1]) for d in dv_list
    ]

    next_pow2_batch = triton.next_power_of_2(B)

    for qi in range(num_q_tensors):
        q_tensor = q_list[qi]
        q_seq_offsets = q_seq_offsets_list[qi]
        attn_scale = attn_scale_list[qi]
        if q_tensor.shape[0] == 0:
            continue

        desc_q = desc_q_list[qi]
        desc_do = desc_do_list[qi]
        desc_dq = desc_dq_list[qi]

        for ki in range(num_kv_tensors):
            mask_type = mask_matrix[qi][ki]
            if mask_type == MaskType.NULL:
                continue

            k_tensor = k_list[ki]
            v_tensor = v_list[ki]
            kv_seq_offsets = kv_seq_offsets_list[ki]
            if k_tensor.shape[0] == 0:
                continue

            desc_k = desc_k_list[ki]
            desc_v = desc_v_list[ki]
            desc_dk = desc_dk_list[ki]
            desc_dv = desc_dv_list[ki]

            total_kv = k_tensor.shape[0]
            # Varlen-aware grid: matches fwd's per-batch tile scan but for KV.
            grid = lambda meta, _tkv=total_kv, _b=B, _h=H: (  # noqa E731
                ((_tkv + _b * (meta["BLOCK_N1"] - 1)) // meta["BLOCK_N1"]),
                _h,
            )
            autotune_max_kv = triton.next_power_of_2(total_kv)

            _block_attn_bwd_ws[grid](
                desc_q=desc_q,
                desc_k=desc_k,
                desc_v=desc_v,
                desc_do=desc_do,
                desc_dq=desc_dq,
                desc_dk=desc_dk,
                desc_dv=desc_dv,
                alpha=alpha,
                Z=B,
                H=H,
                q_seq_offsets_tensor=q_seq_offsets,
                kv_seq_offsets_tensor=kv_seq_offsets,
                attn_scale_ptr=attn_scale,
                stride_q_so_b=q_seq_offsets.stride(0),
                stride_kv_so_b=kv_seq_offsets.stride(0),
                max_attn_len=max_attn_len,
                CUR_MASK=mask_type.value,
                HAS_MAX_ATTN_LEN=max_attn_len > 0,
                AUTOTUNE_MAX_KV_LEN=autotune_max_kv,
                NEXT_POW2_BATCH=next_pow2_batch,
                HEAD_DIM=DimQ,
            )

    return dq_list, dk_list, dv_list


# ---------------------------------------------------------------------------
# Autograd Function wrapping the existing tlx fwd + this new bwd.
# ---------------------------------------------------------------------------


class _TLXBlockAttentionFunction(torch.autograd.Function):
    """Autograd Function for the TLX blocked MHA fwd + bwd.

    The forward delegates to `tlx_block_attention_fwd` (already in-tree).
    The backward calls `tlx_block_attention_bwd`. `attn_scale` is not
    differentiated in this first draft.
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        alpha,
        max_attn_len,
        mask_matrix_tuple,
        num_q,
        num_kv,
        *tensors,
    ):
        # Unpack varargs: q_list ++ k_list ++ v_list ++ q_seq_offsets ++
        # kv_seq_offsets ++ attn_scale.
        idx = 0
        q_list = list(tensors[idx : idx + num_q])
        idx += num_q
        k_list = list(tensors[idx : idx + num_kv])
        idx += num_kv
        v_list = list(tensors[idx : idx + num_kv])
        idx += num_kv
        q_seq_offsets_list = list(tensors[idx : idx + num_q])
        idx += num_q
        kv_seq_offsets_list = list(tensors[idx : idx + num_kv])
        idx += num_kv
        attn_scale_list = list(tensors[idx : idx + num_q])

        # mask_matrix_tuple is num_q*num_kv ints (MaskType values), flattened
        # row-major. Reconstruct.
        mask_matrix = [
            [MaskType(mask_matrix_tuple[qi * num_kv + ki]) for ki in range(num_kv)]
            for qi in range(num_q)
        ]

        # Import lazily to avoid a circular import with tlx_block_attention.py.
        from hammer.v3.ops.triton.tlx_block_attention import tlx_block_attention_fwd

        out_list = tlx_block_attention_fwd(
            alpha=alpha,
            q_list=q_list,
            k_list=k_list,
            v_list=v_list,
            q_seq_offsets_list=q_seq_offsets_list,
            mask_matrix=mask_matrix,
            attn_scale_list=attn_scale_list,
            kv_seq_offsets_list=kv_seq_offsets_list,
            max_attn_len=max_attn_len,
        )

        ctx.save_for_backward(
            *q_list,
            *k_list,
            *v_list,
            *q_seq_offsets_list,
            *kv_seq_offsets_list,
            *attn_scale_list,
        )
        ctx.alpha = alpha
        ctx.max_attn_len = max_attn_len
        ctx.mask_matrix = mask_matrix
        ctx.num_q = num_q
        ctx.num_kv = num_kv

        return tuple(out_list)

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, *grad_outputs):
        num_q = ctx.num_q
        num_kv = ctx.num_kv
        saved = ctx.saved_tensors
        idx = 0
        q_list = list(saved[idx : idx + num_q])
        idx += num_q
        k_list = list(saved[idx : idx + num_kv])
        idx += num_kv
        v_list = list(saved[idx : idx + num_kv])
        idx += num_kv
        q_seq_offsets_list = list(saved[idx : idx + num_q])
        idx += num_q
        kv_seq_offsets_list = list(saved[idx : idx + num_kv])
        idx += num_kv
        attn_scale_list = list(saved[idx : idx + num_q])

        do_list = [g.contiguous() for g in grad_outputs]

        dq_list, dk_list, dv_list = tlx_block_attention_bwd(
            alpha=ctx.alpha,
            q_list=q_list,
            k_list=k_list,
            v_list=v_list,
            do_list=do_list,
            q_seq_offsets_list=q_seq_offsets_list,
            mask_matrix=ctx.mask_matrix,
            attn_scale_list=attn_scale_list,
            kv_seq_offsets_list=kv_seq_offsets_list,
            max_attn_len=ctx.max_attn_len,
        )

        # Order must mirror forward's tensor varargs order:
        # q_list, k_list, v_list, q_seq_offsets_list, kv_seq_offsets_list,
        # attn_scale_list.
        none_for_q_offsets = [None] * num_q
        none_for_kv_offsets = [None] * num_kv
        none_for_scale = [None] * num_q
        return (
            None,  # alpha
            None,  # max_attn_len
            None,  # mask_matrix_tuple
            None,  # num_q
            None,  # num_kv
            *dq_list,
            *dk_list,
            *dv_list,
            *none_for_q_offsets,
            *none_for_kv_offsets,
            *none_for_scale,
        )


def tlx_mha_with_grad(
    alpha: float,
    q_list: List[torch.Tensor],
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
    q_seq_offsets_list: List[torch.Tensor],
    mask_matrix: List[List[MaskType]],
    attn_scale_list: List[torch.Tensor],
    kv_seq_offsets_list: Optional[List[torch.Tensor]] = None,
    max_attn_len: int = 0,
) -> List[torch.Tensor]:
    """Differentiable TLX blocked MHA: forward + backward.

    Use this in place of `tlx_mha` (which is forward-only) when you need
    gradients with respect to q/k/v.
    """
    if kv_seq_offsets_list is None:
        kv_seq_offsets_list = q_seq_offsets_list

    num_q = len(q_list)
    num_kv = len(k_list)

    mask_matrix_tuple = tuple(
        mask_matrix[qi][ki].value for qi in range(num_q) for ki in range(num_kv)
    )

    out = _TLXBlockAttentionFunction.apply(
        alpha,
        max_attn_len,
        mask_matrix_tuple,
        num_q,
        num_kv,
        *q_list,
        *k_list,
        *v_list,
        *q_seq_offsets_list,
        *kv_seq_offsets_list,
        *attn_scale_list,
    )
    return list(out)
