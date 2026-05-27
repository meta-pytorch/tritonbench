# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
TLX block attention kernel for SMALL Q-length workloads (q_tgt @ k_*).

Mirrors the structure of tlx_block_attention.py with one key swap: q rows live
on the tcgen05 N axis (which supports N as small as 32) instead of the M axis
(which has a hard floor of 64). This dodges the 80%+ MMA waste on padded q rows
when q_len ~= 11.

MMA shapes (BLOCK_K=256 fixed, BLOCK_K_SPLIT=128, BLOCK_Q in {32, 64}):
  - K @ Q^T  (replaces Q @ K^T):  M=BLOCK_K_SPLIT=128, N=BLOCK_Q, K=DimQ=128
    Output: S^T tile of shape [BLOCK_K_SPLIT, BLOCK_Q] in TMEM (one per cid_K).
  - V^T @ P^T (replaces P @ V):   M=DimV=128, N=BLOCK_Q, K=BLOCK_K_SPLIT=128
    Output: O^T tile of shape [DimV, BLOCK_Q] in TMEM (transposed accumulator).

NUM_MMA_GROUPS_K = 2 splits BLOCK_K=256 into two halves (cid_K=0, cid_K=1).
Per outer kv-step (BLOCK_K=256 wide), the MMA partition issues:
  - 2x K @ Q^T (one per cid_K)
  - 2x V^T @ P^T (one per cid_K, summing into the same acc tile)

Two silu+correction partitions, one per cid_K, mirror the standard kernel's
cid=0 / cid=1 split. Correction transposes acc[DimV, BLOCK_Q] -> o_tile via
DimV-axis split between the two partitions.
"""

from typing import List

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

from hammer.v2.ops.triton.template.triton_attention_utils import _get_bufidx_phase
from hammer.v3.ops.pytorch.pt_attention import MaskType
from hammer.v3.ops.triton.triton_inline_asm_utils import (
    _fast_silu_pre_halved,
    _mul_f32x2,
)

# @manual=//triton:triton
from triton.tools.tensor_descriptor import TensorDescriptor


MASK_CAUSAL = MaskType.CAUSAL.value
MASK_ALL = MaskType.ALL.value
MASK_DIAGONAL = MaskType.DIAGONAL.value
MASK_NULL = MaskType.NULL.value
MASK_LOCAL = MaskType.LOCAL.value


# ---------------------------------------------------------------------------
# Autotuning configs (small-Q variant).
#
# BLOCK_K is fixed at 256 so that BLOCK_K_SPLIT = 128 satisfies the tcgen05
# M-dim floor (>=64) for the K @ Q^T MMA in each cid_K partition.
# BLOCK_Q in {32, 64}: 32 is the smallest tcgen05 N supported by tlx.async_dot.
# ---------------------------------------------------------------------------


def _host_descriptor_pre_hook(nargs) -> None:
    BLOCK_K_SPLIT = nargs["BLOCK_K"] // 2
    BLOCK_Q = nargs["BLOCK_Q"]
    DimQ = nargs["DimQ"]
    DimV = nargs["DimV"]
    for key in list(nargs.keys()):
        if not isinstance(nargs[key], TensorDescriptor):
            continue
        if key.startswith("desc_q") or key.startswith("desc_o"):
            nargs[key].block_shape = [
                BLOCK_Q,
                DimQ if key.startswith("desc_q") else DimV,
            ]
        elif key.startswith("desc_k"):
            nargs[key].block_shape = [BLOCK_K_SPLIT, DimQ]
        elif key.startswith("desc_v"):
            nargs[key].block_shape = [BLOCK_K_SPLIT, DimV]


def _get_fwd_configs() -> List[triton.Config]:
    return [
        triton.Config(
            {
                "BLOCK_K": 256,  # fixed - split into 2 halves of 128 each
                "BLOCK_Q": BLOCK_Q,
                "NUM_BUFFERS_Q": 1,
                # 3 KV buffers = 6 slots total (NUM_MMA_GROUPS_K * NUM_BUFFERS_KV).
                # Each outer step consumes 4 slots (2K + 2V); the deferred V
                # pattern holds V[1] tiles for 1 extra iter, so peak in-flight
                # is ~5-6 slots. NUM_BUFFERS_KV=3 fits SMEM (192 KB for kv +
                # ~32 KB for q/p/o tiles = ~224 KB).
                "NUM_BUFFERS_KV": NUM_BUFFERS_KV,
                "NUM_REGS_ACT": 196,
                "NUM_REGS_MMA": 24,
                "NUM_REGS_LOAD": 24,
                "NUM_REGS_EPI": 48,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        )
        for BLOCK_Q in [32, 64]
        for NUM_BUFFERS_KV in [2, 3]
    ]


# ---------------------------------------------------------------------------
# Helper: kv-loop bounds for the swapped (small-Q) kernel.
# Loop iterates in BLOCK_K (= 256) steps; each step processes 2 K-tile halves.
# ---------------------------------------------------------------------------


@triton.jit
def _small_q_kv_bounds(
    cur_mask,
    q_tile_start_local,
    q_t_seq_len,
    kv_t_seq_len,
    kv_block_len,
    max_attn_len,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    """Compute (kv_loop_start, left_mask_end, unmasked_end, kv_loop_end).

    Mirror of _block_attn_kv_bounds in tlx_block_attention.py with M<->Q,
    N<->K, start_m_local<->q_tile_start_local. The kv loop is split into up
    to four contiguous regions:
        [kv_loop_start, left_mask_end)  - top-mask only (LOCAL window-left)
        [left_mask_end, unmasked_end)   - fully unmasked middle (no mask op)
        [unmasked_end,  kv_loop_end)    - bottom-mask (CAUSAL boundary; for
                                          LOCAL this region applies BOTH masks
                                          since top/bottom may overlap on
                                          narrow windows)

    For non-LOCAL masks, left_mask_end == kv_loop_start so the leading region
    is empty.

    The loop steps in BLOCK_K (= 256) and each outer iter processes both
    cid_K halves (BLOCK_K_SPLIT = 128 each). Region boundaries are
    BLOCK_K-aligned; when only one half of an outer iter actually needs
    masking, the bottom-mask helper reduces to a no-op (row_limit beyond
    BLOCK_K_SPLIT keeps every row).
    """
    kv_loop_start = 0
    kv_loop_end = kv_block_len

    # max_valid_k_pos is "one past the last valid n" - last q in tile is
    # q_tile_start_local + BLOCK_Q - 1, so last valid k is max_valid_k_pos - 1.
    # Tight ceiling is `((max_valid_k_pos + BLOCK_K - 1) // BLOCK_K) * BLOCK_K`.
    # Same logic for the DIAGONAL upper bound on (q_tile_start_local + BLOCK_Q).
    if cur_mask == MASK_CAUSAL:
        delta = kv_t_seq_len - q_t_seq_len
        max_valid_k_pos = q_tile_start_local + BLOCK_Q + delta
        kv_loop_end = tl.minimum(
            kv_block_len,
            ((max_valid_k_pos + BLOCK_K - 1) // BLOCK_K) * BLOCK_K,
        )
    elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
        delta = kv_t_seq_len - q_t_seq_len
        min_valid_k_pos = q_tile_start_local + delta - max_attn_len
        min_valid_k_pos = tl.maximum(min_valid_k_pos, 0)
        kv_loop_start = (min_valid_k_pos // BLOCK_K) * BLOCK_K
        max_valid_k_pos = q_tile_start_local + BLOCK_Q + delta
        kv_loop_end = tl.minimum(
            kv_block_len,
            ((max_valid_k_pos + BLOCK_K - 1) // BLOCK_K) * BLOCK_K,
        )
    elif cur_mask == MASK_DIAGONAL:
        kv_loop_start = tl.maximum(0, (q_tile_start_local // BLOCK_K) * BLOCK_K)
        kv_loop_end = tl.minimum(
            kv_block_len,
            ((q_tile_start_local + BLOCK_Q + BLOCK_K - 1) // BLOCK_K) * BLOCK_K,
        )

    # Compute unmasked_end (where the trailing bottom-mask region starts).
    # Use the smallest q's max valid k = q_tile_start_local + delta. A block
    # at start_k is fully unmasked iff start_k + BLOCK_K - 1 <= that bound.
    if cur_mask == MASK_CAUSAL or (HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL):
        delta = kv_t_seq_len - q_t_seq_len
        unmasked_end = (q_tile_start_local + delta - BLOCK_K + 1).to(tl.int32)
        unmasked_end = (unmasked_end // BLOCK_K) * BLOCK_K
        unmasked_end = tl.minimum(
            tl.maximum(unmasked_end, kv_loop_start.to(tl.int32)),
            kv_loop_end.to(tl.int32),
        )
    elif cur_mask == MASK_ALL:
        unmasked_end = kv_loop_end.to(tl.int32)
    elif cur_mask == MASK_DIAGONAL:
        unmasked_end = kv_loop_start.to(tl.int32)
    else:
        unmasked_end = kv_loop_end.to(tl.int32)

    # Compute left_mask_end (where the leading window-left mask region ends).
    # Defined only for LOCAL; the smallest BLOCK_K-aligned start_k satisfying
    # `start_k >= max q's L_row` (where L_row = q + 1 - max_attn_len).
    if HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
        delta = kv_t_seq_len - q_t_seq_len
        threshold_left = q_tile_start_local + BLOCK_Q + delta - max_attn_len
        left_mask_end = ((threshold_left + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
        left_mask_end = tl.minimum(
            tl.maximum(left_mask_end, kv_loop_start.to(tl.int32)), unmasked_end
        )
    else:
        left_mask_end = kv_loop_start.to(tl.int32)

    # Ensure unmasked_end only covers full blocks (partial last block goes
    # through the bottom-mask region with kv_block_len clamping).
    kv_seq_aligned = (kv_block_len.to(tl.int32) // BLOCK_K) * BLOCK_K
    unmasked_end = tl.minimum(unmasked_end, kv_seq_aligned)

    return kv_loop_start, left_mask_end, unmasked_end, kv_loop_end


# ---------------------------------------------------------------------------
# Helpers: R2P-style per-element mask zeroing for the transposed layout.
#
# Mirrors _apply_causal_mask_zero / _apply_left_mask_zero in
# tlx_block_attention.py, except the bit-pack runs along the ROW axis (k)
# instead of the COL axis (q). Per-column row limits replace per-row col
# limits. Same constexpr-unrolled bit-test pattern lowers to PTX r2p
# (Register-to-Predicate) + predicated selp.
# ---------------------------------------------------------------------------


@triton.jit
def _mask_scalar_zero_bottom(qk, row_limit_bottom, s, i):
    row_lim_bot_s = row_limit_bottom - s
    row_lim_bot_cur = max(row_lim_bot_s, 0)
    mask = -1 << row_lim_bot_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return tl.where(mask_i_bit, qk, 0.0)


@triton.jit
def _apply_bottom_mask_zero(qk, row_limit_bottom, BLOCK_K_SPLIT: tl.constexpr):
    # Per-col row_limit_bottom shape: [1, BLOCK_Q] (or scalar - broadcasts).
    # Bit chunks of 16 match what r2p.b32 unpacks per call.
    offs_k = tl.arange(0, BLOCK_K_SPLIT)[:, None]
    s = offs_k & ~0xF
    i = offs_k & 0xF
    return tl.map_elementwise(_mask_scalar_zero_bottom, qk, row_limit_bottom, s, i)


# Top-side mirror: zero rows where (i + s) < row_limit_top.
@triton.jit
def _mask_scalar_zero_top(qk, row_limit_top, s, i):
    row_lim_top_s = row_limit_top - s
    row_lim_top_cur = max(row_lim_top_s, 0)
    # Bits 0..row_lim_top_cur-1 set -> zero region.
    mask_bits = (1 << row_lim_top_cur) - 1
    mask_i_bit = (mask_bits & (1 << i)) == 0
    return tl.where(mask_i_bit, qk, 0.0)


@triton.jit
def _apply_top_mask_zero(qk, row_limit_top, BLOCK_K_SPLIT: tl.constexpr):
    # Per-col row_limit_top shape: [1, BLOCK_Q] (or scalar - broadcasts).
    offs_k = tl.arange(0, BLOCK_K_SPLIT)[:, None]
    s = offs_k & ~0xF
    i = offs_k & 0xF
    return tl.map_elementwise(_mask_scalar_zero_top, qk, row_limit_top, s, i)


# ---------------------------------------------------------------------------
# Per-block mask dispatcher for the transposed layout. Each mask type
# collapses to a per-col bottom limit B (zero rows >= B) and an optional
# per-col top limit T (zero rows < T).
# ---------------------------------------------------------------------------


@triton.jit
def _apply_block_mask_T(
    s_T,
    cur_mask,
    offs_q_local,
    slice_start_k,
    q_t_seq_len,
    kv_t_seq_len,
    kv_block_len,
    max_attn_len,
    BLOCK_K_SPLIT: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    if cur_mask == MASK_CAUSAL:
        delta = kv_t_seq_len - q_t_seq_len
        slice_B = (offs_q_local + delta - slice_start_k + 1)[None, :]
        return _apply_bottom_mask_zero(s_T, slice_B, BLOCK_K_SPLIT)
    elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
        delta = kv_t_seq_len - q_t_seq_len
        q_pos = offs_q_local + delta
        kv_lim_slice = kv_block_len.to(tl.int32) - slice_start_k
        slice_B = tl.minimum(q_pos + 1 - slice_start_k, kv_lim_slice)[None, :]
        slice_T = (q_pos + 1 - max_attn_len - slice_start_k)[None, :]
        s_T = _apply_bottom_mask_zero(s_T, slice_B, BLOCK_K_SPLIT)
        return _apply_top_mask_zero(s_T, slice_T, BLOCK_K_SPLIT)
    elif cur_mask == MASK_DIAGONAL:
        # DIAGONAL: keep only k == q. Two-sided bracket around the diagonal.
        kv_lim_slice = kv_block_len.to(tl.int32) - slice_start_k
        slice_B = tl.minimum(offs_q_local + 1 - slice_start_k, kv_lim_slice)[None, :]
        slice_T = (offs_q_local - slice_start_k)[None, :]
        s_T = _apply_bottom_mask_zero(s_T, slice_B, BLOCK_K_SPLIT)
        return _apply_top_mask_zero(s_T, slice_T, BLOCK_K_SPLIT)
    elif cur_mask == MASK_ALL:
        # Only invoked for the partial last KV block; full blocks go through
        # the unmasked path. Mask rows >= remaining kv length (broadcast).
        kv_lim_slice = kv_block_len.to(tl.int32) - slice_start_k
        slice_B = (tl.zeros([BLOCK_Q], tl.int32) + kv_lim_slice)[None, :]
        return _apply_bottom_mask_zero(s_T, slice_B, BLOCK_K_SPLIT)
    else:
        # MASK_NULL (or unhandled): zero everything.
        return _apply_bottom_mask_zero(s_T, 0, BLOCK_K_SPLIT)


# ---------------------------------------------------------------------------
# Small-Q forward kernel (warp-specialized, transposed-MMA orientation,
# 2 silu+correction partitions split across BLOCK_K halves).
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_fwd_configs(),
    key=["AUTOTUNE_TOTAL_Q", "DimQ", "CUR_MASK"],
)
@triton.jit
def _block_attn_fwd_small_q(  # noqa: C901
    Out,
    desc_q_0,
    desc_k_0,
    desc_v_0,
    alpha,
    Z,
    H,
    q_seq_offsets_tensor,
    kv_seq_offsets_tensor,
    attn_scale_ptr,
    stride_q_so_b,
    stride_kv_so_b,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_oh,
    max_attn_len,
    CUR_MASK: tl.constexpr,
    # When True (first KV block for this Q): TMA store directly into Out.
    # When False: read-modify-write accumulate into Out (non-atomic; host
    # serializes launches). Lets the wrapper allocate the output buffer once.
    IS_FIRST_K: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    # Autotune key - bucketed total-Q (next pow2 of total Q tokens across all
    # batches). Avoids the host-GPU sync of computing max_q_len; total_q is
    # just the q tensor's shape[0].
    AUTOTUNE_TOTAL_Q: tl.constexpr,
    # Next power-of-2 of batch size, sized for the per-CTA seqlen scan that
    # decodes (batch, q_tile) on-device from q_seq_offsets.
    NEXT_POW2_BATCH: tl.constexpr,
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_REGS_ACT: tl.constexpr,
    NUM_REGS_MMA: tl.constexpr,
    NUM_REGS_LOAD: tl.constexpr,
    NUM_REGS_EPI: tl.constexpr,
):
    tl.static_assert(DimV == DimQ)
    tl.static_assert(BLOCK_K == 256)  # split into 2 halves of 128 each

    NUM_MMA_GROUPS_K: tl.constexpr = 2  # pyre-ignore[9]
    BLOCK_K_SPLIT: tl.constexpr = BLOCK_K // NUM_MMA_GROUPS_K  # pyre-ignore[9]

    # Named barriers - separate ID block per cid_K so the two silu+correction
    # partitions can independently signal the mma and epilog partitions.
    # P_FULL_BAR_BASE  + cid_K: silu cid_K -> mma  (silu wrote p_tiles[cid_K])
    # O_FULLS_BAR_BASE + cid_K: correction cid_K -> epilog
    # All counts = silu(128) + mma_or_epi(32) = 160.
    # Note: no P_EMPTY needed - the MMA queue serializes K[cid_K]@Q^T_{k+1}
    # after V@P^T_k, which itself only fires after silu_k signaled p_full and
    # the MMA read p_tiles. So silu_{k+1}'s wake-up (gated by s_full[cid_K]
    # from K@Q^T_{k+1}) is naturally ordered after MMA's p_tiles read.
    P_FULL_BAR_BASE: tl.constexpr = 9  # pyre-ignore[9]   IDs 9, 10
    P_FULL_BAR_COUNT: tl.constexpr = 160  # pyre-ignore[9]
    O_FULLS_BAR_BASE: tl.constexpr = 14  # pyre-ignore[9]  IDs 14, 15
    O_FULLS_BAR_COUNT: tl.constexpr = 160  # pyre-ignore[9]

    # Grid is `(total_q + B*(BLOCK_Q-1))//BLOCK_Q * H` (varlen-aware), so each
    # CTA decodes its (batch, q_tile_within_batch) by scanning q_seq_offsets
    # on-device - no host-side max_q_len sync needed.
    tile_idx = tl.program_id(0)
    tile_per_h = tile_idx // H
    off_h = tile_idx % H

    # Compute n_tiles per batch in parallel (load both ends of each batch
    # range, derive seqlen, ceil-div by BLOCK_Q).
    batches = tl.arange(0, NEXT_POW2_BATCH)
    batch_mask = batches < Z
    q_seq_b = tl.load(
        q_seq_offsets_tensor + batches.to(tl.int64) * stride_q_so_b,
        mask=batch_mask,
        other=0,
    )
    q_seq_b1 = tl.load(
        q_seq_offsets_tensor + (batches + 1).to(tl.int64) * stride_q_so_b,
        mask=batch_mask,
        other=0,
    )
    seqlens = (q_seq_b1 - q_seq_b).to(tl.int32)
    n_tiles = (seqlens + BLOCK_Q - 1) // BLOCK_Q
    n_tiles = tl.where(batch_mask, n_tiles, 0)

    # cum_tiles[b] = total q-tiles for batches [0..b]. Find batch_idx as the
    # smallest b such that cum_tiles[b] > tile_per_h.
    cum_tiles = tl.cumsum(n_tiles, axis=0)
    le_mask = cum_tiles <= tile_per_h
    off_z = tl.sum(le_mask.to(tl.int32))
    prev_cum = tl.sum(tl.where(le_mask, n_tiles, 0))
    pid_q = tile_per_h - prev_cum

    # Early exit: this CTA's tile_idx is past total useful work.
    if off_z >= Z:
        return

    off_z_i64 = off_z.to(tl.int64)
    q_tile_start_local = pid_q * BLOCK_Q

    q_seq_start = tl.load(q_seq_offsets_tensor + off_z_i64 * stride_q_so_b).to(tl.int64)
    q_seq_end = tl.load(q_seq_offsets_tensor + (off_z_i64 + 1) * stride_q_so_b).to(
        tl.int64
    )
    q_block_len = q_seq_end - q_seq_start

    kv_seq_start = tl.load(kv_seq_offsets_tensor + off_z_i64 * stride_kv_so_b).to(
        tl.int32
    )
    kv_seq_end = tl.load(kv_seq_offsets_tensor + (off_z_i64 + 1) * stride_kv_so_b).to(
        tl.int32
    )

    if kv_seq_end <= kv_seq_start:
        return

    # SMEM allocations. Single-buffered s/p_tiles per cid_K - the MMA queue's
    # serial execution provides the implicit handshake: silu_{k+1}'s wake-up
    # condition (s_full[cid_K] from K[cid_K]@Q^T_{k+1}) is naturally ordered
    # after MMA's read of p_tiles_k via the V@P^T_k -> K@Q^T_{k+1} queue
    # ordering chain.
    q_tile = tlx.local_alloc((BLOCK_Q, DimQ), tl.bfloat16, NUM_BUFFERS_Q)
    kv_tiles = tlx.local_alloc(
        (BLOCK_K_SPLIT, DimV),
        tl.bfloat16,
        NUM_MMA_GROUPS_K * NUM_BUFFERS_KV,
    )
    p_tiles = tlx.local_alloc((BLOCK_K_SPLIT, BLOCK_Q), tl.bfloat16, NUM_MMA_GROUPS_K)
    o_tile = tlx.local_alloc((BLOCK_Q, DimV), tl.bfloat16, 1)

    q_full = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS_K * NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS_K * NUM_BUFFERS_KV)

    # TMEM allocations.
    s_tiles = tlx.local_alloc(
        (BLOCK_K_SPLIT, BLOCK_Q),
        tl.float32,
        NUM_MMA_GROUPS_K,
        tlx.storage_kind.tmem,
    )
    acc_tile = tlx.local_alloc((DimV, BLOCK_Q), tl.float32, 1, tlx.storage_kind.tmem)

    s_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS_K)
    acc_empty = tlx.alloc_barriers(num_barriers=1)

    with tlx.async_tasks():
        # =================================================================
        # silu + correction for cid_K=0 (default partition, num_warps=4)
        # =================================================================
        with tlx.async_task("default"):
            cid_K: tl.constexpr = 0  # pyre-ignore[9]
            q_t_seq_len = (q_seq_end - q_seq_start).to(tl.int32)
            offs_q_local = q_tile_start_local + tl.arange(0, BLOCK_Q)

            attn_scale_offs = q_seq_start + offs_q_local.to(tl.int64)
            row_scale = tl.load(
                attn_scale_ptr + attn_scale_offs,
                mask=offs_q_local < q_block_len,
                other=1.0,
            )
            combined_scale_half = (alpha * 0.5) * row_scale

            kv_t_seq_len = kv_seq_end - kv_seq_start
            kv_block_len = kv_t_seq_len.to(tl.int64)
            accum_cnt_s = 0

            kv_loop_start, left_mask_end, unmasked_end, kv_loop_end = (
                _small_q_kv_bounds(
                    CUR_MASK,
                    q_tile_start_local,
                    q_t_seq_len,
                    kv_t_seq_len,
                    kv_block_len,
                    max_attn_len,
                    BLOCK_Q,
                    BLOCK_K,
                    HAS_MAX_ATTN_LEN,
                )
            )

            _loop_start = kv_loop_start.to(tl.int32)
            _loop_end = kv_loop_end.to(tl.int32)
            _left_end = left_mask_end.to(tl.int32)
            _unmasked_end = unmasked_end.to(tl.int32)

            # Region 1: top-mask only (LOCAL window-left edge).
            if HAS_MAX_ATTN_LEN and CUR_MASK == MASK_LOCAL:
                delta = kv_t_seq_len - q_t_seq_len
                q_pos = offs_q_local + delta
                for start_k in range(_loop_start, _left_end, BLOCK_K):
                    _, s_phase = _get_bufidx_phase(accum_cnt_s, 1)
                    tlx.barrier_wait(s_fulls[cid_K], s_phase)
                    s_T = tlx.local_load(s_tiles[cid_K])
                    s_T = _mul_f32x2(s_T, combined_scale_half[None, :])
                    start_k_cid = start_k + cid_K * BLOCK_K_SPLIT
                    slice_T = (q_pos + 1 - max_attn_len - start_k_cid)[None, :]
                    s_T = _apply_top_mask_zero(s_T, slice_T, BLOCK_K_SPLIT)
                    p_T = _fast_silu_pre_halved(s_T)
                    tlx.local_store(p_tiles[cid_K], p_T.to(tl.bfloat16))
                    tlx.named_barrier_arrive(P_FULL_BAR_BASE + cid_K, P_FULL_BAR_COUNT)
                    accum_cnt_s += 1

            # Region 2: fully unmasked middle.
            for _ in range(_left_end, _unmasked_end, BLOCK_K):
                _, s_phase = _get_bufidx_phase(accum_cnt_s, 1)
                tlx.barrier_wait(s_fulls[cid_K], s_phase)
                s_T = tlx.local_load(s_tiles[cid_K])
                s_T = _mul_f32x2(s_T, combined_scale_half[None, :])
                p_T = _fast_silu_pre_halved(s_T)
                tlx.local_store(p_tiles[cid_K], p_T.to(tl.bfloat16))
                tlx.named_barrier_arrive(P_FULL_BAR_BASE + cid_K, P_FULL_BAR_COUNT)
                accum_cnt_s += 1

            # Region 3: bottom-mask (CAUSAL boundary; for LOCAL also top-mask
            # when window narrows enough that top/bottom blocks overlap).
            for start_k in range(_unmasked_end, _loop_end, BLOCK_K):
                _, s_phase = _get_bufidx_phase(accum_cnt_s, 1)
                tlx.barrier_wait(s_fulls[cid_K], s_phase)
                s_T = tlx.local_load(s_tiles[cid_K])
                s_T = _mul_f32x2(s_T, combined_scale_half[None, :])
                start_k_cid = start_k + cid_K * BLOCK_K_SPLIT
                s_T = _apply_block_mask_T(
                    s_T,
                    CUR_MASK,
                    offs_q_local,
                    start_k_cid,
                    q_t_seq_len,
                    kv_t_seq_len,
                    kv_block_len,
                    max_attn_len,
                    BLOCK_K_SPLIT,
                    BLOCK_Q,
                    HAS_MAX_ATTN_LEN,
                )
                p_T = _fast_silu_pre_halved(s_T)
                tlx.local_store(p_tiles[cid_K], p_T.to(tl.bfloat16))
                tlx.named_barrier_arrive(P_FULL_BAR_BASE + cid_K, P_FULL_BAR_COUNT)
                accum_cnt_s += 1

            # Correction: this partition handles cols 0..BLOCK_Q/2 of acc.
            # tlx.subslice slices along the tcgen05 N axis (= BLOCK_Q here).
            tlx.barrier_wait(acc_empty[0], 0)
            acc_half = tlx.subslice(acc_tile[0], 0, BLOCK_Q // 2)
            acc = tlx.local_load(acc_half)  # [DimV, BLOCK_Q/2]
            acc_T = tl.trans(acc).to(tl.bfloat16)  # [BLOCK_Q/2, DimV]
            o_slice = tlx.local_slice(o_tile[0], [0, 0], [BLOCK_Q // 2, DimV])
            tlx.local_store(o_slice, acc_T)
            tlx.named_barrier_arrive(O_FULLS_BAR_BASE + cid_K, O_FULLS_BAR_COUNT)

        # =================================================================
        # silu + correction for cid_K=1 (second partition, num_warps=4)
        # =================================================================
        with tlx.async_task(num_warps=4, registers=NUM_REGS_ACT):
            cid_K: tl.constexpr = 1  # pyre-ignore[9]
            q_t_seq_len = (q_seq_end - q_seq_start).to(tl.int32)
            offs_q_local = q_tile_start_local + tl.arange(0, BLOCK_Q)

            attn_scale_offs = q_seq_start + offs_q_local.to(tl.int64)
            row_scale = tl.load(
                attn_scale_ptr + attn_scale_offs,
                mask=offs_q_local < q_block_len,
                other=1.0,
            )
            combined_scale_half = (alpha * 0.5) * row_scale

            kv_t_seq_len = kv_seq_end - kv_seq_start
            kv_block_len = kv_t_seq_len.to(tl.int64)
            accum_cnt_s = 0

            kv_loop_start, left_mask_end, unmasked_end, kv_loop_end = (
                _small_q_kv_bounds(
                    CUR_MASK,
                    q_tile_start_local,
                    q_t_seq_len,
                    kv_t_seq_len,
                    kv_block_len,
                    max_attn_len,
                    BLOCK_Q,
                    BLOCK_K,
                    HAS_MAX_ATTN_LEN,
                )
            )

            _loop_start = kv_loop_start.to(tl.int32)
            _loop_end = kv_loop_end.to(tl.int32)
            _left_end = left_mask_end.to(tl.int32)
            _unmasked_end = unmasked_end.to(tl.int32)

            # Region 1: top-mask only (LOCAL window-left edge).
            if HAS_MAX_ATTN_LEN and CUR_MASK == MASK_LOCAL:
                delta = kv_t_seq_len - q_t_seq_len
                q_pos = offs_q_local + delta
                for start_k in range(_loop_start, _left_end, BLOCK_K):
                    _, s_phase = _get_bufidx_phase(accum_cnt_s, 1)
                    tlx.barrier_wait(s_fulls[cid_K], s_phase)
                    s_T = tlx.local_load(s_tiles[cid_K])
                    s_T = _mul_f32x2(s_T, combined_scale_half[None, :])
                    start_k_cid = start_k + cid_K * BLOCK_K_SPLIT
                    slice_T = (q_pos + 1 - max_attn_len - start_k_cid)[None, :]
                    s_T = _apply_top_mask_zero(s_T, slice_T, BLOCK_K_SPLIT)
                    p_T = _fast_silu_pre_halved(s_T)
                    tlx.local_store(p_tiles[cid_K], p_T.to(tl.bfloat16))
                    tlx.named_barrier_arrive(P_FULL_BAR_BASE + cid_K, P_FULL_BAR_COUNT)
                    accum_cnt_s += 1

            # Region 2: fully unmasked middle.
            for _ in range(_left_end, _unmasked_end, BLOCK_K):
                _, s_phase = _get_bufidx_phase(accum_cnt_s, 1)
                tlx.barrier_wait(s_fulls[cid_K], s_phase)
                s_T = tlx.local_load(s_tiles[cid_K])
                s_T = _mul_f32x2(s_T, combined_scale_half[None, :])
                p_T = _fast_silu_pre_halved(s_T)
                tlx.local_store(p_tiles[cid_K], p_T.to(tl.bfloat16))
                tlx.named_barrier_arrive(P_FULL_BAR_BASE + cid_K, P_FULL_BAR_COUNT)
                accum_cnt_s += 1

            # Region 3: bottom-mask (CAUSAL boundary; for LOCAL also top-mask
            # when window narrows enough that top/bottom blocks overlap).
            for start_k in range(_unmasked_end, _loop_end, BLOCK_K):
                _, s_phase = _get_bufidx_phase(accum_cnt_s, 1)
                tlx.barrier_wait(s_fulls[cid_K], s_phase)
                s_T = tlx.local_load(s_tiles[cid_K])
                s_T = _mul_f32x2(s_T, combined_scale_half[None, :])
                start_k_cid = start_k + cid_K * BLOCK_K_SPLIT
                s_T = _apply_block_mask_T(
                    s_T,
                    CUR_MASK,
                    offs_q_local,
                    start_k_cid,
                    q_t_seq_len,
                    kv_t_seq_len,
                    kv_block_len,
                    max_attn_len,
                    BLOCK_K_SPLIT,
                    BLOCK_Q,
                    HAS_MAX_ATTN_LEN,
                )
                p_T = _fast_silu_pre_halved(s_T)
                tlx.local_store(p_tiles[cid_K], p_T.to(tl.bfloat16))
                tlx.named_barrier_arrive(P_FULL_BAR_BASE + cid_K, P_FULL_BAR_COUNT)
                accum_cnt_s += 1

            # Correction: this partition handles cols BLOCK_Q/2..BLOCK_Q of acc.
            tlx.barrier_wait(acc_empty[0], 0)
            acc_half = tlx.subslice(acc_tile[0], BLOCK_Q // 2, BLOCK_Q // 2)
            acc = tlx.local_load(acc_half)  # [DimV, BLOCK_Q/2]
            acc_T = tl.trans(acc).to(tl.bfloat16)  # [BLOCK_Q/2, DimV]
            o_slice = tlx.local_slice(
                o_tile[0], [BLOCK_Q // 2, 0], [BLOCK_Q // 2, DimV]
            )
            tlx.local_store(o_slice, acc_T)
            tlx.named_barrier_arrive(O_FULLS_BAR_BASE + cid_K, O_FULLS_BAR_COUNT)

        # =================================================================
        # MMA partition: 2 K@Q^T + 2 V^T@P^T per outer kv-step
        # =================================================================
        with tlx.async_task(num_warps=1, registers=NUM_REGS_MMA):
            q_t_seq_len = (q_seq_end - q_seq_start).to(tl.int32)
            kv_t_seq_len = kv_seq_end - kv_seq_start
            kv_block_len = kv_t_seq_len.to(tl.int64)

            tlx.barrier_wait(q_full[0], 0)
            q_T = tlx.local_trans(q_tile[0])  # [DimQ, BLOCK_Q]

            kv_loop_start, _, _, kv_loop_end = _small_q_kv_bounds(
                CUR_MASK,
                q_tile_start_local,
                q_t_seq_len,
                kv_t_seq_len,
                kv_block_len,
                max_attn_len,
                BLOCK_Q,
                BLOCK_K,
                HAS_MAX_ATTN_LEN,
            )

            accum_cnt_kv = 0  # advances by 2*NUM_MMA_GROUPS_K per outer step
            _loop_start = kv_loop_start.to(tl.int32)
            _loop_end = kv_loop_end.to(tl.int32)

            # ============ PEEL: iter 0 - K[0..1]@Q^T + V[0]@P^T[0]; defer V[1] ============
            k0_bufIdx, k0_phase = _get_bufidx_phase(
                accum_cnt_kv, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
            )
            k1_bufIdx, k1_phase = _get_bufidx_phase(
                accum_cnt_kv + 1, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
            )
            v0_bufIdx, v0_phase = _get_bufidx_phase(
                accum_cnt_kv + 2, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
            )
            v1_bufIdx, v1_phase = _get_bufidx_phase(
                accum_cnt_kv + 3, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
            )

            tlx.barrier_wait(kv_fulls[k0_bufIdx], k0_phase)
            tlx.async_dot(
                kv_tiles[k0_bufIdx],
                q_T,
                s_tiles[0],
                use_acc=False,
                mBarriers=[s_fulls[0], kv_empties[k0_bufIdx]],
            )
            tlx.barrier_wait(kv_fulls[k1_bufIdx], k1_phase)
            tlx.async_dot(
                kv_tiles[k1_bufIdx],
                q_T,
                s_tiles[1],
                use_acc=False,
                mBarriers=[s_fulls[1], kv_empties[k1_bufIdx]],
            )

            # V[0] @ P^T[0] for iter 0 (use_acc=False - initializes acc).
            tlx.named_barrier_wait(P_FULL_BAR_BASE + 0, P_FULL_BAR_COUNT)
            tlx.barrier_wait(kv_fulls[v0_bufIdx], v0_phase)
            v_T_0 = tlx.local_trans(kv_tiles[v0_bufIdx])
            tlx.async_dot(
                v_T_0,
                p_tiles[0],
                acc_tile[0],
                use_acc=False,
                mBarriers=[kv_empties[v0_bufIdx]],
            )

            # Defer V[1] @ P^T[1] for iter 0 to the first main-loop iter.
            v1_bufIdx_prev = v1_bufIdx
            v1_phase_prev = v1_phase
            accum_cnt_kv += 4

            # ============ MAIN LOOP: iters 1..N-1, pipelined ============
            for _ in range(_loop_start + BLOCK_K, _loop_end, BLOCK_K):
                k0_bufIdx, k0_phase = _get_bufidx_phase(
                    accum_cnt_kv, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
                )
                k1_bufIdx, k1_phase = _get_bufidx_phase(
                    accum_cnt_kv + 1, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
                )
                v0_bufIdx, v0_phase = _get_bufidx_phase(
                    accum_cnt_kv + 2, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
                )
                v1_bufIdx, v1_phase = _get_bufidx_phase(
                    accum_cnt_kv + 3, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
                )

                # K[0] @ Q^T (current iter)
                tlx.barrier_wait(kv_fulls[k0_bufIdx], k0_phase)
                tlx.async_dot(
                    kv_tiles[k0_bufIdx],
                    q_T,
                    s_tiles[0],
                    use_acc=False,
                    mBarriers=[s_fulls[0], kv_empties[k0_bufIdx]],
                )

                # V[1] @ P^T[1] from PREV iter (deferred). use_acc=True.
                tlx.named_barrier_wait(P_FULL_BAR_BASE + 1, P_FULL_BAR_COUNT)
                tlx.barrier_wait(kv_fulls[v1_bufIdx_prev], v1_phase_prev)
                v_T_1_prev = tlx.local_trans(kv_tiles[v1_bufIdx_prev])
                tlx.async_dot(
                    v_T_1_prev,
                    p_tiles[1],
                    acc_tile[0],
                    use_acc=True,
                    mBarriers=[kv_empties[v1_bufIdx_prev]],
                )

                # K[1] @ Q^T (current iter)
                tlx.barrier_wait(kv_fulls[k1_bufIdx], k1_phase)
                tlx.async_dot(
                    kv_tiles[k1_bufIdx],
                    q_T,
                    s_tiles[1],
                    use_acc=False,
                    mBarriers=[s_fulls[1], kv_empties[k1_bufIdx]],
                )

                # V[0] @ P^T[0] (current iter). use_acc=True.
                tlx.named_barrier_wait(P_FULL_BAR_BASE + 0, P_FULL_BAR_COUNT)
                tlx.barrier_wait(kv_fulls[v0_bufIdx], v0_phase)
                v_T_0 = tlx.local_trans(kv_tiles[v0_bufIdx])
                tlx.async_dot(
                    v_T_0,
                    p_tiles[0],
                    acc_tile[0],
                    use_acc=True,
                    mBarriers=[kv_empties[v0_bufIdx]],
                )

                # Defer V[1] of current iter to next iter / tail.
                v1_bufIdx_prev = v1_bufIdx
                v1_phase_prev = v1_phase
                accum_cnt_kv += 4

            # ============ TAIL: final V[1] @ P^T[1] ============
            tlx.named_barrier_wait(P_FULL_BAR_BASE + 1, P_FULL_BAR_COUNT)
            tlx.barrier_wait(kv_fulls[v1_bufIdx_prev], v1_phase_prev)
            v_T_1_last = tlx.local_trans(kv_tiles[v1_bufIdx_prev])
            tlx.async_dot(
                v_T_1_last,
                p_tiles[1],
                acc_tile[0],
                use_acc=True,
                mBarriers=[kv_empties[v1_bufIdx_prev]],
            )

            tlx.tcgen05_commit(acc_empty[0])

        # =================================================================
        # Load partition: TMA loads of Q, K, V (BLOCK_K_SPLIT-sized tiles)
        # =================================================================
        with tlx.async_task(num_warps=1, registers=NUM_REGS_LOAD):
            q_t_seq_len = (q_seq_end - q_seq_start).to(tl.int32)

            qo_offset_y = q_seq_start + q_tile_start_local
            tlx.barrier_expect_bytes(q_full[0], 2 * BLOCK_Q * DimQ)
            tlx.async_descriptor_load(
                desc_q_0,
                q_tile[0],
                [qo_offset_y.to(tl.int32), off_h * stride_qh],
                q_full[0],
            )

            kv_t_seq_len = kv_seq_end - kv_seq_start
            kv_block_len = kv_t_seq_len.to(tl.int64)
            kv_loop_start, _, _, kv_loop_end = _small_q_kv_bounds(
                CUR_MASK,
                q_tile_start_local,
                q_t_seq_len,
                kv_t_seq_len,
                kv_block_len,
                max_attn_len,
                BLOCK_Q,
                BLOCK_K,
                HAS_MAX_ATTN_LEN,
            )

            accum_cnt_kv = 0
            kv_offset_y = kv_seq_start + kv_loop_start
            _loop_start = kv_loop_start.to(tl.int32)
            _loop_end = kv_loop_end.to(tl.int32)
            # Per outer step: load 2 K halves + 2 V halves = 4 tiles.
            for _ in range(_loop_start, _loop_end, BLOCK_K):
                # K[0]: rows [kv_offset_y, kv_offset_y + BLOCK_K_SPLIT)
                k0_bufIdx, k0_phase = _get_bufidx_phase(
                    accum_cnt_kv, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
                )
                tlx.barrier_wait(kv_empties[k0_bufIdx], k0_phase ^ 1)
                tlx.barrier_expect_bytes(kv_fulls[k0_bufIdx], 2 * BLOCK_K_SPLIT * DimQ)
                tlx.async_descriptor_load(
                    desc_k_0,
                    kv_tiles[k0_bufIdx],
                    [kv_offset_y.to(tl.int32), off_h * stride_kh],
                    kv_fulls[k0_bufIdx],
                )

                # K[1]: rows [kv_offset_y + BLOCK_K_SPLIT, kv_offset_y + BLOCK_K)
                k1_bufIdx, k1_phase = _get_bufidx_phase(
                    accum_cnt_kv + 1, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
                )
                tlx.barrier_wait(kv_empties[k1_bufIdx], k1_phase ^ 1)
                tlx.barrier_expect_bytes(kv_fulls[k1_bufIdx], 2 * BLOCK_K_SPLIT * DimQ)
                tlx.async_descriptor_load(
                    desc_k_0,
                    kv_tiles[k1_bufIdx],
                    [
                        (kv_offset_y + BLOCK_K_SPLIT).to(tl.int32),
                        off_h * stride_kh,
                    ],
                    kv_fulls[k1_bufIdx],
                )

                # V[0]: rows [kv_offset_y, kv_offset_y + BLOCK_K_SPLIT)
                v0_bufIdx, v0_phase = _get_bufidx_phase(
                    accum_cnt_kv + 2, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
                )
                tlx.barrier_wait(kv_empties[v0_bufIdx], v0_phase ^ 1)
                tlx.barrier_expect_bytes(kv_fulls[v0_bufIdx], 2 * BLOCK_K_SPLIT * DimV)
                tlx.async_descriptor_load(
                    desc_v_0,
                    kv_tiles[v0_bufIdx],
                    [kv_offset_y.to(tl.int32), off_h * stride_vh],
                    kv_fulls[v0_bufIdx],
                )

                # V[1]: rows [kv_offset_y + BLOCK_K_SPLIT, kv_offset_y + BLOCK_K)
                v1_bufIdx, v1_phase = _get_bufidx_phase(
                    accum_cnt_kv + 3, NUM_MMA_GROUPS_K * NUM_BUFFERS_KV
                )
                tlx.barrier_wait(kv_empties[v1_bufIdx], v1_phase ^ 1)
                tlx.barrier_expect_bytes(kv_fulls[v1_bufIdx], 2 * BLOCK_K_SPLIT * DimV)
                tlx.async_descriptor_load(
                    desc_v_0,
                    kv_tiles[v1_bufIdx],
                    [
                        (kv_offset_y + BLOCK_K_SPLIT).to(tl.int32),
                        off_h * stride_vh,
                    ],
                    kv_fulls[v1_bufIdx],
                )

                kv_offset_y += BLOCK_K
                accum_cnt_kv += 4

        # =================================================================
        # Epilog partition: TMA store of o_tile (after both correction halves)
        # =================================================================
        with tlx.async_task(num_warps=1, registers=NUM_REGS_EPI):
            qo_offset_y = q_seq_start + q_tile_start_local
            out_offset = off_h.to(tl.int64) * stride_oh

            o_desc = tl.make_tensor_descriptor(
                Out,
                shape=[q_seq_end.to(tl.int32), DimV * H],
                strides=[DimV * H, 1],
                block_shape=[BLOCK_Q, DimV],
            )

            # Wait for both halves of correction to land in o_tile.
            for cid_K in tl.static_range(0, NUM_MMA_GROUPS_K):
                tlx.named_barrier_wait(O_FULLS_BAR_BASE + cid_K, O_FULLS_BAR_COUNT)
            tlx.fence_async_shared()
            # IS_FIRST_K=True: TMA store. False: TMA bulk reduce-add via the
            # beta TLX `store_reduce="add"` parameter (cp.reduce.async.bulk
            # directly from SMEM, no register round-trip). Atomicity isn't
            # required - host serializes launches per (qi, ki).
            tlx.async_descriptor_store(
                o_desc,
                o_tile[0],
                [qo_offset_y.to(tl.int32), out_offset.to(tl.int32)],
                store_reduce="" if IS_FIRST_K else "add",
            )
            tlx.async_descriptor_store_wait(0)


# ---------------------------------------------------------------------------
# Python wrapper for the small-Q kernel
# ---------------------------------------------------------------------------


def tlx_block_attention_fwd_small_q(
    alpha: float,
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    q_seq_offsets: torch.Tensor,
    kv_seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    mask_type: MaskType,
    out_buf: torch.Tensor,
    is_first_k: bool = True,
    max_attn_len: int = 0,
) -> None:
    """Single-launch small-Q forward. Writes into out_buf (in-place when
    is_first_k=True; accumulates into out_buf via atomic_add otherwise).

    Uses total-Q based grid sizing + on-device CTA decoding (mirrors the
    standard kernel) - no host-GPU sync needed.
    """
    assert HAS_TLX, "TLX is not available"

    total_q = q_tensor.shape[0]
    if total_q == 0:
        return

    B = q_seq_offsets.numel() - 1
    H = q_tensor.shape[1]
    DimQ = q_tensor.shape[2]
    DimV = v_tensor.shape[2]
    assert DimQ == DimV, "small-Q kernel requires DimQ == DimV"

    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    dummy_block = [1, 1]

    def make_desc(t, shape, strides):
        return TensorDescriptor(
            t, shape=shape, strides=strides, block_shape=dummy_block
        )

    desc_q = make_desc(q_tensor, [q_tensor.shape[0], H * DimQ], [q_tensor.stride(0), 1])
    desc_k = make_desc(k_tensor, [k_tensor.shape[0], H * DimQ], [k_tensor.stride(0), 1])
    desc_v = make_desc(v_tensor, [v_tensor.shape[0], H * DimV], [v_tensor.stride(0), 1])

    # Varlen-aware grid: same pattern as the standard kernel.
    grid = lambda meta, _tq=total_q, _b=B: (  # noqa E731
        ((_tq + _b * (meta["BLOCK_Q"] - 1)) // meta["BLOCK_Q"]) * H,
    )
    autotune_total_q = triton.next_power_of_2(total_q)
    next_pow2_batch = triton.next_power_of_2(B)

    _block_attn_fwd_small_q[grid](
        Out=out_buf,
        desc_q_0=desc_q,
        desc_k_0=desc_k,
        desc_v_0=desc_v,
        alpha=alpha,
        Z=B,
        H=H,
        q_seq_offsets_tensor=q_seq_offsets,
        kv_seq_offsets_tensor=kv_seq_offsets,
        attn_scale_ptr=attn_scale,
        stride_q_so_b=q_seq_offsets.stride(0),
        stride_kv_so_b=kv_seq_offsets.stride(0),
        stride_qh=q_tensor.stride(1),
        stride_kh=k_tensor.stride(1),
        stride_vh=v_tensor.stride(1),
        stride_oh=out_buf.stride(1),
        max_attn_len=max_attn_len,
        CUR_MASK=mask_type.value,
        IS_FIRST_K=is_first_k,
        HAS_MAX_ATTN_LEN=max_attn_len > 0,
        AUTOTUNE_TOTAL_Q=autotune_total_q,
        NEXT_POW2_BATCH=next_pow2_batch,
        DimQ=DimQ,
        DimV=DimV,
    )
