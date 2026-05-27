# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
TLX (Triton Language Extension) warp-specialized block attention kernel
for Blackwell (SM100+) - NON-PERSISTENT variant.

Each CTA processes exactly one (t_q, pid_m, off_hz) tile.
No persistent tile loop or inter-tile scheduling.
Same warp specialization as persistent variant: correction, silu, mma, load, epilog.

Forward pass only (no backward).
"""

from typing import List, Optional

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
from hammer.v2.ops.triton.template.triton_attention_utils import _get_bufidx_phase
from hammer.v3.ops.pytorch.pt_attention import MaskType
from hammer.v3.ops.triton.tlx_block_attention_small_q import (
    tlx_block_attention_fwd_small_q,
)
from hammer.v3.ops.triton.triton_inline_asm_utils import (
    _fast_silu_pre_halved,
    _mul_f32x2,
)

# Thresholds gating dispatch to the swapped-MMA small-Q kernel for a given
# Otherwise we fall back to the standard kernel.
SMALL_Q_THRESHOLD: int = 32

# @manual=//triton:triton
from triton.tools.tensor_descriptor import TensorDescriptor


MASK_CAUSAL = MaskType.CAUSAL.value
MASK_ALL = MaskType.ALL.value
MASK_DIAGONAL = MaskType.DIAGONAL.value
MASK_NULL = MaskType.NULL.value
MASK_LOCAL = MaskType.LOCAL.value


# ---------------------------------------------------------------------------
# Autotuning configs
# ---------------------------------------------------------------------------


def _host_descriptor_pre_hook(nargs) -> None:
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    DimQ = nargs["DimQ"]
    DimV = nargs["DimV"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    # Patch block shapes for all Q/K/V/O descriptors
    for key in list(nargs.keys()):
        if not isinstance(nargs[key], TensorDescriptor):
            continue
        if key.startswith("desc_q") or key.startswith("desc_o"):
            nargs[key].block_shape = [
                BLOCK_M_SPLIT,
                DimQ if key.startswith("desc_q") else DimV,
            ]
        elif key.startswith("desc_k"):
            nargs[key].block_shape = [BLOCK_N, DimQ]
        elif key.startswith("desc_v"):
            nargs[key].block_shape = [BLOCK_N, DimV]


def _get_fwd_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "NUM_BUFFERS_Q": 1,
                "NUM_BUFFERS_KV": 3,
                "NUM_MMA_GROUPS": 2,
                "NUM_MMA_SLICES": NUM_MMA_SLICES,
                "NUM_REGS_ACT": 196,
                "NUM_REGS_MMA": 24,
                "NUM_REGS_LOAD": 24,
                "NUM_REGS_EPI": 48,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        )
        for BLOCK_M in [128, 256]
        for BLOCK_N in [32, 64, 128, 256]
        for NUM_MMA_SLICES in [1, 2]
    ]
    return configs


# ---------------------------------------------------------------------------
# Helper: split a [M, N] tensor into NUM_MMA_SLICES slices along N
# ---------------------------------------------------------------------------


@triton.jit
def _split_n(x, SPLIT_FACTOR: tl.constexpr):
    if SPLIT_FACTOR == 1:
        return (x,)
    else:
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        return _split_n(x0, SPLIT_FACTOR // 2) + _split_n(x1, SPLIT_FACTOR // 2)


# ---------------------------------------------------------------------------
# Helper: compute block attention masking bounds for a (t_q, t_kv) pair
# ---------------------------------------------------------------------------


@triton.jit
def _block_attn_kv_bounds(
    cur_mask,
    start_m_local,
    q_t_seq_len,
    kv_t_seq_len,
    kv_block_len,
    max_attn_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    """Compute (kv_loop_start, left_mask_end, unmasked_end, kv_loop_end).

    The kv loop is split into up to four contiguous regions:
        [kv_loop_start, left_mask_end)  - left-mask only (LOCAL window-left edge)
        [left_mask_end, unmasked_end)   - fully unmasked middle (no mask op)
        [unmasked_end,  kv_loop_end)    - right-mask (CAUSAL boundary; for LOCAL
                                          this region applies BOTH masks since
                                          left/right may overlap on narrow windows)

    For non-LOCAL masks, left_mask_end == kv_loop_start so the leading region
    is empty and the existing 2-region split is preserved unchanged.

    Mirrors CuteDSL's `BlockInfo.get_n_block_min_before_local_mask` boundary
    (which separates the leading window-left mask region from the unmasked
    middle for LOCAL).
    """
    kv_loop_start = 0
    kv_loop_end = kv_block_len

    # Note: max_valid_k_pos is "one past the last valid n" (the last valid m
    # in the tile is start_m_local + BLOCK_M - 1, so the last valid n is
    # max_valid_k_pos - 1). Tight ceiling is therefore
    # `((max_valid_k_pos + BLOCK_N - 1) // BLOCK_N) * BLOCK_N`. Same logic
    # applies to the DIAGONAL upper bound on (start_m_local + BLOCK_M).
    if cur_mask == MASK_CAUSAL:
        delta = kv_t_seq_len - q_t_seq_len
        max_valid_k_pos = start_m_local + BLOCK_M + delta
        kv_loop_end = tl.minimum(
            kv_block_len,
            ((max_valid_k_pos + BLOCK_N - 1) // BLOCK_N) * BLOCK_N,
        )
    elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
        delta = kv_t_seq_len - q_t_seq_len
        min_valid_k_pos = start_m_local + delta - max_attn_len
        min_valid_k_pos = tl.maximum(min_valid_k_pos, 0)
        kv_loop_start = (min_valid_k_pos // BLOCK_N) * BLOCK_N
        max_valid_k_pos = start_m_local + BLOCK_M + delta
        kv_loop_end = tl.minimum(
            kv_block_len,
            ((max_valid_k_pos + BLOCK_N - 1) // BLOCK_N) * BLOCK_N,
        )
    elif cur_mask == MASK_DIAGONAL:
        kv_loop_start = tl.maximum(0, (start_m_local // BLOCK_N) * BLOCK_N)
        kv_loop_end = tl.minimum(
            kv_block_len,
            ((start_m_local + BLOCK_M + BLOCK_N - 1) // BLOCK_N) * BLOCK_N,
        )

    # Compute unmasked_end (where the trailing right-mask region starts).
    # For LOCAL, use the same CAUSAL formula since LOCAL's right edge is
    # causal-shaped - this differs from the prior behavior which forced
    # unmasked_end = kv_loop_start for LOCAL (sending every block through
    # the masked region).
    if cur_mask == MASK_CAUSAL or (HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL):
        delta = kv_t_seq_len - q_t_seq_len
        unmasked_end = (start_m_local + delta - BLOCK_N + 1).to(tl.int32)
        unmasked_end = (unmasked_end // BLOCK_N) * BLOCK_N
        # tl.clamp is float-only; nested min/max keeps int32 semantics.
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
    # Defined only for LOCAL; for non-LOCAL masks, equals kv_loop_start so the
    # left-mask region is empty.
    #
    # The largest L_row is at the largest m in the tile. A block at start_n
    # needs no left-mask iff start_n >= start_m_local + BLOCK_M + delta -
    # max_attn_len. left_mask_end is the smallest BLOCK_N-aligned start_n
    # satisfying this, clamped to [kv_loop_start, unmasked_end] (the upper
    # clamp keeps the unmasked region contiguous when the window is narrow
    # enough that left-mask and right-mask blocks overlap).
    if HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
        delta = kv_t_seq_len - q_t_seq_len
        threshold_left = start_m_local + BLOCK_M + delta - max_attn_len
        left_mask_end = ((threshold_left + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        # tl.clamp is float-only; nested min/max keeps int32 semantics.
        left_mask_end = tl.minimum(
            tl.maximum(left_mask_end, kv_loop_start.to(tl.int32)), unmasked_end
        )
    else:
        left_mask_end = kv_loop_start.to(tl.int32)

    # Ensure unmasked_end only covers full blocks
    kv_seq_aligned = (kv_block_len.to(tl.int32) // BLOCK_N) * BLOCK_N
    unmasked_end = tl.minimum(unmasked_end, kv_seq_aligned)

    return kv_loop_start, left_mask_end, unmasked_end, kv_loop_end


# ---------------------------------------------------------------------------
# Helper: R2P-style per-element causal mask zeroing.
#
# Replaces tl.where(bool_tile, ...) for the CAUSAL path. The constexpr-
# unrolled bit-test pattern lets ptxas lower to PTX r2p (Register-to-
# Predicate) + predicated selp, avoiding materializing a full bool tile.
# Adapted from third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py
# (originally Tri Dao's flash-attention trick).
# ---------------------------------------------------------------------------


@triton.jit
def _mask_scalar_zero(qk, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s
    col_lim_right_cur = max(col_lim_right_s, 0)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return tl.where(mask_i_bit, qk, 0.0)


@triton.jit
def _apply_causal_mask_zero(qk, col_limit_right, BLOCK_N: tl.constexpr):
    # Per-row col_limit_right shape: [BLOCK_M_SPLIT, 1] (or scalar - broadcasts).
    # Bit chunks of 16 match what r2p.b32 unpacks per call.
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    s = offs_n & ~0xF
    i = offs_n & 0xF
    return tl.map_elementwise(_mask_scalar_zero, qk, col_limit_right, s, i)


# Left-side mirror of _mask_scalar_zero: zero columns where (i + s) < col_limit_left.
@triton.jit
def _mask_scalar_zero_left(qk, col_limit_left, s, i):
    col_lim_left_s = col_limit_left - s
    col_lim_left_cur = max(col_lim_left_s, 0)
    # Bits 0..col_lim_left_cur-1 set -> zero region.
    mask_bits = (1 << col_lim_left_cur) - 1
    mask_i_bit = (mask_bits & (1 << i)) == 0
    return tl.where(mask_i_bit, qk, 0.0)


@triton.jit
def _apply_left_mask_zero(qk, col_limit_left, BLOCK_N: tl.constexpr):
    # Per-row col_limit_left shape: [BLOCK_M_SPLIT, 1] (or scalar - broadcasts).
    # Same r2p-friendly bit-test pattern as the right-side helper.
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    s = offs_n & ~0xF
    i = offs_n & 0xF
    return tl.map_elementwise(_mask_scalar_zero_left, qk, col_limit_left, s, i)


# ---------------------------------------------------------------------------
# Per-slice mask dispatcher - selects R2P limits per CUR_MASK and applies them.
# Each mask type collapses to a per-row right limit R (zero cols >= R) and an
# optional per-row left limit L (zero cols < L). All paths use the same
# r2p-friendly helpers above.
# ---------------------------------------------------------------------------


@triton.jit
def _apply_slice_mask(
    qk_slice,
    cur_mask,
    offs_m_local,
    slice_start_n,
    q_t_seq_len,
    kv_t_seq_len,
    kv_block_len,
    max_attn_len,
    BLOCK_M_SPLIT: tl.constexpr,
    BLOCK_N_SLICE: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    if cur_mask == MASK_CAUSAL:
        delta = kv_t_seq_len - q_t_seq_len
        slice_R = (offs_m_local + delta - slice_start_n + 1)[:, None]
        return _apply_causal_mask_zero(qk_slice, slice_R, BLOCK_N_SLICE)
    elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
        delta = kv_t_seq_len - q_t_seq_len
        q_pos = offs_m_local + delta
        kv_lim_slice = kv_block_len.to(tl.int32) - slice_start_n
        slice_R = tl.minimum(q_pos + 1 - slice_start_n, kv_lim_slice)[:, None]
        slice_L = (q_pos + 1 - max_attn_len - slice_start_n)[:, None]
        qk_slice = _apply_causal_mask_zero(qk_slice, slice_R, BLOCK_N_SLICE)
        return _apply_left_mask_zero(qk_slice, slice_L, BLOCK_N_SLICE)
    elif cur_mask == MASK_DIAGONAL:
        # TODO(perf): DIAGONAL keeps only one column per row, so two-sided R2P
        # zeros ~99% of the tile. Still cheaper than materializing a bool tile,
        # but a dedicated gather/scatter codegen path could be cheaper still -
        # measure before optimizing.
        kv_lim_slice = kv_block_len.to(tl.int32) - slice_start_n
        slice_R = tl.minimum(offs_m_local + 1 - slice_start_n, kv_lim_slice)[:, None]
        slice_L = (offs_m_local - slice_start_n)[:, None]
        qk_slice = _apply_causal_mask_zero(qk_slice, slice_R, BLOCK_N_SLICE)
        return _apply_left_mask_zero(qk_slice, slice_L, BLOCK_N_SLICE)
    elif cur_mask == MASK_ALL:
        # Only invoked for the partial last KV block; full blocks go through
        # the unmasked path. Mask cols >= remaining kv length (broadcast).
        kv_lim_slice = kv_block_len.to(tl.int32) - slice_start_n
        slice_R = (tl.zeros([BLOCK_M_SPLIT], tl.int32) + kv_lim_slice)[:, None]
        return _apply_causal_mask_zero(qk_slice, slice_R, BLOCK_N_SLICE)
    else:
        # MASK_NULL (or unhandled): zero everything.
        return _apply_causal_mask_zero(qk_slice, 0, BLOCK_N_SLICE)


# ---------------------------------------------------------------------------
# Non-persistent block attention forward kernel (warp-specialized)
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_fwd_configs(),
    key=["AUTOTUNE_TOTAL_Q", "DimQ", "CUR_MASK"],
)
@triton.jit
def _block_attn_fwd_np_ws(  # noqa: C901
    # Output base pointer (for device-side TMA store)
    Out,
    # TMA descriptors (single Q/KV pair per launch)
    desc_q_0,
    desc_k_0,
    desc_v_0,
    # Scalar
    alpha,
    Z,  # batch size
    H,  # num heads
    # Offset tensors (1-D, shape [B+1])
    q_seq_offsets_tensor,
    kv_seq_offsets_tensor,
    # Per-row attention scale (1-D, shape [L_q]); folded into alpha per Q row.
    attn_scale_ptr,
    # Strides
    stride_q_so_b,
    stride_kv_so_b,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_oh,
    # Runtime scalars
    max_attn_len,
    # Constexprs
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
    # decodes (batch, m_tile) on-device from q_seq_offsets.
    NEXT_POW2_BATCH: tl.constexpr,
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    NUM_REGS_ACT: tl.constexpr,
    NUM_REGS_MMA: tl.constexpr,
    NUM_REGS_LOAD: tl.constexpr,
    NUM_REGS_EPI: tl.constexpr,
):
    tl.static_assert(NUM_MMA_GROUPS == 2)
    tl.static_assert(DimV == DimQ)

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    # Named barrier IDs for the correction -> epilog handoff. IDs >= 9 are
    # safe per TLX tutorials; we pick 14, 15 to leave headroom for any
    # framework-internal barriers using lower indices.
    O_FULLS_BAR_BASE: tl.constexpr = 14  # pyre-ignore[9]
    # Participant count: correction (default partition, num_warps=4 = 128
    # threads) + epilog (num_warps=1 = 32 threads) = 160.
    O_FULLS_BAR_COUNT: tl.constexpr = 160  # pyre-ignore[9]

    # Named barrier IDs for the silu -> mma p-tile handoff (per (cid, slice)).
    # 4 slots = IDs 9..12 with NUM_MMA_GROUPS=2 and NUM_MMA_SLICES=2.
    # Participants: one silu replica (128 threads) + mma (32 threads) = 160.
    P_FULLS_BAR_BASE: tl.constexpr = 9  # pyre-ignore[9]
    P_FULLS_BAR_COUNT: tl.constexpr = 160  # pyre-ignore[9]

    # Non-persistent: one tile per CTA, one (Q, KV) pair per launch.
    # Grid is `(total_q + B*(BLOCK_M-1))//BLOCK_M * H` (varlen-aware), so each
    # CTA decodes its (batch, m_tile_within_batch) by scanning q_seq_offsets
    # on-device - no host-side max_q_len sync needed.
    tile_idx = tl.program_id(0)
    tile_per_h = tile_idx // H
    off_h = tile_idx % H

    # Compute n_tiles per batch in parallel (load both ends of each batch
    # range, derive seqlen, ceil-div by BLOCK_M).
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
    n_tiles = (seqlens + BLOCK_M - 1) // BLOCK_M
    n_tiles = tl.where(batch_mask, n_tiles, 0)

    # cum_tiles[b] = total m-tiles for batches [0..b]. Find batch_idx as the
    # smallest b such that cum_tiles[b] > tile_per_h, equivalently the count
    # of b where cum_tiles[b] <= tile_per_h.
    cum_tiles = tl.cumsum(n_tiles, axis=0)
    le_mask = cum_tiles <= tile_per_h
    off_z = tl.sum(le_mask.to(tl.int32))
    # local pid_m = tile_per_h - sum(n_tiles for b in [0..off_z)).
    prev_cum = tl.sum(tl.where(le_mask, n_tiles, 0))
    pid_m = tile_per_h - prev_cum

    # Early exit: this CTA's tile_idx is past total useful work (varlen grid
    # over-estimates by at most B*(BLOCK_M-1)/BLOCK_M tiles). off_z hits Z
    # exactly when no remaining batch has tiles for this slot.
    if off_z >= Z:
        return

    off_z_i64 = off_z.to(tl.int64)
    start_m_local = pid_m * BLOCK_M

    # Q bounds: hoisted from per-partition loads. Q sequence offsets are
    # CTA-uniform, so loading once and sharing across all warps avoids 5x
    # redundant L1 traffic.
    q_seq_start = tl.load(q_seq_offsets_tensor + off_z_i64 * stride_q_so_b).to(tl.int64)
    q_seq_end = tl.load(q_seq_offsets_tensor + (off_z_i64 + 1) * stride_q_so_b).to(
        tl.int64
    )
    q_block_len = q_seq_end - q_seq_start

    # KV bounds: hoisted loads only. The cast (kv_t_seq_len /
    # kv_block_len) and _block_attn_kv_bounds stay inside each partition.
    kv_seq_start = tl.load(kv_seq_offsets_tensor + off_z_i64 * stride_kv_so_b).to(
        tl.int32
    )
    kv_seq_end = tl.load(kv_seq_offsets_tensor + (off_z_i64 + 1) * stride_kv_so_b).to(
        tl.int32
    )

    # Early exit: empty KV sequence -> no attention contribution. The host
    # wrapper pre-zeros the output buffer (or accumulates into it), so leaving
    # the rows untouched is the correct semantics.
    if kv_seq_end <= kv_seq_start:
        return

    # Allocate SMEM buffers
    q_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, DimQ), tl.bfloat16, NUM_MMA_GROUPS * NUM_BUFFERS_Q
    )
    kv_tiles = tlx.local_alloc((BLOCK_N, DimV), tl.bfloat16, NUM_BUFFERS_KV)
    o_tiles = tlx.local_alloc((BLOCK_M_SPLIT, DimV), tl.bfloat16, NUM_MMA_GROUPS)

    # SMEM barriers. q_empties / o_empties are omitted: they only matter for
    # producer-consumer reuse cycles in a persistent kernel; here each CTA
    # processes one tile and exits.
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    # o_fulls is implemented as a TLX named barrier (correction -> epilog
    # handoff): cheaper than an SMEM mbarrier since it lowers to PTX bar.sync.
    # Uses barrier IDs O_FULLS_BAR_BASE + cid for cid in [0, NUM_MMA_GROUPS).
    # Participants per cid: correction (4 warps = 128 threads) +
    # epilog (1 warp = 32 threads) = 160 threads.

    # TMEM buffers
    qk_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem
    )
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N // NUM_MMA_SLICES),
        tl.bfloat16,
        NUM_MMA_GROUPS * NUM_MMA_SLICES * 2,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    acc_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, DimV), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem
    )

    # TMEM barriers. p_fulls is implemented as TLX named barriers (one per
    # (cid, slice_id) slot). PTX bar.sync is stateless, so the same barrier
    # ID is reused across kv-block iterations - no phasing or ping-pong.
    # Participants per slot: silu replica `cid` (4 warps = 128 threads) +
    # mma (1 warp = 32 threads) = 160 threads.
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    with tlx.async_tasks():
        # =================================================================
        # Default partition: silu + correction for cid=0. Reclaims the default
        # partition's 4 warps (no longer idle). Body is duplicated with the
        # cid=1 partition below.
        # =================================================================
        with tlx.async_task("default"):
            cid: tl.constexpr = 0  # pyre-ignore[9]
            q_t_seq_len = (q_seq_end - q_seq_start).to(tl.int32)
            offs_m_local = start_m_local + (
                cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
            )
            attn_scale_offs = q_seq_start + offs_m_local.to(tl.int64)
            row_scale = tl.load(
                attn_scale_ptr + attn_scale_offs,
                mask=offs_m_local < q_block_len,
                other=1.0,
            )
            combined_scale_half = (alpha * 0.5) * row_scale
            kv_t_seq_len = kv_seq_end - kv_seq_start
            kv_block_len = kv_t_seq_len.to(tl.int64)
            accum_cnt_qk = 0
            kv_loop_start, left_mask_end, unmasked_end, kv_loop_end = (
                _block_attn_kv_bounds(
                    CUR_MASK,
                    start_m_local,
                    q_t_seq_len,
                    kv_t_seq_len,
                    kv_block_len,
                    max_attn_len,
                    BLOCK_M,
                    BLOCK_N,
                    HAS_MAX_ATTN_LEN,
                )
            )
            BLOCK_N_SLICE: tl.constexpr = BLOCK_N // NUM_MMA_SLICES
            # Left-mask region (LOCAL only - non-empty when window-left bites
            # the leading blocks). Apply left R2P only; right-mask isn't needed
            # since these blocks are below unmasked_end (causal boundary).
            if HAS_MAX_ATTN_LEN and CUR_MASK == MASK_LOCAL:
                delta = kv_t_seq_len - q_t_seq_len
                q_pos = offs_m_local + delta
                for start_n in range(kv_loop_start, left_mask_end, BLOCK_N):
                    _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
                    tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
                    qk = tlx.local_load(tlx.local_view(qk_tiles, cid))
                    qks = _split_n(qk, NUM_MMA_SLICES)
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        slice_start_n = start_n + slice_id * BLOCK_N_SLICE
                        slice_L = (q_pos + 1 - max_attn_len - slice_start_n)[:, None]
                        qk_slice = _apply_left_mask_zero(
                            qks[slice_id], slice_L, BLOCK_N_SLICE
                        )
                        qk_slice = _mul_f32x2(qk_slice, combined_scale_half[:, None])
                        p_bufIdx = (
                            cid * NUM_MMA_GROUPS * NUM_MMA_SLICES
                            + NUM_MMA_SLICES
                            + slice_id
                        )
                        p_i = _fast_silu_pre_halved(qk_slice)
                        tlx.local_store(
                            tlx.local_view(p_tiles, p_bufIdx),
                            p_i.to(tl.bfloat16),
                        )
                        tlx.named_barrier_arrive(
                            P_FULLS_BAR_BASE + cid * NUM_MMA_SLICES + slice_id,
                            P_FULLS_BAR_COUNT,
                        )
                    accum_cnt_qk += 1
            # Unmasked region
            for _ in range(left_mask_end, unmasked_end, BLOCK_N):
                _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
                tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
                qk = tlx.local_load(tlx.local_view(qk_tiles, cid))
                qk = _mul_f32x2(qk, combined_scale_half[:, None])
                qks = _split_n(qk, NUM_MMA_SLICES)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    p_bufIdx = (
                        cid * NUM_MMA_GROUPS * NUM_MMA_SLICES
                        + NUM_MMA_SLICES
                        + slice_id
                    )
                    p_i = _fast_silu_pre_halved(qks[slice_id])
                    tlx.local_store(
                        tlx.local_view(p_tiles, p_bufIdx),
                        p_i.to(tl.bfloat16),
                    )
                    tlx.named_barrier_arrive(
                        P_FULLS_BAR_BASE + cid * NUM_MMA_SLICES + slice_id,
                        P_FULLS_BAR_COUNT,
                    )
                accum_cnt_qk += 1
            # Masked region (right-mask, plus left-mask for LOCAL when the
            # window narrows enough that left/right blocks overlap).
            for start_n in range(unmasked_end, kv_loop_end, BLOCK_N):
                _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
                tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
                qk = tlx.local_load(tlx.local_view(qk_tiles, cid))
                qks = _split_n(qk, NUM_MMA_SLICES)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    slice_start_n = start_n + slice_id * BLOCK_N_SLICE
                    qk_slice = _apply_slice_mask(
                        qks[slice_id],
                        CUR_MASK,
                        offs_m_local,
                        slice_start_n,
                        q_t_seq_len,
                        kv_t_seq_len,
                        kv_block_len,
                        max_attn_len,
                        BLOCK_M_SPLIT,
                        BLOCK_N_SLICE,
                        HAS_MAX_ATTN_LEN,
                    )
                    qk_slice = _mul_f32x2(qk_slice, combined_scale_half[:, None])
                    p_bufIdx = (
                        cid * NUM_MMA_GROUPS * NUM_MMA_SLICES
                        + NUM_MMA_SLICES
                        + slice_id
                    )
                    p_i = _fast_silu_pre_halved(qk_slice)
                    tlx.local_store(
                        tlx.local_view(p_tiles, p_bufIdx),
                        p_i.to(tl.bfloat16),
                    )
                    tlx.named_barrier_arrive(
                        P_FULLS_BAR_BASE + cid * NUM_MMA_SLICES + slice_id,
                        P_FULLS_BAR_COUNT,
                    )
                accum_cnt_qk += 1
            # Correction for cid=0
            tlx.barrier_wait(acc_empties[cid], 0)
            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                subslice = tlx.subslice(
                    acc_tiles[cid],
                    DimQ * slice_id // NUM_MMA_SLICES,
                    DimQ // NUM_MMA_SLICES,
                )
                acc = tlx.local_load(subslice)
                acc = acc.to(tl.bfloat16)
                subslice_o = tlx.local_slice(
                    o_tiles[cid],
                    [0, DimQ * slice_id // NUM_MMA_SLICES],
                    [BLOCK_M_SPLIT, DimQ // NUM_MMA_SLICES],
                )
                tlx.local_store(subslice_o, acc)
            tlx.named_barrier_arrive(O_FULLS_BAR_BASE + cid, O_FULLS_BAR_COUNT)

        # =================================================================
        # SiLU + correction for cid=1 (non-replicated, num_warps=4).
        # =================================================================
        with tlx.async_task(num_warps=4, registers=NUM_REGS_ACT):
            cid: tl.constexpr = 1  # pyre-ignore[9]
            q_t_seq_len = (q_seq_end - q_seq_start).to(tl.int32)
            offs_m_local = start_m_local + (
                cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
            )

            # Per-row attention scale, folded once into alpha. Reused across
            # every kv-block iteration. Out-of-range rows load 1.0 since
            # their output gets dropped by the TMA bounds check anyway.
            attn_scale_offs = q_seq_start + offs_m_local.to(tl.int64)
            row_scale = tl.load(
                attn_scale_ptr + attn_scale_offs,
                mask=offs_m_local < q_block_len,
                other=1.0,
            )
            # Pre-fold 0.5 into the scale so SiLU can skip its leading x*0.5
            # mul (eliminates ~half of the non-fused FP32 ops per kv-block).
            combined_scale_half = (alpha * 0.5) * row_scale  # [BLOCK_M_SPLIT] f32

            # KV-derived bounds (cast kept inside per design).
            kv_t_seq_len = kv_seq_end - kv_seq_start
            kv_block_len = kv_t_seq_len.to(tl.int64)

            accum_cnt_qk = 0

            kv_loop_start, left_mask_end, unmasked_end, kv_loop_end = (
                _block_attn_kv_bounds(
                    CUR_MASK,
                    start_m_local,
                    q_t_seq_len,
                    kv_t_seq_len,
                    kv_block_len,
                    max_attn_len,
                    BLOCK_M,
                    BLOCK_N,
                    HAS_MAX_ATTN_LEN,
                )
            )
            BLOCK_N_SLICE: tl.constexpr = BLOCK_N // NUM_MMA_SLICES
            # Left-mask region (LOCAL only; left R2P only).
            if HAS_MAX_ATTN_LEN and CUR_MASK == MASK_LOCAL:
                delta = kv_t_seq_len - q_t_seq_len
                q_pos = offs_m_local + delta
                for start_n in range(kv_loop_start, left_mask_end, BLOCK_N):
                    _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
                    tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
                    qk = tlx.local_load(tlx.local_view(qk_tiles, cid))
                    qks = _split_n(qk, NUM_MMA_SLICES)
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        slice_start_n = start_n + slice_id * BLOCK_N_SLICE
                        slice_L = (q_pos + 1 - max_attn_len - slice_start_n)[:, None]
                        qk_slice = _apply_left_mask_zero(
                            qks[slice_id], slice_L, BLOCK_N_SLICE
                        )
                        qk_slice = _mul_f32x2(qk_slice, combined_scale_half[:, None])
                        p_bufIdx = (
                            cid * NUM_MMA_GROUPS * NUM_MMA_SLICES
                            + NUM_MMA_SLICES
                            + slice_id
                        )
                        p_i = _fast_silu_pre_halved(qk_slice)
                        tlx.local_store(
                            tlx.local_view(p_tiles, p_bufIdx),
                            p_i.to(tl.bfloat16),
                        )
                        tlx.named_barrier_arrive(
                            P_FULLS_BAR_BASE + cid * NUM_MMA_SLICES + slice_id,
                            P_FULLS_BAR_COUNT,
                        )
                    accum_cnt_qk += 1

            # Unmasked region
            for _ in range(left_mask_end, unmasked_end, BLOCK_N):
                _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
                tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
                qk = tlx.local_load(tlx.local_view(qk_tiles, cid))
                qk = _mul_f32x2(qk, combined_scale_half[:, None])

                qks = _split_n(qk, NUM_MMA_SLICES)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    p_bufIdx = (
                        cid * NUM_MMA_GROUPS * NUM_MMA_SLICES
                        + NUM_MMA_SLICES
                        + slice_id
                    )
                    p_i = _fast_silu_pre_halved(qks[slice_id])
                    tlx.local_store(
                        tlx.local_view(p_tiles, p_bufIdx),
                        p_i.to(tl.bfloat16),
                    )
                    tlx.named_barrier_arrive(
                        P_FULLS_BAR_BASE + cid * NUM_MMA_SLICES + slice_id,
                        P_FULLS_BAR_COUNT,
                    )
                accum_cnt_qk += 1

            # Masked region (right-mask, plus left-mask for LOCAL when the
            # window narrows enough that left/right blocks overlap).
            for start_n in range(unmasked_end, kv_loop_end, BLOCK_N):
                _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
                tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
                qk = tlx.local_load(tlx.local_view(qk_tiles, cid))

                # Unified per-slice R2P masking: split first, then apply
                # mask per slice at BLOCK_N_SLICE = BLOCK_N // NUM_MMA_SLICES.
                qks = _split_n(qk, NUM_MMA_SLICES)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    slice_start_n = start_n + slice_id * BLOCK_N_SLICE
                    qk_slice = _apply_slice_mask(
                        qks[slice_id],
                        CUR_MASK,
                        offs_m_local,
                        slice_start_n,
                        q_t_seq_len,
                        kv_t_seq_len,
                        kv_block_len,
                        max_attn_len,
                        BLOCK_M_SPLIT,
                        BLOCK_N_SLICE,
                        HAS_MAX_ATTN_LEN,
                    )
                    qk_slice = _mul_f32x2(qk_slice, combined_scale_half[:, None])
                    p_bufIdx = (
                        cid * NUM_MMA_GROUPS * NUM_MMA_SLICES
                        + NUM_MMA_SLICES
                        + slice_id
                    )
                    p_i = _fast_silu_pre_halved(qk_slice)
                    tlx.local_store(
                        tlx.local_view(p_tiles, p_bufIdx),
                        p_i.to(tl.bfloat16),
                    )
                    tlx.named_barrier_arrive(
                        P_FULLS_BAR_BASE + cid * NUM_MMA_SLICES + slice_id,
                        P_FULLS_BAR_COUNT,
                    )
                accum_cnt_qk += 1

            # =============================================================
            # Correction work folded in: read acc[cid] from TMEM, store to
            # o_tiles[cid] for the epilog. Each silu replica handles its
            # own cid - no duplication of the silu loop above.
            # =============================================================
            tlx.barrier_wait(acc_empties[cid], 0)
            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                subslice = tlx.subslice(
                    acc_tiles[cid],
                    DimQ * slice_id // NUM_MMA_SLICES,
                    DimQ // NUM_MMA_SLICES,
                )
                acc = tlx.local_load(subslice)
                acc = acc.to(tl.bfloat16)
                subslice_o = tlx.local_slice(
                    o_tiles[cid],
                    [0, DimQ * slice_id // NUM_MMA_SLICES],
                    [BLOCK_M_SPLIT, DimQ // NUM_MMA_SLICES],
                )
                tlx.local_store(subslice_o, acc)
            tlx.named_barrier_arrive(O_FULLS_BAR_BASE + cid, O_FULLS_BAR_COUNT)

        # =================================================================
        # MMA group: Q@K^T -> qk_tiles, P@V -> acc_tiles
        # =================================================================
        with tlx.async_task(num_warps=1, registers=NUM_REGS_MMA):
            # Q bounds + early exit hoisted to CTA scope.
            q_t_seq_len = (q_seq_end - q_seq_start).to(tl.int32)
            kv_t_seq_len = kv_seq_end - kv_seq_start
            kv_block_len = kv_t_seq_len.to(tl.int64)

            accum_cnt_kv = 0
            accum_cnt_qk = 0

            kv_loop_start, _, _, kv_loop_end = _block_attn_kv_bounds(
                CUR_MASK,
                start_m_local,
                q_t_seq_len,
                kv_t_seq_len,
                kv_block_len,
                max_attn_len,
                BLOCK_M,
                BLOCK_N,
                HAS_MAX_ATTN_LEN,
            )

            # kv_loop_end > kv_loop_start is guaranteed by the
            # CTA-scope kv_seq_end > kv_seq_start early exit, so the
            # peel + pipelined loop + tail run unconditionally below.

            # ============ Peeled iter 0: Q@K + P[0]@V only ============
            k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
            v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
            _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

            # Q @ K^T
            tlx.barrier_wait(q_fulls[0], 0)
            tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
            k_tile = tlx.local_trans(kv_tiles[k_bufIdx])

            tlx.async_dot(
                q_tiles[0],
                k_tile,
                qk_tiles[0],
                use_acc=False,
                mBarriers=[qk_fulls[0]],
            )

            tlx.barrier_wait(q_fulls[NUM_BUFFERS_Q], 0)
            tlx.async_dot(
                q_tiles[1],
                k_tile,
                qk_tiles[1],
                use_acc=False,
                mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
            )

            # P[0] @ V (first kv-block, so use_acc=False for slice 0)
            tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                tlx.named_barrier_wait(
                    P_FULLS_BAR_BASE + 0 * NUM_MMA_SLICES + slice_id,
                    P_FULLS_BAR_COUNT,
                )
                kv_slice = tlx.local_slice(
                    kv_tiles[v_bufIdx],
                    [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                    [BLOCK_N // NUM_MMA_SLICES, DimQ],
                )
                p_bufIdx = NUM_MMA_SLICES + slice_id
                tlx.async_dot(
                    p_tiles[p_bufIdx],
                    kv_slice,
                    acc_tiles[0],
                    use_acc=slice_id > 0,
                    force_async=True,
                )

            # Defer P[1]@V to next iter (or tail if no more iters).
            v_bufIdx_prev = v_bufIdx
            acc1_init = False
            accum_cnt_qk += 1
            accum_cnt_kv += 2

            # ============ Main loop: pipelined iter 1..N-1 ============
            _loop_start = (kv_loop_start + BLOCK_N).to(tl.int32)
            _loop_end = kv_loop_end.to(tl.int32)
            for _ in range(_loop_start, _loop_end, BLOCK_N):
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

                # Q[0] @ K (current iter)
                tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                tlx.async_dot(
                    q_tiles[0],
                    k_tile,
                    qk_tiles[0],
                    use_acc=False,
                    mBarriers=[qk_fulls[0]],
                )

                # P[1] @ V from PREVIOUS iter (now safe to issue)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    tlx.named_barrier_wait(
                        P_FULLS_BAR_BASE + 1 * NUM_MMA_SLICES + slice_id,
                        P_FULLS_BAR_COUNT,
                    )
                    kv_slice = tlx.local_slice(
                        kv_tiles[v_bufIdx_prev],
                        [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                        [BLOCK_N // NUM_MMA_SLICES, DimQ],
                    )
                    p_bufIdx_1 = (
                        1 * NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id
                    )
                    use_acc_1 = acc1_init if slice_id == 0 else True
                    mBarriers = (
                        [kv_empties[v_bufIdx_prev]]
                        if slice_id == NUM_MMA_SLICES - 1
                        else []
                    )
                    tlx.async_dot(
                        p_tiles[p_bufIdx_1],
                        kv_slice,
                        acc_tiles[1],
                        use_acc=use_acc_1,
                        mBarriers=mBarriers,
                    )

                acc1_init = True

                # Q[1] @ K (current iter)
                tlx.async_dot(
                    q_tiles[1],
                    k_tile,
                    qk_tiles[1],
                    use_acc=False,
                    mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                )

                # P[0] @ V (current iter, accumulating)
                tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    tlx.named_barrier_wait(
                        P_FULLS_BAR_BASE + 0 * NUM_MMA_SLICES + slice_id,
                        P_FULLS_BAR_COUNT,
                    )
                    kv_slice = tlx.local_slice(
                        kv_tiles[v_bufIdx],
                        [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                        [BLOCK_N // NUM_MMA_SLICES, DimQ],
                    )
                    p_bufIdx = NUM_MMA_SLICES + slice_id
                    tlx.async_dot(
                        p_tiles[p_bufIdx],
                        kv_slice,
                        acc_tiles[0],
                        use_acc=True,
                        force_async=True,
                    )

                v_bufIdx_prev = v_bufIdx
                accum_cnt_qk += 1
                accum_cnt_kv += 2

            # Commit P[0]@V once after the main loop (TMEM safe).
            tlx.tcgen05_commit(acc_empties[0])

            # ============ Tail: final P[1] @ V ============
            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                tlx.named_barrier_wait(
                    P_FULLS_BAR_BASE + 1 * NUM_MMA_SLICES + slice_id,
                    P_FULLS_BAR_COUNT,
                )
                kv_slice = tlx.local_slice(
                    kv_tiles[v_bufIdx_prev],
                    [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                    [BLOCK_N // NUM_MMA_SLICES, DimQ],
                )
                p_bufIdx_1 = (
                    1 * NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id
                )
                use_acc_1 = acc1_init if slice_id == 0 else True
                mBarriers = (
                    [kv_empties[v_bufIdx_prev]]
                    if slice_id == NUM_MMA_SLICES - 1
                    else []
                )
                tlx.async_dot(
                    p_tiles[p_bufIdx_1],
                    kv_slice,
                    acc_tiles[1],
                    use_acc=use_acc_1,
                    mBarriers=mBarriers,
                )

            tlx.tcgen05_commit(acc_empties[1])

        # =================================================================
        # Load group: TMA loads of Q, K, V tiles
        # =================================================================
        with tlx.async_task(num_warps=1, registers=NUM_REGS_LOAD):
            # Q bounds + early exit hoisted to CTA scope.
            q_t_seq_len = (q_seq_end - q_seq_start).to(tl.int32)

            # Load Q tiles (q_empties waits dropped: non-persistent, no reuse).
            qo_offset_y = q_seq_start + start_m_local

            tlx.barrier_expect_bytes(q_fulls[0], 2 * BLOCK_M_SPLIT * DimQ)
            tlx.async_descriptor_load(
                desc_q_0,
                q_tiles[0],
                [qo_offset_y.to(tl.int32), off_h * stride_qh],
                q_fulls[0],
            )

            tlx.barrier_expect_bytes(q_fulls[NUM_BUFFERS_Q], 2 * BLOCK_M_SPLIT * DimQ)
            tlx.async_descriptor_load(
                desc_q_0,
                q_tiles[NUM_BUFFERS_Q],
                [
                    (qo_offset_y + BLOCK_M_SPLIT).to(tl.int32),
                    off_h * stride_qh,
                ],
                q_fulls[NUM_BUFFERS_Q],
            )

            # Load K/V tiles
            kv_t_seq_len = kv_seq_end - kv_seq_start
            kv_block_len = kv_t_seq_len.to(tl.int64)

            kv_loop_start, _, _, kv_loop_end = _block_attn_kv_bounds(
                CUR_MASK,
                start_m_local,
                q_t_seq_len,
                kv_t_seq_len,
                kv_block_len,
                max_attn_len,
                BLOCK_M,
                BLOCK_N,
                HAS_MAX_ATTN_LEN,
            )

            accum_cnt_kv = 0
            kv_offset_y = kv_seq_start + kv_loop_start
            for _ in range(kv_loop_start, kv_loop_end, BLOCK_N):
                # Load K
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                k_empty = tlx.local_view(kv_empties, k_bufIdx)
                tlx.barrier_wait(k_empty, k_phase ^ 1)
                k_full = tlx.local_view(kv_fulls, k_bufIdx)
                k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * DimQ)
                tlx.async_descriptor_load(
                    desc_k_0,
                    k_tile,
                    [kv_offset_y.to(tl.int32), off_h * stride_kh],
                    k_full,
                )

                # Load V
                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                v_empty = tlx.local_view(kv_empties, v_bufIdx)
                tlx.barrier_wait(v_empty, v_phase ^ 1)
                v_full = tlx.local_view(kv_fulls, v_bufIdx)
                v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * DimQ)
                tlx.async_descriptor_load(
                    desc_v_0,
                    v_tile,
                    [kv_offset_y.to(tl.int32), off_h * stride_vh],
                    v_full,
                )

                kv_offset_y += BLOCK_N
                accum_cnt_kv += 2

        # =================================================================
        # Epilog group: TMA stores of output tiles
        # =================================================================
        with tlx.async_task(num_warps=1, registers=NUM_REGS_EPI):
            # Q bounds + early exit hoisted to CTA scope.
            qo_offset_y = q_seq_start + start_m_local
            out_offset = off_h.to(tl.int64) * stride_oh

            o_desc = tl.make_tensor_descriptor(
                Out,
                shape=[q_seq_end.to(tl.int32), DimV * H],
                strides=[DimV * H, 1],
                block_shape=[BLOCK_M_SPLIT, DimV],
            )

            for cid in tl.static_range(0, NUM_MMA_GROUPS):
                tlx.named_barrier_wait(O_FULLS_BAR_BASE + cid, O_FULLS_BAR_COUNT)
                qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                tlx.fence_async_shared()
                # IS_FIRST_K=True: TMA store. False: TMA bulk reduce-add via
                # the beta TLX `store_reduce="add"` parameter, which lowers to
                # cp.reduce.async.bulk.global.shared::cta.add directly from
                # SMEM (no register round-trip). Atomicity isn't required -
                # host serializes launches per (qi, ki); this is just the
                # bulk-DMA read-modify-write primitive.
                tlx.async_descriptor_store(
                    o_desc,
                    o_tiles[cid],
                    [
                        qo_offset_y_split.to(tl.int32),
                        out_offset.to(tl.int32),
                    ],
                    store_reduce="" if IS_FIRST_K else "add",
                )
                tlx.async_descriptor_store_wait(0)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def tlx_block_attention_fwd(
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
    """Forward pass for blocked MHA using TLX (non-persistent).

    Launches a separate kernel per (t_q, t_kv) pair with CUR_MASK as a
    compile-time constant. Accumulates outputs across KV blocks host-side
    (valid because SiLU activation is purely additive).
    """
    if kv_seq_offsets_list is None:
        kv_seq_offsets_list = q_seq_offsets_list

    assert HAS_TLX, "TLX is not available"

    num_q_tensors = len(q_list)
    num_kv_tensors = len(k_list)
    assert num_q_tensors <= 2
    assert num_kv_tensors <= 2

    device = q_list[0].device
    B = q_seq_offsets_list[0].numel() - 1
    H = q_list[0].shape[1]
    DimQ = q_list[0].shape[2]
    DimV = v_list[0].shape[2]

    q_list = [switch_to_contiguous_if_needed(q) for q in q_list]
    k_list = [switch_to_contiguous_if_needed(k) for k in k_list]
    v_list = [switch_to_contiguous_if_needed(v) for v in v_list]

    out_list = [
        torch.zeros((q.shape[0], q.shape[1], DimV), dtype=q.dtype, device=device)
        for q in q_list
    ]

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

    # Track whether each Q output has been written to. First write goes via
    # TMA store (IS_FIRST_K=True); subsequent KV blocks accumulate in-kernel
    # via atomic_add (IS_FIRST_K=False). Lets us allocate `out_list` once.
    first_k_written = [False] * num_q_tensors

    # Pre-build TMA descriptors once per tensor - they don't depend on (qi, ki)
    # iteration state. Saves a per-launch Python alloc + small CPU work.
    desc_q_list = [
        make_desc(q, [q.shape[0], H * DimQ], [q.stride(0), 1]) for q in q_list
    ]
    desc_k_list = [
        make_desc(k, [k.shape[0], H * DimQ], [k.stride(0), 1]) for k in k_list
    ]
    desc_v_list = [
        make_desc(v, [v.shape[0], H * DimV], [v.stride(0), 1]) for v in v_list
    ]
    out_buf_list = out_list  # alias - we write into the same buffer either way

    # Per-Q precompute: total_q comes from the tensor shape (no GPU sync).
    # No `.item()` syncs anywhere in the per-launch path.
    next_pow2_batch = triton.next_power_of_2(B)

    # Iterate over (t_q, t_kv) pairs - one kernel launch per non-NULL pair
    for qi in range(num_q_tensors):
        q_tensor = q_list[qi]
        q_seq_offsets = q_seq_offsets_list[qi]
        attn_scale = attn_scale_list[qi]
        total_q = q_tensor.shape[0]

        if total_q == 0:
            continue

        desc_q = desc_q_list[qi]

        # Varlen-aware grid: launches `(total_q + B*(BLOCK_M-1))/BLOCK_M * H`
        # CTAs (mirrors CuteDSL's SingleTileVarlenScheduler). Each CTA decodes
        # its (batch, m_tile) on-device from q_seq_offsets - no max_q_len sync
        # needed for the kernel itself.
        grid = lambda meta, _tq=total_q, _b=B: (  # noqa E731
            ((_tq + _b * (meta["BLOCK_M"] - 1)) // meta["BLOCK_M"]) * H,
        )
        autotune_total_q = triton.next_power_of_2(total_q)

        for ki in range(num_kv_tensors):
            mask_type = mask_matrix[qi][ki]
            if mask_type == MaskType.NULL:
                continue

            k_tensor = k_list[ki]
            v_tensor = v_list[ki]
            kv_seq_offsets = kv_seq_offsets_list[ki]

            is_first_k = not first_k_written[qi]
            o_buf = out_buf_list[qi]

            desc_k = desc_k_list[ki]
            desc_v = desc_v_list[ki]

            # Dispatch to small-Q kernel for short queries (q_tgt cases)
            # Skip DIAGONAL - its kv_loop is structurally short
            # (1 BLOCK_K_SPLIT iter) regardless of max_kv_len.
            #
            # Sync-free dispatch: `total_q < B * SMALL_Q_THRESHOLD` is a safe
            # upper bound on max_q_len (since max <= total). Conservative -
            # might keep the standard kernel for some cases where small-Q
            # would also be valid, but never the other way (no misdispatch).
            if total_q < B * SMALL_Q_THRESHOLD and mask_type != MaskType.DIAGONAL:
                tlx_block_attention_fwd_small_q(
                    alpha=alpha,
                    q_tensor=q_tensor,
                    k_tensor=k_tensor,
                    v_tensor=v_tensor,
                    q_seq_offsets=q_seq_offsets,
                    kv_seq_offsets=kv_seq_offsets,
                    attn_scale=attn_scale,
                    mask_type=mask_type,
                    out_buf=o_buf,
                    is_first_k=is_first_k,
                    max_attn_len=max_attn_len,
                )
            else:
                _block_attn_fwd_np_ws[grid](
                    Out=o_buf,
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
                    stride_oh=o_buf.stride(1),
                    max_attn_len=max_attn_len,
                    CUR_MASK=mask_type.value,
                    IS_FIRST_K=is_first_k,
                    HAS_MAX_ATTN_LEN=max_attn_len > 0,
                    AUTOTUNE_TOTAL_Q=autotune_total_q,
                    NEXT_POW2_BATCH=next_pow2_batch,
                    DimQ=DimQ,
                    DimV=DimV,
                )

            first_k_written[qi] = True

    return out_list


@torch.fx.wrap
def tlx_mha(
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
    return tlx_block_attention_fwd(
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
