# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

# pyre-unsafe


import enum
import math
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

# pyre-ignore[21]
import cuda.bindings.driver as cuda
import cutlass  # pyre-ignore
import cutlass.cute as cute  # pyre-ignore
import cutlass.cute.nvgpu.tcgen05 as tcgen05  # pyre-ignore
import cutlass.pipeline as pipeline  # pyre-ignore
import cutlass.utils.blackwell_helpers as sm100_utils_basic  # pyre-ignore
import cutlass.utils.blockscaled_layout as blockscaled_utils  # pyre-ignore
import cutlass.utils.hopper_helpers as sm90_utils  # pyre-ignore
import hammer.v3.ops.cutedsl.fa4_helpers.utils as utils
import torch
from cutlass import const_expr, Float32, Int32
from cutlass._mlir.dialects import llvm  # pyre-ignore
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.arch import ProxyKind, SharedSpace  # pyre-ignore
from cutlass.cute.nvgpu import cpasync, warpgroup  # pyre-ignore
from cutlass.cute.runtime import from_dlpack  # pyre-ignore
from cutlass.cutlass_dsl import dsl_user_op, T  # pyre-ignore
from cutlass.utils import LayoutEnum  # pyre-ignore
from cutlass.utils.blockscaled_layout import BlockScaledBasicChunk
from generative_recommenders.ops.utils import is_sm100_plus, is_sm90
from hammer.v3.ops.cutedsl.fa4_helpers import (
    blackwell_helpers as sm100_utils,
    copy_utils,
    hopper_helpers as sm90_helpers,
    pipeline as fa4_pipeline,
)
from hammer.v3.ops.cutedsl.fa4_helpers.block_info import BlockInfo
from hammer.v3.ops.cutedsl.fa4_helpers.block_sparse_utils import (
    consume_block_sparse_loads,
    get_total_block_count,
    handle_block_sparse_empty_tile_correction_sm100,
    produce_block_sparse_loads,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
)
from hammer.v3.ops.cutedsl.fa4_helpers.block_sparsity import BlockSparseTensors
from hammer.v3.ops.cutedsl.fa4_helpers.mask import AttentionMask
from hammer.v3.ops.cutedsl.fa4_helpers.named_barrier import NamedBarrierFwd
from hammer.v3.ops.cutedsl.fa4_helpers.pack_gqa import PackGQA
from hammer.v3.ops.cutedsl.fa4_helpers.paged_kv import PagedKVManager
from hammer.v3.ops.cutedsl.fa4_helpers.seqlen_info import SeqlenInfoQK
from hammer.v3.ops.cutedsl.fa4_helpers.softmax import (
    apply_score_mod_inner,
    Softmax,
    SoftmaxSm100,
)
from hammer.v3.ops.cutedsl.fa4_helpers.tile_scheduler import (
    ParamsBase,
    PersistentVarlenLookupScheduler,
    PersistentVarlenScheduler,
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    StaticPersistentTileScheduler,
    TileSchedulerArguments,
)
from hammer.v3.ops.pytorch.pt_attention import MaskType


# Custom SF layout functions with dtype-aware mma_tile_inst_k computation.
def _compute_mma_tile_inst_k(k_dim: int, dtype_width: int) -> int:
    """Compute k-instructions per MMA K-tile based on element width."""
    mma_inst_bits_k = 256  # Blackwell hardware constant
    k_per_inst = mma_inst_bits_k // dtype_width  # elements per k-instruction
    return k_dim // k_per_inst  # number of k-instructions


@dsl_user_op
def _make_smem_layout_sfa_fp4(
    tiled_mma: cute.TiledMma,  # pyre-ignore
    mma_tiler_mnk: Any,
    sf_vec_size: int,
    num_stages: int,
    mma_tile_inst_k: int,
    *,
    loc: Any = None,
    ip: Any = None,
) -> Any:
    sfa_tile_shape = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[2],
    )
    smem_layout = cute.tile_to_shape(
        BlockScaledBasicChunk(sf_vec_size).layout,
        sfa_tile_shape,
        (2, 1),
    )
    sfa_tile_shape = cute.shape_div(sfa_tile_shape, (1, mma_tile_inst_k))
    smem_layout = cute.tiled_divide(smem_layout, sfa_tile_shape)
    tiler_inst = ((128, sf_vec_size),)
    smem_layout = cute.logical_divide(smem_layout, tiler_inst)
    return cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )


@dsl_user_op
def _make_smem_layout_sfb_fp4(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Any,
    sf_vec_size: int,
    num_stages: int,
    mma_tile_inst_k: int,
    *,
    loc: Any = None,
    ip: Any = None,
) -> Any:
    sfb_tile_shape = (
        cute.round_up(mma_tiler_mnk[1], 128),
        mma_tiler_mnk[2],
    )
    smem_layout = cute.tile_to_shape(
        BlockScaledBasicChunk(sf_vec_size).layout,
        sfb_tile_shape,
        (2, 1),
    )
    sfb_tile_shape = cute.shape_div(sfb_tile_shape, (1, mma_tile_inst_k))
    smem_layout = cute.tiled_divide(smem_layout, sfb_tile_shape)
    tiler_inst = ((128, sf_vec_size),)
    smem_layout = cute.logical_divide(smem_layout, tiler_inst)
    return cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )


import cutlass.cute.core as _cute_core  # pyre-ignore

_original_pretty_str = _cute_core.pretty_str


def _patched_pretty_str(arg: Any, **kwargs: Any) -> str:
    try:
        return _original_pretty_str(arg, **kwargs)
    except TypeError:
        return "<dynamic>"


# pyre-ignore[9]
_cute_core.pretty_str = _patched_pretty_str


class NamedBarrierIds(enum.IntEnum):
    MMA_P0 = 2  # Softmax signals done reading S0, MMA waits before writing SF1
    MMA_P1 = 3  # Softmax signals done reading S1, MMA waits before writing SF0


class TmemLayout:
    """TMEM allocation layout

    TMEM Layout (512 columns total):
        0        64       128      192      256      384      512
        |--------|--------|--------|--------|--------|--------|
        |   S0 (0-128)    |   S1 (128-256)  |   O0   |   O1   |
        |   P0   |        |   P1   |        |(256-384|(384-512|
        |  (64)  |        | (192)  |        |        |        |

    """

    def __init__(self, n_block_size: int, head_dim_v_padded: int, q_stage: int = 2):
        self.n_block_size = n_block_size
        self.head_dim_v_padded = head_dim_v_padded
        self.q_stage = q_stage

        # Compute offsets
        self._s_offset = [0, n_block_size]  # S0 at 0, S1 at n_block_size
        self._s_to_p_offset = n_block_size // 4  # P offset within S region
        self._p_offset = [self._s_offset[i] + self._s_to_p_offset for i in range(2)]
        base_o = self._s_offset[-1] + n_block_size  # After S1
        self._o_offset = [base_o + i * head_dim_v_padded for i in range(q_stage)]
        self._total = self._o_offset[-1] + head_dim_v_padded

        # Prologue SF offsets
        sf_size = 8
        sfk_offset = sf_size  # SFK offset relative to SFQ
        if q_stage >= 2:
            self._sf_prologue_offsets = [
                [
                    self._o_offset[1],
                    self._o_offset[1] + sfk_offset,
                ],  # SF0: SFQ0 in O1, SFK0 after
                [
                    self._o_offset[0],
                    self._o_offset[0] + sfk_offset,
                ],  # SF1: SFQ1 in O0, SFK1 after
            ]
        else:
            # q_stage=1: only one O region, use it for both SF stages
            self._sf_prologue_offsets = [
                [
                    self._o_offset[0],
                    self._o_offset[0] + sfk_offset,
                ],
                [
                    self._o_offset[0],
                    self._o_offset[0] + sfk_offset,
                ],
            ]

    @property
    def s_offset(self) -> list:
        return self._s_offset

    @property
    def o_offset(self) -> list:
        return self._o_offset

    @property
    def p_offset(self) -> list:
        return self._p_offset

    @property
    def s_to_p_offset(self) -> int:
        return self._s_to_p_offset

    @property
    def sf_prologue_offsets(self) -> list:
        return self._sf_prologue_offsets

    @property
    def total_columns(self) -> int:
        return self._total

    def validate(self, max_columns: int = 512) -> None:
        assert self._total <= max_columns, (
            f"TMEM overflow: {self._total} > {max_columns}"
        )


@cute.jit
def mask_r2p_zero(
    X: cute.Tensor,  # pyre-ignore
    col_limit: Int32,  # pyre-ignore
    arch: int = 100,
) -> None:
    """Zeros masked positions for SiLU using R2P bitmask."""
    if const_expr(arch == 90):
        # pyre-ignore[6]
        col_limit_transformed = col_limit // 8 * 2 + min(col_limit % 8, 2)
    else:
        col_limit_transformed = col_limit
    ncol = const_expr(cute.size(X.shape))
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        # pyre-ignore[6]
        col_limit_right_s = max(col_limit_transformed - s * 24, 0)
        mask = (1 << col_limit_right_s) - 1
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            in_bound = cutlass.Boolean(mask & (1 << i))
            c = s * 24 + i
            X[c] = X[c] if in_bound else Float32(0.0)


@cute.jit
def mask_r2p_zero_left(
    X: cute.Tensor,
    col_limit_left: Int32,
    arch: int = 100,
) -> None:
    """Zero positions where col < col_limit_left using R2P bitmask."""
    if const_expr(arch == 90):
        # pyre-ignore[6]
        col_limit_transformed = col_limit_left // 8 * 2 + min(col_limit_left % 8, 2)
    else:
        col_limit_transformed = col_limit_left
    ncol = const_expr(cute.size(X.shape))
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        # pyre-ignore[6]
        col_limit_left_s = max(col_limit_transformed - s * 24, 0)
        # Bits 0..col_limit_left_s-1 are invalid (left of window)
        invalid_mask = (1 << min(col_limit_left_s, 24)) - 1
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            is_left_invalid = cutlass.Boolean(invalid_mask & (1 << i))
            c = s * 24 + i
            if is_left_invalid:
                X[c] = Float32(0.0)


@cute.jit
def mask_r2p_zero_combined(
    X: cute.Tensor,  # pyre-ignore
    col_limit_right: Int32,  # pyre-ignore
    col_limit_left: Int32,  # pyre-ignore
    arch: int = 100,
) -> None:
    """Zero positions where col >= col_limit_right OR col < col_limit_left."""
    if const_expr(arch == 90):
        # pyre-ignore[6]
        col_right_t = col_limit_right // 8 * 2 + min(col_limit_right % 8, 2)
        # pyre-ignore[6]
        col_left_t = col_limit_left // 8 * 2 + min(col_limit_left % 8, 2)
    else:
        col_right_t = col_limit_right
        col_left_t = col_limit_left
    ncol = const_expr(cute.size(X.shape))
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        # pyre-ignore[6]
        col_right_s = max(col_right_t - s * 24, 0)
        # pyre-ignore[6]
        col_left_s = max(col_left_t - s * 24, 0)
        right_mask = (1 << col_right_s) - 1
        left_invalid_mask = (1 << min(col_left_s, 24)) - 1
        valid_mask = right_mask & ~left_invalid_mask
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            in_bound = cutlass.Boolean(valid_mask & (1 << i))
            c = s * 24 + i
            X[c] = X[c] if in_bound else Float32(0.0)


class FlashAttentionForwardSm100:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,  # pyre-ignore
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        # pyre-ignore[9]
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        blockscaled: bool = False,
        sf_vec_size: int = 32,
        is_fp4: bool = False,
        broadcast_q: bool = False,
        use_silu: bool = False,
        is_diagonal: bool = False,
    ):
        self.use_silu = use_silu
        self.is_diagonal = is_diagonal
        if use_silu:
            blockscaled = False
            is_split_kv = False
            score_mod = None
            mask_mod = None
        self.use_tma_KV = not paged_kv_non_tma
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        self.is_fp4 = is_fp4
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.head_dim_v_padded = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        # Reduce n_block_size for large head dims to fit in SMEM
        if max(self.head_dim_padded, self.head_dim_v_padded) > 128:
            n_block_size = min(n_block_size, 64)
        self.n_block_size = n_block_size
        # q_stage=2 processes 2 Q sub-tiles per CTA for better Q reuse.
        # q_stage=1 when head_dim_v > 128 to fit O in TMEM
        self.q_stage = 1 if self.head_dim_v_padded > 128 else 2
        assert self.q_stage in [1, 2]

        # 2 Q tile per CTA
        self.cta_tiler = (
            self.q_stage * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.use_correction_warps_for_epi = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        if pack_gqa:
            # pyre-ignore[58]
            assert m_block_size % self.qhead_per_kvhead == 0, (
                "For PackGQA, m_block_size must be divisible by qhead_per_kvhead"
            )
        assert not (self.is_split_kv and self.head_dim_v_padded >= 192), (
            "SplitKV is not supported for hdim >= 192"
        )
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        if cutlass.const_expr(has_aux_tensors):
            # pyre-ignore[8]
            self.vec_size: cutlass.Constexpr = 1
        else:
            # pyre-ignore[8]
            self.vec_size: cutlass.Constexpr = 2
        self.overlap_sO_sQ = (
            self.head_dim_padded == 192 and self.head_dim_v_padded >= 64
        ) or (self.head_dim_v_padded >= 128 and self.is_split_kv)
        if self.overlap_sO_sQ:
            self.is_persistent = False

        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        # TMEM layout abstraction
        self.tmem_layout = TmemLayout(
            n_block_size=self.n_block_size,
            head_dim_v_padded=self.head_dim_v_padded,
            q_stage=self.q_stage,
        )
        self.tmem_layout.validate(SM100_TMEM_CAPACITY_COLUMNS)

        # Aliases for backward compatibility
        self.tmem_s_offset = self.tmem_layout.s_offset
        self.tmem_o_offset = self.tmem_layout.o_offset
        self.tmem_s_to_p_offset = self.tmem_layout.s_to_p_offset
        self.tmem_p_offset = self.tmem_layout.p_offset

        self.tmem_vec_offset = self.tmem_s_offset

        # Named barrier IDs
        self.mbar_mma_p0_id = NamedBarrierIds.MMA_P0
        self.mbar_mma_p1_id = NamedBarrierIds.MMA_P1
        self.mbar_mma_threads = cute.arch.WARP_SIZE * (len(self.softmax0_warp_ids) + 1)

        # Block scaling specific
        self.blockscaled = blockscaled
        self.sf_vec_size = sf_vec_size
        self.sf_dtype = cutlass.Float8E8M0FNU if blockscaled else None
        self.broadcast_q = broadcast_q

        # unroll_kv: overlaps last PV of current tile
        # with first QK of next tile. Hides prologue latency behind PV GEMM.
        # Only for persistent, non-causal, non-local, non-split-kv scheduling.
        self.unroll_kv = (
            is_persistent
            and not is_causal
            and not is_local
            and not is_split_kv
            and blockscaled
        )

        # When unroll_kv is active, decouple epilogue from correction warps to avoid
        # the correction cascade at tile boundaries. Only for TMA-KV path.
        if self.unroll_kv and self.use_tma_KV:
            self.use_correction_warps_for_epi = False

        # Warp reassignment
        if not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()
        elif not self.use_correction_warps_for_epi:
            # TMA + separate epilogue: TMA load needs only 1 warp,
            # give warp 14 to epilogue for 2-warp (64-thread) GMEM stores
            self.epilogue_warp_ids = (13, 14)
            self.load_warp_ids = (15,)
            self.empty_warp_ids = ()
        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids
        elif self.is_varlen_q:  # fallback
            self.epilogue_warp_ids = (13, 14)

        # TmemLayout for blockscaled MXFP8
        self.tmem_layout = TmemLayout(
            self.n_block_size, self.head_dim_v_padded, self.q_stage
        )

        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        else:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        self.num_regs_empty = 24

        self.buffer_align_bytes = 1024

        # Divisor for splitting TMEM store into two phases for P values
        self.tmem_store_split_divisor = None

    def _get_tmem_store_split_divisor(self) -> int:
        return 4 if self.q_dtype.width >= 16 else 2  # pyre-ignore

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation."""

        self.kv_stage = 4 if self.q_dtype.width <= 8 else 3
        self.acc_stage = 1
        # Reduce epi_stage for large head_dim_v to fit O in SMEM.
        self.epi_stage = 1 if self.head_dim_v_padded > 128 else 2
        self.uneven_kv_smem = (
            self.head_dim_padded == 192
            and self.head_dim_v_padded == 128
            and self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.m_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,  # pyre-ignore
        stream: cuda.CUstream,  # pyre-ignore
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        mSFV: Optional[cute.Tensor] = None,
        mCuSeqlensSFQ: Optional[cute.Tensor] = None,
        mCuSeqlensSFK: Optional[cute.Tensor] = None,
        total_sf_q: Int32 | int | None = None,
        total_sf_k: Int32 | int | None = None,
        mTileToBatch: Optional[cute.Tensor] = None,
        mTileToHead: Optional[cute.Tensor] = None,
        mTileToBlock: Optional[cute.Tensor] = None,
        mCuSeqlensO: Optional[cute.Tensor] = None,
        mAttnScale: Optional[cute.Tensor] = None,
    ):
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type  # pyre-ignore
        self.k_dtype = mK.element_type  # pyre-ignore
        self.v_dtype = mV.element_type  # pyre-ignore
        self.o_dtype = mO.element_type  # pyre-ignore

        if const_expr(self.is_fp4):
            _fp4_fix = lambda t: cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (*t.shape[:-1], t.shape[-1] * 2),
                    stride=(*(s * 2 for s in t.stride[:-1]), t.stride[-1]),
                ),
            )
            mQ = _fp4_fix(mQ)
            mK = _fp4_fix(mK)
            mV = _fp4_fix(mV)

        self.tmem_store_split_divisor = self._get_tmem_store_split_divisor()
        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mO = [
            cute.make_tensor(
                t.iterator, cute.make_layout(t.shape, stride=new_stride(t))
            )
            for t in (mQ, mK, mV, mO)
        ]
        Q_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mQ = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)
        )
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        if const_expr(self.is_split_kv):
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = (
                [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            )
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = (
                [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            )
            LSE_layout_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            )
            num_splits = Int32(1)
        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        mLSE = (
            cute.make_tensor(
                mLSE.iterator,  # pyre-ignore[16]
                cute.select(mLSE.layout, mode=LSE_layout_transpose),  # pyre-ignore[16]
            )
            if const_expr(mLSE is not None)
            else None
        )
        if const_expr(self.is_fp4):
            V_layout_transpose = (
                [0, 1, 2, 3] if const_expr(mCuSeqlensK is None) else [0, 1, 2]
            )
        else:
            V_layout_transpose = (
                [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
            )
        mV = cute.make_tensor(
            mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
        )

        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(  # pyre-ignore[16]
            mQ
        ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(  # pyre-ignore[16]
            mK
        ).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(  # pyre-ignore[16]
            mV
        ).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)  # pyre-ignore[16]

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        if const_expr(self.is_fp4):
            if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError(
                    "FP4 requires K-major V layout. The .kind::mxf4 instruction "
                    "only supports K-major for both operands."
                )
        else:
            if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
                raise RuntimeError("The layout of mV is not supported")

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        # pyre-ignore
        self.use_tma_O = (  # pyre-ignore
            self.arch >= 90 and mCuSeqlensQ is None and mSeqUsedQ is None
        )  # pyre-ignore

        self.e2e_freq = 16  # pyre-ignore
        if const_expr(
            self.head_dim_padded > 64
            and not self.is_causal
            and not self.is_local
            and self.pack_gqa
        ):
            self.e2e_freq = (
                32 if mCuSeqlensQ is not None or mSeqUsedQ is not None else 10
            )

        cta_group = tcgen05.CtaGroup.ONE
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        if const_expr(self.blockscaled):
            tiled_mma_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cta_group,
                self.mma_tiler_qk[:2],
            )
        else:
            tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.qk_acc_dtype,
                cta_group,
                self.mma_tiler_qk[:2],
            )
        if const_expr(self.blockscaled):
            pv_dtype = self.q_dtype  # P is recast to q_dtype in softmax_step
            tiled_mma_pv = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                pv_dtype,
                p_major_mode,
                self.v_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cta_group,
                self.mma_tiler_pv[:2],
                p_source,  # A operand comes from TMEM (TS mode for P*V)
            )
        else:
            tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                self.v_major_mode,
                self.pv_acc_dtype,
                cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)  # pyre-ignore
        self.cluster_layout_vmnk = cute.tiled_divide(  # pyre-ignore
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        self.epi_tile = self.mma_tiler_pv[:2]  # pyre-ignore

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.q_stage,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,  # pyre-ignore
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.q_dtype,
            self.acc_stage,  # pyre-ignore
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.kv_stage,
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,  # pyre-ignore
        )
        # SMEM layouts for scale factors
        sSFQ_layout = None
        sSFK_layout = None
        if const_expr(self.blockscaled):
            # Compute mma_tile_inst_k from dtype width.
            qk_inst_k = _compute_mma_tile_inst_k(
                self.head_dim_padded,
                self.q_dtype.width,  # pyre-ignore[16]
            )
            pv_inst_k = _compute_mma_tile_inst_k(self.n_block_size, self.q_dtype.width)

            sSFQ_layout = _make_smem_layout_sfa_fp4(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                self.q_stage,
                qk_inst_k,
            )
            sSFK_layout = _make_smem_layout_sfb_fp4(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                self.kv_stage,
                qk_inst_k,
            )
            # SFV: V is operand B for PV GEMM, use mma_tiler_pv
            sSFV_layout = _make_smem_layout_sfb_fp4(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                self.kv_stage,
                pv_inst_k,
            )
            # SFP: P's scale factors, 2 stages for P0/P1 double-buffering
            sfp_num_stages = 2
            sSFP_layout = _make_smem_layout_sfa_fp4(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                sfp_num_stages,
                pv_inst_k,
            )
        else:
            sSFV_layout = None
            sSFP_layout = None
        if const_expr(not self.same_hdim_kv_padded):
            stride_sK = const_expr(max(sK_layout.outer.stride[-1], 0))
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem  # pyre-ignore
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )

        if const_expr(self.pack_gqa):
            shape_Q_packed = (
                (
                    self.qhead_per_kvhead,
                    # pyre-ignore[16]
                    mQ.shape[0],
                ),
                mQ.shape[1],
                mK.shape[2],
                *mQ.shape[3:],
            )
            stride_Q_packed = (
                (mQ.stride[2], mQ.stride[0]),
                mQ.stride[1],
                mQ.stride[2] * self.qhead_per_kvhead,
                *mQ.stride[3:],
            )
            mQ = cute.make_tensor(
                mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
            )
            shape_O_packed = (
                (self.qhead_per_kvhead, mO.shape[0]),
                mO.shape[1],
                mK.shape[2],
                *mO.shape[3:],
            )
            stride_O_packed = (
                (mO.stride[2], mO.stride[0]),
                mO.stride[1],
                mO.stride[2] * self.qhead_per_kvhead,
                *mO.stride[3:],
            )
            mO = cute.make_tensor(
                mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
            )
            if const_expr(mLSE is not None):
                shape_LSE_packed = (
                    # pyre-ignore[16]
                    (self.qhead_per_kvhead, mLSE.shape[0]),
                    mK.shape[2],
                    *mLSE.shape[2:],
                )
                stride_LSE_packed = (
                    # pyre-ignore[16]
                    (mLSE.stride[1], mLSE.stride[0]),
                    mLSE.stride[1] * self.qhead_per_kvhead,
                    *mLSE.stride[2:],
                )
                mLSE = cute.make_tensor(
                    # pyre-ignore[16]
                    mLSE.iterator,
                    cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed),
                )

        self.tma_copy_bytes = {  # pyre-ignore
            name: cute.size_in_bytes(
                mX.element_type, cute.select(layout, mode=[0, 1, 2])
            )
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        # Add scale factor TMA copy bytes
        if const_expr(self.blockscaled):
            self.tma_copy_sfq_bytes = int(  # pyre-ignore
                cute.size_in_bytes(
                    self.sf_dtype, cute.select(sSFQ_layout, mode=[0, 1, 2])
                )
            )
            self.tma_copy_sfk_bytes = int(  # pyre-ignore
                cute.size_in_bytes(
                    self.sf_dtype, cute.select(sSFK_layout, mode=[0, 1, 2])
                )
            )
            # Add SF bytes to the transaction counts for Q and K loads
            self.tma_copy_bytes["Q"] += self.tma_copy_sfq_bytes
            self.tma_copy_bytes["K"] += self.tma_copy_sfk_bytes
            # SFV bytes for V loads
            if const_expr(mSFV is not None):
                self.tma_copy_sfv_bytes = int(  # pyre-ignore
                    cute.size_in_bytes(
                        self.sf_dtype, cute.select(sSFV_layout, mode=[0, 1, 2])
                    )
                )
                self.tma_copy_bytes["V"] += self.tma_copy_sfv_bytes
            else:
                self.tma_copy_sfv_bytes = 0
        else:
            self.tma_copy_sfq_bytes = 0
            self.tma_copy_sfk_bytes = 0
            self.tma_copy_sfv_bytes = 0

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        if const_expr(self.use_tma_KV):
            # TMA load for K
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
            # TMA load for V
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                self.cluster_layout_vmnk.shape,
            )
        else:
            tma_atom_K = None
            tma_atom_V = None

        # TMA atoms for scale factors (SFQ, SFK)
        tma_atom_SFQ = None
        tma_atom_SFK = None
        tma_tensor_SFQ = None
        tma_tensor_SFK = None
        if const_expr(self.blockscaled):
            # Create SF tensor layouts
            if const_expr(total_sf_q is not None):
                # Replace first dimension with padded total for SF layout
                sfq_shape = (total_sf_q, mQ.shape[1], mQ.shape[2])
            else:
                sfq_shape = mQ.shape
            sfq_layout = blockscaled_utils.tile_atom_to_shape_SF(
                sfq_shape, self.sf_vec_size
            )
            mSFQ = cute.make_tensor(mSFQ.iterator, sfq_layout)
            sSFQ_layout_per_stage = cute.select(sSFQ_layout, mode=[0, 1, 2])
            tma_atom_SFQ, tma_tensor_SFQ = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mSFQ,
                sSFQ_layout_per_stage,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # SFK TMA atom creation
            if const_expr(total_sf_k is not None):
                # Replace first dimension with padded total for SF layout
                sfk_shape = (
                    (total_sf_k, mK.shape[1], mK.shape[2])
                    if len(mK.shape) == 3
                    else (total_sf_k, mK.shape[1], mK.shape[2], mK.shape[3])
                )
            else:
                sfk_shape = mK.shape
            sfk_layout = blockscaled_utils.tile_atom_to_shape_SF(
                sfk_shape, self.sf_vec_size
            )
            mSFK = cute.make_tensor(mSFK.iterator, sfk_layout)
            sSFK_layout_per_stage = cute.select(sSFK_layout, mode=[0, 1, 2])
            tma_atom_SFK, tma_tensor_SFK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mSFK,
                sSFK_layout_per_stage,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # TMA atom for SFV
            if const_expr(mSFV is not None):
                if const_expr(total_sf_k is not None):
                    sfv_shape = (
                        (mV.shape[0], total_sf_k, mV.shape[2])
                        if len(mV.shape) == 3
                        else (mV.shape[0], total_sf_k, mV.shape[2], mV.shape[3])
                    )
                else:
                    sfv_shape = mV.shape
                sfv_layout = blockscaled_utils.tile_atom_to_shape_SF(
                    sfv_shape, self.sf_vec_size
                )
                mSFV_tensor = cute.make_tensor(mSFV.iterator, sfv_layout)
                sSFV_layout_per_stage = cute.select(sSFV_layout, mode=[0, 1, 2])
                tma_atom_SFV, tma_tensor_SFV = cute.nvgpu.make_tiled_tma_atom_B(
                    tma_load_op,
                    mSFV_tensor,
                    sSFV_layout_per_stage,
                    self.mma_tiler_pv,
                    tiled_mma_pv,
                    self.cluster_layout_vmnk.shape,
                    internal_type=cutlass.Int16,
                )
            else:
                tma_atom_SFV, tma_tensor_SFV = None, None
        else:
            tma_atom_SFV, tma_tensor_SFV = None, None

        o_cta_v_layout = cute.composition(
            cute.make_identity_layout(mO.shape), self.epi_tile
        )

        # pyre-ignore
        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(  # pyre-ignore
            self.epilogue_warp_ids
        )  # pyre-ignore
        # pyre-ignore[16]
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO,
                cute.select(sO_layout, mode=[0, 1]),
                o_cta_v_layout,
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(
                atom_universal_copy, tO_layout, vO_layout
            )

        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            if const_expr(not self.is_persistent):
                TileScheduler = SingleTileVarlenScheduler
            elif const_expr(mTileToBatch is not None):
                TileScheduler = PersistentVarlenLookupScheduler
            else:
                TileScheduler = PersistentVarlenScheduler
        else:
            if const_expr(self.is_causal or self.is_local):
                TileScheduler = SingleTileLPTScheduler
            else:
                TileScheduler = (
                    SingleTileScheduler
                    if const_expr(not self.is_persistent)
                    else StaticPersistentTileScheduler
                )
        tile_sched_args = TileSchedulerArguments(  # pyre-ignore
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            (
                # pyre-ignore[16]
                cute.size(mCuSeqlensO.shape[0] - 1)
                if const_expr(self.broadcast_q)
                else (
                    cute.size(mQ.shape[3])
                    if const_expr(mCuSeqlensQ is None)
                    else cute.size(mCuSeqlensQ.shape[0] - 1)
                )
            ),
            num_splits,
            (
                cute.size(mK.shape[0])
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mQ.shape[1],
            mV.shape[0],
            total_q=(
                cute.size(mO.shape[0])
                if const_expr(self.broadcast_q)
                else (
                    cute.size(mQ.shape[0])
                    if const_expr(mCuSeqlensQ is not None)
                    else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3])
                )
            ),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            # pyre-ignore[6]
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
            # pyre-ignore[6]
            element_size=max(self.k_dtype.width // 8, 1),
            # pyre-ignore[6]
            is_persistent=self.is_persistent,
            # pyre-ignore[6]
            lpt=self.is_causal or self.is_local,
            # pyre-ignore[6]
            is_split_kv=self.is_split_kv,
            mTileToBatch=mTileToBatch,
            mTileToHead=mTileToHead,
            mTileToBlock=mTileToBlock,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler  # pyre-ignore
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)  # pyre-ignore

        self.mbar_load_q_full_offset = 0  # pyre-ignore
        # pyre-ignore
        self.mbar_load_q_empty_offset = (  # pyre-ignore
            self.mbar_load_q_full_offset + self.q_stage
        )  # pyre-ignore
        # pyre-ignore
        self.mbar_load_kv_full_offset = (  # pyre-ignore
            self.mbar_load_q_empty_offset + self.q_stage
        )  # pyre-ignore
        # pyre-ignore
        self.mbar_load_kv_empty_offset = (  # pyre-ignore
            self.mbar_load_kv_full_offset + self.kv_stage
        )  # pyre-ignore
        self.mbar_P_full_O_rescaled_offset = (  # pyre-ignore
            self.mbar_load_kv_empty_offset + self.kv_stage
        )
        self.mbar_S_full_offset = self.mbar_P_full_O_rescaled_offset + 2  # pyre-ignore
        self.mbar_O_full_offset = self.mbar_S_full_offset + 2  # pyre-ignore
        self.mbar_softmax_corr_full_offset = self.mbar_O_full_offset + 2  # pyre-ignore
        # pyre-ignore
        self.mbar_softmax_corr_empty_offset = (  # pyre-ignore
            self.mbar_softmax_corr_full_offset + 2
        )  # pyre-ignore
        self.mbar_corr_epi_full_offset = (  # pyre-ignore
            self.mbar_softmax_corr_empty_offset + self.epi_stage
        )
        self.mbar_corr_epi_empty_offset = (  # pyre-ignore
            self.mbar_corr_epi_full_offset + self.epi_stage
        )
        # pyre-ignore
        self.mbar_s0_s1_sequence_offset = (  # pyre-ignore
            self.mbar_corr_epi_empty_offset + 2
        )  # pyre-ignore
        # pyre-ignore
        self.mbar_tmem_dealloc_offset = (  # pyre-ignore
            self.mbar_s0_s1_sequence_offset + 8
        )  # pyre-ignore
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1  # pyre-ignore
        self.mbar_total = self.mbar_P_full_2_offset + 2  # pyre-ignore

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sQ_size = (
            cute.cosize(sQ_layout)
            if const_expr(not self.overlap_sO_sQ)
            else cutlass.max(
                cute.cosize(sQ_layout),
                cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width,
            )
        )

        @cute.struct
        class SharedStorage:
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            tmem_holding_buf: Int32
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sSFQ: cute.struct.Align[
                cute.struct.MemRange[
                    (
                        self.sf_dtype
                        if const_expr(self.blockscaled)
                        else cutlass.Float8E8M0FNU
                    ),
                    cute.cosize(sSFQ_layout) if const_expr(self.blockscaled) else 0,
                ],
                self.buffer_align_bytes,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[
                    (
                        self.sf_dtype
                        if const_expr(self.blockscaled)
                        else cutlass.Float8E8M0FNU
                    ),
                    cute.cosize(sSFK_layout) if const_expr(self.blockscaled) else 0,
                ],
                self.buffer_align_bytes,
            ]
            sSFV: cute.struct.Align[
                cute.struct.MemRange[
                    (
                        self.sf_dtype
                        if const_expr(self.blockscaled)
                        else cutlass.Float8E8M0FNU
                    ),
                    cute.cosize(sSFV_layout) if const_expr(self.blockscaled) else 0,
                ],
                self.buffer_align_bytes,
            ]
            sSFP: cute.struct.Align[
                cute.struct.MemRange[
                    (
                        self.sf_dtype
                        if const_expr(self.blockscaled)
                        else cutlass.Float8E8M0FNU
                    ),
                    cute.cosize(sSFP_layout) if const_expr(self.blockscaled) else 0,
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage  # pyre-ignore

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            if const_expr(not self.use_silu):
                # pyre-ignore[9]
                softmax_scale = None
        else:
            softmax_scale_log2 = LOG2_E
            softmax_scale = softmax_scale

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = (
                cute.size(mK.shape[0])
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            )
            seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisor(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        # pyre-ignore
        self.use_block_sparsity = cutlass.const_expr(  # pyre-ignore
            blocksparse_tensors is not None
        )  # pyre-ignore
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError(
                "Block sparsity + paged KV not supported on SM100"
            )

        # Launch the kernel
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            tma_tensor_SFQ,
            tma_atom_SFQ,
            sSFQ_layout,
            tma_tensor_SFK,
            tma_atom_SFK,
            sSFK_layout,
            tma_tensor_SFV,
            tma_atom_SFV,
            sSFV_layout,
            sSFP_layout,
            mCuSeqlensSFQ,
            mCuSeqlensSFK,
            mCuSeqlensO,
            mAttnScale,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,  # pyre-ignore
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,  # pyre-ignore
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],  # pyre-ignore
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_tensors: Optional[list] = None,
        fastdiv_mods: Tuple[  # pyre-ignore
            Optional[FastDivmodDivisor], Optional[FastDivmodDivisor]
        ] = (None, None),
        mSFQ: Optional[cute.Tensor] = None,
        tma_atom_SFQ: Optional[cute.CopyAtom] = None,
        sSFQ_layout: Optional[cute.Layout] = None,  # pyre-ignore
        mSFK: Optional[cute.Tensor] = None,
        tma_atom_SFK: Optional[cute.CopyAtom] = None,
        sSFK_layout: Optional[cute.Layout] = None,
        mSFV: Optional[cute.Tensor] = None,
        tma_atom_SFV: Optional[cute.CopyAtom] = None,
        sSFV_layout: Optional[cute.Layout] = None,
        sSFP_layout: Optional[cute.Layout] = None,
        mCuSeqlensSFQ: Optional[cute.Tensor] = None,
        mCuSeqlensSFK: Optional[cute.Tensor] = None,
        mCuSeqlensO: Optional[cute.Tensor] = None,
        mAttnScale: Optional[cute.Tensor] = None,
    ) -> None:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            if const_expr(tma_atom_K is not None):
                cpasync.prefetch_descriptor(tma_atom_K)
            if const_expr(tma_atom_V is not None):
                cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)  # pyre-ignore

        mbar_ptr = storage.mbar_ptr.data_ptr()
        if warp_idx == 1:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_full_offset + i,  # pyre-ignore[16]
                    1,
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_empty_offset + i,  # pyre-ignore[16]
                    len([self.mma_warp_id]),
                )
        if warp_idx == 2:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + i,  # pyre-ignore
                    cute.arch.WARP_SIZE * 4,
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + i,  # pyre-ignore
                    cute.arch.WARP_SIZE * 4,
                )
        if const_expr(not self.use_correction_warps_for_epi) and warp_idx == 4:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_full_offset + i,  # pyre-ignore
                    cute.arch.WARP_SIZE * len(self.correction_warp_ids),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_empty_offset + i,  # pyre-ignore
                    cute.arch.WARP_SIZE * len(self.epilogue_warp_ids),
                )
        if warp_idx == 5:
            for i in cutlass.range_constexpr(self.q_stage):
                if const_expr(self.use_silu):
                    cute.arch.mbarrier_init(
                        mbar_ptr
                        # pyre-ignore
                        + self.mbar_P_full_O_rescaled_offset  # pyre-ignore
                        + i,  # pyre-ignore
                        cute.arch.WARP_SIZE * len(self.softmax0_warp_ids),
                    )
                else:
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_P_full_O_rescaled_offset + i,
                        cute.arch.WARP_SIZE
                        * (len(self.softmax0_warp_ids) + len(self.correction_warp_ids)),
                    )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_S_full_offset + i,  # pyre-ignore[16]
                    len([self.mma_warp_id]),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_O_full_offset + i,  # pyre-ignore[16]
                    len([self.mma_warp_id]),
                )
        if warp_idx == 6:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_2_offset + i,  # pyre-ignore
                    cute.arch.WARP_SIZE * len(self.softmax0_warp_ids),
                )
        if warp_idx == 7:
            cute.arch.mbarrier_init(
                mbar_ptr + self.mbar_tmem_dealloc_offset,  # pyre-ignore
                cute.arch.WARP_SIZE
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        pipeline_kv = self.make_and_init_load_kv_pipeline(
            mbar_ptr + self.mbar_load_kv_full_offset  # pyre-ignore
        )

        #  Generate smem tensor Q/K/V/O
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer
        )
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(
                cute.recast_ptr(
                    sQ.iterator,
                    sO_layout.inner,
                    self.o_dtype,  # pyre-ignore[16]
                ),
                sO_layout.outer,
            )

        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.q_stage * self.m_block_size * 2)
        )

        sSFQ = None
        sSFK = None
        sSFV = None
        sSFP = None
        if const_expr(self.blockscaled):
            sSFQ = storage.sSFQ.get_tensor(sSFQ_layout)
            sSFK = storage.sSFK.get_tensor(sSFK_layout)
            sSFV = storage.sSFV.get_tensor(sSFV_layout)
            sSFP = storage.sSFP.get_tensor(sSFP_layout)

        thr_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        thr_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        tmem_ptr = cute.make_ptr(
            Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)

        tStSs = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout)
            for stage in range(self.q_stage)
        )
        tOtOs = tuple(
            cute.make_tensor(tOtO.iterator + self.tmem_o_offset[stage], tOtO.layout)
            for stage in range(self.q_stage)
        )

        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]

        tOrPs = [
            cute.make_tensor(
                tOrP.iterator
                + self.qk_acc_dtype.width
                // self.q_dtype.width  # pyre-ignore
                * self.tmem_p_offset[stage],
                tOrP.layout,
            )
            for stage in range(self.q_stage)
        ]

        # Create blockscaled PV partitions
        tOtOs_blockscaled = None
        tOrPs_blockscaled = None
        if const_expr(self.blockscaled):
            cta_mma_pv_blockscaled = tiled_mma_pv.get_slice(0)
            pv_acc_shape_bs = cta_mma_pv_blockscaled.partition_shape_C(
                self.mma_tiler_pv[:2]
            )
            tOtO_bs = cta_mma_pv_blockscaled.make_fragment_C(pv_acc_shape_bs)
            tOtOs_blockscaled = tuple(
                cute.make_tensor(
                    tOtO_bs.iterator + self.tmem_o_offset[stage], tOtO_bs.layout
                )
                for stage in range(self.q_stage)
            )
            tP_bs = cute.make_tensor(tStS.iterator, tP_layout.outer)
            tOrP_bs = cta_mma_pv_blockscaled.make_fragment_A(tP_bs)[None, None, None, 0]
            tOrPs_blockscaled = [
                cute.make_tensor(
                    tOrP_bs.iterator
                    + self.qk_acc_dtype.width
                    // self.q_dtype.width
                    * self.tmem_p_offset[stage],
                    tOrP_bs.layout,
                )
                for stage in range(self.q_stage)
            ]

        block_info = BlockInfo(  # pyre-ignore
            self.cta_tiler[0],
            self.cta_tiler[1],
            # pyre-ignore[6]
            self.is_causal,
            # pyre-ignore[6]
            self.is_local,
            # pyre-ignore[6]
            self.is_split_kv,
            window_size_left,
            window_size_right,
            # pyre-ignore[6]
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=(
                # pyre-ignore[16]
                mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=(
                mK.shape[0]
                if const_expr(mPageTable is None)
                # pyre-ignore[16]
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuSeqlensSFQ=mCuSeqlensSFQ,
            mCuSeqlensSFK=mCuSeqlensSFK,
            broadcast_q=self.broadcast_q,
            mCuSeqlensO=mCuSeqlensO,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        TileSchedulerCls = partial(
            self.tile_scheduler_cls.create,  # pyre-ignore[16]
            tile_sched_params,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(len(self.empty_warp_ids) > 0):
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        if const_expr(len(self.empty_warp_ids) > 1):
            if warp_idx == self.empty_warp_ids[1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        assert len(self.empty_warp_ids) <= 2

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                # Scale factor parameters for blockscaled MXFP8
                mSFQ=mSFQ,
                mSFK=mSFK,
                sSFQ=sSFQ,
                sSFK=sSFK,
                tma_atom_SFQ=tma_atom_SFQ,
                tma_atom_SFK=tma_atom_SFK,
                # SFV parameters for PV blockscaled GEMM
                mSFV=mSFV,
                sSFV=sSFV,
                tma_atom_SFV=tma_atom_SFV,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # if warp_idx == self.mma_warp_id or warp_idx == self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
                cute.arch.sync_warp()

            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStSs,
                tOtOs,
                tOrPs,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                sSFQ,
                sSFK,
                sSFV,
                sSFP,
                tOtOs_blockscaled,
                tOrPs_blockscaled,
            )

            # if warp_idx == self.mma_warp_id:
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if (
                warp_idx >= self.epilogue_warp_ids[0]
                and warp_idx <= self.epilogue_warp_ids[-1]
            ):
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    mbar_ptr,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax / SiLU
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            if const_expr(self.use_silu):
                silu_loop = partial(
                    self.silu_loop,
                    softmax_scale=softmax_scale,
                    thr_mma_qk=thr_mma_qk,
                    sScale=sScale,
                    mbar_ptr=mbar_ptr,
                    block_info=block_info,
                    num_splits=num_splits,
                    SeqlenInfoCls=SeqlenInfoCls,
                    TileSchedulerCls=TileSchedulerCls,
                    mAttnScale=mAttnScale,
                    window_size_left=window_size_left,
                )

                if const_expr(self.q_stage == 1):
                    if warp_idx < self.softmax1_warp_ids[0]:
                        silu_loop(
                            stage=Int32(0),
                            tStSi=cute.make_tensor(
                                tStS.iterator + self.tmem_s_offset[0],
                                tStS.layout,
                            ),
                        )
                else:
                    stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
                    silu_loop(
                        stage=stage,
                        tStSi=cute.make_tensor(
                            tStS.iterator
                            + (
                                self.tmem_s_offset[0]
                                if stage == 0
                                else self.tmem_s_offset[1]
                            ),
                            tStS.layout,
                        ),
                    )
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            else:
                softmax_loop = partial(
                    self.softmax_loop,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_scale=softmax_scale,
                    thr_mma_qk=thr_mma_qk,
                    sScale=sScale,
                    mLSE=mLSE,
                    learnable_sink=learnable_sink,
                    mbar_ptr=mbar_ptr,
                    block_info=block_info,
                    num_splits=num_splits,
                    SeqlenInfoCls=SeqlenInfoCls,
                    AttentionMaskCls=AttentionMaskCls,
                    TileSchedulerCls=TileSchedulerCls,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods,
                    blocksparse_tensors=blocksparse_tensors,
                    sSFP=sSFP,
                )

                if const_expr(self.q_stage == 1):
                    if warp_idx < self.softmax1_warp_ids[0]:
                        softmax_loop(
                            stage=Int32(0),
                            tStSi=cute.make_tensor(
                                tStS.iterator + self.tmem_s_offset[0],
                                tStS.layout,
                            ),
                        )
                else:
                    stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
                    softmax_loop(
                        stage=stage,
                        tStSi=cute.make_tensor(
                            tStS.iterator
                            + (
                                self.tmem_s_offset[0]
                                if stage == 0
                                else self.tmem_s_offset[1]
                            ),
                            tStS.layout,
                        ),
                    )
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)
            if const_expr(self.use_silu):
                self.correction_loop_silu(
                    thr_mma_pv,
                    tOtOs,
                    sO,
                    mO,
                    gmem_tiled_copy_O,
                    mbar_ptr,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )
            else:
                self.correction_loop(
                    thr_mma_qk,
                    thr_mma_pv,
                    tStS,
                    tOtOs,
                    sScale,
                    mO,
                    mLSE,
                    sO,
                    learnable_sink,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    mbar_ptr,
                    softmax_scale_log2,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                )
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.core.ThrMma,  # pyre-ignore
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_kv: cutlass.pipeline.PipelineAsync,  # pyre-ignore
        mbar_ptr: cute.Pointer,  # pyre-ignore
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        # Scale factor parameters for blockscaled MXFP8
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        sSFQ: Optional[cute.Tensor] = None,
        sSFK: Optional[cute.Tensor] = None,
        tma_atom_SFQ: Optional[cute.CopyAtom] = None,
        tma_atom_SFK: Optional[cute.CopyAtom] = None,
        # SFV parameters for PV blockscaled GEMM
        mSFV: Optional[cute.Tensor] = None,
        sSFV: Optional[cute.Tensor] = None,
        tma_atom_SFV: Optional[cute.CopyAtom] = None,
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer,
            self.kv_stage,  # pyre-ignore[16]
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(
                mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0)
            )

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead
                if const_expr(not self.pack_gqa)
                else head_idx
            )
            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [
                        t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)
                    ]
                else:
                    mK_cur = cute.domain_offset(
                        (seqlen.offset_k, 0), mK[None, None, head_idx_kv]
                    )
                    if const_expr(self.is_fp4):
                        mV_cur = cute.domain_offset(
                            (0, seqlen.offset_sf_k), mV[None, None, head_idx_kv]
                        )
                    else:
                        mV_cur = cute.domain_offset(
                            (0, seqlen.offset_k), mV[None, None, head_idx_kv]
                        )
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)
                )
            else:
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                )
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(  # pyre-ignore
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )

            if const_expr(self.use_tma_KV):
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                paged_kv_manager = None
            else:
                # pyre-ignore[16]
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    # pyre-ignore[6]
                    mPageTable,
                    mK,
                    mV,
                    FastDivmodDivisor(page_size),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    # pyre-ignore[6]
                    0,  # leftpad_k
                    # pyre-ignore[6]
                    self.n_block_size,
                    # pyre-ignore[6]
                    self.head_dim_padded,
                    # pyre-ignore[6]
                    self.head_dim_v_padded,
                    # pyre-ignore[6]
                    num_load_threads,
                    # pyre-ignore[6]
                    mK.element_type,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None

            # TMA partitions for scale factors (SFQ, SFK)
            tSFQsSFQ, tSFQgSFQ = None, None
            tSFKsSFK, tSFKgSFK = None, None
            if const_expr(self.blockscaled):
                offset = (
                    seqlen.offset_sf_q
                    if const_expr(not self.pack_gqa)
                    else (0, seqlen.offset_sf_q)
                )
                # pyre-ignore[16]
                mSFQ_cur = cute.domain_offset((offset, 0), mSFQ[None, None, head_idx])

                gSFQ = cute.local_tile(
                    mSFQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0)
                )
                tSgSFQ = thr_mma_qk.partition_A(gSFQ)
                tSFQsSFQ, tSFQgSFQ = cpasync.tma_partition(
                    tma_atom_SFQ,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFQ, 0, 3),
                    cute.group_modes(tSgSFQ, 0, 3),
                )
                tSFQsSFQ = cute.filter_zeros(tSFQsSFQ)
                tSFQgSFQ = cute.filter_zeros(tSFQgSFQ)

                # SFK TMA partition
                if const_expr(mPageTable is None):
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        mSFK_cur = mSFK[None, None, head_idx_kv, batch_idx]
                    else:
                        mSFK_cur = cute.domain_offset(
                            (seqlen.offset_sf_k, 0), mSFK[None, None, head_idx_kv]
                        )
                else:
                    mSFK_cur = mSFK[None, None, head_idx_kv, None]
                gSFK = cute.local_tile(
                    mSFK_cur,
                    cute.select(self.mma_tiler_qk, mode=[1, 2]),
                    (None, 0) if const_expr(mPageTable is None) else (None, 0, None),
                )
                tSgSFK = thr_mma_qk.partition_B(gSFK)
                tSFKsSFK, tSFKgSFK = cpasync.tma_partition(
                    tma_atom_SFK,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFK, 0, 3),
                    cute.group_modes(tSgSFK, 0, 3),
                )
                tSFKsSFK = cute.filter_zeros(tSFKsSFK)
                tSFKgSFK = cute.filter_zeros(tSFKgSFK)

            # TMA partitions for scale factors
            tSFVsSFV, tSFVgSFV = None, None
            if const_expr(self.blockscaled and tma_atom_SFV is not None):
                # SFV TMA partition
                if const_expr(mPageTable is None):
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        mSFV_cur = mSFV[None, None, head_idx_kv, batch_idx]
                    else:
                        mSFV_cur = cute.domain_offset(
                            (0, seqlen.offset_sf_k), mSFV[None, None, head_idx_kv]
                        )
                else:
                    mSFV_cur = mSFV[None, None, head_idx_kv, None]
                # Use mma_tiler_pv for V's scale factors
                gSFV = cute.local_tile(
                    mSFV_cur,
                    cute.select(self.mma_tiler_pv, mode=[1, 2]),
                    (0, None) if const_expr(mPageTable is None) else (0, None, None),
                )
                cta_mma_pv_bs = thr_mma_pv
                tSgSFV = cta_mma_pv_bs.partition_B(gSFV)
                tSFVsSFV, tSFVgSFV = cpasync.tma_partition(
                    tma_atom_SFV,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFV, 0, 3),
                    cute.group_modes(tSgSFV, 0, 3),
                )
                tSFVsSFV = cute.filter_zeros(tSFVsSFV)
                tSFVgSFV = cute.filter_zeros(tSFVgSFV)

            load_Q = partial(
                self.load_Q,
                load_Q_fn,
                mbar_ptr + self.mbar_load_q_full_offset,  # pyre-ignore
                mbar_ptr + self.mbar_load_q_empty_offset,  # pyre-ignore
                phase=q_producer_phase,
                # Scale factor TMA parameters
                tma_atom_SFQ=tma_atom_SFQ,
                tSFQgSFQ=tSFQgSFQ,
                tSFQsSFQ=tSFQsSFQ,
            )
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                mbar_ptr + self.mbar_load_kv_full_offset,  # pyre-ignore
                mbar_ptr + self.mbar_load_kv_empty_offset,  # pyre-ignore
                K_or_V="K",
                # Scale factor TMA parameters for SFK
                tma_atom_SFK=tma_atom_SFK,
                tSFKgSFK=tSFKgSFK,
                tSFKsSFK=tSFKsSFK,
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                paged_kv_manager,
                sV,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="V",
                # Scale factor TMA parameters for SFV
                tma_atom_SFV=tma_atom_SFV,
                tSFVgSFV=tSFVgSFV,
                tSFVsSFV=tSFVsSFV,
            )

            if const_expr(not self.use_block_sparsity):  # pyre-ignore
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    if const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE:
                        load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                    page_idx = (
                        mPageTable[batch_idx, n_block_first]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(n_block_first)
                    load_K(
                        block=n_block_max - 1,
                        producer_state=kv_producer_state,
                        page_idx=page_idx,
                    )  # K0
                    kv_producer_state.advance()
                    if const_expr(self.q_stage == 2) and (
                        const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE
                    ):
                        load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1
                    q_producer_phase ^= 1
                    load_V(
                        block=n_block_max - 1,
                        producer_state=kv_producer_state,
                        page_idx=page_idx,
                    )  # V0
                    kv_producer_state.advance()
                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        page_idx = (
                            mPageTable[batch_idx, n_block]
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        load_K(
                            block=n_block,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )  # Ki
                        kv_producer_state.advance()
                        load_V(
                            block=n_block,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )  # Vi
                        kv_producer_state.advance()

            else:
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                )

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        sSFQ: Optional[cute.Tensor] = None,
        sSFK: Optional[cute.Tensor] = None,
        sSFV: Optional[cute.Tensor] = None,
        sSFP: Optional[cute.Tensor] = None,
        tOtOs_blockscaled: Optional[tuple] = None,
        tOrPs_blockscaled: Optional[list] = None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 0])

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op

        # Setup blockscaled GEMM infrastructure
        tCtSFQs, tCtSFKs = [None, None], [None, None]
        tiled_copy_s2t_sfq, tCsSFQ_compact_s2t = None, None
        tCtSFQ_compact_s2ts = [None, None]
        tiled_copy_s2t_sfk, tCsSFK_compact_s2t = None, None
        tCtSFK_compact_s2ts = [None, None]
        tCtSFQs_prologue, tCtSFKs_prologue = [None, None], [None, None]
        tCtSFQ_compact_s2ts_prologue = [None, None]
        tCtSFK_compact_s2ts_prologue = [None, None]

        if const_expr(self.blockscaled):
            tmem_ptr = cute.make_ptr(
                Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
            )
            tilePlikeFP32 = (
                # pyre-ignore
                self.mma_tiler_qk[1] // 32 * self.v_dtype.width  # pyre-ignore
            )  # pyre-ignore

            # SF offsets: SF0 in S1 region, SF1 in S0 region
            sf_offsets = [
                self.tmem_p_offset[1] + tilePlikeFP32 * 2,  # ~224 (in S1)
                self.tmem_p_offset[0] + tilePlikeFP32 * 2,  # ~96 (in S0)
            ]

            # Create TMEM layouts
            # pyre-ignore[16]
            sSFQ_layout_per_stage = cute.slice_(sSFQ.layout, (None, None, None, 0))
            tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                sSFQ_layout_per_stage,
            )
            sSFK_layout_per_stage = cute.slice_(sSFK.layout, (None, None, None, 0))
            tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                sSFK_layout_per_stage,
            )

            # Get SFK offset relative to SFQ
            temp_sfq_ptr = cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype)
            temp_tCtSFQ = cute.make_tensor(temp_sfq_ptr, tCtSFQ_layout)
            sfk_relative_offset = tcgen05.find_tmem_tensor_col_offset(temp_tCtSFQ)

            # Create stage-specific SF TMEM tensors (unrolled)
            sfq_tmem_ptr_0 = cute.recast_ptr(
                tmem_ptr + sf_offsets[0], dtype=self.sf_dtype
            )
            tCtSFQs[0] = cute.make_tensor(sfq_tmem_ptr_0, tCtSFQ_layout)
            sfk_tmem_ptr_0 = cute.recast_ptr(
                tmem_ptr + sf_offsets[0] + sfk_relative_offset, dtype=self.sf_dtype
            )
            tCtSFKs[0] = cute.make_tensor(sfk_tmem_ptr_0, tCtSFK_layout)
            sfq_tmem_ptr_1 = cute.recast_ptr(
                tmem_ptr + sf_offsets[1], dtype=self.sf_dtype
            )
            tCtSFQs[1] = cute.make_tensor(sfq_tmem_ptr_1, tCtSFQ_layout)
            sfk_tmem_ptr_1 = cute.recast_ptr(
                tmem_ptr + sf_offsets[1] + sfk_relative_offset, dtype=self.sf_dtype
            )
            tCtSFKs[1] = cute.make_tensor(sfk_tmem_ptr_1, tCtSFK_layout)

            # Create S2T copy partitions
            # pyre-ignore[6]
            tiled_copy_s2t_sfq, tCsSFQ_compact_s2t, tCtSFQ_compact_s2ts[0] = (
                # pyre-ignore[6]
                sm100_utils.make_s2t_copy_partitions(sSFQ, tCtSFQs[0], self.sf_dtype)
            )
            # pyre-ignore[6]
            _, _, tCtSFQ_compact_s2ts[1] = sm100_utils.make_s2t_copy_partitions(
                sSFQ,  # pyre-ignore[6]
                tCtSFQs[1],  # pyre-ignore[6]
                self.sf_dtype,
            )
            # pyre-ignore[6]
            tiled_copy_s2t_sfk, tCsSFK_compact_s2t, tCtSFK_compact_s2ts[0] = (
                # pyre-ignore[6]
                sm100_utils.make_s2t_copy_partitions(sSFK, tCtSFKs[0], self.sf_dtype)
            )
            # pyre-ignore[6]
            _, _, tCtSFK_compact_s2ts[1] = sm100_utils.make_s2t_copy_partitions(
                sSFK,  # pyre-ignore[6]
                tCtSFKs[1],  # pyre-ignore[6]
                self.sf_dtype,
            )

            # Prologue SF tensors (in O region, safe for first tile before O is used) - unrolled
            sf_prologue_offsets = self.tmem_layout.sf_prologue_offsets
            sfq_prologue_ptr_0 = cute.recast_ptr(
                tmem_ptr + sf_prologue_offsets[0][0], dtype=self.sf_dtype
            )
            tCtSFQs_prologue[0] = cute.make_tensor(sfq_prologue_ptr_0, tCtSFQ_layout)
            sfk_prologue_ptr_0 = cute.recast_ptr(
                tmem_ptr + sf_prologue_offsets[0][1], dtype=self.sf_dtype
            )
            tCtSFKs_prologue[0] = cute.make_tensor(sfk_prologue_ptr_0, tCtSFK_layout)
            # pyre-ignore[6]
            _, _, tCtSFQ_compact_s2ts_prologue[0] = (
                sm100_utils.make_s2t_copy_partitions(
                    sSFQ,  # pyre-ignore[6]
                    tCtSFQs_prologue[0],  # pyre-ignore[6]
                    self.sf_dtype,
                )
            )
            # pyre-ignore[6]
            _, _, tCtSFK_compact_s2ts_prologue[0] = (
                sm100_utils.make_s2t_copy_partitions(
                    sSFK,  # pyre-ignore[6]
                    tCtSFKs_prologue[0],  # pyre-ignore[6]
                    self.sf_dtype,
                )
            )
            sfq_prologue_ptr_1 = cute.recast_ptr(
                tmem_ptr + sf_prologue_offsets[1][0], dtype=self.sf_dtype
            )
            tCtSFQs_prologue[1] = cute.make_tensor(sfq_prologue_ptr_1, tCtSFQ_layout)
            sfk_prologue_ptr_1 = cute.recast_ptr(
                tmem_ptr + sf_prologue_offsets[1][1], dtype=self.sf_dtype
            )
            tCtSFKs_prologue[1] = cute.make_tensor(sfk_prologue_ptr_1, tCtSFK_layout)
            # pyre-ignore[6]
            _, _, tCtSFQ_compact_s2ts_prologue[1] = (
                sm100_utils.make_s2t_copy_partitions(
                    sSFQ,  # pyre-ignore[6]
                    tCtSFQs_prologue[1],  # pyre-ignore[6]
                    self.sf_dtype,
                )
            )
            # pyre-ignore[6]
            _, _, tCtSFK_compact_s2ts_prologue[1] = (
                sm100_utils.make_s2t_copy_partitions(
                    sSFK,  # pyre-ignore[6]
                    tCtSFKs_prologue[1],  # pyre-ignore[6]
                    self.sf_dtype,
                )
            )

        # Setup PV blockscaled GEMM infrastructure
        tCtSFPs, tCtSFVs = [None, None], [None, None]
        tiled_copy_s2t_sfp, tCsSFP_compact_s2t = None, None
        tCtSFP_compact_s2ts = [None, None]
        tiled_copy_s2t_sfv, tCsSFV_compact_s2t = None, None
        tCtSFV_compact_s2ts = [None, None]

        if const_expr(self.blockscaled):
            # Create TMEM layouts for SFP (P's scale) and SFV (V's scale)
            sSFP_layout_per_stage = cute.slice_(sSFP.layout, (None, None, None, 0))
            tCtSFP_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                sSFP_layout_per_stage,
            )
            sSFV_layout_per_stage = cute.slice_(sSFV.layout, (None, None, None, 0))
            tCtSFV_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                sSFV_layout_per_stage,
            )

            # Get SFV offset relative to SFP
            temp_sfp_ptr = cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype)  # pyre-ignore
            temp_tCtSFP = cute.make_tensor(temp_sfp_ptr, tCtSFP_layout)
            sfv_relative_offset = tcgen05.find_tmem_tensor_col_offset(temp_tCtSFP)

            # PV SF offsets: place SF after P for each stage
            # SFP0 = P0 + tilePlikeFP32, SFP1 = P1 + tilePlikeFP32
            sfp_offsets = [
                self.tmem_p_offset[0] + tilePlikeFP32,  # pyre-ignore
                self.tmem_p_offset[1] + tilePlikeFP32,  # pyre-ignore
            ]

            # Create stage-specific SF TMEM tensors for PV GEMM (unrolled)
            # Stage 0: SFP0 at P0 + tilePlikeFP32, SFV0 after SFP0
            sfp_tmem_ptr_0 = cute.recast_ptr(
                tmem_ptr + sfp_offsets[0],  # pyre-ignore[61]
                dtype=self.sf_dtype,
            )
            tCtSFPs[0] = cute.make_tensor(sfp_tmem_ptr_0, tCtSFP_layout)
            sfv_tmem_ptr_0 = cute.recast_ptr(
                tmem_ptr + sfp_offsets[0] + sfv_relative_offset,  # pyre-ignore[61]
                dtype=self.sf_dtype,
            )
            tCtSFVs[0] = cute.make_tensor(sfv_tmem_ptr_0, tCtSFV_layout)

            # Stage 1: SFP1 at P1 + tilePlikeFP32, SFV1 after SFP1
            sfp_tmem_ptr_1 = cute.recast_ptr(
                tmem_ptr + sfp_offsets[1],  # pyre-ignore[61]
                dtype=self.sf_dtype,
            )
            tCtSFPs[1] = cute.make_tensor(sfp_tmem_ptr_1, tCtSFP_layout)
            sfv_tmem_ptr_1 = cute.recast_ptr(
                tmem_ptr + sfp_offsets[1] + sfv_relative_offset,  # pyre-ignore[61]
                dtype=self.sf_dtype,
            )
            tCtSFVs[1] = cute.make_tensor(sfv_tmem_ptr_1, tCtSFV_layout)

            # Create S2T copy partitions for SFP and SFV
            # pyre-ignore[6]
            tiled_copy_s2t_sfp, tCsSFP_compact_s2t, tCtSFP_compact_s2ts[0] = (
                # pyre-ignore[6]
                sm100_utils.make_s2t_copy_partitions(sSFP, tCtSFPs[0], self.sf_dtype)
            )
            # pyre-ignore[6]
            _, _, tCtSFP_compact_s2ts[1] = sm100_utils.make_s2t_copy_partitions(
                sSFP,  # pyre-ignore[6]
                tCtSFPs[1],  # pyre-ignore[6]
                self.sf_dtype,
            )

            # pyre-ignore[6]
            tiled_copy_s2t_sfv, tCsSFV_compact_s2t, tCtSFV_compact_s2ts[0] = (
                # pyre-ignore[6]
                sm100_utils.make_s2t_copy_partitions(sSFV, tCtSFVs[0], self.sf_dtype)
            )
            # pyre-ignore[6]
            _, _, tCtSFV_compact_s2ts[1] = sm100_utils.make_s2t_copy_partitions(
                sSFV,  # pyre-ignore[6]
                tCtSFVs[1],  # pyre-ignore[6]
                self.sf_dtype,
            )

        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_partial,
                qk_mma_op,
                self.tmem_s_offset[stage],
                tSrQs[stage],
                sA=sQ[None, None, None, stage],
                zero_init=True,
            )
            for stage in range(self.q_stage)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage if self.q_stage == 2 else 0],
                tOrPs[stage],
                sA=None,
            )
            for stage in range(self.q_stage)
        ]
        # Blockscaled PV GEMM: P (from TMEM) * V -> O with scale factors SFP and SFV
        gemm_Pi_blockscaled = (
            [
                lambda tCrB,
                sB,
                zero_init,
                stage=stage,  # pyre-ignore
                **kwargs: sm100_utils.gemm_blockscaled(
                    tiled_mma_pv,
                    tOtOs_blockscaled[stage],  # pyre-ignore
                    tOrPs_blockscaled[stage],  # pyre-ignore
                    tCrB,
                    tCtSFPs[stage],
                    tCtSFVs[stage],
                    zero_init=zero_init,
                )
                for stage in range(self.q_stage)
            ]
            if const_expr(self.blockscaled)
            else [None, None]
        )

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer,
            self.kv_stage,  # pyre-ignore[16]
        )
        P_full_O_rescaled_phase = Int32(0)
        is_first_tile = True

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):  # pyre-ignore
                block_iter_count = get_total_block_count(
                    blocksparse_tensors, batch_idx, head_idx, m_block
                )
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                block_iter_count = n_block_max - n_block_min
                if const_expr(not self.is_split_kv):
                    process_tile = True
                else:
                    process_tile = n_block_min < n_block_max

            if process_tile:
                if is_first_tile or const_expr(not self.unroll_kv):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                        # 1. wait for Q0 / Q1
                        cute.arch.mbarrier_wait(
                            mbar_ptr
                            # pyre-ignore
                            + self.mbar_load_q_full_offset  # pyre-ignore
                            + stage,  # pyre-ignore
                            mma_q_consumer_phase,
                        )
                        # 2. wait for K0
                        if const_expr(stage == 0):
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                        # 3. gemm
                        sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                        if const_expr(self.uneven_kv_smem):  # pyre-ignore
                            sK_cur = self.offset_kv_smem(
                                sK_cur,
                                mma_kv_consumer_state.index,
                                mma_kv_consumer_state.phase,
                            )
                        if const_expr(self.blockscaled):
                            s2t_sfq_stage_coord = (None, None, None, None, stage)
                            s2t_sfk_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                mma_kv_consumer_state.index,
                            )
                            if is_first_tile:
                                # First tile: use prologue SF offsets (in O region, safe before O is used)
                                cute.copy(
                                    tiled_copy_s2t_sfq,
                                    # pyre-ignore[16]
                                    tCsSFQ_compact_s2t[s2t_sfq_stage_coord],
                                    tCtSFQ_compact_s2ts_prologue[stage],
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfk,
                                    tCsSFK_compact_s2t[s2t_sfk_stage_coord],
                                    tCtSFK_compact_s2ts_prologue[stage],
                                )
                                cute.arch.fence_view_async_tmem_store()
                                sm100_utils.gemm_blockscaled(
                                    tiled_mma_qk,
                                    tStSs[stage],
                                    tSrQs[stage],
                                    tSrKi,
                                    tCtSFQs_prologue[stage],
                                    tCtSFKs_prologue[stage],
                                    zero_init=True,
                                )
                            else:
                                # Non-first tile: use regular SF (overlaps S region)
                                # with P1/P0 barrier waits (same as main loop pattern).
                                if stage == 0:
                                    cute.arch.barrier(
                                        barrier_id=self.mbar_mma_p1_id,
                                        number_of_threads=self.mbar_mma_threads,
                                    )
                                else:
                                    cute.arch.barrier(
                                        barrier_id=self.mbar_mma_p0_id,
                                        number_of_threads=self.mbar_mma_threads,
                                    )
                                cute.copy(
                                    tiled_copy_s2t_sfq,
                                    tCsSFQ_compact_s2t[s2t_sfq_stage_coord],
                                    tCtSFQ_compact_s2ts[stage],
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfk,
                                    tCsSFK_compact_s2t[s2t_sfk_stage_coord],
                                    tCtSFK_compact_s2ts[stage],
                                )
                                cute.arch.fence_view_async_tmem_store()
                                sm100_utils.gemm_blockscaled(
                                    tiled_mma_qk,
                                    tStSs[stage],
                                    tSrQs[stage],
                                    tSrKi,
                                    tCtSFQs[stage],
                                    tCtSFKs[stage],
                                    zero_init=True,
                                )
                        else:
                            gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                        # 4. release S0 / S1
                        with cute.arch.elect_one():
                            tcgen05.commit(
                                # pyre-ignore
                                mbar_ptr
                                + self.mbar_S_full_offset  # pyre-ignore
                                + stage  # pyre-ignore
                            )  # pyre-ignore
                    mma_q_consumer_phase ^= 1
                    # 5. release K0
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    # End of GEMM (Q1 * K0 -> S1)
                    is_first_tile = False

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                for _ in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        cute.arch.mbarrier_wait(
                            mbar_ptr
                            # pyre-ignore
                            + self.mbar_P_full_O_rescaled_offset  # pyre-ignore
                            + stage,  # pyre-ignore
                            P_full_O_rescaled_phase,
                        )
                        # 3. gemm
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        if const_expr(self.blockscaled):
                            # Copy SFP and SFV from SMEM to TMEM before blockscaled PV GEMM
                            s2t_sfp_stage_coord = (None, None, None, None, stage)
                            cute.copy(
                                tiled_copy_s2t_sfp,
                                tCsSFP_compact_s2t[s2t_sfp_stage_coord],
                                tCtSFP_compact_s2ts[stage],
                            )
                            s2t_sfv_stage_coord = (None, None, None, None, Vi_index)
                            cute.copy(
                                tiled_copy_s2t_sfv,
                                tCsSFV_compact_s2t[s2t_sfv_stage_coord],
                                tCtSFV_compact_s2ts[stage],
                            )
                            cute.arch.fence_view_async_tmem_store()
                            cute.arch.mbarrier_wait(
                                mbar_ptr
                                # pyre-ignore
                                + self.mbar_P_full_2_offset  # pyre-ignore
                                + stage,  # pyre-ignore
                                P_full_O_rescaled_phase,
                            )
                            gemm_Pi_blockscaled[stage](  # pyre-ignore
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=not O_should_accumulate,
                            )
                        else:
                            gemm_Pi[stage](
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=not O_should_accumulate,
                                mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                                mbar_phase=P_full_O_rescaled_phase,
                            )
                        # 4. release accumulated O0_partial / O1_partial
                        # 5. release V(i-1)
                        if const_expr(stage == self.q_stage - 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        # 2. gemm
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        if const_expr(self.blockscaled):
                            if stage == 0:
                                cute.arch.barrier(
                                    barrier_id=self.mbar_mma_p1_id,
                                    number_of_threads=self.mbar_mma_threads,
                                )
                            else:
                                cute.arch.barrier(
                                    barrier_id=self.mbar_mma_p0_id,
                                    number_of_threads=self.mbar_mma_threads,
                                )
                            # Copy SF from SMEM to TMEM
                            s2t_sfq_stage_coord = (None, None, None, None, stage)
                            s2t_sfk_stage_coord = (None, None, None, None, Ki_index)
                            cute.copy(
                                tiled_copy_s2t_sfq,
                                tCsSFQ_compact_s2t[s2t_sfq_stage_coord],
                                tCtSFQ_compact_s2ts[stage],
                            )
                            cute.copy(
                                tiled_copy_s2t_sfk,
                                tCsSFK_compact_s2t[s2t_sfk_stage_coord],
                                tCtSFK_compact_s2ts[stage],
                            )
                            cute.arch.fence_view_async_tmem_store()
                            sm100_utils.gemm_blockscaled(
                                tiled_mma_qk,
                                tStSs[stage],
                                tSrQs[stage],
                                tSrK[None, None, None, Ki_index],
                                tCtSFQs[stage],
                                tCtSFKs[stage],
                                zero_init=True,
                            )
                        else:
                            gemm_Si[stage](
                                tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur
                            )
                        # 3. release S0
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.q_stage):
                        tcgen05.commit(
                            # pyre-ignore
                            mbar_ptr
                            + self.mbar_load_q_empty_offset  # pyre-ignore
                            + stage  # pyre-ignore
                        )  # pyre-ignore

                if const_expr(not self.unroll_kv):
                    # unroll_kv disabled: regular last PV, then advance (original order)
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index, Vi_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                            P_full_O_rescaled_phase,
                        )
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        if const_expr(self.blockscaled):
                            s2t_sfp_stage_coord = (None, None, None, None, stage)
                            cute.copy(
                                tiled_copy_s2t_sfp,
                                tCsSFP_compact_s2t[s2t_sfp_stage_coord],
                                tCtSFP_compact_s2ts[stage],
                            )
                            s2t_sfv_stage_coord = (None, None, None, None, Vi_index)
                            cute.copy(
                                tiled_copy_s2t_sfv,
                                tCsSFV_compact_s2t[s2t_sfv_stage_coord],
                                tCtSFV_compact_s2ts[stage],
                            )
                            cute.arch.fence_view_async_tmem_store()
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_P_full_2_offset + stage,
                                P_full_O_rescaled_phase,
                            )
                            gemm_Pi_blockscaled[stage](  # pyre-ignore
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=not O_should_accumulate,
                            )
                        else:
                            gemm_Pi[stage](
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=not O_should_accumulate,
                                mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                                mbar_phase=P_full_O_rescaled_phase,
                            )
                        with cute.arch.elect_one():
                            tcgen05.commit(
                                # pyre-ignore
                                mbar_ptr
                                + self.mbar_O_full_offset  # pyre-ignore
                                + stage  # pyre-ignore
                            )  # pyre-ignore
                    P_full_O_rescaled_phase ^= 1
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    # Advance to next tile after last PV
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                else:
                    # Advance to next tile before last PV
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                    is_next_tile_valid = work_tile.is_valid_tile
                    # unroll_kv enabled: check next tile validity at runtime
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index, Vi_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tOrVi = tOrV[None, None, None, Vi_index]
                    # Initialize before control flow (DSL requires initial value
                    # before dynamic branches that may define the variable)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.blockscaled):
                        s2t_sfp_stage_coord = (None, None, None, None, 0)
                        s2t_sfv_stage_coord = (None, None, None, None, Vi_index)
                    if is_next_tile_valid:
                        # Transition: overlap last PV of current tile with first QK of next tile
                        # Hides prologue latency behind PV GEMM pipeline
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        for stage in cutlass.range_constexpr(self.q_stage):
                            # --- Last PV GEMM of current tile ---
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                                P_full_O_rescaled_phase,
                            )
                            sV_cur = sV[None, None, None, Vi_index]
                            if const_expr(self.uneven_kv_smem):
                                sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                            if const_expr(self.blockscaled):
                                s2t_sfp_stage_coord = (None, None, None, None, stage)
                                cute.copy(
                                    tiled_copy_s2t_sfp,
                                    tCsSFP_compact_s2t[s2t_sfp_stage_coord],
                                    tCtSFP_compact_s2ts[stage],
                                )
                                s2t_sfv_stage_coord = (None, None, None, None, Vi_index)
                                cute.copy(
                                    tiled_copy_s2t_sfv,
                                    tCsSFV_compact_s2t[s2t_sfv_stage_coord],
                                    tCtSFV_compact_s2ts[stage],
                                )
                                cute.arch.fence_view_async_tmem_store()
                                cute.arch.mbarrier_wait(
                                    mbar_ptr + self.mbar_P_full_2_offset + stage,
                                    P_full_O_rescaled_phase,
                                )
                                gemm_Pi_blockscaled[stage](  # pyre-ignore
                                    tCrB=tOrVi,
                                    sB=sV_cur,
                                    zero_init=not O_should_accumulate,
                                )
                            else:
                                gemm_Pi[stage](
                                    tCrB=tOrVi,
                                    sB=sV_cur,
                                    zero_init=not O_should_accumulate,
                                    mbar_ptr=mbar_ptr
                                    + self.mbar_P_full_2_offset
                                    + stage,
                                    mbar_phase=P_full_O_rescaled_phase,
                                )
                            # Commit O_full — tells correction warps O is ready
                            with cute.arch.elect_one():
                                tcgen05.commit(
                                    mbar_ptr + self.mbar_O_full_offset + stage
                                )

                            # Release V after last stage PV GEMM (staggered release)
                            if const_expr(stage == self.q_stage - 1):
                                pipeline_kv.consumer_release(mma_kv_release_state)
                                mma_kv_release_state.advance()

                            # --- First QK GEMM of next tile (overlapped) ---
                            # Wait for Q (next tile)
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_load_q_full_offset + stage,
                                mma_q_consumer_phase,
                            )
                            # Wait for K (next tile, stage 0 only)
                            if const_expr(stage == 0):
                                mma_kv_consumer_state.advance()
                                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                            tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                            sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                            if const_expr(self.uneven_kv_smem):
                                sK_cur = self.offset_kv_smem(
                                    sK_cur,
                                    mma_kv_consumer_state.index,
                                    mma_kv_consumer_state.phase,
                                )
                            if const_expr(self.blockscaled):
                                # Barrier for SF TMEM safety — hidden behind PV GEMM pipeline
                                if stage == 0:
                                    cute.arch.barrier(
                                        barrier_id=self.mbar_mma_p1_id,
                                        number_of_threads=self.mbar_mma_threads,
                                    )
                                else:
                                    cute.arch.barrier(
                                        barrier_id=self.mbar_mma_p0_id,
                                        number_of_threads=self.mbar_mma_threads,
                                    )
                                # Copy SF from SMEM to TMEM
                                s2t_sfq_stage_coord = (None, None, None, None, stage)
                                s2t_sfk_stage_coord = (
                                    None,
                                    None,
                                    None,
                                    None,
                                    mma_kv_consumer_state.index,
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfq,
                                    tCsSFQ_compact_s2t[s2t_sfq_stage_coord],
                                    tCtSFQ_compact_s2ts[stage],
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfk,
                                    tCsSFK_compact_s2t[s2t_sfk_stage_coord],
                                    tCtSFK_compact_s2ts[stage],
                                )
                                cute.arch.fence_view_async_tmem_store()
                                sm100_utils.gemm_blockscaled(
                                    tiled_mma_qk,
                                    tStSs[stage],
                                    tSrQs[stage],
                                    tSrKi,
                                    tCtSFQs[stage],
                                    tCtSFKs[stage],
                                    zero_init=True,
                                )
                            else:
                                gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                            # Commit S_full
                            with cute.arch.elect_one():
                                tcgen05.commit(
                                    mbar_ptr + self.mbar_S_full_offset + stage
                                )
                        mma_q_consumer_phase ^= 1
                        P_full_O_rescaled_phase ^= 1
                        # Release K of next tile
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                    else:
                        # Regular last PV: no next tile (last tile in persistent loop)
                        for stage in cutlass.range_constexpr(self.q_stage):
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                                P_full_O_rescaled_phase,
                            )
                            sV_cur = sV[None, None, None, Vi_index]
                            if const_expr(self.uneven_kv_smem):
                                sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                            if const_expr(self.blockscaled):
                                s2t_sfp_stage_coord = (None, None, None, None, stage)
                                cute.copy(
                                    tiled_copy_s2t_sfp,
                                    tCsSFP_compact_s2t[s2t_sfp_stage_coord],
                                    tCtSFP_compact_s2ts[stage],
                                )
                                s2t_sfv_stage_coord = (None, None, None, None, Vi_index)
                                cute.copy(
                                    tiled_copy_s2t_sfv,
                                    tCsSFV_compact_s2t[s2t_sfv_stage_coord],
                                    tCtSFV_compact_s2ts[stage],
                                )
                                cute.arch.fence_view_async_tmem_store()
                                cute.arch.mbarrier_wait(
                                    mbar_ptr + self.mbar_P_full_2_offset + stage,
                                    P_full_O_rescaled_phase,
                                )
                                gemm_Pi_blockscaled[stage](  # pyre-ignore
                                    tCrB=tOrVi,
                                    sB=sV_cur,
                                    zero_init=not O_should_accumulate,
                                )
                            else:
                                gemm_Pi[stage](
                                    tCrB=tOrVi,
                                    sB=sV_cur,
                                    zero_init=not O_should_accumulate,
                                    mbar_ptr=mbar_ptr
                                    + self.mbar_P_full_2_offset
                                    + stage,
                                    mbar_phase=P_full_O_rescaled_phase,
                                )
                            with cute.arch.elect_one():
                                tcgen05.commit(
                                    mbar_ptr + self.mbar_O_full_offset + stage
                                )
                        P_full_O_rescaled_phase ^= 1
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
            else:
                # Empty tile (block_sparsity/split_kv): just advance
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: Optional[list] = None,
        fastdiv_mods: Tuple[
            Optional[FastDivmodDivisor], Optional[FastDivmodDivisor]
        ] = (None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        sSFP: Optional[cute.Tensor] = None,
    ) -> None:
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * (len(self.softmax0_warp_ids))
        )

        tStScale = cute.composition(tStSi, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width  # pyre-ignore
        tStP_layout = cute.composition(
            tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)

        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)),
            Float32,
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(
            tmem_store_scale_atom, tStScale
        ).get_slice(tidx)

        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = (
            # pyre-ignore
            self.mbar_s0_s1_sequence_offset + warp_idx_in_wg  # pyre-ignore
        )  # pyre-ignore

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        is_first_persistent = True
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            shared_mask_kwargs = dict(
                # pyre-ignore[58]
                m_block=self.q_stage * m_block + stage,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_tensors=aux_tensors,
            )

            # Recompute fastdiv_mods if necessary
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    (
                        seqlen_q_divmod
                        if not recompute_fastdiv_mods_q
                        else FastDivmodDivisor(seqlen.seqlen_q)
                    ),
                    (
                        seqlen_k_divmod
                        if not recompute_fastdiv_mods_k
                        else FastDivmodDivisor(seqlen.seqlen_k)
                    ),
                )

            mask_mod = self.mask_mod if const_expr(self.mask_mod is not None) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=mask_mod,
                fastdiv_mods=fastdiv_mods,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):  # pyre-ignore
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    fastdiv_mods=fastdiv_mods,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                # pyre-ignore[6]
                rescale_threshold=8.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(
                    blocksparse_tensors, batch_idx, head_idx, m_block
                )
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(
                    0
                )

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                # pyre-ignore[58]
                m_block=self.q_stage * m_block + stage,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                sSFP=sSFP,
            )

            if has_work:
                # Softmax acts as the producer: wait until correction signals the stage is empty
                cute.arch.mbarrier_wait(
                    mbar_ptr
                    # pyre-ignore
                    + self.mbar_softmax_corr_empty_offset  # pyre-ignore
                    + stage,  # pyre-ignore
                    si_corr_producer_phase,
                )
                si_corr_producer_phase ^= 1

            # Block sparse or dense iteration
            if const_expr(self.use_block_sparsity):
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mbar_ptr,
                    self.mbar_softmax_corr_full_offset,  # pyre-ignore
                    self.mbar_softmax_corr_empty_offset,
                    self.mbar_P_full_O_rescaled_offset,  # pyre-ignore
                    self.mbar_P_full_2_offset,  # pyre-ignore
                    self.q_stage,
                    Int32(stage),
                    is_first_persistent=is_first_persistent,
                )
                if not empty_tile:
                    is_first_persistent = False
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.m_block_size * 2
                        ] = softmax.row_max[0]
                    cute.arch.mbarrier_arrive(
                        # pyre-ignore[58]
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage
                    )
            else:
                if const_expr(not self.is_split_kv) or tile_block_count > Int32(0):
                    (
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                    ) = softmax_step(
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        is_first_persistent=is_first_persistent,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                    )
                    is_first_persistent = False
                    n_block_max -= 1
                    # Next couple of iterations with causal masking
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = (
                            block_info.get_n_block_min_causal_local_mask(
                                seqlen, m_block, n_block_min
                            )
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_causal_local_mask
                        )
                    # The remaining iterations have no masking
                    n_block_min_before_local_mask = (
                        block_info.get_n_block_min_before_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        n_block = n_block_max - n_tile - 1
                        if const_expr(self.mask_mod is not None):
                            (
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        else:
                            (
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                            )
                    # Separate iterations with local masking on the left
                    if const_expr(
                        self.is_local and block_info.window_size_left is not None
                    ):
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_before_local_mask
                        )
                        # pyre-ignore[28]
                        for n_tile in cutlass.range(
                            0, n_block_max - n_block_min, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )

                    # Dense path always writes scale / signals
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.m_block_size * 2
                        ] = softmax.row_max[0]
                    cute.arch.mbarrier_arrive(
                        # pyre-ignore[58]
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage
                    )

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_tensors: Optional[list] = None,
        fastdiv_mods: Tuple[
            Optional[FastDivmodDivisor], Optional[FastDivmodDivisor]
        ] = (None, None),
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        is_first_persistent: bool = False,
        sSFP: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:  # pyre-ignore
        tilePlikeFP32 = (
            # pyre-ignore
            self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width  # pyre-ignore
        )  # pyre-ignore
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tScP = cute.composition(
            tScS, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )

        # Wait for Si
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_S_full_offset + stage,  # pyre-ignore[16, 58]
            mma_si_consumer_phase,
        )
        tSrS_t2r = cute.make_fragment(
            thr_tmem_load.partition_D(tScS).shape,  # pyre-ignore[16]
            self.qk_acc_dtype,
        )
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        # Signal MMA warp that softmax has finished reading S (for blockscaled SF overlap)
        # - Stage 0: Skip signaling on first persistent tile (MMA uses prologue SF in O region)
        # - Stage 1: Always signal MMA_P1 - MMA stage 0 waits on P1 before stage 1
        if const_expr(self.blockscaled):
            if stage == 0:
                # Use if/else instead of `not` to avoid DSL __bool__ error on runtime var
                if is_first_persistent:
                    pass  # First tile: prologue uses O-region SF, no P0 barrier needed
                else:
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.barrier_arrive(
                        barrier_id=self.mbar_mma_p0_id,
                        number_of_threads=self.mbar_mma_threads,
                    )
            else:
                # Stage 1: Always signal - MMA stage 0 waits on P1 before stage 1 runs
                cute.arch.fence_view_async_tmem_load()
                cute.arch.barrier_arrive(
                    barrier_id=self.mbar_mma_p1_id,
                    number_of_threads=self.mbar_mma_threads,
                )

        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_tensors,
                fastdiv_mods,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)  # pyre-ignore

        # Use blockscaled update_row_max to get block maxes for MXFP8 optimization
        block_maxes = None
        if const_expr(self.blockscaled):
            row_max, acc_scale, block_maxes = softmax.update_row_max_blockscaled(
                tSrS_t2r.load(), is_first
            )
        else:
            row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            # tSrScale_r2t = cute.make_fragment(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            # tSrScale_r2t[0] = acc_scale
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            # pyre-ignore[16]
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready
        cute.arch.mbarrier_arrive(
            # pyre-ignore
            mbar_ptr + self.mbar_softmax_corr_full_offset + stage  # pyre-ignore
        )  # pyre-ignore

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        tSrP_r2t_f32 = cute.make_fragment(
            thr_tmem_store.partition_S(tScP).shape,  # pyre-ignore[16]
            Float32,
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),  # pyre-ignore
            tSrS_t2r.layout,
        )
        # Use blockscaled conversion for MXFP8 to compute proper scales
        sf0, sf1, sf2, sf3 = (
            cutlass.Uint8(127),
            cutlass.Uint8(127),
            cutlass.Uint8(127),
            cutlass.Uint8(127),
        )
        if const_expr(self.blockscaled):
            from hammer.v3.ops.cutedsl.fa4_helpers.softmax import (
                E2M1_MAX_NORM_RCP,
                E4M3_MAX_NORM_RCP,
            )

            max_norm_rcp_val = (
                E2M1_MAX_NORM_RCP
                if const_expr(self.q_dtype == cutlass.Float4E2M1FN)
                else E4M3_MAX_NORM_RCP
            )
            sf0, sf1, sf2, sf3 = softmax.scale_apply_exp2_convert_blockscaled(
                tSrS_t2r,
                tSrP_r2t,
                row_max=row_max,
                block_maxes=block_maxes,
                e2e=mask_fn is None and self.head_dim_padded <= 128,
                e2e_freq=self.e2e_freq,  # pyre-ignore
                e2e_res=0,
                max_norm_rcp_val=max_norm_rcp_val,
            )
        else:
            softmax.apply_exp2_convert(
                tSrS_t2r,
                tSrP_r2t,
                e2e=mask_fn is None and self.head_dim_padded <= 128,
                e2e_freq=self.e2e_freq,
            )

        # Fence before P R2T store for blockscaled: ensures apply_exp2_convert_blockscaled
        if const_expr(self.blockscaled):
            cute.arch.fence_view_async_tmem_store()

        # Write SFP (P's scale factors) to SMEM BEFORE mbar_P_full_O_rescaled arrive.
        if const_expr(self.blockscaled and sSFP is not None):
            from hammer.v3.ops.cutedsl.fa4_helpers.softmax import pack_4xu8_to_u32

            tidx = thr_tmem_load.thr_idx
            scale_packed = pack_4xu8_to_u32(sf0, sf1, sf2, sf3)
            # pyre-ignore[16]
            sSFP_u32_ptr = cute.recast_ptr(sSFP.iterator, dtype=cutlass.Uint32)
            sSFP_filtered = cute.filter_zeros(sSFP)
            sSFP_grouped = cute.group_modes(
                sSFP_filtered, 0, cute.rank(sSFP_filtered.layout) - 1
            )
            sSFP_u32_layout = cute.recast_layout(32, 8, sSFP_grouped.layout)
            sSFP_u32 = cute.make_tensor(sSFP_u32_ptr, sSFP_u32_layout)
            sfp_stage = Int32(0) if stage == 0 else Int32(1)
            # Each thread writes its portion of scale factors
            if tidx < cute.size(sSFP_u32.shape[0]):
                sSFP_u32[tidx, sfp_stage] = scale_packed
            # Fence SMEM write to ensure SFP is visible to MMA warp after barrier
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )

        for i in cutlass.range_constexpr(
            cute.size(tStP_r2t.shape[2])  # pyre-ignore[16]
            // self.tmem_store_split_divisor
            * 3
        ):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that P is ready (and SFP is in SMEM)
        cute.arch.mbarrier_arrive(
            # pyre-ignore
            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage  # pyre-ignore
        )  # pyre-ignore
        for i in cutlass.range_constexpr(
            cute.size(tStP_r2t.shape[2]) // self.tmem_store_split_divisor * 3,
            cute.size(tStP_r2t.shape[2]),
        ):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.mbarrier_arrive(
            # pyre-ignore
            mbar_ptr + self.mbar_P_full_2_offset + stage  # pyre-ignore
        )  # pyre-ignore

        if const_expr(self.unroll_kv):
            # Overlap update_row_sum with correction warp processing to reduce cascade stalls.
            softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)

        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage,  # pyre-ignore
            si_corr_producer_phase,
        )

        if const_expr(not self.unroll_kv):
            softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        return (
            mma_si_consumer_phase ^ 1,
            si_corr_producer_phase ^ 1,
            s0_s1_sequence_phase ^ 1,
        )

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtOs: tuple[cute.Tensor],
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        mbar_ptr: cute.Pointer,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tStScale_layout = cute.composition(
            tStS.layout, cute.make_layout((self.m_block_size, 1))
        )
        tStScales = tuple(
            cute.make_tensor(
                tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout
            )
            for stage in range(self.q_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
            self.qk_acc_dtype,
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(
            tmem_load_v_atom, tStScales[0]
        ).get_slice(tidx)

        tStScales_t2r = [
            thr_tmem_load_vec.partition_S(tStScales[stage])
            for stage in range(self.q_stage)
        ]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # First iter: no correction is required
        for _s in cutlass.range_constexpr(self.q_stage):
            cute.arch.mbarrier_arrive(
                mbar_ptr + self.mbar_P_full_O_rescaled_offset + _s  # pyre-ignore
            )

        softmax_corr_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            if const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_O(mO, batch_idx, dim=3)[
                    None, None, head_idx, split_idx
                ]
            else:
                mO_cur = seqlen.offset_batch_O(mO, batch_idx, dim=3)[
                    None, None, head_idx
                ]
            gO = cute.local_tile(
                mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0)
            )

            # Default LSE to -inf for invalid split_idx tiles
            stats = [
                (
                    0.0,
                    (
                        -Float32.inf
                        if const_expr(mLSE is not None or learnable_sink is not None)
                        else None
                    ),
                    True,
                )
            ] * self.q_stage

            if const_expr(self.use_block_sparsity):  # pyre-ignore
                total_block_count = get_total_block_count(
                    blocksparse_tensors, batch_idx, head_idx, m_block
                )
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = const_expr(
                    not self.is_split_kv
                ) or total_block_count > Int32(0)

            if has_work:
                # Ignore first signal from softmax as no correction is required
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + 0,  # pyre-ignore
                    softmax_corr_consumer_phase,
                )
                cute.arch.mbarrier_arrive(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + 0  # pyre-ignore
                )
                if const_expr(self.q_stage >= 2):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + 1,
                        softmax_corr_consumer_phase,
                    )
                softmax_corr_consumer_phase ^= 1

                tSrScale_t2r = cute.make_fragment(tSrScale_t2r_shape, Float32)
                for _ in cutlass.range(total_block_count - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait for S0 / S1
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                            softmax_corr_consumer_phase,
                        )
                        # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                        # cute.arch.fence_view_async_tmem_load()
                        # scale = tSrScale_t2r[0]
                        scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        # Fence TMEM loads to ensure MMA warp's TMEM writes are visible
                        # to correction warps after mbarrier synchronization
                        cute.arch.fence_view_async_tmem_load()
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv,
                                tOtOs[stage if self.q_stage == 2 else 0],
                                tidx,
                                scale,
                            )
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage
                        )
                        if const_expr(self.q_stage >= 2):
                            cute.arch.mbarrier_arrive(
                                mbar_ptr
                                + self.mbar_softmax_corr_empty_offset
                                + (1 - stage)
                            )
                    softmax_corr_consumer_phase ^= 1
                    # o_corr_consumer_phase ^= 1
                if const_expr(self.q_stage >= 2):
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_empty_offset + 1
                    )
                else:
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_empty_offset + 0
                    )
                # End of seqlen_corr_loop_steps

                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        # pyre-ignore[16]
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                (self.q_stage * m_block + stage) * self.m_block_size
                                + tidx
                                # pyre-ignore[58]
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            # pyre-ignore[6]
                            learnable_sink_val[stage] = Float32(
                                learnable_sink[q_head_idx]
                            )
                for stage in cutlass.range_constexpr(self.q_stage):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                        softmax_corr_consumer_phase,
                    )
                    # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[0]
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[
                            tidx + stage * self.m_block_size + self.m_block_size * 2
                        ]
                    else:
                        row_max = None
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_empty_offset + stage
                    )
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                row_max = sink_val * (LOG2_E / softmax_scale_log2)
                                row_sum = Float32(1.0)
                            else:
                                row_sum += utils.exp2f(
                                    # pyre-ignore[58]
                                    sink_val * LOG2_E - row_max * softmax_scale_log2
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(
                        row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0
                    )
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_O_full_offset + stage,  # pyre-ignore
                        o_corr_consumer_phase,
                    )
                    # Fence TMEM loads to ensure MMA warp's TMEM writes are visible
                    # to correction warps after mbarrier synchronization
                    cute.arch.fence_view_async_tmem_load()
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_wait(
                            mbar_ptr
                            # pyre-ignore
                            + self.mbar_corr_epi_empty_offset  # pyre-ignore
                            + stage,  # pyre-ignore
                            corr_epi_producer_phase,
                        )
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtOs[stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O,
                    )
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_arrive(
                            mbar_ptr
                            # pyre-ignore
                            + self.mbar_corr_epi_full_offset  # pyre-ignore
                            + stage  # pyre-ignore
                        )
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage
                    )

                o_corr_consumer_phase ^= 1
                softmax_corr_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                else:
                    gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_block_sparsity):
                    (
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtOs,
                        sO,
                        mbar_ptr,
                        self.mbar_softmax_corr_full_offset,
                        self.mbar_softmax_corr_empty_offset,
                        self.mbar_P_full_O_rescaled_offset,
                        self.mbar_P_full_2_offset,  # pyre-ignore
                        self.mbar_corr_epi_full_offset,
                        self.mbar_corr_epi_empty_offset,
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    # Use offset_o for LSE (matches O addressing, not Q)
                    offset = (
                        seqlen.offset_o
                        if const_expr(not self.pack_gqa)
                        else (0, seqlen.offset_o)
                    )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset(
                            (offset,), mLSE[None, head_idx, split_idx]
                        )
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    gLSE = cute.local_tile(
                        mLSE_cur,
                        (self.m_block_size,),
                        (self.q_stage * m_block + stage,),
                    )
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + utils.log2f(row_sum)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = (
                        seqlen.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen.seqlen_q * self.qhead_per_kvhead
                    )
                    if (
                        tidx
                        < seqlen_q
                        - (self.q_stage * m_block + stage) * self.m_block_size
                    ):
                        gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    # SiLU loop for both softmax0 and softmax1 warp groups (mirrors softmax_loop)
    @cute.jit
    def silu_loop(
        self,
        stage: int | Int32,
        softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mAttnScale: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
    ) -> None:
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * (len(self.softmax0_warp_ids))
        )

        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width  # pyre-ignore
        tStP_layout = cute.composition(
            tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = (
            # pyre-ignore
            self.mbar_s0_s1_sequence_offset + warp_idx_in_wg  # pyre-ignore
        )  # pyre-ignore

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            tile_block_count = n_block_max - n_block_min
            has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(0)

            silu_step = partial(
                self.silu_step,
                softmax_scale=softmax_scale,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                tStS_t2r=tStS_t2r,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                # pyre-ignore[58]
                m_block=self.q_stage * m_block + stage,
                seqlen_q=seqlen.seqlen_q,
                seqlen_k=seqlen.seqlen_k,
                mAttnScale=mAttnScale,
                window_size_left=window_size_left,
            )

            if has_work:
                # First iteration (highest n_block): boundary + causal/local masking
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = silu_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    n_block=n_block_max - 1,
                    apply_causal_mask=self.is_causal or self.is_local,
                    mask_seqlen=True,
                )
                n_block_max -= 1
                # Next iterations with causal masking (no boundary)
                if const_expr(self.is_causal or self.is_local):
                    n_block_min_causal_local_mask = (
                        block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_causal_local_mask, unroll=1
                    ):
                        n_block = n_block_max - 1 - n_tile
                        (
                            mma_si_consumer_phase,
                            si_corr_producer_phase,
                            s0_s1_sequence_phase,
                        ) = silu_step(
                            mma_si_consumer_phase,
                            si_corr_producer_phase,
                            s0_s1_sequence_phase,
                            n_block=n_block,
                            apply_causal_mask=True,
                        )
                    n_block_max = cutlass.min(
                        n_block_max, n_block_min_causal_local_mask
                    )
                # Unmasked iterations (between local-left boundary and causal boundary)
                n_block_min_before_local_mask = (
                    block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                )
                for n_tile in cutlass.range(
                    n_block_max - n_block_min_before_local_mask, unroll=1
                ):
                    (
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                    ) = silu_step(
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                    )
                # Local left-window masking iterations
                if const_expr(
                    self.is_local and block_info.window_size_left is not None
                ):
                    n_block_max = cutlass.min(
                        n_block_max, n_block_min_before_local_mask
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                        n_block = n_block_max - 1 - n_tile
                        (
                            mma_si_consumer_phase,
                            si_corr_producer_phase,
                            s0_s1_sequence_phase,
                        ) = silu_step(
                            mma_si_consumer_phase,
                            si_corr_producer_phase,
                            s0_s1_sequence_phase,
                            n_block=n_block,
                            apply_causal_mask=True,
                        )

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def silu_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        softmax_scale: Float32,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        n_block: Int32 = Int32(0),
        m_block: Int32 = Int32(0),
        seqlen_q: Int32 = Int32(0),
        seqlen_k: Int32 = Int32(0),
        apply_causal_mask: bool = False,
        mask_seqlen: bool = False,
        mAttnScale: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        tilePlikeFP32 = (
            # pyre-ignore
            self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width  # pyre-ignore
        )  # pyre-ignore
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScP = cute.composition(
            tScS, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )

        # Wait for Si from MMA
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_S_full_offset + stage,  # pyre-ignore[16, 58]
            mma_si_consumer_phase,
        )
        tSrS_t2r = cute.make_fragment(
            thr_tmem_load.partition_D(tScS).shape,  # pyre-ignore[16]
            self.qk_acc_dtype,
        )
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        # Apply alpha scaling and SiLU activation: silu(x) = x/2 * (tanh(x/2) + 1)
        # half_scale = softmax_scale * 0.5 (the LN2 roundtrip cancels out)
        half_scale = softmax_scale * Float32(0.5)

        if const_expr(mAttnScale is not None):
            # Apply SiLU with alpha only, then scale by attn_scale after
            c_half_scale = (half_scale, half_scale)
            tScS_t2r_fused = thr_tmem_load.partition_D(tScS)
            for i in cutlass.range_constexpr(0, cute.size(tSrS_t2r), 2):
                row_scale = mAttnScale[  # pyre-ignore[16]
                    tScS_t2r_fused[i][0] + m_block * self.mma_tiler_qk[0]
                ]
                x = (tSrS_t2r[i], tSrS_t2r[i + 1])
                half_x = cute.arch.mul_packed_f32x2(x, c_half_scale)
                t0 = llvm.inline_asm(
                    cutlass.Float32.mlir_type,
                    [half_x[0].ir_value()],
                    "tanh.approx.f32 $0, $1;",
                    "=f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
                t1 = llvm.inline_asm(
                    cutlass.Float32.mlir_type,
                    [half_x[1].ir_value()],
                    "tanh.approx.f32 $0, $1;",
                    "=f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
                r0, r1 = cute.arch.fma_packed_f32x2(
                    half_x, (Float32(t0), Float32(t1)), half_x
                )
                tSrS_t2r[i] = r0 * row_scale
                tSrS_t2r[i + 1] = r1 * row_scale
        else:
            c_half_scale = (half_scale, half_scale)
            for i in cutlass.range_constexpr(0, cute.size(tSrS_t2r), 2):
                x = (tSrS_t2r[i], tSrS_t2r[i + 1])
                half_x = cute.arch.mul_packed_f32x2(x, c_half_scale)
                t0 = llvm.inline_asm(
                    cutlass.Float32.mlir_type,
                    [half_x[0].ir_value()],
                    "tanh.approx.f32 $0, $1;",
                    "=f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
                t1 = llvm.inline_asm(
                    cutlass.Float32.mlir_type,
                    [half_x[1].ir_value()],
                    "tanh.approx.f32 $0, $1;",
                    "=f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
                r0, r1 = cute.arch.fma_packed_f32x2(
                    half_x, (Float32(t0), Float32(t1)), half_x
                )
                tSrS_t2r[i] = r0
                tSrS_t2r[i + 1] = r1

        # Apply seqlen boundary masking
        # For boundary tiles where seqlen_k doesn't fill the tile, zero out-of-bounds cols
        if const_expr(mask_seqlen):
            seqlenk_col_limit = seqlen_k - n_block * self.mma_tiler_qk[1]
            if const_expr(not apply_causal_mask):
                # Non-causal: only seqlen boundary masking needed
                mask_r2p_zero(tSrS_t2r, seqlenk_col_limit)

        # Apply causal mask: zero out positions where col >= col_limit_right
        if const_expr(apply_causal_mask):
            tScS_t2r = thr_tmem_load.partition_D(tScS)
            # Global row for this thread (all elements share the same row)
            row_idx = tScS_t2r[0][0] + m_block * self.mma_tiler_qk[0]
            if const_expr(self.is_diagonal):
                # DIAGONAL: attend where q_pos == k_pos (no delta shift)
                causal_row_offset = Int32(1) - n_block * self.mma_tiler_qk[1]
            else:
                # Causal: col must be < row + 1 + (seqlen_k - seqlen_q)
                causal_row_offset = (
                    Int32(1) + seqlen_k - n_block * self.mma_tiler_qk[1] - seqlen_q
                )
            col_limit_right = row_idx + causal_row_offset
            # Also clamp by seqlen_k boundary if needed
            if const_expr(mask_seqlen):
                col_limit_right = cutlass.min(
                    col_limit_right,
                    seqlenk_col_limit,  # pyre-ignore[61]
                )
            mask_r2p_zero(tSrS_t2r, col_limit_right)

            # Apply left-window masking for LOCAL/DIAGONAL: zero cols < col_limit_left
            if const_expr(window_size_left is not None):
                col_limit_left = (
                    row_idx + causal_row_offset - window_size_left - Int32(1)
                )
                # pyre-ignore[6]
                col_limit_left = max(col_limit_left, Int32(0))
                mask_r2p_zero_left(tSrS_t2r, col_limit_left)

        # attn_scale is applied as a post-SiLU multiplier when mAttnScale is not None

        # Convert to P format for TMEM store
        tSrP_r2t_f32 = cute.make_fragment(
            thr_tmem_store.partition_S(tScP).shape,  # pyre-ignore[16]
            Float32,
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),  # pyre-ignore
            tSrS_t2r.layout,
        )
        # Convert f32 SiLU result to bf16 packed format
        from hammer.v3.ops.cutedsl.fa4_helpers import utils as fa4_utils

        fa4_utils.cvt_f16(tSrS_t2r, tSrP_r2t)

        # Write P to TMEM (two-phase, same as softmax_step)
        for i in cutlass.range_constexpr(
            cute.size(tStP_r2t.shape[2])  # pyre-ignore[16]
            // self.tmem_store_split_divisor
            * 3
        ):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
        cute.arch.fence_view_async_tmem_store()
        # Notify MMA warp that P is ready
        cute.arch.mbarrier_arrive(
            # pyre-ignore
            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage  # pyre-ignore
        )  # pyre-ignore
        for i in cutlass.range_constexpr(
            cute.size(tStP_r2t.shape[2]) // self.tmem_store_split_divisor * 3,
            cute.size(tStP_r2t.shape[2]),
        ):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
        cute.arch.fence_view_async_tmem_store()
        # Notify MMA warp that the 2nd half of P is ready
        cute.arch.mbarrier_arrive(
            # pyre-ignore
            mbar_ptr + self.mbar_P_full_2_offset + stage  # pyre-ignore
        )  # pyre-ignore

        return (
            mma_si_consumer_phase ^ 1,
            si_corr_producer_phase ^ 1,
            s0_s1_sequence_phase ^ 1,
        )

    @cute.jit
    def correction_loop_silu(
        self,
        thr_mma_pv: cute.core.ThrMma,
        tOtOs: tuple[cute.Tensor],
        sO: cute.Tensor,
        mO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )

        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            mO_cur = seqlen.offset_batch_O(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = cute.local_tile(
                mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0)
            )

            total_block_count = n_block_max - n_block_min
            has_work = const_expr(not self.is_split_kv) or total_block_count > Int32(0)

            if has_work:
                # Final epilogue: write O to gmem
                for stage in cutlass.range_constexpr(self.q_stage):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_O_full_offset + stage,  # pyre-ignore
                        o_corr_consumer_phase,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_wait(
                            mbar_ptr
                            # pyre-ignore
                            + self.mbar_corr_epi_empty_offset  # pyre-ignore
                            + stage,  # pyre-ignore
                            corr_epi_producer_phase,
                        )
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtOs[stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen.seqlen_q,
                        Float32(1.0),  # No normalization for SiLU
                        sO[None, None, stage],
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O,
                    )
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_arrive(
                            mbar_ptr
                            # pyre-ignore
                            + self.mbar_corr_epi_full_offset  # pyre-ignore
                            + stage  # pyre-ignore
                        )

                o_corr_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.composition(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_fragment((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_fragment(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(
                tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout
            )
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOtO_r2t_i = cute.make_tensor(
                tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout
            )
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        corr_tile_size = 32 * 8 // self.o_dtype.width  # pyre-ignore
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.logical_divide(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOsO_i = cute.logical_divide(
            tOsO, cute.make_layout((self.m_block_size, corr_tile_size))
        )

        epi_subtile = (self.epi_tile[0], corr_tile_size)  # pyre-ignore
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,  # pyre-ignore
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO_i[(None, None), 0]
        ).get_slice(tidx)
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range_constexpr(self.head_dim_v_padded // corr_tile_size):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_fragment(
                tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            if const_expr(not self.use_silu):
                for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                    tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                        (tOrO_frg[j], tOrO_frg[j + 1]),
                        (scale, scale),
                    )
            tOrO_frg_cvt = cute.make_fragment(tOrO_frg.shape, self.o_dtype)
            tOrO_frg_cvt.store(tOrO_frg.load().to(self.o_dtype))
            cute.copy(tiled_smem_store, tOrO_frg_cvt, tOsO_r2s_i)
        # fence view async shared
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

        if const_expr(self.use_correction_warps_for_epi):
            assert not self.use_tma_O  # pyre-ignore
            assert gmem_tiled_copy_O is not None
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE,
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            # pyre-ignore[16]
            tOpO = utils.predicate_k(tOcO, limit=mO_cur.shape[1])
            assert not self.pack_gqa
            pack_gqa = PackGQA(
                # pyre-ignore[6]
                self.m_block_size,
                # pyre-ignore[6]
                self.head_dim_v_padded,
                self.check_hdim_v_oob,
                # pyre-ignore[6]
                self.qhead_per_kvhead,
            )

            # load acc O from smem to rmem for wider vectorization
            tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
            cute.autovec_copy(tOsO, tOrO)

            # copy acc O from rmem to gmem
            if const_expr(not self.pack_gqa):
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if (
                        t0OcO[0, rest_m, 0][0]
                        < seqlen_q
                        - (self.q_stage * m_block + stage) * self.m_block_size
                        - tOcO[0][0]
                    ):
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None, self.q_stage * m_block + stage],
                            pred=tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None,
                        )
            else:
                pack_gqa.store_O(
                    mO_cur,
                    tOrO,
                    gmem_tiled_copy_O,
                    tidx,
                    self.q_stage * m_block + stage,
                    seqlen_q,
                )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ) -> None:
        epi_consumer_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_O(mO, batch_idx, dim=3)[
                        None, None, head_idx, split_idx
                    ]
                else:
                    mO_cur = seqlen.offset_batch_O(mO, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                gO = cute.local_tile(
                    mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0)
                )
                if const_expr(self.use_tma_O):  # pyre-ignore[16]
                    store_O, _, _ = copy_utils.tma_get_copy_fn(  # pyre-ignore[23]
                        tma_atom_O,  # pyre-ignore[6]
                        0,
                        cute.make_layout(1),
                        sO,
                        gO,
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        cute.arch.mbarrier_wait(
                            mbar_ptr
                            # pyre-ignore
                            + self.mbar_corr_epi_full_offset  # pyre-ignore
                            + stage,  # pyre-ignore
                            epi_consumer_phase,
                        )
                        # 2. copy O0 / O1 to gmem
                        store_O(src_idx=stage, dst_idx=self.q_stage * m_block + stage)
                        cute.arch.cp_async_bulk_commit_group()
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Ensure O0 / O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(1 - stage, read=True)
                        cute.arch.mbarrier_arrive(
                            mbar_ptr
                            # pyre-ignore
                            + self.mbar_corr_epi_empty_offset  # pyre-ignore
                            + stage  # pyre-ignore
                        )
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                    tOsO = gmem_thr_copy_O.partition_S(sO)
                    cO = cute.make_identity_tensor(
                        (self.m_block_size, self.head_dim_v_padded)
                    )
                    tOgO = gmem_thr_copy_O.partition_D(gO)
                    tOcO = gmem_thr_copy_O.partition_S(cO)
                    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                    # pyre-ignore[16]
                    tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                    assert not self.pack_gqa
                    pack_gqa = PackGQA(
                        # pyre-ignore[6]
                        self.m_block_size,
                        # pyre-ignore[6]
                        self.head_dim_v_padded,
                        self.check_hdim_v_oob,
                        # pyre-ignore[6]
                        self.qhead_per_kvhead,
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage,
                            epi_consumer_phase,
                        )
                        # 2. copy O0 / O1 to gmem
                        # load acc O from smem to rmem for wider vectorization
                        tOrO = cute.make_fragment_like(
                            tOsO[None, None, None, 0],
                            self.o_dtype,  # pyre-ignore[16]
                        )
                        cute.autovec_copy(tOsO[None, None, None, stage], tOrO)

                        # copy acc O from rmem to gmem
                        if const_expr(not self.pack_gqa):
                            for rest_m in cutlass.range_constexpr(
                                cute.size(tOrO.shape[1])
                            ):
                                if (
                                    t0OcO[0, rest_m, 0][0]
                                    < seqlen.seqlen_q
                                    - (self.q_stage * m_block + stage)
                                    * self.m_block_size
                                    - tOcO[0][0]
                                ):
                                    cute.copy(
                                        gmem_tiled_copy_O,
                                        tOrO[None, rest_m, None],
                                        tOgO[
                                            None,
                                            rest_m,
                                            None,
                                            self.q_stage * m_block + stage,
                                        ],
                                        pred=(
                                            tOpO[None, rest_m, None]
                                            if const_expr(self.check_hdim_v_oob)
                                            else None
                                        ),
                                    )
                        else:
                            pack_gqa.store_O(
                                mO_cur,
                                tOrO,
                                gmem_tiled_copy_O,
                                tidx,
                                self.q_stage * m_block + stage,
                                seqlen.seqlen_q,
                            )
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + stage
                        )

                epi_consumer_phase ^= 1

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
        # Scale factor TMA parameters (for blockscaled MXFP8)
        tma_atom_SFQ: Optional[cute.CopyAtom] = None,
        tSFQgSFQ: Optional[cute.Tensor] = None,
        tSFQsSFQ: Optional[cute.Tensor] = None,
    ):
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_full_ptr + stage,
                self.tma_copy_bytes["Q"],  # pyre-ignore[16]
            )
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)
        # Load scale factor SFQ together with Q (using same barrier)
        if const_expr(self.blockscaled):
            cute.copy(
                tma_atom_SFQ,
                # pyre-ignore[16]
                tSFQgSFQ[None, block],
                tSFQsSFQ[None, stage],
                tma_bar_ptr=mbar_full_ptr + stage,
            )

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,  # pyre-ignore
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        # Scale factor TMA parameters for SFK
        tma_atom_SFK: Optional[cute.CopyAtom] = None,
        tSFKgSFK: Optional[cute.Tensor] = None,
        tSFKsSFK: Optional[cute.Tensor] = None,
        # Scale factor TMA parameters for SFV
        tma_atom_SFV: Optional[cute.CopyAtom] = None,
        tSFVgSFV: Optional[cute.Tensor] = None,
        tSFVsSFV: Optional[cute.Tensor] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):  # pyre-ignore
            if stage == 0:
                cute.arch.mbarrier_wait(mbar_empty_ptr + 1, phase)

        if const_expr(self.use_tma_KV):
            assert tXgX is not None and tXsX is not None and tma_atom is not None
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_full_ptr + stage,
                    self.tma_copy_bytes[K_or_V],  # pyre-ignore
                )
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            tXgX_cur = (
                tXgX[None, block]
                if const_expr(page_idx is None)
                else tXgX[None, 0, page_idx]
            )
            cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)

            # Load scale factor SFK together with K (using same barrier)
            if const_expr(K_or_V == "K" and self.blockscaled):
                # pyre-ignore[16]
                tSFKsSFK_cur = tSFKsSFK[None, stage]
                tSFKgSFK_cur = (
                    tSFKgSFK[None, block]
                    if const_expr(page_idx is None)
                    else tSFKgSFK[None, 0, page_idx]
                )
                cute.copy(
                    tma_atom_SFK,
                    tSFKgSFK_cur,
                    tSFKsSFK_cur,
                    tma_bar_ptr=mbar_full_ptr + stage,
                )

            # Load scale factor SFV together with V (using same barrier)
            if const_expr(
                K_or_V == "V" and self.blockscaled and tma_atom_SFV is not None
            ):
                tSFVsSFV_cur = tSFVsSFV[None, stage]
                tSFVgSFV_cur = (
                    tSFVgSFV[None, block]
                    if const_expr(page_idx is None)
                    else tSFVgSFV[None, 0, page_idx]
                )
                cute.copy(
                    tma_atom_SFV,
                    tSFVgSFV_cur,
                    tSFVsSFV_cur,
                    tma_bar_ptr=mbar_full_ptr + stage,
                )
        else:
            assert paged_kv_manager is not None
            paged_kv_manager.load_KV(block, sX[None, None, None, stage], K_or_V)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_mbarrier_arrive_noinc(mbar_full_ptr + stage)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):  # pyre-ignore
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = (
                # pyre-ignore
                0
                if stage != 1
                else self.uneven_kv_smem_offset * (1 - 2 * phase)  # pyre-ignore
            )  # pyre-ignore
            # pyre-ignore[6]
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        load_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        if self.use_tma_KV:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.load_warp_ids)
            )
            return cutlass.pipeline.PipelineTmaUmma.create(
                barrier_storage=load_kv_mbar_ptr,
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
            )
        else:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread,
                len(self.load_warp_ids) * cute.arch.WARP_SIZE,
            )
            return cutlass.pipeline.PipelineAsyncUmma.create(
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                barrier_storage=load_kv_mbar_ptr,
            )

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset(
            (m_block * self.m_block_size, n_block * self.n_block_size), cS
        )
        tScS = thr_mma_qk.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa):
            q_physical = q_idx_logical
            q_idx_logical = q_physical // self.qhead_per_kvhead
            head_offset = q_physical - q_idx_logical * self.qhead_per_kvhead
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = divmod(q_idx_logical, seqlen_q_divmod)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=(
                self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1
            ),
        )


# =============================================================================
# Constants
# =============================================================================

MASK_CAUSAL = MaskType.CAUSAL.value
MASK_ALL = MaskType.ALL.value
MASK_DIAGONAL = MaskType.DIAGONAL.value
MASK_NULL = MaskType.NULL.value

LOG2_E = 1.4426950408889634074

# Block sizes
DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# Cache for compiled Hopper kernels
_compiled_kernel_cache_fwd_hopper: Dict[Tuple, object] = {}


# =============================================================================
# FA4 Hopper (SM90) forward - base class and SM90 kernel
# Adapted from ads_mkl/ops/cute_dsl/fa4/src/flash_fwd.py
# =============================================================================


class FlashAttentionForwardBase:
    arch: int = 80

    def __init__(
        self,
        dtype,
        head_dim: int,
        head_dim_v: int | None = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
        Q_in_regs: bool = False,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors: bool = False,
        use_silu: bool = False,
        is_diagonal: bool = False,
    ):
        self.dtype = dtype
        self.use_silu = use_silu
        self.is_diagonal = is_diagonal
        if use_silu:
            score_mod = None
            mask_mod = None
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.pack_gqa = pack_gqa
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.Q_in_regs = Q_in_regs
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.qk_acc_dtype = Float32
        if const_expr(has_aux_tensors):
            self.vec_size = 1
        else:
            self.vec_size = 2

    def _check_type(
        self,
        mQ_type,
        mK_type,
        mV_type,
        mO_type,
        mLSE_type,
        mCuSeqlensQ_type=None,
        mCuSeqlensK_type=None,
        mSeqUsedQ_type=None,
        mSeqUsedK_type=None,
    ):
        if const_expr(not (mQ_type == mK_type == mV_type == mO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ_type not in [None, Int32]):
            raise TypeError("cu_seqlens_q tensor must be Int32")
        if const_expr(mCuSeqlensK_type not in [None, Int32]):
            raise TypeError("cu_seqlens_k tensor must be Int32")
        if const_expr(mSeqUsedQ_type not in [None, Int32]):
            raise TypeError("seqused_q tensor must be Int32")
        if const_expr(mSeqUsedK_type not in [None, Int32]):
            raise TypeError("seqused_k tensor must be Int32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        ) = self._get_smem_layout_atom()
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self.tile_m, self.tile_hdim), (0, 1)
        )
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom, (self.tile_n, self.tile_hdim, self.num_stages), (0, 1, 2)
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom, (self.tile_n, self.tile_hdimv, self.num_stages), (0, 1, 2)
        )
        self.sO_layout = cute.tile_to_shape(
            sO_layout_atom, (self.tile_m, self.tile_hdimv), (0, 1)
        )
        if const_expr(sP_layout_atom is not None):
            self.sP_layout = cute.tile_to_shape(
                sP_layout_atom, (self.tile_m, self.tile_n), (0, 1)
            )
        else:
            self.sP_layout = None

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_Q_load_threads % tQK_shape_dim_1 == 0
        assert self.num_producer_threads % tQK_shape_dim_1 == 0
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        tK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        assert self.tile_m % tQ_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tV_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        assert self.tile_m % tO_layout.shape[0] == 0

        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(
            atom_async_copy, tQ_layout, vQKV_layout
        )
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(
            atom_async_copy, tK_layout, vQKV_layout
        )
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(
            atom_async_copy, tV_layout, vQKV_layout
        )
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

    def _get_smem_layout_atom(self):
        raise NotImplementedError()

    def _get_tiled_mma(self):
        raise NotImplementedError()

    def _get_shared_storage_cls(self):
        raise NotImplementedError()

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom | None,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            # pyre-ignore[16]
            number_of_threads=self.num_epilogue_threads,
        )
        # pyre-ignore[6]
        smem_copy_atom_O = utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(
            tidx
        )
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(
            # pyre-ignore[6]
            self.tile_m,
            # pyre-ignore[6]
            self.tile_hdimv,
            self.check_hdim_v_oob,
            # pyre-ignore[6]
            self.qhead_per_kvhead,
        )

        # Write LSE from rmem -> gmem
        if const_expr(mLSE is not None):
            if const_expr(not seqlen.has_cu_seqlens_q):
                # pyre-ignore[16]
                mLSE_cur = mLSE[None, head_idx, batch_idx]
            else:
                offset = (
                    seqlen.offset_q
                    if const_expr(not self.pack_gqa)
                    else (0, seqlen.offset_q)
                )
                mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded_layout = cute.append(
                    gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
                )
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = utils.make_acc_tensor_mn_view(
                    thr_mma.partition_C(gLSE_expanded)
                )
                assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
                taccOcO = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cO))
                t0accOcO = utils.make_acc_tensor_mn_view(
                    thr_mma.get_slice(0).partition_C(cO)
                )
                if taccOcO[0][1] == 0:
                    # pyre-ignore[16]
                    for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                        if (
                            t0accOcO[m, 0][0]
                            < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                        ):
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa.store_LSE(
                    mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q
                )

        if const_expr(not seqlen.has_cu_seqlens_q):
            mO_cur = mO[None, None, head_idx, batch_idx]
        else:
            offset = (
                seqlen.offset_q
                if const_expr(not self.pack_gqa)
                else (0, seqlen.offset_q)
            )
            mO_cur = cute.domain_offset((offset, 0), mO[None, None, head_idx])
        # pyre-ignore[16]
        if const_expr(self.use_tma_O):
            cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            # pyre-ignore[23]
            store_O, _, _ = copy_utils.tma_get_copy_fn(
                # pyre-ignore[6]
                tma_atom_O,
                0,
                cute.make_layout(1),
                sO,
                gO,
                single_stage=True,
            )
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 4:
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
                )
                store_O()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads,
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOrO = cute.make_fragment_like(tOsO, self.dtype)
            cute.autovec_copy(tOsO, tOrO)
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(
                    mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
                )
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if (
                        t0OcO[0, rest_m, 0][0]
                        < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]
                    ):
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None],
                            pred=tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None,
                        )
            else:
                pack_gqa.store_O(
                    mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q
                )


class FlashAttentionForwardSm90(FlashAttentionForwardBase):
    arch = 90

    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim
            ),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype, self.tile_n
                ),
                self.dtype,
            )
        else:
            sP_layout_atom = None
        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM
            if self.mma_pv_is_rs
            else warpgroup.OperandSource.SMEM,
        )
        tiled_mma_pv_rs = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_rs

    def _get_shared_storage_cls(self):
        sQ_alignment = 128 if const_expr(self.use_tma_Q) else 1024
        sK_alignment = 128
        sV_alignment = 128
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], alignment
            ]
            for layout, alignment in zip(
                (self.sQ_layout, self.sK_layout, self.sV_layout),
                (sQ_alignment, sK_alignment, sV_alignment),
            )
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQV], 1024
        ]
        cosize_sP = (
            cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        )
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        mbar_ptr_QO_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr: mbar_ptr_QO_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr: mbar_ptr_QO_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        return (
            SharedStorageQKV
            if const_expr(not self.Q_in_regs)
            else SharedStorageSharedQV
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mSeqUsedQ: cute.Tensor | None = None,
        mSeqUsedK: cute.Tensor | None = None,
        mPageTable: cute.Tensor | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: cute.Tensor | None = None,
        blocksparse_tensors: BlockSparseTensors | None = None,
        aux_tensors: list | None = None,
        mAttnScale: cute.Tensor | None = None,
        mK_alt: cute.Tensor | None = None,
        mV_alt: cute.Tensor | None = None,
        mCuSeqlensK_alt: cute.Tensor | None = None,
    ):
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                )
            )
        )

        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mO = [
            cute.make_tensor(
                t.iterator, cute.make_layout(t.shape, stride=new_stride(t))
            )
            for t in (mQ, mK, mV, mO)
        ]
        QO_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mQ, mO = [utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        mK_alt_t = None
        mV_alt_t = None
        if const_expr(mK_alt is not None):
            mK_alt_t = cute.make_tensor(
                # pyre-ignore[16]
                mK_alt.iterator,
                # pyre-ignore[16]
                cute.make_layout(mK_alt.shape, stride=new_stride(mK_alt)),
            )
            mV_alt_t = cute.make_tensor(
                mV_alt.iterator,
                cute.make_layout(mV_alt.shape, stride=new_stride(mV_alt)),
            )
            mK_alt_t = utils.select(mK_alt_t, KV_layout_transpose)
            mV_alt_t = utils.select(mV_alt_t, KV_layout_transpose)
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = (
            # pyre-ignore[6]
            utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )

        tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_rs = self._get_tiled_mma()
        # pyre-ignore[16]
        self.num_mma_threads = tiled_mma_qk.size
        # pyre-ignore[16]
        self.num_threads_per_warp_group = 128
        # pyre-ignore[16]
        self.num_mma_warp_groups = (
            self.num_mma_threads // self.num_threads_per_warp_group
        )
        self.num_threads = self.num_threads_per_warp_group * (
            self.num_mma_warp_groups + 1
        )
        # pyre-ignore[16]
        self.num_producer_threads = 32
        # pyre-ignore[16]
        self.num_Q_load_threads = self.num_mma_threads
        # pyre-ignore[16]
        self.num_epilogue_threads = self.num_mma_threads
        # pyre-ignore[16]
        self.num_mma_regs = (
            256
            if self.num_mma_warp_groups == 1
            else (240 if self.num_mma_warp_groups == 2 else 160)
        )
        # pyre-ignore[16]
        self.num_producer_regs = (
            56
            if self.num_mma_warp_groups == 1
            else (24 if self.num_mma_warp_groups == 2 else 32)
        )
        # pyre-ignore[16]
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        # pyre-ignore[16]
        self.use_scheduler_barrier = (
            (self.num_mma_warp_groups >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_mma_warp_groups == 2)
        )
        # pyre-ignore[16]
        self.use_tma_Q = self.arch >= 90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        # pyre-ignore[16]
        self.use_tma_O = (
            self.arch >= 90
            and mCuSeqlensQ is None
            and mSeqUsedQ is None
            and not self.pack_gqa
        )
        self._setup_attributes()
        # pyre-ignore[16]
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_helpers.make_smem_layout(
                mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage
            )
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
                (mO, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        # pyre-ignore[16]
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_helpers.make_smem_layout(
                # pyre-ignore[16]
                mV.dtype,
                LayoutEnum.ROW_MAJOR,
                (self.tile_m, self.tile_n),
            )

        SharedStorage = self._get_shared_storage_cls()

        if const_expr(self.pack_gqa):
            shape_Q_packed = (
                (
                    self.qhead_per_kvhead,
                    # pyre-ignore[16]
                    mQ.shape[0],
                ),
                mQ.shape[1],
                mK.shape[2],
                *mQ.shape[3:],
            )
            stride_Q_packed = (
                (mQ.stride[2], mQ.stride[0]),
                mQ.stride[1],
                mQ.stride[2] * self.qhead_per_kvhead,
                *mQ.stride[3:],
            )
            mQ = cute.make_tensor(
                mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
            )
            shape_O_packed = (
                (self.qhead_per_kvhead, mO.shape[0]),
                mK.shape[1],
                mK.shape[2],
                *mO.shape[3:],
            )
            stride_O_packed = (
                (mO.stride[2], mO.stride[0]),
                mO.stride[1],
                mO.stride[2] * self.qhead_per_kvhead,
                *mO.stride[3:],
            )
            mO = cute.make_tensor(
                mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
            )
            if const_expr(mLSE is not None):
                shape_LSE_packed = (
                    # pyre-ignore[16]
                    (self.qhead_per_kvhead, mLSE.shape[0]),
                    mK.shape[2],
                    # pyre-ignore[16]
                    *mLSE.shape[2:],
                )
                stride_LSE_packed = (
                    # pyre-ignore[16]
                    (mLSE.stride[1], mLSE.stride[0]),
                    mLSE.stride[1] * self.qhead_per_kvhead,
                    *mLSE.stride[2:],
                )
                mLSE = cute.make_tensor(
                    # pyre-ignore[16]
                    mLSE.iterator,
                    cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed),
                )

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        # pyre-ignore[16]
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_Q,
                mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),
            )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
            1,
        )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_O,
                mO,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),
            )
        tma_atom_K_alt = None
        tma_tensor_K_alt = None
        tma_atom_V_alt = None
        tma_tensor_V_alt = None
        if const_expr(mK_alt is not None):
            tma_atom_K_alt, tma_tensor_K_alt = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK_alt_t,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_V_alt, tma_tensor_V_alt = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV_alt_t,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,
            )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            # pyre-ignore[6]
            1,  # num_splits
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            # pyre-ignore[6]
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            # pyre-ignore[6]
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            # pyre-ignore[6]
            element_size=self.dtype.width // 8,
            # pyre-ignore[6]
            is_persistent=False,
            # pyre-ignore[6]
            lpt=self.is_causal and not self.is_local,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        # pyre-ignore[6]
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            if const_expr(not self.use_silu):
                # pyre-ignore[9]
                softmax_scale = None
        else:
            softmax_scale_log2 = LOG2_E
            softmax_scale = softmax_scale
        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = (
                cute.size(mK.shape[0])
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            )
            seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisor(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            # pyre-ignore[16]
            self.gmem_tiled_copy_Q,
            # pyre-ignore[16]
            self.gmem_tiled_copy_K,
            # pyre-ignore[16]
            self.gmem_tiled_copy_V,
            # pyre-ignore[16]
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_pv_rs,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_tensors,
            fastdiv_mods,
            mAttnScale,
            tma_tensor_K_alt if const_expr(mK_alt is not None) else None,
            tma_atom_K_alt,
            tma_tensor_V_alt if const_expr(mK_alt is not None) else None,
            tma_atom_V_alt,
            mCuSeqlensK_alt,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        mSeqUsedK: cute.Tensor | None,
        tma_atom_Q: cute.CopyAtom | None,
        tma_atom_K: cute.CopyAtom | None,
        tma_atom_V: cute.CopyAtom | None,
        tma_atom_O: cute.CopyAtom | None,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Int32 | None,
        window_size_right: Int32 | None,
        learnable_sink: cute.Tensor | None,
        blocksparse_tensors: BlockSparseTensors | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_tensors=None,
        fastdiv_mods=None,
        mAttnScale: cute.Tensor | None = None,
        mK_alt: cute.Tensor | None = None,
        tma_atom_K_alt: cute.CopyAtom | None = None,
        mV_alt: cute.Tensor | None = None,
        tma_atom_V_alt: cute.CopyAtom | None = None,
        mCuSeqlensK_alt: cute.Tensor | None = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        mbar_ptr_Q = storage.mbar_ptr.data_ptr()
        if warp_idx == 1:
            # pyre-ignore[16]
            if const_expr(not self.use_tma_Q):
                # pyre-ignore[16]
                cute.arch.mbarrier_init(mbar_ptr_Q, self.num_Q_load_threads)
        pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            # pyre-ignore[16]
            self.num_mma_threads // self.num_threads_per_warp_group,
        )
        pipeline_k = fa4_pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            # pyre-ignore[16]
            tx_count=self.tma_copy_bytes["K"],
            # pyre-ignore[6]
            init_wait=False,
        )
        pipeline_v = fa4_pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["V"],
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        sVt = utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            # pyre-ignore[16]
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sO = storage.sQ.get_tensor(
            sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype
        )

        block_info = BlockInfo(
            # pyre-ignore[6]
            self.tile_m,
            # pyre-ignore[6]
            self.tile_n,
            # pyre-ignore[6]
            self.is_causal,
            # pyre-ignore[6]
            self.is_local,
            # pyre-ignore[6]
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            # pyre-ignore[6]
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            # pyre-ignore[16]
            seqlen_q_static=mQ.shape[0]
            if const_expr(not self.pack_gqa)
            else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        # pyre-ignore[16]
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        SeqlenInfoCls_alt = None
        block_info_alt = None
        AttentionMaskCls_alt = None
        if const_expr(mK_alt is not None):
            SeqlenInfoCls_alt = partial(
                SeqlenInfoQK.create,
                seqlen_q_static=mQ.shape[0]
                if const_expr(not self.pack_gqa)
                else mQ.shape[0][1],
                # pyre-ignore[16]
                seqlen_k_static=mK_alt.shape[0],
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK_alt,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=None,
            )
            block_info_alt = BlockInfo(
                # pyre-ignore[6]
                self.tile_m,
                # pyre-ignore[6]
                self.tile_n,
                # pyre-ignore[6]
                True,  # is_causal
                # pyre-ignore[6]
                False,  # is_local
                # pyre-ignore[6]
                False,
                None,
                None,
                # pyre-ignore[6]
                qhead_per_kvhead_packgqa=self.qhead_per_kvhead
                if const_expr(self.pack_gqa)
                else 1,
            )
            AttentionMaskCls_alt = partial(
                AttentionMask,
                self.tile_m,
                self.tile_n,
                window_size_left=None,
                window_size_right=None,
                qhead_per_kvhead_packgqa=self.qhead_per_kvhead
                if const_expr(self.pack_gqa)
                else 1,
            )

        if warp_idx < 4:  # Producer
            # pyre-ignore[16]
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                blocksparse_tensors,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mK_alt,
                tma_atom_K_alt,
                mV_alt,
                tma_atom_V_alt,
                SeqlenInfoCls_alt,
                block_info_alt,
            )
        else:  # Consumer
            # pyre-ignore[16]
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                tiled_mma_pv_rs,
                mQ,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                gmem_tiled_copy_Q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                blocksparse_tensors,
                aux_tensors,
                fastdiv_mods,
                mAttnScale,
                SeqlenInfoCls_alt,
                AttentionMaskCls_alt,
                block_info_alt,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        blocksparse_tensors: BlockSparseTensors | None,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mK_alt: cute.Tensor | None = None,
        tma_atom_K_alt: cute.CopyAtom | None = None,
        mV_alt: cute.Tensor | None = None,
        tma_atom_V_alt: cute.CopyAtom | None = None,
        SeqlenInfoCls_alt: Callable | None = None,
        block_info_alt: BlockInfo | None = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            q_producer_phase = Int32(1)
            kv_producer_state = fa4_pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead
                    if const_expr(not self.pack_gqa)
                    else head_idx
                )
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                # pyre-ignore[16]
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(
                        mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0)
                    )
                    # pyre-ignore[23]
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )
                # pyre-ignore[23]
                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK
                )
                load_K = copy_utils.tma_producer_copy_fn(load_K, pipeline_k)
                # pyre-ignore[23]
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV
                )
                load_V = copy_utils.tma_producer_copy_fn(load_V, pipeline_v)

                # pyre-ignore[16]
                if const_expr(not self.use_block_sparsity):
                    n_block_min, n_block_max = block_info.get_n_block_min_max(
                        seqlen, m_block
                    )
                    n_block = n_block_max - 1
                    pipeline_k.producer_acquire(
                        kv_producer_state,
                        # pyre-ignore[16]
                        extra_tx_count=self.tma_copy_bytes["Q"]
                        if const_expr(self.use_tma_Q)
                        else 0,
                    )
                    if const_expr(self.use_tma_Q):
                        # pyre-ignore[61]
                        load_Q(
                            tma_bar_ptr=pipeline_k.producer_get_barrier(
                                kv_producer_state
                            )
                        )
                    load_K(src_idx=n_block, producer_state=kv_producer_state)

                    if const_expr(not self.intra_wg_overlap):
                        pipeline_v.producer_acquire(kv_producer_state)
                        load_V(src_idx=n_block, producer_state=kv_producer_state)
                        kv_producer_state.advance()
                        for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - i - 1
                            pipeline_k.producer_acquire(kv_producer_state)
                            load_K(src_idx=n_block, producer_state=kv_producer_state)
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(src_idx=n_block, producer_state=kv_producer_state)
                            kv_producer_state.advance()
                    else:
                        for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                            n_block_prev = n_block_max - i - 1
                            n_block = n_block_prev - 1
                            kv_producer_state_prev = kv_producer_state.clone()
                            kv_producer_state.advance()
                            pipeline_k.producer_acquire(kv_producer_state)
                            load_K(src_idx=n_block, producer_state=kv_producer_state)
                            pipeline_v.producer_acquire(kv_producer_state_prev)
                            load_V(
                                src_idx=n_block_prev,
                                producer_state=kv_producer_state_prev,
                            )
                        n_block = n_block_min
                        pipeline_v.producer_acquire(kv_producer_state)
                        load_V(src_idx=n_block, producer_state=kv_producer_state)
                        kv_producer_state.advance()
                    if const_expr(mK_alt is not None):
                        # pyre-ignore[29]
                        seqlen_alt = SeqlenInfoCls_alt(batch_idx)
                        mK_alt_cur = seqlen_alt.offset_batch_K(
                            mK_alt, batch_idx, dim=3
                        )[None, None, head_idx_kv]
                        mV_alt_cur = seqlen_alt.offset_batch_K(
                            mV_alt, batch_idx, dim=3
                        )[None, None, head_idx_kv]
                        gK_alt = cute.local_tile(
                            mK_alt_cur, (self.tile_n, self.tile_hdim), (None, 0)
                        )
                        gV_alt = cute.local_tile(
                            mV_alt_cur, (self.tile_n, self.tile_hdimv), (None, 0)
                        )
                        # pyre-ignore[6, 23]
                        load_K_alt, _, _ = copy_utils.tma_get_copy_fn(
                            # pyre-ignore[6]
                            tma_atom_K_alt,
                            0,
                            cute.make_layout(1),
                            gK_alt,
                            sK,
                        )
                        load_K_alt = copy_utils.tma_producer_copy_fn(
                            load_K_alt, pipeline_k
                        )
                        # pyre-ignore[6, 23]
                        load_V_alt, _, _ = copy_utils.tma_get_copy_fn(
                            # pyre-ignore[6]
                            tma_atom_V_alt,
                            0,
                            cute.make_layout(1),
                            gV_alt,
                            sV,
                        )
                        load_V_alt = copy_utils.tma_producer_copy_fn(
                            load_V_alt, pipeline_v
                        )

                        n_block_min_alt, n_block_max_alt = (
                            # pyre-ignore[16]
                            block_info_alt.get_n_block_min_max(seqlen_alt, m_block)
                        )
                        n_block = n_block_max_alt - 1
                        pipeline_k.producer_acquire(kv_producer_state)
                        load_K_alt(src_idx=n_block, producer_state=kv_producer_state)

                        if const_expr(not self.intra_wg_overlap):
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V_alt(
                                src_idx=n_block, producer_state=kv_producer_state
                            )
                            kv_producer_state.advance()
                            for i in cutlass.range(
                                n_block_max_alt - 1 - n_block_min_alt, unroll=1
                            ):
                                n_block = n_block_max_alt - 1 - i - 1
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K_alt(
                                    src_idx=n_block,
                                    producer_state=kv_producer_state,
                                )
                                pipeline_v.producer_acquire(kv_producer_state)
                                load_V_alt(
                                    src_idx=n_block,
                                    producer_state=kv_producer_state,
                                )
                                kv_producer_state.advance()
                        else:
                            for i in cutlass.range(
                                n_block_max_alt - 1 - n_block_min_alt, unroll=1
                            ):
                                n_block_prev = n_block_max_alt - i - 1
                                n_block = n_block_prev - 1
                                kv_producer_state_prev = kv_producer_state.clone()
                                kv_producer_state.advance()
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K_alt(
                                    src_idx=n_block,
                                    producer_state=kv_producer_state,
                                )
                                pipeline_v.producer_acquire(kv_producer_state_prev)
                                load_V_alt(
                                    src_idx=n_block_prev,
                                    producer_state=kv_producer_state_prev,
                                )
                            n_block = n_block_min_alt
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V_alt(
                                src_idx=n_block, producer_state=kv_producer_state
                            )
                            kv_producer_state.advance()
                else:
                    kv_producer_state = produce_block_sparse_loads(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        m_block,
                        kv_producer_state,
                        # pyre-ignore[61]
                        load_Q,
                        load_K,
                        load_V,
                        pipeline_k,
                        pipeline_v,
                        self.use_tma_Q,
                        self.tma_copy_bytes["Q"],
                        self.intra_wg_overlap,
                    )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: cute.Tensor | None,
        sO: cute.Tensor,
        learnable_sink: cute.Tensor | None,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom | None,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors | None,
        aux_tensors: list | None,
        fastdiv_mods=None,
        mAttnScale: cute.Tensor | None = None,
        SeqlenInfoCls_alt: Callable | None = None,
        AttentionMaskCls_alt: Callable | None = None,
        block_info_alt: BlockInfo | None = None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(
            # pyre-ignore[16]
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            # pyre-ignore[16]
            self.num_mma_warp_groups,
            stride=self.num_threads_per_warp_group,
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        tSrQ = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sK))
        if const_expr(self.mma_pv_is_rs):
            acc_S_shape = tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
            tOrP = cute.make_fragment(
                utils.convert_layout_acc_frgA(cute.make_layout(acc_S_shape)), self.dtype
            )
        else:
            tOrP = tiled_mma_pv.make_fragment_A(wg_mma_pv.partition_A(sP))
        tOrVt = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sVt))

        # pyre-ignore[6]
        smem_copy_atom_P = utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_P = cute.make_tiled_copy_C(
            smem_copy_atom_P, tiled_mma_qk
        ).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None

        self.mma_init()

        acc_shape_O = tiled_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_fragment(acc_shape_O, Float32)
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        mma_qk_fn = partial(
            sm90_helpers.gemm_zero_init,
            tiled_mma_qk,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrK,
        )
        mma_pv_fn = partial(sm90_helpers.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap
            if const_expr(self.intra_wg_overlap)
            else self.mma_one_n_block,
            mma_qk_fn=mma_qk_fn,
            tiled_mma_pv_rs=tiled_mma_pv_rs,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
        )

        q_consumer_phase = Int32(0)
        kv_consumer_state = fa4_pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        if const_expr(self.use_silu):
            # SiLU mainloop — no softmax, no LSE
            while work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)

                # pyre-ignore[16]
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(
                        # pyre-ignore[6]
                        self.tile_m,
                        # pyre-ignore[6]
                        self.tile_hdim,
                        self.check_hdim_oob,
                        # pyre-ignore[6]
                        self.qhead_per_kvhead,
                    )
                    mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    pack_gqa.load_Q(
                        mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                    )
                    cute.arch.cp_async_mbarrier_arrive_noinc(mbar_ptr_Q)

                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block
                )
                if const_expr(not self.use_tma_Q):
                    cute.arch.mbarrier_wait(mbar_ptr_Q, phase=q_consumer_phase)
                q_consumer_phase ^= 1
                O_should_accumulate = False

                # SiLU mainloop

                cached_combined = self.precompute_silu_combined(
                    softmax_scale, thr_mma_qk, m_block, mAttnScale
                )

                if const_expr(self.intra_wg_overlap):
                    # Overlap pattern: QK[N+1] || PV[N]
                    silu_overlap = partial(
                        self.mma_one_n_block_intrawg_overlap_silu,
                        m_block=m_block,
                        mma_qk_fn=mma_qk_fn,
                        tiled_mma_pv_rs=tiled_mma_pv_rs,
                        pipeline_k=pipeline_k,
                        pipeline_v=pipeline_v,
                        acc_O=acc_O,
                        tOrP=tOrP,
                        smem_copy_params=smem_copy_params,
                        thr_mma_qk=thr_mma_qk,
                        softmax_scale=softmax_scale,
                        seqlen=seqlen,
                        mAttnScale=mAttnScale,
                        window_size_left=block_info.window_size_left,
                        cached_combined=cached_combined,
                    )
                    process_last_half_silu = partial(
                        self.last_half_block_overlap,
                        pipeline_v=pipeline_v,
                        mma_pv_fn=mma_pv_fn,
                    )

                    # First half: QK + SiLU -> P (no PV yet)
                    kv_consumer_state = self.first_half_block_silu(
                        n_block=n_block_max - 1,
                        m_block=m_block,
                        mma_qk_fn=mma_qk_fn,
                        kv_consumer_state=kv_consumer_state,
                        pipeline_k=pipeline_k,
                        tOrP=tOrP,
                        smem_copy_params=smem_copy_params,
                        thr_mma_qk=thr_mma_qk,
                        softmax_scale=softmax_scale,
                        seqlen=seqlen,
                        mAttnScale=mAttnScale,
                        apply_causal_mask=self.is_causal or self.is_local,
                        window_size_left=block_info.window_size_left,
                        cached_combined=cached_combined,
                    )
                    n_block_max -= 1

                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = (
                            block_info.get_n_block_min_causal_local_mask(
                                seqlen, m_block, n_block_min
                            )
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            kv_consumer_state = silu_overlap(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                mma_pv_fn=partial(
                                    mma_pv_fn, zero_init=not O_should_accumulate
                                ),
                                apply_causal_mask=True,
                                mask_seqlen=False,
                            )
                            O_should_accumulate = True
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_causal_local_mask
                        )

                    # Unmasked iterations (with overlap)
                    n_block_min_before_local_mask = (
                        block_info.get_n_block_min_before_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        kv_consumer_state = silu_overlap(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            mma_pv_fn=partial(
                                mma_pv_fn, zero_init=not O_should_accumulate
                            ),
                            apply_causal_mask=False,
                            mask_seqlen=False,
                        )
                        O_should_accumulate = True

                    # Local masking on the left (with overlap)
                    if const_expr(
                        self.is_local and block_info.window_size_left is not None
                    ):
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_before_local_mask
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min, unroll=1
                        ):
                            kv_consumer_state = silu_overlap(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                mma_pv_fn=partial(
                                    mma_pv_fn, zero_init=not O_should_accumulate
                                ),
                                apply_causal_mask=True,
                                mask_seqlen=False,
                            )
                            O_should_accumulate = True

                    # Last half: final PV
                    kv_consumer_state = process_last_half_silu(
                        kv_consumer_state=kv_consumer_state,
                        zero_init=not O_should_accumulate,
                    )
                    O_should_accumulate = True
                else:
                    # Non-overlap path
                    silu_common = partial(
                        self.mma_one_n_block_silu,
                        m_block=m_block,
                        mma_qk_fn=mma_qk_fn,
                        tiled_mma_pv_rs=tiled_mma_pv_rs,
                        pipeline_k=pipeline_k,
                        pipeline_v=pipeline_v,
                        acc_O=acc_O,
                        tOrP=tOrP,
                        smem_copy_params=smem_copy_params,
                        thr_mma_qk=thr_mma_qk,
                        softmax_scale=softmax_scale,
                        seqlen=seqlen,
                        mAttnScale=mAttnScale,
                        window_size_left=block_info.window_size_left,
                        cached_combined=cached_combined,
                    )

                    # First iteration with seqlen masking
                    self.warp_scheduler_barrier_sync()
                    kv_consumer_state = silu_common(
                        kv_consumer_state,
                        n_block=n_block_max - 1,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                        apply_causal_mask=self.is_causal or self.is_local,
                        mask_seqlen=True,
                    )
                    O_should_accumulate = True
                    n_block_max -= 1

                    # Causal/local masking iterations
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = (
                            block_info.get_n_block_min_causal_local_mask(
                                seqlen, m_block, n_block_min
                            )
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            kv_consumer_state = silu_common(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                mma_pv_fn=partial(
                                    mma_pv_fn, zero_init=not O_should_accumulate
                                ),
                                apply_causal_mask=True,
                                mask_seqlen=False,
                            )
                            O_should_accumulate = True
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_causal_local_mask
                        )

                    # Unmasked iterations
                    n_block_min_before_local_mask = (
                        block_info.get_n_block_min_before_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        kv_consumer_state = silu_common(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            mma_pv_fn=partial(
                                mma_pv_fn, zero_init=not O_should_accumulate
                            ),
                            apply_causal_mask=False,
                            mask_seqlen=False,
                        )
                        O_should_accumulate = True

                    # Local masking on the left
                    if const_expr(
                        self.is_local and block_info.window_size_left is not None
                    ):
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_before_local_mask
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min, unroll=1
                        ):
                            kv_consumer_state = silu_common(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                mma_pv_fn=partial(
                                    mma_pv_fn, zero_init=not O_should_accumulate
                                ),
                                apply_causal_mask=True,
                                mask_seqlen=False,
                            )
                            O_should_accumulate = True

                    self.warp_scheduler_barrier_arrive()

                if const_expr(SeqlenInfoCls_alt is not None):
                    # pyre-ignore[29]
                    seqlen_alt = SeqlenInfoCls_alt(batch_idx)
                    n_block_min_alt, n_block_max_alt = (
                        # pyre-ignore[16]
                        block_info_alt.get_n_block_min_max(seqlen_alt, m_block)
                    )
                    cached_combined_alt = self.precompute_silu_combined(
                        softmax_scale, thr_mma_qk, m_block, mAttnScale
                    )
                    if const_expr(self.intra_wg_overlap):
                        silu_overlap_alt = partial(
                            self.mma_one_n_block_intrawg_overlap_silu,
                            m_block=m_block,
                            mma_qk_fn=mma_qk_fn,
                            tiled_mma_pv_rs=tiled_mma_pv_rs,
                            pipeline_k=pipeline_k,
                            pipeline_v=pipeline_v,
                            acc_O=acc_O,
                            tOrP=tOrP,
                            smem_copy_params=smem_copy_params,
                            thr_mma_qk=thr_mma_qk,
                            softmax_scale=softmax_scale,
                            seqlen=seqlen_alt,
                            mAttnScale=mAttnScale,
                            window_size_left=None,
                            cached_combined=cached_combined_alt,
                        )
                        process_last_half_silu_alt = partial(
                            self.last_half_block_overlap,
                            pipeline_v=pipeline_v,
                            mma_pv_fn=mma_pv_fn,
                        )

                        kv_consumer_state = self.first_half_block_silu(
                            n_block=n_block_max_alt - 1,
                            m_block=m_block,
                            mma_qk_fn=mma_qk_fn,
                            kv_consumer_state=kv_consumer_state,
                            pipeline_k=pipeline_k,
                            tOrP=tOrP,
                            smem_copy_params=smem_copy_params,
                            thr_mma_qk=thr_mma_qk,
                            softmax_scale=softmax_scale,
                            seqlen=seqlen_alt,
                            mAttnScale=mAttnScale,
                            apply_causal_mask=True,
                            window_size_left=None,
                            cached_combined=cached_combined_alt,
                        )
                        n_block_max_alt -= 1

                        # Causal masking iters
                        n_block_min_causal_alt = (
                            # pyre-ignore[16]
                            block_info_alt.get_n_block_min_causal_local_mask(
                                seqlen_alt, m_block, n_block_min_alt
                            )
                        )
                        for n_tile in cutlass.range(
                            n_block_max_alt - n_block_min_causal_alt, unroll=1
                        ):
                            kv_consumer_state = silu_overlap_alt(
                                kv_consumer_state,
                                n_block=n_block_max_alt - 1 - n_tile,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                                apply_causal_mask=True,
                                mask_seqlen=False,
                            )
                        n_block_max_alt = cutlass.min(
                            n_block_max_alt, n_block_min_causal_alt
                        )

                        # Unmasked iters
                        for n_tile in cutlass.range(
                            n_block_max_alt - n_block_min_alt, unroll=1
                        ):
                            kv_consumer_state = silu_overlap_alt(
                                kv_consumer_state,
                                n_block=n_block_max_alt - 1 - n_tile,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                                apply_causal_mask=False,
                                mask_seqlen=False,
                            )

                        kv_consumer_state = process_last_half_silu_alt(
                            kv_consumer_state=kv_consumer_state,
                            zero_init=False,
                        )
                    else:
                        silu_common_alt = partial(
                            self.mma_one_n_block_silu,
                            m_block=m_block,
                            mma_qk_fn=mma_qk_fn,
                            tiled_mma_pv_rs=tiled_mma_pv_rs,
                            pipeline_k=pipeline_k,
                            pipeline_v=pipeline_v,
                            acc_O=acc_O,
                            tOrP=tOrP,
                            smem_copy_params=smem_copy_params,
                            thr_mma_qk=thr_mma_qk,
                            softmax_scale=softmax_scale,
                            seqlen=seqlen_alt,
                            mAttnScale=mAttnScale,
                            window_size_left=None,
                            cached_combined=cached_combined_alt,
                        )
                        self.warp_scheduler_barrier_sync()
                        kv_consumer_state = silu_common_alt(
                            kv_consumer_state,
                            n_block=n_block_max_alt - 1,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                            apply_causal_mask=True,
                            mask_seqlen=True,
                        )
                        n_block_max_alt -= 1
                        n_block_min_causal_alt = (
                            block_info_alt.get_n_block_min_causal_local_mask(
                                seqlen_alt, m_block, n_block_min_alt
                            )
                        )
                        for n_tile in cutlass.range(
                            n_block_max_alt - n_block_min_causal_alt, unroll=1
                        ):
                            kv_consumer_state = silu_common_alt(
                                kv_consumer_state,
                                n_block=n_block_max_alt - 1 - n_tile,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                                apply_causal_mask=True,
                                mask_seqlen=False,
                            )
                        n_block_max_alt = cutlass.min(
                            n_block_max_alt, n_block_min_causal_alt
                        )
                        for n_tile in cutlass.range(
                            n_block_max_alt - n_block_min_alt, unroll=1
                        ):
                            kv_consumer_state = silu_common_alt(
                                kv_consumer_state,
                                n_block=n_block_max_alt - 1 - n_tile,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                                apply_causal_mask=False,
                                mask_seqlen=False,
                            )
                        self.warp_scheduler_barrier_arrive()

                # Epilogue — no LSE for SiLU
                self.epilogue(
                    acc_O,
                    None,
                    mO,
                    None,
                    sO,
                    seqlen,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    tiled_mma_pv,
                    tidx,
                    m_block,
                    head_idx,
                    batch_idx,
                )

                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
        else:
            # Standard softmax mainloop
            softmax = Softmax.create(
                softmax_scale_log2,
                num_rows=acc_O.shape[0][0] * acc_O.shape[1],
                softmax_scale=softmax_scale,
            )

            process_first_half_block = partial(
                self.first_half_block_overlap,
                mma_qk_fn=mma_qk_fn,
                pipeline_k=pipeline_k,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                softmax=softmax,
            )
            process_last_half_block = partial(
                self.last_half_block_overlap,
                pipeline_v=pipeline_v,
                mma_pv_fn=mma_pv_fn,
            )
            while work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)

                recompute_fastdiv_mods_q = cutlass.const_expr(
                    aux_tensors is not None
                    and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
                )
                recompute_fastdiv_mods_k = cutlass.const_expr(
                    aux_tensors is not None
                    and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
                )
                if cutlass.const_expr(fastdiv_mods is not None):
                    seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                    fastdiv_mods = (
                        seqlen_q_divmod
                        if not recompute_fastdiv_mods_q
                        else FastDivmodDivisor(seqlen.seqlen_q),
                        seqlen_k_divmod
                        if not recompute_fastdiv_mods_k
                        else FastDivmodDivisor(seqlen.seqlen_k),
                    )

                mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
                mask_fn = partial(
                    mask.apply_mask,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=m_block,
                    thr_mma=thr_mma_qk,
                    mask_causal=self.is_causal,
                    mask_local=self.is_local,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods,
                )
                score_mod_fn = None
                if const_expr(self.score_mod is not None):
                    score_mod_fn = partial(
                        self.apply_score_mod,
                        thr_mma_qk,
                        batch_idx,
                        head_idx,
                        m_block,
                        softmax_scale=softmax_scale,
                        aux_tensors=aux_tensors,
                        fastdiv_mods=fastdiv_mods,
                    )
                mma_one_n_block = partial(
                    mma_one_n_block_all,
                    softmax=softmax,
                    score_mod_fn=score_mod_fn,
                )
                # Load Q if not TMA_Q
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(
                        # pyre-ignore[6]
                        self.tile_m,
                        # pyre-ignore[6]
                        self.tile_hdim,
                        self.check_hdim_oob,
                        # pyre-ignore[6]
                        self.qhead_per_kvhead,
                    )
                    mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    pack_gqa.load_Q(
                        mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                    )
                    cute.arch.cp_async_mbarrier_arrive_noinc(mbar_ptr_Q)

                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block
                )
                if const_expr(not self.use_tma_Q):
                    cute.arch.mbarrier_wait(mbar_ptr_Q, phase=q_consumer_phase)
                q_consumer_phase ^= 1
                O_should_accumulate = False

                # MAINLOOP
                # pyre-ignore[16]
                if const_expr(not self.use_block_sparsity):
                    # First iteration with seqlen masking
                    if const_expr(self.intra_wg_overlap):
                        kv_consumer_state = process_first_half_block(
                            n_block=n_block_max - 1,
                            seqlen=seqlen,
                            kv_consumer_state=kv_consumer_state,
                            mask_fn=partial(mask_fn, mask_mod=self.mask_mod),
                            score_mod_fn=score_mod_fn,
                            is_first_block=True,
                        )
                    else:
                        self.warp_scheduler_barrier_sync()
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1,
                            seqlen=seqlen,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                            is_first_n_block=True,
                            mask_fn=partial(
                                mask_fn, mask_mod=self.mask_mod, mask_seqlen=True
                            ),
                        )
                        O_should_accumulate = True
                    n_block_max -= 1
                    # Causal masking iterations
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = (
                            block_info.get_n_block_min_causal_local_mask(
                                seqlen, m_block, n_block_min
                            )
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            kv_consumer_state = mma_one_n_block(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                seqlen=seqlen,
                                mma_pv_fn=partial(
                                    mma_pv_fn, zero_init=not O_should_accumulate
                                ),
                                mask_fn=partial(
                                    mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                                ),
                            )
                            O_should_accumulate = True
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_causal_local_mask
                        )
                    # Unmasked iterations
                    n_block_min_before_local_mask = (
                        block_info.get_n_block_min_before_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=partial(
                                mma_pv_fn, zero_init=not O_should_accumulate
                            ),
                            mask_fn=partial(
                                mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                            ),
                        )
                        O_should_accumulate = True
                    # Local masking on the left
                    if const_expr(
                        self.is_local and block_info.window_size_left is not None
                    ):
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_before_local_mask
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min, unroll=1
                        ):
                            kv_consumer_state = mma_one_n_block(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                seqlen=seqlen,
                                mma_pv_fn=partial(
                                    mma_pv_fn, zero_init=not O_should_accumulate
                                ),
                                mask_fn=partial(
                                    mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                                ),
                            )
                            O_should_accumulate = True
                    # Last half iteration
                    if const_expr(self.intra_wg_overlap):
                        kv_consumer_state = process_last_half_block(
                            kv_consumer_state=kv_consumer_state,
                            zero_init=not O_should_accumulate,
                        )
                        O_should_accumulate = True
                    else:
                        self.warp_scheduler_barrier_arrive()
                else:
                    # Block sparsity
                    kv_consumer_state, O_should_accumulate, processed_any = (
                        consume_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            kv_consumer_state,
                            mma_pv_fn,
                            mma_one_n_block,
                            process_first_half_block,
                            process_last_half_block,
                            mask_fn,
                            score_mod_fn,
                            O_should_accumulate,
                            self.mask_mod,
                            fastdiv_mods,
                            self.intra_wg_overlap,
                            self.warp_scheduler_barrier_sync,
                            self.warp_scheduler_barrier_arrive,
                        )
                    )
                    if not processed_any:
                        softmax.reset()
                        acc_O.fill(0.0)

                sink_val = None
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        # pyre-ignore[16]
                        sink_val = Float32(learnable_sink[head_idx])
                    else:
                        sink_val = cute.make_fragment_like(softmax.row_max, Float32)
                        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                        tScS_mn = utils.make_acc_tensor_mn_view(
                            thr_mma_qk.partition_C(cS)
                        )
                        for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                            row = m_block * self.tile_m + tScS_mn[r][0]
                            q_head_idx = (
                                row % self.qhead_per_kvhead
                                + head_idx * self.qhead_per_kvhead
                            )
                            sink_val[r] = Float32(learnable_sink[q_head_idx])

                row_scale = softmax.finalize(sink_val=sink_val)
                softmax.rescale_O(acc_O, row_scale)

                # Epilogue
                self.epilogue(
                    acc_O,
                    softmax.row_sum,
                    mO,
                    mLSE,
                    sO,
                    seqlen,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    tiled_mma_pv,
                    tidx,
                    m_block,
                    head_idx,
                    batch_idx,
                )

                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        # pyre-ignore[9]
        mask_fn: Callable = None,
        score_mod_fn: Callable | None = None,
        is_first_block: bool = False,
    ):
        pipeline_k.consumer_wait(
            kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state)
        )
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        if const_expr(score_mod_fn is not None):
            # pyre-ignore[29]
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)

        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

        softmax.online_softmax(acc_S, is_first=is_first_block)

        tOrP_acc = cute.make_tensor(
            acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout)
        )
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_fragment_like(tOrP_acc, self.dtype)
        )
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))

        if const_expr(not self.mma_pv_is_rs):
            # pyre-ignore[16]
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            # pyre-ignore[16]
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.sync_warp()

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
    ):
        pipeline_v.consumer_wait(
            kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state)
        )
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def first_half_block_silu(
        self,
        n_block: Int32,
        m_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        thr_mma_qk,
        softmax_scale: Float32,
        seqlen: SeqlenInfoQK,
        mAttnScale: cute.Tensor | None,
        apply_causal_mask: cutlass.Constexpr[bool],
        window_size_left: Int32 | None,
        cached_combined=None,
    ):
        """First half of overlap: QK + SiLU + convert P (no PV yet)."""
        pipeline_k.consumer_wait(
            kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state)
        )
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        self.silu_activate(
            acc_S,
            softmax_scale,
            thr_mma_qk,
            m_block,
            n_block,
            seqlen.seqlen_q,
            seqlen.seqlen_k,
            mAttnScale,
            apply_causal_mask,
            True,
            window_size_left,
            cached_combined=cached_combined,
        )

        tOrP_acc = cute.make_tensor(
            acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout)
        )
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_fragment_like(tOrP_acc, self.dtype)
        )
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))

        if const_expr(not self.mma_pv_is_rs):
            # pyre-ignore[16]
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            # pyre-ignore[16]
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            cute.arch.sync_warp()

        return kv_consumer_state

    @cute.jit
    def mma_one_n_block_intrawg_overlap_silu(
        self,
        smem_pipe_read,
        n_block: Int32,
        m_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        tiled_mma_pv_rs: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        thr_mma_qk,
        softmax_scale: Float32,
        seqlen: SeqlenInfoQK,
        mAttnScale: cute.Tensor | None,
        apply_causal_mask: cutlass.Constexpr[bool],
        mask_seqlen: cutlass.Constexpr[bool],
        window_size_left: Int32 | None,
        cached_combined=None,
    ):
        """Middle block overlap: QK(current) + PV(previous) concurrent."""
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        pipeline_v.consumer_wait(
            smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v)
        )
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        self.silu_activate(
            acc_S,
            softmax_scale,
            thr_mma_qk,
            m_block,
            n_block,
            seqlen.seqlen_q,
            seqlen.seqlen_k,
            mAttnScale,
            apply_causal_mask,
            mask_seqlen,
            window_size_left,
            cached_combined=cached_combined,
        )

        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = cute.make_tensor(
            acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout)
        )
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_fragment_like(tOrP_acc, self.dtype)
        )
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            # pyre-ignore[16]
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            # pyre-ignore[16]
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        # No rescale_O for SiLU
        if const_expr(not self.mma_pv_is_rs):
            cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
            cute.arch.sync_warp()
        return smem_pipe_read

    @cute.jit
    def precompute_silu_combined(
        self,
        softmax_scale: Float32,
        thr_mma_qk,
        m_block: Int32,
        mAttnScale: cute.Tensor | None,
    ):
        if const_expr(mAttnScale is not None):
            cS_scale = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS_scale = thr_mma_qk.partition_C(cS_scale)
            # pyre-ignore[16]
            attn_scale_len = cute.size(mAttnScale.shape[0])
            acc_S_shape = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
            num_pairs = const_expr(cute.size(acc_S_shape) // 2)
            combined_cache = cute.make_fragment(cute.make_layout(num_pairs), Float32)
            for i in cutlass.range_constexpr(0, num_pairs):
                row_idx = tScS_scale[i * 2][0] + m_block * self.tile_m
                row_idx = cutlass.min(row_idx, attn_scale_len - Int32(1))
                combined_cache[i] = mAttnScale[row_idx]  # pyre-ignore[16]
            return combined_cache
        return None

    @cute.jit
    def silu_activate(
        self,
        acc_S: cute.Tensor,
        softmax_scale: Float32,
        thr_mma_qk,
        m_block: Int32,
        n_block: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        mAttnScale: cute.Tensor | None,
        apply_causal_mask: cutlass.Constexpr[bool],
        mask_seqlen: cutlass.Constexpr[bool],
        window_size_left: Int32 | None,
        cached_combined=None,
    ):
        """SiLU activation for SM90. silu(x) = x/2 * (1 + tanh(x/2))"""
        half_scale = softmax_scale * Float32(0.5)

        @dsl_user_op
        # pyre-ignore[11]
        def _silu_pair(a: T, b: T, hs: T, *, loc=None) -> Tuple[T, T]:
            half_a = a * hs
            half_b = b * hs
            t0 = llvm.inline_asm(
                cutlass.Float32.mlir_type,
                [half_a.ir_value()],
                "tanh.approx.f32 $0, $1;",
                "=f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
            t1 = llvm.inline_asm(
                cutlass.Float32.mlir_type,
                [half_b.ir_value()],
                "tanh.approx.f32 $0, $1;",
                "=f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
            r0 = half_a * Float32(t0) + half_a
            r1 = half_b * Float32(t1) + half_b
            return r0, r1

        if const_expr(mAttnScale is not None and cached_combined is None):
            cS_scale = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS_scale = thr_mma_qk.partition_C(cS_scale)
            # pyre-ignore[16]
            attn_scale_len = cute.size(mAttnScale.shape[0])

        if const_expr(mAttnScale is not None):
            if const_expr(cached_combined is not None):
                for i in cutlass.range_constexpr(0, cute.size(acc_S), 2):
                    row_scale = cached_combined[i // 2]
                    acc_S[i], acc_S[i + 1] = _silu_pair(
                        acc_S[i], acc_S[i + 1], half_scale
                    )
                    acc_S[i] = acc_S[i] * row_scale
                    acc_S[i + 1] = acc_S[i + 1] * row_scale
            else:
                for i in cutlass.range_constexpr(0, cute.size(acc_S), 2):
                    # pyre-ignore[61]
                    row_idx_p1 = tScS_scale[i][0] + m_block * self.tile_m
                    # pyre-ignore[61]
                    row_idx_p1 = cutlass.min(row_idx_p1, attn_scale_len - Int32(1))
                    row_scale = mAttnScale[row_idx_p1]  # pyre-ignore[16]
                    acc_S[i], acc_S[i + 1] = _silu_pair(
                        acc_S[i], acc_S[i + 1], half_scale
                    )
                    acc_S[i] = acc_S[i] * row_scale
                    acc_S[i + 1] = acc_S[i + 1] * row_scale
        else:
            for i in cutlass.range_constexpr(0, cute.size(acc_S), 2):
                acc_S[i], acc_S[i + 1] = _silu_pair(acc_S[i], acc_S[i + 1], half_scale)

        # Phase 2: Apply masking using R2P bitmask (per-row)
        if const_expr(apply_causal_mask or mask_seqlen):
            acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS_mn = utils.make_acc_tensor_mn_view(thr_mma_qk.partition_C(cS))
            thr_col_offset = tScS_mn[0][1]
            safe_n_block = cutlass.max(n_block, Int32(0))
            n_offset = safe_n_block * self.tile_n

            row_idx = Int32(0)
            col_limit_right = Int32(0)

            if const_expr(apply_causal_mask):
                if const_expr(self.is_diagonal):
                    causal_row_offset = Int32(1) - n_offset - thr_col_offset
                else:
                    causal_row_offset = (
                        Int32(1) + seqlen_k - n_offset - seqlen_q - thr_col_offset
                    )
                seqlenk_col_limit = seqlen_k - n_offset - thr_col_offset
                if const_expr(window_size_left is not None):
                    # LOCAL: fuse right + left masks into one R2P pass per row.
                    local_row_offset_left = (
                        causal_row_offset - Int32(1) - window_size_left
                    )
                    # pyre-ignore[16]
                    for r in cutlass.range(
                        # pyre-ignore[16]
                        cute.size(acc_S_mn.shape[0]),
                        unroll_full=True,
                    ):
                        row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        col_limit_right = row_idx + causal_row_offset
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(
                                col_limit_right, seqlenk_col_limit
                            )
                        col_limit_left = cutlass.max(
                            row_idx + local_row_offset_left, Int32(0)
                        )
                        mask_r2p_zero_combined(
                            acc_S_mn[r, None],
                            col_limit_right,
                            col_limit_left,
                            arch=90,
                        )
                else:
                    # pyre-ignore[16]
                    for r in cutlass.range(
                        cute.size(acc_S_mn.shape[0]), unroll_full=True
                    ):
                        row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        col_limit_right = row_idx + causal_row_offset
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(
                                col_limit_right, seqlenk_col_limit
                            )
                        mask_r2p_zero(acc_S_mn[r, None], col_limit_right, arch=90)
            else:
                seqlenk_col_limit = seqlen_k - n_offset - thr_col_offset
                for r in cutlass.range(cute.size(acc_S_mn.shape[0]), unroll_full=True):
                    mask_r2p_zero(acc_S_mn[r, None], seqlenk_col_limit, arch=90)

    @cute.jit
    def mma_one_n_block_silu(
        self,
        smem_pipe_read,
        n_block: Int32,
        m_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        tiled_mma_pv_rs: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        thr_mma_qk,
        softmax_scale: Float32,
        seqlen: SeqlenInfoQK,
        mAttnScale: cute.Tensor | None,
        apply_causal_mask: cutlass.Constexpr[bool],
        mask_seqlen: cutlass.Constexpr[bool],
        window_size_left: Int32 | None,
        cached_combined=None,
    ):
        """Per-n_block SiLU attention: QK GEMM -> SiLU + mask -> PV GEMM."""
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        self.silu_activate(
            acc_S,
            softmax_scale,
            thr_mma_qk,
            m_block,
            n_block,
            seqlen.seqlen_q,
            seqlen.seqlen_k,
            mAttnScale,
            apply_causal_mask,
            mask_seqlen,
            window_size_left,
            cached_combined=cached_combined,
        )

        tOrP_acc = cute.make_tensor(
            acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout)
        )
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_fragment_like(tOrP_acc, self.dtype)
        )
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            # pyre-ignore[16]
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            # pyre-ignore[16]
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
            cute.arch.sync_warp()
        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        tiled_mma_pv_rs: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        score_mod_fn: Callable | None = None,
        mask_fn: Callable | None = None,
        # pyre-ignore[9]
        is_first_n_block: cutlass.Constexpr = False,
        # pyre-ignore[9]
        check_inf: cutlass.Constexpr = True,
    ):
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        if const_expr(score_mod_fn is not None):
            # pyre-ignore[29]
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            # pyre-ignore[29]
            mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        tOrP_acc = cute.make_tensor(
            acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout)
        )
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_fragment_like(tOrP_acc, self.dtype)
        )
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            # pyre-ignore[16]
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            # pyre-ignore[16]
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
            cute.arch.sync_warp()
        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        tiled_mma_pv_rs: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        score_mod_fn: Callable | None = None,
        mask_fn: Callable | None = None,
        # pyre-ignore[9]
        check_inf: cutlass.Constexpr = True,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        pipeline_v.consumer_wait(
            smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v)
        )
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        if const_expr(score_mod_fn is not None):
            # pyre-ignore[29]
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            # pyre-ignore[29]
            mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = cute.make_tensor(
            acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout)
        )
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_fragment_like(tOrP_acc, self.dtype)
        )
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            # pyre-ignore[16]
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            # pyre-ignore[16]
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
            cute.arch.sync_warp()
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_tensors: list | None = None,
        fastdiv_mods=None,
    ):
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_mma_warp_groups in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_mma_warp_groups == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_mma_warp_groups
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )


def _cutedsl_mha_hopper(
    Q_list: List[torch.Tensor],
    K_list: List[torch.Tensor],
    V_list: List[torch.Tensor],
    q_seq_offsets: torch.Tensor,
    kv_seq_offsets: torch.Tensor,
    attn_scale_list: List[torch.Tensor],
    mask_matrix: List[List["MaskType"]],
    alpha: float = 1.0,
    max_attn_len: int | None = None,
) -> List[torch.Tensor]:
    """Hopper (SM90) forward using FA4 FlashAttentionForwardSm90 kernel."""
    dtype = Q_list[0].dtype
    device = Q_list[0].device
    H = Q_list[0].shape[1]  # num heads
    dim_q = Q_list[0].shape[2]
    dim_v = V_list[0].shape[2]
    B = q_seq_offsets.shape[1] - 1  # batch size

    # Pad dimensions to multiple of 16 for MMA tile alignment
    dim_q_padded = (dim_q + 15) // 16 * 16
    dim_v_padded = (dim_v + 15) // 16 * 16

    # Pad input tensors to padded dimensions
    if dim_q < dim_q_padded:
        Q_list = [torch.nn.functional.pad(q, (0, dim_q_padded - dim_q)) for q in Q_list]
        K_list = [torch.nn.functional.pad(k, (0, dim_q_padded - dim_q)) for k in K_list]
    if dim_v < dim_v_padded:
        V_list = [torch.nn.functional.pad(v, (0, dim_v_padded - dim_v)) for v in V_list]

    stream_handle = torch.cuda.current_stream().cuda_stream
    cu_stream = cuda.CUstream(stream_handle)  # pyre-ignore[16]

    # Allocate per-Q output tensors
    Out_list = []
    for Q in Q_list:
        Out_list.append(
            torch.empty(Q.shape[0], H, dim_v_padded, dtype=dtype, device=device)
        )

    first_k_written = [False] * len(Q_list)

    # When True, fuse the ALL + CAUSAL launch
    _is_semilocal_specialized_fwd = _is_semilocal_mask_matrix(mask_matrix)

    # Pre-build cute tensors outside the loop
    q_cute_list = [
        from_dlpack(Q_list[qi].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=Q_list[qi].ndim - 1
        )
        for qi in range(len(Q_list))
    ]
    k_cute_list = [
        from_dlpack(K_list[ki].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=K_list[ki].ndim - 1
        )
        for ki in range(len(K_list))
    ]
    v_cute_list = [
        from_dlpack(V_list[ki].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=V_list[ki].ndim - 1
        )
        for ki in range(len(V_list))
    ]
    cu_seqlens_q_list = [
        q_seq_offsets[qi].to(torch.int32).contiguous() for qi in range(len(Q_list))
    ]
    cu_seqlens_k_list = [
        kv_seq_offsets[ki].to(torch.int32).contiguous() for ki in range(len(K_list))
    ]
    cu_seqlens_q_cute_list = [
        from_dlpack(cs, assumed_align=4).mark_layout_dynamic(leading_dim=0)
        for cs in cu_seqlens_q_list
    ]
    cu_seqlens_k_cute_list = [
        from_dlpack(cs, assumed_align=4).mark_layout_dynamic(leading_dim=0)
        for cs in cu_seqlens_k_list
    ]
    attn_scale_cute_list = [
        from_dlpack(
            attn_scale_list[qi].to(torch.float32).contiguous(), assumed_align=4
        ).mark_layout_dynamic(leading_dim=0)
        for qi in range(len(Q_list))
    ]

    # Iterate over Q-K pairs per mask_matrix
    for qi in range(len(Q_list)):
        for ki in range(len(K_list)):
            mask_type = mask_matrix[qi][ki]
            if mask_type == MaskType.NULL:
                continue

            if (
                _is_semilocal_specialized_fwd
                and qi == 1
                and ki == 1
                and mask_type == MaskType.CAUSAL
            ):
                continue

            # Upstream FA4 semantics: is_causal and is_local are mutually
            # exclusive.  When a window is present (DIAGONAL / LOCAL),
            # the causal constraint is handled by the window bounds,
            # so is_causal must be False.
            window_size_left = None
            window_size_right = None
            if mask_type == MaskType.DIAGONAL:
                window_size_left = 0
                window_size_right = 0
            elif mask_type == MaskType.LOCAL:
                assert max_attn_len is not None, (
                    "max_attn_len must be provided for LOCAL mask type"
                )
                window_size_left = max_attn_len - 1
                window_size_right = 0
            is_local = window_size_left is not None
            is_causal = (
                mask_type in (MaskType.CAUSAL, MaskType.DIAGONAL, MaskType.LOCAL)
                and not is_local
            )
            is_diagonal = mask_type == MaskType.DIAGONAL

            is_first_k = not first_k_written[qi]
            if is_first_k:
                cur_o_buf = Out_list[qi]
            else:
                cur_o_buf = torch.empty(Out_list[qi].shape, dtype=dtype, device=device)

            q_tensor = q_cute_list[qi]
            k_tensor = k_cute_list[ki]
            v_tensor = v_cute_list[ki]
            o_tensor = from_dlpack(cur_o_buf, assumed_align=16).mark_layout_dynamic(
                leading_dim=cur_o_buf.ndim - 1
            )
            cu_seqlens_q_tensor = cu_seqlens_q_cute_list[qi]
            cu_seqlens_k_tensor = cu_seqlens_k_cute_list[ki]
            attn_scale_tensor = attn_scale_cute_list[qi]

            softmax_scale = alpha

            _use_semilocal_fused_fwd = (
                _is_semilocal_specialized_fwd
                and qi == 1
                and ki == 0
                and mask_type == MaskType.ALL
            )

            tile_m_const = 64 if _use_semilocal_fused_fwd else 192
            tile_n_const = 64
            k_alt_cute = None
            v_alt_cute = None
            cu_seqlens_k_alt_cute = None
            if _use_semilocal_fused_fwd:
                _alt_ki = 1
                k_alt_cute = k_cute_list[_alt_ki]
                v_alt_cute = v_cute_list[_alt_ki]
                cu_seqlens_k_alt_cute = cu_seqlens_k_cute_list[_alt_ki]

            cache_key = (
                "hopper_fwd",
                dtype,
                dim_q_padded,
                dim_v_padded,
                H,
                B,
                _next_power_of_2(Q_list[qi].shape[0]),
                _next_power_of_2(K_list[ki].shape[0]),
                is_causal,
                is_local,
                is_diagonal,
                window_size_left,
                window_size_right,
                _use_semilocal_fused_fwd,
                _next_power_of_2(K_list[1].shape[0])
                if _use_semilocal_fused_fwd
                else None,
                tile_m_const,
            )

            if cache_key not in _compiled_kernel_cache_fwd_hopper:
                fa_fwd = FlashAttentionForwardSm90(
                    dtype=cutlass.BFloat16
                    if dtype == torch.bfloat16
                    else cutlass.Float16,
                    head_dim=dim_q_padded,
                    head_dim_v=dim_v_padded,
                    is_causal=is_causal,
                    is_local=is_local,
                    pack_gqa=False,
                    tile_m=tile_m_const,
                    tile_n=tile_n_const,
                    num_stages=4,
                    use_silu=True,
                    is_diagonal=is_diagonal,
                )

                _compile_kwargs = {}
                if _use_semilocal_fused_fwd:
                    _compile_kwargs["mK_alt"] = k_alt_cute
                    _compile_kwargs["mV_alt"] = v_alt_cute
                    _compile_kwargs["mCuSeqlensK_alt"] = cu_seqlens_k_alt_cute
                _compiled_kernel_cache_fwd_hopper[cache_key] = cute.compile(
                    fa_fwd,
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    o_tensor,
                    None,  # mLSE — not needed for SiLU
                    softmax_scale,
                    cu_stream,
                    cu_seqlens_q_tensor,  # mCuSeqlensQ
                    cu_seqlens_k_tensor,  # mCuSeqlensK
                    None,  # mSeqUsedQ
                    None,  # mSeqUsedK
                    None,  # mPageTable
                    window_size_left,  # window_size_left
                    window_size_right,  # window_size_right
                    None,  # learnable_sink
                    None,  # blocksparse_tensors
                    None,  # aux_tensors
                    attn_scale_tensor,  # mAttnScale
                    **_compile_kwargs,
                )

            compiled = _compiled_kernel_cache_fwd_hopper[cache_key]
            _invoke_kwargs = {}
            if _use_semilocal_fused_fwd:
                _invoke_kwargs["mK_alt"] = k_alt_cute
                _invoke_kwargs["mV_alt"] = v_alt_cute
                _invoke_kwargs["mCuSeqlensK_alt"] = cu_seqlens_k_alt_cute
            compiled(  # pyre-ignore[29]
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                None,  # mLSE — not needed for SiLU
                softmax_scale,
                cu_stream,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                None,  # mSeqUsedQ
                None,  # mSeqUsedK
                None,  # mPageTable
                window_size_left,
                window_size_right,
                None,  # learnable_sink
                None,  # blocksparse_tensors
                None,  # aux_tensors
                attn_scale_tensor,  # mAttnScale
                **_invoke_kwargs,
            )

            if is_first_k:
                first_k_written[qi] = True
            else:
                Out_list[qi] += cur_o_buf

    # Zero any output entries never written
    for qi in range(len(Q_list)):
        if not first_k_written[qi]:
            Out_list[qi].zero_()

    # Slice to original dims if padded
    if dim_v < dim_v_padded:
        Out_list = [o[:, :, :dim_v].contiguous() for o in Out_list]

    return Out_list


# =============================================================================
# Blackwell (SM100) forward
# =============================================================================

# Cache for compiled Blackwell kernels
_compiled_kernel_cache_fwd_blackwell: Dict[Tuple, object] = {}


def _cutedsl_mha_blackwell(
    Q_list: List[torch.Tensor],
    K_list: List[torch.Tensor],
    V_list: List[torch.Tensor],
    q_seq_offsets: torch.Tensor,
    kv_seq_offsets: torch.Tensor,
    attn_scale_list: List[torch.Tensor],
    mask_matrix: List[List["MaskType"]],
    alpha: float = 1.0,
    max_attn_len: Optional[int] = None,
) -> List[torch.Tensor]:
    """Blackwell (SM100) SiLU attention forward"""
    dtype = Q_list[0].dtype
    device = Q_list[0].device
    H = Q_list[0].shape[1]  # num heads
    dim_q = Q_list[0].shape[2]
    dim_v = V_list[0].shape[2]
    B = q_seq_offsets.shape[1] - 1  # batch size

    # Pad dimensions to multiple of 32 for MMA tile alignment (matching Hopper path)
    dim_q_padded = (dim_q + 31) // 32 * 32
    dim_v_padded = (dim_v + 31) // 32 * 32

    # Pad input tensors to padded dimensions
    if dim_q < dim_q_padded:
        Q_list = [torch.nn.functional.pad(q, (0, dim_q_padded - dim_q)) for q in Q_list]
        K_list = [torch.nn.functional.pad(k, (0, dim_q_padded - dim_q)) for k in K_list]
    if dim_v < dim_v_padded:
        V_list = [torch.nn.functional.pad(v, (0, dim_v_padded - dim_v)) for v in V_list]

    stream_handle = torch.cuda.current_stream().cuda_stream
    # pyre-ignore[16]
    cu_stream = cuda.CUstream(stream_handle)

    # Allocate per-Q output tensors in input dtype for accumulation
    Out_list = []
    for Q in Q_list:
        Out_list.append(
            torch.empty(Q.shape[0], H, dim_v_padded, dtype=dtype, device=device)
        )

    # Track whether each Q has received its first K pair's output
    first_k_written = [False] * len(Q_list)

    # Pre-build cute tensors outside the loop (avoid repeated from_dlpack)
    q_cute_list = [
        from_dlpack(Q_list[qi].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=Q_list[qi].ndim - 1
        )
        for qi in range(len(Q_list))
    ]
    k_cute_list = [
        from_dlpack(K_list[ki].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=K_list[ki].ndim - 1
        )
        for ki in range(len(K_list))
    ]
    v_cute_list = [
        from_dlpack(V_list[ki].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=V_list[ki].ndim - 1
        )
        for ki in range(len(V_list))
    ]
    cu_seqlens_q_list = [
        q_seq_offsets[qi].to(torch.int32).contiguous() for qi in range(len(Q_list))
    ]
    cu_seqlens_k_list = [
        kv_seq_offsets[ki].to(torch.int32).contiguous() for ki in range(len(K_list))
    ]
    cu_seqlens_q_cute_list = [
        from_dlpack(cs, assumed_align=4).mark_layout_dynamic(leading_dim=0)
        for cs in cu_seqlens_q_list
    ]
    cu_seqlens_k_cute_list = [
        from_dlpack(cs, assumed_align=4).mark_layout_dynamic(leading_dim=0)
        for cs in cu_seqlens_k_list
    ]
    attn_scale_cute_list = [
        from_dlpack(
            attn_scale_list[qi].to(torch.float32).contiguous(), assumed_align=4
        ).mark_layout_dynamic(leading_dim=0)
        for qi in range(len(Q_list))
    ]

    # Iterate over Q-K pairs per mask_matrix, with per-Q streams for concurrency
    for qi in range(len(Q_list)):
        for ki in range(len(K_list)):
            mask_type = mask_matrix[qi][ki]
            if mask_type == MaskType.NULL:
                continue

            is_causal = mask_type in (
                MaskType.CAUSAL,
                MaskType.DIAGONAL,
                MaskType.LOCAL,
            )
            is_diagonal = mask_type == MaskType.DIAGONAL
            window_size_left = None
            window_size_right = None
            if mask_type == MaskType.DIAGONAL:
                window_size_left = 0
                window_size_right = 0
            elif mask_type == MaskType.LOCAL:
                assert max_attn_len is not None, (
                    "max_attn_len must be provided for LOCAL mask type"
                )
                window_size_left = max_attn_len - 1
                window_size_right = 0

            # For the first K pair per Q, write directly to Out_list[qi] (pre-zeroed)
            is_first_k = not first_k_written[qi]
            if is_first_k:
                o_buf = Out_list[qi]
            else:
                o_buf = torch.empty(Out_list[qi].shape, dtype=dtype, device=device)

            q_tensor = q_cute_list[qi]
            k_tensor = k_cute_list[ki]
            v_tensor = v_cute_list[ki]
            o_tensor = from_dlpack(o_buf, assumed_align=16).mark_layout_dynamic(
                leading_dim=o_buf.ndim - 1
            )
            cu_seqlens_q_tensor = cu_seqlens_q_cute_list[qi]
            cu_seqlens_k_tensor = cu_seqlens_k_cute_list[ki]
            attn_scale_tensor = attn_scale_cute_list[qi]

            softmax_scale = alpha

            # Cache key includes pair indices, causal flag, and window size
            cache_key = (
                dtype,
                dim_q_padded,
                dim_v_padded,
                H,
                B,
                _next_power_of_2(Q_list[qi].shape[0]),
                _next_power_of_2(K_list[ki].shape[0]),
                is_causal,
                is_diagonal,
                window_size_left,
            )

            if cache_key not in _compiled_kernel_cache_fwd_blackwell:
                fa_fwd = FlashAttentionForwardSm100(
                    head_dim=dim_q_padded,
                    head_dim_v=dim_v_padded,
                    is_causal=is_causal,
                    is_local=window_size_left is not None,
                    is_varlen_q=True,
                    is_persistent=False,
                    use_silu=True,
                    is_diagonal=is_diagonal,
                )

                _compiled_kernel_cache_fwd_blackwell[cache_key] = cute.compile(
                    fa_fwd,
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    o_tensor,
                    None,  # mLSE
                    softmax_scale,
                    cu_stream,
                    cu_seqlens_q_tensor,  # mCuSeqlensQ
                    cu_seqlens_k_tensor,  # mCuSeqlensK
                    None,  # mSeqUsedQ
                    None,  # mSeqUsedK
                    None,  # mPageTable
                    window_size_left,  # window_size_left
                    window_size_right,  # window_size_right
                    None,  # learnable_sink
                    None,  # blocksparse_tensors
                    None,  # aux_tensors
                    None,  # mSFQ
                    None,  # mSFK
                    None,  # mSFV
                    None,  # mCuSeqlensSFQ
                    None,  # mCuSeqlensSFK
                    None,  # total_sf_q
                    None,  # total_sf_k
                    None,  # mTileToBatch
                    None,  # mTileToHead
                    None,  # mTileToBlock
                    None,  # mCuSeqlensO
                    attn_scale_tensor,  # mAttnScale
                )

            compiled = _compiled_kernel_cache_fwd_blackwell[cache_key]
            # pyre-ignore[29]
            compiled(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                None,  # mLSE
                softmax_scale,
                cu_stream,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                None,  # mSeqUsedQ
                None,  # mSeqUsedK
                None,  # mPageTable
                window_size_left,  # window_size_left
                window_size_right,  # window_size_right
                None,  # learnable_sink
                None,  # blocksparse_tensors
                None,  # aux_tensors
                None,  # mSFQ
                None,  # mSFK
                None,  # mSFV
                None,  # mCuSeqlensSFQ
                None,  # mCuSeqlensSFK
                None,  # total_sf_q
                None,  # total_sf_k
                None,  # mTileToBatch
                None,  # mTileToHead
                None,  # mTileToBlock
                None,  # mCuSeqlensO
                attn_scale_tensor,  # mAttnScale
            )

            # Accumulate this pair's output into the Q tensor's total output
            if is_first_k:
                first_k_written[qi] = True
            else:
                Out_list[qi] += o_buf

    # Zero any output entries never written
    for qi in range(len(Q_list)):
        if not first_k_written[qi]:
            Out_list[qi].zero_()

    # Slice to original dim_v if padded
    if dim_v < dim_v_padded:
        Out_list = [o[:, :, :dim_v].contiguous() for o in Out_list]

    return Out_list


# Cache for compiled Hopper backward kernels
_compiled_kernel_cache_bwd_hopper: Dict[Tuple, object] = {}


def _is_semilocal_mask_matrix(mask_matrix: List[List["MaskType"]]) -> bool:
    if len(mask_matrix) != 2 or any(len(row) != 2 for row in mask_matrix):
        return False
    return (
        mask_matrix[0][0] == MaskType.LOCAL
        and mask_matrix[0][1] == MaskType.NULL
        and mask_matrix[1][0] == MaskType.ALL
        and mask_matrix[1][1] == MaskType.CAUSAL
    )


def _cutedsl_mha_hopper_backward(
    Q_list: List[torch.Tensor],
    K_list: List[torch.Tensor],
    V_list: List[torch.Tensor],
    dO_list: List[torch.Tensor],
    q_seq_offsets: torch.Tensor,
    kv_seq_offsets: torch.Tensor,
    attn_scale_list: List[torch.Tensor],
    mask_matrix: List[List["MaskType"]],
    alpha: float = 1.0,
    max_attn_len: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Hopper (SM90) attention backward."""
    from hammer.v3.ops.cutedsl.cutedsl_attention_bwd import FlashAttentionBackwardSm90

    dtype = Q_list[0].dtype
    device = Q_list[0].device
    H = Q_list[0].shape[1]
    dim_q = Q_list[0].shape[2]
    dim_v = V_list[0].shape[2]
    B = q_seq_offsets.shape[1] - 1

    # Pad dimensions to multiple of 16 for SM90 MMA tile alignment
    dim_q_padded = (dim_q + 15) // 16 * 16
    dim_v_padded = (dim_v + 15) // 16 * 16

    if dim_q < dim_q_padded:
        Q_list = [torch.nn.functional.pad(q, (0, dim_q_padded - dim_q)) for q in Q_list]
        K_list = [torch.nn.functional.pad(k, (0, dim_q_padded - dim_q)) for k in K_list]
    if dim_v < dim_v_padded:
        V_list = [torch.nn.functional.pad(v, (0, dim_v_padded - dim_v)) for v in V_list]
        dO_list = [
            torch.nn.functional.pad(do, (0, dim_v_padded - dim_v)) for do in dO_list
        ]

    stream_handle = torch.cuda.current_stream().cuda_stream
    cu_stream = cuda.CUstream(stream_handle)  # pyre-ignore[16]

    # dQ needs to be zero-initialized since TMA reduce-add target
    dQ_list = [torch.zeros_like(q) for q in Q_list]
    dK_list = [torch.empty_like(k) for k in K_list]
    dV_list = [torch.empty_like(v) for v in V_list]

    _is_semilocal_specialized = _is_semilocal_mask_matrix(mask_matrix)

    cu_seqlens_q_list = [
        q_seq_offsets[qi].to(torch.int32).contiguous() for qi in range(len(Q_list))
    ]
    cu_seqlens_k_list = [
        kv_seq_offsets[ki].to(torch.int32).contiguous() for ki in range(len(K_list))
    ]

    m_block_size = 64
    n_block_size = 128
    softmax_scale = alpha

    # Precompute CPU seqlens
    # pyre-ignore[9]
    _cu_k_cpu_list: List[Optional[torch.Tensor]] = [None] * len(cu_seqlens_k_list)
    # pyre-ignore[9]
    _cu_q_cpu_list: List[Optional[torch.Tensor]] = [None] * len(cu_seqlens_q_list)

    def _get_cu_k_cpu(idx: int) -> torch.Tensor:
        if _cu_k_cpu_list[idx] is None:
            _cu_k_cpu_list[idx] = cu_seqlens_k_list[idx].cpu()
        # pyre-ignore[7]
        return _cu_k_cpu_list[idx]

    def _get_cu_q_cpu(idx: int) -> torch.Tensor:
        if _cu_q_cpu_list[idx] is None:
            _cu_q_cpu_list[idx] = cu_seqlens_q_list[idx].cpu()
        # pyre-ignore[7]
        return _cu_q_cpu_list[idx]

    _dK_written = set()
    _dV_written = set()
    _dQ_postprocessed = set()
    _ki_dK_orig: dict[int, torch.Tensor] = {}
    _ki_dV_orig: dict[int, torch.Tensor] = {}

    _per_qi_state: List[Optional[Dict[str, object]]] = []
    for qi in range(len(Q_list)):
        if all(mask_matrix[qi][ki] == MaskType.NULL for ki in range(len(K_list))):
            _per_qi_state.append(None)
            continue
        Q_qi = Q_list[qi]
        cu_seqlens_q_qi = cu_seqlens_q_list[qi]
        total_q_padded_qi = (
            (Q_qi.shape[0] + cu_seqlens_q_qi.shape[0] * m_block_size - 1)
            // m_block_size
            * m_block_size
        )
        _per_qi_state.append(
            {
                "total_q_padded": total_q_padded_qi,
                "LSE": torch.empty(
                    (H, total_q_padded_qi),
                    dtype=torch.float32,
                    device=device,
                ),
                "dPsum": torch.empty(
                    (H, total_q_padded_qi),
                    dtype=torch.float32,
                    device=device,
                ),
                "attn_scale_f32": (
                    attn_scale_list[qi].to(torch.float32).contiguous()
                    if attn_scale_list[qi] is not None
                    else None
                ),
            }
        )

    for qi in range(len(Q_list)):
        _qi_state = _per_qi_state[qi]
        _qi_LSE = _qi_state["LSE"] if _qi_state is not None else None
        _qi_dPsum = _qi_state["dPsum"] if _qi_state is not None else None
        _qi_attn_scale_f32 = (
            _qi_state["attn_scale_f32"] if _qi_state is not None else None
        )
        _qi_dQ_wrote = False
        _qi_K_diag: torch.Tensor | None = None
        _qi_V_diag: torch.Tensor | None = None

        for ki in range(len(K_list)):
            mask_type = mask_matrix[qi][ki]
            if mask_type == MaskType.NULL:
                continue

            if (
                _is_semilocal_specialized
                and qi == 1
                and ki == 0
                and mask_type == MaskType.ALL
            ):
                _qi_dQ_wrote = True
                continue

            window_size_left = None
            window_size_right = None
            if mask_type == MaskType.DIAGONAL:
                window_size_left = 0
                window_size_right = 0
            elif mask_type == MaskType.LOCAL:
                assert max_attn_len is not None
                window_size_left = max_attn_len - 1
                window_size_right = 0
            is_local = window_size_left is not None
            is_causal = (
                mask_type in (MaskType.CAUSAL, MaskType.DIAGONAL, MaskType.LOCAL)
                and not is_local
            )

            Q = Q_list[qi]
            K = K_list[ki]
            V = V_list[ki]
            dO = dO_list[qi]
            cu_seqlens_q = cu_seqlens_q_list[qi]
            cu_seqlens_k = cu_seqlens_k_list[ki]

            # DIAGONAL: rearrange K/V to Q's layout so seqlen_q == seqlen_k
            _diag_rearrange = False
            if mask_type == MaskType.DIAGONAL:
                _diag_q_offsets_cpu = _get_cu_q_cpu(qi)
                _diag_k_offsets_cpu = _get_cu_k_cpu(ki)
                q_lens = _diag_q_offsets_cpu[1:] - _diag_q_offsets_cpu[:-1]
                k_lens = _diag_k_offsets_cpu[1:] - _diag_k_offsets_cpu[:-1]
                max_delta = int((q_lens - k_lens).abs().max())
                if max_delta > 0:
                    _diag_rearrange = True
                    _K_diag_shape = (Q.shape[0], K.shape[1], K.shape[2])
                    _V_diag_shape = (Q.shape[0], V.shape[1], V.shape[2])
                    if (
                        _qi_K_diag is None
                        or _qi_K_diag.shape != _K_diag_shape
                        or _qi_K_diag.dtype != K.dtype
                    ):
                        _qi_K_diag = torch.zeros(
                            _K_diag_shape, dtype=K.dtype, device=device
                        )
                    else:
                        _qi_K_diag.zero_()
                    if (
                        _qi_V_diag is None
                        or _qi_V_diag.shape != _V_diag_shape
                        or _qi_V_diag.dtype != V.dtype
                    ):
                        _qi_V_diag = torch.zeros(
                            _V_diag_shape, dtype=V.dtype, device=device
                        )
                    else:
                        _qi_V_diag.zero_()
                    K_diag = _qi_K_diag
                    V_diag = _qi_V_diag
                    eff_lens = torch.minimum(q_lens, k_lens).long()
                    total_eff = int(eff_lens.sum().item())
                    if total_eff > 0:
                        q_starts = _diag_q_offsets_cpu[:-1].long()
                        k_starts = _diag_k_offsets_cpu[:-1].long()
                        k_base = torch.repeat_interleave(k_starts, eff_lens)
                        q_base = torch.repeat_interleave(q_starts, eff_lens)
                        eff_cs = torch.zeros(B + 1, dtype=torch.long)
                        eff_cs[1:] = eff_lens.cumsum(0)
                        within = torch.arange(total_eff) - torch.repeat_interleave(
                            eff_cs[:-1], eff_lens
                        )
                        _diag_src_idx = (k_base + within).to(device)
                        _diag_dst_idx = (q_base + within).to(device)
                        K_diag[_diag_dst_idx] = K[_diag_src_idx]
                        V_diag[_diag_dst_idx] = V[_diag_src_idx]
                    else:
                        _diag_src_idx = None
                        _diag_dst_idx = None
                    K = K_diag
                    V = V_diag
                    cu_seqlens_k = cu_seqlens_q

            LSE = _qi_LSE
            dPsum = _qi_dPsum
            dQ_buf_qi = dQ_list[qi]

            # Direct write to output tensors
            kernel_accumulate = (
                ki in _dK_written and ki in _dV_written and not _diag_rearrange
            )

            if not _diag_rearrange:
                dK_buf = dK_list[ki]
                dV_buf = dV_list[ki]
            else:
                dK_buf = torch.empty_like(K)
                dV_buf = torch.empty_like(V)

            # Cute tensors - 3D varlen layout (total_seq, nheads, hdim)
            q_cute = from_dlpack(Q.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=Q.ndim - 1
            )
            k_cute = from_dlpack(K.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=K.ndim - 1
            )
            v_cute = from_dlpack(V.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=V.ndim - 1
            )
            do_cute = from_dlpack(dO.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=dO.ndim - 1
            )
            lse_cute = from_dlpack(LSE, assumed_align=16).mark_layout_dynamic(
                # pyre-ignore[16]
                leading_dim=LSE.ndim - 1
            )
            dpsum_cute = from_dlpack(dPsum, assumed_align=16).mark_layout_dynamic(
                leading_dim=dPsum.ndim - 1  # pyre-ignore[16]
            )
            dqaccum_cute = from_dlpack(dQ_buf_qi, assumed_align=16).mark_layout_dynamic(
                leading_dim=dQ_buf_qi.ndim - 1
            )
            dk_cute = from_dlpack(dK_buf, assumed_align=16).mark_layout_dynamic(
                leading_dim=dK_buf.ndim - 1
            )
            dv_cute = from_dlpack(dV_buf, assumed_align=16).mark_layout_dynamic(
                leading_dim=dV_buf.ndim - 1
            )
            cu_seqlens_q_cute = from_dlpack(
                cu_seqlens_q, assumed_align=4
            ).mark_layout_dynamic(leading_dim=0)
            cu_seqlens_k_cute = from_dlpack(
                cu_seqlens_k, assumed_align=4
            ).mark_layout_dynamic(leading_dim=0)
            has_attn_scale = _qi_attn_scale_f32 is not None
            if has_attn_scale:
                attn_scale_cute = from_dlpack(
                    _qi_attn_scale_f32,
                    assumed_align=4,
                ).mark_layout_dynamic(leading_dim=0)
            else:
                attn_scale_cute = None

            has_attn_scale_alt = False
            if _is_semilocal_specialized and is_local:
                _alt_state_for_key = _per_qi_state[qi + 1]
                assert _alt_state_for_key is not None
                has_attn_scale_alt = _alt_state_for_key["attn_scale_f32"] is not None

            bwd_key = (
                "hopper_bwd",
                dtype,
                dim_q_padded,
                dim_v_padded,
                H,
                B,
                _next_power_of_2(Q.shape[0]),
                _next_power_of_2(K.shape[0]),
                is_causal,
                is_local,
                window_size_left,
                window_size_right,
                has_attn_scale,
                not is_causal and not is_local,
                mask_type == MaskType.DIAGONAL,
                kernel_accumulate,
                _is_semilocal_specialized,
                has_attn_scale_alt,
            )

            _use_persistent = False

            # Build tile mapping tables for persistent varlen scheduler
            if _use_persistent:
                _tile_key = ("hopper_tile_map", ki, K.shape[0], H, B, device)
                if _tile_key not in _compiled_kernel_cache_bwd_hopper:
                    _cpu_csl = _get_cu_k_cpu(ki)
                    _k_lens = _cpu_csl[1:] - _cpu_csl[:-1]
                    _n_blocks = (_k_lens + n_block_size - 1) // n_block_size
                    _tiles_per_batch = H * _n_blocks
                    _total_tiles = int(_tiles_per_batch.sum().item())
                    _batch_ids = torch.repeat_interleave(
                        torch.arange(B, dtype=torch.int32),
                        _tiles_per_batch.to(torch.int64),
                    )
                    _pos_in_batch = torch.cat(
                        [
                            torch.arange(int(t), dtype=torch.int32)
                            for t in _tiles_per_batch
                        ]
                    )
                    _head_ids = _pos_in_batch // _n_blocks.repeat_interleave(
                        _tiles_per_batch.to(torch.int64)
                    ).to(torch.int32)
                    _block_ids = _pos_in_batch % _n_blocks.repeat_interleave(
                        _tiles_per_batch.to(torch.int64)
                    ).to(torch.int32)
                    _num_sms = 132  # H100
                    _grid = min(_num_sms, _total_tiles)
                    _padded = (_total_tiles + _grid - 1) // _grid * _grid
                    if _padded > _total_tiles:
                        _pad_n = _padded - _total_tiles
                        _batch_ids = torch.cat(
                            [
                                _batch_ids,
                                torch.full((_pad_n,), B, dtype=torch.int32),
                            ]
                        )
                        _head_ids = torch.cat(
                            [_head_ids, torch.zeros(_pad_n, dtype=torch.int32)]
                        )
                        _block_ids = torch.cat(
                            [_block_ids, torch.zeros(_pad_n, dtype=torch.int32)]
                        )
                    _compiled_kernel_cache_bwd_hopper[_tile_key] = (
                        _batch_ids.to(device=device),
                        _head_ids.to(device=device),
                        _block_ids.to(device=device),
                    )
                # pyre-ignore[23]
                _ttb, _tth, _ttbl = _compiled_kernel_cache_bwd_hopper[_tile_key]
                _ttb_cute = from_dlpack(_ttb, assumed_align=4).mark_layout_dynamic(
                    leading_dim=0
                )
                _tth_cute = from_dlpack(_tth, assumed_align=4).mark_layout_dynamic(
                    leading_dim=0
                )
                _ttbl_cute = from_dlpack(_ttbl, assumed_align=4).mark_layout_dynamic(
                    leading_dim=0
                )
            else:
                _ttb_cute = None
                _tth_cute = None
                _ttbl_cute = None

            _use_semilocal_fused = _is_semilocal_specialized and is_local

            q_alt_cute = None
            do_alt_cute = None
            lse_alt_cute = None
            dpsum_alt_cute = None
            dqaccum_alt_cute = None
            cu_seqlens_q_alt_cute = None
            attn_scale_alt_cute = None
            has_attn_scale_alt = False
            if _use_semilocal_fused:
                _alt_qi = qi + 1
                _alt_state = _per_qi_state[_alt_qi]
                assert _alt_state is not None, (
                    "semi-local fused requires alt qi state pre-allocated"
                )
                Q_alt = Q_list[_alt_qi]
                dO_alt = dO_list[_alt_qi]
                cu_seqlens_q_alt = cu_seqlens_q_list[_alt_qi]
                q_alt_cute = from_dlpack(
                    Q_alt.detach(), assumed_align=16
                ).mark_layout_dynamic(leading_dim=Q_alt.ndim - 1)
                do_alt_cute = from_dlpack(
                    dO_alt.detach(), assumed_align=16
                ).mark_layout_dynamic(leading_dim=dO_alt.ndim - 1)
                lse_alt_cute = from_dlpack(
                    _alt_state["LSE"],
                    assumed_align=16,
                    # pyre-ignore[16]
                ).mark_layout_dynamic(leading_dim=_alt_state["LSE"].ndim - 1)
                dpsum_alt_cute = from_dlpack(
                    _alt_state["dPsum"], assumed_align=16
                ).mark_layout_dynamic(leading_dim=_alt_state["dPsum"].ndim - 1)
                # alt-Q also writes directly into its own dQ output tensor.
                _alt_dQ_buf = dQ_list[_alt_qi]
                dqaccum_alt_cute = from_dlpack(
                    _alt_dQ_buf, assumed_align=16
                ).mark_layout_dynamic(leading_dim=_alt_dQ_buf.ndim - 1)
                cu_seqlens_q_alt_cute = from_dlpack(
                    cu_seqlens_q_alt, assumed_align=4
                ).mark_layout_dynamic(leading_dim=0)
                _alt_attn_scale_f32 = _alt_state["attn_scale_f32"]
                has_attn_scale_alt = _alt_attn_scale_f32 is not None
                if has_attn_scale_alt:
                    attn_scale_alt_cute = from_dlpack(
                        _alt_attn_scale_f32,
                        assumed_align=4,
                    ).mark_layout_dynamic(leading_dim=0)

            if bwd_key not in _compiled_kernel_cache_bwd_hopper:
                _bwd_kwargs = dict(
                    dtype=cutlass.BFloat16
                    if dtype == torch.bfloat16
                    else cutlass.Float16,
                    head_dim=dim_q_padded,
                    head_dim_v=dim_v_padded,
                    is_causal=is_causal,
                    is_local=is_local,
                    tile_m=64,
                    tile_n=128,
                    PdS_stage=1,
                    SdP_swapAB=True,
                    use_silu=True,
                    is_persistent=_use_persistent,
                    is_diagonal=(mask_type == MaskType.DIAGONAL),
                    accumulate_dKV=kernel_accumulate,
                    reorder_sdp=True,
                )
                # pyre-ignore[6]
                fa_bwd = FlashAttentionBackwardSm90(**_bwd_kwargs)

                _compile_kwargs = dict(
                    mAttnScale=attn_scale_cute,
                    mTileToBatch=_ttb_cute,
                    mTileToHead=_tth_cute,
                    mTileToBlock=_ttbl_cute,
                )
                if _use_semilocal_fused:
                    _compile_kwargs["mQ_alt"] = q_alt_cute
                    _compile_kwargs["mdO_alt"] = do_alt_cute
                    _compile_kwargs["mLSE_alt"] = lse_alt_cute
                    _compile_kwargs["mdPsum_alt"] = dpsum_alt_cute
                    _compile_kwargs["mdQaccum_alt"] = dqaccum_alt_cute
                    _compile_kwargs["mCuSeqlensQ_alt"] = cu_seqlens_q_alt_cute
                    _compile_kwargs["mAttnScale_alt"] = attn_scale_alt_cute
                _compiled_kernel_cache_bwd_hopper[bwd_key] = cute.compile(
                    fa_bwd,
                    q_cute,
                    k_cute,
                    v_cute,
                    do_cute,
                    lse_cute,
                    dpsum_cute,
                    dqaccum_cute,
                    dk_cute,
                    dv_cute,
                    softmax_scale,
                    cu_stream,
                    cu_seqlens_q_cute,
                    cu_seqlens_k_cute,
                    None,  # mSeqUsedQ
                    None,  # mSeqUsedK
                    None,  # softcap
                    window_size_left,
                    window_size_right,
                    options="--opt-level 2",
                    **_compile_kwargs,
                )

            _invoke_kwargs = dict(
                mAttnScale=attn_scale_cute,
                mTileToBatch=_ttb_cute,
                mTileToHead=_tth_cute,
                mTileToBlock=_ttbl_cute,
            )
            if _use_semilocal_fused:
                _invoke_kwargs["mQ_alt"] = q_alt_cute
                _invoke_kwargs["mdO_alt"] = do_alt_cute
                _invoke_kwargs["mLSE_alt"] = lse_alt_cute
                _invoke_kwargs["mdPsum_alt"] = dpsum_alt_cute
                _invoke_kwargs["mdQaccum_alt"] = dqaccum_alt_cute
                _invoke_kwargs["mAttnScale_alt"] = attn_scale_alt_cute
                _invoke_kwargs["mCuSeqlensQ_alt"] = cu_seqlens_q_alt_cute
            _compiled_kernel_cache_bwd_hopper[bwd_key](  # pyre-ignore[29]
                q_cute,
                k_cute,
                v_cute,
                do_cute,
                lse_cute,
                dpsum_cute,
                dqaccum_cute,
                dk_cute,
                dv_cute,
                softmax_scale,
                cu_stream,
                cu_seqlens_q_cute,
                cu_seqlens_k_cute,
                None,  # mSeqUsedQ
                None,  # mSeqUsedK
                None,  # softcap
                window_size_left,
                window_size_right,
                **_invoke_kwargs,
            )

            _qi_dQ_wrote = True
            if _use_semilocal_fused:
                # pyre-ignore[61]
                _dQ_postprocessed.add(_alt_qi)

            if _diag_rearrange:
                # Reverse rearrangement: dK/dV are in Q's layout, scatter back
                _dK_orig_template = K_list[ki]
                _dV_orig_template = K_list[ki]
                _cached_dK = _ki_dK_orig.get(ki)
                if (
                    _cached_dK is None
                    or _cached_dK.shape != _dK_orig_template.shape
                    or _cached_dK.dtype != _dK_orig_template.dtype
                ):
                    dK_orig = torch.zeros_like(_dK_orig_template)
                    _ki_dK_orig[ki] = dK_orig
                else:
                    _cached_dK.zero_()
                    dK_orig = _cached_dK
                _cached_dV = _ki_dV_orig.get(ki)
                if (
                    _cached_dV is None
                    or _cached_dV.shape != _dV_orig_template.shape
                    or _cached_dV.dtype != _dV_orig_template.dtype
                ):
                    dV_orig = torch.zeros_like(_dV_orig_template)
                    _ki_dV_orig[ki] = dV_orig
                else:
                    _cached_dV.zero_()
                    dV_orig = _cached_dV
                # pyre-ignore[61]
                if _diag_src_idx is not None and _diag_dst_idx is not None:
                    # pyre-ignore[61]
                    dK_orig[_diag_src_idx] = dK_buf[_diag_dst_idx]
                    # pyre-ignore[61]
                    dV_orig[_diag_src_idx] = dV_buf[_diag_dst_idx]
                if ki not in _dK_written:
                    dK_list[ki].copy_(dK_orig)
                else:
                    dK_list[ki] += dK_orig
                if ki not in _dV_written:
                    dV_list[ki].copy_(dV_orig)
                else:
                    dV_list[ki] += dV_orig
            _dK_written.add(ki)
            _dV_written.add(ki)

        # Kernel writes dQ directly via TMA reduce-add
        if _qi_dQ_wrote:
            _dQ_postprocessed.add(qi)

    # Zero any unwritten entries (NULL mask pairs)
    for qi in range(len(Q_list)):
        if qi not in _dQ_postprocessed:
            dQ_list[qi].zero_()
    for ki in range(len(K_list)):
        if ki not in _dK_written:
            dK_list[ki].zero_()
        if ki not in _dV_written:
            dV_list[ki].zero_()

    if dim_q < dim_q_padded:
        dQ_list = [dq[:, :, :dim_q].contiguous() for dq in dQ_list]
        dK_list = [dk[:, :, :dim_q].contiguous() for dk in dK_list]
    if dim_v < dim_v_padded:
        dV_list = [dv[:, :, :dim_v].contiguous() for dv in dV_list]

    return dQ_list, dK_list, dV_list


# Cache for compiled Blackwell backward kernels
_compiled_kernel_cache_bwd_blackwell: Dict[Tuple, object] = {}


def _cutedsl_mha_blackwell_backward(
    Q_list: List[torch.Tensor],
    K_list: List[torch.Tensor],
    V_list: List[torch.Tensor],
    dO_list: List[torch.Tensor],
    q_seq_offsets: torch.Tensor,
    kv_seq_offsets: torch.Tensor,
    attn_scale_list: List[torch.Tensor],
    mask_matrix: List[List["MaskType"]],
    alpha: float = 1.0,
    max_attn_len: Optional[int] = None,
    q_seq_offsets_cpu: Optional[List[torch.Tensor]] = None,
    kv_seq_offsets_cpu: Optional[List[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Blackwell (SM100) attention backward."""
    from hammer.v3.ops.cutedsl.cutedsl_attention_bwd import FlashAttentionBackwardSm100
    from hammer.v3.ops.cutedsl.fa4_helpers.flash_bwd_postprocess import (
        FlashAttentionBackwardPostprocess,
    )

    dtype = Q_list[0].dtype
    device = Q_list[0].device
    H = Q_list[0].shape[1]
    dim_q = Q_list[0].shape[2]
    dim_v = V_list[0].shape[2]
    B = q_seq_offsets.shape[1] - 1

    dim_padded = max((dim_q + 31) // 32 * 32, (dim_v + 31) // 32 * 32)
    dim_q_padded = dim_padded
    dim_v_padded = dim_padded

    if dim_q < dim_q_padded:
        Q_list = [torch.nn.functional.pad(q, (0, dim_q_padded - dim_q)) for q in Q_list]
        K_list = [torch.nn.functional.pad(k, (0, dim_q_padded - dim_q)) for k in K_list]
    if dim_v < dim_v_padded:
        V_list = [torch.nn.functional.pad(v, (0, dim_v_padded - dim_v)) for v in V_list]
        dO_list = [
            torch.nn.functional.pad(do, (0, dim_v_padded - dim_v)) for do in dO_list
        ]

    stream_handle = torch.cuda.current_stream().cuda_stream
    # pyre-ignore[16]
    cu_stream = cuda.CUstream(stream_handle)

    dQ_list = [torch.empty_like(q) for q in Q_list]
    dK_list = [torch.empty_like(k) for k in K_list]
    dV_list = [torch.empty_like(v) for v in V_list]

    cu_seqlens_q_list = [
        q_seq_offsets[qi].to(torch.int32).contiguous() for qi in range(len(Q_list))
    ]
    cu_seqlens_k_list = [
        kv_seq_offsets[ki].to(torch.int32).contiguous() for ki in range(len(K_list))
    ]

    m_block_size = 128
    softmax_scale = alpha

    _dK_written = set()
    _dV_written = set()
    _dQaccum_used = {}
    _dQ_postprocessed = set()

    for qi in range(len(Q_list)):
        _qi_dQaccum = None
        _qi_needs_postprocess = False
        _qi_LSE = None
        _qi_dPsum = None

        for ki in range(len(K_list)):
            mask_type = mask_matrix[qi][ki]
            if mask_type == MaskType.NULL:
                continue

            is_causal = mask_type in (
                MaskType.CAUSAL,
                MaskType.DIAGONAL,
                MaskType.LOCAL,
            )
            is_local = mask_type == MaskType.LOCAL
            window_size_left = None
            window_size_right = None
            if mask_type == MaskType.DIAGONAL:
                window_size_left = 0
                window_size_right = 0
            elif mask_type == MaskType.LOCAL:
                assert max_attn_len is not None
                window_size_left = max_attn_len - 1
                window_size_right = 0

            Q = Q_list[qi]
            K = K_list[ki]
            V = V_list[ki]
            dO = dO_list[qi]
            cu_seqlens_q = cu_seqlens_q_list[qi]
            cu_seqlens_k = cu_seqlens_k_list[ki]

            _diag_rearrange = False
            _diag_q_offsets_cpu = None
            _diag_k_offsets_cpu = None
            if mask_type == MaskType.DIAGONAL:
                _diag_q_offsets_cpu = (
                    q_seq_offsets_cpu[qi]
                    if q_seq_offsets_cpu is not None
                    else cu_seqlens_q.cpu()
                )
                _diag_k_offsets_cpu = (
                    kv_seq_offsets_cpu[ki]
                    if kv_seq_offsets_cpu is not None
                    else cu_seqlens_k.cpu()
                )
                q_lens = _diag_q_offsets_cpu[1:] - _diag_q_offsets_cpu[:-1]
                k_lens = _diag_k_offsets_cpu[1:] - _diag_k_offsets_cpu[:-1]
                max_delta = int((q_lens - k_lens).abs().max())
                if max_delta >= m_block_size:
                    _diag_rearrange = True
                    K_diag = torch.zeros(
                        Q.shape[0], K.shape[1], K.shape[2], dtype=K.dtype, device=device
                    )
                    V_diag = torch.zeros(
                        Q.shape[0], V.shape[1], V.shape[2], dtype=V.dtype, device=device
                    )
                    eff_lens = torch.minimum(q_lens, k_lens).long()
                    total_eff = int(eff_lens.sum().item())
                    if total_eff > 0:
                        q_starts = _diag_q_offsets_cpu[:-1].long()
                        k_starts = _diag_k_offsets_cpu[:-1].long()
                        k_base = torch.repeat_interleave(k_starts, eff_lens)
                        q_base = torch.repeat_interleave(q_starts, eff_lens)
                        eff_cs = torch.zeros(B + 1, dtype=torch.long)
                        eff_cs[1:] = eff_lens.cumsum(0)
                        within = torch.arange(total_eff) - torch.repeat_interleave(
                            eff_cs[:-1], eff_lens
                        )
                        _diag_src_idx = (k_base + within).to(device)
                        _diag_dst_idx = (q_base + within).to(device)
                        K_diag[_diag_dst_idx] = K[_diag_src_idx]
                        V_diag[_diag_dst_idx] = V[_diag_src_idx]
                    else:
                        _diag_src_idx = None
                        _diag_dst_idx = None
                    K = K_diag
                    V = V_diag
                    cu_seqlens_k = cu_seqlens_q

            total_q = Q.shape[0]

            # dQaccum: (H, total_q_padded * dim_q_padded)
            total_q_padded = (
                (total_q + cu_seqlens_q.shape[0] * m_block_size - 1)
                // m_block_size
                * m_block_size
            )

            # dQaccum: allocate for first ki, reuse for subsequent ki
            if _qi_dQaccum is None:
                dQaccum_shape = (H, total_q_padded * dim_q_padded)
                dQaccum_key = ("dQaccum", device)
                if (
                    dQaccum_key in _compiled_kernel_cache_bwd_blackwell
                    # pyre-ignore[16]
                    and _compiled_kernel_cache_bwd_blackwell[dQaccum_key].shape
                    == dQaccum_shape
                ):
                    dQaccum = _compiled_kernel_cache_bwd_blackwell[dQaccum_key]
                    if dQaccum_key in _dQaccum_used:
                        # pyre-ignore[16]
                        dQaccum.zero_()
                else:
                    dQaccum = torch.zeros(
                        *dQaccum_shape, dtype=torch.float32, device=device
                    )
                    _compiled_kernel_cache_bwd_blackwell[dQaccum_key] = dQaccum
                _qi_dQaccum = dQaccum
            else:
                dQaccum = _qi_dQaccum

            LSE_shape = (H, total_q_padded)
            if _qi_LSE is None or _qi_LSE.shape != LSE_shape:
                _qi_LSE = torch.empty(LSE_shape, dtype=torch.float32, device=device)
                _qi_dPsum = torch.empty(
                    H, total_q_padded, dtype=torch.float32, device=device
                )
            LSE = _qi_LSE
            dPsum = _qi_dPsum

            if ki not in _dK_written and not _diag_rearrange:
                dK_buf = dK_list[ki]
            else:
                dK_buf = torch.empty_like(K)
            if ki not in _dV_written and not _diag_rearrange:
                dV_buf = dV_list[ki]
            else:
                dV_buf = torch.empty_like(V)

            # Cute tensors
            q_cute = from_dlpack(Q.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=Q.ndim - 1
            )
            k_cute = from_dlpack(K.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=K.ndim - 1
            )
            v_cute = from_dlpack(V.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=V.ndim - 1
            )
            do_cute = from_dlpack(dO.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=dO.ndim - 1
            )
            lse_cute = from_dlpack(LSE, assumed_align=16).mark_layout_dynamic(
                leading_dim=LSE.ndim - 1
            )
            dpsum_cute = from_dlpack(dPsum, assumed_align=16).mark_layout_dynamic(
                # pyre-ignore[16]
                leading_dim=dPsum.ndim - 1
            )
            dqaccum_cute = from_dlpack(dQaccum, assumed_align=16).mark_layout_dynamic(
                # pyre-ignore[16]
                leading_dim=dQaccum.ndim - 1
            )
            dk_cute = from_dlpack(dK_buf, assumed_align=16).mark_layout_dynamic(
                leading_dim=dK_buf.ndim - 1
            )
            dv_cute = from_dlpack(dV_buf, assumed_align=16).mark_layout_dynamic(
                leading_dim=dV_buf.ndim - 1
            )
            cu_seqlens_q_cute = from_dlpack(
                cu_seqlens_q, assumed_align=4
            ).mark_layout_dynamic(leading_dim=0)
            cu_seqlens_k_cute = from_dlpack(
                cu_seqlens_k, assumed_align=4
            ).mark_layout_dynamic(leading_dim=0)
            attn_scale_cute = from_dlpack(
                attn_scale_list[qi].to(torch.float32).contiguous(), assumed_align=4
            ).mark_layout_dynamic(leading_dim=0)

            # BWD kernel — use persistent scheduling for non-causal
            _use_persistent = not is_causal and not is_local
            bwd_key = (
                "bwd",
                dtype,
                dim_q_padded,
                dim_v_padded,
                H,
                B,
                _next_power_of_2(Q.shape[0]),
                _next_power_of_2(K.shape[0]),
                is_causal,
                is_local,
                window_size_left,
                _use_persistent,
            )

            # Build tile mapping tables for persistent varlen scheduler
            if _use_persistent:
                _tile_key = ("tile_map", K.shape[0], H, device)
                if _tile_key not in _compiled_kernel_cache_bwd_blackwell:
                    # Compute on CPU
                    _cpu_csl = cu_seqlens_k.cpu()
                    _k_lens = _cpu_csl[1:] - _cpu_csl[:-1]
                    _n_blocks = (_k_lens + m_block_size - 1) // m_block_size
                    _tiles_per_batch = H * _n_blocks
                    _total_tiles = int(_tiles_per_batch.sum().item())
                    # Block-innermost ordering
                    _batch_ids = torch.repeat_interleave(
                        torch.arange(B, dtype=torch.int32),
                        _tiles_per_batch.to(torch.int64),
                    )
                    _pos_in_batch = torch.cat(
                        [
                            torch.arange(int(t), dtype=torch.int32)
                            for t in _tiles_per_batch
                        ]
                    )
                    _head_ids = _pos_in_batch // _n_blocks.repeat_interleave(
                        _tiles_per_batch.to(torch.int64)
                    ).to(torch.int32)
                    _block_ids = _pos_in_batch % _n_blocks.repeat_interleave(
                        _tiles_per_batch.to(torch.int64)
                    ).to(torch.int32)
                    # Pad to grid-aligned size
                    _num_sms = 148  # B200
                    _grid = min(_num_sms, _total_tiles)
                    _padded = (_total_tiles + _grid - 1) // _grid * _grid
                    if _padded > _total_tiles:
                        _pad_n = _padded - _total_tiles
                        _batch_ids = torch.cat(
                            [
                                _batch_ids,
                                torch.full((_pad_n,), B, dtype=torch.int32),
                            ]
                        )
                        _head_ids = torch.cat(
                            [
                                _head_ids,
                                torch.zeros(_pad_n, dtype=torch.int32),
                            ]
                        )
                        _block_ids = torch.cat(
                            [
                                _block_ids,
                                torch.zeros(_pad_n, dtype=torch.int32),
                            ]
                        )
                    _compiled_kernel_cache_bwd_blackwell[_tile_key] = (
                        _batch_ids.to(device=device),
                        _head_ids.to(device=device),
                        _block_ids.to(device=device),
                    )
                # pyre-ignore[23]
                _ttb, _tth, _ttbl = _compiled_kernel_cache_bwd_blackwell[_tile_key]
                _ttb_cute = from_dlpack(_ttb, assumed_align=4).mark_layout_dynamic(
                    leading_dim=0
                )
                _tth_cute = from_dlpack(_tth, assumed_align=4).mark_layout_dynamic(
                    leading_dim=0
                )
                _ttbl_cute = from_dlpack(_ttbl, assumed_align=4).mark_layout_dynamic(
                    leading_dim=0
                )
            else:
                _ttb_cute = None
                _tth_cute = None
                _ttbl_cute = None

            if bwd_key not in _compiled_kernel_cache_bwd_blackwell:
                fa_bwd = FlashAttentionBackwardSm100(
                    head_dim=dim_q_padded,
                    head_dim_v=dim_v_padded,
                    is_causal=is_causal,
                    is_local=is_local,
                    is_persistent=_use_persistent,
                    deterministic=False,
                    blockscaled=False,
                    use_silu=True,
                )
                _compiled_kernel_cache_bwd_blackwell[bwd_key] = cute.compile(
                    fa_bwd,
                    q_cute,
                    k_cute,
                    v_cute,
                    do_cute,
                    lse_cute,
                    dpsum_cute,
                    dqaccum_cute,
                    dk_cute,
                    dv_cute,
                    softmax_scale,
                    cu_stream,
                    cu_seqlens_q_cute,
                    cu_seqlens_k_cute,
                    None,
                    None,
                    None,  # mSeqUsedQ, mSeqUsedK, softcap
                    window_size_left,
                    window_size_right,
                    mAttnScale=attn_scale_cute,
                    mTileToBatch=_ttb_cute,
                    mTileToHead=_tth_cute,
                    mTileToBlock=_ttbl_cute,
                    options="--opt-level 2",
                )
            _compiled_kernel_cache_bwd_blackwell[bwd_key](  # pyre-ignore[29]
                q_cute,
                k_cute,
                v_cute,
                do_cute,
                lse_cute,
                dpsum_cute,
                dqaccum_cute,
                dk_cute,
                dv_cute,
                softmax_scale,
                cu_stream,
                cu_seqlens_q_cute,
                cu_seqlens_k_cute,
                None,
                None,
                None,
                window_size_left,
                window_size_right,
                mAttnScale=attn_scale_cute,
                mTileToBatch=_ttb_cute,
                mTileToHead=_tth_cute,
                mTileToBlock=_ttbl_cute,
            )

            _qi_needs_postprocess = True

            # DIAGONAL: rearrange dK/dV from Q's layout back to K's layout
            if _diag_rearrange:
                dK_orig = torch.zeros_like(K_list[ki])
                dV_orig = torch.zeros_like(V_list[ki])
                # pyre-ignore[61]
                if _diag_src_idx is not None and _diag_dst_idx is not None:
                    dK_orig[_diag_src_idx] = dK_buf[_diag_dst_idx]  # pyre-ignore[61]
                    dV_orig[_diag_src_idx] = dV_buf[_diag_dst_idx]  # pyre-ignore[61]
                dK_buf = dK_orig
                dV_buf = dV_orig

            if ki not in _dK_written:
                _dK_written.add(ki)
                if _diag_rearrange:
                    dK_list[ki].copy_(dK_buf)
            else:
                dK_list[ki] += dK_buf
            if ki not in _dV_written:
                _dV_written.add(ki)
                if _diag_rearrange:
                    dV_list[ki].copy_(dV_buf)
            else:
                dV_list[ki] += dV_buf

        # Deferred postprocess: run once per qi after all ki pairs
        if _qi_needs_postprocess:
            dQ_buf = dQ_list[qi]
            dQaccum = _qi_dQaccum
            cu_seqlens_q = cu_seqlens_q_list[qi]

            dqaccum_cute = from_dlpack(dQaccum, assumed_align=16).mark_layout_dynamic(
                # pyre-ignore[16]
                leading_dim=dQaccum.ndim - 1
            )
            dq_cute = from_dlpack(dQ_buf, assumed_align=16).mark_layout_dynamic(
                leading_dim=dQ_buf.ndim - 1
            )
            cu_seqlens_q_cute = from_dlpack(
                cu_seqlens_q, assumed_align=4
            ).mark_layout_dynamic(leading_dim=0)

            post_key = (
                "post",
                dtype,
                dim_q_padded,
                _next_power_of_2(Q_list[qi].shape[0]),
            )
            if post_key not in _compiled_kernel_cache_bwd_blackwell:
                fa_bwd_post = FlashAttentionBackwardPostprocess(
                    cutlass.BFloat16,
                    dim_q_padded,
                    arch=100,
                    num_threads=128,
                )
                _compiled_kernel_cache_bwd_blackwell[post_key] = cute.compile(
                    fa_bwd_post,
                    dqaccum_cute,
                    dq_cute,
                    softmax_scale,
                    cu_seqlens_q_cute,
                    None,
                    cu_stream,
                    options="--opt-level 2",
                )
            _compiled_kernel_cache_bwd_blackwell[post_key](  # pyre-ignore[29]
                dqaccum_cute,
                dq_cute,
                softmax_scale,
                cu_seqlens_q_cute,
                None,
                cu_stream,
            )

            _dQ_postprocessed.add(qi)
            dQaccum_key = ("dQaccum", device)
            _dQaccum_used[dQaccum_key] = True

    # Zero any unwritten entries (NULL mask pairs)
    for qi in range(len(Q_list)):
        if qi not in _dQ_postprocessed:
            dQ_list[qi].zero_()
    for ki in range(len(K_list)):
        if ki not in _dK_written:
            dK_list[ki].zero_()
        if ki not in _dV_written:
            dV_list[ki].zero_()

    # Zero cached dQaccum for next backward call
    for key in _dQaccum_used:
        _compiled_kernel_cache_bwd_blackwell[key].zero_()

    if dim_q < dim_q_padded:
        dQ_list = [dq[:, :, :dim_q].contiguous() for dq in dQ_list]
        dK_list = [dk[:, :, :dim_q].contiguous() for dk in dK_list]
    if dim_v < dim_v_padded:
        dV_list = [dv[:, :, :dim_v].contiguous() for dv in dV_list]

    return dQ_list, dK_list, dV_list


class _CuteDSLBlockedMHAFunction(torch.autograd.Function):
    """Autograd function for cutedsl blocked MHA (Blackwell)."""

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        alpha: float,
        max_attn_len: int,
        mask_matrix_tuple: Tuple[int, ...],
        num_q_tensors: int,
        num_kv_tensors: int,
        q_seq_offsets: torch.Tensor,
        kv_seq_offsets: torch.Tensor,
        *tensors,
    ) -> Tuple[torch.Tensor, ...]:
        q_list = list(tensors[:num_q_tensors])
        k_list = list(tensors[num_q_tensors : num_q_tensors + num_kv_tensors])
        v_list = list(
            tensors[num_q_tensors + num_kv_tensors : num_q_tensors + 2 * num_kv_tensors]
        )
        attn_scale_list = list(
            tensors[
                num_q_tensors + 2 * num_kv_tensors : 2 * num_q_tensors
                + 2 * num_kv_tensors
            ]
        )

        mask_matrix = [
            [MaskType(m) for m in row]
            for row in [
                mask_matrix_tuple[i * num_kv_tensors : (i + 1) * num_kv_tensors]
                for i in range(num_q_tensors)
            ]
        ]

        out_list = _cutedsl_mha_blackwell(
            Q_list=q_list,
            K_list=k_list,
            V_list=v_list,
            q_seq_offsets=q_seq_offsets,
            kv_seq_offsets=kv_seq_offsets,
            attn_scale_list=attn_scale_list,
            mask_matrix=mask_matrix,
            alpha=alpha,
            max_attn_len=max_attn_len if max_attn_len > 0 else None,
        )

        # Save for backward
        ctx.save_for_backward(
            *q_list,
            *k_list,
            *v_list,
            *attn_scale_list,
            q_seq_offsets,
            kv_seq_offsets,
        )
        ctx.alpha = alpha
        ctx.max_attn_len = max_attn_len
        ctx.mask_matrix_tuple = mask_matrix_tuple
        ctx.num_q_tensors = num_q_tensors
        ctx.num_kv_tensors = num_kv_tensors
        ctx.q_seq_offsets_cpu = [
            q_seq_offsets[i].to(torch.int32).cpu() for i in range(num_q_tensors)
        ]
        ctx.kv_seq_offsets_cpu = [
            kv_seq_offsets[i].to(torch.int32).cpu() for i in range(num_kv_tensors)
        ]

        return tuple(out_list)

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        num_q = ctx.num_q_tensors
        num_kv = ctx.num_kv_tensors

        # Unpack saved tensors
        saved = ctx.saved_tensors
        q_list = list(saved[:num_q])
        k_list = list(saved[num_q : num_q + num_kv])
        v_list = list(saved[num_q + num_kv : num_q + 2 * num_kv])
        attn_scale_list = list(saved[num_q + 2 * num_kv : 2 * num_q + 2 * num_kv])
        q_seq_offsets = saved[2 * num_q + 2 * num_kv]
        kv_seq_offsets = saved[2 * num_q + 2 * num_kv + 1]

        mask_matrix = [
            [MaskType(m) for m in row]
            for row in [
                ctx.mask_matrix_tuple[i * num_kv : (i + 1) * num_kv]
                for i in range(num_q)
            ]
        ]

        dout_list = list(grad_outputs)
        alpha = ctx.alpha
        max_attn_len = ctx.max_attn_len

        dq_list, dk_list, dv_list = _cutedsl_mha_blackwell_backward(
            Q_list=q_list,
            K_list=k_list,
            V_list=v_list,
            dO_list=dout_list,
            q_seq_offsets=q_seq_offsets,
            kv_seq_offsets=kv_seq_offsets,
            attn_scale_list=attn_scale_list,
            mask_matrix=mask_matrix,
            alpha=alpha,
            max_attn_len=max_attn_len if max_attn_len > 0 else None,
            q_seq_offsets_cpu=ctx.q_seq_offsets_cpu,
            kv_seq_offsets_cpu=ctx.kv_seq_offsets_cpu,
        )

        # pyre-ignore[60]
        return (
            None,  # alpha
            None,  # max_attn_len
            None,  # mask_matrix_tuple
            None,  # num_q_tensors
            None,  # num_kv_tensors
            None,  # q_seq_offsets
            None,  # kv_seq_offsets
            *dq_list,  # dQ
            *dk_list,  # dK
            *dv_list,  # dV
            *[None] * num_q,  # attn_scale (no grad)
        )


class _CuteDSLBlockedMHAFunctionHopper(torch.autograd.Function):
    """Autograd function for cutedsl blocked MHA (Hopper)."""

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        alpha: float,
        max_attn_len: int,
        mask_matrix_tuple: Tuple[int, ...],
        num_q_tensors: int,
        num_kv_tensors: int,
        q_seq_offsets: torch.Tensor,
        kv_seq_offsets: torch.Tensor,
        *tensors,
    ) -> Tuple[torch.Tensor, ...]:
        q_list = list(tensors[:num_q_tensors])
        k_list = list(tensors[num_q_tensors : num_q_tensors + num_kv_tensors])
        v_list = list(
            tensors[num_q_tensors + num_kv_tensors : num_q_tensors + 2 * num_kv_tensors]
        )
        attn_scale_list = list(
            tensors[
                num_q_tensors + 2 * num_kv_tensors : 2 * num_q_tensors
                + 2 * num_kv_tensors
            ]
        )

        mask_matrix = [
            [MaskType(m) for m in row]
            for row in [
                mask_matrix_tuple[i * num_kv_tensors : (i + 1) * num_kv_tensors]
                for i in range(num_q_tensors)
            ]
        ]

        out_list = _cutedsl_mha_hopper(
            Q_list=q_list,
            K_list=k_list,
            V_list=v_list,
            q_seq_offsets=q_seq_offsets,
            kv_seq_offsets=kv_seq_offsets,
            attn_scale_list=attn_scale_list,
            mask_matrix=mask_matrix,
            alpha=alpha,
            max_attn_len=max_attn_len if max_attn_len > 0 else None,
        )

        ctx.save_for_backward(
            *q_list,
            *k_list,
            *v_list,
            *attn_scale_list,
            q_seq_offsets,
            kv_seq_offsets,
        )
        ctx.alpha = alpha
        ctx.max_attn_len = max_attn_len
        ctx.mask_matrix_tuple = mask_matrix_tuple
        ctx.num_q_tensors = num_q_tensors
        ctx.num_kv_tensors = num_kv_tensors

        return tuple(out_list)

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        num_q = ctx.num_q_tensors
        num_kv = ctx.num_kv_tensors

        saved = ctx.saved_tensors
        q_list = list(saved[:num_q])
        k_list = list(saved[num_q : num_q + num_kv])
        v_list = list(saved[num_q + num_kv : num_q + 2 * num_kv])
        attn_scale_list = list(saved[num_q + 2 * num_kv : 2 * num_q + 2 * num_kv])
        q_seq_offsets = saved[2 * num_q + 2 * num_kv]
        kv_seq_offsets = saved[2 * num_q + 2 * num_kv + 1]

        mask_matrix = [
            [MaskType(m) for m in row]
            for row in [
                ctx.mask_matrix_tuple[i * num_kv : (i + 1) * num_kv]
                for i in range(num_q)
            ]
        ]

        dout_list = list(grad_outputs)
        alpha = ctx.alpha
        max_attn_len = ctx.max_attn_len

        dq_list, dk_list, dv_list = _cutedsl_mha_hopper_backward(
            Q_list=q_list,
            K_list=k_list,
            V_list=v_list,
            dO_list=dout_list,
            q_seq_offsets=q_seq_offsets,
            kv_seq_offsets=kv_seq_offsets,
            attn_scale_list=attn_scale_list,
            mask_matrix=mask_matrix,
            alpha=alpha,
            max_attn_len=max_attn_len if max_attn_len > 0 else None,
        )

        # pyre-ignore[60]
        return (
            None,  # alpha
            None,  # max_attn_len
            None,  # mask_matrix_tuple
            None,  # num_q_tensors
            None,  # num_kv_tensors
            None,  # q_seq_offsets
            None,  # kv_seq_offsets
            *dq_list,  # dQ
            *dk_list,  # dK
            *dv_list,  # dV
            *[None] * num_q,  # attn_scale (no grad)
        )


def cutedsl_mha(
    Q_list: List[torch.Tensor],
    K_list: List[torch.Tensor],
    V_list: List[torch.Tensor],
    q_seq_offsets: torch.Tensor,
    kv_seq_offsets: torch.Tensor,
    attn_scale_list: List[torch.Tensor],
    mask_matrix: List[List[MaskType]],
    alpha: float = 1.0,
    max_attn_len: Optional[int] = None,
) -> List[torch.Tensor]:
    assert len(Q_list) > 0, "Q_list must not be empty"
    assert len(K_list) > 0, "K_list must not be empty"
    assert len(V_list) == len(K_list), "K_list and V_list must have same length"

    # Dispatch to Blackwell (SM100) or Hopper (SM90) kernel
    if is_sm100_plus():
        requires_grad = any(t.requires_grad for t in Q_list + K_list + V_list)
        if requires_grad:
            num_q_tensors = len(Q_list)
            num_kv_tensors = len(K_list)
            mask_matrix_tuple = tuple(m.value for row in mask_matrix for m in row)
            # pyre-ignore[60]
            tensors = (
                *Q_list,
                *K_list,
                *V_list,
                *attn_scale_list,
            )
            out_tuple = _CuteDSLBlockedMHAFunction.apply(
                alpha,
                max_attn_len if max_attn_len is not None else 0,
                mask_matrix_tuple,
                num_q_tensors,
                num_kv_tensors,
                q_seq_offsets,
                kv_seq_offsets,
                *tensors,
            )
            return list(out_tuple)
        else:
            return _cutedsl_mha_blackwell(
                Q_list,
                K_list,
                V_list,
                q_seq_offsets,
                kv_seq_offsets,
                attn_scale_list,
                mask_matrix,
                alpha,
                max_attn_len,
            )
    elif not is_sm90():
        raise RuntimeError(
            "CuTeDSL attention kernel requires Hopper (SM90) or Blackwell (SM100+)."
        )

    requires_grad = any(t.requires_grad for t in Q_list + K_list + V_list)
    if requires_grad:
        num_q_tensors = len(Q_list)
        num_kv_tensors = len(K_list)
        mask_matrix_tuple = tuple(m.value for row in mask_matrix for m in row)
        # pyre-ignore[60]
        tensors = (
            *Q_list,
            *K_list,
            *V_list,
            *attn_scale_list,
        )
        out_tuple = _CuteDSLBlockedMHAFunctionHopper.apply(
            alpha,
            max_attn_len if max_attn_len is not None else 0,
            mask_matrix_tuple,
            num_q_tensors,
            num_kv_tensors,
            q_seq_offsets,
            kv_seq_offsets,
            *tensors,
        )
        return list(out_tuple)

    return _cutedsl_mha_hopper(
        Q_list,
        K_list,
        V_list,
        q_seq_offsets,
        kv_seq_offsets,
        attn_scale_list,
        mask_matrix,
        alpha,
        max_attn_len,
    )
