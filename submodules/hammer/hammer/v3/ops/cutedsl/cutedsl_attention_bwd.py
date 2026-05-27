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

# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.

# pyre-unsafe

import enum
import math
from functools import partial
from typing import Callable, Optional, Type

# pyre-ignore[21]
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
import hammer.v3.ops.cutedsl.fa4_helpers.pipeline as pipeline  # pyre-ignore
import hammer.v3.ops.cutedsl.fa4_helpers.utils as utils
from cutlass import Boolean, const_expr, Float32, Int32, Uint32, Uint8
from cutlass._mlir.dialects import llvm  # pyre-ignore
from cutlass.cute.arch import ProxyKind, SharedSpace
from cutlass.cute.nvgpu import cpasync, tcgen05, warpgroup
from cutlass.cutlass_dsl import dsl_user_op, T  # pyre-ignore
from cutlass.pipeline import PipelineAsync, PipelineConsumer
from cutlass.utils import LayoutEnum
from hammer.v3.ops.cutedsl.fa4_helpers import (
    blackwell_helpers as sm100_utils,
    copy_utils,
    hopper_helpers as sm90_utils,
)
from hammer.v3.ops.cutedsl.fa4_helpers.blackwell_helpers import (  # noqa
    gemm_blockscaled,
    gemm_ptx_w_idx,
    gemm_w_idx,
    make_s2t_copy_partitions,
)
from hammer.v3.ops.cutedsl.fa4_helpers.block_info import BlockInfo
from hammer.v3.ops.cutedsl.fa4_helpers.hopper_helpers import (
    gemm_w_idx as gemm_w_idx_sm90,
    gemm_zero_init,
)
from hammer.v3.ops.cutedsl.fa4_helpers.mask import AttentionMask
from hammer.v3.ops.cutedsl.fa4_helpers.named_barrier import (
    NamedBarrierBwd,
    NamedBarrierBwdSm100,
    NamedBarrierFwd,
)
from hammer.v3.ops.cutedsl.fa4_helpers.seqlen_info import SeqlenInfoQK
from hammer.v3.ops.cutedsl.fa4_helpers.softmax import (
    abs_f32,
    E4M3_MAX_NORM_RCP,
    fused_amax_to_e8m0_scale_f32,
    max_f32,
    redux_sync_max_abs_f32,
)
from hammer.v3.ops.cutedsl.fa4_helpers.tile_scheduler import (
    ParamsBase,
    PersistentVarlenLookupScheduler,
    SingleTileLPTBwdScheduler,  # noqa
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    StaticPersistentTileScheduler,
    TileSchedulerArguments,
)


@dsl_user_op
def _ld_acquire(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    # pyre-ignore[16]
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(state)


@dsl_user_op
def _red_release(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    # pyre-ignore[16]
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _wait_eq(
    lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: Int32
) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        read_val = Int32(0)
        while read_val != val:
            read_val = _ld_acquire(flag_ptr)


@cute.jit
def _arrive_inc(
    lock_ptr: cute.Pointer,
    thread_idx: int | Int32,
    flag_offset: int,
    val: cutlass.Constexpr[Int32],
) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        _red_release(flag_ptr, val)


class barrier:
    ld_acquire = staticmethod(_ld_acquire)
    red_release = staticmethod(_red_release)
    wait_eq = staticmethod(_wait_eq)
    arrive_inc = staticmethod(_arrive_inc)


class FlashAttentionBackwardSm100:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        # pyre-ignore[9]
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        is_persistent: bool = False,
        deterministic: bool = False,
        cluster_size: int = 1,
        blockscaled: bool = False,
        sf_vec_size: int = 32,
        output_mxfp8_dkv: bool = False,
        broadcast_q: bool = False,
        use_silu: bool = False,
    ):
        self.use_silu = use_silu
        if use_silu:
            blockscaled = False
            output_mxfp8_dkv = False

        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        assert head_dim == head_dim_v, (
            "head_dim and head_dim_v must be the same for now"
        )
        self.tile_hdimv = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        assert self.tile_hdim == self.tile_hdimv, (
            "tile_hdim and tile_hdimv must be the same for now"
        )
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.tile_m = tile_m
        self.tile_n = tile_n

        # CTA tiler
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (tile_n, tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (tile_n, tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (tile_n, self.tile_hdimv, tile_m)
        # dK = dS.T @ Q (N, M) (M, D)
        self.mma_tiler_dsq = (tile_n, self.tile_hdimv, tile_m)
        # dQ = dS @ K
        self.mma_tiler_dsk = (tile_m, self.tile_hdimv, tile_n)

        self.acc_dtype = Float32

        assert cluster_size in (1, 2), "Only cluster_size=1 or 2 is supported"
        self.cluster_shape_mn = (cluster_size, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = False
        self.deterministic = deterministic
        self.blockscaled = blockscaled
        self.sf_vec_size = sf_vec_size
        self.sf_dtype = cutlass.Float8E8M0FNU if blockscaled else None
        self.output_mxfp8_dkv = output_mxfp8_dkv
        self.broadcast_q = broadcast_q

        self.shuffle_LSE = False
        self.shuffle_dPsum = False
        self.use_smem_dS_for_mma_dK = (
            self.deterministic and self.is_causal
        ) or self.use_silu

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.epi_warp_id = 14
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epi_warp_id,
                self.empty_warp_id,
            )
        )

        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )

        # TMEM setup
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.tmem_S_offset = 0
        self.tmem_P_offset = 0  # overlap with S
        self.tmem_dV_offset = self.tmem_S_offset + self.tile_n
        self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
        self.tmem_dQ_offset = self.tmem_dP_offset  # overlap with dP
        self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m
        self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP

        self.tmem_SF_prologue_offset = self.tmem_dK_offset
        self.tmem_SF_offset = self.tmem_dP_offset + 48  # 304 (S GEMM SFs in dP region)
        self.tmem_SF_offset_dP = (
            self.tmem_S_offset + 48
        )  # 48 (dP/dK/dV GEMM SFs in S region)

        self.tmem_SF_phase2_offset = self.tmem_dK_offset

        if (not is_causal and not is_local) or deterministic:
            self.num_regs_reduce = 152
            self.num_regs_compute = 136
        else:
            self.num_regs_reduce = 136
            self.num_regs_compute = 144
        self.num_regs_other = 96 - 8
        self.num_regs_empty = 24
        assert (
            self.num_regs_reduce + self.num_regs_compute * 2 + self.num_regs_other
            <= 512
        )

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.Q_stage = 2
        self.dO_stage = 1
        # number of tma reduce adds per dQacc mma
        self.dQ_reduce_ncol = 32
        self.sdQaccum_stage = 64 // self.dQ_reduce_ncol
        assert self.tile_hdim % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.cluster_reduce_dQ = False and cute.size(self.cluster_shape_mn) > 1
        # number of tma reduce adds for dKacc and dVacc epilogue
        self.dK_reduce_ncol = 32

    def _get_tiled_mma(self):
        cta_group = tcgen05.CtaGroup.ONE
        # S = K @ Q.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_kq[:2],
        )
        # dP = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_vdo[:2],
        )
        # dV += P @ dO --> (K, MN) major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # P_major_mode
            tcgen05.OperandMajorMode.MN,  # dO_major_mode
            self.acc_dtype,
            cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dK += dS.T @ Q
        if const_expr(self.use_smem_dS_for_mma_dK):
            mma_dK_a_src = tcgen05.OperandSource.SMEM
        else:
            mma_dK_a_src = tcgen05.OperandSource.TMEM
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Q_major_mode
            self.acc_dtype,
            cta_group,
            self.mma_tiler_dsq[:2],
            a_source=mma_dK_a_src,
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Kt_major_mode
            self.acc_dtype,
            cta_group,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_tiled_mma_blockscaled(self):
        """Create blockscaled tiled_mma objects for SMEM/TMA layouts."""
        cta_group = tcgen05.CtaGroup.ONE

        # S = K @ Q.T - blockscaled version for SF layouts
        tiled_mma_S_bs = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.mma_tiler_kq[:2],
        )

        # dP = V @ dO.T - blockscaled version
        tiled_mma_dP_bs = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.mma_tiler_vdo[:2],
        )

        # dV = P.T @ dO - blockscaled version (P from TMEM)
        tiled_mma_dV_bs = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # P_major_mode (transposed)
            tcgen05.OperandMajorMode.MN,  # dO_major_mode
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )

        # dK = dS.T @ Q - blockscaled version
        # dS_major_mode = K (transposed), Q_major_mode = MN
        # TMEM variant: used for SF layouts (sSFDS, tCtSFDS, etc.)
        tiled_mma_dK_bs = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # dS_major_mode (transposed)
            tcgen05.OperandMajorMode.MN,  # Q_major_mode
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.mma_tiler_dsq[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # SMEM variant: used for actual GEMM execution (reads dS from SMEM)
        # K-major A to match dS.T transposition
        tiled_mma_dK_bs_smem = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # dS_major_mode (K-major)
            tcgen05.OperandMajorMode.MN,  # Q_major_mode
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.mma_tiler_dsq[:2],
            a_source=tcgen05.OperandSource.SMEM,
        )

        # dQ = dS @ K - blockscaled version (dS from SMEM)
        # MN-major A: compute warp writes dS in M-major (ROW_MAJOR R2S), so
        # the UMMA must read M-contiguous data. SF covers 32 M-elements per group.
        tiled_mma_dQ_bs = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,  # dS_major_mode — MN-major to match ROW_MAJOR R2S
            tcgen05.OperandMajorMode.MN,  # K_major_mode
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.mma_tiler_dsk[:2],
        )

        return (
            tiled_mma_S_bs,
            tiled_mma_dP_bs,
            tiled_mma_dV_bs,
            tiled_mma_dK_bs,
            tiled_mma_dK_bs_smem,
            tiled_mma_dQ_bs,
        )

    def _setup_smem_layout(self) -> None:
        # S = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            # pyre-ignore[16]
            self.tiled_mma_S,
            self.mma_tiler_kq,
            # pyre-ignore[16]
            self.k_dtype,
            1,
        )
        # pyre-ignore[16]
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        # pyre-ignore[16]
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            # pyre-ignore[16]
            self.q_dtype,
            # pyre-ignore[16]
            self.Q_stage,
        )
        # dP = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            # pyre-ignore[16]
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            # pyre-ignore[16]
            self.v_dtype,
            1,
        )
        # pyre-ignore[16]
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        # pyre-ignore[16]
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            # pyre-ignore[16]
            self.do_dtype,
            # pyre-ignore[16]
            self.dO_stage,
        )
        # dV += P @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            # pyre-ignore[16]
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        # pyre-ignore[16]
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        # pyre-ignore[16]
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            # pyre-ignore[16]
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            # pyre-ignore[16]
            self.ds_dtype,
            1,
        )
        # pyre-ignore[16]
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        # pyre-ignore[16]
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        # pyre-ignore[16]
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            # pyre-ignore[16]
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        # pyre-ignore[16]
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        # pyre-ignore[16]
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        # Blockscaled dQ: separate layouts from tiled_mma_dQ_bs for fragment creation
        # sdS_dQ_layout: A operand for blockscaled dQ GEMM (MN-major A)
        # ROW_MAJOR R2S writes dS in M-major order, matching MN-major UMMA read.
        if const_expr(self.blockscaled):
            sdS_dQ_layout_full = sm100_utils_basic.make_smem_layout_a(
                # pyre-ignore[16]
                self.tiled_mma_dQ_bs,
                self.mma_tiler_dsk,
                self.ds_dtype,
                1,
            )
            # pyre-ignore[16]
            self.sdS_dQ_data_layout = cute.slice_(
                sdS_dQ_layout_full, (None, None, None, 0)
            )
            # B operand layout for dQ MMA (K stays MN-major)
            sKt_dQ_layout = sm100_utils_basic.make_smem_layout_b(
                self.tiled_mma_dQ_bs,
                self.mma_tiler_dsk,
                self.k_dtype,
                1,
            )
            # pyre-ignore[16]
            self.sKt_dQ_data_layout = cute.slice_(sKt_dQ_layout, (None, None, None, 0))
            # Blockscaled dK: separate SMEM layout for fragment A creation (K-major A)
            # dK reads dS.T from SMEM with blockscaled MMA (a_source=SMEM).
            # Must use the SMEM variant MMA to get correct swizzle pattern.
            sdS_dK_layout_full = sm100_utils_basic.make_smem_layout_a(
                # pyre-ignore[16]
                self.tiled_mma_dK_bs_smem,
                self.mma_tiler_dsq,
                self.ds_dtype,
                1,
            )
            # pyre-ignore[16]
            self.sdS_dK_data_layout = cute.slice_(
                sdS_dK_layout_full, (None, None, None, 0)
            )
        else:
            self.sdS_dQ_data_layout = None
            self.sKt_dQ_data_layout = None
            self.sdS_dK_data_layout = None
        # pyre-ignore[16]
        self.sdQaccum_layout = cute.make_layout(
            # pyre-ignore[16]
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
        )
        # pyre-ignore[16]
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        # pyre-ignore[16]
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        # pyre-ignore[16]
        self.sdKV_epi_tile = (
            self.tile_n,
            # pyre-ignore[16]
            min(128 // (self.dk_dtype.width // 8), self.tile_hdim // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        # headdim_64 gets 1 stage
        # pyre-ignore[16]
        self.num_epi_stages = max(1, (self.tile_hdim // 2) // self.sdKV_epi_tile[1])
        # pyre-ignore[16]
        self.sdKV_flat_epi_tile = (
            self.tile_n * (self.tile_hdim // 2) // self.num_epi_stages
        )
        # TODO: dK and dV could have different shapes
        # pyre-ignore[16]
        if const_expr(not self.dKV_postprocess):
            # pyre-ignore[16]
            self.sdKV_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dk_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdKV_epi_tile,
                2,  # num compute wgs
            )
        else:
            # pyre-ignore[16]
            self.sdKV_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))

        # SMEM layouts for scale factors (MXFP8 blockscaled)
        # pyre-ignore[16]
        self.sSFQ_layout = None
        # pyre-ignore[16]
        self.sSFK_layout = None
        # pyre-ignore[16]
        self.sSFV_layout = None
        # pyre-ignore[16]
        self.sSFDO_layout = None

    @staticmethod
    def _make_smem_u32_view(tensor):
        """Recast an SMEM SF tensor as a u32 view for byte-level packing."""
        ptr = cute.recast_ptr(tensor.iterator, dtype=cutlass.Uint32)
        filtered = cute.filter_zeros(tensor)
        grouped = cute.group_modes(filtered, 0, cute.rank(filtered.layout) - 1)
        layout = cute.recast_layout(32, 8, grouped.layout)
        return cute.make_tensor(ptr, layout)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        # pyre-ignore[11]
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        # MXFP8 scale factors
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        mSFV: Optional[cute.Tensor] = None,
        mSFDO: Optional[cute.Tensor] = None,
        mSFDO_dV: Optional[cute.Tensor] = None,  # Transposed dO scales for dV GEMM
        mdO_dV: Optional[cute.Tensor] = None,  # M-block quantized dO for dV GEMM
        mQ_dK: Optional[cute.Tensor] = None,  # M-block quantized Q for dK GEMM
        mK_dQ: Optional[cute.Tensor] = None,  # M-block quantized K for dQ GEMM
        mSFQ_dK: Optional[cute.Tensor] = None,  # M-block scales for Q in dK GEMM
        mSFK_dQ: Optional[cute.Tensor] = None,  # M-block scales for K in dQ GEMM
        mdSFK_out: Optional[cute.Tensor] = None,  # Output scale factors for dK (MXFP8)
        mdSFV_out: Optional[cute.Tensor] = None,  # Output scale factors for dV (MXFP8)
        mCuSeqlensO: Optional[cute.Tensor] = None,  # O addressing for broadcast_q
        # 128-aligned cu_seqlens for scale factor offsets (varlen MXFP8)
        mCuSeqlensSFQ: Optional[cute.Tensor] = None,
        mCuSeqlensSFK: Optional[cute.Tensor] = None,
        # Total padded tokens for SF layout (SF-only approach)
        total_sf_q: Int32 | int | None = None,
        total_sf_k: Int32 | int | None = None,
        # Persistent backward: grouped tile lookup tables
        mTileToBatch: Optional[cute.Tensor] = None,
        mTileToHead: Optional[cute.Tensor] = None,
        mTileToBlock: Optional[cute.Tensor] = None,
        mAttnScale: Optional[cute.Tensor] = None,  # SiLU: per-row attention scale
    ):
        # pyre-ignore[16]
        self.q_dtype = mQ.element_type
        # pyre-ignore[16]
        self.k_dtype = mK.element_type
        # pyre-ignore[16]
        self.v_dtype = mV.element_type
        # pyre-ignore[16]
        self.do_dtype = mdO.element_type
        # pyre-ignore[16]
        self.lse_dtype = mLSE.element_type
        # pyre-ignore[16]
        self.dpsum_dtype = mdPsum.element_type
        # pyre-ignore[16]
        self.dqaccum_dtype = mdQaccum.element_type
        # pyre-ignore[16]
        self.dk_dtype = mdK.element_type
        # pyre-ignore[16]
        self.dv_dtype = mdV.element_type
        # pyre-ignore[16]
        self.ds_dtype = self.q_dtype

        # pyre-ignore[16]
        self.is_varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None
        # pyre-ignore[16]
        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        # pyre-ignore[16]
        assert self.v_dtype.width == 16 or self.v_dtype.width == 8, (
            "Only support v_dtype.width = 16 or 8"
        )

        # pyre-ignore[16]
        self.use_tma_store = self.v_dtype.width != 8 and not (
            self.qhead_per_kvhead == 1 and mCuSeqlensK is not None
        )
        # pyre-ignore[16, 58]
        self.dKV_postprocess = self.qhead_per_kvhead > 1

        # pyre-ignore[58]
        if const_expr(self.qhead_per_kvhead > 1):
            assert self.dk_dtype.width == 32, (
                "Must accumulate dK in float precision for GQA"
            )
            assert self.dv_dtype.width == 32, (
                "Must accumulate dV in float precision for GQA"
            )

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        (
            mdQaccum,
            mdK,
            mdV,
        ) = [
            cute.make_tensor(
                t.iterator, cute.make_layout(t.shape, stride=new_stride(t))
            )
            if t is not None
            else None
            for t in (
                mdQaccum,
                mdK,
                mdV,
            )
        ]

        QO_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mQ, mdO = [utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]
        mK, mV = [utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]
        LSE_dPsum_dQaccum_transpose = (
            [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        )
        mLSE, mdPsum, mdQaccum = [
            utils.select(t, mode=LSE_dPsum_dQaccum_transpose)
            for t in (mLSE, mdPsum, mdQaccum)
        ]
        if const_expr(not self.dKV_postprocess):
            layout_dKV_transpose = KV_layout_transpose
        else:
            layout_dKV_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensK is None) else [1, 0]
            )
        mdK, mdV = [utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)]
        # Apply same transformations to output scale factor tensors (MXFP8 dK/dV)
        if const_expr(mdSFK_out is not None):
            # pyre-ignore[6]
            mdSFK_out = utils.select(mdSFK_out, mode=layout_dKV_transpose)
        else:
            mdSFK_out = None
        if const_expr(mdSFV_out is not None):
            # pyre-ignore[6]
            mdSFV_out = utils.select(mdSFV_out, mode=layout_dKV_transpose)
        else:
            mdSFV_out = None
        dO_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensQ is None) else [1, 0, 2]
        mdO = utils.select(mdO, mode=dO_transpose)

        # Apply same transformations to mdO_dV (M-block quantized dO for dV GEMM)
        if const_expr(mdO_dV is not None):
            # pyre-ignore[6]
            mdO_dV = utils.select(mdO_dV, mode=QO_layout_transpose)
            mdO_dV = utils.select(mdO_dV, mode=dO_transpose)

        # Apply same transformations to mQ_dK (M-block quantized Q for dK GEMM)
        if const_expr(mQ_dK is not None):
            # pyre-ignore[6]
            mQ_dK = utils.select(mQ_dK, mode=QO_layout_transpose)

        # Apply same transformations to mK_dQ (M-block quantized K for dQ GEMM)
        if const_expr(mK_dQ is not None):
            # pyre-ignore[6]
            mK_dQ = utils.select(mK_dQ, mode=KV_layout_transpose)

        semaphore_transpose = [
            2,
            3,
            1,
            0,
        ]  # (b, n, block, stage) -> (block, stage, n, b)
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = utils.select(mdQ_semaphore, mode=semaphore_transpose)

        # pyre-ignore[58]
        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                utils.select(t, mode=semaphore_transpose)
                for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self._setup_attributes()
        (
            # pyre-ignore[16]
            self.tiled_mma_S,
            # pyre-ignore[16]
            self.tiled_mma_dP,
            # pyre-ignore[16]
            self.tiled_mma_dK,
            # pyre-ignore[16]
            self.tiled_mma_dV,
            # pyre-ignore[16]
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()

        # For blockscaled mode, override MMAs with blockscaled versions
        if const_expr(self.blockscaled):
            # Save non-blockscaled MMAs for SMEM fragment creation
            # pyre-ignore[16]
            self.tiled_mma_dK_nonbs = self.tiled_mma_dK
            # pyre-ignore[16]
            self.tiled_mma_dV_nonbs = self.tiled_mma_dV
            # pyre-ignore[16]
            self.tiled_mma_dQ_nonbs = self.tiled_mma_dQ
            (
                # pyre-ignore[16]
                self.tiled_mma_S_bs,
                # pyre-ignore[16]
                self.tiled_mma_dP_bs,
                # pyre-ignore[16]
                self.tiled_mma_dV_bs,
                # pyre-ignore[16]
                self.tiled_mma_dK_bs,
                # pyre-ignore[16]
                self.tiled_mma_dK_bs_smem,
                # pyre-ignore[16]
                self.tiled_mma_dQ_bs,
            ) = self._get_tiled_mma_blockscaled()
            self.tiled_mma_S = self.tiled_mma_S_bs
            self.tiled_mma_dP = self.tiled_mma_dP_bs
            self.tiled_mma_dV = self.tiled_mma_dV_bs
            self.tiled_mma_dK = self.tiled_mma_dK_bs

        self._setup_smem_layout()

        # Create blockscaled SMEM layouts for scale factors (MXFP8)
        if const_expr(self.blockscaled):
            # SFQ: Q's scale factors (operand B for S = K @ Q.T)
            # pyre-ignore[16]
            self.sSFQ_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_S_bs,
                self.mma_tiler_kq,
                self.sf_vec_size,
                # pyre-ignore[16]
                self.Q_stage,
            )

            # SFK: K's scale factors (operand A for S = K @ Q.T)
            # pyre-ignore[16]
            self.sSFK_layout = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma_S_bs,
                self.mma_tiler_kq,
                self.sf_vec_size,
                1,  # K has 1 stage
            )

            # SFV: V's scale factors (operand A for dP = V @ dO.T)
            # pyre-ignore[16]
            self.sSFV_layout = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma_dP_bs,
                self.mma_tiler_vdo,
                self.sf_vec_size,
                1,  # V has 1 stage
            )

            # SFDO: dO's scale factors (operand B for dP = V @ dO.T)
            # pyre-ignore[16]
            self.sSFDO_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_dP_bs,
                self.mma_tiler_vdo,
                self.sf_vec_size,
                # pyre-ignore[16]
                self.dO_stage,
            )

            # SFP: P's scale factors (operand A for dV = P.T @ dO)
            # 2 stages for double-buffered P0/P1
            # pyre-ignore[16]
            self.sSFP_layout = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma_dV_bs,
                self.mma_tiler_pdo,
                self.sf_vec_size,
                2,  # 2 stages for double-buffering
            )

            # SFDS: dS's scale factors (operand A for dK = dS.T @ Q)
            # 2 stages for double-buffered dS0/dS1
            # Use TMEM MMA variant for SF layouts
            # pyre-ignore[16]
            self.sSFDS_layout = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma_dK_bs,
                self.mma_tiler_dsq,
                self.sf_vec_size,
                2,  # 2 stages for double-buffering
            )

            # SFQ_dK: Use dK GEMM params with TMEM MMA for SF layout
            # pyre-ignore[16]
            self.sSFQ_dK_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_dK_bs,
                self.mma_tiler_dsq,
                self.sf_vec_size,
                self.Q_stage,
            )

            # SFDO_dV: dO's scale factors for dV GEMM (operand B for dV = P.T @ dO)
            # Separate layout from sSFDO because dV uses tiled_mma_dV_bs, not tiled_mma_dP_bs
            # S2T copy requires matching SMEM/TMEM layouts
            # pyre-ignore[16]
            self.sSFDO_dV_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_dV_bs,
                self.mma_tiler_pdo,
                self.sf_vec_size,
                self.dO_stage,
            )

            # SFDS_dQ: dS's scale factors for dQ GEMM (operand A for dQ = dS @ K)
            # Separate layout from sSFDS because dQ uses tiled_mma_dQ_bs and mma_tiler_dsk,
            # not tiled_mma_dK_bs and mma_tiler_dsq
            # pyre-ignore[16]
            self.sSFDS_dQ_layout = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma_dQ_bs,
                self.mma_tiler_dsk,
                self.sf_vec_size,
                2,  # 2 stages for double-buffered dS0/dS1
            )

            # SFK_dQ: K's scale factors for dQ GEMM (operand B for dQ = dS @ K)
            # Separate layout from sSFK because dQ uses tiled_mma_dQ_bs and mma_tiler_dsk,
            # not tiled_mma_S_bs and mma_tiler_kq
            # pyre-ignore[16]
            self.sSFK_dQ_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_dQ_bs,
                self.mma_tiler_dsk,
                self.sf_vec_size,
                1,  # K has 1 stage
            )

        else:
            self.tiled_mma_dV_bs = None
            self.tiled_mma_dK_bs = None
            self.tiled_mma_dK_bs_smem = None
            self.tiled_mma_dQ_bs = None
            self.sSFP_layout = None
            self.sSFDS_layout = None
            self.sSFQ_dK_layout = None
            self.sSFDO_dV_layout = None
            self.sSFDS_dQ_layout = None
            self.sSFK_dQ_layout = None

        cta_group = tcgen05.CtaGroup.ONE

        # pyre-ignore[16, 60]
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        # pyre-ignore[16]
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        # pyre-ignore[16]
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        # pyre-ignore[16]
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(not self.dKV_postprocess):
            # pyre-ignore[16]
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            # pyre-ignore[16]
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        if const_expr(self.use_tma_store and not self.dKV_postprocess):
            tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                # pyre-ignore[16]
                cute.select(self.sdKV_layout, mode=[0, 1]),
                # pyre-ignore[16]
                self.sdKV_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(self.sdKV_layout, mode=[0, 1]),
                self.sdKV_epi_tile,
                1,  # no mcast
            )
        else:
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        if const_expr(not self.dKV_postprocess):
            thr_layout_r2s_dKV = cute.make_ordered_layout(
                (128, 1), order=(1, 0)
            )  # 128 threads
            val_layout_r2s_dKV = cute.make_ordered_layout(
                (1, 128 // self.dk_dtype.width), order=(1, 0)
            )  # 4 or 8 vals for 16 byte store
            copy_atom_r2s_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
                copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
            )
        else:
            tiled_copy_r2s_dKV = copy_utils.tiled_copy_1d(
                Float32, 128, num_copy_elems=128 // Float32.width
            )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_load_op_multicast = cpasync.CopyBulkTensorTileG2SMulticastOp(cta_group)

        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            # pyre-ignore[16]
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_tma_op,
            mQ,
            # pyre-ignore[16]
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            # pyre-ignore[16]
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op,
            mdO,
            # pyre-ignore[16]
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )

        # TMA atom for mdO_dV (M-block quantized dO for dV GEMM)
        tma_atom_dO_dV, tma_tensor_dO_dV = None, None
        if const_expr(self.blockscaled and mdO_dV is not None):
            # mdO_dV uses the same layout as mdO (sdO_layout)
            tma_atom_dO_dV, tma_tensor_dO_dV = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op,
                mdO_dV,
                cute.select(self.sdO_layout, mode=[0, 1, 2]),
                self.mma_tiler_pdo,
                self.tiled_mma_dV,
                self.cluster_layout_vmnk.shape,
            )

        # TMA atom for mQ_dK (M-block quantized Q for dK GEMM)
        tma_atom_Q_dK, tma_tensor_Q_dK = None, None
        if const_expr(self.blockscaled and mQ_dK is not None):
            # Use same TMA setup as original Q: sQ_layout, mma_tiler_kq, tiled_mma_S
            # Physical SMEM arrangement is identical to sQ; dK GEMM reads through sQt_dK
            Q_dK_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_S.thr_id
            )
            tma_atom_Q_dK, tma_tensor_Q_dK = cute.nvgpu.make_tiled_tma_atom_B(
                Q_dK_tma_op,
                mQ_dK,
                cute.select(self.sQ_layout, mode=[0, 1, 2]),
                self.mma_tiler_kq,
                self.tiled_mma_S,
                self.cluster_layout_vmnk.shape,
            )

        # TMA atom for mK_dQ (M-block quantized K for dQ GEMM)
        # K is loaded as A operand of S GEMM, then reinterpreted via sKt for dQ GEMM
        # Use same TMA setup as original K: make_tiled_tma_atom_A with sK_layout
        tma_atom_K_dQ, tma_tensor_K_dQ = None, None
        if const_expr(self.blockscaled and mK_dQ is not None):
            tma_atom_K_dQ, tma_tensor_K_dQ = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mK_dQ,
                cute.select(self.sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_kq,
                self.tiled_mma_S,
                self.cluster_layout_vmnk.shape,
            )

        # TMA atoms for scale factors (MXFP8 blockscaled)
        tma_atom_SFQ, tma_tensor_SFQ = None, None
        tma_atom_SFQ_dK, tma_tensor_SFQ_dK = None, None  # SFQ for dK GEMM
        tma_atom_SFK, tma_tensor_SFK = None, None
        tma_atom_SFK_dQ, tma_tensor_SFK_dQ = None, None  # SFK for dQ GEMM
        tma_atom_SFV, tma_tensor_SFV = None, None
        tma_atom_SFDO, tma_tensor_SFDO = None, None
        tma_atom_SFDO_dV, tma_tensor_SFDO_dV = None, None  # SFDO for dV GEMM

        if const_expr(self.blockscaled):
            # SFQ TMA atom (Q's scales, operand B)
            if const_expr(total_sf_q is not None):
                # pyre-ignore[16]
                sfq_shape = (total_sf_q, mQ.shape[1], mQ.shape[2])
            else:
                sfq_shape = mQ.shape
            sfq_layout = blockscaled_utils.tile_atom_to_shape_SF(
                sfq_shape, self.sf_vec_size
            )
            # pyre-ignore[16]
            mSFQ_tma = cute.make_tensor(mSFQ.iterator, sfq_layout)
            sSFQ_layout_per_stage = cute.select(self.sSFQ_layout, mode=[0, 1, 2])

            Q_tma_op_SF = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_S_bs.thr_id
            )
            tma_atom_SFQ, tma_tensor_SFQ = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op_SF,
                mSFQ_tma,
                sSFQ_layout_per_stage,
                self.mma_tiler_kq,
                self.tiled_mma_S_bs,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # SFQ_dK TMA atom — use dK GEMM params + transposed shape (like SFDO_dV)
            sSFQ_dK_layout_per_stage = cute.select(self.sSFQ_dK_layout, mode=[0, 1, 2])
            Q_tma_op_SF_dK = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_dK_bs.thr_id
            )
            # Create GMEM tensor for M-block SFQ if provided
            if const_expr(mSFQ_dK is not None):
                # M-block scale factors for Q in dK GEMM
                # pyre-ignore[16]
                sfq_dk_shape = (mQ_dK.shape[2], mQ_dK.shape[0], mQ_dK.shape[1])
                sfq_dk_layout = blockscaled_utils.tile_atom_to_shape_SF(
                    sfq_dk_shape, self.sf_vec_size
                )
                mSFQ_dK_tma = cute.make_tensor(mSFQ_dK.iterator, sfq_dk_layout)
            else:
                # Fallback to K-block scales (same as SFQ)
                mSFQ_dK_tma = mSFQ_tma
            tma_atom_SFQ_dK, tma_tensor_SFQ_dK = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op_SF_dK,
                mSFQ_dK_tma,
                sSFQ_dK_layout_per_stage,
                self.mma_tiler_dsq,
                self.tiled_mma_dK_bs,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # SFK TMA atom (K's scales, operand A)
            # For SF-only approach: use total_sf_k if provided (128-aligned total)
            if const_expr(total_sf_k is not None):
                sfk_shape = (total_sf_k, mK.shape[1], mK.shape[2])
            else:
                sfk_shape = mK.shape
            sfk_layout = blockscaled_utils.tile_atom_to_shape_SF(
                sfk_shape, self.sf_vec_size
            )
            mSFK_tma = cute.make_tensor(mSFK.iterator, sfk_layout)
            sSFK_layout_per_stage = cute.select(self.sSFK_layout, mode=[0, 1, 2])
            tma_atom_SFK, tma_tensor_SFK = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mSFK_tma,
                sSFK_layout_per_stage,
                self.mma_tiler_kq,
                self.tiled_mma_S_bs,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # SFV TMA atom (V's scales, operand A)
            # For SF-only approach: use total_sf_k for V (V has same seq dim as K)
            if const_expr(total_sf_k is not None):
                sfv_shape = (total_sf_k, mV.shape[1], mV.shape[2])
            else:
                sfv_shape = mV.shape
            sfv_layout = blockscaled_utils.tile_atom_to_shape_SF(
                sfv_shape, self.sf_vec_size
            )
            mSFV_tma = cute.make_tensor(mSFV.iterator, sfv_layout)
            sSFV_layout_per_stage = cute.select(self.sSFV_layout, mode=[0, 1, 2])
            tma_atom_SFV, tma_tensor_SFV = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mSFV_tma,
                sSFV_layout_per_stage,
                self.mma_tiler_vdo,
                self.tiled_mma_dP_bs,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # SFDO TMA atom (dO's scales, operand B)
            # NOTE: SFDO uses mdO.shape (O-side dimensions), NOT total_sf_q.
            # In broadcast_q, dO has B*dense_q_len tokens which differs from Q's total.
            # The SFDO scale tensor is always co-located with dO data.
            sfdo_layout = blockscaled_utils.tile_atom_to_shape_SF(
                mdO.shape, self.sf_vec_size
            )
            mSFDO_tma = cute.make_tensor(mSFDO.iterator, sfdo_layout)
            sSFDO_layout_per_stage = cute.select(self.sSFDO_layout, mode=[0, 1, 2])

            dO_tma_op_SF = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_dP_bs.thr_id
            )
            tma_atom_SFDO, tma_tensor_SFDO = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op_SF,
                mSFDO_tma,
                sSFDO_layout_per_stage,
                self.mma_tiler_vdo,
                self.tiled_mma_dP_bs,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # SFDO_dV TMA atom (dO's scales for dV GEMM, uses tiled_mma_dV_bs)
            # dV = P.T @ dO uses dO non-transposed, with MMA K dimension = M (tokens).
            # dO's scales are computed along hdim, so we need a TRANSPOSED scale tensor
            # (mSFDO_dV) that has scales along M instead of hdim.
            sSFDO_dV_layout_per_stage = cute.select(
                self.sSFDO_dV_layout, mode=[0, 1, 2]
            )
            dO_tma_op_SF_dV = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_dV_bs.thr_id
            )
            # Use mSFDO_dV if provided (transposed quantization), otherwise fall back to mSFDO
            if const_expr(mSFDO_dV is not None):
                # For dV GEMM, MMA K dimension is M (tokens), not hdim (K).
                # Use TRANSPOSED dO shape for layout: (K, H, M) instead of (M, H, K).
                # This matches the transposed SF tensor data which has scales along M.
                mdO_shape_transposed_dv = (mdO.shape[2], mdO.shape[1], mdO.shape[0])
                sfdo_dv_layout = blockscaled_utils.tile_atom_to_shape_SF(
                    mdO_shape_transposed_dv, self.sf_vec_size
                )
                mSFDO_dV_tma = cute.make_tensor(mSFDO_dV.iterator, sfdo_dv_layout)
            else:
                # Fallback: use same layout as SFDO
                mSFDO_dV_tma = mSFDO_tma
            tma_atom_SFDO_dV, tma_tensor_SFDO_dV = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op_SF_dV,
                mSFDO_dV_tma,
                sSFDO_dV_layout_per_stage,
                self.mma_tiler_pdo,  # dV GEMM tiler
                self.tiled_mma_dV_bs,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # SFK_dQ TMA atom (K's scales for dQ GEMM, operand B)
            # Separate TMA from SFK because dQ uses different MMA/tiler (dQ = dS @ K)
            sSFK_dQ_layout_per_stage = cute.select(self.sSFK_dQ_layout, mode=[0, 1, 2])
            K_tma_op_SF_dQ = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_dQ_bs.thr_id
            )
            # Use M-block SFK_dQ if provided, otherwise fall back to K-block SFK
            if const_expr(mSFK_dQ is not None):
                # M-block scale factors for K in dQ GEMM
                mK_shape_transposed = (mK.shape[2], mK.shape[0], mK.shape[1])
                sfk_dq_layout = blockscaled_utils.tile_atom_to_shape_SF(
                    mK_shape_transposed, self.sf_vec_size
                )
                mSFK_dQ_tma = cute.make_tensor(mSFK_dQ.iterator, sfk_dq_layout)
            else:
                # Fallback to K-block scales (same as SFK)
                mSFK_dQ_tma = mSFK_tma
            tma_atom_SFK_dQ, tma_tensor_SFK_dQ = cute.nvgpu.make_tiled_tma_atom_B(
                K_tma_op_SF_dQ,
                mSFK_dQ_tma,
                sSFK_dQ_layout_per_stage,
                self.mma_tiler_dsk,  # dQ GEMM tiler
                self.tiled_mma_dQ_bs,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        # pyre-ignore[16]
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(
                mX.element_type, cute.select(layout, mode=[0, 1, 2])
            )
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = (
            # pyre-ignore[16]
            self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        )
        self.tma_copy_bytes["dKacc"] = (
            # pyre-ignore[16]
            self.tile_n * self.dK_reduce_ncol * Float32.width // 8
        )

        # Add SF bytes to transaction counts (MXFP8 blockscaled)
        # pyre-ignore[16]
        self.tma_copy_sfq_bytes = 0
        # pyre-ignore[16]
        self.tma_copy_sfk_bytes = 0
        # pyre-ignore[16]
        self.tma_copy_sfv_bytes = 0
        # pyre-ignore[16]
        self.tma_copy_sfdo_bytes = 0

        if const_expr(self.blockscaled):
            self.tma_copy_sfq_bytes = int(
                cute.size_in_bytes(
                    self.sf_dtype, cute.select(self.sSFQ_layout, mode=[0, 1, 2])
                )
            )
            self.tma_copy_bytes["Q"] += (
                self.tma_copy_sfq_bytes
            )  # Enabled - SFQ loading active
            # SFQ_dK: Q's scale factors for dK GEMM (separate from SFQ due to layout mismatch)
            # pyre-ignore[16]
            self.tma_copy_sfq_dk_bytes = int(
                cute.size_in_bytes(
                    self.sf_dtype, cute.select(self.sSFQ_dK_layout, mode=[0, 1, 2])
                )
            )
            self.tma_copy_bytes["Q"] += (
                self.tma_copy_sfq_dk_bytes
            )  # Enabled - SFQ_dK loading active
            self.tma_copy_sfk_bytes = int(
                cute.size_in_bytes(
                    self.sf_dtype, cute.select(self.sSFK_layout, mode=[0, 1, 2])
                )
            )
            self.tma_copy_bytes["K"] += (
                self.tma_copy_sfk_bytes
            )  # Enabled - SFK loading active
            self.tma_copy_sfv_bytes = int(
                cute.size_in_bytes(
                    self.sf_dtype, cute.select(self.sSFV_layout, mode=[0, 1, 2])
                )
            )
            self.tma_copy_bytes["V"] += (
                self.tma_copy_sfv_bytes
            )  # Enabled - SFV loading active
            self.tma_copy_sfdo_bytes = int(
                cute.size_in_bytes(
                    self.sf_dtype, cute.select(self.sSFDO_layout, mode=[0, 1, 2])
                )
            )
            self.tma_copy_bytes["dO"] += (
                self.tma_copy_sfdo_bytes
            )  # Enabled - SFDO loading active
            # SFDO_dV: dO's scale factors for dV GEMM (separate from SFDO due to layout mismatch)
            # pyre-ignore[16]
            self.tma_copy_sfdo_dv_bytes = int(
                cute.size_in_bytes(
                    self.sf_dtype, cute.select(self.sSFDO_dV_layout, mode=[0, 1, 2])
                )
            )
            self.tma_copy_bytes["dO"] += (
                self.tma_copy_sfdo_dv_bytes
            )  # Enabled - SFDO_dV loading active
            # SFK_dQ: K's scale factors for dQ GEMM (separate from SFK due to layout mismatch)
            # pyre-ignore[16]
            self.tma_copy_sfk_dq_bytes = int(
                cute.size_in_bytes(
                    self.sf_dtype, cute.select(self.sSFK_dQ_layout, mode=[0, 1, 2])
                )
            )
            # Enable SFK_dQ TMA loading - add to K's transaction count
            self.tma_copy_bytes["K"] += (
                self.tma_copy_sfk_dq_bytes
            )  # Enabled - SFK_dQ loading with K

        # TMA copy bytes for mdO_dV (M-block quantized dO for dV GEMM)
        if const_expr(self.blockscaled and mdO_dV is not None):
            self.tma_copy_bytes["dO_dV"] = cute.size_in_bytes(
                # pyre-ignore[16]
                mdO_dV.element_type,
                cute.select(self.sdO_layout, mode=[0, 1, 2]),
            )
        else:
            self.tma_copy_bytes["dO_dV"] = 0

        # TMA copy bytes for mQ_dK (M-block quantized Q for dK GEMM)
        if const_expr(self.blockscaled and mQ_dK is not None):
            self.tma_copy_bytes["Q_dK"] = cute.size_in_bytes(
                mQ_dK.element_type, cute.select(self.sQ_layout, mode=[0, 1, 2])
            )
        else:
            self.tma_copy_bytes["Q_dK"] = 0

        # TMA copy bytes for mK_dQ (M-block quantized K for dQ GEMM)
        if const_expr(self.blockscaled and mK_dQ is not None):
            self.tma_copy_bytes["K_dQ"] = cute.size_in_bytes(
                mK_dQ.element_type, cute.select(self.sK_layout, mode=[0, 1, 2])
            )
        else:
            self.tma_copy_bytes["K_dQ"] = 0

        if const_expr(
            self.is_persistent
            and self.is_varlen_k
            and not self.deterministic
            and not self.is_causal
            and not self.is_local
        ):
            # Varlen persistent: lookup tables with block-innermost ordering
            TileScheduler = PersistentVarlenLookupScheduler
        elif const_expr(
            self.is_persistent
            and not self.is_varlen_k
            and not self.deterministic
            and not self.is_causal
            and not self.is_local
        ):
            # Non-varlen persistent: static scheduler (good L2, no lookup tables)
            TileScheduler = StaticPersistentTileScheduler
        elif const_expr(self.is_varlen_k):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        # reads n_blocks right-to-left
        # pyre-ignore[16]
        self.spt = (self.is_causal or self.is_local) and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            cute.size(mK.shape[3])
            if mCuSeqlensK is None
            else cute.size(mCuSeqlensK) - 1,
            # pyre-ignore[6]
            1,  # num_splits
            cute.size(mK.shape[0]),  # seqlen_k: bwd tiles over N-blocks (K dim)
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mK.shape[0]),  # total K tokens for grid sizing
            tile_shape_mn=self.cta_tiler[:2],
            # pyre-ignore[6]
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            # pyre-ignore[6]
            qhead_per_kvhead_packgqa=1,
            # pyre-ignore[6]
            element_size=self.k_dtype.width // 8,
            # pyre-ignore[6]
            is_persistent=self.is_persistent,
            # pyre-ignore[6]
            lpt=self.spt,
            # pyre-ignore[6]
            head_swizzle=self.spt,
            mTileToBatch=mTileToBatch,
            mTileToHead=mTileToHead,
            mTileToBlock=mTileToBlock,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        # pyre-ignore[16]
        self.tile_scheduler_cls = TileScheduler
        # pyre-ignore[6]
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # Compute allocation sizes for shared buffers that are reused
        # sQ is reused for sdK, sdO is reused for sdV
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdKV_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(self.dv_dtype, self.sdKV_layout),
            cute.size_in_bytes(self.do_dtype, self.sdO_layout),
        )
        # sdO_dV for M-block quantized dO (blockscaled dV GEMM only)
        # Uses same layout as sdO since shape is identical
        sdO_dV_alloc_bytes = (
            cute.size_in_bytes(self.do_dtype, self.sdO_layout)
            if const_expr(self.blockscaled and mdO_dV is not None)
            else 0
        )
        # sQ_dK for M-block quantized Q (blockscaled dK GEMM only)
        # Uses sQ_layout for physical SMEM allocation (same physical layout as sQ)
        sQ_dK_alloc_bytes = (
            cute.size_in_bytes(self.q_dtype, self.sQ_layout)
            if const_expr(self.blockscaled and mQ_dK is not None)
            else 0
        )
        # sK_dQ for M-block quantized K (blockscaled dQ GEMM only)
        # Uses same layout as sK since shape is identical
        sK_dQ_alloc_bytes = (
            cute.size_in_bytes(self.k_dtype, self.sK_layout)
            if const_expr(self.blockscaled and mK_dQ is not None)
            else 0
        )
        # Sanity check that layouts fit in allocation
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdKV_layout)
        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdKV_layout)
        assert sdV_bytes <= sdO_alloc_bytes, "sdV doesn't fit in sdO storage allocation"
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
            dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.dQaccum_reduce_stage // 2
            ]
            dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.dQaccum_reduce_stage // 2
            ]
            tmem_holding_buf: Int32
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            K_done_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]

            # Smem tensors

            # sQ is reused for sdK which in the non-MHA case needs float32
            sQ: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]
            # sdO is reused for sdV which in the non-MHA case needs float32
            sdO: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                self.buffer_align_bytes,
            ]
            # sdO_dV for M-block quantized dO (blockscaled dV GEMM only)
            sdO_dV: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sdO_dV_alloc_bytes],
                self.buffer_align_bytes,
            ]
            # sQ_dK for M-block quantized Q (blockscaled dK GEMM only)
            sQ_dK: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sQ_dK_alloc_bytes],
                self.buffer_align_bytes,
            ]
            # sK_dQ for M-block quantized K (blockscaled dQ GEMM only)
            sK_dQ: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sK_dQ_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                128,
            ]
            # sP: FP8 P data (blockscaled) or S scratch (SiLU backward)
            sP: cute.struct.Align[
                cute.struct.MemRange[
                    self.do_dtype,
                    cute.cosize(self.tP_layout) if const_expr(self.blockscaled) else 0,
                ],
                128,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                128,
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                128,
            ]
            sdQaccum: cute.struct.Align[
                cute.struct.MemRange[
                    self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)
                ],
                self.buffer_align_bytes,
            ]

            # Scale factor SMEM buffers for blockscaled MXFP8
            sSFQ: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU,
                    cute.cosize(self.sSFQ_layout)
                    if const_expr(self.blockscaled)
                    else 0,
                ],
                self.buffer_align_bytes,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU,
                    cute.cosize(self.sSFK_layout)
                    if const_expr(self.blockscaled)
                    else 0,
                ],
                self.buffer_align_bytes,
            ]
            sSFV: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU,
                    cute.cosize(self.sSFV_layout)
                    if const_expr(self.blockscaled)
                    else 0,
                ],
                self.buffer_align_bytes,
            ]
            sSFDO: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU,
                    cute.cosize(self.sSFDO_layout)
                    if const_expr(self.blockscaled)
                    else 0,
                ],
                self.buffer_align_bytes,
            ]
            # Scale factor SMEM buffers (SFP computed from P's AMAX, SFDS computed from dS's AMAX)
            # These are filled with constant E8M0=0x7F (scale=1.0) at kernel start
            sSFP: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU,
                    cute.cosize(self.sSFP_layout)
                    if const_expr(self.blockscaled)
                    else 0,
                ],
                self.buffer_align_bytes,
            ]
            sSFDS: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU,
                    cute.cosize(self.sSFDS_layout)
                    if const_expr(self.blockscaled)
                    else 0,
                ],
                self.buffer_align_bytes,
            ]
            # SFQ_dK: Q's scale factors for dK GEMM (separate from sSFQ due to layout mismatch)
            sSFQ_dK: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU,
                    cute.cosize(self.sSFQ_dK_layout)
                    if const_expr(self.blockscaled)
                    else 0,
                ],
                self.buffer_align_bytes,
            ]
            # SFDO_dV: dO's scale factors for dV GEMM (separate from sSFDO due to layout mismatch)
            sSFDO_dV: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU,
                    cute.cosize(self.sSFDO_dV_layout)
                    if const_expr(self.blockscaled)
                    else 0,
                ],
                self.buffer_align_bytes,
            ]
            # SFDS_dQ: dS's scale factors for dQ GEMM (separate from sSFDS due to layout mismatch)
            sSFDS_dQ: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU,
                    cute.cosize(self.sSFDS_dQ_layout)
                    if const_expr(self.blockscaled)
                    else 0,
                ],
                self.buffer_align_bytes,
            ]
            # SFK_dQ: K's scale factors for dQ GEMM (separate from sSFK due to layout mismatch)
            sSFK_dQ: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU,
                    cute.cosize(self.sSFK_dQ_layout)
                    if const_expr(self.blockscaled)
                    else 0,
                ],
                self.buffer_align_bytes,
            ]

        # pyre-ignore[16]
        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            mdV,
            mdK,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dO_dV,  # TMA atom for M-block quantized dO (dV GEMM)
            tma_tensor_dO_dV,  # TMA tensor for M-block quantized dO (dV GEMM)
            tma_atom_Q_dK,  # TMA atom for M-block quantized Q (dK GEMM)
            tma_tensor_Q_dK,  # TMA tensor for M-block quantized Q (dK GEMM)
            tma_atom_K_dQ,  # TMA atom for M-block quantized K (dQ GEMM)
            tma_tensor_K_dQ,  # TMA tensor for M-block quantized K (dQ GEMM)
            tma_atom_dV,
            tma_atom_dK,
            # SF TMA atoms and tensors
            tma_atom_SFQ,
            tma_tensor_SFQ,
            tma_atom_SFQ_dK,
            tma_tensor_SFQ_dK,
            tma_atom_SFK,
            tma_tensor_SFK,
            tma_atom_SFK_dQ,
            tma_tensor_SFK_dQ,
            tma_atom_SFV,
            tma_tensor_SFV,
            tma_atom_SFDO,
            tma_tensor_SFDO,
            tma_atom_SFDO_dV,
            tma_tensor_SFDO_dV,
            self.sSFQ_layout,
            self.sSFQ_dK_layout,
            self.sSFK_layout,
            self.sSFV_layout,
            self.sSFDO_layout,
            self.sSFDO_dV_layout,
            self.sSFP_layout,
            self.sSFDS_layout,
            self.sSFDS_dQ_layout,
            self.sSFK_dQ_layout,
            self.sQ_layout,
            # pyre-ignore[16]
            self.sQt_layout,
            self.sK_layout,
            self.sV_layout,
            # pyre-ignore[16]
            self.sLSE_layout,
            # pyre-ignore[16]
            self.sdPsum_layout,
            self.sdO_layout,
            # pyre-ignore[16]
            self.sdOt_layout,
            # pyre-ignore[16]
            self.sdSt_layout,
            # pyre-ignore[16]
            self.sdS_layout,
            # pyre-ignore[16]
            self.sKt_layout,
            # pyre-ignore[16]
            self.sdS_dQ_data_layout if const_expr(self.blockscaled) else None,
            # pyre-ignore[16]
            self.sdS_dK_data_layout if const_expr(self.blockscaled) else None,
            # pyre-ignore[16]
            self.sdQaccum_layout,
            self.sdKV_layout,
            # pyre-ignore[16]
            self.tP_layout,
            # pyre-ignore[16]
            self.tdS_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            self.tiled_mma_S_bs if const_expr(self.blockscaled) else None,
            self.tiled_mma_dP_bs if const_expr(self.blockscaled) else None,
            self.tiled_mma_dV_bs if const_expr(self.blockscaled) else None,
            self.tiled_mma_dK_bs if const_expr(self.blockscaled) else None,
            self.tiled_mma_dK_bs_smem if const_expr(self.blockscaled) else None,
            self.tiled_mma_dQ_bs if const_expr(self.blockscaled) else None,
            tiled_copy_r2s_dKV,
            softmax_scale,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            tile_sched_params,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mdSFK_out,
            mdSFV_out,
            mCuSeqlensO,
            mCuSeqlensSFQ,
            mCuSeqlensSFK,
            mAttnScale,  # SiLU: per-row attention scale
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk
            if cute.size(self.cluster_shape_mnk) > 1
            else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dO_dV: Optional[
            cute.CopyAtom
        ],  # TMA atom for M-block quantized dO (dV GEMM)
        tma_tensor_dO_dV: Optional[
            cute.Tensor
        ],  # TMA tensor for M-block quantized dO (dV GEMM)
        tma_atom_Q_dK: Optional[
            cute.CopyAtom
        ],  # TMA atom for M-block quantized Q (dK GEMM)
        tma_tensor_Q_dK: Optional[
            cute.Tensor
        ],  # TMA tensor for M-block quantized Q (dK GEMM)
        tma_atom_K_dQ: Optional[
            cute.CopyAtom
        ],  # TMA atom for M-block quantized K (dQ GEMM)
        tma_tensor_K_dQ: Optional[
            cute.Tensor
        ],  # TMA tensor for M-block quantized K (dQ GEMM)
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        # SF TMA atoms and tensors
        tma_atom_SFQ: Optional[cute.CopyAtom],
        tma_tensor_SFQ: Optional[cute.Tensor],
        tma_atom_SFQ_dK: Optional[cute.CopyAtom],
        tma_tensor_SFQ_dK: Optional[cute.Tensor],
        tma_atom_SFK: Optional[cute.CopyAtom],
        tma_tensor_SFK: Optional[cute.Tensor],
        tma_atom_SFK_dQ: Optional[cute.CopyAtom],  # SFK for dQ GEMM
        tma_tensor_SFK_dQ: Optional[cute.Tensor],  # SFK for dQ GEMM
        tma_atom_SFV: Optional[cute.CopyAtom],
        tma_tensor_SFV: Optional[cute.Tensor],
        tma_atom_SFDO: Optional[cute.CopyAtom],
        tma_tensor_SFDO: Optional[cute.Tensor],
        tma_atom_SFDO_dV: Optional[cute.CopyAtom],  # SFDO for dV GEMM
        tma_tensor_SFDO_dV: Optional[cute.Tensor],  # SFDO for dV GEMM
        sSFQ_layout: Optional[cute.Layout],
        sSFQ_dK_layout: Optional[cute.Layout],
        sSFK_layout: Optional[cute.Layout],
        sSFV_layout: Optional[cute.Layout],
        sSFDO_layout: Optional[cute.Layout],
        sSFDO_dV_layout: Optional[cute.Layout],  # SFDO for dV GEMM (separate layout)
        sSFP_layout: Optional[cute.Layout],  # Constant SF for P (scale=1.0)
        sSFDS_layout: Optional[cute.Layout],  # Constant SF for dS (scale=1.0)
        sSFDS_dQ_layout: Optional[cute.Layout],  # SFDS for dQ GEMM (separate layout)
        sSFK_dQ_layout: Optional[cute.Layout],  # SFK for dQ GEMM (separate layout)
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sdS_dQ_data_layout: Optional[
            cute.ComposedLayout
        ],  # Blockscaled A layout for dQ
        sdS_dK_data_layout: Optional[
            cute.ComposedLayout
        ],  # Blockscaled A layout for dK
        sdQaccum_layout: cute.Layout,
        sdKV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_mma_S_bs: Optional[cute.TiledMma],  # Blockscaled MMA for S = K @ Q.T
        tiled_mma_dP_bs: Optional[cute.TiledMma],  # Blockscaled MMA for dP = V @ dO.T
        tiled_mma_dV_bs: Optional[cute.TiledMma],  # Blockscaled MMA for dV = P.T @ dO
        tiled_mma_dK_bs: Optional[
            cute.TiledMma
        ],  # Blockscaled MMA for dK SF layouts (TMEM)
        tiled_mma_dK_bs_smem: Optional[
            cute.TiledMma
        ],  # Blockscaled MMA for dK GEMM (SMEM)
        tiled_mma_dQ_bs: Optional[cute.TiledMma],  # Blockscaled MMA for dQ = dS @ K
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        tile_sched_params: ParamsBase,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mdSFK_out: Optional[cute.Tensor],  # Output scale factors for dK (MXFP8)
        mdSFV_out: Optional[cute.Tensor],  # Output scale factors for dV (MXFP8)
        mCuSeqlensO: Optional[cute.Tensor],  # O addressing for broadcast_q
        # 128-aligned cu_seqlens for scale factor offsets (varlen MXFP8)
        mCuSeqlensSFQ: Optional[cute.Tensor] = None,
        mCuSeqlensSFK: Optional[cute.Tensor] = None,
        mAttnScale: Optional[cute.Tensor] = None,  # SiLU: per-row attention scale
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)

        cluster_layout_vmnk = cute.tiled_divide(
            # pyre-ignore[16]
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        # pyre-ignore[16]
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        dQ_cluster_full_mbar_ptr = storage.dQ_cluster_full_mbar_ptr.data_ptr()
        dQ_cluster_empty_mbar_ptr = storage.dQ_cluster_empty_mbar_ptr.data_ptr()

        if warp_idx == 1:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr, cute.arch.WARP_SIZE * len(self.compute_warp_ids)
            )
        # pyre-ignore[16]
        if const_expr(self.cluster_reduce_dQ):
            if warp_idx == 4:
                # pyre-ignore[16]
                for i in range(self.dQaccum_reduce_stage // 2):
                    cute.arch.mbarrier_init(dQ_cluster_full_mbar_ptr + i, 1)
                    cute.arch.mbarrier_init(dQ_cluster_empty_mbar_ptr + i, 1)

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        # Only 1 thread per warp will signal
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)
        )
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
        )
        pipeline_K_done = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, 1
            ),
            barrier_storage=storage.K_done_mbar_ptr.data_ptr(),
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids),
        )  # Compute
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
        )

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        # The arrive count is the number of mcast size
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            # pyre-ignore[16]
            len([self.mma_warp_id]) * self.num_mcast_ctas_b,
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            # cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.num_mcast_ctas_b
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            # pyre-ignore[16]
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            # pyre-ignore[16]
            tx_count=self.tma_copy_bytes["LSE"],
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            # pyre-ignore[16]
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            # pyre-ignore[6]
            init_wait=False,
        )
        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            # pyre-ignore[6]
            init_wait=True,
        )

        sQ = storage.sQ.get_tensor(
            sQ_layout.outer,
            swizzle=sQ_layout.inner,
            # pyre-ignore[16]
            dtype=self.q_dtype,
        )
        sQt = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype),
            sQt_layout.outer,
        )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sKt = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer
        )
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(
            cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer
        )
        # Blockscaled dQ SMEM view: uses blockscaled layout for fragment A creation
        sdS_dQ = None
        if const_expr(self.blockscaled and sdS_dQ_data_layout is not None):
            sdS_dQ = cute.make_tensor(
                # pyre-ignore[16]
                cute.recast_ptr(sdSt.iterator, sdS_dQ_data_layout.inner),
                # pyre-ignore[16]
                sdS_dQ_data_layout.outer,
            )
        # Blockscaled dK SMEM view: uses blockscaled K-major layout for fragment A
        sdS_dK = None
        if const_expr(self.blockscaled and sdS_dK_data_layout is not None):
            sdS_dK = cute.make_tensor(
                cute.recast_ptr(sdSt.iterator, sdS_dK_data_layout.inner),
                sdS_dK_data_layout.outer,
            )
        # sP for FP8 P data — dV GEMM reads P from SMEM in blockscaled mode
        sP = None
        if const_expr(self.blockscaled):
            sP = storage.sP.get_tensor(
                tP_layout.outer,
                swizzle=tP_layout.inner,
                # pyre-ignore[16]
                dtype=self.do_dtype,
            )
        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )
        sdOt = cute.make_tensor(
            cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
            sdOt_layout.outer,
        )
        # sdO_dV for M-block quantized dO (blockscaled dV GEMM only)
        sdO_dV = None
        if const_expr(self.blockscaled and tma_tensor_dO_dV is not None):
            sdO_dV = storage.sdO_dV.get_tensor(
                sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
            )
        # sQ_dK for M-block quantized Q (blockscaled dK GEMM only)
        sQ_dK = None
        sQt_dK = None  # transposed view for dK GEMM
        if const_expr(self.blockscaled and tma_tensor_Q_dK is not None):
            sQ_dK = storage.sQ_dK.get_tensor(
                sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype
            )
            sQt_dK = cute.make_tensor(
                cute.recast_ptr(sQ_dK.iterator, sQt_layout.inner, dtype=self.q_dtype),
                sQt_layout.outer,
            )
        # sK_dQ for M-block quantized K (blockscaled dQ GEMM only)
        sK_dQ = None
        sKt_dQ = None  # transposed view for dQ GEMM
        if const_expr(self.blockscaled and tma_tensor_K_dQ is not None):
            sK_dQ = storage.sK_dQ.get_tensor(
                sK_layout.outer,
                swizzle=sK_layout.inner,
                # pyre-ignore[16]
                dtype=self.k_dtype,
            )
            sKt_dQ = cute.make_tensor(
                cute.recast_ptr(sK_dQ.iterator, sKt_layout.inner, dtype=self.k_dtype),
                sKt_layout.outer,
            )
        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        # pyre-ignore[16]
        if const_expr(not self.dKV_postprocess):
            sdV = storage.sdO.get_tensor(
                # pyre-ignore[16]
                sdKV_layout.outer,
                # pyre-ignore[16]
                swizzle=sdKV_layout.inner,
                # pyre-ignore[16]
                dtype=self.dv_dtype,
            )
            sdK = storage.sQ.get_tensor(
                # pyre-ignore[16]
                sdKV_layout.outer,
                # pyre-ignore[16]
                swizzle=sdKV_layout.inner,
                # pyre-ignore[16]
                dtype=self.dk_dtype,
            )
        else:
            sdV = storage.sdO.get_tensor(sdKV_layout, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdKV_layout, dtype=self.dk_dtype)

        # Buffer sizing is guaranteed by max(...) in SharedStorage declarations
        # for both sQ (reused as sdK) and sdO (reused as sdV)

        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        # Get scale factor SMEM tensors
        sSFQ, sSFK, sSFV, sSFDO = None, None, None, None
        sSFP, sSFDS = None, None
        sSFQ_dK = None  # SFQ for dK GEMM (separate layout)
        sSFDO_dV = None  # SFDO for dV GEMM (separate layout)
        sSFDS_dQ = None  # SFDS for dQ GEMM (separate layout)
        sSFK_dQ = None  # SFK for dQ GEMM (separate layout)
        if const_expr(self.blockscaled):
            sSFQ = storage.sSFQ.get_tensor(sSFQ_layout)
            sSFK = storage.sSFK.get_tensor(sSFK_layout)
            sSFV = storage.sSFV.get_tensor(sSFV_layout)
            sSFDO = storage.sSFDO.get_tensor(sSFDO_layout)
            # SF tensors (SFP computed from P's AMAX, SFDS computed from dS's AMAX)
            sSFP = storage.sSFP.get_tensor(sSFP_layout)
            sSFDS = storage.sSFDS.get_tensor(sSFDS_layout)
            # SFQ for dK GEMM (separate storage due to layout mismatch)
            sSFQ_dK = storage.sSFQ_dK.get_tensor(sSFQ_dK_layout)
            # SFDO for dV GEMM (separate storage due to layout mismatch)
            sSFDO_dV = storage.sSFDO_dV.get_tensor(sSFDO_dV_layout)
            # SFDS and SFK for dQ GEMM (separate storage due to layout mismatch)
            sSFDS_dQ = storage.sSFDS_dQ.get_tensor(sSFDS_dQ_layout)
            sSFK_dQ = storage.sSFK_dQ.get_tensor(sSFK_dQ_layout)

        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(
            Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )
        # S
        # For blockscaled mode, use blockscaled MMA for layout consistency
        if const_expr(self.blockscaled):
            # pyre-ignore[16]
            thr_mma_S = tiled_mma_S_bs.get_slice(0)
        else:
            thr_mma_S = tiled_mma_S.get_slice(0)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        # For blockscaled mode, use blockscaled MMA for layout consistency
        if const_expr(self.blockscaled):
            thr_mma_dP = tiled_mma_dP_bs.get_slice(0)
        else:
            thr_mma_dP = tiled_mma_dP.get_slice(0)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        # For blockscaled mode, use blockscaled MMA for layout consistency
        if const_expr(self.blockscaled):
            thr_mma_dV = tiled_mma_dV_bs.get_slice(0)
        else:
            thr_mma_dV = tiled_mma_dV.get_slice(0)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype),
            tP_layout.outer,
        )
        # dK
        # For blockscaled mode, use blockscaled MMA for layout consistency
        if const_expr(self.blockscaled):
            thr_mma_dK = tiled_mma_dK_bs.get_slice(0)
        else:
            thr_mma_dK = tiled_mma_dK.get_slice(0)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(
            # pyre-ignore[16]
            cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype),
            tdS_layout.outer,
        )
        # dQ
        # For blockscaled mode, use blockscaled MMA for layout consistency
        if const_expr(self.blockscaled):
            thr_mma_dQ = tiled_mma_dQ_bs.get_slice(0)
        else:
            thr_mma_dQ = tiled_mma_dQ.get_slice(0)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        # SF TMEM tensors for blockscaled MMA (S = K @ Q.T and dP = V @ dO.T)
        tCtSFK, tCtSFQ, tCtSFV, tCtSFDO = None, None, None, None
        tCtSFK_prologue, tCtSFQ_prologue = None, None
        tCtSFV_prologue, tCtSFDO_prologue = None, None
        # Phase 2 SF TMEM tensors (SFP computed from P, SFDS computed from dS)
        tCtSFP, tCtSFDO_dV = None, None
        tCtSFDS, tCtSFQ_dK = None, None
        # dQ GEMM SF TMEM tensors
        tCtSFDS_dQ, tCtSFK_dQ = None, None

        if const_expr(self.blockscaled):
            # Create TMEM layouts for S = K @ Q.T
            # SFK is operand A, SFQ is operand B
            sSFK_layout_per_stage = cute.slice_(sSFK.layout, (None, None, None, 0))
            tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfa(
                # pyre-ignore[16]
                self.tiled_mma_S_bs,
                self.mma_tiler_kq,
                self.sf_vec_size,
                sSFK_layout_per_stage,
            )
            sSFQ_layout_per_stage = cute.slice_(sSFQ.layout, (None, None, None, 0))
            tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfb(
                self.tiled_mma_S_bs,
                self.mma_tiler_kq,
                self.sf_vec_size,
                sSFQ_layout_per_stage,
            )

            # Create TMEM layouts for dP = V @ dO.T
            # SFV is operand A, SFDO is operand B
            sSFV_layout_per_stage = cute.slice_(sSFV.layout, (None, None, None, 0))
            tCtSFV_layout = blockscaled_utils.make_tmem_layout_sfa(
                # pyre-ignore[16]
                self.tiled_mma_dP_bs,
                self.mma_tiler_vdo,
                self.sf_vec_size,
                sSFV_layout_per_stage,
            )
            sSFDO_layout_per_stage = cute.slice_(sSFDO.layout, (None, None, None, 0))
            tCtSFDO_layout = blockscaled_utils.make_tmem_layout_sfb(
                self.tiled_mma_dP_bs,
                self.mma_tiler_vdo,
                self.sf_vec_size,
                sSFDO_layout_per_stage,
            )

            # SFDO_dV TMEM layout (for dV GEMM, uses tiled_mma_dV_bs)
            sSFDO_dV_layout_per_stage = cute.slice_(
                sSFDO_dV.layout, (None, None, None, 0)
            )
            tCtSFDO_dV_layout = blockscaled_utils.make_tmem_layout_sfb(
                # pyre-ignore[16]
                self.tiled_mma_dV_bs,
                self.mma_tiler_pdo,
                self.sf_vec_size,
                sSFDO_dV_layout_per_stage,
            )

            # Get relative offsets for SFB vs SFA
            temp_sfk_ptr = cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype)
            temp_tCtSFK = cute.make_tensor(temp_sfk_ptr, tCtSFK_layout)
            sfq_relative_offset = tcgen05.find_tmem_tensor_col_offset(temp_tCtSFK)

            temp_sfv_ptr = cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype)
            temp_tCtSFV = cute.make_tensor(temp_sfv_ptr, tCtSFV_layout)
            sfdo_relative_offset = tcgen05.find_tmem_tensor_col_offset(temp_tCtSFV)

            # MAIN LOOP SF tensors (use dP region - recomputed each iteration)
            sf_ptr = cute.recast_ptr(
                tmem_ptr + self.tmem_SF_offset, dtype=self.sf_dtype
            )
            tCtSFK = cute.make_tensor(sf_ptr, tCtSFK_layout)
            tCtSFQ = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_SF_offset + sfq_relative_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFQ_layout,
            )
            # dP GEMM SFs at separate TMEM location (tmem_SF_offset_dP = S+96)
            sf_ptr_dP = cute.recast_ptr(
                tmem_ptr + self.tmem_SF_offset_dP, dtype=self.sf_dtype
            )
            tCtSFV = cute.make_tensor(sf_ptr_dP, tCtSFV_layout)
            tCtSFDO = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_SF_offset_dP + sfdo_relative_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFDO_layout,
            )

            # PROLOGUE SF tensors (use dK region - dK is zero in first iteration)
            sf_prologue_ptr = cute.recast_ptr(
                tmem_ptr + self.tmem_SF_prologue_offset, dtype=self.sf_dtype
            )
            tCtSFK_prologue = cute.make_tensor(sf_prologue_ptr, tCtSFK_layout)
            tCtSFQ_prologue = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_SF_prologue_offset + sfq_relative_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFQ_layout,
            )
            tCtSFV_prologue = cute.make_tensor(sf_prologue_ptr, tCtSFV_layout)
            tCtSFDO_prologue = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_SF_prologue_offset + sfdo_relative_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFDO_layout,
            )

            # Create TMEM layouts for SFP and SFDS (computed from P and dS AMAX respectively)
            # SFP is operand A for dV = P.T @ dO
            sSFP_layout_per_stage = cute.slice_(sSFP.layout, (None, None, None, 0))
            tCtSFP_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dV_bs,
                self.mma_tiler_pdo,
                self.sf_vec_size,
                sSFP_layout_per_stage,
            )
            # SFDS is operand A for dK = dS.T @ Q
            sSFDS_layout_per_stage = cute.slice_(sSFDS.layout, (None, None, None, 0))
            tCtSFDS_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dK_bs,
                self.mma_tiler_dsq,
                self.sf_vec_size,
                sSFDS_layout_per_stage,
            )
            # SFQ_dK is operand B for dK = dS.T @ Q
            sSFQ_dK_layout_per_stage = cute.slice_(
                sSFQ_dK.layout, (None, None, None, 0)
            )
            tCtSFQ_dK_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_dK_bs,
                self.mma_tiler_dsq,
                self.sf_vec_size,
                sSFQ_dK_layout_per_stage,
            )

            # Get relative offsets for Phase 2 SFs
            # For dV = P.T @ dO: SFP is operand A, SFDO is operand B
            temp_sfp_ptr = cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype)
            temp_tCtSFP = cute.make_tensor(temp_sfp_ptr, tCtSFP_layout)
            sfdo_dv_relative_offset = tcgen05.find_tmem_tensor_col_offset(temp_tCtSFP)

            # For dK = dS.T @ Q: SFDS is operand A, SFQ is operand B
            temp_sfds_ptr = cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype)
            temp_tCtSFDS = cute.make_tensor(temp_sfds_ptr, tCtSFDS_layout)
            sfq_dk_relative_offset = tcgen05.find_tmem_tensor_col_offset(temp_tCtSFDS)

            # Create Phase 2 TMEM tensors for dV GEMM SFs
            sfp_phase2_offset = self.tmem_S_offset + 56  # 56
            sfdo_dv_phase2_offset = sfp_phase2_offset + sfdo_dv_relative_offset

            sf_ptr_phase2 = cute.recast_ptr(
                tmem_ptr + sfp_phase2_offset, dtype=self.sf_dtype
            )
            tCtSFP = cute.make_tensor(sf_ptr_phase2, tCtSFP_layout)
            tCtSFDO_dV = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + sfdo_dv_phase2_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFDO_dV_layout,  # Use dV GEMM layout instead of dP GEMM layout
            )
            # SFDS and SFQ_dK placement for dK blockscaled GEMM.
            # dK SFs in dP region (col 336). dK fires before dP in the
            # mainloop, so the dP region is free. dP's zero_init clears stale SFs.
            sfds_phase2_offset = self.tmem_dP_offset + 80  # 336 (dP region)
            sfq_dk_phase2_offset = sfds_phase2_offset + sfq_dk_relative_offset

            sf_ptr_phase2_sfds = cute.recast_ptr(
                tmem_ptr + sfds_phase2_offset, dtype=self.sf_dtype
            )
            tCtSFDS = cute.make_tensor(sf_ptr_phase2_sfds, tCtSFDS_layout)
            tCtSFQ_dK = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + sfq_dk_phase2_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFQ_dK_layout,
            )

            # Create TMEM layouts for dQ GEMM (dQ = dS @ K)
            # SFDS_dQ is operand A (SFA), SFK_dQ is operand B (SFB)
            # Using proper layouts created for tiled_mma_dQ_bs and mma_tiler_dsk
            sSFDS_dQ_layout_per_stage = cute.slice_(
                sSFDS_dQ.layout, (None, None, None, 0)
            )
            tCtSFDS_dQ_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dQ_bs,
                self.mma_tiler_dsk,
                self.sf_vec_size,
                sSFDS_dQ_layout_per_stage,
            )
            sSFK_dQ_layout_per_stage = cute.slice_(
                sSFK_dQ.layout, (None, None, None, 0)
            )
            tCtSFK_dQ_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_dQ_bs,
                self.mma_tiler_dsk,
                self.sf_vec_size,
                sSFK_dQ_layout_per_stage,
            )

            # Get relative offset for SFK_dQ (operand B) from SFDS_dQ (operand A)
            temp_sfds_dq_ptr = cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype)
            temp_tCtSFDS_dQ = cute.make_tensor(temp_sfds_dq_ptr, tCtSFDS_dQ_layout)
            sfk_dq_relative_offset = tcgen05.find_tmem_tensor_col_offset(
                temp_tCtSFDS_dQ
            )

            # dQ GEMM SFs — S region (col 80).
            sfds_dq_phase2_offset = self.tmem_S_offset + 80  # 80 (S region)
            sfk_dq_phase2_offset = sfds_dq_phase2_offset + sfk_dq_relative_offset

            tCtSFDS_dQ = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + sfds_dq_phase2_offset, dtype=self.sf_dtype),
                tCtSFDS_dQ_layout,
            )
            tCtSFK_dQ = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + sfk_dq_phase2_offset, dtype=self.sf_dtype),
                tCtSFK_dQ_layout,
            )

        block_info = BlockInfo(
            # pyre-ignore[6]
            self.tile_m,
            # self.tile_n,
            # pyre-ignore[6]
            self.tile_n * self.cluster_shape_mnk[0],
            # pyre-ignore[6]
            self.is_causal,
            # pyre-ignore[6]
            self.is_local,
            # pyre-ignore[6]
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            # pyre-ignore[6]
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            # pyre-ignore[16]
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuSeqlensSFQ=mCuSeqlensSFQ,
            mCuSeqlensSFK=mCuSeqlensSFK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            broadcast_q=self.broadcast_q,
            mCuSeqlensO=mCuSeqlensO,
        )
        # pyre-ignore[16]
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            swap_AB=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )

        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        #  EPI
        # (14)
        if warp_idx == self.epi_warp_id:
            # currently no-op, could use for tma store/reduce
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                thr_mma_dQ,
                mQ,
                mK,
                mV,
                mLSE,
                mdPsum,
                mdO,
                sQ,
                sK,
                sV,
                sLSE,
                sdPsum,
                sdO,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                # SF TMA atoms and SMEM tensors
                tma_atom_SFQ,
                tma_tensor_SFQ,
                sSFQ,
                tma_atom_SFQ_dK,
                tma_tensor_SFQ_dK,
                sSFQ_dK,
                tma_atom_SFK,
                tma_tensor_SFK,
                sSFK,
                tma_atom_SFK_dQ,
                tma_tensor_SFK_dQ,
                sSFK_dQ,
                tma_atom_SFV,
                tma_tensor_SFV,
                sSFV,
                tma_atom_SFDO,
                tma_tensor_SFDO,
                sSFDO,
                tma_atom_SFDO_dV,
                tma_tensor_SFDO_dV,
                sSFDO_dV,
                # mdO_dV: M-block quantized dO for dV GEMM
                tma_tensor_dO_dV,  # Pass tma_tensor_dO_dV as mdO_dV parameter
                tma_atom_dO_dV,
                sdO_dV,
                # mQ_dK: M-block quantized Q for dK GEMM
                tma_tensor_Q_dK,
                tma_atom_Q_dK,
                sQ_dK,
                # mK_dQ: M-block quantized K for dQ GEMM
                tma_tensor_K_dQ,
                tma_atom_K_dQ,
                sK_dQ,
                pipeline_Q,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_K_done,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                should_load_Q=True,
                should_load_dO=True,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            cute.arch.sync_warp()

            # Note: SFP and SFDS are computed during S→P and dP→dS conversions
            # in the compute warp (compute_loop). The scales are written to SMEM
            # after being computed from the actual AMAX of P and dS values.
            # SFDS_dQ is also written in compute_loop with the same scale factors as SFDS.

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                tiled_mma_S_bs,
                tiled_mma_dP_bs,
                tiled_mma_dV_bs,
                tiled_mma_dK_bs,
                tiled_mma_dK_bs_smem,
                tiled_mma_dQ_bs,
                sQ,
                sQt,
                sK,
                sV,
                sdO,
                sdO_dV,  # M-block quantized dO for dV GEMM
                sQt_dK,  # M-block quantized Q transposed for dK GEMM
                sKt_dQ,  # M-block quantized K transposed for dQ GEMM
                sdOt,
                sdSt,
                sdS,
                sdS_dQ,  # Blockscaled A SMEM view for dQ GEMM
                sdS_dK,  # Blockscaled A SMEM view for dK GEMM
                sKt,
                tP,
                tdS,
                sP,  # SMEM P buffer for blockscaled dV GEMM
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                pipeline_Q.make_consumer(),
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                pipeline_K_done,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                # SF SMEM tensors for S2T copies to TMEM
                sSFQ,
                sSFK,
                sSFV,
                sSFDO,
                sSFDO_dV,
                # SF TMEM tensors for blockscaled MMA
                tCtSFK,
                tCtSFQ,
                tCtSFV,
                tCtSFDO,
                tCtSFK_prologue,
                tCtSFQ_prologue,
                tCtSFV_prologue,
                tCtSFDO_prologue,
                # Phase 2: SF tensors (SFP computed from P, SFDS computed from dS)
                sSFP,
                sSFDS,
                tCtSFP,
                tCtSFDO_dV,
                tCtSFDS,
                tCtSFQ_dK,
                sSFQ_dK,
                # dQ GEMM scale factors
                sSFDS_dQ,
                sSFK_dQ,
                tCtSFDS_dQ,
                tCtSFK_dQ,
            )
            cute.arch.relinquish_tmem_alloc_permit()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )

            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols, is_two_cta=False)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if (
            warp_idx >= self.compute_warp_ids[0]
            and warp_idx <= self.compute_warp_ids[-1]
        ):
            cute.arch.warpgroup_reg_alloc(self.num_regs_compute)  # 8 warps
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                sLSE,
                sdPsum,
                tdVtdV,
                tdKtdK,
                mdV,
                mdK,
                sdS,
                sdS_dQ,  # Blockscaled A SMEM view for dQ GEMM
                sdS_dK,  # Blockscaled A SMEM view for dK GEMM
                sdSt,  # N-contiguous sdS SMEM for compute warp R2S write
                tdPtdP,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
                mdK_semaphore,
                mdV_semaphore,
                sSFP,
                sSFDS,
                sSFDS_dQ,
                sP,  # SMEM P buffer for blockscaled dV GEMM
                mdSFK_out,
                mdSFV_out,
                mAttnScale,  # SiLU: per-row attention scale
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_reduce)
            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mdQ_semaphore,
            )

        return

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        thr_mma_dQ: cute.core.ThrMma,  # For SFK_dQ TMA partition
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdO: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        # SF TMA atoms and SMEM tensors
        tma_atom_SFQ: Optional[cute.CopyAtom],
        tma_tensor_SFQ: Optional[cute.Tensor],
        sSFQ: Optional[cute.Tensor],
        tma_atom_SFQ_dK: Optional[cute.CopyAtom],
        tma_tensor_SFQ_dK: Optional[cute.Tensor],
        sSFQ_dK: Optional[cute.Tensor],
        tma_atom_SFK: Optional[cute.CopyAtom],
        tma_tensor_SFK: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        tma_atom_SFK_dQ: Optional[cute.CopyAtom],  # SFK for dQ GEMM
        tma_tensor_SFK_dQ: Optional[cute.Tensor],  # SFK for dQ GEMM
        sSFK_dQ: Optional[cute.Tensor],  # SFK for dQ GEMM
        tma_atom_SFV: Optional[cute.CopyAtom],
        tma_tensor_SFV: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        tma_atom_SFDO: Optional[cute.CopyAtom],
        tma_tensor_SFDO: Optional[cute.Tensor],
        sSFDO: Optional[cute.Tensor],
        tma_atom_SFDO_dV: Optional[cute.CopyAtom],  # SFDO for dV GEMM
        tma_tensor_SFDO_dV: Optional[cute.Tensor],  # SFDO for dV GEMM
        sSFDO_dV: Optional[cute.Tensor],  # SFDO for dV GEMM
        # mdO_dV: M-block quantized dO for dV GEMM
        mdO_dV: Optional[cute.Tensor],
        tma_atom_dO_dV: Optional[cute.CopyAtom],
        sdO_dV: Optional[cute.Tensor],
        # mQ_dK: M-block quantized Q for dK GEMM
        mQ_dK: Optional[cute.Tensor],
        tma_atom_Q_dK: Optional[cute.CopyAtom],
        sQ_dK: Optional[cute.Tensor],
        # mK_dQ: M-block quantized K for dQ GEMM
        mK_dQ: Optional[cute.Tensor],
        tma_atom_K_dQ: Optional[cute.CopyAtom],
        sK_dQ: Optional[cute.Tensor],
        pipeline_Q: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_K_done: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ):
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer,
            # pyre-ignore[16]
            self.Q_stage,
        )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer,
            # pyre-ignore[16]
            self.dO_stage,
        )

        # Compute multicast mask for Q & dO buffer full
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        # pyre-ignore[16]
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        q_do_mcast_mask = None
        # pyre-ignore[16]
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # pyre-ignore[29]
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        is_first_tile = True
        K_done_phase = Int32(0)
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            # pyre-ignore[29]
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen,
                # pyre-ignore[16]
                n_block // self.cluster_shape_mnk[0],
            )
            head_idx_kvv = head_idx // self.qhead_per_kvhead
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                None, None, head_idx_kvv
            ]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                None, None, head_idx_kvv
            ]
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdO_cur = mdO[None, None, head_idx, batch_idx]
            else:
                mdO_cur = cute.domain_offset(
                    (0, seqlen.offset_o), mdO[None, None, head_idx]
                )
            mLSE_cur = seqlen.offset_batch_O(mLSE, batch_idx, dim=2, padded=True)[
                None, head_idx
            ]
            mPsum_cur = seqlen.offset_batch_O(mdPsum, batch_idx, dim=2, padded=True)[
                None, head_idx
            ]

            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0)
            )
            tSgK = thr_mma_S.partition_A(gK)
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, 0)
            )
            tdPgV = thr_mma_dP.partition_A(gV)
            gQ = cute.local_tile(
                mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0)
            )
            tSgQ = thr_mma_S.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mPsum_cur, (self.tile_m,), (None,))
            gdO = cute.local_tile(
                mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None)
            )
            tdPgdO = thr_mma_dV.partition_B(gdO)

            # gdO_dV: M-block quantized dO for dV GEMM
            gdO_dV, tdPgdO_dV = None, None
            if const_expr(self.blockscaled and mdO_dV is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    # pyre-ignore[16]
                    mdO_dV_cur = mdO_dV[None, None, head_idx, batch_idx]
                else:
                    mdO_dV_cur = cute.domain_offset(
                        (0, seqlen.offset_o), mdO_dV[None, None, head_idx]
                    )
                gdO_dV = cute.local_tile(
                    mdO_dV_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None)
                )
                tdPgdO_dV = thr_mma_dV.partition_B(gdO_dV)

            # gQ_dK: M-block quantized Q for dK GEMM
            # Use S GEMM params (mma_tiler_kq, thr_mma_S) to match TMA atom setup
            # TMA atom was created with make_tiled_tma_atom_B using tiled_mma_S
            gQ_dK, tdKgQ_dK = None, None
            if const_expr(self.blockscaled and mQ_dK is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mQ_dK_cur = mQ_dK[None, None, head_idx, batch_idx]
                else:
                    mQ_dK_cur = cute.domain_offset(
                        (seqlen.offset_q, 0), mQ_dK[None, None, head_idx]
                    )
                gQ_dK = cute.local_tile(
                    mQ_dK_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0)
                )
                tdKgQ_dK = thr_mma_S.partition_B(gQ_dK)

            # gK_dQ: M-block quantized K for dQ GEMM
            # K is loaded as A operand (same pattern as original K)
            gK_dQ, tdQgK_dQ = None, None
            if const_expr(self.blockscaled and mK_dQ is not None):
                mK_dQ_cur = seqlen.offset_batch_K(mK_dQ, batch_idx, dim=3)[
                    None, None, head_idx_kvv
                ]
                gK_dQ = cute.local_tile(
                    mK_dQ_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0)
                )
                tdQgK_dQ = thr_mma_S.partition_A(gK_dQ)

            # Define b_cta_layout early for SF TMA partition (also used later for Q/dO)
            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )

            # TMA partitions for scale factors (SFQ, SFQ_dK, SFK, SFK_dQ, SFV, SFDO)
            tSFQsSFQ, tSFQgSFQ = None, None
            tSFQ_dKsSFQ_dK, tSFQ_dKgSFQ_dK = None, None  # SFQ for dK GEMM
            tSFKsSFK, tSFKgSFK = None, None
            tSFK_dQsSFK_dQ, tSFK_dQgSFK_dQ = None, None  # SFK for dQ GEMM
            tSFVsSFV, tSFVgSFV = None, None
            tSFDOsSFDO, tSFDOgSFDO = None, None

            if const_expr(self.blockscaled):
                # SFQ TMA partition — offset-at-copy-time for B>1 varlen fix
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mSFQ_cur = tma_tensor_SFQ[None, None, head_idx, batch_idx]
                    sfq_m_offset = Int32(0)
                else:
                    mSFQ_cur = tma_tensor_SFQ[None, None, head_idx]
                    sfq_m_offset = seqlen.offset_sf_q // self.tile_m
                gSFQ = cute.local_tile(
                    mSFQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0)
                )
                tSgSFQ = thr_mma_S.partition_B(gSFQ)
                tSFQsSFQ, tSFQgSFQ = cpasync.tma_partition(
                    tma_atom_SFQ,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFQ, 0, 3),
                    cute.group_modes(tSgSFQ, 0, 3),
                )
                tSFQsSFQ = cute.filter_zeros(tSFQsSFQ)
                tSFQgSFQ = cute.filter_zeros(tSFQgSFQ)

                # SFQ_dK — domain_offset for batch (transposed shape, like SFDO_dV)
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mSFQ_dK_cur = tma_tensor_SFQ_dK[None, None, head_idx, batch_idx]
                else:
                    mSFQ_dK_cur = cute.domain_offset(
                        (0, seqlen.offset_sf_q), tma_tensor_SFQ_dK[None, None, head_idx]
                    )
                gSFQ_dK = cute.local_tile(
                    mSFQ_dK_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None)
                )
                tSgSFQ_dK = thr_mma_dK.partition_B(gSFQ_dK)
                tSFQ_dKsSFQ_dK, tSFQ_dKgSFQ_dK = cpasync.tma_partition(
                    tma_atom_SFQ_dK,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFQ_dK, 0, 3),
                    cute.group_modes(tSgSFQ_dK, 0, 3),
                )
                tSFQ_dKsSFQ_dK = cute.filter_zeros(tSFQ_dKsSFQ_dK)
                tSFQ_dKgSFQ_dK = cute.filter_zeros(tSFQ_dKgSFQ_dK)

                # SFK TMA partition (K's scale factors, operand A for S = K @ Q.T)
                # Use (None, 0) to keep all n blocks, then index when copying (like fwd)
                # SFK — offset-at-copy-time
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mSFK_cur = tma_tensor_SFK[None, None, head_idx_kvv, batch_idx]
                    sfk_n_offset = Int32(0)
                else:
                    mSFK_cur = tma_tensor_SFK[None, None, head_idx_kvv]
                    sfk_n_offset = seqlen.offset_sf_k // self.tile_n
                gSFK = cute.local_tile(
                    mSFK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (None, 0)
                )
                tSgSFK = thr_mma_S.partition_A(gSFK)
                tSFKsSFK, tSFKgSFK = cpasync.tma_partition(
                    tma_atom_SFK,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFK, 0, 3),
                    cute.group_modes(tSgSFK, 0, 3),
                )
                tSFKsSFK = cute.filter_zeros(tSFKsSFK)
                tSFKgSFK = cute.filter_zeros(tSFKgSFK)

                # SFK_dQ TMA partition (K's scale factors for dQ GEMM, operand B for dQ = dS @ K)
                # Separate partition from SFK because dQ uses different MMA/tiler
                # SFK_dQ — offset-at-copy-time
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mSFK_dQ_cur = tma_tensor_SFK_dQ[None, None, head_idx_kvv, batch_idx]
                    sfk_dq_n_offset = Int32(0)
                else:
                    mSFK_dQ_cur = tma_tensor_SFK_dQ[None, None, head_idx_kvv]
                    sfk_dq_n_offset = seqlen.offset_sf_k // self.tile_n
                gSFK_dQ = cute.local_tile(
                    mSFK_dQ_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, None)
                )
                tSgSFK_dQ = thr_mma_dQ.partition_B(gSFK_dQ)
                tSFK_dQsSFK_dQ, tSFK_dQgSFK_dQ = cpasync.tma_partition(
                    tma_atom_SFK_dQ,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFK_dQ, 0, 3),
                    cute.group_modes(tSgSFK_dQ, 0, 3),
                )
                tSFK_dQsSFK_dQ = cute.filter_zeros(tSFK_dQsSFK_dQ)
                tSFK_dQgSFK_dQ = cute.filter_zeros(tSFK_dQgSFK_dQ)

                # SFV TMA partition (V's scale factors, operand A for dP = V @ dO.T)
                # Use (None, 0) to keep all n blocks, then index when copying (like fwd)
                # SFV — offset-at-copy-time
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mSFV_cur = tma_tensor_SFV[None, None, head_idx_kvv, batch_idx]
                    sfv_n_offset = Int32(0)
                else:
                    mSFV_cur = tma_tensor_SFV[None, None, head_idx_kvv]
                    sfv_n_offset = seqlen.offset_sf_k // self.tile_n
                gSFV = cute.local_tile(
                    mSFV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (None, 0)
                )
                tSgSFV = thr_mma_dP.partition_A(gSFV)
                tSFVsSFV, tSFVgSFV = cpasync.tma_partition(
                    tma_atom_SFV,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFV, 0, 3),
                    cute.group_modes(tSgSFV, 0, 3),
                )
                tSFVsSFV = cute.filter_zeros(tSFVsSFV)
                tSFVgSFV = cute.filter_zeros(tSFVgSFV)

                # SFDO — offset-at-copy-time
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mSFDO_cur = tma_tensor_SFDO[None, None, head_idx, batch_idx]
                    sfdo_m_offset = Int32(0)
                else:
                    mSFDO_cur = tma_tensor_SFDO[None, None, head_idx]
                    sfdo_m_offset = seqlen.offset_o // self.tile_m
                gSFDO = cute.local_tile(
                    mSFDO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None)
                )
                tSgSFDO = thr_mma_dP.partition_B(gSFDO)
                tSFDOsSFDO, tSFDOgSFDO = cpasync.tma_partition(
                    tma_atom_SFDO,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFDO, 0, 3),
                    cute.group_modes(tSgSFDO, 0, 3),
                )
                tSFDOsSFDO = cute.filter_zeros(tSFDOsSFDO)
                tSFDOgSFDO = cute.filter_zeros(tSFDOgSFDO)

                # SFDO_dV — keep original offset_batch pattern (transposed tensor)
                # NOTE: Use O-side offset (offset_o), not Q-side (offset_sf_q).
                # SFDO_dV follows dO data layout which is O-side in broadcast_q.
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mSFDO_dV_cur = tma_tensor_SFDO_dV[None, None, head_idx, batch_idx]
                else:
                    mSFDO_dV_cur = cute.domain_offset(
                        (0, seqlen.offset_o), tma_tensor_SFDO_dV[None, None, head_idx]
                    )
                gSFDO_dV = cute.local_tile(
                    mSFDO_dV_cur,
                    cute.select(self.mma_tiler_pdo, mode=[1, 2]),
                    (0, None),
                )
                tSgSFDO_dV = thr_mma_dV.partition_B(gSFDO_dV)
                tSFDO_dVsSFDO_dV, tSFDO_dVgSFDO_dV = cpasync.tma_partition(
                    tma_atom_SFDO_dV,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFDO_dV, 0, 3),
                    cute.group_modes(tSgSFDO_dV, 0, 3),
                )
                tSFDO_dVsSFDO_dV = cute.filter_zeros(tSFDO_dVsSFDO_dV)
                tSFDO_dVgSFDO_dV = cute.filter_zeros(tSFDO_dVgSFDO_dV)

            # pyre-ignore[23]
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), tSgK, sK, single_stage=True
            )
            # pyre-ignore[23]
            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
            )
            # pyre-ignore[23]
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
            # pyre-ignore[23]
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdPgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

            # load_dO_dV: M-block quantized dO for dV GEMM
            load_dO_dV = None
            if const_expr(self.blockscaled and tma_atom_dO_dV is not None):
                # pyre-ignore[23]
                load_dO_dV, _, _ = copy_utils.tma_get_copy_fn(
                    # pyre-ignore[6]
                    tma_atom_dO_dV,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdPgdO_dV,
                    # pyre-ignore[6]
                    dst_tensor=sdO_dV,
                    mcast_mask=q_do_mcast_mask,
                )
                load_dO_dV = copy_utils.tma_producer_copy_fn(load_dO_dV, pipeline_dO)

            # load_Q_dK: M-block quantized Q for dK GEMM
            # Must use same condition as tdKgQ_dK creation (mQ_dK is not None)
            # and also check that tdKgQ_dK and sQ_dK are not None
            load_Q_dK = None
            if const_expr(self.blockscaled and mQ_dK is not None and sQ_dK is not None):
                # pyre-ignore[23]
                load_Q_dK, _, _ = copy_utils.tma_get_copy_fn(
                    # pyre-ignore[6]
                    tma_atom_Q_dK,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdKgQ_dK,
                    # pyre-ignore[6]
                    dst_tensor=sQ_dK,
                    mcast_mask=q_do_mcast_mask,
                )
                load_Q_dK = copy_utils.tma_producer_copy_fn(load_Q_dK, pipeline_Q)

            # load_K_dQ: M-block quantized K for dQ GEMM
            # K is loaded as A operand (same pattern as original K load)
            load_K_dQ = None
            if const_expr(self.blockscaled and tma_atom_K_dQ is not None):
                # pyre-ignore[23]
                load_K_dQ, _, _ = copy_utils.tma_get_copy_fn(
                    # pyre-ignore[6]
                    tma_atom_K_dQ,
                    0,
                    cute.make_layout(1),
                    tdQgK_dQ,
                    # pyre-ignore[6]
                    sK_dQ,
                    single_stage=True,
                )

            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(cute.copy, copy_atom_stats)
            # copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SMulticastOp(), Float32)
            # sLSE = cute.logical_divide(sLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gLSE = cute.logical_divide(gLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # sdPsum = cute.logical_divide(sdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gdPsum = cute.logical_divide(gdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # copy_stats = partial(cute.copy, copy_atom_stats, mcast_mask=q_do_mcast_mask)

            if const_expr(not self.is_local) or m_block_min < m_block_max:
                if const_expr(self.is_persistent):
                    if not is_first_tile:
                        # pyre-ignore[19]
                        pipeline_K_done.sync_object_full.wait(0, K_done_phase)
                        K_done_phase ^= 1
                    is_first_tile = False
                # First iteration: load K together w Q & LSE, then V together w dO & dPsum
                if const_expr(should_load_Q):
                    # K bundled with Q via pipeline_Q
                    extra_tx_K_Q = (
                        # pyre-ignore[16]
                        self.tma_copy_bytes["K"]
                        + self.tma_copy_bytes["Q_dK"]
                        + self.tma_copy_bytes.get("K_dQ", 0)
                    )
                    pipeline_Q.producer_acquire(
                        producer_state_Q_LSE, extra_tx_count=extra_tx_K_Q
                    )
                    load_K(
                        tma_bar_ptr=pipeline_Q.producer_get_barrier(
                            producer_state_Q_LSE
                        )
                    )
                    if const_expr(load_K_dQ is not None):
                        load_K_dQ(
                            tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                producer_state_Q_LSE
                            )
                        )
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tma_atom_SFK,
                            # pyre-ignore[61]
                            tSFKgSFK[None, n_block + sfk_n_offset],
                            tSFKsSFK[None, 0],
                            tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                producer_state_Q_LSE
                            ),
                        )
                        cute.copy(
                            tma_atom_SFK_dQ,
                            # pyre-ignore[61]
                            tSFK_dQgSFK_dQ[None, n_block + sfk_dq_n_offset],
                            tSFK_dQsSFK_dQ[None, 0],
                            tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                producer_state_Q_LSE
                            ),
                        )
                    load_Q(m_block_min, producer_state=producer_state_Q_LSE)
                    # Load SFQ together with Q
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tma_atom_SFQ,
                            # pyre-ignore[61]
                            tSFQgSFQ[None, m_block_min + sfq_m_offset],
                            tSFQsSFQ[None, producer_state_Q_LSE.index],
                            tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                producer_state_Q_LSE
                            ),
                        )
                        # Load SFQ_dK together with SFQ (for dK GEMM)
                        cute.copy(
                            tma_atom_SFQ_dK,
                            tSFQ_dKgSFQ_dK[None, m_block_min],
                            tSFQ_dKsSFQ_dK[None, producer_state_Q_LSE.index],
                            tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                producer_state_Q_LSE
                            ),
                        )
                        # Load M-block quantized Q for dK GEMM
                        if const_expr(load_Q_dK is not None):
                            load_Q_dK(m_block_min, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)
                    if const_expr(not self.use_silu):
                        pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                        with cute.arch.elect_one():
                            copy_stats(
                                gLSE[None, m_block_min],
                                sLSE[None, producer_state_Q_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(
                                    producer_state_Q_LSE
                                ),
                            )
                    producer_state_Q_LSE.advance()
                if const_expr(should_load_dO):
                    # V bundled with dO via pipeline_dO
                    extra_tx = self.tma_copy_bytes["V"] + self.tma_copy_bytes["dO_dV"]
                    pipeline_dO.producer_acquire(
                        producer_state_dO_dPsum, extra_tx_count=extra_tx
                    )
                    load_V(
                        tma_bar_ptr=pipeline_dO.producer_get_barrier(
                            producer_state_dO_dPsum
                        )
                    )
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tma_atom_SFV,
                            # pyre-ignore[61]
                            tSFVgSFV[None, n_block + sfv_n_offset],
                            tSFVsSFV[None, 0],
                            tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                producer_state_dO_dPsum
                            ),
                        )
                    load_dO(m_block_min, producer_state=producer_state_dO_dPsum)
                    # Load dO_dV: M-block quantized dO for dV GEMM
                    if const_expr(self.blockscaled and load_dO_dV is not None):
                        load_dO_dV(m_block_min, producer_state=producer_state_dO_dPsum)
                    # Load SFDO together with dO
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tma_atom_SFDO,
                            # pyre-ignore[61]
                            tSFDOgSFDO[None, m_block_min + sfdo_m_offset],
                            tSFDOsSFDO[None, producer_state_dO_dPsum.index],
                            tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                producer_state_dO_dPsum
                            ),
                        )
                        # Also load SFDO_dV for dV GEMM (separate layout)
                        cute.copy(
                            tma_atom_SFDO_dV,
                            # pyre-ignore[61]
                            tSFDO_dVgSFDO_dV[None, m_block_min],
                            # pyre-ignore[61]
                            tSFDO_dVsSFDO_dV[None, producer_state_dO_dPsum.index],
                            tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                producer_state_dO_dPsum
                            ),
                        )
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)
                    if const_expr(not self.use_silu):
                        pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                        with cute.arch.elect_one():
                            copy_stats(
                                gdPsum[None, m_block_min],
                                sdPsum[None, producer_state_dO_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                    producer_state_dO_dPsum
                                ),
                            )
                    producer_state_dO_dPsum.advance()

                # pyre-ignore[28]
                for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                    if const_expr(should_load_Q):
                        # Q (with SFQ for blockscaled, and Q_dK for dK GEMM)
                        extra_tx_Q_main = self.tma_copy_bytes["Q_dK"]
                        pipeline_Q.producer_acquire(
                            producer_state_Q_LSE, extra_tx_count=extra_tx_Q_main
                        )
                        load_Q(m_block, producer_state=producer_state_Q_LSE)
                        # Load SFQ together with Q
                        if const_expr(self.blockscaled):
                            cute.copy(
                                tma_atom_SFQ,
                                # pyre-ignore[58, 61]
                                tSFQgSFQ[None, m_block + sfq_m_offset],
                                tSFQsSFQ[None, producer_state_Q_LSE.index],
                                tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                    producer_state_Q_LSE
                                ),
                            )
                            # Load SFQ_dK together with SFQ (for dK GEMM)
                            cute.copy(
                                tma_atom_SFQ_dK,
                                tSFQ_dKgSFQ_dK[None, m_block],
                                tSFQ_dKsSFQ_dK[None, producer_state_Q_LSE.index],
                                tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                    producer_state_Q_LSE
                                ),
                            )
                            # Load M-block quantized Q for dK GEMM
                            if const_expr(load_Q_dK is not None):
                                load_Q_dK(m_block, producer_state=producer_state_Q_LSE)
                        pipeline_Q.producer_commit(producer_state_Q_LSE)
                        if const_expr(not self.use_silu):
                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gLSE[None, m_block],
                                    sLSE[None, producer_state_Q_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(
                                        producer_state_Q_LSE
                                    ),
                                )
                        producer_state_Q_LSE.advance()
                    if const_expr(should_load_dO):
                        # dO (with SFDO for blockscaled)
                        extra_tx_main = self.tma_copy_bytes["dO_dV"]
                        pipeline_dO.producer_acquire(
                            producer_state_dO_dPsum, extra_tx_count=extra_tx_main
                        )
                        load_dO(m_block, producer_state=producer_state_dO_dPsum)
                        # Load dO_dV: M-block quantized dO for dV GEMM
                        if const_expr(self.blockscaled and load_dO_dV is not None):
                            load_dO_dV(m_block, producer_state=producer_state_dO_dPsum)
                        # Load SFDO together with dO
                        if const_expr(self.blockscaled):
                            cute.copy(
                                tma_atom_SFDO,
                                # pyre-ignore[58, 61]
                                tSFDOgSFDO[None, m_block + sfdo_m_offset],
                                tSFDOsSFDO[None, producer_state_dO_dPsum.index],
                                tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                    producer_state_dO_dPsum
                                ),
                            )
                            # Also load SFDO_dV for dV GEMM (separate layout)
                            cute.copy(
                                tma_atom_SFDO_dV,
                                # pyre-ignore[61]
                                tSFDO_dVgSFDO_dV[None, m_block],
                                # pyre-ignore[61]
                                tSFDO_dVsSFDO_dV[None, producer_state_dO_dPsum.index],
                                tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                    producer_state_dO_dPsum
                                ),
                            )
                        pipeline_dO.producer_commit(producer_state_dO_dPsum)
                        if const_expr(not self.use_silu):
                            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gdPsum[None, m_block],
                                    sdPsum[None, producer_state_dO_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    ),
                                )
                        producer_state_dO_dPsum.advance()

                if const_expr(not self.is_persistent):
                    if const_expr(should_load_Q):
                        pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                        if const_expr(not self.use_silu):
                            pipeline_LSE.producer_tail(producer_state_Q_LSE)
                    if const_expr(should_load_dO):
                        pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                        if const_expr(not self.use_silu):
                            pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        # Persistent: call producer_tail after all N-blocks are done
        if const_expr(self.is_persistent):
            if const_expr(should_load_Q):
                pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                if const_expr(not self.use_silu):
                    pipeline_LSE.producer_tail(producer_state_Q_LSE)
            if const_expr(should_load_dO):
                pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                if const_expr(not self.use_silu):
                    pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_mma_S_bs: Optional[cute.TiledMma],  # Blockscaled MMA for S = K @ Q.T
        tiled_mma_dP_bs: Optional[cute.TiledMma],  # Blockscaled MMA for dP = V @ dO.T
        tiled_mma_dV_bs: Optional[cute.TiledMma],  # Blockscaled MMA for dV = P.T @ dO
        tiled_mma_dK_bs: Optional[
            cute.TiledMma
        ],  # Blockscaled MMA for dK SF layouts (TMEM)
        tiled_mma_dK_bs_smem: Optional[
            cute.TiledMma
        ],  # Blockscaled MMA for dK GEMM (SMEM)
        tiled_mma_dQ_bs: Optional[cute.TiledMma],  # Blockscaled MMA for dQ = dS @ K
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdO_dV: Optional[cute.Tensor],  # M-block quantized dO for dV GEMM
        sQt_dK: Optional[cute.Tensor],  # M-block quantized Q transposed for dK GEMM
        sKt_dQ: Optional[cute.Tensor],  # M-block quantized K transposed for dQ GEMM
        sdOt: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        sdS_dQ: Optional[cute.Tensor],  # Blockscaled A SMEM view for dQ GEMM
        sdS_dK: Optional[cute.Tensor],  # Blockscaled A SMEM view for dK GEMM
        sKt: cute.Tensor,
        tP: cute.Tensor,
        tdS: cute.Tensor,
        sP: Optional[cute.Tensor],  # SMEM P buffer for blockscaled dV GEMM
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        pipeline_Q_consumer: PipelineConsumer,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        pipeline_K_done: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        # SF SMEM tensors for S2T copies to TMEM
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        sSFDO: Optional[cute.Tensor],
        sSFDO_dV: Optional[
            cute.Tensor
        ],  # dO's scale factors for dV GEMM (separate layout)
        # SF TMEM tensors for blockscaled MMA
        tCtSFK: Optional[cute.Tensor],
        tCtSFQ: Optional[cute.Tensor],
        tCtSFV: Optional[cute.Tensor],
        tCtSFDO: Optional[cute.Tensor],
        tCtSFK_prologue: Optional[cute.Tensor],
        tCtSFQ_prologue: Optional[cute.Tensor],
        tCtSFV_prologue: Optional[cute.Tensor],
        tCtSFDO_prologue: Optional[cute.Tensor],
        # Phase 2: Constant SF tensors (SFP=1.0, SFDS=1.0)
        sSFP: Optional[cute.Tensor],
        sSFDS: Optional[cute.Tensor],
        tCtSFP: Optional[cute.Tensor],
        tCtSFDO_dV: Optional[cute.Tensor],
        tCtSFDS: Optional[cute.Tensor],
        tCtSFQ_dK: Optional[cute.Tensor],
        sSFQ_dK: Optional[cute.Tensor],
        # dQ GEMM scale factors (separate layouts for dQ = dS @ K)
        sSFDS_dQ: Optional[cute.Tensor],
        sSFK_dQ: Optional[cute.Tensor],
        tCtSFDS_dQ: Optional[cute.Tensor],
        tCtSFK_dQ: Optional[cute.Tensor],
    ) -> None:
        # Partition smem / tmem tensors
        # S = K @ Q.T
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dO.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        if const_expr(self.use_smem_dS_for_mma_dK):
            tdKrdS = tiled_mma_dK.make_fragment_A(sdSt)
            tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        else:
            if const_expr(self.blockscaled):
                # Blockscaled dK: A operand (dS) from SMEM via blockscaled layout
                # pyre-ignore[16]
                tdKrdS = tiled_mma_dK_bs_smem.make_fragment_A(sdS_dK)
                # B operand (Q) uses non-blockscaled MMA layout
                tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
                # tdKrQ_dK: M-block quantized Q for dK GEMM (blockscaled)
                if const_expr(sQt_dK is not None):
                    tdKrQ_dK = tiled_mma_dK.make_fragment_B(sQt_dK)
                else:
                    tdKrQ_dK = tdKrQ  # fallback to K-block quantized Q
            else:
                tdKrdS = tiled_mma_dK.make_fragment_A(tdS)
                tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
                tdKrQ_dK = tdKrQ  # non-blockscaled doesn't need separate Q
        # dQ = dS @ K
        if const_expr(self.blockscaled):
            # Use non-blockscaled MMA for SMEM fragments
            tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
            tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
            # Blockscaled path: fragment A from blockscaled MMA with blockscaled SMEM view
            if const_expr(sdS_dQ is not None):
                tdQrdS_bs = tiled_mma_dQ_bs.make_fragment_A(sdS_dQ)
            else:
                tdQrdS_bs = tiled_mma_dQ_bs.make_fragment_A(sdS)
            # tdQrK_dQ: M-block quantized K for dQ GEMM (blockscaled)
            if const_expr(sKt_dQ is not None):
                # pyre-ignore[16]
                tdQrK_dQ = tiled_mma_dQ_bs.make_fragment_B(sKt_dQ)
            else:
                tdQrK_dQ = tiled_mma_dQ_bs.make_fragment_B(sKt)  # fallback
        else:
            tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
            tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
            tdQrK_dQ = tdQrK  # non-blockscaled doesn't need separate K
        # dV = P @ dO.T
        if const_expr(self.blockscaled):
            tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
            # tdVrdO_dV: M-block quantized dO for dV GEMM (blockscaled)
            # Use sdO_dV if available, otherwise fall back to sdO
            if const_expr(sdO_dV is not None):
                tdVrdO_dV = tiled_mma_dV.make_fragment_B(sdO_dV)
            else:
                tdVrdO_dV = tdVrdO  # fallback to K-block quantized dO
            # Issue 008: P from TMEM via R2T
            tdVrP = tiled_mma_dV_bs.make_fragment_A(tP)
        else:
            tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
            tdVrdO_dV = tdVrdO  # non-blockscaled doesn't need separate dO
            tdVrP = tiled_mma_dV.make_fragment_A(tP)

        # S2T copy partitions for scale factors (blockscaled MMA)
        tiled_copy_s2t_sfk, tCsSFK_s2t, tCtSFK_s2t = None, None, None
        tiled_copy_s2t_sfq, tCsSFQ_s2t, tCtSFQ_s2t = None, None, None
        tiled_copy_s2t_sfv, tCsSFV_s2t, tCtSFV_s2t = None, None, None
        tiled_copy_s2t_sfdo, tCsSFDO_s2t, tCtSFDO_s2t = None, None, None
        # Prologue partitions (different TMEM region)
        tCtSFK_s2t_prologue, tCtSFQ_s2t_prologue = None, None
        tCtSFV_s2t_prologue, tCtSFDO_s2t_prologue = None, None

        if const_expr(self.blockscaled):
            # S2T copy partitions for S = K @ Q.T (main loop)
            tiled_copy_s2t_sfk, tCsSFK_s2t, tCtSFK_s2t = make_s2t_copy_partitions(
                # pyre-ignore[6]
                sSFK,
                # pyre-ignore[6]
                tCtSFK,
                self.sf_dtype,
            )
            tiled_copy_s2t_sfq, tCsSFQ_s2t, tCtSFQ_s2t = make_s2t_copy_partitions(
                # pyre-ignore[6]
                sSFQ,
                # pyre-ignore[6]
                tCtSFQ,
                self.sf_dtype,
            )

            # S2T copy partitions for dP = V @ dO.T (main loop)
            tiled_copy_s2t_sfv, tCsSFV_s2t, tCtSFV_s2t = make_s2t_copy_partitions(
                # pyre-ignore[6]
                sSFV,
                # pyre-ignore[6]
                tCtSFV,
                self.sf_dtype,
            )
            tiled_copy_s2t_sfdo, tCsSFDO_s2t, tCtSFDO_s2t = make_s2t_copy_partitions(
                # pyre-ignore[6]
                sSFDO,
                # pyre-ignore[6]
                tCtSFDO,
                self.sf_dtype,
            )

            # Prologue partitions (use dK region - same tiled_copy, different TMEM partitions)
            _, _, tCtSFK_s2t_prologue = make_s2t_copy_partitions(
                # pyre-ignore[6]
                sSFK,
                # pyre-ignore[6]
                tCtSFK_prologue,
                self.sf_dtype,
            )
            _, _, tCtSFQ_s2t_prologue = make_s2t_copy_partitions(
                # pyre-ignore[6]
                sSFQ,
                # pyre-ignore[6]
                tCtSFQ_prologue,
                self.sf_dtype,
            )
            _, _, tCtSFV_s2t_prologue = make_s2t_copy_partitions(
                # pyre-ignore[6]
                sSFV,
                # pyre-ignore[6]
                tCtSFV_prologue,
                self.sf_dtype,
            )
            _, _, tCtSFDO_s2t_prologue = make_s2t_copy_partitions(
                # pyre-ignore[6]
                sSFDO,
                # pyre-ignore[6]
                tCtSFDO_prologue,
                self.sf_dtype,
            )

            # Phase 2: S2T copy partitions for SFP and SFDS (computed dynamically)
            # dV = P.T @ dO: needs SFP (computed from P's AMAX), SFDO (from input)
            tiled_copy_s2t_sfp, tCsSFP_s2t, tCtSFP_s2t = make_s2t_copy_partitions(
                # pyre-ignore[6]
                sSFP,
                # pyre-ignore[6]
                tCtSFP,
                self.sf_dtype,
            )
            tiled_copy_s2t_sfdo_dv, tCsSFDO_dV_s2t, tCtSFDO_dV_s2t = (
                # pyre-ignore[6]
                make_s2t_copy_partitions(sSFDO_dV, tCtSFDO_dV, self.sf_dtype)
            )
            # dK = dS.T @ Q: needs SFDS (constant 1.0), SFQ (from input)
            tiled_copy_s2t_sfds, tCsSFDS_s2t, tCtSFDS_s2t = make_s2t_copy_partitions(
                # pyre-ignore[6]
                sSFDS,
                # pyre-ignore[6]
                tCtSFDS,
                self.sf_dtype,
            )
            tiled_copy_s2t_sfq_dk, tCsSFQ_dK_s2t, tCtSFQ_dK_s2t = (
                # pyre-ignore[6]
                make_s2t_copy_partitions(sSFQ_dK, tCtSFQ_dK, self.sf_dtype)
            )

            # dQ GEMM S2T partition creation (reusing dK layouts - workaround for JIT hang)
            # dQ = dS @ K: needs SFDS_dQ (constant 1.0), SFK_dQ (from input)
            tiled_copy_s2t_sfds_dq, tCsSFDS_dQ_s2t, tCtSFDS_dQ_s2t = (
                # pyre-ignore[6]
                make_s2t_copy_partitions(sSFDS_dQ, tCtSFDS_dQ, self.sf_dtype)
            )
            tiled_copy_s2t_sfk_dq, tCsSFK_dQ_s2t, tCtSFK_dQ_s2t = (
                # pyre-ignore[6]
                make_s2t_copy_partitions(sSFK_dQ, tCtSFK_dQ, self.sf_dtype)
            )

        # S2T for P data removed entirely (Issue 008+010: JIT dead-code corruption)

        mma_qk_fn = partial(
            gemm_ptx_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, sA=sK, sB=sQ, zero_init=True
        )
        mma_dov_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
        )
        mma_pdo_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            sA=None,
            sB=sdO,
            tA_addr=self.tmem_P_offset,
        )
        mma_dsk_fn = partial(
            gemm_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, zero_init=True
        )
        if const_expr(self.use_smem_dS_for_mma_dK):
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, tdKtdK, tdKrdS, tdKrQ)
        else:
            mma_dsq_fn = partial(
                gemm_ptx_w_idx,
                tiled_mma_dK,
                tdKtdK,
                tdKrdS,
                tdKrQ,
                sA=None,
                sB=sQt,
                tA_addr=self.tmem_dS_offset,
            )

        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer,
            # pyre-ignore[16]
            self.dO_stage,
        )
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_phase_dKV = Int32(1)
        # pyre-ignore[16]
        cta_group = pipeline_S_P.cta_group

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)  # must be seqlen_k
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen,
                # pyre-ignore[16]
                n_block // self.cluster_shape_mnk[0],
            )
            if const_expr(not self.is_local) or m_block_min < m_block_max:
                accumulate_dK = False

                # Persistent: do NOT reset producer_phase_acc or producer_phase_dKV.
                # Mbarrier phases carry forward across tiles. Pipeline credits
                # at tile boundaries keep software phase in sync with hardware.

                # -----------------------------------------------------------
                ###### Prologue
                # -----------------------------------------------------------
                # 1. S  = Q0 @ K.T
                # 2. dP = V @ dO.T
                # 3. dV = P @ dO
                # 1) S  = Q0 @ K.T
                handle_Q = pipeline_Q_consumer.wait_and_advance()

                # pyre-ignore[19]
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                # 1) S = K @ Q.T (PROLOGUE - uses dK region for SFs)
                if const_expr(self.blockscaled):
                    # S2T copy: SFK and SFQ to TMEM (prologue uses dK region)
                    # Stage coordinates: (None, None, None, None, stage_idx)
                    sfk_stage_coord = (None, None, None, None, 0)
                    sfq_stage_coord = (None, None, None, None, handle_Q.index)
                    cute.copy(
                        tiled_copy_s2t_sfk,
                        # pyre-ignore[16]
                        tCsSFK_s2t[sfk_stage_coord],
                        tCtSFK_s2t_prologue,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfq,
                        tCsSFQ_s2t[sfq_stage_coord],
                        tCtSFQ_s2t_prologue,
                    )
                    cute.arch.fence_view_async_tmem_store()
                    # Blockscaled GEMM
                    # K is single-stage (no stage slice needed)
                    # Q is double-buffered (slice by stage)
                    gemm_blockscaled(
                        tiled_mma_S_bs,
                        tStS,
                        tSrK,  # K is single-stage, no slice
                        tSrQ[None, None, None, handle_Q.index],  # Q is double-buffered
                        tCtSFK_prologue,
                        tCtSFQ_prologue,
                        zero_init=True,
                    )
                else:
                    mma_qk_fn(B_idx=handle_Q.index)
                # Don't release Q yet
                # pyre-ignore[19]
                pipeline_S_P.sync_object_full.arrive(
                    0, pipeline_S_P.producer_mask, cta_group
                )

                # 2) dP = V @ dO.T
                pipeline_dO.consumer_wait(consumer_state_dO)

                # pyre-ignore[19]
                pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                # dQ uses the same tmem as dP
                # pyre-ignore[19]
                pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                # 2) dP = V @ dO.T (PROLOGUE - uses dK region for SFs)
                if const_expr(self.blockscaled):
                    # S2T copy: SFV and SFDO to TMEM (prologue uses dK region)
                    sfv_stage_coord = (None, None, None, None, 0)
                    sfdo_stage_coord = (None, None, None, None, consumer_state_dO.index)
                    cute.copy(
                        tiled_copy_s2t_sfv,
                        tCsSFV_s2t[sfv_stage_coord],
                        tCtSFV_s2t_prologue,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfdo,
                        tCsSFDO_s2t[sfdo_stage_coord],
                        tCtSFDO_s2t_prologue,
                    )
                    cute.arch.fence_view_async_tmem_store()
                    # Blockscaled GEMM
                    # V is single-stage (no stage slice needed)
                    # dO is single-stage (dO_stage=1, but has stage dimension)
                    gemm_blockscaled(
                        tiled_mma_dP_bs,
                        tdPtdP,
                        tdPrV,  # V is single-stage, no slice
                        tdPrdOt[
                            None, None, None, consumer_state_dO.index
                        ],  # dO stage slice
                        tCtSFV_prologue,
                        tCtSFDO_prologue,
                        zero_init=True,
                    )
                else:
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                # Don't release dO yet
                # pyre-ignore[19]
                pipeline_dP.sync_object_full.arrive(
                    0, pipeline_dP.producer_mask, cta_group
                )

                producer_phase_acc ^= 1
                # 3) dV = P.T @ dO
                # wait for P to be ready (in SMEM for blockscaled, TMEM for non-blockscaled)
                # pyre-ignore[19]
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                if const_expr(self.blockscaled):
                    # Phase 2 blockscaled: dV = P.T @ dO
                    # S2T copy: SFP (computed from P's AMAX) and SFDO to TMEM
                    sfp_stage_coord = (None, None, None, None, 0)
                    sfdo_stage_coord = (None, None, None, None, consumer_state_dO.index)

                    cute.copy(
                        # pyre-ignore[61]
                        tiled_copy_s2t_sfp,
                        # pyre-ignore[61]
                        tCsSFP_s2t[sfp_stage_coord],
                        # pyre-ignore[61]
                        tCtSFP_s2t,
                    )
                    cute.copy(
                        # pyre-ignore[61]
                        tiled_copy_s2t_sfdo_dv,
                        # pyre-ignore[61]
                        tCsSFDO_dV_s2t[sfdo_stage_coord],
                        # pyre-ignore[61]
                        tCtSFDO_dV_s2t,
                    )
                    cute.arch.fence_view_async_tmem_store()

                    # Blockscaled GEMM: dV = P.T @ dO
                    gemm_blockscaled(
                        tiled_mma_dV_bs,
                        tdVtdV,
                        tdVrP,  # P from TMEM
                        tdVrdO_dV[
                            None, None, None, consumer_state_dO.index
                        ],  # M-block dO from sdO_dV
                        tCtSFP,
                        tCtSFDO_dV,
                        zero_init=True,
                    )
                else:
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()
                # -----------------------------------------------------------
                ###### MAIN LOOP
                # -----------------------------------------------------------
                # 1. S  = K    @ Q.T
                # 2. dQ = dS   @ K
                # 3. dK = dS.T @ Q
                # 4. dP = V    @ dO.T
                # 5. dV = P.T  @ dO

                # pyre-ignore[28]
                for _ in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                    # 1) S = K @ Q_i (MAIN LOOP - uses dP region for SFs)
                    handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                    # Don't need to wait for S, as P must have been ready ealier, i.e., S is ready
                    # Issue 013 fix: Wait for compute warp to finish dP T2R before
                    # writing Phase 1 SFs to dP region (col 304). Without this,
                    # the SF S2T can race with dP T2R across iterations.
                    # pyre-ignore[19]
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    if const_expr(self.blockscaled):
                        # S2T copy: SFK and SFQ to TMEM (main loop)
                        # FIX: Double S2T copy to "prime" TMEM with SF format.
                        # First copy establishes SF byte format at TMEM location.
                        # Second copy writes actual SF values (same data, same location).
                        # Without this, S2T partial-write on top of F32 stale data
                        # leaves stale bits → corrupt SFs → garbage S output.
                        sfk_stage_coord = (None, None, None, None, 0)
                        sfq_stage_coord = (None, None, None, None, handle_Q_next.index)
                        cute.copy(
                            tiled_copy_s2t_sfk, tCsSFK_s2t[sfk_stage_coord], tCtSFK_s2t
                        )
                        cute.copy(
                            tiled_copy_s2t_sfq, tCsSFQ_s2t[sfq_stage_coord], tCtSFQ_s2t
                        )
                        cute.arch.fence_view_async_tmem_store()
                        # Second copy (same data, same destination)
                        cute.copy(
                            tiled_copy_s2t_sfk, tCsSFK_s2t[sfk_stage_coord], tCtSFK_s2t
                        )
                        cute.copy(
                            tiled_copy_s2t_sfq, tCsSFQ_s2t[sfq_stage_coord], tCtSFQ_s2t
                        )
                        cute.arch.fence_view_async_tmem_store()
                        # Blockscaled GEMM
                        # K is single-stage (no stage slice needed)
                        # Q is double-buffered (slice by stage)
                        gemm_blockscaled(
                            tiled_mma_S_bs,
                            tStS,
                            tSrK,  # K is single-stage, no slice
                            tSrQ[
                                None, None, None, handle_Q_next.index
                            ],  # Q is double-buffered
                            tCtSFK,
                            tCtSFQ,
                            zero_init=True,
                        )
                    else:
                        mma_qk_fn(B_idx=handle_Q_next.index)
                    # pyre-ignore[19]
                    pipeline_S_P.sync_object_full.arrive(
                        0, pipeline_S_P.producer_mask, cta_group
                    )

                    # 2-3)
                    # Do dK = dS.T @ Q, then dQ = dS @ K if dS in tmem for first mma
                    # Otherwise, reverse order
                    pipeline_dS.consumer_wait(consumer_state_dS)

                    if const_expr(self.use_smem_dS_for_mma_dK):
                        if const_expr(self.blockscaled):
                            sfds_dq_stage_coord = (None, None, None, None, 0)
                            sfk_dq_stage_coord = (None, None, None, None, 0)
                            cute.copy(
                                # pyre-ignore[61]
                                tiled_copy_s2t_sfds_dq,
                                # pyre-ignore[61]
                                tCsSFDS_dQ_s2t[sfds_dq_stage_coord],
                                # pyre-ignore[61]
                                tCtSFDS_dQ_s2t,
                            )
                            cute.copy(
                                # pyre-ignore[61]
                                tiled_copy_s2t_sfk_dq,
                                # pyre-ignore[61]
                                tCsSFK_dQ_s2t[sfk_dq_stage_coord],
                                # pyre-ignore[61]
                                tCtSFK_dQ_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_dQ_bs,
                                tdQtdQ,
                                # pyre-ignore[61]
                                tdQrdS_bs,
                                tdQrK_dQ,
                                tCtSFDS_dQ,
                                tCtSFK_dQ,
                                zero_init=True,
                            )
                        else:
                            mma_dsk_fn()
                        # pyre-ignore[19]
                        pipeline_dQ.sync_object_full.arrive(
                            0, pipeline_dQ.producer_mask, cta_group
                        )
                        mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                        accumulate_dK = True
                        handle_Q.release()
                    else:
                        # Order: S → dQ → dK → dP → dV
                        if const_expr(self.blockscaled):
                            sfds_dq_stage_coord = (None, None, None, None, 0)
                            sfk_dq_stage_coord = (None, None, None, None, 0)
                            cute.copy(
                                # pyre-ignore[61]
                                tiled_copy_s2t_sfds_dq,
                                # pyre-ignore[61]
                                tCsSFDS_dQ_s2t[sfds_dq_stage_coord],
                                # pyre-ignore[61]
                                tCtSFDS_dQ_s2t,
                            )
                            cute.copy(
                                # pyre-ignore[61]
                                tiled_copy_s2t_sfk_dq,
                                # pyre-ignore[61]
                                tCsSFK_dQ_s2t[sfk_dq_stage_coord],
                                # pyre-ignore[61]
                                tCtSFK_dQ_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_dQ_bs,
                                tdQtdQ,
                                # pyre-ignore[61]
                                tdQrdS_bs,
                                tdQrK_dQ,
                                tCtSFDS_dQ,
                                tCtSFK_dQ,
                                zero_init=True,
                            )
                        else:
                            mma_dsk_fn()
                        # pyre-ignore[19]
                        pipeline_dQ.sync_object_full.arrive(
                            0, pipeline_dQ.producer_mask, cta_group
                        )
                        if const_expr(self.blockscaled):
                            # dK SFs in dP region (col 336)
                            sfds_stage_coord = (None, None, None, None, 0)
                            sfq_stage_coord = (None, None, None, None, handle_Q.index)
                            cute.copy(
                                # pyre-ignore[61]
                                tiled_copy_s2t_sfds,
                                # pyre-ignore[61]
                                tCsSFDS_s2t[sfds_stage_coord],
                                # pyre-ignore[61]
                                tCtSFDS_s2t,
                            )
                            cute.copy(
                                # pyre-ignore[61]
                                tiled_copy_s2t_sfq_dk,
                                # pyre-ignore[61]
                                tCsSFQ_dK_s2t[sfq_stage_coord],
                                # pyre-ignore[61]
                                tCtSFQ_dK_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_dK_bs_smem,
                                tdKtdK,
                                tdKrdS,  # dS from SMEM
                                # pyre-ignore[61]
                                tdKrQ_dK[None, None, None, handle_Q.index],
                                tCtSFDS,
                                tCtSFQ_dK,
                                zero_init=not accumulate_dK,
                            )
                        else:
                            mma_dsq_fn(
                                B_idx=handle_Q.index, zero_init=not accumulate_dK
                            )
                        accumulate_dK = True
                        handle_Q.release()

                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

                    # dP = V @ dO.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    # pyre-ignore[19]
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                    if const_expr(self.blockscaled):
                        sfv_stage_coord = (None, None, None, None, 0)
                        sfdo_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            consumer_state_dO.index,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfv, tCsSFV_s2t[sfv_stage_coord], tCtSFV_s2t
                        )
                        cute.copy(
                            tiled_copy_s2t_sfdo,
                            tCsSFDO_s2t[sfdo_stage_coord],
                            tCtSFDO_s2t,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        gemm_blockscaled(
                            tiled_mma_dP_bs,
                            tdPtdP,
                            tdPrV,
                            tdPrdOt[None, None, None, consumer_state_dO.index],
                            tCtSFV,
                            tCtSFDO,
                            zero_init=True,
                        )
                    else:
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                    # pyre-ignore[19]
                    pipeline_dP.sync_object_full.arrive(
                        0, pipeline_dP.producer_mask, cta_group
                    )

                    producer_phase_acc ^= 1
                    # dV += P.T @ dO
                    # pyre-ignore[19]
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    if const_expr(self.blockscaled):
                        # Issue 008 fix: removed dead-code tiled_copy_s2t_P line
                        # (JIT bug: const_expr(None is not None) branch corrupts TMEM)
                        sfp_stage_coord = (None, None, None, None, 0)
                        sfdo_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            consumer_state_dO.index,
                        )
                        cute.copy(
                            # pyre-ignore[61]
                            tiled_copy_s2t_sfp,
                            # pyre-ignore[61]
                            tCsSFP_s2t[sfp_stage_coord],
                            # pyre-ignore[61]
                            tCtSFP_s2t,
                        )
                        cute.copy(
                            # pyre-ignore[61]
                            tiled_copy_s2t_sfdo_dv,
                            # pyre-ignore[61]
                            tCsSFDO_dV_s2t[sfdo_stage_coord],
                            # pyre-ignore[61]
                            tCtSFDO_dV_s2t,
                        )
                        cute.arch.fence_view_async_tmem_store()

                        gemm_blockscaled(
                            tiled_mma_dV_bs,
                            tdVtdV,
                            tdVrP,
                            tdVrdO_dV[
                                None, None, None, consumer_state_dO.index
                            ],  # M-block dO from sdO_dV
                            tCtSFP,
                            tCtSFDO_dV,
                            zero_init=False,  # Accumulate
                        )
                    else:
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    handle_Q = handle_Q_next

                # Orphaned S_P.full.arrive: signals "S is ready" for a non-existent
                # extra M-block. In persistent mode, skip to prevent phase mismatch.
                if const_expr(not self.is_persistent):
                    # pyre-ignore[19]
                    pipeline_S_P.sync_object_full.arrive(
                        0, pipeline_S_P.producer_mask, cta_group
                    )

                # signal to the epilogue that dV is ready
                # pipeline_dKV.producer_acquire(producer_state_dKV)
                # pyre-ignore[19]
                pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                # pipeline_dKV.producer_commit(producer_state_dKV)
                # pyre-ignore[19]
                pipeline_dKV.sync_object_full.arrive(
                    0, pipeline_dKV.producer_mask, cta_group
                )
                # producer_state_dKV.advance()
                # pipeline_dKV.producer_acquire(producer_state_dKV)
                # pyre-ignore[19]
                pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                # -----------------------------------------------------------
                ###### Remaining 2
                # -----------------------------------------------------------
                # 1) dK += dS.T @ Q
                pipeline_dS.consumer_wait(consumer_state_dS)
                if const_expr(self.blockscaled and not self.use_smem_dS_for_mma_dK):
                    # Phase 2 blockscaled: dK = dS.T @ Q
                    sfds_stage_coord = (None, None, None, None, 0)
                    sfq_stage_coord = (None, None, None, None, handle_Q.index)
                    cute.copy(
                        # pyre-ignore[61]
                        tiled_copy_s2t_sfds,
                        # pyre-ignore[61]
                        tCsSFDS_s2t[sfds_stage_coord],
                        # pyre-ignore[61]
                        tCtSFDS_s2t,
                    )
                    cute.copy(
                        # pyre-ignore[61]
                        tiled_copy_s2t_sfq_dk,
                        # pyre-ignore[61]
                        tCsSFQ_dK_s2t[sfq_stage_coord],
                        # pyre-ignore[61]
                        tCtSFQ_dK_s2t,
                    )
                    cute.arch.fence_view_async_tmem_store()
                    gemm_blockscaled(
                        tiled_mma_dK_bs_smem,
                        tdKtdK,
                        tdKrdS,  # dS from SMEM
                        # pyre-ignore[61]
                        tdKrQ_dK[None, None, None, handle_Q.index],  # M-block Q for dK
                        tCtSFDS,
                        tCtSFQ_dK,
                        zero_init=not accumulate_dK,
                    )
                else:
                    mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)

                # signal to the epilogue that dK is ready
                # pipeline_dKV.producer_commit(producer_state_dKV)
                # pyre-ignore[19]
                pipeline_dKV.sync_object_full.arrive(
                    1, pipeline_dKV.producer_mask, cta_group
                )
                # producer_state_dKV.advance()
                producer_phase_dKV ^= 1

                # 2) dQ = dS @ K
                # dS is done, so dP must have been ready, we don't need to wait
                # Using reused dK layouts for dQ GEMM TMEM (TMA loading disabled)
                if const_expr(self.blockscaled):
                    # S2T copy: SFDS_dQ and SFK_dQ to TMEM for dQ GEMM
                    sfds_dq_stage_coord = (
                        None,
                        None,
                        None,
                        None,
                        0,
                    )  # Stage 0 for simplicity
                    sfk_dq_stage_coord = (
                        None,
                        None,
                        None,
                        None,
                        0,
                    )  # K is single-stage
                    cute.copy(
                        # pyre-ignore[61]
                        tiled_copy_s2t_sfds_dq,
                        # pyre-ignore[61]
                        tCsSFDS_dQ_s2t[sfds_dq_stage_coord],
                        # pyre-ignore[61]
                        tCtSFDS_dQ_s2t,
                    )
                    cute.copy(
                        # pyre-ignore[61]
                        tiled_copy_s2t_sfk_dq,
                        # pyre-ignore[61]
                        tCsSFK_dQ_s2t[sfk_dq_stage_coord],
                        # pyre-ignore[61]
                        tCtSFK_dQ_s2t,
                    )
                    cute.arch.fence_view_async_tmem_store()

                    gemm_blockscaled(
                        tiled_mma_dQ_bs,
                        tdQtdQ,
                        # pyre-ignore[61]
                        tdQrdS_bs,
                        tdQrK_dQ,
                        tCtSFDS_dQ,
                        tCtSFK_dQ,
                        zero_init=True,
                    )
                else:
                    mma_dsk_fn()
                # pyre-ignore[19]
                pipeline_dQ.sync_object_full.arrive(
                    0, pipeline_dQ.producer_mask, cta_group
                )
                if const_expr(self.is_persistent):
                    # pyre-ignore[19]
                    pipeline_K_done.sync_object_full.arrive(
                        0, pipeline_K_done.producer_mask, cta_group
                    )
                handle_Q.release()
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()

                producer_phase_acc ^= 1

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        tStS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        sdS_dQ: Optional[cute.Tensor],  # Blockscaled A SMEM view for dQ GEMM
        sdS_dK: Optional[cute.Tensor],  # Blockscaled A SMEM view for dK GEMM
        sdSt: cute.Tensor,  # N-contiguous sdS SMEM for R2S write
        tdPtdP: cute.Tensor,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor] = None,  # SFP SMEM tensor for blockscaled P
        sSFDS: Optional[cute.Tensor] = None,  # SFDS SMEM tensor for blockscaled dS
        sSFDS_dQ: Optional[cute.Tensor] = None,  # SFDS for dQ GEMM (separate layout)
        sP: Optional[cute.Tensor] = None,  # SMEM P buffer for blockscaled dV GEMM
        mdSFK_out: Optional[cute.Tensor] = None,  # Output SF tensor for dK (MXFP8)
        mdSFV_out: Optional[cute.Tensor] = None,  # Output SF tensor for dV (MXFP8)
        mAttnScale: Optional[cute.Tensor] = None,  # SiLU: per-row attention scale
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                # pyre-ignore[16]
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                # pyre-ignore[16]
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        # if const_expr(self.SdP_swapAB):
        if const_expr(True):
            sLSE_2D = utils.transpose_view(sLSE_2D)
            sdPsum_2D = utils.transpose_view(sdPsum_2D)

        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        )
        # tidx = cute.arch.thread_idx()[0] - (cute.arch.WARP_SIZE * self.compute_warp_ids[0])
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        tileP_f32_like = (
            # pyre-ignore[16]
            self.mma_tiler_kq[0] // 32 * self.v_dtype.width
        )  # 64 for tile_n = 128
        # tStS has shape ((128, 128), 1, 1), tStP has shape ((128, 64), 1, 1)
        # tP overlap with tS
        tStP = cute.composition(
            tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1)
        )
        tStP = cute.make_tensor(
            tStS.iterator, tStP.layout
        )  # Otherwise the tmem address is wrong
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(
            tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1)
        )
        # tdS overlap with tdP
        tdPtdS = cute.composition(
            tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1)
        )
        tdPtdS = cute.make_tensor(
            tdPtdP.iterator, tdPtdS.layout
        )  # Fix tmem address like P does - otherwise the address is wrong
        tdPcdP = thr_mma_dP.partition_C(
            cute.make_identity_tensor(self.mma_tiler_vdo[:2])
        )
        tdPcdS = cute.composition(
            tdPcdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1)
        )

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        tmem_store_repetition = 16 if self.v_dtype.width == 16 else 8
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(tmem_store_repetition)),
            Float32,
        )

        # tmem -> rmem
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)  # (((32, 32), 1), 2, 1, 1)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
        t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)  # ((32, 1), 2, 1, 1)
        # ((32, 1), 2, 1, 1, STAGE)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        # rmem -> tmem
        thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(
            tidx
        )
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
        thr_copy_r2t_P = thr_copy_r2t
        # rmem -> smem
        # This part is a bit iffy, we might be making a lot of assumptions here
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR,
            # pyre-ignore[16]
            self.ds_dtype,
            Float32,
            thr_copy_t2r,
        )
        thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(
            tidx
        )
        # We assume the swizzle (i.e. layout.inner) stays the same
        sdS_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        ).outer  # ((8,16), (64,2), (1, 1))
        sdS_layout = cute.slice_(sdS_layout, (None, None, 0))  # ((8,16), (64,2))
        # Need to group into 1 mode to be compatible w thr_copy_r2s
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        sdS_epi = cute.make_tensor(sdSt.iterator, sdS_layout)
        tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)

        # R2S setup for P data (blockscaled: write FP8 P to SMEM for dV GEMM)
        if const_expr(self.blockscaled and sP is not None):
            # P has same tile dimensions as dS: (tile_n, tile_m) in FP8
            # Use same epi layout pattern and R2S copy as dS (same dtype, same shape)
            # pyre-ignore[16]
            sP_epi = cute.make_tensor(sP.iterator, sdS_layout)

        consumer_state_S_P_dP = (
            pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
                cutlass.pipeline.PipelineUserType.Consumer, 1
            )
        )
        producer_state_dS = (
            pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
                cutlass.pipeline.PipelineUserType.Producer, 1
            )
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )

        # pyre-ignore[29]
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            # pyre-ignore[29]
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen,
                # pyre-ignore[16]
                n_block // self.cluster_shape_mnk[0],
            )
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            # TODO: condition mask_seqlen
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=n_block,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
            )

            prefetch_LSE = False

            # Mainloop
            # pyre-ignore[28]
            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                # Prefetch 1 stage of LSE (skip for SiLU — not loaded)
                if const_expr(not self.use_silu):
                    pipeline_LSE.consumer_wait(consumer_state_LSE)
                tSrLSE_s2r = cute.make_fragment(tScS_t2r[None, 0, 0, 0].shape, Float32)
                if const_expr(prefetch_LSE and not self.shuffle_LSE):
                    cute.autovec_copy(
                        tSsLSE[None, 0, 0, 0, consumer_state_LSE.index], tSrLSE_s2r
                    )

                pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                # pipeline_S_P.sync_object_full.wait(0, consumer_phase_S_P_dP)
                #### TMEM->RMEM (Load S from TMEM)
                tSrS_t2r = cute.make_fragment(tScS_t2r.shape, Float32)
                cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)
                cute.arch.fence_view_async_tmem_load()

                #### APPLY MASK
                if const_expr(not self.use_silu):
                    # Softmax: mask with -inf before exp2
                    mask_fn(tSrS_t2r, m_block=m_block)

                num_stages = cute.size(tScS_t2r, mode=[1])

                # ---------------------------------------------
                #### Phase 1: S -> P (activation)
                # ---------------------------------------------
                if const_expr(self.use_silu):
                    # compute P and dS in one pass

                    # Wait for dP to be ready (dP GEMM follows S GEMM in MMA warp)
                    pipeline_dP.consumer_wait(consumer_state_S_P_dP)

                    # Create dS output fragments
                    if const_expr(not self.use_smem_dS_for_mma_dK):
                        tdPrdS_r2t_f32 = cute.make_fragment(tdPcdS_r2t.shape, Float32)
                        tdPrdS_r2t = cute.recast_tensor(tdPrdS_r2t_f32, self.ds_dtype)

                    tSrP_r2t_f32 = cute.make_fragment(tScP_r2t.shape, Float32)
                    # pyre-ignore[16]
                    tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
                    elems_per_stage = cute.size(tSrS_t2r, mode=[0])

                    # Precompute tile validity once per m_block
                    seqlen_k_limit = seqlen.seqlen_k - n_block * self.tile_n
                    seqlen_q_limit = seqlen.seqlen_q - m_block * self.tile_m
                    tile_fully_valid = (seqlen_k_limit >= self.tile_n) & (
                        seqlen_q_limit >= self.tile_m
                    )
                    if const_expr(self.is_causal or self.is_local):
                        if const_expr(
                            block_info.window_size_left is not None
                            and not self.is_local
                        ):
                            causal_delta_tile = Int32(0)
                        else:
                            causal_delta_tile = seqlen.seqlen_k - seqlen.seqlen_q
                        tile_fully_valid = tile_fully_valid & (
                            m_block * self.tile_m + causal_delta_tile
                            >= (n_block + 1) * self.tile_n - 1
                        )
                        if const_expr(block_info.window_size_left is not None):
                            tile_fully_valid = tile_fully_valid & (
                                (m_block + 1) * self.tile_m
                                - 1
                                + causal_delta_tile
                                - n_block * self.tile_n
                                # pyre-ignore[58]
                                <= block_info.window_size_left
                            )

                    for stage in cutlass.range(num_stages, unroll=2):
                        tSrS_cur = tSrS_t2r[None, stage, 0, 0]

                        # Read dP from TMEM for this stage
                        tdPrdP_t2r = cute.make_fragment(
                            tScS_t2r[None, 0, None, None].shape, Float32
                        )
                        cute.copy(
                            thr_copy_t2r,
                            tdPtdP_t2r[None, stage, None, None],
                            tdPrdP_t2r,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        tdPrdP_cur = tdPrdP_t2r[None, 0, 0]

                        # --- SiLU forward + derivative ---
                        half_scale = softmax_scale * Float32(0.5)
                        tScS_stage_fused = tScS_t2r[None, stage, 0, 0]
                        for v in cutlass.range_constexpr(elems_per_stage // 2):
                            if const_expr(mAttnScale is not None):
                                m_idx = (
                                    tScS_stage_fused[2 * v][1] + m_block * self.tile_m
                                )
                                # pyre-ignore[16]
                                row_scale = Float32(mAttnScale[m_idx])
                            else:
                                row_scale = Float32(1.0)

                            x = (tSrS_cur[2 * v], tSrS_cur[2 * v + 1])
                            c_hs = (half_scale, half_scale)
                            half_x = cute.arch.mul_packed_f32x2(x, c_hs)

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
                            tanh0 = Float32(t0)
                            tanh1 = Float32(t1)

                            sig0, sig1 = utils.fma_packed_f32x2(
                                (tanh0, tanh1),
                                (Float32(0.5), Float32(0.5)),
                                (Float32(0.5), Float32(0.5)),
                            )

                            qk0, qk1 = utils.mul_packed_f32x2(
                                (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                                (softmax_scale, softmax_scale),
                            )

                            oms0, oms1 = utils.fma_packed_f32x2(
                                (tanh0, tanh1),
                                (Float32(-0.5), Float32(-0.5)),
                                (Float32(0.5), Float32(0.5)),
                            )

                            p0, p1 = utils.mul_packed_f32x2((qk0, qk1), (sig0, sig1))
                            if const_expr(mAttnScale is not None):
                                tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = (
                                    utils.mul_packed_f32x2(
                                        (p0, p1), (row_scale, row_scale)
                                    )
                                )
                            else:
                                tSrS_cur[2 * v] = p0
                                tSrS_cur[2 * v + 1] = p1

                            deriv0, deriv1 = utils.fma_packed_f32x2(
                                (p0, p1), (oms0, oms1), (sig0, sig1)
                            )

                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = (
                                utils.mul_packed_f32x2(
                                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                                    (deriv0, deriv1),
                                )
                            )
                            if const_expr(mAttnScale is not None):
                                tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = (
                                    utils.mul_packed_f32x2(
                                        (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                                        (row_scale, row_scale),
                                    )
                                )

                        # Zero out-of-bounds positions in BOTH P and dS
                        if not tile_fully_valid:
                            tScS_stage_mask = tScS_t2r[None, stage, 0, 0]
                            for v in cutlass.range_constexpr(elems_per_stage):
                                row = tScS_stage_mask[v][0]
                                col = tScS_stage_mask[v][1]
                                out_of_bounds = (row >= seqlen_k_limit) | (
                                    col >= seqlen_q_limit
                                )
                                if const_expr(self.is_causal or self.is_local):
                                    global_q = m_block * self.tile_m + col
                                    global_k = n_block * self.tile_n + row
                                    if const_expr(
                                        block_info.window_size_left is not None
                                        and not self.is_local
                                    ):
                                        causal_delta = Int32(0)
                                    else:
                                        causal_delta = seqlen.seqlen_k - seqlen.seqlen_q
                                    if global_q + causal_delta < global_k:
                                        out_of_bounds = True
                                    if const_expr(
                                        block_info.window_size_left is not None
                                    ):
                                        if (
                                            global_q + causal_delta - global_k
                                            # pyre-ignore[58]
                                            > block_info.window_size_left
                                        ):
                                            out_of_bounds = True
                                if out_of_bounds:
                                    tSrS_cur[v] = Float32(0.0)
                                    tdPrdP_cur[v] = Float32(0.0)

                        # Write P to TMEM
                        utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
                        cute.copy(
                            thr_copy_r2t_P,
                            tSrP_r2t_f32[None, stage, None, None],
                            tStP_r2t[None, stage, None, None],
                        )

                        # Write dS to TMEM and/or SMEM
                        if const_expr(not self.use_smem_dS_for_mma_dK):
                            # pyre-ignore[61]
                            tdPrdS_cvt = tdPrdS_r2t[None, stage, 0, 0]
                        else:
                            tdPrdS_cvt = cute.make_fragment_like(
                                tdPrdP_cur, self.ds_dtype
                            )
                        utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
                        if const_expr(not self.use_smem_dS_for_mma_dK):
                            cute.copy(
                                thr_copy_r2t,
                                # pyre-ignore[61]
                                tdPrdS_r2t_f32[None, stage, None, None],
                                tdPtdS_r2t[None, stage, None, None],
                            )
                        cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

                    # Fence P TMEM store + dS TMEM store + dS SMEM store
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.compute_sync_barrier.arrive_and_wait()

                    # Release pipelines
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                    with cute.arch.elect_one():
                        # pyre-ignore[19]
                        pipeline_dP.sync_object_empty.arrive(
                            0, pipeline_dP.consumer_mask
                        )
                    consumer_state_S_P_dP.advance()
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()
                else:
                    # ---------------------------------------------
                    lane_idx = cute.arch.lane_idx()
                    tSrP_r2t_f32 = cute.make_fragment(tScP_r2t.shape, Float32)  # 64
                    tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
                    for stage in cutlass.range_constexpr(num_stages):
                        tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                        tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                        if const_expr(not self.shuffle_LSE):
                            if const_expr(stage > 0 or not prefetch_LSE):
                                cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                            tSrLSE = tSrLSE_s2r
                        else:
                            tSrLSE = tSsLSE_cur[lane_idx]
                        if const_expr(
                            self.blockscaled
                            and sSFP is not None
                            and self.q_dtype
                            in [cutlass.Float8E4M3FN, cutlass.Float8E5M2]
                        ):
                            pre_exp2_max = -Float32.inf
                        for v in cutlass.range_constexpr(
                            cute.size(tSrS_t2r, mode=[0]) // 2
                        ):
                            if const_expr(not self.shuffle_LSE):
                                lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                            else:
                                lse_pair = (
                                    utils.shuffle_sync(tSrLSE, offset=2 * v),
                                    utils.shuffle_sync(tSrLSE, offset=2 * v + 1),
                                )
                            tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = (
                                utils.fma_packed_f32x2(
                                    ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                                    (softmax_scale_log2, softmax_scale_log2),
                                    (-lse_pair[0], -lse_pair[1]),
                                )
                            )
                            if const_expr(
                                self.blockscaled
                                and sSFP is not None
                                and self.q_dtype
                                in [cutlass.Float8E4M3FN, cutlass.Float8E5M2]
                            ):
                                pre_exp2_max = max_f32(pre_exp2_max, tSrS_cur[2 * v])
                                pre_exp2_max = max_f32(
                                    pre_exp2_max, tSrS_cur[2 * v + 1]
                                )
                            tSrS_cur[2 * v] = cute.math.exp2(
                                tSrS_cur[2 * v], fastmath=True
                            )
                            tSrS_cur[2 * v + 1] = cute.math.exp2(
                                tSrS_cur[2 * v + 1], fastmath=True
                            )
                        # For FP16/BF16, use cvt_f16 which uses packed conversion
                        if const_expr(
                            self.q_dtype in [cutlass.Float8E4M3FN, cutlass.Float8E5M2]
                        ):
                            # Get the FP8 destination slice
                            tSrP_dst = tSrP_r2t[None, stage, 0, 0]

                            if const_expr(self.blockscaled and sSFP is not None):
                                # Blockscaled conversion: compute AMAX, derive scale, scale and convert
                                BLOCK_SIZE = 32
                                max_norm_rcp = Float32(E4M3_MAX_NORM_RCP)

                                src_frg = cute.logical_divide(
                                    tSrS_cur, cute.make_layout(BLOCK_SIZE)
                                )
                                dst_frg = cute.logical_divide(
                                    tSrP_dst, cute.make_layout(BLOCK_SIZE)
                                )

                                # Derive block_amax from pre-computed max of pre-exp2 values
                                # exp2 is monotonic: max(exp2(x_i)) = exp2(max(x_i))
                                # pyre-ignore[61]
                                block_amax = cute.math.exp2(pre_exp2_max, fastmath=True)
                                sfp, inv_scale, sfp_u32 = fused_amax_to_e8m0_scale_f32(
                                    block_amax, max_norm_rcp
                                )
                                sSFDS_scratch = self._make_smem_u32_view(sSFDS)
                                if dp_idx < cute.size(sSFDS_scratch.shape[0]):
                                    sSFDS_scratch[dp_idx, Int32(1)] = sfp_u32 & Uint32(
                                        0xFF
                                    )
                                # Use packed FP8 conversion (avoids PRMT instructions)
                                _sfp_scaled = cute.make_fragment_like(tSrS_cur, Float32)
                                _sfp_scaled.store(tSrS_cur.load() * inv_scale)
                                utils.cvt_f16(_sfp_scaled, tSrP_dst)

                                sSFP_u32 = self._make_smem_u32_view(sSFP)

                                wg_idx = tidx // 128  # 0 for WG0, 1 for WG1

                                _sfp_u32_val = Uint32(
                                    sSFDS_scratch[dp_idx, Int32(1)]
                                ) & Uint32(0xFF)

                                # Byte store: write SF byte to unique byte offset
                                # within the u32 (no barriers needed).
                                if dp_idx < cute.size(sSFP_u32.shape[0]):
                                    _sfp_byte_addr = Int32(
                                        utils.elem_pointer(
                                            sSFP_u32, (dp_idx, Int32(0))
                                        ).toint()
                                    )
                                    _sfp_byte_off = Int32(
                                        const_expr(stage * 2)
                                    ) + Int32(wg_idx)
                                    utils.st_shared_b8(
                                        _sfp_byte_addr + _sfp_byte_off,
                                        Int32(_sfp_u32_val),
                                    )

                                # SMEM fence: order byte stores with async ops
                                cute.arch.fence_proxy(
                                    cute.arch.ProxyKind.async_shared,
                                    space=cute.arch.SharedSpace.shared_cta,
                                )
                            else:
                                # Non-blockscaled FP8: use packed conversion
                                # to avoid PRMT byte-permutation instructions
                                utils.cvt_f16(tSrS_cur, tSrP_dst)
                        else:
                            utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])

                        if const_expr(stage == 0):
                            cute.arch.fence_view_async_tmem_load()
                            # Without this barrier, we could have 1 warp writing to P in tmem while
                            # another warp is still reading S from tmem.
                            self.compute_sync_barrier.arrive_and_wait()

                        # R2T: write FP8 P to TMEM (layout-aware copy from Issue 008 fix)
                        cute.copy(
                            thr_copy_r2t_P,
                            tSrP_r2t_f32[None, stage, None, None],
                            tStP_r2t[None, stage, None, None],
                        )

                    cute.arch.fence_view_async_tmem_store()
                    self.compute_sync_barrier.arrive_and_wait()

                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                    pipeline_LSE.consumer_release(consumer_state_LSE)
                    consumer_state_LSE.advance()

                # ---------------------------------------------
                # Phase 2: dP -> dS (softmax only; SiLU handled above)
                # ---------------------------------------------
                if const_expr(not self.use_silu):
                    pipeline_dPsum.consumer_wait(consumer_state_dPsum)

                    pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                    consumer_state_S_P_dP.advance()

                    # Create Float32 fragment for dS R2T copy
                    # Use tdPcdS_r2t.shape for consistency with dS destination
                    if const_expr(not self.use_smem_dS_for_mma_dK):
                        # Use dS's coordinate tensor shape (not P's)
                        tdPrdS_r2t_f32 = cute.make_fragment(tdPcdS_r2t.shape, Float32)
                        tdPrdS_r2t = cute.recast_tensor(tdPrdS_r2t_f32, self.ds_dtype)

                    ##### Softmax dS.T = P.T * (dP.T - Psum)
                    for stage in cutlass.range_constexpr(num_stages):
                        tdPrdP_t2r = cute.make_fragment(
                            tScS_t2r[None, 0, None, None].shape, Float32
                        )
                        cute.copy(
                            thr_copy_t2r,
                            tdPtdP_t2r[None, stage, None, None],
                            tdPrdP_t2r,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        self.compute_sync_barrier.arrive_and_wait()
                        tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                        tSrS_cur = tSrS_t2r[None, stage, 0, 0]

                        tSsdPsum_cur = tSsdPsum[
                            None, stage, 0, 0, consumer_state_dPsum.index
                        ]
                        if const_expr(not self.shuffle_dPsum):
                            tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                            cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                        else:
                            # pyre-ignore[61]
                            tSrdPsum = tSsdPsum_cur[lane_idx]

                        for v in cutlass.range_constexpr(
                            cute.size(tdPrdP_t2r, mode=[0]) // 2
                        ):
                            if const_expr(not self.shuffle_dPsum):
                                dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                            else:
                                dPsum_pair = (
                                    utils.shuffle_sync(tSrdPsum, offset=2 * v),
                                    utils.shuffle_sync(tSrdPsum, offset=2 * v + 1),
                                )
                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = (
                                utils.sub_packed_f32x2(
                                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                                    dPsum_pair,
                                )
                            )
                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = (
                                utils.mul_packed_f32x2(
                                    (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                                )
                            )

                        # Use pre-allocated Float32 fragment for R2T copy (matching P pattern)
                        # This ensures correct size when recasting FP8 to Float32
                        if const_expr(not self.use_smem_dS_for_mma_dK):
                            # FP8 view of Float32 storage for this stage
                            # pyre-ignore[61]
                            tdPrdS_cvt = tdPrdS_r2t[None, stage, 0, 0]
                        else:
                            # SMEM-only path: create separate fragment
                            tdPrdS_cvt = cute.make_fragment_like(
                                tdPrdP_cur, self.ds_dtype
                            )

                        # Convert dS from Float32 to FP8/BF16
                        if const_expr(self.blockscaled):
                            # Blockscaled conversion matching SFP pattern exactly
                            BLOCK_SIZE = 32
                            max_norm_rcp = Float32(E4M3_MAX_NORM_RCP)

                            src_frg = cute.logical_divide(
                                tdPrdP_cur, cute.make_layout(BLOCK_SIZE)
                            )
                            dst_frg = cute.logical_divide(
                                tdPrdS_cvt, cute.make_layout(BLOCK_SIZE)
                            )

                            # AMAX over the single block (32 elements), with abs since dS can be negative
                            block_amax = Float32(0.0)
                            for k in cutlass.range_constexpr(0, BLOCK_SIZE, 2):
                                block_amax = max_f32(block_amax, abs_f32(src_frg[k, 0]))
                                block_amax = max_f32(
                                    block_amax, abs_f32(src_frg[k + 1, 0])
                                )

                            # Warp-wide AMAX: redux across 32 lanes for uniform SF.
                            # When reading dS from SMEM, all lanes see the same data
                            # via the MMA's broadcast pattern, so they must share a
                            # single SF to avoid scale mismatches.
                            _dk_warp_amax = redux_sync_max_abs_f32(block_amax)
                            sfds0, inv_scale, sfds0_u32 = fused_amax_to_e8m0_scale_f32(
                                _dk_warp_amax, max_norm_rcp
                            )

                            # Use packed FP8 conversion (avoids PRMT instructions)
                            _sfds_scaled = cute.make_fragment_like(tdPrdP_cur, Float32)
                            _sfds_scaled.store(tdPrdP_cur.load() * inv_scale)
                            utils.cvt_f16(_sfds_scaled, tdPrdS_cvt)

                            # SFDS byte store: write SF byte to unique byte offset
                            # within the u32 (no barriers needed).
                            wg_idx_sfds = tidx // 128
                            _sfds_val = sfds0_u32 & Uint32(0xFF)

                            sSFDS_u32 = self._make_smem_u32_view(sSFDS)

                            if dp_idx < cute.size(sSFDS_u32.shape[0]):
                                _sfds_byte_addr = Int32(
                                    utils.elem_pointer(
                                        sSFDS_u32, (dp_idx, Int32(0))
                                    ).toint()
                                )
                                _sfds_byte_off = Int32(const_expr(stage * 2)) + Int32(
                                    wg_idx_sfds
                                )
                                utils.st_shared_b8(
                                    _sfds_byte_addr + _sfds_byte_off,
                                    Int32(_sfds_val),
                                )

                            # SMEM fence: order byte stores with async ops
                            cute.arch.fence_proxy(
                                cute.arch.ProxyKind.async_shared,
                                space=cute.arch.SharedSpace.shared_cta,
                            )

                        else:
                            # Non-blockscaled path: simple conversion
                            utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)

                        if const_expr(stage == 0):
                            pipeline_dS.producer_acquire(producer_state_dS)
                        if const_expr(
                            not self.use_smem_dS_for_mma_dK and not self.blockscaled
                        ):
                            # R2T copy: write dS to TMEM for non-blockscaled dK GEMM.
                            # Blockscaled dK reads dS from SMEM, so R2T is not needed.
                            cute.copy(
                                thr_copy_r2t,
                                # pyre-ignore[61]
                                tdPrdS_r2t_f32[None, stage, None, None],
                                tdPtdS_r2t[None, stage, None, None],
                            )

                        if const_expr(not self.use_smem_dS_for_mma_dK):
                            if const_expr(self.blockscaled):
                                # dQ GEMM quantization: warp-wide AMAX
                                # Per-thread AMAX over 32 m-values, then redux across
                                # 32 lanes (n-values) → single SF per warp per k_group.
                                # Quantize all elements with this warp-wide SF.
                                # pyre-ignore[61]
                                _dq_per_thread_amax = block_amax  # Reuse SFDS AMAX (same 32 elements of tdPrdP_cur, unmodified)
                                _dq_warp_amax = redux_sync_max_abs_f32(
                                    _dq_per_thread_amax
                                )
                                _, _dq_inv, _dq_sf_u32 = fused_amax_to_e8m0_scale_f32(
                                    _dq_warp_amax,
                                    # pyre-ignore[61]
                                    max_norm_rcp,
                                )

                                # Scale in-place, then use packed FP8 conversion (avoids PRMT)
                                tdPrdP_cur.store(tdPrdP_cur.load() * _dq_inv)
                                _dq_cvt = cute.make_fragment_like(
                                    tdPrdP_cur, self.ds_dtype
                                )
                                utils.cvt_f16(tdPrdP_cur, _dq_cvt)
                                cute.autovec_copy(_dq_cvt, tRS_sdS[None, stage])

                                # Save warp SF to scratch for cross-warp combine.
                                # Use stage 1 of sSFDS_dQ u32 view as scratch space
                                # (MMA always reads stage 0, so stage 1 is free).
                                # 16 scratch positions: wg_idx*4 + warp + stage*8
                                # Stage 0: positions 0..7, Stage 1: positions 8..15
                                if const_expr(sSFDS_dQ is not None):
                                    sSFDS_dQ_u32 = self._make_smem_u32_view(sSFDS_dQ)
                                    wg_idx_sfds = tidx // 128
                                    _warp_within_WG = (tidx % 128) // 32

                                    # st.shared.b8: write SF byte directly to final
                                    # packed u32, no scratch + barrier + combine needed.
                                    # Target dp_idx: each WG+stage covers 32 M-rows.
                                    _target_dp_idx = (
                                        Int32(const_expr(stage * 2))
                                        + Int32(wg_idx_sfds)
                                        # pyre-ignore[61]
                                    ) * Int32(32) + lane_idx

                                    if _target_dp_idx < cute.size(
                                        sSFDS_dQ_u32.shape[0]
                                    ):
                                        _dq_sf_byte_addr = Int32(
                                            utils.elem_pointer(
                                                sSFDS_dQ_u32,
                                                (_target_dp_idx, Int32(0)),
                                            ).toint()
                                        )
                                        _dq_sf_byte_off = Int32(_warp_within_WG)
                                        utils.st_shared_b8(
                                            _dq_sf_byte_addr + _dq_sf_byte_off,
                                            Int32(_dq_sf_u32),
                                        )
                            else:
                                # Non-blockscaled: use standard conversion
                                cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
                        else:
                            cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

                    if const_expr(
                        not self.use_smem_dS_for_mma_dK and not self.blockscaled
                    ):
                        cute.arch.fence_view_async_tmem_store()
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.compute_sync_barrier.arrive_and_wait()

                    # K-major R2S writes data directly in K-major order; no SMEM transpose needed.

                    with cute.arch.elect_one():
                        # Issue 013 fix: Signal that compute warp is done reading from
                        # dP TMEM region. MMA warp waits for this before writing Phase 1
                        # SFs to dP region (col 304) in the next iteration.
                        # pyre-ignore[19]
                        pipeline_dP.sync_object_empty.arrive(
                            0, pipeline_dP.consumer_mask
                        )
                    pipeline_dPsum.consumer_release(consumer_state_dPsum)
                    consumer_state_dPsum.advance()
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

            # Epilogue
            if const_expr(not self.is_local) or m_block_min < m_block_max:
                # pyre-ignore[16]
                if const_expr(not self.use_tma_store):
                    consumer_state_dKV = self.epilogue_dKV(
                        dp_idx,
                        warp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV,
                        mdK,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                        mdSFK_out,
                        mdSFV_out,
                    )
                else:
                    # pyre-ignore[16]
                    thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                    #### STORE dV
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        thr_mma_dV,
                        tdVtdV,
                        mdV_tma_tensor,
                        sdV,
                        tma_atom_dV,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        None,  # Don't scale
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdV_semaphore,
                        mdSFV_out,
                        seqlen,
                    )
                    #### STORE dK
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        thr_mma_dK,
                        tdKtdK,
                        mdK_tma_tensor,
                        sdK,
                        tma_atom_dK,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale
                        if const_expr(self.qhead_per_kvhead == 1 or self.use_silu)
                        else None,
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdK_semaphore,
                        mdSFK_out,
                        seqlen,
                    )
            if const_expr(self.qhead_per_kvhead == 1 and self.is_local):
                if m_block_min >= m_block_max:
                    # like other epis, currently assumes hdim == hdimv
                    gmem_tiled_copy_zero_dKV = copy_utils.tiled_copy_2d(
                        # pyre-ignore[16]
                        self.dk_dtype,
                        self.tile_hdim,
                        128,  # num_threads
                    )
                    gmem_thr_copy_zero_dKV = gmem_tiled_copy_zero_dKV.get_slice(dp_idx)
                    mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    gdK = cute.local_tile(
                        mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0)
                    )
                    gdV = cute.local_tile(
                        mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0)
                    )
                    tdKgdK = gmem_thr_copy_zero_dKV.partition_D(gdK)
                    tdVgdV = gmem_thr_copy_zero_dKV.partition_D(gdV)
                    assert tdKgdK.shape[2] == 1
                    assert tdVgdV.shape[2] == 1
                    cdKV = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
                    tdKVcdKV = gmem_thr_copy_zero_dKV.partition_D(cdKV)
                    zero = cute.make_fragment_like(tdKgdK[None, 0, 0])
                    zero.fill(0.0)
                    if tidx < 128:
                        for i in cutlass.range_constexpr(tdKgdK.shape[1]):
                            row_idx = tdKVcdKV[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - self.tile_n * n_block:
                                cute.copy(
                                    gmem_tiled_copy_zero_dKV, zero, tdKgdK[None, i, 0]
                                )
                    else:
                        for i in cutlass.range_constexpr(tdVgdV.shape[1]):
                            row_idx = tdKVcdKV[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - self.tile_n * n_block:
                                cute.copy(
                                    gmem_tiled_copy_zero_dKV, zero, tdVgdV[None, i, 0]
                                )

            # Pipeline credits: replenish init credits for S_P and dP.
            # Init arrives arrive_count times on each empty barrier, flipping
            # the phase once. Per tile, m consumer_releases (8 warps × elect_one
            # = 8 arrives = arrive_count) cause m phase flips. The init's extra
            # flip means the phase is off by 1 after each tile. This credit
            # (8 warps × elect_one = 8 arrives) adds one more flip to compensate.
            if const_expr(self.is_persistent):
                with cute.arch.elect_one():
                    # pyre-ignore[19]
                    pipeline_S_P.sync_object_empty.arrive(0, pipeline_S_P.consumer_mask)
                    # pyre-ignore[19]
                    pipeline_dP.sync_object_empty.arrive(0, pipeline_dP.consumer_mask)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.core.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mdQ_semaphore: Optional[cute.Tensor],
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(
            cute.arch.warp_idx() % len(self.reduce_warp_ids)
        )
        is_tma_warp = warp_idx == 0
        # TMEM -> RMEM
        tmem_load_atom = cute.make_copy_atom(
            # pyre-ignore[16]
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)),
            Float32,
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(
            cute.make_identity_tensor(self.mma_tiler_dsk[:2])
        )
        tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        # pyre-ignore[16]
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == self.dQaccum_reduce_stage, (
            "dQaccum reduce stage mismatch"
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            # pyre-ignore[16]
            self.dqaccum_dtype,
            num_reduce_threads,
            num_copy_elems=128 // self.dqaccum_dtype.width,
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        read_flag = const_expr(not self.deterministic)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            # pyre-ignore[16]
            self.sdQaccum_stage,
        )
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen,
                # pyre-ignore[16]
                n_block // self.cluster_shape_mnk[0],
            )
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            else:
                mdQaccum_cur = cute.domain_offset(
                    (seqlen.padded_offset_q * self.tile_hdim,), mdQaccum[None, head_idx]
                )
            gdQaccum_ = cute.local_tile(
                mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,)
            )
            # (M * K / STAGE, STAGE, _)
            gdQaccum = cute.flat_divide(
                gdQaccum_, (self.tile_m * self.tile_hdim // self.dQaccum_reduce_stage,)
            )

            if const_expr(self.deterministic):
                # pyre-ignore[16]
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            delay_semaphore_release = self.is_causal
            n_block_global_max = cute.ceil_div(seqlen.seqlen_k, self.tile_n)

            # pyre-ignore[28]
            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                pipeline_dQ.consumer_wait(dQ_consumer_state)
                # TMEM -> RMEM
                tdQrdQ_t2r = cute.make_fragment(tdQrdQ_t2r_shape, Float32)
                cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
                cute.arch.fence_view_async_tmem_load()

                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()

                gdQaccum_cur = gdQaccum[None, None, m_block]

                for stage in cutlass.range_constexpr(
                    cute.size(tdQrdQ_t2r, mode=[1])
                ):  # 4
                    smem_idx = dQ_tma_store_producer_state.index
                    tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                    tdQrdQ_r2s = cute.make_tensor(
                        tdQrdQ_t2r[None, stage, None, None].iterator, tdQsdQ_r2s.shape
                    )
                    cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)

                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    # semaphore acquire
                    if const_expr(self.deterministic and stage == 0):
                        # pyre-ignore[16]
                        if const_expr(self.spt):
                            if const_expr(
                                self.is_causal
                                or block_info.window_size_right is not None
                            ):
                                n_idx_right = (
                                    (m_block + 1) * self.tile_m
                                    + seqlen.seqlen_k
                                    - seqlen.seqlen_q
                                )
                                if const_expr(block_info.window_size_right is not None):
                                    # pyre-ignore[58]
                                    n_idx_right += block_info.window_size_right
                                n_block_max_for_m_block = min(
                                    n_block_global_max,
                                    cute.ceil_div(n_idx_right, self.tile_n),
                                )
                            else:
                                n_block_max_for_m_block = n_block_global_max
                            lock_value = n_block_max_for_m_block - 1 - n_block
                        else:
                            lock_value = n_block
                        barrier.wait_eq(
                            # pyre-ignore[61]
                            mdQ_semaphore_cur[(m_block, None)].iterator,
                            tidx,
                            0,
                            lock_value,
                        )
                    self.reduce_sync_barrier.arrive_and_wait()
                    # Copy from shared memory to global memory
                    if is_tma_warp:
                        with cute.arch.elect_one():
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdQaccum[None, smem_idx].iterator,
                                gdQaccum_cur[None, stage].iterator,
                                # pyre-ignore[16]
                                self.tma_copy_bytes["dQ"] // 1,
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(
                            self.sdQaccum_stage - 1, read=read_flag
                        )
                    self.reduce_sync_barrier.arrive_and_wait()
                    dQ_tma_store_producer_state.advance()
                    if const_expr(
                        self.deterministic and stage == 0 and delay_semaphore_release
                    ):
                        if m_block > m_block_min:
                            barrier.arrive_inc(
                                # pyre-ignore[61]
                                mdQ_semaphore_cur[(m_block - 1, None)].iterator,
                                tidx,
                                0,
                                1,
                            )

                # semaphore release
                # NOTE: arrive_inc calls red_release which issues membar
                if const_expr(self.deterministic and not delay_semaphore_release):
                    if is_tma_warp:
                        cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                    self.reduce_sync_barrier.arrive_and_wait()
                    barrier.arrive_inc(
                        # pyre-ignore[61]
                        mdQ_semaphore_cur[m_block, None].iterator,
                        tidx,
                        0,
                        1,
                    )

            if const_expr(not self.is_local) or m_block_min < m_block_max:
                if is_tma_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                self.reduce_sync_barrier.arrive_and_wait()
                # final semaphore release
                if const_expr(self.deterministic and delay_semaphore_release):
                    barrier.arrive_inc(
                        # pyre-ignore[61]
                        mdQ_semaphore_cur[(m_block_max - 1, None)].iterator,
                        tidx,
                        0,
                        1,
                    )

            if const_expr(
                self.deterministic
                and not self.spt
                and block_info.window_size_left is not None
            ):
                m_block_global_max = cute.ceil_div(seqlen.seqlen_q, self.tile_m)
                # pyre-ignore[28]
                for m_block in cutlass.range(m_block_max, m_block_global_max, unroll=1):
                    barrier.arrive_inc(
                        # pyre-ignore[61]
                        mdQ_semaphore_cur[(m_block, None)].iterator,
                        tidx,
                        0,
                        1,
                    )

            # Pipeline credit: replenish dQ empty init credit for next tile.
            # All 4 reduce warps must participate (4 × elect_one = 4 arrives
            # = arrive_count) to flip the mbarrier phase.
            if const_expr(self.is_persistent):
                with cute.arch.elect_one():
                    # pyre-ignore[19]
                    pipeline_dQ.sync_object_empty.arrive(0, pipeline_dQ.consumer_mask)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        warp_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen: SeqlenInfoQK,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        softmax_scale: Float32,
        mdSFK_out: Optional[cute.Tensor] = None,  # Output SF tensor for dK (MXFP8)
        mdSFV_out: Optional[cute.Tensor] = None,  # Output SF tensor for dV (MXFP8)
    ):
        wg_idx = (
            cute.arch.thread_idx()[0]
            % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        ) // 128
        num_wg = cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128

        assert self.qhead_per_kvhead == 1, "This epilogue path is only for MHA"
        mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]
        mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, head_idx]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )

        # dV
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom, tdVtdV)
        thr_tmem_ld_dV = tiled_tmem_ld_dV.get_slice(tidx)

        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)
        tdVtdV_t2r = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV = thr_mma_dV.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV.iterator, tdVcdV.layout)

        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_fragment(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            # pyre-ignore[16]
            self.dv_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # Compute layout_tv for the output dtype (may differ from F32 TMEM layout)
        # num_rep=16 from Repetition(16), elems_per_copy depends on output dtype width
        num_rep_dV = 16
        elems_per_copy_dV = universal_copy_bits // self.dv_dtype.width
        layout_tv_dV = cute.make_layout(
            ((32, elems_per_copy_dV, num_wg), (num_rep_dV, 32)),
            stride=(
                (0, 1, elems_per_copy_dV * num_rep_dV),
                (elems_per_copy_dV, elems_per_copy_dV * num_rep_dV * num_wg),
            ),
        )
        tiled_gmem_store_dV = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=layout_tv_dV,
            tiler_mn=tiled_tmem_ld_dV.tiler_mn,
        )

        tdVrdV_r2s = cute.make_fragment(tdVrdV_t2r.shape, self.dv_dtype)
        if const_expr(self.output_mxfp8_dkv):
            # MXFP8 blockscaled conversion for dV:
            # From coordinate tensor layout ((16,1),4,1,1):((1@1,0),16@1,0,0):
            #   - Each thread owns 64 elements from ONE row (row implicit from thread)
            #   - Mode 0 (size 16): 16 consecutive columns (stride 1@1)
            #   - Mode 1 (size 4): column groups stepping by 16 (stride 16@1)
            #   - Flat index = column within this workgroup's column range
            #   - MXFP8 block of 32 = 2 consecutive mode-1 groups
            max_norm_rcp_dV = Float32(E4M3_MAX_NORM_RCP)
            head_idx_kv = head_idx // self.qhead_per_kvhead
            # pyre-ignore[6]
            mdSFV_cur = seqlen.offset_batch_K(mdSFV_out, batch_idx, dim=3)[
                None, None, head_idx_kv
            ]
            # Row for this thread (tidx = dp_idx, 0..127, one row per thread)
            global_row_dV = n_block * Int32(self.tile_n) + tidx
            # Predicate for SF write: skip OOB rows when seqlen_k not multiple of tile_n
            sf_row_valid_dV = global_row_dV < seqlen.seqlen_k
            # Load entire fragment into flat temporary
            flat_dV = cute.make_fragment(tdVrdV_t2r.shape, Float32)
            flat_dV.store(tdVrdV_t2r.load())
            total_elems_dV = cute.size(tdVrdV_t2r.shape)
            num_sf_blocks_dV = total_elems_dV // 32
            for sf_blk in cutlass.range_constexpr(num_sf_blocks_dV):
                base = sf_blk * 32
                # Per-thread AMAX across 32 contiguous columns (no warp redux)
                block_amax_dV = Float32(0.0)
                for k in cutlass.range_constexpr(0, 32, 2):
                    block_amax_dV = max_f32(block_amax_dV, abs_f32(flat_dV[base + k]))
                    block_amax_dV = max_f32(
                        block_amax_dV, abs_f32(flat_dV[base + k + 1])
                    )
                sf_dV, inv_scale_dV, sf_u32_dV = fused_amax_to_e8m0_scale_f32(
                    block_amax_dV, max_norm_rcp_dV
                )
                # Scale the 32 elements
                for k in cutlass.range_constexpr(32):
                    flat_dV[base + k] = flat_dV[base + k] * inv_scale_dV
                # SF column index from coordinate tensor (mode-1 block i0 = sf_blk * 2)
                i_ref_dV = sf_blk * 2
                col_base_dV = Int32(tdVcdV_t2r[((0, 0), i_ref_dV, 0, 0)][1])
                sf_col_idx_dV = col_base_dV >> Int32(5)
                if sf_row_valid_dV:
                    mdSFV_cur[(global_row_dV, sf_col_idx_dV)] = Uint8(
                        sf_u32_dV & Uint32(0xFF)
                    )
            # Store scaled values back to register fragment
            tdVrdV_t2r.store(flat_dV.load())
            _scaled_dV = cute.make_fragment_like(tdVrdV_t2r, Float32)
            _scaled_dV.store(tdVrdV_t2r.load())
            utils.cvt_f16(_scaled_dV, tdVrdV_r2s)
        else:
            for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
                dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
                tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
        gdV_tile = gdV[None, None, n_block]

        tdVgdV = thr_mma_dV.partition_C(gdV_tile)
        tdVgdV_r2g_p = thr_tmem_ld_dV.partition_D(tdVgdV)
        tdVgdV_r2g = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)

        # Predicated store: skip OOB rows when seqlen_k is not a multiple of tile_n
        seqlen_k_remaining = seqlen.seqlen_k - n_block * self.tile_n
        cdV_gmem = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
        tdVcdV_gmem_p = thr_tmem_ld_dV.partition_D(thr_mma_dV.partition_C(cdV_gmem))
        tdVcdV_gmem = self.split_wg(tdVcdV_gmem_p, wg_idx, num_wg)
        for rest_n in cutlass.range(cute.size(tdVgdV_r2g.shape[1]), unroll_full=True):
            if tdVcdV_gmem[0, rest_n, 0, 0][0] < seqlen_k_remaining:
                cute.copy(
                    tiled_gmem_store_dV,
                    tdVrdV_r2s[None, rest_n, None, None],
                    tdVgdV_r2g[None, rest_n, None, None],
                )

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # dK
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom, tdKtdK)
        thr_tmem_ld_dK = tiled_tmem_ld_dK.get_slice(tidx)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK = thr_mma_dK.partition_C(cdK)
        tdKcdK_tensor = cute.make_tensor(tdKcdK.iterator, tdKcdK.layout)

        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_fragment(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            # pyre-ignore[16]
            self.dk_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # Compute layout_tv for the output dtype (may differ from F32 TMEM layout)
        # num_rep=16 from Repetition(16), elems_per_copy depends on output dtype width
        num_rep_dK = 16
        elems_per_copy_dK = universal_copy_bits // self.dk_dtype.width
        layout_tv_dK = cute.make_layout(
            ((32, elems_per_copy_dK, num_wg), (num_rep_dK, 32)),
            stride=(
                (0, 1, elems_per_copy_dK * num_rep_dK),
                (elems_per_copy_dK, elems_per_copy_dK * num_rep_dK * num_wg),
            ),
        )
        tiled_gmem_store_dK = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=layout_tv_dK,
            tiler_mn=tiled_tmem_ld_dK.tiler_mn,
        )

        tdKrdK_r2s = cute.make_fragment(tdKrdK_t2r.shape, self.dk_dtype)

        if const_expr(self.output_mxfp8_dkv):
            # MXFP8 blockscaled conversion for dK (includes softmax_scale):
            # Same per-thread AMAX pattern as dV.
            max_norm_rcp_dK = Float32(E4M3_MAX_NORM_RCP)
            head_idx_kv_dk = head_idx // self.qhead_per_kvhead
            # pyre-ignore[6]
            mdSFK_cur = seqlen.offset_batch_K(mdSFK_out, batch_idx, dim=3)[
                None, None, head_idx_kv_dk
            ]
            global_row_dK = n_block * Int32(self.tile_n) + tidx
            # Predicate for SF write: skip OOB rows when seqlen_k not multiple of tile_n
            sf_row_valid_dK = global_row_dK < seqlen.seqlen_k
            # Load entire fragment into flat temporary
            flat_dK = cute.make_fragment(tdKrdK_t2r.shape, Float32)
            flat_dK.store(tdKrdK_t2r.load())
            # Apply softmax_scale to all elements first (packed f32x2)
            total_elems_dK = cute.size(tdKrdK_t2r.shape)
            for k_s in cutlass.range_constexpr(total_elems_dK // 2):
                flat_dK[2 * k_s], flat_dK[2 * k_s + 1] = utils.mul_packed_f32x2(
                    (flat_dK[2 * k_s], flat_dK[2 * k_s + 1]),
                    (softmax_scale, softmax_scale),
                )
            num_sf_blocks_dK = total_elems_dK // 32
            for sf_blk in cutlass.range_constexpr(num_sf_blocks_dK):
                base = sf_blk * 32
                # Per-thread AMAX across 32 contiguous columns (no warp redux)
                block_amax_dK = Float32(0.0)
                for k in cutlass.range_constexpr(0, 32, 2):
                    block_amax_dK = max_f32(block_amax_dK, abs_f32(flat_dK[base + k]))
                    block_amax_dK = max_f32(
                        block_amax_dK, abs_f32(flat_dK[base + k + 1])
                    )
                sf_dK, inv_scale_dK, sf_u32_dK = fused_amax_to_e8m0_scale_f32(
                    block_amax_dK, max_norm_rcp_dK
                )
                # Scale the 32 elements
                for k in cutlass.range_constexpr(32):
                    flat_dK[base + k] = flat_dK[base + k] * inv_scale_dK
                # SF column index from coordinate tensor
                i_ref_dK = sf_blk * 2
                col_base_dK = Int32(tdKcdK_t2r[((0, 0), i_ref_dK, 0, 0)][1])
                sf_col_idx_dK = col_base_dK >> Int32(5)
                if sf_row_valid_dK:
                    mdSFK_cur[(global_row_dK, sf_col_idx_dK)] = Uint8(
                        sf_u32_dK & Uint32(0xFF)
                    )
            # Store scaled values back
            tdKrdK_t2r.store(flat_dK.load())
            _scaled_dK = cute.make_fragment_like(tdKrdK_t2r, Float32)
            _scaled_dK.store(tdKrdK_t2r.load())
            utils.cvt_f16(_scaled_dK, tdKrdK_r2s)
        else:
            for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
                dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load() * softmax_scale
                tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdimv), (None, 0))
        gdK_tile = gdK[None, None, n_block]

        tdKgdK = thr_mma_dK.partition_C(gdK_tile)
        tdKgdK_r2g_p = thr_tmem_ld_dK.partition_D(tdKgdK)
        tdKgdK_r2g = self.split_wg(tdKgdK_r2g_p, wg_idx, num_wg)

        # Predicated store: skip OOB rows when seqlen_k is not a multiple of tile_n
        cdK_gmem = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
        tdKcdK_gmem_p = thr_tmem_ld_dK.partition_D(thr_mma_dK.partition_C(cdK_gmem))
        tdKcdK_gmem = self.split_wg(tdKcdK_gmem_p, wg_idx, num_wg)
        for rest_n in cutlass.range(cute.size(tdKgdK_r2g.shape[1]), unroll_full=True):
            if tdKcdK_gmem[0, rest_n, 0, 0][0] < seqlen_k_remaining:
                cute.copy(
                    tiled_gmem_store_dK,
                    tdKrdK_r2s[None, rest_n, None, None],
                    tdKgdK_r2g[None, rest_n, None, None],
                )

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        thr_mma: cute.core.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Optional[Float32],
        barrier_id: Int32,
        mdKV_semaphore: Optional[cute.Tensor],
        mdSF_out: Optional[cute.Tensor] = None,  # Output SF tensor for MXFP8 dK/dV
        seqlen: Optional[SeqlenInfoQK] = None,  # Seqlen info for OOB predication
    ) -> cutlass.pipeline.PipelineState:
        # assumes mma_tiler_pdo = mma_tiler_dsq = (tile_n, head_dim)
        # head_dim = head_dim_v, dk_dtype = dv_dtype
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        # pyre-ignore[16]
        if const_expr(not self.dKV_postprocess):
            sdKV = sdKV[None, None, wg_idx]  # (tile_n, 64) for bf16
        else:
            sdKV = sdKV[None, wg_idx]  # (tile_n * 32) for fp32

        # (8, tile_n / 128, 64 / 8) = (8, 1, 8) or (4, tile_n * 32 / (128 * 4)) = (4, 8)
        # pyre-ignore[16]
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)

        head_idx_kvv = head_idx // self.qhead_per_kvhead
        if const_expr(not self.dKV_postprocess):
            mdKV_cur = mdKV[None, None, head_idx_kvv, batch_idx]  # (seqlen, hdim)
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n, self.tile_hdim), (n_block, 0)
            )  # (tile_n, hdim)
            gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)  # (tile_n, hdim / 2)
            gdKV_epi = cute.local_tile(
                gdKV,
                # pyre-ignore[16]
                self.sdKV_epi_tile,
                (0, None),
            )  # (tile_n, 64, epi_stage = (hdim / 2) / 64)
        else:
            mdKV_cur = mdKV[None, head_idx_kvv, batch_idx]  # (seqlen * hdim)
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n * self.tile_hdim,), (n_block,)
            )  # (tile_n * hdim)
            gdKV = cute.logical_divide(
                gdKV_p, (self.tile_n * self.tile_hdim // num_wg,)
            )[((None, wg_idx),)]  # (tile_n * hdim / 2)
            gdKV_epi = cute.flat_divide(
                gdKV,
                # pyre-ignore[16]
                (self.sdKV_flat_epi_tile,),
            )  # (tile_n * hdim / 2 / epi_stage, epi_stage)

        # pyre-ignore[58]
        deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
        if const_expr(deterministic_KV):
            # pyre-ignore[16]
            mdKV_semaphore_cur = mdKV_semaphore[n_block, None, head_idx_kvv, batch_idx]

        if const_expr(not self.dKV_postprocess):
            tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
                tma_atom_dKV,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sdKV, 0, 2),
                cute.group_modes(gdKV_epi, 0, 2),
            )  # (TMA) and (TMA, EPI_STAGE)
            assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
            assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"
            num_epi_stages = cute.size(tdKVgdKV.shape[1])
            # pyre-ignore[16]
            assert num_epi_stages == self.num_epi_stages, (
                "Epi stage calculation is wrong"
            )
        else:
            num_epi_stages = self.num_epi_stages

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )

        read_flag = const_expr(not deterministic_KV)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # semaphore acquire
        if const_expr(deterministic_KV):
            barrier.wait_eq(
                mdKV_semaphore_cur.iterator,
                tidx,
                wg_idx,
                head_idx % self.qhead_per_kvhead,
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(
                tidx
            )
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[
                None, None, 0, 0
            ]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[
                None, None, 0, 0
            ]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_fragment(tdKVcdKV_t2r.shape, Float32)

            assert (
                cute.size(tdKVrdKV_t2r)
                == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE
            ), "RMEM<->TMEM fragment size mismatch"

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -- scale and convert
            if const_expr(scale is not None):
                for i in cutlass.range(
                    cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True
                ):
                    tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = (
                        utils.mul_packed_f32x2(
                            (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]),
                            (scale, scale),
                        )
                    )
            if const_expr(self.output_mxfp8_dkv):
                # MXFP8 blockscaled conversion:
                # After epi_stage selection, fragment may be 1D (32 elements)
                # or 2D multi-modal. Use flat iteration which works when
                # the fragment is 1D after epi_stage selection.
                # TODO: verify multi-modal shapes when num_epi_stages == 1
                max_norm_rcp = Float32(E4M3_MAX_NORM_RCP)
                lane_id = cute.arch.lane_idx()

                # Offset SF tensor to current batch/head
                head_idx_kv = head_idx // self.qhead_per_kvhead
                mdSF_cur = mdSF_out[None, None, head_idx_kv, batch_idx]

                col_coord = Int32(tdKVcdKV_t2r[0][1])
                sf_col_idx = Int32(col_coord / Int32(32))

                for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape), unroll_full=True):
                    amax = redux_sync_max_abs_f32(tdKVrdKV_t2r[i])
                    sf, inv_scale, sf_u32 = fused_amax_to_e8m0_scale_f32(
                        amax, max_norm_rcp
                    )
                    tdKVrdKV_t2r[i] = tdKVrdKV_t2r[i] * inv_scale
                    if lane_id == 0:
                        row_coord = Int32(tdKVcdKV_t2r[i][0])
                        global_row = n_block * Int32(self.tile_n) + row_coord
                        # pyre-ignore[16]
                        if global_row < seqlen.seqlen_k:
                            mdSF_cur[(global_row, sf_col_idx)] = Uint8(
                                sf_u32 & Uint32(0xFF)
                            )

                # Convert scaled FP32 to output FP8 dtype
                # pyre-ignore[16]
                tdKVrdKV = cute.make_fragment(tdKVrdKV_t2r.shape, self.dv_dtype)
                _scaled_frg = cute.make_fragment_like(tdKVrdKV_t2r, Float32)
                _scaled_frg.store(tdKVrdKV_t2r.load())
                utils.cvt_f16(_scaled_frg, tdKVrdKV)
            else:
                tdKVrdKV = cute.make_fragment(
                    tdKVrdKV_t2r.shape, self.dv_dtype
                )  # (32 columns)
                tdKVrdKV.store(tdKVrdKV_t2r.load().to(self.dv_dtype))

            # RMEM -> SMEM -- copy, fence and barrier
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                if const_expr(not self.dKV_postprocess):
                    # pyre-ignore[61]
                    cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                else:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKV.iterator,
                            gdKV_epi[None, epi_stage].iterator,
                            # pyre-ignore[16]
                            self.tma_copy_bytes["dKacc"],
                        )
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx,
                    number_of_threads=128 + cute.arch.WARP_SIZE,
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx,
                number_of_threads=128 + cute.arch.WARP_SIZE,
            )

        # semaphore release
        # NOTE: arrive_inc calls red_release which issues membar
        if const_expr(deterministic_KV):
            if leader_warp:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV


def mma_partition_fragment_AB(
    thr_mma: cute.core.ThrMma,
    sA: Optional[cute.Tensor],
    sB: Optional[cute.Tensor],
    swap_AB: bool,
):
    if const_expr(not swap_AB):
        return (
            thr_mma.make_fragment_A(thr_mma.partition_A(sA))
            if sA is not None
            else None,
            thr_mma.make_fragment_B(thr_mma.partition_B(sB))
            if sB is not None
            else None,
        )
    else:
        return (
            thr_mma.make_fragment_B(thr_mma.partition_B(sA))
            if sA is not None
            else None,
            thr_mma.make_fragment_A(thr_mma.partition_A(sB))
            if sB is not None
            else None,
        )


class FlashAttentionBackwardSm90:
    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        tile_m: int = 64,
        tile_n: int = 128,
        Q_stage: int = 2,
        dO_stage: int = 2,
        PdS_stage: int = 2,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 2,
        AtomLayoutMdQ: int = 1,
        num_threads: int = 384,
        V_in_regs: bool = False,
        use_silu: bool = False,
        is_persistent: bool = False,
        is_diagonal: bool = False,
        accumulate_dKV: bool = False,
        reorder_sdp: bool = False,
    ):
        self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        assert self.tile_hdim == self.tile_hdimv, (
            f"tile_hdim ({self.tile_hdim}) must equal tile_hdimv ({self.tile_hdimv}) "
            "for SM90 backward (dKVaccum f32 reduce-add assumes same hdim)"
        )
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_diagonal = is_diagonal
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.Q_stage = Q_stage
        self.dO_stage = dO_stage
        self.PdS_stage = PdS_stage
        assert self.dO_stage in [1, self.Q_stage]
        assert self.PdS_stage in [1, self.Q_stage]
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.num_mma_warp_groups = (self.num_threads // 128) - 1
        self.mma_dkv_is_rs = (
            AtomLayoutMSdP == 1
            and AtomLayoutNdKV == self.num_mma_warp_groups
            and SdP_swapAB
            and not dKV_swapAB
        )
        self.V_in_regs = V_in_regs
        self.use_silu = use_silu
        self.is_persistent = is_persistent
        self.accumulate_dKV = accumulate_dKV
        self.reorder_sdp = reorder_sdp
        self.shuffle_LSE = self.SdP_swapAB and self.tile_hdim <= 64
        self.shuffle_dPsum = self.SdP_swapAB and self.tile_hdim <= 64

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        Q_stage,
        num_threads,
        V_in_regs=False,
    ) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQaccum_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQaccum_type not in [self.dtype]):
            raise TypeError("dQ output tensor must match Q dtype (BF16/FP16)")
        if const_expr(self.qhead_per_kvhead == 1):
            if const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError(
                    "mdK and mdV tensors must have the same data type as mQ"
                )
        else:
            if const_expr(not (mdK_type == mdV_type == Float32)):
                raise TypeError(
                    "mdKaccum and mdVaccum tensors must have the data type Float32"
                )
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        (
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sdO_layout,
            self.sPdS_layout,
        ) = [
            sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for shape, stage in [
                ((self.tile_m, self.tile_hdim), self.Q_stage),
                ((self.tile_n, self.tile_hdim), None),
                ((self.tile_n, self.tile_hdimv), None),
                ((self.tile_m, self.tile_hdimv), self.dO_stage),
                ((self.tile_m, self.tile_n), self.PdS_stage),
            ]
        ]
        # 2D row-major BF16 layout for direct TMA reduce-add into mdQ.
        self.sdQaccum_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_hdim),
        )

    def _get_tiled_mma(self):
        # S = Q @ K.T, dP = dO @ V.T
        atom_layout_SdP = (
            self.AtomLayoutMSdP,
            self.num_mma_warp_groups // self.AtomLayoutMSdP,
        )
        tiler_mn_SdP = (
            self.tile_m // atom_layout_SdP[0],
            self.tile_n // atom_layout_SdP[1],
        )
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(
                atom_layout_SdP if not self.SdP_swapAB else atom_layout_SdP[::-1]
            )
            + (1,),
            tiler_mn=tiler_mn_SdP if not self.SdP_swapAB else tiler_mn_SdP[::-1],
        )
        # dV = P.T @ dO, dK = dS.T @ Q
        atom_layout_dKV = (
            self.AtomLayoutNdKV,
            self.num_mma_warp_groups // self.AtomLayoutNdKV,
        )
        tiler_mn_dK = (
            self.tile_n // atom_layout_dKV[0],
            self.tile_hdim // atom_layout_dKV[1],
        )
        tiler_mn_dV = (
            self.tile_n // atom_layout_dKV[0],
            self.tile_hdimv // atom_layout_dKV[1],
        )
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.MN
                if not self.mma_dkv_is_rs
                else warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=(
                    atom_layout_dKV if not self.dKV_swapAB else atom_layout_dKV[::-1]
                )
                + (1,),
                tiler_mn=tiler_mn_d if not self.dKV_swapAB else tiler_mn_d[::-1],
                a_source=warpgroup.OperandSource.RMEM
                if self.mma_dkv_is_rs
                else warpgroup.OperandSource.SMEM,
            )
            for tiler_mn_d in (tiler_mn_dK, tiler_mn_dV)
        ]
        # dQ = dS @ K
        atom_layout_dQ = (
            self.AtomLayoutMdQ,
            self.num_mma_warp_groups // self.AtomLayoutMdQ,
        )
        tiler_mn_dQ = (
            self.tile_m // atom_layout_dQ[0],
            self.tile_hdim // atom_layout_dQ[1],
        )
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K
            if not self.dQ_swapAB
            else warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN
            if not self.dQ_swapAB
            else warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(
                atom_layout_dQ if not self.dQ_swapAB else atom_layout_dQ[::-1]
            )
            + (1,),
            tiler_mn=tiler_mn_dQ if not self.dQ_swapAB else tiler_mn_dQ[::-1],
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        sQ_alignment = sK_alignment = sV_alighment = sdQaccum_alignment = (
            sdO_alignment
        ) = 1024

        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[
                cute.struct.MemRange[type, cute.cosize(layout)], alignment
            ]
            for (layout, type, alignment) in [
                (self.sQ_layout, self.dtype, sQ_alignment),
                (self.sK_layout, self.dtype, sK_alignment),
                (self.sV_layout, self.dtype, sV_alighment),
                (self.sdO_layout, self.dtype, sdO_alignment),
                (self.sdQaccum_layout, self.dtype, sdQaccum_alignment),
            ]
        ]

        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = (
            cute.cosize(self.sPdS_layout) if const_expr(not self.mma_dkv_is_rs) else 0
        )
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, cute.round_up(self.tile_m, 64) * self.Q_stage
            ],
            128,
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, cute.round_up(self.tile_m, 64) * self.dO_stage
            ],
            128,
        ]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        mAttnScale: Optional[cute.Tensor] = None,
        mTileToBatch: Optional[cute.Tensor] = None,
        mTileToHead: Optional[cute.Tensor] = None,
        mTileToBlock: Optional[cute.Tensor] = None,
        mQ_alt: Optional[cute.Tensor] = None,
        mdO_alt: Optional[cute.Tensor] = None,
        mLSE_alt: Optional[cute.Tensor] = None,
        mdPsum_alt: Optional[cute.Tensor] = None,
        mdQaccum_alt: Optional[cute.Tensor] = None,
        mCuSeqlensQ_alt: Optional[cute.Tensor] = None,
        mAttnScale_alt: Optional[cute.Tensor] = None,
    ):
        assert (
            mdQ_semaphore is None and mdK_semaphore is None and mdV_semaphore is None
        ), "determinism not supported yet for Sm90"

        self._check_type(
            # pyre-ignore[6]
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
            )
        )

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            cute.make_tensor(
                t.iterator, cute.make_layout(t.shape, stride=new_stride(t))
            )
            if t is not None
            else None
            for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]

        # pyre-ignore[16]
        self.is_varlen = mCuSeqlensQ is not None
        QKV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mdK_raw = mdK
        mdV_raw = mdV
        mQ, mK, mV, mdK, mdV, mdO, mdQaccum = [
            utils.select(t, QKV_layout_transpose)
            for t in (mQ, mK, mV, mdK, mdV, mdO, mdQaccum)
        ]
        LSE_dPsum_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE, mdPsum = [utils.select(t, LSE_dPsum_transpose) for t in (mLSE, mdPsum)]

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()

        # pyre-ignore[16]
        self.num_mma_threads = tiled_mma_SdP.size
        assert self.num_mma_threads + 128 == self.num_threads

        # pyre-ignore[16]
        self.num_threads_per_warp_group = 128
        # pyre-ignore[16]
        self.num_producer_threads = 32

        # pyre-ignore[16]
        self.num_mma_regs = 240
        # pyre-ignore[16]
        self.num_producer_regs = 24

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        # pyre-ignore[16]
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                # pyre-ignore[16]
                ("Q", mQ, self.sQ_layout),
                # pyre-ignore[16]
                ("K", mK, self.sK_layout),
                # pyre-ignore[16]
                ("V", mV, self.sV_layout),
                # pyre-ignore[16]
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dQ, tma_tensor_dQ = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp(),
            mdQaccum,
            # pyre-ignore[16]
            cute.select(self.sdQaccum_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )

        tma_atom_Q_alt = None
        tma_tensor_Q_alt = None
        tma_atom_dO_alt = None
        tma_tensor_dO_alt = None
        tma_atom_dQ_alt = None
        tma_tensor_dQ_alt = None
        mLSE_alt_t = None
        mdPsum_alt_t = None
        if const_expr(mQ_alt is not None):
            mQ_alt_t = cute.make_tensor(
                # pyre-ignore[16]
                mQ_alt.iterator,
                # pyre-ignore[16]
                cute.make_layout(mQ_alt.shape, stride=new_stride(mQ_alt)),
            )
            mdO_alt_t = cute.make_tensor(
                mdO_alt.iterator,
                cute.make_layout(mdO_alt.shape, stride=new_stride(mdO_alt)),
            )
            mLSE_alt_t = cute.make_tensor(
                mLSE_alt.iterator,
                cute.make_layout(mLSE_alt.shape, stride=new_stride(mLSE_alt)),
            )
            mdPsum_alt_t = cute.make_tensor(
                mdPsum_alt.iterator,
                cute.make_layout(mdPsum_alt.shape, stride=new_stride(mdPsum_alt)),
            )
            mdQaccum_alt_t = cute.make_tensor(
                mdQaccum_alt.iterator,
                cute.make_layout(mdQaccum_alt.shape, stride=new_stride(mdQaccum_alt)),
            )
            mQ_alt_t = utils.select(mQ_alt_t, QKV_layout_transpose)
            mdO_alt_t = utils.select(mdO_alt_t, QKV_layout_transpose)
            mLSE_alt_t = utils.select(mLSE_alt_t, LSE_dPsum_transpose)
            mdPsum_alt_t = utils.select(mdPsum_alt_t, LSE_dPsum_transpose)
            mdQaccum_alt_t = utils.select(mdQaccum_alt_t, QKV_layout_transpose)
            tma_atom_Q_alt, tma_tensor_Q_alt = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mQ_alt_t,
                cute.select(self.sQ_layout, mode=[0, 1]),
                (self.tile_m, self.tile_hdim),
            )
            tma_atom_dO_alt, tma_tensor_dO_alt = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mdO_alt_t,
                cute.select(self.sdO_layout, mode=[0, 1]),
                (self.tile_m, self.tile_hdimv),
            )
            tma_atom_dQ_alt, tma_tensor_dQ_alt = cpasync.make_tiled_tma_atom(
                cpasync.CopyReduceBulkTensorTileS2GOp(),
                mdQaccum_alt_t,
                cute.select(self.sdQaccum_layout, mode=[0, 1]),
                (self.tile_m, self.tile_hdim),
            )

        if const_expr(
            self.is_persistent
            and mCuSeqlensK is not None
            and not self.is_causal
            and not self.is_local
        ):
            TileScheduler = PersistentVarlenLookupScheduler
        elif const_expr(mCuSeqlensK is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            # pyre-ignore[16]
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mK.shape[2]),
            cute.size(mK.shape[3])
            if mCuSeqlensK is None
            else cute.size(mCuSeqlensK) - 1,
            # pyre-ignore[6]
            1,  # num_splits
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mK.shape[0]),
            # pyre-ignore[6]
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            # pyre-ignore[6]
            qhead_per_kvhead_packgqa=1,
            # pyre-ignore[6]
            element_size=self.dtype.width // 8,
            # pyre-ignore[6]
            is_persistent=self.is_persistent,
            # pyre-ignore[6]
            lpt=False,
            mTileToBatch=mTileToBatch,
            mTileToHead=mTileToHead,
            mTileToBlock=mTileToBlock,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        # pyre-ignore[6]
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        if const_expr(self.is_varlen):
            _univ_bits = 128
            _async_elems = _univ_bits // self.dtype.width
            _copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=_univ_bits,
            )
            _sK_dim1 = self.sK_layout.outer.shape[1][0] // _async_elems
            _num_epi_threads = 32
            _thr_layout = cute.make_ordered_layout(
                (_num_epi_threads // _sK_dim1, _sK_dim1), order=(1, 0)
            )
            _val_layout = cute.make_layout((1, _async_elems))
            gmem_tiled_copy_dKV = cute.make_tiled_copy_tv(
                _copy_atom, _thr_layout, _val_layout
            )
        else:
            gmem_tiled_copy_dKV = None

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dK,
            tma_tensor_dV,
            tma_tensor_dQ,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            tma_atom_dQ,
            mLSE,
            mdPsum,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            # pyre-ignore[16]
            self.sPdS_layout,
            self.sdO_layout,
            # pyre-ignore[16]
            self.sdQaccum_layout,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            mCuSeqlensQ,
            mCuSeqlensK,
            mAttnScale,
            window_size_left,
            window_size_right,
            gmem_tiled_copy_dKV,
            mdK_raw,
            mdV_raw,
            tma_tensor_Q_alt,
            tma_atom_Q_alt,
            tma_tensor_dO_alt,
            tma_atom_dO_alt,
            tma_tensor_dQ_alt,
            tma_atom_dQ_alt,
            mLSE_alt_t,
            mdPsum_alt_t,
            mCuSeqlensQ_alt,
            mAttnScale_alt,
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
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQ: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tma_atom_dQ: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.ComposedLayout,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mAttnScale: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        gmem_tiled_copy_dKV: Optional[cute.TiledCopy] = None,
        mdK_raw: Optional[cute.Tensor] = None,
        mdV_raw: Optional[cute.Tensor] = None,
        mQ_alt: Optional[cute.Tensor] = None,
        tma_atom_Q_alt: Optional[cute.CopyAtom] = None,
        mdO_alt: Optional[cute.Tensor] = None,
        tma_atom_dO_alt: Optional[cute.CopyAtom] = None,
        mdQ_alt: Optional[cute.Tensor] = None,
        tma_atom_dQ_alt: Optional[cute.CopyAtom] = None,
        mLSE_alt: Optional[cute.Tensor] = None,
        mdPsum_alt: Optional[cute.Tensor] = None,
        mCuSeqlensQ_alt: Optional[cute.Tensor] = None,
        mAttnScale_alt: Optional[cute.Tensor] = None,
    ):
        # pyre-ignore[16]
        self.window_size_left = window_size_left
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # prefetch TMA descriptors
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            # pyre-ignore[16]
            self.num_mma_threads // self.num_threads_per_warp_group,
        )
        # pyre-ignore[16]
        tx_count_Q = self.tma_copy_bytes["Q"]
        tx_count_dO = self.tma_copy_bytes["dO"]
        if const_expr(not self.use_silu):
            tx_count_Q = tx_count_Q + self.tma_copy_bytes["LSE"]
            tx_count_dO = tx_count_dO + self.tma_copy_bytes["dPsum"]
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=tx_count_Q,
            # pyre-ignore[6]
            init_wait=False,
        )
        pipeline_dO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=tx_count_dO,
            # pyre-ignore[6]
            init_wait=True,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = None
        if const_expr(not self.mma_dkv_is_rs):
            sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdQaccum = storage.sdQaccum.get_tensor(
            sdQaccum_layout.outer, swizzle=sdQaccum_layout.inner
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
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            # pyre-ignore[16]
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=None,
            mSeqUsedK=None,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        # pyre-ignore[16]
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        SeqlenInfoCls_alt = None
        block_info_alt = None
        if const_expr(mQ_alt is not None):
            SeqlenInfoCls_alt = partial(
                SeqlenInfoQK.create,
                # pyre-ignore[16]
                seqlen_q_static=mQ_alt.shape[0],
                seqlen_k_static=mK.shape[0],
                mCuSeqlensQ=mCuSeqlensQ_alt,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=None,
                mSeqUsedK=None,
                tile_m=self.tile_m,
                tile_n=self.tile_n,
            )
            block_info_alt = BlockInfo(
                # pyre-ignore[6]
                self.tile_m,
                # pyre-ignore[6]
                self.tile_n,
                # pyre-ignore[6]
                False,
                # pyre-ignore[6]
                False,
                # pyre-ignore[6]
                False,
                None,
                None,
                # pyre-ignore[6]
                qhead_per_kvhead_packgqa=1,
            )

        if warp_idx < 4:
            # pyre-ignore[16]
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    mQ_alt,
                    mdO_alt,
                    tma_atom_Q_alt,
                    tma_atom_dO_alt,
                    SeqlenInfoCls_alt,
                    block_info_alt,
                )
            if warp_idx == 1:
                for warp_group_idx in cutlass.range(self.num_mma_warp_groups):
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                        number_of_threads=self.num_threads_per_warp_group
                        + cute.arch.WARP_SIZE,
                    )
                self.dQaccum_store(
                    mdQ,
                    tma_atom_dQ,
                    sdQaccum,
                    block_info,
                    TileSchedulerCls,
                    SeqlenInfoCls,
                    mdQ_alt,
                    tma_atom_dQ_alt,
                    SeqlenInfoCls_alt,
                    block_info_alt,
                )
        else:
            # pyre-ignore[16]
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                mAttnScale,
                gmem_tiled_copy_dKV,
                mdK_raw,
                mdV_raw,
                mQ_alt,
                SeqlenInfoCls_alt,
                block_info_alt,
                mAttnScale_alt,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mQ_alt: Optional[cute.Tensor] = None,
        mdO_alt: Optional[cute.Tensor] = None,
        tma_atom_Q_alt: Optional[cute.CopyAtom] = None,
        tma_atom_dO_alt: Optional[cute.CopyAtom] = None,
        SeqlenInfoCls_alt: Optional[Callable] = None,
        block_info_alt: Optional[BlockInfo] = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_idx_in_wg == 0:
            producer_state_Q = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
            )
            producer_state_dO = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                gK = cute.local_tile(
                    mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0)
                )
                gV = cute.local_tile(
                    mV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0)
                )

                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
                gdO = cute.local_tile(
                    mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0)
                )
                if const_expr(not self.use_silu):
                    mLSE_cur = seqlen.offset_batch_Q(
                        mLSE, batch_idx, dim=2, padded=True
                    )[None, head_idx]
                    mdPsum_cur = seqlen.offset_batch_Q(
                        mdPsum, batch_idx, dim=2, padded=True
                    )[None, head_idx]
                    gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                    gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

                # pyre-ignore[23]
                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK, single_stage=True
                )
                # pyre-ignore[23]
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV, single_stage=True
                )
                # pyre-ignore[23]
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ
                )
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
                # pyre-ignore[23]
                load_dO, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO, 0, cute.make_layout(1), gdO, sdO
                )
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
                if const_expr(not self.use_silu):
                    # pyre-ignore[61]
                    load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                    load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_Q)
                    # pyre-ignore[61]
                    load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
                    load_dPsum = copy_utils.tma_producer_copy_fn(
                        load_dPsum, pipeline_dO
                    )

                m_block_min, m_block_max = block_info.get_m_block_min_max(
                    seqlen, n_block
                )
                if const_expr(not self.is_local) or m_block_min < m_block_max:
                    # First iteration: load K together w Q & LSE, then V together w dO & dPsum
                    m_block = m_block_min
                    pipeline_Q.producer_acquire(
                        producer_state_Q,
                        # pyre-ignore[16]
                        extra_tx_count=self.tma_copy_bytes["K"],
                    )
                    load_K(
                        tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q)
                    )
                    load_Q(m_block, producer_state=producer_state_Q)
                    if const_expr(not self.use_silu):
                        with cute.arch.elect_one():
                            # pyre-ignore[61]
                            load_LSE(m_block, producer_state=producer_state_Q)
                    producer_state_dO_cur = (
                        producer_state_dO
                        if const_expr(self.Q_stage != self.dO_stage)
                        else producer_state_Q
                    )
                    pipeline_dO.producer_acquire(
                        producer_state_dO_cur, extra_tx_count=self.tma_copy_bytes["V"]
                    )
                    load_V(
                        tma_bar_ptr=pipeline_dO.producer_get_barrier(
                            producer_state_dO_cur
                        )
                    )
                    load_dO(m_block, producer_state=producer_state_dO_cur)
                    if const_expr(not self.use_silu):
                        with cute.arch.elect_one():
                            # pyre-ignore[61]
                            load_dPsum(m_block, producer_state=producer_state_dO_cur)
                    producer_state_Q.advance()
                    producer_state_dO.advance()
                    # Subsequent iterations: load Q & LSE, then dO & dPsum
                    # pyre-ignore[28]
                    for m_block in cutlass.range(
                        m_block_min + 1, m_block_max, unroll=1
                    ):
                        pipeline_Q.producer_acquire(producer_state_Q)
                        load_Q(m_block, producer_state=producer_state_Q)
                        if const_expr(not self.use_silu):
                            with cute.arch.elect_one():
                                # pyre-ignore[61]
                                load_LSE(m_block, producer_state=producer_state_Q)
                        producer_state_dO_cur = (
                            producer_state_dO
                            if const_expr(self.Q_stage != self.dO_stage)
                            else producer_state_Q
                        )
                        pipeline_dO.producer_acquire(producer_state_dO_cur)
                        load_dO(m_block, producer_state=producer_state_dO_cur)
                        if const_expr(not self.use_silu):
                            with cute.arch.elect_one():
                                # pyre-ignore[61]
                                load_dPsum(
                                    m_block, producer_state=producer_state_dO_cur
                                )
                        producer_state_Q.advance()
                        producer_state_dO.advance()

                if const_expr(mQ_alt is not None):
                    # pyre-ignore[29]
                    seqlen_alt = SeqlenInfoCls_alt(batch_idx)
                    mQ_alt_cur = seqlen_alt.offset_batch_Q(mQ_alt, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    mdO_alt_cur = seqlen_alt.offset_batch_Q(mdO_alt, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    gQ_alt = cute.local_tile(
                        mQ_alt_cur, (self.tile_m, self.tile_hdim), (None, 0)
                    )
                    gdO_alt = cute.local_tile(
                        mdO_alt_cur, (self.tile_m, self.tile_hdimv), (None, 0)
                    )
                    # pyre-ignore[6, 23]
                    load_Q_alt, _, _ = copy_utils.tma_get_copy_fn(
                        # pyre-ignore[6]
                        tma_atom_Q_alt,
                        0,
                        cute.make_layout(1),
                        gQ_alt,
                        sQ,
                    )
                    load_Q_alt = copy_utils.tma_producer_copy_fn(load_Q_alt, pipeline_Q)
                    # pyre-ignore[6, 23]
                    load_dO_alt, _, _ = copy_utils.tma_get_copy_fn(
                        # pyre-ignore[6]
                        tma_atom_dO_alt,
                        0,
                        cute.make_layout(1),
                        gdO_alt,
                        sdO,
                    )
                    load_dO_alt = copy_utils.tma_producer_copy_fn(
                        load_dO_alt, pipeline_dO
                    )
                    m_block_alt_min, m_block_alt_max = (
                        # pyre-ignore[16]
                        block_info_alt.get_m_block_min_max(seqlen_alt, n_block)
                    )

                    if const_expr(self.is_local):
                        if (
                            m_block_min >= m_block_max
                            and m_block_alt_min < m_block_alt_max
                        ):
                            m_block_alt0 = m_block_alt_min
                            pipeline_Q.producer_acquire(
                                producer_state_Q,
                                extra_tx_count=self.tma_copy_bytes["K"],
                            )
                            load_K(
                                tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                    producer_state_Q
                                )
                            )
                            load_Q_alt(m_block_alt0, producer_state=producer_state_Q)
                            producer_state_dO_cur_first = (
                                producer_state_dO
                                if const_expr(self.Q_stage != self.dO_stage)
                                else producer_state_Q
                            )
                            pipeline_dO.producer_acquire(
                                producer_state_dO_cur_first,
                                extra_tx_count=self.tma_copy_bytes["V"],
                            )
                            load_V(
                                tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                    producer_state_dO_cur_first
                                )
                            )
                            load_dO_alt(
                                m_block_alt0,
                                producer_state=producer_state_dO_cur_first,
                            )
                            producer_state_Q.advance()
                            producer_state_dO.advance()
                            m_block_alt_min = m_block_alt_min + 1

                    # pyre-ignore[28]
                    for m_block in cutlass.range(
                        m_block_alt_min, m_block_alt_max, unroll=1
                    ):
                        pipeline_Q.producer_acquire(producer_state_Q)
                        load_Q_alt(m_block, producer_state=producer_state_Q)
                        producer_state_dO_cur = (
                            producer_state_dO
                            if const_expr(self.Q_stage != self.dO_stage)
                            else producer_state_Q
                        )
                        pipeline_dO.producer_acquire(producer_state_dO_cur)
                        load_dO_alt(m_block, producer_state=producer_state_dO_cur)
                        producer_state_Q.advance()
                        producer_state_dO.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        mAttnScale: Optional[cute.Tensor] = None,
        gmem_tiled_copy_dKV: Optional[cute.TiledCopy] = None,
        mdK_raw: Optional[cute.Tensor] = None,
        mdV_raw: Optional[cute.Tensor] = None,
        mQ_alt: Optional[cute.Tensor] = None,
        SeqlenInfoCls_alt: Optional[Callable] = None,
        block_info_alt: Optional[BlockInfo] = None,
        mAttnScale_alt: Optional[cute.Tensor] = None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(
            # pyre-ignore[16]
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(warp_group_idx))
        # S = Q @ K.T
        # pyre-ignore[6]
        tSrQ, tSrK = mma_partition_fragment_AB(wg_mma_SdP, sQ, sK, self.SdP_swapAB)
        # dP = dO @ V.T
        # pyre-ignore[6]
        tdPrdO, tdPrV = mma_partition_fragment_AB(wg_mma_SdP, sdO, sV, self.SdP_swapAB)
        # dV += P.T @ dO
        sPt = utils.transpose_view(sP) if sP is not None else None
        sdOt = utils.transpose_view(sdO)
        tdVrPt, tdVrdOt = mma_partition_fragment_AB(
            # pyre-ignore[6]
            wg_mma_dV,
            sPt,
            sdOt,
            self.dKV_swapAB,
        )
        # dK += dS.T @ Q
        sdSt = utils.transpose_view(sdS)
        sQt = utils.transpose_view(sQ)
        tdKrdSt, tdKrQt = mma_partition_fragment_AB(
            # pyre-ignore[6]
            wg_mma_dK,
            sdSt,
            sQt,
            self.dKV_swapAB,
        )
        # dQ = dS @ K
        sKt = utils.transpose_view(sK)
        # pyre-ignore[6]
        tdQrdS, tdQrKt = mma_partition_fragment_AB(wg_mma_dQ, sdS, sKt, self.dQ_swapAB)

        # Smem copy atom tiling
        smem_copy_atom_PdS = utils.get_smem_store_atom(
            # pyre-ignore[6]
            self.arch,
            self.dtype,
            transpose=self.SdP_swapAB,
        )
        smem_thr_copy_PdS = cute.make_tiled_copy_C(
            smem_copy_atom_PdS, tiled_mma_SdP
        ).get_slice(tidx)
        tPsP = None
        if const_expr(sP is not None):
            tPsP = smem_thr_copy_PdS.partition_D(
                sP if const_expr(not self.SdP_swapAB) else sPt
            )
        tdSsdS = smem_thr_copy_PdS.partition_D(
            sdS if const_expr(not self.SdP_swapAB) else sdSt
        )

        sLSE_mma = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_mma = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        if const_expr(self.SdP_swapAB):
            sLSE_mma = utils.transpose_view(sLSE_mma)
            sdPsum_mma = utils.transpose_view(sdPsum_mma)
        LSEslice = (
            (None, 0, None) if const_expr(not self.SdP_swapAB) else (0, None, None)
        )
        tLSEsLSE = utils.make_acc_tensor_mn_view(thr_mma_SdP.partition_C(sLSE_mma))[
            LSEslice
        ]
        tLSEsdPsum = utils.make_acc_tensor_mn_view(thr_mma_SdP.partition_C(sdPsum_mma))[
            LSEslice
        ]

        smem_copy_atom_dQ = utils.get_smem_store_atom(
            # pyre-ignore[6]
            self.arch,
            self.dtype,
            transpose=self.dQ_swapAB,
        )
        smem_thr_copy_dQaccum = cute.make_tiled_copy_C(
            smem_copy_atom_dQ, tiled_mma_dQ
        ).get_slice(tidx)
        sdQaccum_t = (
            sdQaccum
            if const_expr(not self.dQ_swapAB)
            else utils.transpose_view(sdQaccum)
        )
        tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum_t)

        dV_shape = (self.tile_n, self.tile_hdimv)
        acc_dV = cute.make_fragment(
            tiled_mma_dV.partition_shape_C(
                dV_shape if not self.dKV_swapAB else dV_shape[::-1]
            ),
            Float32,
        )
        dK_shape = (self.tile_n, self.tile_hdim)
        acc_dK = cute.make_fragment(
            tiled_mma_dK.partition_shape_C(
                dK_shape if not self.dKV_swapAB else dK_shape[::-1]
            ),
            Float32,
        )

        mma_qk_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrK,
            swap_AB=self.SdP_swapAB,
        )
        mma_dov_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            (self.tile_m, self.tile_n),
            tdPrdO,
            tdPrV,
            swap_AB=self.SdP_swapAB,
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn = partial(
                gemm_w_idx_sm90,
                tiled_mma_dV,
                acc_dV,
                tdVrPt,
                tdVrdOt,
                swap_AB=self.dKV_swapAB,
            )
            mma_dsq_fn = partial(
                gemm_w_idx_sm90,
                tiled_mma_dK,
                acc_dK,
                tdKrdSt,
                tdKrQt,
                swap_AB=self.dKV_swapAB,
            )
        else:
            assert not self.dKV_swapAB
            mma_pdo_fn = partial(gemm_w_idx_sm90, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
            mma_dsq_fn = partial(gemm_w_idx_sm90, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        mma_dsk_fn = partial(
            gemm_zero_init,
            tiled_mma_dQ,
            (self.tile_m, self.tile_hdim),
            tdQrdS,
            tdQrKt,
            swap_AB=self.dQ_swapAB,
        )

        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tPsP=tPsP,
            tdSsdS=tdSsdS,
            tdQsdQaccum=tdQsdQaccum,
            smem_thr_copy_PdS=smem_thr_copy_PdS,
            smem_thr_copy_dQaccum=smem_thr_copy_dQaccum,
            softmax_scale_log2=softmax_scale_log2,
            softmax_scale=softmax_scale,
            mAttnScale=mAttnScale,
            thr_mma_SdP=thr_mma_SdP,
        )

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=None,
                head_idx=None,
                n_block=n_block,
                thr_mma=thr_mma_SdP,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            primary_ran = False
            if const_expr(not self.is_local) or m_block_min < m_block_max:
                dKV_accumulate = False
                # pyre-ignore[28]
                for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                    consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                        m_block,
                        consumer_state_Q,
                        consumer_state_dO,
                        mask_fn=mask_fn,
                        dKV_accumulate=dKV_accumulate,
                        n_block=n_block,
                        seqlen_q=seqlen.seqlen_q,
                        seqlen_k=seqlen.seqlen_k,
                    )
                    dKV_accumulate = True
                    primary_ran = True

            if const_expr(mQ_alt is not None):
                # pyre-ignore[29]
                seqlen_alt = SeqlenInfoCls_alt(batch_idx)
                mask_alt = AttentionMaskCls(seqlen_alt.seqlen_q, seqlen.seqlen_k)
                mask_fn_alt = partial(
                    mask_alt.apply_mask,
                    batch_idx=None,
                    head_idx=None,
                    n_block=n_block,
                    thr_mma=thr_mma_SdP,
                    mask_seqlen=True,
                    mask_causal=False,
                    mask_local=False,
                )
                mma_one_m_block_alt = partial(
                    self.mma_one_m_block,
                    warp_group_idx=warp_group_idx,
                    mma_qk_fn=mma_qk_fn,
                    mma_dov_fn=mma_dov_fn,
                    mma_pdo_fn=mma_pdo_fn,
                    mma_dsq_fn=mma_dsq_fn,
                    mma_dsk_fn=mma_dsk_fn,
                    pipeline_Q=pipeline_Q,
                    pipeline_dO=pipeline_dO,
                    tLSEsLSE=tLSEsLSE,
                    tLSEsdPsum=tLSEsdPsum,
                    tPsP=tPsP,
                    tdSsdS=tdSsdS,
                    tdQsdQaccum=tdQsdQaccum,
                    smem_thr_copy_PdS=smem_thr_copy_PdS,
                    smem_thr_copy_dQaccum=smem_thr_copy_dQaccum,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_scale=softmax_scale,
                    mAttnScale=mAttnScale_alt,
                    thr_mma_SdP=thr_mma_SdP,
                    is_causal_override=False,
                    is_local_override=False,
                )
                # pyre-ignore[16]
                m_block_alt_min, m_block_alt_max = block_info_alt.get_m_block_min_max(
                    seqlen_alt, n_block
                )
                # pyre-ignore[28]
                for m_block in cutlass.range(
                    m_block_alt_min, m_block_alt_max, unroll=1
                ):
                    consumer_state_Q, consumer_state_dO = mma_one_m_block_alt(
                        m_block,
                        consumer_state_Q,
                        consumer_state_dO,
                        mask_fn=mask_fn_alt,
                        dKV_accumulate=primary_ran,
                        n_block=n_block,
                        seqlen_q=seqlen_alt.seqlen_q,
                        seqlen_k=seqlen.seqlen_k,
                    )
                    primary_ran = True

            if primary_ran:
                acc_dK.store(acc_dK.load() * softmax_scale)
            if const_expr(self.is_local):
                if not primary_ran:
                    acc_dK.fill(0.0)
                    acc_dV.fill(0.0)
            self.epilogue_dKV(
                acc_dV,
                mdV,
                sV,
                acc_dK,
                mdK,
                sK,
                seqlen,
                tma_atom_dK,
                tma_atom_dV,
                tiled_mma_dK,
                tiled_mma_dV,
                tidx,
                n_block,
                head_idx,
                batch_idx,
                gmem_tiled_copy_dKV,
                mdK_raw,
                mdV_raw,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: cutlass.pipeline.PipelineState
        | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tPsP: Optional[cute.Tensor],
        tdSsdS: Optional[cute.Tensor],
        tdQsdQaccum: cute.Tensor,
        smem_thr_copy_PdS: cute.TiledCopy,
        smem_thr_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 = Float32(1.0),
        mask_fn: Optional[Callable] = None,
        # pyre-ignore[9]
        dKV_accumulate: Boolean = True,
        mAttnScale: Optional[cute.Tensor] = None,
        thr_mma_SdP: Optional[cute.TiledMma] = None,
        n_block: Int32 = Int32(0),
        seqlen_q: Int32 = Int32(0),
        seqlen_k: Int32 = Int32(0),
        is_causal_override: Optional[bool] = None,
        is_local_override: Optional[bool] = None,
    ):
        is_causal_eff = (
            self.is_causal if is_causal_override is None else is_causal_override
        )
        is_local_eff = self.is_local if is_local_override is None else is_local_override
        consumer_state_dO_cur = (
            consumer_state_dO
            if const_expr(self.Q_stage == self.dO_stage)
            else consumer_state_Q
        )
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = (
            consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
        )
        smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0
        # S = Q @ K^T
        pipeline_Q.consumer_wait(
            consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q)
        )
        acc_S = mma_qk_fn(
            A_idx=smem_idx_Q,
            wg_wait=0 if const_expr(self.reorder_sdp) else -1,
        )
        if const_expr(not self.use_silu):
            tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])
        # dP = dO @ V.T
        if const_expr(not self.reorder_sdp):
            pipeline_dO.consumer_wait(
                consumer_state_dO_cur,
                pipeline_dO.consumer_try_wait(consumer_state_dO_cur),
            )
            acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)
        # P = activation(S)
        if const_expr(not self.use_silu):
            if cutlass.const_expr(mask_fn is not None):
                # pyre-ignore[29]
                mask_fn(acc_S, m_block=m_block)
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S, transpose=self.SdP_swapAB)
        if const_expr(not self.use_silu):
            # P = exp2(S * scale_log2 - LSE)
            for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
                for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                    acc_S_mn[r, c] = cute.math.exp2(
                        # pyre-ignore[61]
                        acc_S_mn[r, c] * softmax_scale_log2 - tLSErLSE[r],
                        fastmath=True,
                    )
            tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])
        else:
            # SiLU: P = silu(S * scale) = S*scale * sigmoid(S*scale)
            # silu'(x) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x)) = P*(1-sig) + sig
            acc_silu_deriv = cute.make_fragment(acc_S.shape, Float32)
            acc_silu_deriv_mn = utils.make_acc_tensor_mn_view(
                acc_silu_deriv, transpose=self.SdP_swapAB
            )
            # Build coordinate tensors for mAttnScale and zero masking
            acc_shape_silu = (self.tile_m, self.tile_n)
            cS_silu = cute.make_identity_tensor(
                acc_shape_silu if not self.SdP_swapAB else acc_shape_silu[::-1]
            )
            tScS_mn_silu = utils.make_acc_tensor_mn_view(
                # pyre-ignore[16]
                thr_mma_SdP.partition_C(cS_silu),
                transpose=self.SdP_swapAB,
            )
            t0ScS_mn_silu = utils.make_acc_tensor_mn_view(
                # pyre-ignore[16]
                thr_mma_SdP.get_slice(0).partition_C(cS_silu),
                transpose=self.SdP_swapAB,
            )
            ROW_SILU = 0 if const_expr(not self.SdP_swapAB) else 1
            COL_SILU = 1 if const_expr(not self.SdP_swapAB) else 0
            thr_col_offset_silu = tScS_mn_silu[0][COL_SILU]
            seqlenk_col_limit_silu = (
                seqlen_k - n_block * self.tile_n - thr_col_offset_silu
            )
            seqlenq_row_limit_silu = seqlen_q - m_block * self.tile_m
            tile_fully_valid = (seqlen_k - n_block * self.tile_n >= self.tile_n) & (
                seqlenq_row_limit_silu >= self.tile_m
            )
            if const_expr(is_causal_eff or is_local_eff):
                causal_delta_tv = (
                    Int32(0) if const_expr(self.is_diagonal) else seqlen_k - seqlen_q
                )
                tile_fully_valid = tile_fully_valid & (
                    m_block * self.tile_m + causal_delta_tv
                    >= (n_block + 1) * self.tile_n - 1
                )
                if const_expr(is_local_eff):
                    tile_fully_valid = tile_fully_valid & (
                        (m_block + 1) * self.tile_m
                        - 1
                        + causal_delta_tv
                        - n_block * self.tile_n
                        # pyre-ignore[16]
                        <= self.window_size_left
                    )
            half_scale = softmax_scale * Float32(0.5)
            for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
                if const_expr(mAttnScale is not None):
                    m_idx = tScS_mn_silu[r, 0][ROW_SILU] + m_block * self.tile_m
                    # pyre-ignore[16]
                    row_scale = Float32(mAttnScale[m_idx])
                for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                    s_val = acc_S_mn[r, c]
                    half_x = s_val * half_scale
                    tanh_val_ir = llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [half_x.ir_value()],
                        "tanh.approx.f32 $0, $1;",
                        "=f,f",
                        has_side_effects=False,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                    tanh_val = Float32(tanh_val_ir)
                    sig = tanh_val * Float32(0.5) + Float32(0.5)
                    qk = s_val * softmax_scale
                    p = qk * sig
                    oms = Float32(0.5) - tanh_val * Float32(0.5)
                    deriv = p * oms + sig
                    if const_expr(mAttnScale is not None):
                        # pyre-ignore[61]
                        acc_S_mn[r, c] = p * row_scale
                        # pyre-ignore[61]
                        acc_silu_deriv_mn[r, c] = deriv * row_scale
                    else:
                        acc_S_mn[r, c] = p
                        acc_silu_deriv_mn[r, c] = deriv
            # Zero out-of-bounds positions in both P and silu derivative
            if not tile_fully_valid:
                for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                    col_idx = t0ScS_mn_silu[0, c][COL_SILU]
                    col_oob = col_idx >= seqlenk_col_limit_silu
                    for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
                        row_idx = tScS_mn_silu[r, 0][ROW_SILU]
                        out_of_bounds = col_oob | (row_idx >= seqlenq_row_limit_silu)
                        if const_expr(is_causal_eff or is_local_eff):
                            global_q = m_block * self.tile_m + row_idx
                            global_k = (
                                n_block * self.tile_n + thr_col_offset_silu + col_idx
                            )
                            causal_delta = (
                                Int32(0)
                                if const_expr(self.is_diagonal)
                                else seqlen_k - seqlen_q
                            )
                            if global_q + causal_delta < global_k:
                                out_of_bounds = True
                            if const_expr(is_local_eff):
                                if (
                                    global_q + causal_delta - global_k
                                    > self.window_size_left
                                ):
                                    out_of_bounds = True
                        if out_of_bounds:
                            acc_S_mn[r, c] = Float32(0.0)
                            acc_silu_deriv_mn[r, c] = Float32(0.0)

        # Convert P from f32 -> f16
        tdVrP = utils.cvt_f16(utils.make_acc_tensor_frgA_view(acc_S), self.dtype)
        # R2S for P
        if const_expr(not self.mma_dkv_is_rs):
            # sync to ensure P has already been used in the previous iteration before overwriting
            if const_expr(self.PdS_stage == 1):
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.PdS),
                    # pyre-ignore[16]
                    number_of_threads=self.num_mma_threads,
                )
            tPrP = smem_thr_copy_PdS.retile(tdVrP)
            cute.copy(smem_thr_copy_PdS, tPrP, tPsP[None, None, None, smem_idx_PdS])

        if const_expr(self.reorder_sdp):
            pipeline_dO.consumer_wait(
                consumer_state_dO_cur,
                pipeline_dO.consumer_try_wait(consumer_state_dO_cur),
            )
            acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=0)

        # dS computation
        warpgroup.wait_group(0)
        # pyre-ignore[61]
        acc_dP_mn = utils.make_acc_tensor_mn_view(acc_dP, transpose=self.SdP_swapAB)
        if const_expr(not self.use_silu):
            # Softmax: dS = P * (dP - dPsum)
            for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
                for c in cutlass.range(
                    cute.size(acc_dP_mn, mode=[1]), unroll_full=True
                ):
                    # pyre-ignore[61]
                    acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - tLSErdPsum[r])
        else:
            # SiLU: dS = dP * silu'(S*scale)
            for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
                for c in cutlass.range(
                    cute.size(acc_dP_mn, mode=[1]), unroll_full=True
                ):
                    # pyre-ignore[61]
                    acc_dP_mn[r, c] = acc_dP_mn[r, c] * acc_silu_deriv_mn[r, c]
        # Convert dS from f32 -> f16
        # pyre-ignore[61]
        tdKrdS = utils.cvt_f16(utils.make_acc_tensor_frgA_view(acc_dP), self.dtype)

        # This sync is to ensure P is written in case of !mma_dkv_is_rs and
        # dS is already read by the Mma in the previous iteration in case of mma_dkv_is_rs.
        if const_expr(
            not self.mma_dkv_is_rs or (self.PdS_stage == 1 and self.mma_dkv_is_rs)
        ):
            cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.PdS),
                number_of_threads=self.num_mma_threads,
            )

        # R2S for dS
        tdSrdS = smem_thr_copy_PdS.retile(tdKrdS)
        cute.copy(smem_thr_copy_PdS, tdSrdS, tdSsdS[None, None, None, smem_idx_PdS])

        # dV += P.T @ dO
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn(
                A_idx=smem_idx_PdS,
                B_idx=smem_idx_dO,
                zero_init=not dKV_accumulate,
                wg_wait=-1,
            )
        else:
            mma_pdo_fn(
                tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1
            )

        # smem fence to make sure sdS is written before it's read by WGMMA
        cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads
        )
        # dQ = dS @ K
        acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
        pipeline_dO.consumer_release(
            consumer_state_dO_cur
        )  # release dO as dV mma is done

        # dK += dS.T @ Q
        if const_expr(not self.mma_dkv_is_rs):
            mma_dsq_fn(
                A_idx=smem_idx_PdS,
                B_idx=smem_idx_Q,
                zero_init=not dKV_accumulate,
                wg_wait=1,
            )
        else:
            mma_dsq_fn(
                tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
            )

        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
            # pyre-ignore[16]
            number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
        acc_dQ_bf = cute.make_fragment_like(acc_dQ, self.dtype)
        acc_dQ_bf.store((acc_dQ.load() * softmax_scale).to(self.dtype))
        taccdQrdQ = smem_thr_copy_dQaccum.retile(acc_dQ_bf)
        cute.copy(smem_thr_copy_dQaccum, taccdQrdQ, tdQsdQaccum)
        cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
            number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )

        warpgroup.wait_group(0)
        pipeline_Q.consumer_release(consumer_state_Q)

        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return consumer_state_Q, consumer_state_dO

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        gmem_tiled_copy_dKV: Optional[cute.TiledCopy] = None,
        mdK_raw: Optional[cute.Tensor] = None,
        mdV_raw: Optional[cute.Tensor] = None,
    ):
        rdV = cute.make_fragment_like(acc_dV, self.dtype)
        rdV.store(acc_dV.load().to(self.dtype))
        rdK = utils.cvt_f16(acc_dK, self.dtype)

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            # pyre-ignore[16]
            number_of_threads=self.num_mma_threads,
        )

        smem_copy_atom_dKV = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                transpose=self.dKV_swapAB, num_matrices=4
            ),
            self.dtype,
        )
        smem_thr_copy_dK = cute.make_tiled_copy_C(
            smem_copy_atom_dKV, tiled_mma_dK
        ).get_slice(tidx)
        smem_thr_copy_dV = cute.make_tiled_copy_C(
            smem_copy_atom_dKV, tiled_mma_dV
        ).get_slice(tidx)
        # pyre-ignore[16]
        if const_expr(not self.is_varlen):
            mdV_cur = mdV[None, None, head_idx, batch_idx]
            mdK_cur = mdK[None, None, head_idx, batch_idx]
        else:
            mdV_cur = cute.domain_offset(
                (seqlen.offset_k, 0), mdV[None, None, head_idx]
            )
            mdK_cur = cute.domain_offset(
                (seqlen.offset_k, 0), mdK[None, None, head_idx]
            )
        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
        k_residue = seqlen.seqlen_k - n_block * self.tile_n
        row_start = seqlen.offset_k + n_block * self.tile_n

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # rmem -> smem (stmatrix) for dV
        taccdVrdV = smem_thr_copy_dV.retile(rdV)
        sdV = sV if const_expr(not self.dKV_swapAB) else utils.transpose_view(sV)
        taccdVsdV = smem_thr_copy_dV.partition_D(sdV)
        cute.copy(smem_copy_atom_dKV, taccdVrdV, taccdVsdV)
        cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_mma_threads,
        )
        if const_expr(self.accumulate_dKV):
            # pyre-ignore[16]
            gdV_ptr_acc = mdV_raw.iterator + (
                # pyre-ignore[16]
                row_start * mdV_raw.stride[0] + head_idx * mdV_raw.stride[1]
            )
            gdV_plain_acc = cute.make_tensor(
                gdV_ptr_acc,
                cute.make_layout(
                    (self.tile_n, self.tile_hdimv),
                    stride=(mdV_raw.stride[0], mdV_raw.stride[2]),
                ),
            )
            if warp_idx == 4:
                thr_idx = tidx % 32
                # pyre-ignore[16]
                gmem_thr_dV = gmem_tiled_copy_dKV.get_slice(thr_idx)
                tdVsV_src = gmem_thr_dV.partition_S(sV)
                tdVgdV = gmem_thr_dV.partition_D(gdV_plain_acc)
                cdV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
                tdVcV = gmem_thr_dV.partition_S(cdV)
                t0dVcV = gmem_tiled_copy_dKV.get_slice(0).partition_S(cdV)
                tdVrV_new = cute.make_fragment_like(tdVsV_src, self.dtype)
                cute.autovec_copy(tdVsV_src, tdVrV_new)
                tdVrV_old = cute.make_fragment_like(tdVsV_src, self.dtype)
                tdVrV_old.fill(0.0)
                for rest_n in cutlass.range(
                    cute.size(tdVrV_old.shape[1]), unroll_full=True
                ):
                    if t0dVcV[0, rest_n, 0][0] < k_residue - tdVcV[0][0]:
                        cute.copy(
                            gmem_tiled_copy_dKV,
                            tdVgdV[None, rest_n, None],
                            tdVrV_old[None, rest_n, None],
                        )
                tdVrV_sum = cute.make_fragment_like(tdVrV_new, self.dtype)
                tdVrV_sum.store(
                    (tdVrV_new.load().to(Float32) + tdVrV_old.load().to(Float32)).to(
                        self.dtype
                    )
                )
                for rest_n in cutlass.range(
                    cute.size(tdVrV_sum.shape[1]), unroll_full=True
                ):
                    if t0dVcV[0, rest_n, 0][0] < k_residue - tdVcV[0][0]:
                        cute.copy(
                            gmem_tiled_copy_dKV,
                            tdVrV_sum[None, rest_n, None],
                            tdVgdV[None, rest_n, None],
                        )
        else:
            if const_expr(not self.is_varlen) or k_residue >= self.tile_n:
                # pyre-ignore[23]
                store_dV, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dV, 0, cute.make_layout(1), sV, gdV, single_stage=True
                )
                if warp_idx == 4:
                    store_dV()
            else:
                # pyre-ignore[16]
                gdV_ptr = mdV_raw.iterator + (
                    # pyre-ignore[16]
                    row_start * mdV_raw.stride[0] + head_idx * mdV_raw.stride[1]
                )
                gdV_plain = cute.make_tensor(
                    gdV_ptr,
                    cute.make_layout(
                        (self.tile_n, self.tile_hdimv),
                        stride=(mdV_raw.stride[0], mdV_raw.stride[2]),
                    ),
                )
                if warp_idx == 4:
                    thr_idx = tidx % 32
                    # pyre-ignore[16]
                    gmem_thr_dV = gmem_tiled_copy_dKV.get_slice(thr_idx)
                    tdVsV_src = gmem_thr_dV.partition_S(sV)
                    tdVgdV_dst = gmem_thr_dV.partition_D(gdV_plain)
                    cdV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
                    tdVcV = gmem_thr_dV.partition_S(cdV)
                    t0dVcV = gmem_tiled_copy_dKV.get_slice(0).partition_S(cdV)
                    tdVrV = cute.make_fragment_like(tdVsV_src, self.dtype)
                    cute.autovec_copy(tdVsV_src, tdVrV)
                    for rest_n in cutlass.range(
                        cute.size(tdVrV.shape[1]), unroll_full=True
                    ):
                        if t0dVcV[0, rest_n, 0][0] < k_residue - tdVcV[0][0]:
                            cute.copy(
                                gmem_tiled_copy_dKV,
                                tdVrV[None, rest_n, None],
                                tdVgdV_dst[None, rest_n, None],
                            )

        # rmem -> smem (stmatrix) for dK
        taccdKrdK = smem_thr_copy_dK.retile(rdK)
        sdK = sK if const_expr(not self.dKV_swapAB) else utils.transpose_view(sK)
        taccdKsdK = smem_thr_copy_dK.partition_D(sdK)
        cute.copy(smem_copy_atom_dKV, taccdKrdK, taccdKsdK)
        cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_mma_threads,
        )
        if const_expr(self.accumulate_dKV):
            gdK_ptr_acc = mdK_raw.iterator + (
                row_start * mdK_raw.stride[0] + head_idx * mdK_raw.stride[1]
            )
            gdK_plain_acc = cute.make_tensor(
                gdK_ptr_acc,
                cute.make_layout(
                    (self.tile_n, self.tile_hdim),
                    stride=(mdK_raw.stride[0], mdK_raw.stride[2]),
                ),
            )
            if warp_idx == 4:
                thr_idx = tidx % 32
                gmem_thr_dK = gmem_tiled_copy_dKV.get_slice(thr_idx)
                tdKsK_src = gmem_thr_dK.partition_S(sK)
                tdKgdK = gmem_thr_dK.partition_D(gdK_plain_acc)
                cdK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
                tdKcK = gmem_thr_dK.partition_S(cdK)
                t0dKcK = gmem_tiled_copy_dKV.get_slice(0).partition_S(cdK)
                tdKrK_new = cute.make_fragment_like(tdKsK_src, self.dtype)
                cute.autovec_copy(tdKsK_src, tdKrK_new)
                tdKrK_old = cute.make_fragment_like(tdKsK_src, self.dtype)
                tdKrK_old.fill(0.0)
                for rest_n in cutlass.range(
                    cute.size(tdKrK_old.shape[1]), unroll_full=True
                ):
                    if t0dKcK[0, rest_n, 0][0] < k_residue - tdKcK[0][0]:
                        cute.copy(
                            gmem_tiled_copy_dKV,
                            tdKgdK[None, rest_n, None],
                            tdKrK_old[None, rest_n, None],
                        )
                tdKrK_sum = cute.make_fragment_like(tdKrK_new, self.dtype)
                tdKrK_sum.store(
                    (tdKrK_new.load().to(Float32) + tdKrK_old.load().to(Float32)).to(
                        self.dtype
                    )
                )
                for rest_n in cutlass.range(
                    cute.size(tdKrK_sum.shape[1]), unroll_full=True
                ):
                    if t0dKcK[0, rest_n, 0][0] < k_residue - tdKcK[0][0]:
                        cute.copy(
                            gmem_tiled_copy_dKV,
                            tdKrK_sum[None, rest_n, None],
                            tdKgdK[None, rest_n, None],
                        )
        else:
            if const_expr(not self.is_varlen) or k_residue >= self.tile_n:
                # pyre-ignore[23]
                store_dK, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dK, 0, cute.make_layout(1), sK, gdK, single_stage=True
                )
                if warp_idx == 4:
                    store_dK()
                    cute.arch.cp_async_bulk_commit_group()
                    if const_expr(self.is_persistent):
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
            else:
                gdK_ptr = mdK_raw.iterator + (
                    row_start * mdK_raw.stride[0] + head_idx * mdK_raw.stride[1]
                )
                gdK_plain = cute.make_tensor(
                    gdK_ptr,
                    cute.make_layout(
                        (self.tile_n, self.tile_hdim),
                        stride=(mdK_raw.stride[0], mdK_raw.stride[2]),
                    ),
                )
                if warp_idx == 4:
                    thr_idx = tidx % 32
                    gmem_thr_dK = gmem_tiled_copy_dKV.get_slice(thr_idx)
                    tdKsK_src = gmem_thr_dK.partition_S(sK)
                    tdKgdK_dst = gmem_thr_dK.partition_D(gdK_plain)
                    cdK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
                    tdKcK = gmem_thr_dK.partition_S(cdK)
                    t0dKcK = gmem_tiled_copy_dKV.get_slice(0).partition_S(cdK)
                    tdKrK = cute.make_fragment_like(tdKsK_src, self.dtype)
                    cute.autovec_copy(tdKsK_src, tdKrK)
                    for rest_n in cutlass.range(
                        cute.size(tdKrK.shape[1]), unroll_full=True
                    ):
                        if t0dKcK[0, rest_n, 0][0] < k_residue - tdKcK[0][0]:
                            cute.copy(
                                gmem_tiled_copy_dKV,
                                tdKrK[None, rest_n, None],
                                tdKgdK_dst[None, rest_n, None],
                            )

    @cute.jit
    def dQaccum_store(
        self,
        mdQ: cute.Tensor,
        tma_atom_dQ: cute.CopyAtom,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
        mdQ_alt: Optional[cute.Tensor] = None,
        tma_atom_dQ_alt: Optional[cute.CopyAtom] = None,
        SeqlenInfoCls_alt: Optional[Callable] = None,
        block_info_alt: Optional[BlockInfo] = None,
    ):
        # pyre-ignore[29]
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            # pyre-ignore[29]
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdQ_cur = mdQ[None, None, head_idx, batch_idx]
            else:
                mdQ_cur = cute.domain_offset(
                    (seqlen.offset_q, 0),
                    mdQ[None, None, head_idx],
                )
            gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
            tdQsdQ, tdQgdQ = cpasync.tma_partition(
                tma_atom_dQ,
                0,
                cute.make_layout(1),
                cute.group_modes(sdQaccum, 0, 2),
                cute.group_modes(gdQ, 0, 2),
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            # pyre-ignore[28]
            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                for warp_group_idx in cutlass.range_constexpr(self.num_mma_warp_groups):
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                        # pyre-ignore[16]
                        number_of_threads=self.num_threads_per_warp_group
                        + cute.arch.WARP_SIZE,
                    )
                cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ[None, m_block])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                for warp_group_idx in cutlass.range_constexpr(self.num_mma_warp_groups):
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                        number_of_threads=self.num_threads_per_warp_group
                        + cute.arch.WARP_SIZE,
                    )

            if const_expr(mdQ_alt is not None):
                # pyre-ignore[29]
                seqlen_alt = SeqlenInfoCls_alt(batch_idx)
                if const_expr(not seqlen_alt.has_cu_seqlens_q):
                    # pyre-ignore[16]
                    mdQ_alt_cur = mdQ_alt[None, None, head_idx, batch_idx]
                else:
                    mdQ_alt_cur = cute.domain_offset(
                        (seqlen_alt.offset_q, 0),
                        mdQ_alt[None, None, head_idx],
                    )
                gdQ_alt = cute.local_tile(
                    mdQ_alt_cur, (self.tile_m, self.tile_hdim), (None, 0)
                )
                tdQsdQ_alt, tdQgdQ_alt = cpasync.tma_partition(
                    tma_atom_dQ_alt,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sdQaccum, 0, 2),
                    cute.group_modes(gdQ_alt, 0, 2),
                )
                # pyre-ignore[16]
                m_block_alt_min, m_block_alt_max = block_info_alt.get_m_block_min_max(
                    seqlen_alt, n_block
                )
                # pyre-ignore[28]
                for m_block in cutlass.range(
                    m_block_alt_min, m_block_alt_max, unroll=1
                ):
                    for warp_group_idx in cutlass.range_constexpr(
                        self.num_mma_warp_groups
                    ):
                        cute.arch.barrier(
                            barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                            number_of_threads=self.num_threads_per_warp_group
                            + cute.arch.WARP_SIZE,
                        )
                    cute.copy(tma_atom_dQ_alt, tdQsdQ_alt, tdQgdQ_alt[None, m_block])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    for warp_group_idx in cutlass.range_constexpr(
                        self.num_mma_warp_groups
                    ):
                        cute.arch.barrier_arrive(
                            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                            number_of_threads=self.num_threads_per_warp_group
                            + cute.arch.WARP_SIZE,
                        )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
