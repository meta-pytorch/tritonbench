# pyre-ignore-all-errors
# Copyright (c) 2025, Tri Dao.

import math
import operator
from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
import hammer.v3.ops.cutedsl.fa4_helpers.utils as utils
from cutlass import Float32, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T
from hammer.v3.ops.cutedsl.fa4_helpers.cute_dsl_utils import ParamsBase
from hammer.v3.ops.cutedsl.fa4_helpers.seqlen_info import SeqlenInfoQK

# MXFP8 block scaling constants
E4M3_MAX_NORM_RCP = 1.0 / 448.0  # FP8 E4M3 max value reciprocal
E2M1_MAX_NORM_RCP = 1.0 / 6.0  # FP4 E2M1 max value reciprocal


@dsl_user_op
def optimization_barrier(val: Float32, *, loc=None, ip=None) -> Float32:
    """Identity function that acts as a compiler optimization barrier.
    The @dsl_user_op decorator forces a function call boundary which
    prevents the JIT from reordering instructions across it."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "mov.f32 $0, $1;",
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def max_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute max of two F32 values using PTX."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def abs_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    """Compute absolute value of a float32 using PTX abs.f32 instruction."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fused_abs_max_f32(
    running_max: Float32, val: Float32, *, loc=None, ip=None
) -> Float32:
    """Compute max(running_max, abs(val)) in a single PTX asm block.
    has_side_effects=True prevents the JIT from reordering or CSE'ing
    this operation, which is critical for correct AMAX computation
    when register aliasing through logical_divide views can cause
    the JIT to miscompile separate abs_f32/max_f32 calls."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(running_max).ir_value(loc=loc, ip=ip),
                Float32(val).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .f32 _abs_val;\n"
            "abs.f32 _abs_val, $2;\n"
            "max.f32 $0, $1, _abs_val;\n"
            "}\n",
            "=f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def shfl_xor_b32(
    val: cutlass.Uint32, lane_mask: int, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Warp shuffle XOR for uint32 values (butterfly exchange)."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Uint32(val).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {lane_mask}, 0x1f, 0xffffffff;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def prmt_b32(
    a: cutlass.Uint32, b: cutlass.Uint32, selector: int, *, loc=None, ip=None
) -> cutlass.Uint32:
    """PTX prmt (permute bytes) instruction. Selects 4 bytes from a and b.
    selector is a 16-bit immediate: each 4-bit nibble selects a byte source.
    Nibble values 0-3 select bytes from a, 4-7 select bytes from b."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(a).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(b).ir_value(loc=loc, ip=ip),
            ],
            f"prmt.b32 $0, $1, $2, {hex(selector)};",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def redux_sync_max_abs_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    """Warp-level reduction: compute max(abs(val)) across all 32 threads in warp.

    Uses Blackwell SM100 redux.sync.max.abs.f32 PTX instruction.
    Each thread gets back the result (all-reduce, not just lane 0).

    Args:
        val: Float32 value from each thread

    Returns:
        max(abs(val)) across all 32 threads in the warp
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "redux.sync.max.abs.f32 $0, $1, 0xffffffff;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def subwarp8_max_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    """8-lane sub-warp max reduction using butterfly shuffles.

    Computes max(val) across groups of 8 consecutive lanes:
      lanes 0-7, lanes 8-15, lanes 16-23, lanes 24-31.
    All lanes in a group get the same result.

    Input val should already be non-negative (e.g., result of abs/max_abs).
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "{\n"
            "    .reg .f32 sw_val, sw_tmp_f;\n"
            "    .reg .b32 sw_tmp_b, sw_shfl;\n"
            "    mov.f32 sw_val, $1;\n"
            "    mov.b32 sw_tmp_b, sw_val;\n"
            "    shfl.sync.bfly.b32 sw_shfl, sw_tmp_b, 1, 31, 0xffffffff;\n"
            "    mov.b32 sw_tmp_f, sw_shfl;\n"
            "    max.f32 sw_val, sw_val, sw_tmp_f;\n"
            "    mov.b32 sw_tmp_b, sw_val;\n"
            "    shfl.sync.bfly.b32 sw_shfl, sw_tmp_b, 2, 31, 0xffffffff;\n"
            "    mov.b32 sw_tmp_f, sw_shfl;\n"
            "    max.f32 sw_val, sw_val, sw_tmp_f;\n"
            "    mov.b32 sw_tmp_b, sw_val;\n"
            "    shfl.sync.bfly.b32 sw_shfl, sw_tmp_b, 4, 31, 0xffffffff;\n"
            "    mov.b32 sw_tmp_f, sw_shfl;\n"
            "    max.f32 sw_val, sw_val, sw_tmp_f;\n"
            "    mov.f32 $0, sw_val;\n"
            "}\n",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fused_amax_to_e8m0_scale_f32(
    amax: Float32,
    max_norm_rcp: Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Uint8, Float32]:
    """
    Convert F32 AMAX to E8M0 scale and inverse scale.

    This matches the C++ fused_amax_to_e8m0_rceil function:
    1. scale_f32 = amax * max_norm_rcp
    2. Extract exponent, round up (RCEIL) if mantissa != 0
    3. Compute inverse scale = 2^(127 - e8m0_exp)

    Args:
        amax: Maximum absolute value of the block (F32)
        max_norm_rcp: Reciprocal of max normal value (1/448 for E4M3)

    Returns:
        Tuple of (e8m0_scale, inv_scale):
        - e8m0_scale: E8M0 biased exponent (uint8)
        - inv_scale: Inverse scale factor = 2^(127 - e8m0_exp) (float32)
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [
            Float32(amax).ir_value(loc=loc, ip=ip),
            Float32(max_norm_rcp).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .f32 fae_scale;\n"
        ".reg .u32 fae_bits, fae_exp, fae_mantissa;\n"
        ".reg .pred fae_has_mantissa;\n"
        "mul.f32 fae_scale, $2, $3;\n"
        "mov.b32 fae_bits, fae_scale;\n"
        "bfe.u32 fae_exp, fae_bits, 23, 8;\n"
        "and.b32 fae_mantissa, fae_bits, 0x7FFFFF;\n"
        "setp.ne.u32 fae_has_mantissa, fae_mantissa, 0;\n"
        "@fae_has_mantissa add.u32 fae_exp, fae_exp, 1;\n"
        "mov.b32 $0, fae_exp;\n"
        ".reg .u32 fae_inv_exp, fae_inv_bits;\n"
        "sub.u32 fae_inv_exp, 254, fae_exp;\n"
        "shl.b32 fae_inv_bits, fae_inv_exp, 23;\n"
        "mov.b32 $1, fae_inv_bits;\n"
        "}\n",
        "=r,=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    e8m0_u32 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    inv_scale_val = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    e8m0_scale = Uint8(
        llvm.inline_asm(
            T.i8(),
            [e8m0_u32],
            "cvt.u8.u32 $0, $1;\n",
            "=c,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

    # Also return the raw u32 exponent (before Uint8 truncation)
    # for callers that need to avoid Uint8 register clobber
    e8m0_u32_val = cutlass.Uint32(e8m0_u32)

    return e8m0_scale, Float32(inv_scale_val), e8m0_u32_val


@dsl_user_op
def pack_4xu8_to_u32(
    a: Uint8, b: Uint8, c: Uint8, d: Uint8, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Pack 4 uint8 values into a uint32 (little-endian: a is lowest byte)."""
    result = llvm.inline_asm(
        T.i32(),
        [
            Uint8(a).ir_value(loc=loc, ip=ip),
            Uint8(b).ir_value(loc=loc, ip=ip),
            Uint8(c).ir_value(loc=loc, ip=ip),
            Uint8(d).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 t0, t1, t2, t3;\n"
        "cvt.u32.u8 t0, $1;\n"
        "cvt.u32.u8 t1, $2;\n"
        "cvt.u32.u8 t2, $3;\n"
        "cvt.u32.u8 t3, $4;\n"
        "shl.b32 t1, t1, 8;\n"
        "shl.b32 t2, t2, 16;\n"
        "shl.b32 t3, t3, 24;\n"
        "or.b32 $0, t0, t1;\n"
        "or.b32 $0, $0, t2;\n"
        "or.b32 $0, $0, t3;\n"
        "}\n",
        "=r,c,c,c,c",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint32(result)


@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    def _compute_row_max(
        self, acc_S_row: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fmax_reduce(acc_S_row, init_val, arch=self.arch)

    def _compute_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Apply online softmax and return the row_scale to rescale O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first: is first n_block
        :type is_first: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch

        # Each iteration processes one row of acc_S
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)

            row_max_cur = utils.fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )

            row_max_cur = utils.warp_reduce(row_max_cur, cute.arch.fmax, width=4)
            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)

                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=None, arch=arch
                )
                row_scale[r] = 1.0
            else:
                row_max_prev = row_max[r]
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)
                # row_scale[r] = utils.exp2f(row_max_prev * self.scale_log2 - row_max_cur_scaled)
                row_scale[r] = utils.exp2f((row_max_prev - row_max_cur) * scale_log2)

                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
                )

            row_max[r] = row_max_cur
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(
        self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None
    ) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp."""
        if cutlass.const_expr(
            sink_val is not None and isinstance(sink_val, cute.Tensor)
        ):
            assert cute.size(sink_val) == cute.size(self.row_sum)
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2

        # quad reduction for row_sum as we didn't do it during each iteration of online softmax
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            if cutlass.const_expr(sink_val is not None):
                sink_val_cur = (
                    sink_val if not isinstance(sink_val, cute.Tensor) else sink_val[r]
                )
                LOG2_E = math.log2(math.e)
                row_sum[r] += utils.exp2f(
                    sink_val_cur * LOG2_E - row_max[r] * scale_log2
                )

            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(
                    row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0
                )
            ) * final_scale
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = (
                (row_max[r] * scale_log2 + utils.log2f(row_sum_cur)) * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        """Scale each row of acc_O by the given scale tensor.
        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_scale: row_scale tensor
        :type row_scale: cute.Tensor
        """
        acc_O_mn = utils.make_acc_tensor_mn_view(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


@dataclass
class SoftmaxSm100(Softmax):
    rescale_threshold: cutlass.Constexpr[float] = 0.0

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
    ):
        num_rows = 1
        arch = 100
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return SoftmaxSm100(
            scale_log2,
            num_rows,
            row_max,
            row_sum,
            arch,
            softmax_scale,
            rescale_threshold=rescale_threshold,
        )

    @cute.jit
    def update_row_max(
        self, acc_S_row: cute.TensorSSA, is_first: int
    ) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = utils.exp2f(acc_scale_)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    @cute.jit
    def update_row_max_blockscaled(
        self, acc_S_row: cute.TensorSSA, is_first: int
    ) -> Tuple[Float32, Float32, Tuple[Float32, Float32, Float32, Float32]]:
        """Compute row_max and per-block maxes for MXFP8 blockscaling optimization.

        This function computes the max of the 128-element row as 4 block maxes (32 elements each),
        then derives row_max from them. The block maxes can be reused in apply_exp2_convert_blockscaled
        to derive block_amax without recomputing max over exp2 values.

        Args:
            acc_S_row: 128-element TensorSSA containing S values (before softmax)
            is_first: Whether this is the first N tile

        Returns:
            Tuple of (row_max_safe, acc_scale, (block_max0, block_max1, block_max2, block_max3))
            - row_max_safe: The row max (with -inf replaced by 0)
            - acc_scale: Scale factor for O accumulator rescaling
            - block_maxes: Max of S values for each 32-element block
        """
        BLOCK_SIZE: cutlass.Constexpr[int] = 32

        # Create a fragment to work with the values
        res = cute.make_fragment(acc_S_row.shape, Float32)
        res.store(acc_S_row)

        # Compute max for each 32-element block using tree reduction
        # Block 0: elements [0, 32)
        block_max_0 = utils.fmax(res[0], res[1])
        for i in cutlass.range_constexpr(2, BLOCK_SIZE, 2):
            block_max_0 = utils.fmax(block_max_0, res[i], res[i + 1])

        # Block 1: elements [32, 64)
        block_max_1 = utils.fmax(res[BLOCK_SIZE], res[BLOCK_SIZE + 1])
        for i in cutlass.range_constexpr(2, BLOCK_SIZE, 2):
            block_max_1 = utils.fmax(
                block_max_1, res[BLOCK_SIZE + i], res[BLOCK_SIZE + i + 1]
            )

        # Block 2: elements [64, 96)
        block_max_2 = utils.fmax(res[2 * BLOCK_SIZE], res[2 * BLOCK_SIZE + 1])
        for i in cutlass.range_constexpr(2, BLOCK_SIZE, 2):
            block_max_2 = utils.fmax(
                block_max_2, res[2 * BLOCK_SIZE + i], res[2 * BLOCK_SIZE + i + 1]
            )

        # Block 3: elements [96, 128)
        block_max_3 = utils.fmax(res[3 * BLOCK_SIZE], res[3 * BLOCK_SIZE + 1])
        for i in cutlass.range_constexpr(2, BLOCK_SIZE, 2):
            block_max_3 = utils.fmax(
                block_max_3, res[3 * BLOCK_SIZE + i], res[3 * BLOCK_SIZE + i + 1]
            )

        # Compute row_max as max of block maxes
        row_max_new = max_f32(
            max_f32(block_max_0, block_max_1), max_f32(block_max_2, block_max_3)
        )

        if cutlass.const_expr(is_first):
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            # Include previous row_max in the comparison
            row_max_new = max_f32(row_max_new, row_max_old)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = utils.exp2f(acc_scale_)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0

        self.row_max[0] = row_max_new
        return (
            row_max_safe,
            acc_scale,
            (block_max_0, block_max_1, block_max_2, block_max_3),
        )

    def update_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, row_scale: Float32, is_first: int = False
    ) -> None:
        init_val = (
            self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else None
        )
        # self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=self.row_sum[0] * row_scale)
        self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=init_val)
        # tmp = self._compute_row_sum(acc_S_row_exp)
        # self.row_sum[0] = self.row_sum[0] * row_scale + tmp

    @cute.jit
    def scale_subtract_rowmax(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        row_max_scaled = row_max * self.scale_log2
        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
            acc_S_row[i], acc_S_row[i + 1] = utils.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (-row_max_scaled, -row_max_scaled),
            )

    @cute.jit
    def apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        e2e: cutlass.Constexpr[bool] = False,
        e2e_freq: cutlass.Constexpr[int] = 16,
        e2e_res: cutlass.Constexpr[int] = 4,
        e2e_frg_limit: cutlass.Constexpr[int] = 1,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                # acc_S_row_frg[k, j] = utils.exp2f(acc_S_row_frg[k, j])
                # acc_S_row_frg[k + 1, j] = utils.exp2f(acc_S_row_frg[k + 1, j])
                if cutlass.const_expr(not e2e):
                    acc_S_row_frg[k, j] = cute.arch.exp2(acc_S_row_frg[k, j])
                    acc_S_row_frg[k + 1, j] = cute.arch.exp2(acc_S_row_frg[k + 1, j])
                else:
                    if cutlass.const_expr(
                        k % e2e_freq < e2e_freq - e2e_res
                        or j >= frg_cnt - e2e_frg_limit
                    ):
                        acc_S_row_frg[k, j] = cute.arch.exp2(acc_S_row_frg[k, j])
                        acc_S_row_frg[k + 1, j] = cute.arch.exp2(
                            acc_S_row_frg[k + 1, j]
                        )
                    else:
                        # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.e2e_asm2(acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
                            utils.ex2_emulation_2(
                                acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                            )
                        )
        # Use optimized FP8 conversion (cvt_fp8x4_f32) to avoid PRMT instructions
        utils.cvt_f16(acc_S_row, acc_S_row_converted)

    @cute.jit
    def scale_apply_exp2_convert_blockscaled(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        row_max: Float32,
        block_maxes: Tuple[Float32, Float32, Float32, Float32],
        e2e: cutlass.Constexpr[bool] = False,
        e2e_freq: cutlass.Constexpr[int] = 16,
        e2e_res: cutlass.Constexpr[int] = 4,
        max_norm_rcp_val: float = E4M3_MAX_NORM_RCP,
    ) -> Tuple[Uint8, Uint8, Uint8, Uint8]:
        """Apply exp2, compute per-32-block scales, and quantize for MXFP8.

        This follows the GDPA pattern for blockscaled softmax:
        - Process 128 elements in 4 blocks of 32 elements each
        - For each block: derive E8M0 scale from pre-computed block_maxes
        - Apply exp2 and quantize values using inverse scale

        block_amax is derived mathematically as exp2((block_max - row_max) * scale_log2)
        instead of computing max over exp2 values. This saves 128 max operations per tile.

        IMPORTANT: This function leaves acc_S_row with exp2 values (NOT scaled)
        so that update_row_sum can correctly compute the softmax normalization.
        Only acc_S_row_converted gets the scaled FP8 values.

        Args:
            acc_S_row: Input tensor with 128 F32 values (S * scale_log2 - row_max * scale_log2, pre-exp)
                       After return, contains exp2 values (for row_sum computation)
            acc_S_row_converted: Output tensor for 128 FP8 values (P for TMEM)
            row_max: Pre-computed row_max from update_row_max_blockscaled
            block_maxes: Pre-computed block maxes from update_row_max_blockscaled
            e2e: Whether to use e2e emulation for exp2
            e2e_freq: Frequency of e2e emulation
            e2e_res: Resolution of e2e emulation

        Returns:
            Tuple of 4 E8M0 scale factors (one per 32-element block)
        """
        BLOCK_SIZE = 32
        max_norm_rcp = Float32(max_norm_rcp_val)

        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(BLOCK_SIZE))

        # Derive block_amax from pre-computed block_maxes
        # Mathematical derivation: block_amax = exp2((block_max_S - row_max) * scale_log2)
        # This works because exp2 is monotonic, so max(exp2(x)) = exp2(max(x))
        block_amax_0 = cute.arch.exp2((block_maxes[0] - row_max) * self.scale_log2)
        block_amax_1 = cute.arch.exp2((block_maxes[1] - row_max) * self.scale_log2)
        block_amax_2 = cute.arch.exp2((block_maxes[2] - row_max) * self.scale_log2)
        block_amax_3 = cute.arch.exp2((block_maxes[3] - row_max) * self.scale_log2)

        # Derive scales from pre-computed block_amax
        scale0, inv_scale_0, _ = fused_amax_to_e8m0_scale_f32(
            block_amax_0, max_norm_rcp
        )
        scale1, inv_scale_1, _ = fused_amax_to_e8m0_scale_f32(
            block_amax_1, max_norm_rcp
        )
        scale2, inv_scale_2, _ = fused_amax_to_e8m0_scale_f32(
            block_amax_2, max_norm_rcp
        )
        scale3, inv_scale_3, _ = fused_amax_to_e8m0_scale_f32(
            block_amax_3, max_norm_rcp
        )

        inv_scales = (inv_scale_0, inv_scale_1, inv_scale_2, inv_scale_3)

        # Apply exp2, multiply by inv_scale using packed f32x2, and quantize
        # Create temporary tensor for scaled values
        scaled_frg = cute.make_fragment(acc_S_row_frg.shape, Float32)

        for j in cutlass.range_constexpr(4):
            inv_scale_j = inv_scales[j]
            for k in cutlass.range_constexpr(0, BLOCK_SIZE, 2):
                # Apply exp2 (with optional e2e emulation)
                if cutlass.const_expr(not e2e):
                    exp_val_0 = cute.arch.exp2(acc_S_row_frg[k, j])
                    exp_val_1 = cute.arch.exp2(acc_S_row_frg[k + 1, j])
                else:
                    if cutlass.const_expr(k % e2e_freq < e2e_freq - e2e_res):
                        exp_val_0 = cute.arch.exp2(acc_S_row_frg[k, j])
                        exp_val_1 = cute.arch.exp2(acc_S_row_frg[k + 1, j])
                    else:
                        exp_val_0, exp_val_1 = utils.ex2_emulation_2(
                            acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                        )
                # Store exp2 values back (needed for row_sum computation)
                acc_S_row_frg[k, j] = exp_val_0
                acc_S_row_frg[k + 1, j] = exp_val_1
                # Multiply by inv_scale using packed f32x2
                scaled_0, scaled_1 = utils.mul_packed_f32x2(
                    (exp_val_0, exp_val_1), (inv_scale_j, inv_scale_j)
                )
                scaled_frg[k, j] = scaled_0
                scaled_frg[k + 1, j] = scaled_1

        # Use optimized FP8 conversion (cvt_fp8x4_f32) to avoid PRMT instructions
        utils.cvt_f16(scaled_frg, acc_S_row_converted)

        return scale0, scale1, scale2, scale3

    @cute.jit
    def scale_apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
        acc_S_row_converted: cute.Tensor,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        minus_row_max_scaled = -row_max * self.scale_log2
        for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
            acc_S_row[i], acc_S_row[i + 1] = utils.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (minus_row_max_scaled, minus_row_max_scaled),
            )

        # for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
        #     acc_S_row[i], acc_S_row[i + 1] = utils.fma_packed_f32x2(
        #         (acc_S_row[i], acc_S_row[i + 1]),
        #         (self.scale_log2, self.scale_log2),
        #         (minus_row_max_scaled, minus_row_max_scaled),
        #     )
        #     acc_S_row[i] = cute.arch.exp2(acc_S_row[i])
        #     acc_S_row[i + 1] = cute.arch.exp2(acc_S_row[i + 1])

        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
                #     utils.fma_packed_f32x2(
                #         (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]),
                #         (self.scale_log2, self.scale_log2),
                #         (minus_row_max_scaled, minus_row_max_scaled),
                #     )
                # )
                # acc_S_row_frg[k, j] = utils.exp2f(acc_S_row_frg[k, j])
                # acc_S_row_frg[k + 1, j] = utils.exp2f(acc_S_row_frg[k + 1, j])
                acc_S_row_frg[k, j] = cute.arch.exp2(acc_S_row_frg[k, j])
                acc_S_row_frg[k + 1, j] = cute.arch.exp2(acc_S_row_frg[k + 1, j])
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )


@cute.jit
def floor_if_packed(
    q_idx,
    qhead_per_kvhead: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Convert q_idx to packed format for Pack-GQA."""
    if cutlass.const_expr(qhead_per_kvhead == 1):
        return q_idx
    return q_idx // qhead_per_kvhead


@cute.jit
def apply_score_mod_inner(
    score_tensor,
    index_tensor,
    score_mod: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_tensors,
    fastdiv_mods,
    seqlen_info: SeqlenInfoQK,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
):
    """Shared implementation for applying score modification.

    Args:
        score_tensor: The scores to modify (acc_S for flash_fwd, tSrS_t2r for sm100)
        index_tensor: Index positions (tScS for flash_fwd, tScS_t2r for sm100)
        score_mod: The score modification function to apply
        batch_idx: Batch index
        head_idx: Head index
        softmax_scale: Scale to apply
        vec_size: Vector size for processing elements
        qk_acc_dtype: Data type for accumulator
        aux_tensors: Optional aux_tensors for FlexAttention
        fastdiv_mods: Tuple of (seqlen_q_divmod, seqlen_k_divmod) for wrapping
        seqlen_info: Sequence length info
        constant_q_idx: If provided, use this constant for all q_idx values
                        If None, compute q_idx per-element
        qhead_per_kvhead_packgqa: Pack-GQA replication factor. Divide q_idx by this
                                  when greater than 1 so score mods see logical heads.
    """
    n_vals = cutlass.const_expr(cute.size(score_tensor.shape))
    score_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
    kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # SSA values for batch (constant across all elements)
    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32).broadcast_to(
        (vec_size,)
    )

    # Handle q_idx based on whether it's constant
    q_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # For Pack-GQA with non-constant q_idx, we need per-element head indices
    # since a thread my process multiple query head indices
    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        head_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
        for j in cutlass.range(vec_size, unroll_full=True):
            score_vec[j] = score_tensor[i + j] * softmax_scale

            # Extract head offset from packed q_idx for Pack-GQA
            if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
                q_idx_packed = index_tensor[i + j][0]
                # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
                q_idx_logical = q_idx_packed // qhead_per_kvhead
                head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
                head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

            # If we will do loads we mod, in order to not read OOB
            if cutlass.const_expr(aux_tensors is not None and fastdiv_mods is not None):
                if cutlass.const_expr(constant_q_idx is None):
                    seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                    q_idx_floored = floor_if_packed(
                        index_tensor[i + j][0], qhead_per_kvhead
                    )
                    _, q_idx_wrapped = divmod(q_idx_floored, seqlen_q_divmod)
                    q_idx_vec[j] = q_idx_wrapped
                else:
                    _, seqlen_k_divmod = fastdiv_mods

                _, kv_idx_wrapped = divmod(index_tensor[i + j][1], seqlen_k_divmod)
                kv_idx_vec[j] = kv_idx_wrapped
            else:
                # No bounds checking - direct indexing
                if constant_q_idx is None:
                    q_idx_vec[j] = floor_if_packed(
                        index_tensor[i + j][0], qhead_per_kvhead
                    )
                kv_idx_vec[j] = index_tensor[i + j][1]

        # Convert to SSA for score_mod call
        score_ssa = score_vec.load()
        kv_idx_ssa = kv_idx_vec.load()
        if cutlass.const_expr(constant_q_idx is None):
            q_idx_ssa = q_idx_vec.load()
        else:
            # NB we do not apply Pack-GQA division here, as constant_q_idx is assumed to already be logical
            q_idx_const = constant_q_idx
            q_idx_ssa = utils.scalar_to_ssa(q_idx_const, cutlass.Int32).broadcast_to(
                (vec_size,)
            )

        # Compute head_idx_ssa: per-element for Pack-GQA with non-constant q_idx, constant otherwise
        if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
            head_idx_ssa = head_idx_vec.load()
        else:
            head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32).broadcast_to(
                (vec_size,)
            )

        aux_args = []
        if cutlass.const_expr(aux_tensors is not None):
            aux_args = aux_tensors

        post_mod_scores = score_mod(
            score_ssa,
            batch_idx_ssa,
            head_idx_ssa,
            q_idx=q_idx_ssa,
            kv_idx=kv_idx_ssa,
            seqlen_info=seqlen_info,
            aux_tensors=aux_args,
        )

        # Write back modified scores
        score_vec.store(post_mod_scores)
        for j in cutlass.range(vec_size, unroll_full=True):
            score_tensor[i + j] = score_vec[j]
