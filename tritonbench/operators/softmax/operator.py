import argparse
from typing import Generator, List, Optional

import torch
import triton
import triton.language as tl
from tritonbench.utils.data_utils import get_production_shapes
from tritonbench.utils.env_utils import is_fbcode, is_hip
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
    register_x_val,
)

try:
    from quack.softmax import softmax as quack_softmax

    HAS_QUACK = True
except ImportError:
    HAS_QUACK = False


def _softmax_heuristic(BLOCK_SIZE: int, n_cols: int = 0):
    """Select num_warps and num_stages based on BLOCK_SIZE to avoid autotuning.

    Each warp has 32 threads (NVIDIA) or 64 threads (AMD).  We want enough
    elements per thread (~8-16) to amortise warp-scheduling overhead, but
    enough warps to hide memory latency.

    When n_cols is provided and significantly smaller than BLOCK_SIZE (non-
    power-of-2 columns), we base warp count on n_cols to avoid over-
    subscribing warps to masked-out elements.
    """
    # Use actual column count when available, otherwise fall back to BLOCK_SIZE
    effective_size = n_cols if n_cols > 0 else BLOCK_SIZE

    def _round_down_pow2(x: int) -> int:
        """Round down to nearest power of 2, minimum 1."""
        x = max(x, 1)
        # Clear all bits except the highest set bit
        while x & (x - 1):
            x &= x - 1
        return x

    if is_hip():
        # AMD: wavefront=64 threads. For small blocks (< 4096), fewer warps
        # avoids cross-warp reduction overhead. For large blocks (≥ 4096),
        # more warps provide enough wavefronts for memory latency hiding.
        # Use n_cols-based raw (not BLOCK_SIZE) to avoid over-subscribing
        # warps to masked-out elements at non-power-of-2 N.
        raw = effective_size // 512
        if BLOCK_SIZE >= 4096:
            num_warps = _round_down_pow2(max(4, min(raw, 16)))
        else:
            num_warps = _round_down_pow2(max(1, min(raw, 4)))
        num_stages = 1
    else:
        # NVIDIA: warp=32, target ~8 elements per thread
        raw = effective_size // 256
        num_warps = _round_down_pow2(max(1, min(raw, 16)))
        # Fewer stages at large BLOCK_SIZE to reduce register pressure
        if BLOCK_SIZE >= 4096:
            num_stages = 1
        elif BLOCK_SIZE >= 1024:
            num_stages = 2
        else:
            num_stages = 4
    return num_warps, num_stages


QUACK_SHAPES = [
    (32 * 1024, 256),
    (32 * 1024, 512),
    (32 * 1024, 1024),
    (32 * 1024, 2 * 1024),
    (32 * 1024, 4 * 1024),
    (32 * 1024, 8 * 1024),
    (32 * 1024, 16 * 1024),
    (32 * 1024, 32 * 1024),
    (32 * 1024, 65 * 1024),
    (16 * 1024, 131 * 1024),
    (8 * 1024, 262 * 1024),
]


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--M",
        type=int,
        default=4096,
        help="[Optional] Size of dimension 0 in input shape (integer), default: 4096",
    )
    parser.add_argument(
        "--N",
        type=int,
        help="[Optional] Size of dimension 1 in input shape (integer)",
    )
    parser.add_argument(
        "--quack-shapes",
        action="store_true",
        help="[Optional] Use the QuACK benchmark shapes for softmax evaluation",
    )
    return parser.parse_args(args)


class TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        y = torch.empty_like(x)
        num_warps, num_stages = _softmax_heuristic(BLOCK_SIZE, n_cols)
        # For small N with many rows, use multi-row kernel to reduce grid size
        # and improve SM utilization by giving each program more work.
        # 2D layout: [ROWS_PER_PROGRAM, BLOCK_SIZE] processed in parallel.
        if BLOCK_SIZE <= 512 and n_rows >= 8192:
            # Target: ~2048-4096 total elements per program for good occupancy
            ROWS_PER_PROGRAM = min(4096 // BLOCK_SIZE, 8)
            grid = ((n_rows + ROWS_PER_PROGRAM - 1) // ROWS_PER_PROGRAM,)
            # More warps for 2D layout: each warp handles a subset of rows
            multirow_warps = max(2, min(ROWS_PER_PROGRAM * BLOCK_SIZE // 256, 8))
            Operator.softmax_kernel_multirow[grid](
                y,
                x,
                x.stride(0),
                y.stride(0),
                n_cols,
                n_rows,
                num_warps=multirow_warps,
                num_stages=num_stages,
                BLOCK_SIZE=BLOCK_SIZE,
                ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
            )
        else:
            Operator.softmax_kernel[(n_rows,)](
                y,
                x,
                x.stride(0),
                y.stride(0),
                n_cols,
                num_warps=num_warps,
                num_stages=num_stages,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        return Operator.softmax_bwd_triton(grad_output, y)


triton_softmax_fn = TritonSoftmax.apply


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "fp16"
    is_compute_bound = False

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.M = args.M
        self.N = args.N
        self.quack_shapes = args.quack_shapes

    @register_benchmark()
    def triton_softmax(self, x):
        n_cols = x.shape[-1]
        if n_cols > 65536:
            return None
        return lambda: triton_softmax_fn(x)

    @triton.jit
    def softmax_kernel(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        # The rows of the softmax are independent, so we parallelize across those
        row_idx = tl.program_id(0)
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    @triton.jit
    def softmax_kernel_multirow(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_cols,
        n_rows,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        # 2D parallel layout: each program processes ROWS_PER_PROGRAM rows
        # simultaneously using [ROWS_PER_PROGRAM, BLOCK_SIZE] tensor layout.
        # This mirrors torch.compile's persistent_reduction approach.
        prog_idx = tl.program_id(0)
        row_offsets = prog_idx * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
        col_offsets = tl.arange(0, BLOCK_SIZE)

        # [ROWS_PER_PROGRAM, BLOCK_SIZE] index grid
        indices = row_offsets[:, None] * input_row_stride + col_offsets[None, :]
        mask = (row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols)

        # Load all rows at once
        data = tl.load(input_ptr + indices, mask=mask, other=-float("inf"))

        # Row-wise max, exp, sum, normalize — reductions along axis=1
        row_max = tl.max(data, axis=1)[:, None]
        numerator = tl.exp(data - row_max)
        denominator = tl.sum(numerator, axis=1)[:, None]
        softmax_output = numerator / denominator

        # Store
        out_indices = row_offsets[:, None] * output_row_stride + col_offsets[None, :]
        tl.store(output_ptr + out_indices, softmax_output, mask=mask)

    @triton.jit
    def softmax_bwd_kernel(
        grad_input_ptr,
        grad_output_ptr,
        softmax_output_ptr,
        row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        # One row per program — mirrors the forward kernel design.
        # Single pass: load full row, compute dot product, compute gradient.
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        offset = row_idx * row_stride
        grad_out = tl.load(grad_output_ptr + offset + col_offsets, mask=mask, other=0.0)
        soft_out = tl.load(
            softmax_output_ptr + offset + col_offsets, mask=mask, other=0.0
        )

        # dot = sum(softmax * grad_output) per row
        dot = tl.sum(soft_out * grad_out, axis=0)

        # grad_input = softmax * (grad_output - dot)
        grad_in = soft_out * (grad_out - dot)

        tl.store(grad_input_ptr + offset + col_offsets, grad_in, mask=mask)

    @triton.jit
    def softmax_bwd_kernel_tiled(
        grad_input_ptr,
        grad_output_ptr,
        softmax_output_ptr,
        row_stride,
        n_cols,
        TILE_SIZE: tl.constexpr,
    ):
        # Tiled softmax backward for large N (> 65536) where next_power_of_2(N)
        # would exceed GPU register capacity.  Two-pass algorithm:
        # Pass 1: accumulate dot = sum(y * dy) across tiles
        # Pass 2: compute grad_input = y * (dy - dot) per tile
        row_idx = tl.program_id(0)
        offset = row_idx * row_stride
        col_offsets = tl.arange(0, TILE_SIZE)

        # Pass 1: tiled dot product accumulation (in fp32 for precision)
        dot = tl.zeros([], dtype=tl.float32)
        for col_start in tl.range(0, n_cols, TILE_SIZE):
            idx = col_start + col_offsets
            mask = idx < n_cols
            dy_blk = tl.load(grad_output_ptr + offset + idx, mask=mask, other=0.0).to(
                tl.float32
            )
            y_blk = tl.load(softmax_output_ptr + offset + idx, mask=mask, other=0.0).to(
                tl.float32
            )
            dot += tl.sum(dy_blk * y_blk, axis=0)

        # Pass 2: compute and store gradient per tile
        for col_start in tl.range(0, n_cols, TILE_SIZE):
            idx = col_start + col_offsets
            mask = idx < n_cols
            dy_blk = tl.load(grad_output_ptr + offset + idx, mask=mask, other=0.0).to(
                tl.float32
            )
            y_blk = tl.load(softmax_output_ptr + offset + idx, mask=mask, other=0.0).to(
                tl.float32
            )
            grad_blk = y_blk * (dy_blk - dot)
            tl.store(grad_input_ptr + offset + idx, grad_blk, mask=mask)

    @staticmethod
    def softmax_bwd_triton(grad_output, softmax_output):
        """
        Optimized backward kernel with tiled fallback for large N.
        Uses single-pass for N <= 65536 and two-pass tiled kernel for larger N
        to avoid exceeding GPU register capacity.
        """
        m, n = grad_output.size()
        grad_input = torch.empty_like(grad_output)

        if n > 65536:
            TILE_SIZE = 2048
            tiled_warps, tiled_stages = _softmax_heuristic(TILE_SIZE, TILE_SIZE)
            Operator.softmax_bwd_kernel_tiled[(m,)](
                grad_input,
                grad_output,
                softmax_output,
                grad_output.stride(0),
                n,
                num_warps=tiled_warps,
                num_stages=tiled_stages,
                TILE_SIZE=TILE_SIZE,
            )
        else:
            BLOCK_SIZE = triton.next_power_of_2(n)
            num_warps, num_stages = _softmax_heuristic(BLOCK_SIZE, n)
            Operator.softmax_bwd_kernel[(m,)](
                grad_input,
                grad_output,
                softmax_output,
                grad_output.stride(0),
                n,
                num_warps=num_warps,
                num_stages=num_stages,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        return grad_input

    @register_benchmark(baseline=True)
    def naive_softmax(self, x):
        """Compute row-wise softmax of X using native pytorch."""

        def _inner():
            return torch.nn.functional.softmax(x, dim=1)

        return _inner

    @register_benchmark(enabled=HAS_QUACK)
    def quack(self, x):
        inner = lambda: quack_softmax(x)
        return inner

    @register_benchmark()
    def torch_compile_softmax(self, x):
        @torch.compile(mode="max-autotune-no-cudagraphs")
        def _inner(x):
            return torch.nn.functional.softmax(x, dim=1)

        return lambda: _inner(x)

    def get_input_iter(self):
        # If quack-shapes is provided, use the QuACK benchmark shapes
        if self.quack_shapes:
            shapes = QUACK_SHAPES
        # If N is provided, use only that value; otherwise use the default range
        elif self.N is not None:
            shapes = [(self.M, self.N)]
        else:
            shapes = [(self.M, 128 * i) for i in range(2, 100)]

        if is_fbcode() and self.tb_args.production_shapes:
            additional_shapes = get_production_shapes(
                self.name, "softmax", self.tb_args.shuffle_shapes
            )
            if additional_shapes:
                shapes.extend(additional_shapes)

        requires_grad = not (self.mode == Mode.FWD_NO_GRAD)

        for M, N in shapes:
            yield (
                torch.randn(
                    [M, N],
                    dtype=self.dtype,
                    device=self.device,
                    requires_grad=requires_grad,
                ),
            )

    @register_x_val(label="(M, N)")
    def get_x_val(self, example_inputs):
        M, N = example_inputs[0].shape
        return (M, N)

    @register_metric()
    def gbps(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics) -> float:
        return (
            2
            * example_inputs[0].nelement()
            * example_inputs[0].element_size()
            * 1e-9
            / (metrics.latency * 1e-3)
        )

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["N"],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "triton_softmax",
                    "naive_softmax",
                ],  # possible values for `line_arg``
                line_names=[
                    "Triton",
                    "Torch (native)",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-"), ("green", "--")],  # line styles
                ylabel="GB/s",  # label name for the y-axis
                plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={
                    "M": self.M
                },  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(M, N, provider):
            gbps, max_gbps, min_gbps = self.output.get_y_vals(N, provider, "gbps")
            return gbps, max_gbps, min_gbps

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/test_softmax")
