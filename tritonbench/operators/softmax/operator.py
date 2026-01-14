import argparse
from typing import Generator, List, Optional

import torch
import triton
import triton.language as tl
from tritonbench.utils.data_utils import get_production_shapes
from tritonbench.utils.env_utils import is_fbcode
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
        # The block size is the smallest power of two greater than the number of columns in `x`
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        # Another trick we can use is to ask the compiler to use more threads per row by
        # increasing the number of warps (`num_warps`) over which each row is distributed.
        # You will see in the next tutorial how to auto-tune this value in a more natural
        # way so you don't have to come up with manual heuristics yourself.
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        # Allocate output
        y = torch.empty_like(x)

        # Enqueue kernel
        Operator.softmax_kernel[(n_rows,)](
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_cols,
            num_warps=num_warps,
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
    def softmax_bwd_kernel(
        softmax_output,
        grad_output,
        grad_input,
        grad_input_stride_0,
        grad_input_stride_1,
        grad_output_stride_0,
        grad_output_stride_1,
        softmax_output_stride_0,
        softmax_output_stride_1,
        m,
        n,
        BLOCK_SIZE_0: tl.constexpr,
        BLOCK_SIZE_1: tl.constexpr,
        BLOCK_SIZE_2: tl.constexpr,
    ):
        pid_0 = tl.program_id(0)
        offset_0 = pid_0 * BLOCK_SIZE_0
        indices_0 = (offset_0 + tl.arange(0, BLOCK_SIZE_0)).to(tl.int32)
        mask_0 = indices_0 < m
        sum_per_row = tl.full([BLOCK_SIZE_0], 0.0, tl.float32)
        for offset_1 in tl.range(0, n.to(tl.int32), BLOCK_SIZE_1):
            indices_1 = offset_1 + tl.arange(0, BLOCK_SIZE_1).to(tl.int32)
            mask_1 = indices_1 < n
            sum_per_row_copy = sum_per_row
            sum_per_row_copy_0 = sum_per_row_copy
            load = tl.load(
                softmax_output
                + (
                    indices_0[:, None] * softmax_output_stride_0
                    + indices_1[None, :] * softmax_output_stride_1
                ),
                mask_0[:, None] & mask_1[None, :],
                other=0,
            )
            load_1 = tl.load(
                grad_output
                + (
                    indices_0[:, None] * grad_output_stride_0
                    + indices_1[None, :] * grad_output_stride_1
                ),
                mask_0[:, None] & mask_1[None, :],
                other=0,
            )
            v_0 = load * load_1
            sum_1 = tl.cast(tl.sum(v_0, 1), tl.float16)
            v_1 = tl.cast(sum_1, tl.float32)
            sum_per_row = sum_per_row_copy_0 + v_1
        for offset_2 in tl.range(0, n.to(tl.int32), BLOCK_SIZE_2):
            indices_2 = offset_2 + tl.arange(0, BLOCK_SIZE_2).to(tl.int32)
            mask_2 = indices_2 < n
            sum_per_row_copy_1 = sum_per_row
            sum_per_row_copy_1_0 = sum_per_row_copy_1
            load_2 = tl.load(
                softmax_output
                + (
                    indices_0[:, None] * softmax_output_stride_0
                    + indices_2[None, :] * softmax_output_stride_1
                ),
                mask_0[:, None] & mask_2[None, :],
                other=0,
            )
            load_3 = tl.load(
                grad_output
                + (
                    indices_0[:, None] * grad_output_stride_0
                    + indices_2[None, :] * grad_output_stride_1
                ),
                mask_0[:, None] & mask_2[None, :],
                other=0,
            )
            subscript = sum_per_row_copy_1_0[:, None]
            v_3 = tl.cast(load_3, tl.float32)
            v_4 = v_3 - subscript
            v_5 = tl.cast(load_2, tl.float32)
            v_6 = v_5 * v_4
            v_7 = tl.cast(v_6, tl.float16)
            tl.store(
                grad_input
                + (
                    indices_0[:, None] * grad_input_stride_0
                    + indices_2[None, :] * grad_input_stride_1
                ),
                v_7,
                mask_0[:, None] & mask_2[None, :],
            )

    @staticmethod
    def softmax_bwd_triton(grad_output, softmax_output):
        """
        Helion generated triton kernel for softmax backward pass
        PR: https://github.com/pytorch/helion/pull/744
        """
        m, n = grad_output.size()
        grad_input = torch.empty_like(grad_output)

        BLOCK_SIZE_0 = min(32, triton.next_power_of_2(m))
        BLOCK_SIZE_1 = triton.next_power_of_2(n)
        BLOCK_SIZE_2 = BLOCK_SIZE_1

        Operator.softmax_bwd_kernel[(triton.cdiv(m, BLOCK_SIZE_0),)](
            softmax_output,
            grad_output,
            grad_input,
            grad_input.stride(0),
            grad_input.stride(1),
            grad_output.stride(0),
            grad_output.stride(1),
            softmax_output.stride(0),
            softmax_output.stride(1),
            m,
            n,
            BLOCK_SIZE_0,
            BLOCK_SIZE_1,
            BLOCK_SIZE_2,
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
