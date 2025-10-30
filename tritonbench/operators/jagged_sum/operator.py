import argparse
import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
import triton

from tritonbench.utils.jagged_utils import (
    ABSOLUTE_TOLERANCE,
    generate_input_vals,
    generate_random_nested_tensors,
    get_param_fstrings,
    get_parse_op_args,
    get_plot_args,
    get_styles,
    get_tensor_bytes_limit,
    GIGABYTES_PER_BYTE,
    jagged_to_nested_tensor,
    RANDOM_CHOICE_MARGIN,
    RELATIVE_TOLERANCE,
)

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernels import (
    triton_jagged_sum_kernel_simple_fused_buffer_then_sum,
    triton_jagged_sum_kernel_simple_fused_sum_then_buffer,
    triton_jagged_sum_kernel_variable_length_loop_buffer_then_sum,
    triton_jagged_sum_kernel_variable_length_loop_sum_then_buffer,
)


def parse_op_args(args: List[str]):
    parser = get_parse_op_args(
        "B", "M", "seqlen", "sparsity", "sum_then_buffer", "plot_benchmarks"
    )
    return parser.parse_args(args)


def execute_kernel_simple_fused(x, max_seqlen, sum_then_buffer):
    B, M = x.shape[0], x.shape[2]
    grid = lambda meta: ((len(x.offsets()) - 1) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    kernel_output = torch.zeros((B, M), device=x.device)

    if sum_then_buffer:
        triton_jagged_sum_kernel_simple_fused_sum_then_buffer[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
            MAX_SEQLEN=max_seqlen,
        )
    else:
        triton_jagged_sum_kernel_simple_fused_buffer_then_sum[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
            MAX_SEQLEN=max_seqlen,
        )

    return kernel_output


def execute_kernel_variable_length_loop(x, sum_then_buffer):
    B, M = x.shape[0], x.shape[2]
    grid = lambda meta: ((len(x.offsets()) - 1) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    kernel_output = torch.zeros((B, M), device=x.device)
    # The size of the profile memory will be determined by the autotuner
    profile_mem = torch.empty(0, dtype=torch.int64, device=x.device)

    if sum_then_buffer:
        triton_jagged_sum_kernel_variable_length_loop_sum_then_buffer[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
            profile_mem=profile_mem,
        )
    else:
        triton_jagged_sum_kernel_variable_length_loop_buffer_then_sum[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
            profile_mem=profile_mem,
        )

    return {"output": kernel_output, "profile_mem": profile_mem}


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "best_config"]
    DEFAULT_PRECISION = "fp32"
    FWD_ONLY = True

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.sizes = list(
            range(2, 12, 4)
        )  # bias towards larger sizes, which are more representative of real-world shapes

        args = parse_op_args(self.extra_args)
        self.B = args.B
        self.M = args.M
        self.seqlen = args.seqlen
        self.sparsity = args.sparsity
        self.sum_then_buffer = args.sum_then_buffer
        self.plot_benchmarks = args.plot_benchmarks

        self.tensor_bytes_limit = get_tensor_bytes_limit(tb_args.test_only)

    @register_benchmark(baseline=True)
    def torch_jagged_sum_no_pad(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        return lambda: torch.tensor(
            [
                torch.sum(t, dim=0).tolist() for t in x.unbind()
            ],  # in 3D tensor (B, *, M), sums B 2D tensors (*, M)
            device=self.device,
            dtype=self.dtype,
        )

    @register_benchmark()
    def torch_jagged_sum_pad(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        return (
            lambda: torch.sum(
                torch.ops.aten._jagged_to_padded_dense_forward(
                    x.values(),
                    [
                        x.offsets()
                    ],  # pyre-ignore: Undefined attribute [16]: `torch._tensor.Tensor` has no attribute `offsets`.
                    max_lengths=[seqlen],  # max length of ragged dimension
                ),
                dim=1,
            )
        )  # sum along ragged dimension (dim == 1)

    @register_benchmark()
    def triton_jagged_sum_no_pad_simple_fused(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            return execute_kernel_simple_fused(x, seqlen, self.sum_then_buffer)

        return _inner

    @register_benchmark()
    def triton_jagged_sum_no_pad_variable_length_loop(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            return execute_kernel_variable_length_loop(x, self.sum_then_buffer)

        return _inner

    @register_benchmark()
    def torch_compile_nested_tensor_integration(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner(x: torch.Tensor):  # sum along ragged dimension (dim == 1)
            return torch.sum(
                x, dim=x._ragged_idx
            )  # pyre-ignore: Undefined attribute [16]: `torch._tensor.Tensor` has no attribute `_ragged_idx`.

        torch_compile_func = torch.compile(_inner)
        return lambda: torch_compile_func(x)

    def get_x_val(self, example_inputs):
        if self.B is None:
            return example_inputs[1]
        if self.M is None:
            return example_inputs[2]
        if self.seqlen is None:
            return example_inputs[3]
        if self.sparsity is None:
            return example_inputs[4]

    def get_x_vals(self) -> Tuple[List[int], List[int], List[int], List[float]]:
        return generate_input_vals(
            self.B, self.M, self.seqlen, self.sparsity, self.sizes
        )

    def get_input_iter(self) -> Generator:
        """
        Generate random nested tensors of shape (B, *, M), where * is the ragged dimension
        """
        if not self.prod_shapes:
            B_vals, M_vals, seqlen_vals, sparsity_vals = self.get_x_vals()
            yield from generate_random_nested_tensors(
                B_vals,
                M_vals,
                seqlen_vals,
                sparsity_vals,
                device=self.device,
                dtype=self.dtype,
                TENSOR_BYTES_LIMIT=self.tensor_bytes_limit,
                RANDOM_CHOICE_MARGIN=RANDOM_CHOICE_MARGIN,
            )
        else:
            from tritonbench.data.fb.input_loader import get_input_loader_jagged

            loader = get_input_loader_jagged(self)
            for jagged_values, jagged_offsets, dense_0, _ in loader():
                nested_tensor = jagged_to_nested_tensor(jagged_values, jagged_offsets)
                # Yueming: in the future, if we integrate more input shapes for other jagged operators,
                # the dense_0 may be None. In that case, we should use another way to obtain the batch size
                # and max seq len.
                batch_size, max_seq_len, _ = dense_0.shape
                yield (nested_tensor, batch_size, 1, max_seq_len, 0.0)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        return torch.allclose(
            output, baseline_output, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        )

    @register_metric()
    def gbps(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics):
        return (
            example_inputs[0].element_size()
            * example_inputs[0].numel()
            / metrics.latency
            * GIGABYTES_PER_BYTE
        )

    @register_metric(x_only=True)
    def input_shape(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return (
            f"B: {example_inputs[1]}",  # B
            "*",
            f"M: {example_inputs[2]}",  # M
            f"max seqlen: {example_inputs[3]}",  # seqlen
            f"sparsity: {example_inputs[4]}",  # sparsity
        )  # return (B, '*', M, max seqlen, sparsity) for each example input

    @register_metric()
    def occupancy(
        self, fn: Callable, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        profile_mem = fn().get("profile_mem", None)
        if profile_mem is None:
            return None

        # each row of profile_mem is (smid, start, end)
        smids = profile_mem[:, 0]
        start_times = profile_mem[:, 1]
        end_times = profile_mem[:, 2]

        # We use the actual number of SMs that are active to calculate occupancy.
        # max_sm counts the total number of SMs on the device, but not all of them
        # are active.
        active_sm = torch.unique(smids).numel()
        # max_sm = torch.cuda.get_device_properties("cuda").multi_processor_count

        # Wall time measures the actual time taken to run the kernel.
        # GPU time measures the time spent on the GPU, aggregated across all SMs.
        wall_time = torch.max(end_times) - torch.min(start_times)
        gpu_time = torch.sum(end_times - start_times)

        # We define the occupancy to be the ratio of actual GPU time to the maximum
        # possible GPU time using the active SMs.
        NUM_WAVES = 2
        occupancy = gpu_time / (wall_time * active_sm) / NUM_WAVES
        return occupancy

    def plot(self):
        x_axis, params = get_param_fstrings(self.B, self.M, self.seqlen, self.sparsity)

        line_vals_all = [
            "torch_jagged_sum_no_pad",
            "torch_jagged_sum_pad",
            "triton_jagged_sum_no_pad_simple_fused",
            "triton_jagged_sum_no_pad_variable_length_loop",
            "torch_compile_nested_tensor_integration",
        ]
        line_names_all = [
            "PyTorch jagged sum, no padding",
            "PyTorch jagged sum, padding",
            "Triton kernel jagged sum, simple fused",
            "Triton kernel jagged sum, variable length loop",
            "Inductor, NestedTensor integration",
        ]
        styles_all = get_styles(len(line_vals_all))

        line_vals, line_names, styles = get_plot_args(
            self.plot_benchmarks, 2, line_vals_all, line_names_all, styles_all
        )

        plot_name = f"jagged-sum-perf-var-{x_axis}" + params

        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["x_axis"],
                x_vals=self.output.x_vals,
                line_arg="provider",
                line_vals=line_vals,
                line_names=line_names,
                styles=styles,
                xlabel=x_axis,
                ylabel="latency",
                plot_name=plot_name,
                args={},
            )
        )
        def _plot(x_axis, provider):
            latency = self.output.get_y_vals(x_axis, provider, "latency")
            return latency.p50, latency.max, latency.min

        save_path = (
            os.getcwd()
            + f"/pytorch/tritonbench/tritonbench/operators/jagged_sum/jagged_sum_performance/{plot_name}"
        )

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
