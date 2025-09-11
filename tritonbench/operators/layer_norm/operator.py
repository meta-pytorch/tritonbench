import argparse
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
import triton

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
    register_x_val,
)

from . import tutorial


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
    return parser.parse_args(args)


try:
    from liger_kernel.ops.layer_norm import LigerLayerNormFunction

    HAS_LIGER_KERNEL = True
except ModuleNotFoundError:
    LigerLayerNormFunction = None
    HAS_LIGER_KERNEL = False


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.M = args.M
        self.N = args.N

    @register_benchmark()
    def triton_layer_norm(self, *args):
        return lambda: tutorial.layer_norm(*args)

    @register_benchmark(baseline=True)
    def torch_layer_norm(self, *args):
        return lambda: F.layer_norm(*args)

    @register_benchmark()
    def torch_compile_layer_norm(self, *args):
        # TODO: remove this once we have a better way to handle backward benchmarking
        # We need to run backward multiple times for proper benchmarking
        # so donated buffer have to be disabled
        if self.mode == Mode.BWD or self.mode == Mode.FWD_BWD:
            from torch._functorch import config as functorch_config

            functorch_config.donated_buffer = False
        import torch

        @torch.compile(mode="max-autotune-no-cudagraphs")
        def inner(*args):
            return F.layer_norm(*args)

        return lambda: inner(*args)

    @register_benchmark(enabled=HAS_LIGER_KERNEL)
    def liger_layer_norm(self, *args):
        (x, w_shape, weight, bias, eps) = args
        return lambda: LigerLayerNormFunction.apply(x, weight, bias, eps)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        from torch.utils._pytree import tree_map

        # Run forward once to get output
        y = fwd_fn()
        dy = 0.1 * torch.randn_like(y)

        # Extract tensors that require gradients from example_inputs
        grad_tensors = []

        def extract_if_requires_grad(x):
            if isinstance(x, torch.Tensor) and x.requires_grad:
                grad_tensors.append(x)
            return x

        # Use tree_map to find all grad tensors in example_inputs
        # example_inputs is set by the benchmark framework and contains the current input
        tree_map(extract_if_requires_grad, self.example_inputs)

        def bwd_fn():
            # Clear existing gradients
            for t in grad_tensors:
                if t.grad is not None:
                    t.grad = None

            # Run backward
            y.backward(dy, retain_graph=True)

            # Return the tensors (not gradients) for accuracy checking
            return grad_tensors

        return bwd_fn

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        x = args[0]
        return [x]

    def get_input_iter(self):
        eps = 1e-5

        # If N is provided, use only that value; otherwise use the default range
        if self.N is not None:
            N_values = [self.N]
        else:
            N_values = [512 * i for i in range(2, 32)]

        for N in N_values:
            x_shape = (self.M, N)
            w_shape = (x_shape[-1],)
            x = -2.3 + 0.5 * torch.randn(
                x_shape,
                dtype=self.dtype,
                device=self.device,
            )
            x.requires_grad_()
            weight = torch.rand(
                w_shape, dtype=self.dtype, device=self.device, requires_grad=True
            )
            bias = torch.rand(
                w_shape, dtype=self.dtype, device=self.device, requires_grad=True
            )
            yield (x, w_shape, weight, bias, eps)

    @register_x_val(label="(M, N)")
    def get_x_val(self, args):
        M, N = args[0].shape
        return (M, N)

    @register_metric()
    def gbps(self, fn, args, metrics: BenchmarkOperatorMetrics) -> float:
        x = args[0]
        base = x.numel() * x.element_size() / metrics.latency * 1e-6
        return {
            Mode.FWD: 2 * base,
            Mode.BWD: 3 * base,
            Mode.FWD_BWD: 5 * base,
        }[self.mode]

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["N"],
                x_vals=self.output.x_vals,
                line_arg="provider",
                line_vals=[
                    "triton_layer_norm",
                    "torch_layer_norm",
                ],
                line_names=[
                    "triton_layer_norm",
                    "torch_layer_norm",
                ],
                styles=[("blue", "-"), ("green", "-")],
                ylabel="GB/s",
                plot_name="layer-norm-fwd",
                args={"M": self.M},
            )
        )
        def _plot(M, N, provider):
            gbps, max_gbps, min_gbps = self.output.get_y_vals(N, provider, "gbps")
            return gbps, max_gbps, min_gbps

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/test_layer_norm")
