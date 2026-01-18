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

from . import fused_triton, tutorial


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
        help="[Optional] Use the QuACK benchmark shapes for layer norm evaluation",
    )
    return parser.parse_args(args)


try:
    from liger_kernel.ops.layer_norm import LigerLayerNormFunction

    HAS_LIGER_KERNEL = True
except ModuleNotFoundError:
    LigerLayerNormFunction = None
    HAS_LIGER_KERNEL = False

try:
    from quack.rmsnorm import layernorm_fwd as quack_layernorm

    HAS_QUACK_KERNEL = True
except (ModuleNotFoundError, ImportError):
    HAS_QUACK_KERNEL = False


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.M = args.M
        self.N = args.N
        self.quack_shapes = args.quack_shapes
        if self.tb_args.rtol is None:
            self.tb_args.rtol = 1e-5
        if self.tb_args.atol is None:
            self.tb_args.atol = 5e-3

    @register_benchmark()
    def triton_layer_norm(self, *args):
        return lambda: tutorial.layer_norm(*args)

    @register_benchmark()
    def triton_fused_layer_norm(self, *args):
        # Fused bwd Triton Layer Norm
        return lambda: fused_triton.layer_norm(*args)

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

    @register_benchmark(enabled=HAS_QUACK_KERNEL, fwd_only=True)
    def quack_layer_norm(self, *args) -> Callable:
        (x, w_shape, weight, bias, eps) = args
        return lambda: quack_layernorm(x, weight, bias=bias, eps=eps)

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        x = args[0]
        return [x]

    def get_input_iter(self):
        eps = 1e-5

        # If quack-shapes is provided, use the QuACK benchmark shapes
        if self.quack_shapes:
            shapes = QUACK_SHAPES
        # If N is provided, use only that value; otherwise use the default range
        elif self.N is not None:
            shapes = [(self.M, self.N)]
        else:
            shapes = [(self.M, 512 * i) for i in range(2, 32)]

        for M, N in shapes:
            x_shape = (M, N)
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
