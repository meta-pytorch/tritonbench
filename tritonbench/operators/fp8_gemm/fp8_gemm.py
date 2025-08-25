import argparse
import logging

from typing import Any, Callable, List, Optional

import torch
import torch._inductor.config as inductor_config
import triton

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    llama_shapes,
    register_benchmark,
    register_metric,
)

from .tutorial import matmul as tutorial_matmul

logger = logging.getLogger(__name__)
try:
    from .persistent import (
        allocate_matmul_tma,
        matmul_persistent,
        matmul_tma_persistent,
    )

    HAS_TMA = True
except ModuleNotFoundError:
    HAS_TMA = False
    logger.warning("Failed to import TMA due to module not being found")
except Exception as e:
    HAS_TMA = False
    logger.warning(f"Failed to import TMA: {e}")


def parse_args(args):
    parser = argparse.ArgumentParser(description="TritonBench fp8_gemm")
    parser.add_argument("--llama", action="store_true")
    parser.add_argument("--scaling_rowwise", action="store_true")
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--per-tensor-scale-a", type=float, default=None)
    parser.add_argument("--per-tensor-scale-b", type=float, default=None)
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "gbps", "latency"]
    DEFAULT_PRECISION = "fp8"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.extra_args = parse_args(extra_args)

    def _get_dtype(self):
        if self.extra_args.scaling_rowwise:
            return torch.bfloat16
        else:
            return torch.float16

    def get_input_iter(self):
        def _get_scale_per_tensor(
            x: torch.Tensor, custom_scale: float = None
        ) -> torch.Tensor:
            # For tensor-wise scaling, kernel requires a float32 scale tensor
            if custom_scale:
                return torch.tensor(custom_scale, dtype=torch.float32, device=x.device)
            scale = torch.finfo(torch.float8_e4m3fn).max / x.abs().max()
            return scale.to(torch.float32)

        def _get_scale_per_row(
            x: torch.Tensor, transpose: bool = False
        ) -> torch.Tensor:
            if transpose:  # scale_b.shape should be [1, N]
                scale = (
                    torch.finfo(torch.float8_e4m3fn).max
                    / x.abs().max(dim=0, keepdim=True).values
                )
            else:  # scale_a.shape should be [M, 1]
                scale = (
                    torch.finfo(torch.float8_e4m3fn).max
                    / x.abs().max(dim=1, keepdim=True).values
                )
            return scale.to(
                torch.float32
            )  # For row-wise scaling, kernel requires a float32 scale tensor

        def args(m, n, k):
            a = torch.randn(m, k, device=self.device).to(self._get_dtype())
            b = (
                torch.randn(k, n, device=self.device)
                .to(self._get_dtype())
                .T.contiguous()
                .T
            )

            if self.extra_args.scaling_rowwise:
                scale_a = _get_scale_per_row(a)
                scale_b = _get_scale_per_row(b, transpose=True)
            else:
                scale_a = _get_scale_per_tensor(
                    a, custom_scale=self.extra_args.per_tensor_scale_a
                )
                scale_b = _get_scale_per_tensor(
                    b, custom_scale=self.extra_args.per_tensor_scale_b
                )

            # Kernels expect dtype=float8_e4m3fn
            a = a.to(torch.float8_e4m3fn)
            b = b.to(torch.float8_e4m3fn)

            return (a, b, scale_a, scale_b)

        if (
            hasattr(self, "external_shapes") and self.external_shapes
        ):  # Check for external shapes loaded from input-loader
            for shape in self.external_shapes:
                if len(shape) == 3:
                    m, n, k = shape
                    yield args(m, n, k)
                else:
                    logger.warning(
                        f"Skipping invalid shape: {shape}, expected [M, N, K]"
                    )
        elif self.extra_args.llama:
            for m, n, k, _bias in llama_shapes():
                yield args(m, n, k)
        elif self.extra_args.m:
            yield args(self.extra_args.m, self.extra_args.n, self.extra_args.k)
        else:
            for i in range(10, 15):
                for j in range(0, 4):
                    k = 2**i
                    k += k // 4 * j
                    m = n = k
                    yield args(m, n, k)

    def get_x_val(self, example_inputs) -> float:
        a, b, _, _ = example_inputs
        m, k = a.size()
        _, n = b.size()
        return (m, n, k)

    @register_benchmark(baseline=True)
    def torch_fp8_gemm(self, a, b, scale_a, scale_b):
        return lambda: torch._scaled_mm(
            a, b, scale_a, scale_b, use_fast_accum=True, out_dtype=self._get_dtype()
        )

    @register_benchmark()
    def pt2_fp8_gemm(self, a, b, scale_a, scale_b) -> Callable:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            autotune_fallback_to_aten=False,
        ):
            f = lambda a, b: torch._scaled_mm(
                a, b, scale_a, scale_b, use_fast_accum=True, out_dtype=self._get_dtype()
            )
            compiled = torch.compile(f, dynamic=False)
            compiled(a, b)

        return lambda: compiled(a, b)

    @register_benchmark()
    def triton_fp8_gemm(self, a, b, scale_a, scale_b):
        return lambda: tutorial_matmul(a, b)

    @register_benchmark(enabled=HAS_TMA)
    def triton_persistent_fp8_gemm(self, a, b, scale_a, scale_b):
        return lambda: matmul_persistent(a, b)

    @register_benchmark(enabled=HAS_TMA)
    def triton_tma_persistent_fp8_gemm(self, a, b, scale_a, scale_b):
        b = b.T.contiguous()
        c, desc_a, desc_b, desc_c = allocate_matmul_tma(a, b)
        return lambda: matmul_tma_persistent(a, b, c, desc_a, desc_b, desc_c)

    @register_metric()
    def gbps(self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics) -> float:
        def nbytes(t):
            return t.numel() * t.element_size()

        a, b, _, _ = example_inputs
        c = fn()
        c = c[0] if isinstance(c, tuple) else c

        m, k = a.shape
        _, n = b.shape
        gb = (nbytes(a) + nbytes(b) + nbytes(c)) / 1e9
        return gb / metrics.latency * 1e3

    @register_metric()
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        a, b, _, _ = example_inputs
        m, k = a.size()
        _, n = b.size()
        flops = 2 * m * n * k
        return flops

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=[
                    "m",
                    "n",
                    "k",
                ],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "torch_fp8_gemm",
                    "triton_fp8_gemm",
                ],  # possible values for `line_arg``
                line_names=[
                    "torch_fp8_gemm",
                    "triton_fp8_gemm",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-")],
                ylabel="tflops",  # label name for the y-axis
                plot_name="fp8-gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(m, n, k, provider):
            tflops = self.output.get_y_vals((m, n, k), provider, "tflops")
            return tflops

        save_path = "/tmp/fp8_gemm"

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
