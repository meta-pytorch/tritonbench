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
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "gbps", "latency"]
    DEFAULT_PRECISION = "fp8"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.extra_args = parse_args(extra_args)

    def get_input_iter(self):
        def args(m, n, k):
            a = torch.randn(m, k, device=self.device).to(torch.float8_e4m3fn)
            b = (
                torch.randn(k, n, device=self.device)
                .to(torch.float8_e4m3fn)
                .T.contiguous()
                .T
            )
            return (a, b)

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
        a, b = example_inputs
        m, k = a.size()
        _, n = b.size()
        return (m, n, k)

    @register_benchmark(baseline=True)
    def torch_fp8_gemm(self, a, b):
        if self.extra_args.scaling_rowwise:
            M, N = a.shape[0], b.shape[1]
            scale_a = torch.ones((M, 1), dtype=torch.float32, device=a.device)
            scale_b = torch.ones((1, N), dtype=torch.float32, device=b.device)
            out_dtype = torch.bfloat16
        else:
            scale_a = torch.tensor(1.0, device=a.device)
            scale_b = torch.tensor(1.0, device=a.device)
            out_dtype = torch.float16

        return lambda: torch._scaled_mm(
            a, b, scale_a, scale_b, use_fast_accum=True, out_dtype=out_dtype
        )

    @register_benchmark()
    def pt2_fp8_gemm(self, a, b) -> Callable:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            autotune_fallback_to_aten=False,
        ):
            if self.extra_args.scaling_rowwise:
                M, N = a.shape[0], b.shape[1]
                scale_a = torch.ones((M, 1), dtype=torch.float32, device=a.device)
                scale_b = torch.ones((1, N), dtype=torch.float32, device=b.device)
                out_dtype = torch.bfloat16
            else:
                scale_a = torch.tensor(1.0, device=a.device)
                scale_b = torch.tensor(1.0, device=b.device)
                out_dtype = torch.float16
            f = lambda a, b: torch._scaled_mm(
                a, b, scale_a, scale_b, use_fast_accum=True, out_dtype=out_dtype
            )
            compiled = torch.compile(f, dynamic=False)
            compiled(a, b)

        return lambda: compiled(a, b)

    @register_benchmark()
    def triton_fp8_gemm(self, a, b):
        return lambda: tutorial_matmul(a, b)

    @register_benchmark(enabled=HAS_TMA)
    def triton_persistent_fp8_gemm(self, a, b):
        return lambda: matmul_persistent(a, b)

    @register_benchmark(enabled=HAS_TMA)
    def triton_tma_persistent_fp8_gemm(self, a, b):
        b = b.T.contiguous()
        c, desc_a, desc_b, desc_c = allocate_matmul_tma(a, b)
        return lambda: matmul_tma_persistent(a, b, c, desc_a, desc_b, desc_c)

    @register_metric()
    def gbps(self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics) -> float:
        def nbytes(t):
            return t.numel() * t.element_size()

        a, b = example_inputs
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
        a, b = example_inputs
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
