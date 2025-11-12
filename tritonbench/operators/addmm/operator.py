import argparse
import itertools
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
import torch._inductor.config as inductor_config
import triton
from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.python_utils import try_import

try:
    from hammer.ops.triton.triton_hstu_linear import triton_addmm as hstu_triton_addmm
except ModuleNotFoundError:
    from .hstu import triton_addmm as hstu_triton_addmm

with try_import("HAS_STREAMK"):
    from tritonbench.operators.gemm.stream_k import streamk_cuda_matmul

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)

from .data_io import parse_args

if is_fbcode():
    from tritonbench.utils.fb.addmm_prod import get_prod_shapes
else:
    get_prod_shapes = lambda x: None


# Shape encoding information: (M, K, N, BIAS_1D_Y)
BUILDIN_SHAPES = [
    (20120, 1536, 512, False),
    (34579, 1536, 512, False),
    (34839, 1536, 512, False),
    (35561, 1536, 512, False),
    (35916, 1536, 512, False),
    (19735, 1536, 512, False),
    (34533, 1536, 512, False),
    (35791, 1536, 512, False),
    (35844, 1536, 512, False),
    (20116, 1536, 512, False),
    (33887, 1536, 512, False),
    (20203, 1536, 512, False),
    (33961, 1536, 512, False),
    (19747, 1536, 512, False),
    (34181, 1536, 512, False),
    (35541, 1536, 512, False),
    (36032, 1536, 512, False),
    (15168, 1536, 512, False),
    (35249, 1536, 512, False),
    (33894, 1536, 512, False),
    (20067, 1536, 512, False),
    (27456, 1536, 512, False),
    (19410, 1536, 512, False),
    (35884, 1536, 512, False),
    (35917, 1536, 512, False),
    (19632, 1536, 512, False),
    (35656, 1536, 512, False),
    (35405, 1536, 512, False),
    (35503, 1536, 512, False),
    (35504, 1536, 512, False),
    (35605, 1536, 512, False),
    (34238, 1536, 512, False),
    (33660, 1536, 512, False),
    (35410, 1536, 512, False),
    (20211, 1536, 512, False),
    (34308, 1536, 512, False),
    (34516, 1536, 512, False),
    (20224, 1536, 512, False),
    (35678, 1536, 512, False),
    (35380, 1536, 512, False),
    (35901, 1536, 512, False),
    (20068, 1536, 512, False),
]

# M=13, K=2^6..2^25, N=2, BIAS_1D_Y=False
LARGE_K_SHAPES = list(
    itertools.product([13], [2**i for i in range(6, 26)], [2], [False])
)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "best_config"]
    DEFAULT_PRECISION = "fp16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        addmm_args = parse_args(self.extra_args)
        prod_shapes = get_prod_shapes(addmm_args.config)
        if prod_shapes:
            self.shapes = prod_shapes
        elif addmm_args.m and addmm_args.n and addmm_args.k and addmm_args.bias_1D_y:
            self.shapes = [
                (addmm_args.m, addmm_args.k, addmm_args.n, addmm_args.bias_1D_y)
            ]
        elif addmm_args.large_k_shapes:
            self.shapes = LARGE_K_SHAPES
        else:
            self.shapes = BUILDIN_SHAPES
        self.col_major = addmm_args.col_major

    @register_benchmark()
    def triton_addmm(self, a, mat1, mat2) -> Callable:
        return lambda: hstu_triton_addmm(a, mat1, mat2)

    # FIXME: bwd has some problem, need to re-enable it
    @register_benchmark(enabled=False)
    def streamk_addmm(self, a, mat1, mat2) -> Callable:
        return lambda: streamk_cuda_matmul(mat1, mat2.T.contiguous()) + a

    @register_benchmark(baseline=True)
    def aten_addmm(self, a, mat1, mat2) -> Callable:
        return lambda: torch.addmm(a, mat1, mat2)

    @register_benchmark()
    def pt2_triton_matmul(self, a, mat1, mat2) -> Callable:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            autotune_fallback_to_aten=False,
        ):
            f = lambda a, mat1, mat2: torch.addmm(a, mat1, mat2)
            compiled = torch.compile(f, dynamic=False)
            compiled(a, mat1, mat2)
        return lambda: compiled(a, mat1, mat2)

    @register_benchmark(enabled=False)
    def pt2_addmm_maxautotune(self, a, mat1, mat2) -> Callable:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="ATEN,TRITON",
            autotune_num_choices_displayed=None,
        ):
            f = lambda a, mat1, mat2: torch.addmm(a, mat1, mat2)
            compiled = torch.compile(f, dynamic=False)
            compiled(a, mat1, mat2)
        return lambda: compiled(a, mat1, mat2)

    @register_metric()
    def gbps(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        a, mat1, mat2 = example_inputs
        numel = (
            a.numel()
            + mat1.numel()
            + mat2.numel()
            + (torch.addmm(a, mat1, mat2).numel())
        )
        numel = numel * a.element_size() / 1e9
        return numel / metrics.latency * 1e3

    @register_metric()
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        _, mat1, mat2 = example_inputs
        m, k = mat1.size()
        k, n = mat2.size()
        flops = (2 * m * k * n) + (m * n)
        return flops

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        # x-value: computation intensity
        a, mat1, mat2 = example_inputs
        m, k = mat1.size()
        k, n = mat2.size()
        return (m, n, k)

    def get_input_iter(self) -> Generator:
        for _shape_id, shape in enumerate(self.shapes):
            m, k, n, bias_1D_y = shape
            if bias_1D_y:
                a = torch.randn(n, device=self.device, dtype=self.dtype).requires_grad_(
                    self.requires_grad
                )
            else:
                a = torch.randn(
                    (m, n), device=self.device, dtype=self.dtype
                ).requires_grad_(self.requires_grad)
            mat1 = torch.randn(
                (m, k), device=self.device, dtype=self.dtype
            ).requires_grad_(self.requires_grad)
            mat2 = torch.randn(
                (k, n), device=self.device, dtype=self.dtype
            ).requires_grad_(self.requires_grad)
            if self.col_major:
                mat2 = mat2.T.contiguous().T
            yield a, mat1, mat2

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        accuracy = True
        try:
            torch.testing.assert_close(output, baseline_output, atol=1e-5, rtol=0.5)
        except Exception:
            accuracy = False
        finally:
            return accuracy

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["density"],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "aten_addmm",
                    "triton_addmm",
                ],  # possible values for `line_arg``
                line_names=[
                    "ATen AddMM",
                    "Triton AddMM",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-")],  # line styles
                ylabel="tflops",  # label name for the y-axis
                plot_name="gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(density, provider):
            tflops = self.output.get_y_vals(density, provider, "tflops")
            return tflops

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/test_addmm")
