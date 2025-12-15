import argparse

import logging

from typing import Any, Callable, List, Optional

import torch
import torch._inductor.config as inductor_config
import triton

from torch._inductor.kernel.mm import scaling_pairs, ScalingType

from tritonbench.data.llama import llama_shapes
from tritonbench.operators.fp8_gemm.persistent import blackwell_persistent_tma
from tritonbench.utils.env_utils import IS_BLACKWELL

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from tritonbench.utils.triton_utils import has_experimental_descriptor

from .tutorial import matmul as tutorial_matmul


torch._dynamo.config.recompile_limit = 10000

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

HAS_CUDA_129 = (
    torch.cuda.is_available() and torch.version.cuda and torch.version.cuda >= "12.9"
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="TritonBench fp8_gemm")
    parser.add_argument("--llama", action="store_true")
    parser.add_argument("--scaling-pair", type=str, default="TensorWise,TensorWise")
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--per-tensor-scale-a", type=float, default=None)
    parser.add_argument("--per-tensor-scale-b", type=float, default=None)
    return parser.parse_args(args)


def get_fp8_dtype():
    if torch.version.cuda:
        return torch.float8_e4m3fn
    elif torch.version.hip:
        if torch.cuda.get_device_capability() < (9, 5):
            return torch.float8_e4m3fnuz
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz


def get_scaling_recipe(scaling_recipe: str) -> int:
    if scaling_recipe == "TensorWise":
        return ScalingType.TensorWise
    elif scaling_recipe == "RowWise":
        return ScalingType.RowWise
    elif scaling_recipe == "BlockWise1x128":
        return ScalingType.BlockWise1x128
    elif scaling_recipe == "BlockWise128x128":
        return ScalingType.BlockWise128x128
    else:
        raise ValueError(f"Invalid scaling recipe: {scaling_recipe}")


def get_scale(
    x: torch.Tensor,
    scaling_recipe: ScalingType,
    transpose: bool = False,
    custom_scale: float = None,
) -> (torch.Tensor, torch.Tensor):
    def _get_scale_per_tensor(
        x: torch.Tensor, custom_scale: float = None
    ) -> (torch.Tensor, torch.Tensor):
        # For tensor-wise scaling, kernel requires a float32 scale tensor
        if custom_scale:
            return torch.tensor(custom_scale, dtype=torch.float32, device=x.device)
        scale = (torch.finfo(torch.float8_e4m3fn).max / x.abs().max()).reciprocal()
        x *= scale
        return x, scale.to(torch.float32)

    def _get_scale_per_row(
        x: torch.Tensor, transpose: bool = False
    ) -> (torch.Tensor, torch.Tensor):
        if transpose:  # scale_b.shape should be [1, N]
            scale = (
                torch.finfo(torch.float8_e4m3fn).max
                / x.abs().max(dim=0, keepdim=True).values
            ).reciprocal()
        else:  # scale_a.shape should be [M, 1]
            scale = (
                torch.finfo(torch.float8_e4m3fn).max
                / x.abs().max(dim=1, keepdim=True).values
            ).reciprocal()
        x = x.mul(scale)
        return x, scale.to(
            torch.float32
        )  # For row-wise scaling, kernel requires a float32 scale tensor

    def _get_scale_per_block(
        x: torch.Tensor, block_outer: int, block_inner: int
    ) -> (torch.Tensor, torch.Tensor):
        x = x.unflatten(1, (-1, block_inner)).unflatten(0, (-1, block_outer))
        amax = x.abs().amax(dim=[1, 3], keepdim=True).float()
        scale = (
            torch.finfo(torch.float8_e4m3fn).max / amax
        ).reciprocal()  # keeps scale small enough such that scaling doesn't cause inf values
        x = (
            x.mul(scale).flatten(2, 3).flatten(0, 1)
        )  # scale input up to dynamic range of float8_e4m3fn
        scale = scale.flatten(2, 3).flatten(0, 1)

        if block_outer == 1 and block_inner == 128:
            scale = (
                scale.t().contiguous().t()
            )  # 1x128 blocks need scales to be outer-dim-major

        return x, scale.to(torch.float32)

    match scaling_recipe:
        case ScalingType.TensorWise:
            return _get_scale_per_tensor(x, custom_scale=custom_scale)
        case ScalingType.RowWise:
            return _get_scale_per_row(x, transpose=transpose)
        case ScalingType.BlockWise1x128:
            return _get_scale_per_block(x, 1, 128)
        case ScalingType.BlockWise128x128:
            return _get_scale_per_block(x, 128, 128)
        case _:
            raise AssertionError(f"Unsupported scaling type {scaling_recipe}")


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "gbps", "latency"]
    DEFAULT_PRECISION = "fp8"
    FWD_ONLY = True

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.extra_args = parse_args(extra_args)

        self.fp8_dtype = get_fp8_dtype()

        scaling_recipe_a, scaling_recipe_b = self.extra_args.scaling_pair.split(",")
        if (scaling_recipe_a, scaling_recipe_b) not in [
            (a.name, b.name) for a, b in scaling_pairs
        ]:
            raise ValueError(
                f"Invalid scaling pair: {scaling_recipe_a}, {scaling_recipe_b}. See torch/_inductor/kernel/mm.py::scaling_pairs for valid pairs."
            )
        self.scaling_recipe_a = get_scaling_recipe(scaling_recipe_a)
        self.scaling_recipe_b = get_scaling_recipe(scaling_recipe_b)

        blockwise_scaling_types = [
            ScalingType.BlockWise1x128,
            ScalingType.BlockWise128x128,
        ]
        self.contains_blockwise_scaling = (
            self.scaling_recipe_a in blockwise_scaling_types
            or self.scaling_recipe_b in blockwise_scaling_types
        )

        self.use_fast_accum = (
            False if self.contains_blockwise_scaling else True
        )  # BlockWise scaled_gemm does not support use_fast_accum=True

    def _get_dtype(self):
        if (
            self.scaling_recipe_a == ScalingType.TensorWise
            and self.scaling_recipe_b == ScalingType.TensorWise
        ):
            return torch.float16
        return torch.bfloat16

    def get_input_iter(self):
        def args(m, n, k):
            a = torch.randn(m, k, device=self.device).to(self._get_dtype())
            b = torch.randn(n, k, device=self.device).to(self._get_dtype())

            a, scale_a = get_scale(
                a,
                self.scaling_recipe_a,
                custom_scale=self.extra_args.per_tensor_scale_a,
            )
            b, scale_b = get_scale(
                b,
                self.scaling_recipe_b,
                custom_scale=self.extra_args.per_tensor_scale_b,
            )

            # Kernels expect dtype=float8_e4m3fn(uz)
            a = a.to(self.fp8_dtype)
            b = b.to(self.fp8_dtype)

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
        assert (
            not self.contains_blockwise_scaling or HAS_CUDA_129
        ), "BlockWise scaling variants for scaled_gemm require CUDA 12.9+"

        return lambda: torch._scaled_mm(
            a,
            b.t(),
            scale_a,
            scale_b.t(),
            use_fast_accum=self.use_fast_accum,
            out_dtype=self._get_dtype(),
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
                a,
                b.t(),
                scale_a,
                scale_b.t(),
                use_fast_accum=self.use_fast_accum,
                out_dtype=self._get_dtype(),
            )
            compiled = torch.compile(f, dynamic=False)
            compiled(a, b)

        return lambda: compiled(a, b)

    if IS_BLACKWELL:

        @register_benchmark(enabled=True)
        def blackwell_persistent_tma_fp8_gemm(self, a, b, scale_a, scale_b):
            if self.scaling_recipe_a == self.scaling_recipe_b == ScalingType.TensorWise:
                scaling_recipe_int = 0
            elif self.scaling_recipe_a == self.scaling_recipe_b == ScalingType.RowWise:
                scaling_recipe_int = 1
            else:
                raise ValueError(
                    f"Invalid scaling pair: {self.scaling_recipe_a}, {self.scaling_recipe_b} for blackwell_persistent_tma_fp8_gemm."
                )
            return lambda: blackwell_persistent_tma(
                a,
                b,
                scale_a,
                scale_b,
                self._get_dtype(),
                scaling_recipe_int,
            )

        @register_benchmark(enabled=True)
        def blackwell_pt2_fp8_gemm(self, a, b, scale_a, scale_b):
            torch._dynamo.reset()
            with inductor_config.patch(
                max_autotune=True,
                max_autotune_gemm_backends="TRITON",
                autotune_fallback_to_aten=False,
            ):
                f = lambda a, b: torch._scaled_mm(
                    a,
                    b.t(),
                    scale_a,
                    scale_b.t(),
                    use_fast_accum=True,
                    out_dtype=self._get_dtype(),
                )
                compiled = torch.compile(f, dynamic=False)
                compiled(a, b)

            return lambda: compiled(a, b)

    @register_benchmark()
    def triton_fp8_gemm(self, a, b, scale_a, scale_b):
        return lambda: tutorial_matmul(a, b.t())

    @register_benchmark(enabled=HAS_TMA)
    def triton_persistent_fp8_gemm(self, a, b, scale_a, scale_b):
        return lambda: matmul_persistent(a, b.t())

    @register_benchmark(enabled=HAS_TMA and has_experimental_descriptor())
    def triton_tma_persistent_fp8_gemm(self, a, b, scale_a, scale_b):
        b = b.T.contiguous()
        c, desc_a, desc_b, desc_c = allocate_matmul_tma(a, b.t())
        return lambda: matmul_tma_persistent(a, b.t(), c, desc_a, desc_b, desc_c)

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
