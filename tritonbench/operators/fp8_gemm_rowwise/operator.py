import argparse
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
import triton

from tritonbench.utils.data_utils import get_production_shapes

from tritonbench.utils.env_utils import get_nvidia_gpu_model, is_cuda, is_hip

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    gemm_shapes,
    register_benchmark,
    register_metric,
    register_x_val,
)

from .aoti_fp8_triton_mm import aoti_fp8_triton_mm


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TorchBench fp8 gemm rowwise operator Benchmark"
    )
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--llama", action="store_true")
    parser.add_argument("--prefill", default=False, action="store_true")
    parser.add_argument(
        "--no_fp8_fast_accum", dest="fp8_fast_accum", action="store_false"
    )
    parser.add_argument(
        "--no_use_tma", dest="use_tma", default=None, action="store_false"
    )
    parser.add_argument("--use_tma", dest="use_tma", action="store_true")
    parser.add_argument(
        "--use_persistent",
        dest="no_use_persistent",
        default=None,
        action="store_false",
    )
    parser.add_argument(
        "--no_use_persistent",
        dest="no_use_persistent",
        action="store_true",
    )
    parser.add_argument("--warp_specialization", action="store_true")
    parsed_args = parser.parse_args(args)
    if parsed_args.use_tma is None:
        # Default to True for CUDA, False for ROCm
        parsed_args.use_tma = True if torch.version.hip is None else False
    if parsed_args.no_use_persistent is None:
        # Default to True for ROCm, False for CUDA
        parsed_args.no_use_persistent = True if torch.version.hip is not None else False
    if torch.version.hip is not None:
        if parsed_args.use_tma:
            raise RuntimeError("TMA is not supported on ROCm")
        if parsed_args.warp_specialization:
            parsed_args.warp_specialization = False
            print("Warp specialization is not supported on ROCm defaulting to False")
    return parsed_args


HAS_TRITON = False
HAS_CUTLASS_OR_CK = False
HAS_CUBLAS = False

try:
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        get_fp8_constants as get_fp8_constants,
        matmul_fp8_row as triton_fp8_row,
    )

    if not torch.version.hip:
        assert hasattr(
            triton.runtime.driver.active.utils, "fill_1d_tma_descriptor"
        ), "TMA is required by the Triton kernel."
    HAS_TRITON = True
except (ImportError, AssertionError):
    HAS_TRITON = False

try:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401

    cutlass_or_ck_fp8_row = torch.ops.fbgemm.f8f8bf16_rowwise
    # TODO: remove these b200 hacks.
    HAS_CUTLASS_OR_CK = is_hip() or (
        is_cuda() and get_nvidia_gpu_model() != "NVIDIA B200"
    )
except (ImportError, AttributeError, FileNotFoundError):
    HAS_CUTLASS_OR_CK = False

try:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401

    cublas_fp8_row = torch.ops.fbgemm.f8f8bf16_cublas
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import scale_fp8_row

    # TODO: remove these b200 hacks.
    HAS_CUBLAS = is_cuda() and get_nvidia_gpu_model() != "NVIDIA B200"
except (ImportError, IOError, AttributeError, FileNotFoundError):
    HAS_CUBLAS = False


BUILDIN_SHAPES = [
    (1, 2304, 2048),
    (1, 8192, 16384),
    (4, 4096, 2304),
    (4, 13312, 2048),
    (8, 2304, 2304),
    (8, 8192, 6656),
    (16, 4096, 6656),
    (16, 13312, 13312),
    (32, 2304, 16384),
    (32, 8192, 13312),
    (64, 4096, 2048),
    (64, 13312, 2048),
    (128, 2304, 6656),
    (128, 8192, 2304),
    (2048, 8192, 2048),
    (2048, 13312, 6656),
    (4096, 2304, 13312),
    (4096, 13312, 2304),
    (16384, 4096, 16384),
    (16384, 8192, 13312),
]

FP8_DTYPE, _, _, _ = get_fp8_constants()
E4M3_MAX_POS: float = torch.finfo(FP8_DTYPE).max
EPS: float = 1e-12
FP16_MAX_POS: float = torch.finfo(torch.float16).max


def fp8_row_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Quantize an input tensor and return the fp8 tensor and its inverse scale.
    x_row_max = torch.max(torch.abs(x), dim=1).values
    scale = E4M3_MAX_POS / torch.clamp(x_row_max, EPS)
    if x.dtype is torch.float16:
        scale = torch.clamp(scale, max=FP16_MAX_POS)
    xq = torch.clamp(x * scale[:, None], min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS).to(
        FP8_DTYPE
    )
    return xq, scale.reciprocal().to(torch.float32)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "speedup", "accuracy"]
    DEFAULT_PRECISION = "fp8"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.use_cuda_graphs = True
        addmm_args = parse_args(self.extra_args)
        if addmm_args.m and addmm_args.n and addmm_args.k:
            self.shapes = [(addmm_args.m, addmm_args.n, addmm_args.k)]
        elif addmm_args.llama:
            self.shapes = gemm_shapes(addmm_args.prefill)
        else:
            self.shapes = BUILDIN_SHAPES
        if hasattr(tb_args, "production_shapes") and tb_args.production_shapes:
            extras = get_production_shapes(self.name, "fp32_gemm")
            if len(extras):
                self.shapes.extend(extras)
        self.bias = addmm_args.bias
        self.fp8_fast_accum = addmm_args.fp8_fast_accum
        self.use_tma = addmm_args.use_tma
        self.no_use_persistent = addmm_args.no_use_persistent
        self.warp_specialization = addmm_args.warp_specialization

        # This is the only config currently tested in b200.
        # TODO: Remove it when the other variants are supported.
        if is_cuda() and get_nvidia_gpu_model() == "NVIDIA B200":
            self.fp8_fast_accum = True
            self.use_tma = True
            self.no_use_persistent = False
            self.warp_specialization = False
            self.shapes = (
                gemm_shapes(addmm_args.prefill)
                if (addmm_args.llama)
                else BUILDIN_SHAPES
            )

    @register_benchmark(enabled=HAS_TRITON)
    def _triton(self, xq, wq, x_scale, w_scale, bias) -> Callable:
        return lambda: triton_fp8_row(
            xq,
            wq,
            x_scale,
            w_scale,
            bias=bias,
            fp8_fast_accum=self.fp8_fast_accum,
            tma_persistent=self.use_tma,
            no_use_persistent=self.no_use_persistent,
            use_warp_specialization=self.warp_specialization,
        )

    @register_benchmark(enabled=HAS_TRITON)
    def _aoti_fp8_triton_mm(self, xq, wq, x_scale, w_scale, bias) -> Callable:
        return lambda: aoti_fp8_triton_mm(
            xq,
            x_scale,
            wq,
            w_scale,
            bias=bias,
            fp8_fast_accum=self.fp8_fast_accum,
        )

    @register_benchmark(
        enabled=HAS_CUTLASS_OR_CK,
        label="ck" if torch.version.hip else "cutlass",
        baseline=True,
    )
    def _cutlass_or_ck(self, xq, wq, x_scale, w_scale, bias) -> Callable:
        if bias is not None:
            return (
                lambda: cutlass_or_ck_fp8_row(
                    xq, wq, x_scale, w_scale, use_fast_accum=self.fp8_fast_accum
                )
                + bias
            )
        else:
            return lambda: cutlass_or_ck_fp8_row(
                xq, wq, x_scale, w_scale, use_fast_accum=self.fp8_fast_accum
            )

    @register_benchmark(enabled=HAS_CUBLAS, label=f"cublas_{torch.version.cuda}")
    def _cublas(self, xq, wq, x_scale, w_scale, bias) -> Callable:
        if bias is not None:
            return (
                lambda: scale_fp8_row(
                    cublas_fp8_row(xq, wq, use_fast_accum=self.fp8_fast_accum),
                    x_scale,
                    w_scale,
                )
                + bias
            )
        else:
            return lambda: scale_fp8_row(
                cublas_fp8_row(xq, wq, use_fast_accum=self.fp8_fast_accum),
                x_scale,
                w_scale,
            )

    # TODO: add cublas rowwise FP8 kernel
    # @register_benchmark(baseline=True)
    # def _torch(self, xq, wq, x_scale, w_scale) -> Callable:
    #     def _cublass(xq, wq, x_scale, w_scale):
    #         output, _ = torch._scaled_mm(
    #             xq, wq.T, use_fast_accum=True, out_dtype=torch.bfloat16
    #         )
    #         return output * x_scale[:, None] * w_scale[None, :]

    #     return lambda: _cublass(xq, wq, x_scale, w_scale)

    @register_metric()
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> List[float]:
        xq, wq, _, _, bias = example_inputs
        m, k = xq.size()
        n, k = wq.size()
        if bias is not None:
            flops = m * k * 2 * n + 2 * m * n
        else:
            flops = m * k * 2 * n
        return flops

    @register_metric()
    def gbps(self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics) -> float:
        def nbytes(t):
            return t.numel() * t.element_size()

        a, b, _, _, _ = example_inputs
        c = fn()
        c = c[0] if isinstance(c, tuple) else c

        gb = (nbytes(a) + nbytes(b) + nbytes(c)) / 1e9
        return gb / metrics.latency * 1e3

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        xq, wq, _, _, _ = example_inputs
        m, k = xq.size()
        n, k = wq.size()
        return (m, n, k)

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            m, n, k = shape
            x = torch.randn(
                (m, k), device=self.device, dtype=torch.bfloat16
            ).requires_grad_(False)
            w = torch.randn(
                (n, k), device=self.device, dtype=torch.bfloat16
            ).requires_grad_(False)
            xq, x_scale = fp8_row_quantize(x)
            wq, w_scale = fp8_row_quantize(w)
            if self.bias:
                bias = torch.randn(
                    (n), device=self.device, dtype=torch.bfloat16
                ).requires_grad_(False)
            else:
                bias = None
            yield xq, wq, x_scale, w_scale, bias

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        accuracy = True
        try:
            torch.testing.assert_close(output, baseline_output, atol=1e-2, rtol=0.5)
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
                    "_torch",
                    "_triton",
                    "_ck" if torch.version.hip else "_cutlass",
                    "_cublas",
                ],  # possible values for `line_arg``
                line_names=[
                    "Torch",
                    "Triton",
                    "CK" if torch.version.hip else "Cutlass",
                    "cuBLAS",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-"), ("yellow", "-")],  # line styles
                ylabel="tflops",  # label name for the y-axis
                plot_name="gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(density, provider):
            tflops = self.output.get_y_vals(density, provider, "tflops")
            return tflops

        save_path = "/tmp/test_fp8_gemm_rowwise"

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
