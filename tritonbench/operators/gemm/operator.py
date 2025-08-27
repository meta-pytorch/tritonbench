import argparse
import contextlib
import csv
import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
import torch._inductor.config as inductor_config
import triton

from tritonbench.operators.gemm.kernels import matmul as kernels
from tritonbench.operators.gemm.partition_k import matmul_partition_k
from tritonbench.operators.gemm.stream_k import streamk_amd_matmul, streamk_cuda_matmul
from tritonbench.operators.gemm.warp_spec_persistent_matmul import (
    blackwell_matmul_descriptor_persistent,
    blackwell_matmul_tma,
    blackwell_matmul_tma_persistent,
)
from tritonbench.utils.triton_utils import has_tlx

if has_tlx():
    from tritonbench.operators.gemm.tlx_matmul import tlx_matmul as _tlx_matmul
else:

    def _tlx_matmul(*args, **kwargs):
        raise RuntimeError("TLX not available in this Triton version")


from tritonbench.utils.data_utils import get_production_shapes
from tritonbench.utils.env_utils import (
    get_nvidia_gpu_model,
    is_cuda,
    is_fbcode,
    supports_tma,
)

from tritonbench.utils.path_utils import REPO_PATH

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    llama_shapes,
    PRECISION_DTYPE_MAPPING,
    register_benchmark,
    register_metric,
    register_x_val,
)

try:
    from tritonbench.operators.gemm.persistent_matmul import (
        matmul_persistent,
        matmul_tma_persistent,
        matmul_tma_persistent_cached,
    )

    HAS_PERSISTENT = True
except ModuleNotFoundError:
    HAS_PERSISTENT = False

from tritonbench.operators.gemm.triton_matmul import matmul as triton_tutorial_matmul

if is_fbcode():
    import generative_recommenders.ops.triton.triton_addmm as hstu_triton_addmm

    # without this set we can only pick a single config for AMD, Nvidia has 8
    # with this set AMD will pick from 256 different configs (not the actual full
    # tuning space, so some perf may be left on the table)
    hstu_triton_addmm.ENABLE_FULL_TURNING_SPACE = True
    from hammer.ops.triton.triton_matmul import (
        triton_matmul as hstu_triton_matmul_kernel,
    )

    HAS_HAMMER = True
else:
    HAS_HAMMER = False

BUILDIN_SHAPES = [
    (256, 256, 256, None),
    (384, 384, 384, None),
    (512, 512, 512, None),
    (640, 640, 640, None),
    (768, 768, 768, None),
    (896, 896, 896, None),
    (1024, 1024, 1024, None),
    (1152, 1152, 1152, None),
    (1280, 1280, 1280, None),
    (1408, 1408, 1408, None),
    (1536, 1536, 1536, None),
    (1664, 1664, 1664, None),
    (1792, 1792, 1792, None),
    (1920, 1920, 1920, None),
    (2048, 2048, 2048, None),
    (2176, 2176, 2176, None),
    (2304, 2304, 2304, None),
    (2432, 2432, 2432, None),
    (2560, 2560, 2560, None),
    (2688, 2688, 2688, None),
    (2816, 2816, 2816, None),
    (2944, 2944, 2944, None),
    (3072, 3072, 3072, None),
    (3200, 3200, 3200, None),
    (3328, 3328, 3328, None),
    (3456, 3456, 3456, None),
    (3584, 3584, 3584, None),
    (3712, 3712, 3712, None),
    (3840, 3840, 3840, None),
    (3968, 3968, 3968, None),
    (4096, 4096, 4096, None),
]

SPLIT_K_SHAPES = [
    (m, m, k, None)
    for m in [16 * i for i in range(1, 5)]
    for k in [4096 * i for i in range(1, 9)]
]

IS_B200 = is_cuda() and get_nvidia_gpu_model() == "NVIDIA B200"


@contextlib.contextmanager
def set_env_variable(key, value):
    """Context manager to temporarily set an environment variable."""
    original = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original is not None:
            os.environ[key] = original
        else:
            del os.environ[key]


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TritonBench Gemm operator Benchmark")
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--bias", type=int)
    parser.add_argument("--input", type=str)
    parser.add_argument("--splitk", action="store_true", default=False)
    parser.add_argument("--llama", action="store_true", default=False)
    parser.add_argument("--buffer-ops", action="store_true", default=False)
    parser.add_argument("--layout", type=str, default="tn")
    parser.add_argument(
        "--verbose-autotune",
        action="store_true",
        help="Being verbose with autotuning results",
    )
    args = parser.parse_args(args)
    return args


def read_shapes_from_csv(csv_path: str) -> List[List[int]]:
    input_file_path = os.path.join(
        REPO_PATH, "tritonbench", "operators", "gemm", csv_path
    )
    shapes = []
    with open(input_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape = [
                int(row.get(f)) if row.get(f) else None for f in ("M", "N", "K", "Bias")
            ]
            shapes.append(shape)
    return shapes


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "speedup", "tflops"]
    DEFAULT_PRECISION = "fp16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        gemm_args = parse_args(self.extra_args)
        self.layout = gemm_args.layout
        self.inductor_autotune_num_choices_displayed = (
            None
            if gemm_args.verbose_autotune
            else inductor_config.autotune_num_choices_displayed
        )
        if gemm_args.input:
            self.shapes = read_shapes_from_csv(gemm_args.input)
        elif gemm_args.splitk:
            self.shapes = SPLIT_K_SHAPES
        elif gemm_args.llama:
            self.shapes = llama_shapes()
        elif gemm_args.m and gemm_args.k and gemm_args.n:
            self.shapes = [(gemm_args.m, gemm_args.n, gemm_args.k, gemm_args.bias)]
        else:
            self.shapes = BUILDIN_SHAPES

        if is_fbcode() and tb_args.production_shapes:
            additional_shapes = get_production_shapes(
                self.name, f"{tb_args.precision}_gemm", self.tb_args.shuffle_shapes
            )
            if len(additional_shapes):  # only append if not empty
                self.shapes.append(
                    get_production_shapes(
                        self.name,
                        f"{tb_args.precision}_gemm",
                        self.tb_args.shuffle_shapes,
                    )
                )

        self.use_buffer_ops = gemm_args.buffer_ops

        if self.use_buffer_ops and torch.version.hip is None:
            raise ValueError("Buffer ops are only supported on AMD GPUs.")

    @register_benchmark()
    def triton_tutorial_matmul(self, a, b, bias) -> Callable:
        if bias is not None:
            return lambda: triton_tutorial_matmul(a, b) + bias
        else:
            return lambda: triton_tutorial_matmul(a, b)

    @register_benchmark()
    def matmul_partition_k(self, a, b, bias) -> Callable:
        bt = b.contiguous()
        if bias is not None:
            return lambda: matmul_partition_k(a, bt) + bias
        else:
            return lambda: matmul_partition_k(a, bt)

    @register_benchmark(enabled=HAS_PERSISTENT)
    def triton_persistent_matmul(self, a, b, bias) -> Callable:
        if bias is not None:
            return lambda: matmul_persistent(a, b) + bias
        else:
            return lambda: matmul_persistent(a, b)

    @register_benchmark(enabled=not is_fbcode() and HAS_PERSISTENT and supports_tma())
    def triton_tma_persistent_matmul(self, a, b, bias) -> Callable:
        b = b.T.contiguous()
        if bias is not None:
            return lambda: matmul_tma_persistent(a, b) + bias
        else:
            return lambda: matmul_tma_persistent(a, b)

    @register_benchmark(enabled=not is_fbcode() and HAS_PERSISTENT and supports_tma())
    def triton_tma_persistent_cached_matmul(self, a, b, bias) -> Callable:
        b = b.T.contiguous()
        if bias is not None:
            return lambda: matmul_tma_persistent_cached(a, b) + bias
        else:
            return lambda: matmul_tma_persistent_cached(a, b)

    @register_benchmark(enabled=is_cuda())
    def triton_ops_matmul(self, a, b, bias) -> Callable:
        # kwargs are not allowed in torch autograd functions, so passing
        # in as parameter is messy. Instead, we set env var and extract
        # it in the triton kernel call

        def func():
            with set_env_variable(
                "AMDGCN_USE_BUFFER_OPS", "1" if self.use_buffer_ops else "0"
            ):
                if bias is not None:
                    return kernels.matmul(a, b) + bias
                else:
                    return kernels.matmul(a, b)

        return func

    @register_benchmark(baseline=True)
    def aten_matmul(self, a, b, bias) -> Callable:
        if bias is not None:
            return lambda: torch.matmul(a, b) + bias
        else:
            return lambda: torch.matmul(a, b)

    @register_benchmark()
    def aten_tunableop_matmul(self, a, b, bias) -> Callable:
        is_enabled = torch.cuda.tunable.is_enabled()

        def op():
            torch.cuda.tunable.enable(True)
            output = (
                torch.matmul(a, b) + bias if bias is not None else torch.matmul(a, b)
            )
            torch.cuda.tunable.enable(is_enabled)
            return output

        torch.cuda.tunable.enable(True)

        # trigger tuning
        op()

        return op

    @register_benchmark(enabled=HAS_HAMMER)
    def hstu_triton_matmul(self, a, b, bias) -> Callable:
        if bias is not None:
            return lambda: hstu_triton_matmul_kernel(a, b) + bias
        else:
            return lambda: hstu_triton_matmul_kernel(a, b)

    @register_benchmark()
    def pt2_triton_matmul(self, a, b, bias) -> Callable:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            autotune_fallback_to_aten=False,
            autotune_num_choices_displayed=self.inductor_autotune_num_choices_displayed,
        ):
            if bias is not None:
                f = lambda a, b: a.matmul(b) + bias
            else:
                f = lambda a, b: a.matmul(b)
            compiled = torch.compile(f, dynamic=False)
            compiled(a, b)

        return lambda: compiled(a, b)

    @register_benchmark(enabled=False)
    def pt2_matmul_maxautotune(self, a, b, bias) -> Callable:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="ATEN,TRITON",
            autotune_num_choices_displayed=self.inductor_autotune_num_choices_displayed,
        ):
            if bias is not None:
                f = lambda a, b: a.matmul(b) + bias
            else:
                f = lambda a, b: a.matmul(b)
            compiled = torch.compile(f, dynamic=False)
            compiled(a, b)

        return lambda: compiled(a, b)

    @register_benchmark(enabled=not is_cuda())
    def streamk_matmul(self, a, b, bias) -> Callable:
        return lambda: (
            streamk_amd_matmul(a, b, bias) if bias else streamk_amd_matmul(a, b)
        )

    @register_benchmark(enabled=is_cuda())
    def streamk_matmul(self, a, b, bias) -> Callable:
        print(f"Testing shape: {a.shape} x {b.shape}...")
        streamk = torch.matmul(a, b)
        b = b.T.contiguous()
        baseline = streamk_cuda_matmul(a, b)
        if not torch.allclose(streamk, baseline):
            print(
                f"StreamK matmul on {a.shape} x {b.shape} result does not match baseline matmul result. Max abs(streamk/baseline - 1):  {torch.max(torch.abs(streamk / baseline - 1))}"
            )
        return lambda: (
            streamk_cuda_matmul(a, b) + bias if bias else streamk_cuda_matmul(a, b)
        )

    @register_benchmark(enabled=is_cuda())
    def pt2_cutlass_matmul(self, a, b, bias) -> Callable:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="CUTLASS",
            autotune_fallback_to_aten=False,
        ):
            if bias is not None:
                f = lambda a, b: a.matmul(b) + bias
            else:
                f = lambda a, b: a.matmul(b)
            # cutlass needs to know the static shape, so set dynamic to False
            compiled = torch.compile(f, dynamic=False)
            compiled(a, b)
        return lambda: compiled(a, b)

    @register_benchmark(enabled=False)
    def matmul_decompose_k(self, a, b, bias) -> Callable:
        def decompose_func(a_in, b_in):
            M, K = a_in.shape
            K, N = b_in.shape

            # TODO: Ideally we want to autotune over this parameter
            kPartitions = 256
            assert K % kPartitions == 0, "K must be divisible by Kmini"
            B = K // kPartitions

            a_reshaped = a.reshape(M, B, kPartitions).transpose(
                0, 1
            )  # Shape: (B, M, kPartitions)
            b_reshaped = b.reshape(B, kPartitions, N)  # Shape: (B, kPartitions, N)
            result = torch.bmm(a_reshaped, b_reshaped).to(
                torch.float32
            )  # Shape: (B, M, N)
            return result.sum(dim=0)  # Sum over B dimension, Shape: (M, N)

        compiled_decompose_k = torch.compile(decompose_func)
        compiled_decompose_k(a, b)
        if bias is not None:
            return lambda: compiled_decompose_k(a, b) + bias
        else:
            return lambda: compiled_decompose_k(a, b)

    if IS_B200:

        @register_benchmark(enabled=False)
        def triton_blackwell_warpspec_persistent_matmul(self, a, b, bias) -> Callable:
            if bias is not None:
                return (
                    lambda: blackwell_matmul_tma_persistent(a, b, warp_specialize=True)
                    + bias
                )
            else:
                return lambda: blackwell_matmul_tma_persistent(
                    a, b, warp_specialize=True
                )

        @register_benchmark(enabled=False)
        def triton_blackwell_persistent_matmul(self, a, b, bias) -> Callable:
            if bias is not None:
                return (
                    lambda: blackwell_matmul_tma_persistent(a, b, warp_specialize=False)
                    + bias
                )
            else:
                return lambda: blackwell_matmul_tma_persistent(
                    a, b, warp_specialize=False
                )

        @register_benchmark(enabled=False)
        def triton_blackwell_warpspec_tma_matmul(self, a, b, bias) -> Callable:
            if bias is not None:
                return lambda: blackwell_matmul_tma(a, b, warp_specialize=True) + bias
            else:
                return lambda: blackwell_matmul_tma(a, b, warp_specialize=True)

        @register_benchmark(enabled=False)
        def triton_blackwell_tma_matmul(self, a, b, bias) -> Callable:
            if bias is not None:
                return lambda: blackwell_matmul_tma(a, b, warp_specialize=False) + bias
            else:
                return lambda: blackwell_matmul_tma(a, b, warp_specialize=False)

        @register_benchmark(enabled=False)
        def triton_blackwell_warpspec_descriptor_matmul(self, a, b, bias) -> Callable:
            if bias is not None:
                return (
                    lambda: blackwell_matmul_descriptor_persistent(
                        a, b, warp_specialize=True
                    )
                    + bias
                )
            else:
                return lambda: blackwell_matmul_descriptor_persistent(
                    a, b, warp_specialize=True
                )

        @register_benchmark(enabled=False)
        def triton_blackwell_descriptor_matmul(self, a, b, bias) -> Callable:
            if bias is not None:
                return (
                    lambda: blackwell_matmul_descriptor_persistent(
                        a, b, warp_specialize=False
                    )
                    + bias
                )
            else:
                return lambda: blackwell_matmul_descriptor_persistent(
                    a, b, warp_specialize=False
                )

        @register_benchmark(enabled=False)
        def tlx_matmul(self, a, b, bias) -> Callable:
            if bias is not None:
                return lambda: _tlx_matmul(a, b) + bias
            else:
                return lambda: _tlx_matmul(a, b)

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        # x-value: computation intensity
        a, w, bias = example_inputs
        m, k = a.size()
        k, n = w.size()
        return (m, n, k)

    @register_metric()
    def gbps(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        a, w, bias = example_inputs
        numel = a.numel() + w.numel() + (torch.mm(a, w).numel())
        numel = numel * a.element_size() / 1e9
        return numel / metrics.latency * 1e3

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        a, w, bias = example_inputs
        m, k = a.size()
        k, n = w.size()
        if bias is not None:
            flops = m * k * 2 * n + 2 * m * n
        else:
            flops = m * k * 2 * n
        return flops / metrics.latency / 1e12 * 1e3

    @staticmethod
    def _scaled_randn(*args, scale: float, **kwargs) -> torch.Tensor:
        """
        This provides more numerically stable inputs for GEMMs. The +1
        eliminates very small values that could result in denormals, and the
        scale (which should be set to K in an M*N*K GEMM) reduces the size of
        the absolute error.

        In particular, for a given element in the output tensor, the cumulative
        error is eps * 2 * K, where eps is the smallest precision representable
        in the dtype. By scaling the element by K, we avoid the error growing
        with the size of the tensor.
        """
        return (torch.randn(*args, **kwargs) + 1) / scale

    def get_input_iter(self) -> Generator:
        for shape_id, shape in enumerate(self.shapes):
            if len(shape) == 4:
                m, n, k, bias = shape
            elif len(shape) == 3:
                m, n, k = shape
                bias = None
            else:
                raise ValueError(f"Invalid shape {shape}")
            if hasattr(self, "dtypes") and self.dtypes:
                self.tb_args.precision = "bypass"
                self.dtype = PRECISION_DTYPE_MAPPING[self.dtypes[shape_id]]
            if hasattr(self, "strides"):
                strides = self.strides[shape_id]
                assert (
                    len(strides) == 2
                ), f"Can only have 2 strides from input, get: {strides}"
                assert (
                    len(strides[0]) == 2 and len(strides[1]) == 2
                ), f"Can only deal with 2D strides, get: {strides}"
                # The shape might from a tensor view, which is different from the original shape
                # Try to infer the original shape from both shape and strides
                actual_m = max(m, strides[0][1])
                actual_k = max(k, strides[0][0], strides[1][1])
                actual_n = max(n, strides[1][0])
                a = self._scaled_randn(
                    (actual_m, actual_k), scale=k, device=self.device, dtype=self.dtype
                )
                w = self._scaled_randn(
                    (actual_k, actual_n), scale=k, device=self.device, dtype=self.dtype
                )
                a = a.as_strided(size=[m, k], stride=strides[0])
                w = w.as_strided(size=[k, n], stride=strides[1])
            else:
                a = self._scaled_randn(
                    (m, k), scale=k, device=self.device, dtype=self.dtype
                )
                w = self._scaled_randn(
                    (k, n), scale=k, device=self.device, dtype=self.dtype
                )
                # Convert inputs to column-major if layout is "n" (non-transposed)
                if self.layout[0] == "n":
                    a = a.T.contiguous().T
                if self.layout[1] == "n":
                    w = w.T.contiguous().T
            if not bias == None:
                bias = torch.randn(
                    (bias), device=self.device, dtype=self.dtype
                ).requires_grad_(False)

            yield a, w, bias

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        # Float atomics introduce non-determinism for some GEMMs (e.g., Stream-K)
        # So we use a slightly larger tolerance here.
        return torch.allclose(output, baseline_output, atol=1e-5, rtol=0.5)

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
                    "aten_matmul",
                    "triton_tutorial_matmul",
                    "triton_kernels_matmul",
                    "hstu_triton_matmul",
                ],  # possible values for `line_arg``
                line_names=[
                    "ATen GEMM",
                    "Triton Tutorial GEMM",
                    "triton/kernels/matmul",
                    "HSTU Triton GEMM",
                ],  # label name for the lines
                styles=[
                    ("blue", "-"),
                    ("green", "-"),
                    ("red", "-"),
                    ("yellow", "-"),
                ],  # line styles
                ylabel="tflops",  # label name for the y-axis
                plot_name="gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(m, n, k, provider):
            tflops = self.output.get_y_vals((m, n, k), provider, "tflops")
            return tflops

        save_path = "/tmp/test_gemm"

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
