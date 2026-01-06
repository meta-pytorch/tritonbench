import logging

from importlib.metadata import PackageNotFoundError, version
from itertools import accumulate
from typing import Any, Generator, List, Tuple

import torch
from torch._inductor import config as inductor_config
from torch._inductor.utils import ensure_cute_available
from tritonbench.utils.env_utils import IS_BLACKWELL, is_cuda, is_fbcode
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

if is_fbcode():
    from tritonbench.utils.fb.grouped_gemm import get_fb_shapes

logger = logging.getLogger(__name__)


if ensure_cute_available():
    import cutlass

    from .cutedsl.kernels import compile_cutedsl_grouped_gemm, grouped_gemm_sm100_tuned

    # Set HAS_CUTEDSL to True if import succeeds
    HAS_CUTEDSL = True
else:
    logger.warning("Failed to import CuteDSL and/or Cutlass")
    HAS_CUTEDSL = False

try:
    CUTLASS_VERSION = version("nvidia-cutlass-dsl")
except PackageNotFoundError:
    CUTLASS_VERSION = "0.0.0"


from .kernels import triton_group_gemm_fn

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore

    HAS_TLX = True
except ImportError:
    # suppress type checking errors
    tlx = None

    HAS_TLX = False

if HAS_TLX:
    from .kernels import tlx_group_gemm_fn


def get_default_shapes():
    group_size = 4
    x_vals = [2**i for i in range(10, 14)]  # 128, 256, 512, 1024

    shapes = []
    for N in x_vals:
        M = K = N
        N_out = N
        A_shapes = [(M, K)] * group_size
        B_shape = (K, N_out)
        shapes.append((A_shapes, B_shape, None, None))

    return shapes


# TODO(nikhilap): Add a separate 3D grouped_gemm operator to alleviate the restriction that all B tensors must be the same.
class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["latency", "speedup", "accuracy"]
    FWD_ONLY = True

    @register_benchmark(baseline=True)
    def aten_grouped_mm(self, group_A, group_B, w=None, split=None):
        def _inner():
            A_packed, B_shared, offs = self.list_input_to_jagged(group_A, group_B)
            return torch._grouped_mm(A_packed, B_shared, offs=offs, bias=None)

        return _inner

    # Version of the ATen benchmark that doesn't time input preprocessing
    @register_benchmark()
    def preprocessed_aten_grouped_mm(self, group_A, group_B, w=None, split=None):
        A_packed, B_shared, offs = self.list_input_to_jagged(group_A, group_B)

        def _inner():
            return torch._grouped_mm(A_packed, B_shared, offs=offs, bias=None)

        return _inner

    @register_benchmark()
    def naive(self, group_A, group_B, w=None, split=None):
        b_shared = group_B[0]

        def _inner():
            outs = [torch.matmul(a, b_shared) for a in group_A]
            # TODO(nikhilap): consider removing this cat and handling packing outside timing if you want
            # a pure-matmul baseline without the extra copy kernel. Decide whether the
            # baseline should (a) include cat for end-to-end parity, or (b) exclude cat and
            # let the harness flatten for accuracy outside timing for a micro-kernel apples-to-apples.
            # Maybe consider only doing the cat if accuracy is a current metric.
            return torch.cat(outs, dim=0)

        return _inner

    # TODO: Does not work on hip
    @register_benchmark(enabled=is_cuda())
    def torch_compile_grouped_gemm(self, group_A, group_B, w=None, split=None):
        def _inner():
            torch._dynamo.reset()

            with inductor_config.patch(
                max_autotune=True,
                max_autotune_gemm_backends="TRITON",
                autotune_fallback_to_aten=False,
            ):
                A_packed, B_shared, offs = self.list_input_to_jagged(group_A, group_B)
                compiled = torch.compile(torch._grouped_mm, dynamic=False)
                return compiled(A_packed, B_shared, offs=offs, bias=None)

        return _inner

    # Version of the Inductor Triton benchmark that doesn't time input preprocessing
    # TODO: Does not work on hip
    @register_benchmark(enabled=is_cuda())
    def preprocessed_pt2_triton_grouped_mm(self, group_A, group_B, w=None, split=None):
        torch._dynamo.reset()

        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            autotune_fallback_to_aten=False,
        ):
            A_packed, B_shared, offs = self.list_input_to_jagged(group_A, group_B)
            compiled = torch.compile(torch._grouped_mm, dynamic=False)

        def _inner():
            return compiled(A_packed, B_shared, offs=offs, bias=None)

        return _inner

    @register_benchmark(
        enabled=HAS_CUTEDSL and IS_BLACKWELL,
        label=f"preprocessed_pt2_cute_grouped_mm-{CUTLASS_VERSION}",
    )
    def preprocessed_pt2_cute_grouped_mm(self, group_A, group_B, w=None, split=None):
        torch._dynamo.reset()

        A_packed, B_shared, offs = self.list_input_to_jagged(group_A, group_B)
        compiled = torch.compile(torch._grouped_mm, dynamic=False)

        def _inner():
            with inductor_config.patch(
                max_autotune=True,
                max_autotune_gemm_backends="CUTEDSL",
                autotune_fallback_to_aten=False,
                cutedsl_enable_autotuning=False,
                autotune_num_choices_displayed=None,
                max_autotune_gemm_search_space="DEFAULT",
            ):
                return compiled(A_packed, B_shared, offs=offs, bias=None)

        return _inner

    @register_benchmark()
    def triton_grouped_gemm(self, group_A, group_B, w=None, split=None):
        def _inner():
            (d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_C) = (
                self.list_input_to_triton_input(group_A, group_B)
            )
            outs = triton_group_gemm_fn(
                d_a_ptrs,
                d_b_ptrs,
                d_c_ptrs,
                d_g_sizes,
                d_g_lds,
                group_C,
                len(group_A),
                group_A[0].dtype,
            )
            return torch.cat(outs, dim=0)

        return _inner

    @register_benchmark(enabled=HAS_TLX and IS_BLACKWELL)
    def tlx_grouped_gemm(self, group_A, group_B, w=None, split=None):
        def _inner():
            (d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_C) = (
                self.list_input_to_triton_input(group_A, group_B)
            )
            outs = tlx_group_gemm_fn(
                d_a_ptrs,
                d_b_ptrs,
                d_c_ptrs,
                d_g_sizes,
                d_g_lds,
                group_C,
                len(group_A),
                group_A[0].dtype,
            )
            return torch.cat(outs, dim=0)

        return _inner

    # NOTE(nikhilap): These CuteDSL kernels are highly experimental, and certain design decisions may affect the accuracy of their measurements.
    # In the kernel below, it was decided to NOT include the time it takes to convert from Torch to Cute tensors and construct the tensors of
    # dims, strides, etc. that the kernel needs. Additionally, we are not including the compile time. This was done so we can measure raw kernel
    # performance. The Inductor implementation has not yet been fleshed out, but should hopefully hide some of the preprocessing steps we have chosen
    # to omit here.
    @register_benchmark(enabled=HAS_CUTEDSL and IS_BLACKWELL)
    def precompiled_cutedsl_grouped_mm(self, group_A, group_B, w=None, split=None):
        (
            compiled_grouped_gemm,
            initial_cute_tensors_abc,
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc,
            tensor_of_ptrs_abc,
            tensor_of_tensormap,
            current_stream,
            torch_tensors_abc,
        ) = compile_cutedsl_grouped_gemm(
            group_A,
            group_B,
            ab_dtype=cutlass.Float16
            if self.dtype == torch.float16
            else cutlass.BFloat16,
            c_dtype=cutlass.Float16
            if self.dtype == torch.float16
            else cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            a_major="m",
            b_major="n",
            c_major="m",
            mma_tiler_mn=(128, 256),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=False,
            tensormap_update_mode=cutlass.utils.TensorMapUpdateMode.SMEM,
            tolerance=0.5,
            warmup_iterations=0,
            iterations=1,
            skip_ref_check=True,
        )

        def _inner():
            compiled_grouped_gemm(
                initial_cute_tensors_abc[0],
                initial_cute_tensors_abc[1],
                initial_cute_tensors_abc[2],
                tensor_of_dim_size_mnkl,
                tensor_of_strides_abc,
                tensor_of_ptrs_abc,
                tensor_of_tensormap,
                current_stream,
            )

            outs = [C.squeeze(-1) for (_, _, C) in torch_tensors_abc]
            return torch.cat(outs, dim=0)

        return _inner

    @register_benchmark(enabled=HAS_CUTEDSL and IS_BLACKWELL)
    def cutedsl_grouped_mm(self, group_A, group_B, w=None, split=None):
        def _inner():
            (
                compiled_grouped_gemm,
                initial_cute_tensors_abc,
                tensor_of_dim_size_mnkl,
                tensor_of_strides_abc,
                tensor_of_ptrs_abc,
                tensor_of_tensormap,
                current_stream,
                torch_tensors_abc,
            ) = compile_cutedsl_grouped_gemm(
                group_A,
                group_B,
                ab_dtype=cutlass.Float16
                if self.dtype == torch.float16
                else cutlass.BFloat16,
                c_dtype=cutlass.Float16
                if self.dtype == torch.float16
                else cutlass.BFloat16,
                acc_dtype=cutlass.Float32,
                a_major="m",
                b_major="n",
                c_major="m",
                mma_tiler_mn=(128, 256),
                cluster_shape_mn=(2, 1),
                use_2cta_instrs=False,
                tensormap_update_mode=cutlass.utils.TensorMapUpdateMode.SMEM,
                tolerance=0.5,
                warmup_iterations=0,
                iterations=1,
                skip_ref_check=True,
            )
            compiled_grouped_gemm(
                initial_cute_tensors_abc[0],
                initial_cute_tensors_abc[1],
                initial_cute_tensors_abc[2],
                tensor_of_dim_size_mnkl,
                tensor_of_strides_abc,
                tensor_of_ptrs_abc,
                tensor_of_tensormap,
                current_stream,
            )

            outs = [C.squeeze(-1) for (_, _, C) in torch_tensors_abc]
            return torch.cat(outs, dim=0)

        return _inner

    # Autotunes the CuteDSL kernel then returns the version compiled with the best configs. Similar to the Triton benchmarks,
    # autotuning time is not included in benchmark measurements.
    # NOTE(nikhilap): Right now we use the shape as an autotune key much like Triton. It is unclear whether that is the right approach for CuteDSL,
    # given how Quack keys instead on dynamic scheduling.
    @register_benchmark(enabled=HAS_CUTEDSL and IS_BLACKWELL)
    def precompiled_cutedsl_grouped_mm_tuned(
        self, group_A, group_B, w=None, split=None
    ):
        # --- Trigger autotune outside of timing ---
        shape_sig = tuple(
            (A.shape[0], B.shape[1], A.shape[1]) for A, B in zip(group_A, group_B)
        )
        grouped_gemm_sm100_tuned(
            group_A,
            group_B,
            ab_dtype=cutlass.Float16
            if self.dtype == torch.float16
            else cutlass.BFloat16,
            c_dtype=cutlass.Float16
            if self.dtype == torch.float16
            else cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            a_major="m",
            b_major="n",
            c_major="m",
            skip_ref_check=True,
            shape_sig=shape_sig,
        )
        torch.cuda.synchronize()

        # --- Return timed closure ---
        def _inner():
            outs = grouped_gemm_sm100_tuned(
                group_A,
                group_B,
                ab_dtype=cutlass.Float16
                if self.dtype == torch.float16
                else cutlass.BFloat16,
                c_dtype=cutlass.Float16
                if self.dtype == torch.float16
                else cutlass.BFloat16,
                acc_dtype=cutlass.Float32,
                skip_ref_check=True,
                shape_sig=shape_sig,
            )
            outs = [C.squeeze(-1) for (_, _, C) in outs]
            return torch.cat(outs, dim=0)

        return _inner

    def get_input_iter(self) -> Generator:
        """
        If external shapes are provided, generate inputs for those shapes.
        If not, fall back to the default sweep:
            group_size = 4
            x_vals = [128, 256, 512, 1024]
        NOTE:
        The 2D+offs variant of torch._grouped_mm only supports a *single shared B* across groups.
        That’s why group_B here just repeats the same B_shared reference.
        If you need truly different B_i per group, you cannot use the 2D+offs API —
        instead, you must switch to the 3D variant (both A and B 3D) where offs is not required.
        """
        if hasattr(self, "external_shapes") and self.external_shapes:
            self.shapes = self.external_shapes
        else:
            self.shapes = get_default_shapes()
        if is_fbcode():
            self.shapes += get_fb_shapes()

        # Generate tensors from shapes
        for A_shapes, B_shape, W_shapes, split_size in self.shapes:
            G = len(A_shapes)

            B_shared = torch.rand(
                B_shape, device=self.device, dtype=self.dtype
            ).contiguous()

            group_A = [
                torch.rand(A_shape, device=self.device, dtype=self.dtype).contiguous()
                for A_shape in A_shapes
            ]
            group_B = [B_shared] * G
            w = None
            if W_shapes is not None:
                w = [
                    torch.randn(
                        W_shape, device=self.device, dtype=self.dtype
                    ).contiguous()
                    for W_shape in W_shapes
                ]
            split = None
            if split_size is not None:
                split = torch.tensor(split_size, device=self.device, dtype=torch.int64)

            yield (group_A, group_B, w, split)

    def list_input_to_jagged(
        self,
        group_A: List[torch.Tensor],
        group_B: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        G = len(group_A)

        A_packed = torch.cat(group_A, dim=0).contiguous()

        B0 = group_B[0]
        B_batched = B0.unsqueeze(0).expand(G, -1, -1).contiguous()

        # Offsets over rows of each A_i (NO leading 0), dtype=int32
        M_sizes = [a.shape[0] for a in group_A]
        offs = torch.tensor(
            list(accumulate(M_sizes)), device=self.device, dtype=torch.int32
        )

        return A_packed, B_batched, offs

    def list_input_to_triton_input(
        self,
        group_A: List[torch.Tensor],
        group_B: List[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        group_size = len(group_A)
        device = group_A[0].device

        A_addrs = []
        B_addrs = []
        C_addrs = []
        g_sizes = []
        g_lds = []
        group_C = []
        for i in range(group_size):
            A = group_A[i]
            B = group_B[i]
            assert A.shape[1] == B.shape[0]
            M, K = A.shape
            K, N = B.shape
            C = torch.zeros((M, N), device=device, dtype=A.dtype)
            group_C.append(C)
            A_addrs.append(A.data_ptr())
            B_addrs.append(B.data_ptr())
            C_addrs.append(C.data_ptr())
            g_sizes += [M, N, K]
            g_lds += [A.stride(0), B.stride(0), C.stride(0)]

        # note these are device tensors
        d_a_ptrs = torch.tensor(A_addrs, device=device)
        d_b_ptrs = torch.tensor(B_addrs, device=device)
        d_c_ptrs = torch.tensor(C_addrs, device=device)
        d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
        d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)

        return (d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_C)

    @register_metric()
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        group_A, group_B = example_inputs
        flops = 0
        for a, b in zip(group_A, group_B):
            m, k = a.size()
            k, n = b.size()
            flops += m * k * 2 * n
        return flops

    def get_x_val(self, example_inputs):
        N = example_inputs[0][0].shape[0]
        return N
