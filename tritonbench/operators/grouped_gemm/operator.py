from itertools import accumulate
from typing import Any, Generator, List, Tuple

import torch
from torch._inductor import config as inductor_config
from tritonbench.utils.env_utils import get_nvidia_gpu_model, is_cuda

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

# Try to import Cutlass and CuteDSL
try:
    import cutlass
    import cutlass.utils as utils

    from .cutedsl import compile_cutedsl_grouped_gemm

    # Set HAS_CUTEDSL to True if import succeeds
    HAS_CUTEDSL = True
except (ImportError, AttributeError):
    HAS_CUTEDSL = False


from .kernels import triton_group_gemm_fn

IS_B200 = is_cuda() and get_nvidia_gpu_model() == "NVIDIA B200"


# TODO(nikhilap): Add a separate 3D grouped_gemm operator to alleviate the restriction that all B tensors must be the same.
class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["latency", "speedup", "accuracy"]

    @register_benchmark(baseline=True)
    def aten_grouped_mm(self, group_A, group_B):
        A_packed, B_shared, offs = self.list_input_to_jagged(group_A, group_B)
        return lambda: torch._grouped_mm(A_packed, B_shared, offs=offs, bias=None)

    @register_benchmark()
    def naive(self, group_A, group_B):
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

    @register_benchmark()
    def pt2_triton_grouped_mm(self, group_A, group_B):
        torch._dynamo.reset()

        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            autotune_fallback_to_aten=False,
        ):
            A_packed, B_shared, offs = self.list_input_to_jagged(group_A, group_B)
            compiled = torch.compile(torch._grouped_mm, dynamic=False)
            return lambda: compiled(A_packed, B_shared, offs=offs, bias=None)

    @register_benchmark()
    def triton(self, group_A, group_B):
        def _inner():
            outs = triton_group_gemm_fn(group_A, group_B)
            return torch.cat(outs, dim=0)

        return _inner

    @register_benchmark(enabled=HAS_CUTEDSL and IS_B200)
    def cutedsl_grouped_mm(self, group_A, group_B):
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
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
            tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
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

    def get_input_iter(self) -> Generator:
        # NOTE:
        # The 2D+offs variant of torch._grouped_mm only supports a *single shared B* across groups.
        # That’s why group_B here just repeats the same B_shared reference.
        # If you need truly different B_i per group, you cannot use the 2D+offs API —
        # instead, you must switch to the 3D variant (both A and B 3D) where offs is not required.
        self.group_size = 4
        x_vals = [2**i for i in range(7, 11)]  # 128, 256, 512, 1024

        for N in x_vals:
            G = self.group_size
            M = K = N
            N_out = N

            B_shared = torch.rand(
                (K, N_out), device=self.device, dtype=self.dtype
            ).contiguous()

            group_A = [
                torch.rand((M, K), device=self.device, dtype=self.dtype).contiguous()
                for _ in range(G)
            ]
            group_B = [B_shared] * G  # same weight per group in list-form

            yield (group_A, group_B)

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
