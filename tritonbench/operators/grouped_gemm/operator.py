from itertools import accumulate
from typing import Any, Generator, List, Tuple

import torch
from torch._inductor import config as inductor_config

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernels import triton_group_gemm_fn


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
        (d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_C) = (
            self.list_input_to_triton_input(group_A, group_B)
        )

        def _inner():
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
