from typing import Any, Generator

import torch

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernels import triton_group_gemm_fn


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "fp16"
    DEFAULT_METRICS = ["latency", "speedup", "accuracy"]

    @register_benchmark(baseline=True)
    def eager(self, group_A, group_B):
        def _inner():
            out = []
            for a, b in zip(group_A, group_B):
                out.append(torch.matmul(a, b))
            return out

        return _inner

    @register_benchmark()
    def torch_compile(self, group_A, group_B):
        return torch.compile(
            self.eager(group_A, group_B), mode="max-autotune-no-cudagraphs"
        )

    @register_benchmark()
    def triton(self, group_A, group_B):
        return lambda: triton_group_gemm_fn(group_A, group_B)

    def get_input_iter(self) -> Generator:
        self.group_size = 4
        x_vals = [2**i for i in range(7, 11)]
        for N in x_vals:
            group_A = []
            group_B = []
            for i in range(self.group_size):
                A = torch.rand((N, N), device=self.device, dtype=self.dtype)
                B = torch.rand((N, N), device=self.device, dtype=self.dtype)
                group_A.append(A)
                group_B.append(B)
            yield group_A, group_B

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
