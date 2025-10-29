import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from tritonbench.utils.jagged_utils import GIGABYTES_PER_BYTE
from tritonbench.utils.python_utils import try_import
from tritonbench.utils.triton_op import BenchmarkOperatorMetrics, register_metric

# We are benchmarking the kernel used inside quantize_comm. Insofar, we are using the fp32_to_mx4 fbgemm API rather than the quantize_mx API.
with try_import("HAS_FBGEMM"):
    from fbgemm_gpu.quantize_utils import fp32_to_mx4, RoundingMode

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)


class Operator(BenchmarkOperator):
    is_compute_bound: bool = False

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        # they are generated later
        self.reset_dynamo = True

    def get_input_iter(self) -> Generator:
        for sz in [24048, 1024 * 1024, 64 * 1024 * 1024, 64 * 1024 * 1024 + 16]:
            _input = torch.randn((sz,), device=self.device, dtype=torch.float32)
            yield _input, 32, 2, 1, RoundingMode.even, False

    @register_benchmark(baseline=True, fwd_only=True, enabled=HAS_FBGEMM)
    def fbgemm_fp32_to_mx4(self, *args) -> Callable:
        return lambda: fp32_to_mx4(*args, use_triton=True)

    @register_x_val(
        label="(Size, Group Size, ebits, mbits, rounding_mode, stochastic_casting)"
    )
    def get_x_val(self, example_inputs) -> Tuple[int, int, int, int, RoundingMode, int]:
        input_tensor, group_size, ebits, mbits, rounding_mode, stochastic_casting = (
            example_inputs
        )
        return (
            input_tensor.numel(),
            group_size,
            ebits,
            mbits,
            rounding_mode,
            stochastic_casting,
        )

    @register_metric()
    def gbps(
        self,
        fn,
        example_inputs: Tuple[torch.Tensor, int, int, int, RoundingMode, bool],
        metrics: BenchmarkOperatorMetrics,
    ) -> float:
        # fp32_to_mx4: a[M] -> out[M / 2 + M / group_size] (int8)
        return (
            (
                example_inputs[0].element_size() * example_inputs[0].numel()
                + (example_inputs[0].numel() + 1) // 2
                + (example_inputs[0].numel() + example_inputs[1] - 1)
                // example_inputs[1]
            )
            / metrics.latency
            * 1e3
            * GIGABYTES_PER_BYTE
        )
