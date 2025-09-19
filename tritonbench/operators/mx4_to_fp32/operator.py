import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch

# We are benchmarking the kernel used inside quantize_comm. Insofar, we are using the fp32_to_mx4 fbgemm API rather than the quantize_mx API.
from fbgemm_gpu.quantize_utils import fp32_to_mx4, mx4_to_fp32

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        # they are generated later
        self.reset_dynamo = True

    def get_input_iter(self) -> Generator:
        for sz in [12024, 512 * 1024, 32 * 1024 * 1024, 32 * 1024 * 1024 + 16]:
            ebits = 2
            mbits = 1
            group_size = 32
            _input = fp32_to_mx4(
                torch.randn((sz,), device=self.device, dtype=torch.float32),
                group_size,
                ebits,
                mbits,
            )
            yield _input, group_size, ebits, mbits

    @register_benchmark(baseline=True, fwd_only=True)
    def fbgemm_mx4_to_fp32(
        self, tensor: torch.Tensor, group_size: int, ebits: int, mbits: int
    ) -> Callable:
        return lambda: mx4_to_fp32(
            tensor=tensor,
            group_size=group_size,
            use_triton=True,
            ebits=ebits,
            mbits=mbits,
        )

    @register_x_val(label="(Size, Group Size, ebits, mbits)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int, int]:
        input_tensor, group_size, ebits, mbits = example_inputs
        return (input_tensor.numel(), group_size, ebits, mbits)
