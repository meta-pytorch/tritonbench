import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from tritonbench.utils.jagged_utils import GIGABYTES_PER_BYTE
# We are benchmarking the kernel used inside quantize_comm. Insofar, we are using the fp32_to_mx4 fbgemm API rather than the quantize_mx API.

from tritonbench.utils.python_utils import try_import
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)

with try_import("HAS_FBGEMM"):
    from fbgemm_gpu.quantize_utils import fp32_to_mx4, mx4_to_fp32

if not HAS_FBGEMM:
    # Fall back to the pure-torch re-implementation so that the operator still
    # loads (input generation, x_vals) without fbgemm installed. The benchmark
    # itself stays gated on HAS_FBGEMM.
    from tritonbench.utils.fbgemm_utils import fp32_to_mx4, mx4_to_fp32


class Operator(BenchmarkOperator):
    is_compute_bound: bool = False

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
        # Decode side of the large MX4 comm (quantize_comm / All2All_Pooled) production
        # cases in fp32_to_mx4. The input is a random packed MX4 tensor of numel
        # M / 2 + M / group_size: each group is group_size // 2 e2m1 bytes plus a
        # shared exponent byte, pinned to 2^0 to avoid inf/NaN. It is built directly
        # to avoid materializing the fp32 message and encoding it.
        for numel, group_size in [
            (1_409_400_832, 16),  # backward
            (1_409_286_144, 8),  # forward
            (2_421_580_288, 8),  # forward
        ]:
            ebits = 2
            mbits = 1
            _input = torch.randint(
                0,
                256,
                (numel // group_size, group_size // 2 + 1),
                dtype=torch.uint8,
                device=self.device,
            )
            _input[:, -1] = 127
            yield _input.view(-1), group_size, ebits, mbits

    @register_benchmark(baseline=True, fwd_only=True, enabled=HAS_FBGEMM)
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

    @register_metric()
    def gbps(
        self,
        fn,
        example_inputs: Tuple[torch.Tensor, int, int, int],
        metrics: BenchmarkOperatorMetrics,
    ) -> float:
        # mx4_to_fp32: a[M / 2 + M / group_size] (int 8) -> out[M]
        packed_group_size = example_inputs[1] // 2 + 1
        num_groups = example_inputs[0].numel() // packed_group_size
        out_size = num_groups * example_inputs[1]
        return (
            (
                example_inputs[0].element_size() * example_inputs[0].numel()
                + out_size * 4
            )
            / metrics.latency
            * 1e3
            * GIGABYTES_PER_BYTE
        )
