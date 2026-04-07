# pyre-strict
import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)


torch._dynamo.config.automatic_dynamic_shapes = False


def fused_softmax(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused softmax: permute (B, S, H, S) -> (B, H, S, S), add bias, softmax."""
    return F.softmax(x.permute(0, 2, 1, 3) + bias, dim=-1)


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--n-heads", type=int, default=16, help="Number of heads")
    parser.add_argument(
        "--seq-len", type=int, default=None, help="Fixed sequence length"
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "speedup", "gbps"]
    DEFAULT_PRECISION = "bf16"
    FWD_ONLY = True
    is_compute_bound = False  # memory-bound reduction

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.batch_size = args.batch
        self.num_heads = args.n_heads
        self.seq_len = args.seq_len

    @register_x_val(label="(B, S, H, S)")
    def get_x_val(self, example_inputs) -> str:
        x, bias = example_inputs
        B, S, H, S2 = x.shape
        return f"({B}, {S}, {H}, {S2})"

    @register_benchmark(baseline=True)
    def aten(self, x: torch.Tensor, bias: torch.Tensor) -> Callable:
        return lambda: fused_softmax(x, bias)

    @register_benchmark()
    def inductor(self, x: torch.Tensor, bias: torch.Tensor) -> Callable:
        compiled_fn = torch.compile(fused_softmax, mode="max-autotune-no-cudagraphs")
        return lambda: compiled_fn(x, bias)

    @register_metric()
    def gbps(
        self, fn_name: str, example_inputs: Tuple, metrics: BenchmarkOperatorMetrics
    ):
        x, bias = example_inputs
        # 1 read + 1 write of the main tensor
        total_bytes = 2 * x.numel() * x.element_size()
        gbps = total_bytes / metrics.latency / 1e6
        return (
            gbps,
            total_bytes / metrics.latency.max / 1e6,
            total_bytes / metrics.latency.min / 1e6,
        )

    def get_input_iter(self) -> Generator:
        B = self.batch_size
        H = self.num_heads

        if self.seq_len:
            seq_lens = [self.seq_len]
        else:
            seq_lens = [2**i for i in range(7, 12)]  # 128 to 16384

        for S in seq_lens:
            # x: (B, S, H, S) attention scores before permute
            x = torch.randn(
                B, S, H, S, device=self.device, dtype=self.dtype, requires_grad=False
            )
            # bias: (1, 1, 1, S) broadcastable bias
            bias = torch.randn(
                1, 1, 1, S, device=self.device, dtype=self.dtype, requires_grad=False
            )
            yield x, bias
