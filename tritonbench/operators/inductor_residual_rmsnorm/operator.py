import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)


torch._dynamo.config.automatic_dynamic_shapes = False


def residual_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fused residual add + RMS normalization.

    Common in every transformer block: norm(x + residual) * weight.
    Inductor fuses this into a single persistent reduction kernel
    (triton_per_fused__fused_rms_norm_*).
    """
    hidden = x + residual
    variance = hidden.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    return weight * hidden


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument(
        "--hidden-size", type=int, default=None, help="Fixed hidden dimension"
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
        self.seq_len = args.seq_len or 2048
        self.hidden_size = args.hidden_size
        self.eps = 1e-6

    @register_x_val(label="(B*S, H)")
    def get_x_val(self, example_inputs) -> str:
        x, residual, weight = example_inputs
        M, H = x.shape
        return f"({M}, {H})"

    @register_benchmark(baseline=True)
    def aten(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> Callable:
        eps = self.eps
        return lambda: residual_rmsnorm(x, residual, weight, eps)

    @register_benchmark()
    def inductor(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> Callable:
        compiled_fn = torch.compile(residual_rmsnorm, mode="max-autotune-no-cudagraphs")
        eps = self.eps
        return lambda: compiled_fn(x, residual, weight, eps)

    @register_metric()
    def gbps(
        self, fn_name: str, example_inputs: Tuple, metrics: BenchmarkOperatorMetrics
    ):
        x, residual, weight = example_inputs
        # Read x + residual + weight, write output
        read_bytes = (
            x.numel() * x.element_size()
            + residual.numel() * residual.element_size()
            + weight.numel() * weight.element_size()
        )
        write_bytes = x.numel() * x.element_size()
        total_bytes = read_bytes + write_bytes
        gbps = total_bytes / metrics.latency / 1e6
        return (
            gbps,
            total_bytes / metrics.latency.max / 1e6,
            total_bytes / metrics.latency.min / 1e6,
        )

    def get_input_iter(self) -> Generator:
        B = self.batch_size
        S = self.seq_len
        M = B * S  # flatten batch and seq dims

        if self.hidden_size:
            hidden_sizes = [self.hidden_size]
        else:
            hidden_sizes = [1024, 2048, 4096, 5120, 8192, 16384]

        for H in hidden_sizes:
            x = torch.randn(M, H, device=self.device, dtype=self.dtype)
            residual = torch.randn(M, H, device=self.device, dtype=self.dtype)
            weight = torch.randn(H, device=self.device, dtype=self.dtype)
            yield x, residual, weight
