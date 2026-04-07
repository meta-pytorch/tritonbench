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


def fused_linear_gelu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Fused linear + GeLU activation.

    Common in transformer FFN layers. Inductor fuses the GeLU into the
    matmul epilogue as a template kernel (triton_tem_fused_addmm_gelu_*).
    """
    return F.gelu(F.linear(x, weight, bias))


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--in-features", type=int, default=None, help="Input features")
    parser.add_argument(
        "--out-features", type=int, default=None, help="Output features"
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "speedup", "tflops"]
    DEFAULT_PRECISION = "bf16"
    FWD_ONLY = True
    is_compute_bound = True

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.batch_size = args.batch
        self.seq_len = args.seq_len
        self.in_features = args.in_features
        self.out_features = args.out_features

    @register_x_val(label="(M, K, N)")
    def get_x_val(self, example_inputs) -> str:
        x, weight, bias = example_inputs
        M = x.shape[0]
        K = x.shape[1]
        N = weight.shape[0]
        return f"({M}, {K}, {N})"

    @register_benchmark(baseline=True)
    def aten(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> Callable:
        return lambda: fused_linear_gelu(x, weight, bias)

    @register_benchmark()
    def inductor(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> Callable:
        compiled_fn = torch.compile(fused_linear_gelu, fullgraph=True)
        return lambda: compiled_fn(x, weight, bias)

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Tuple, metrics: BenchmarkOperatorMetrics
    ):
        x, weight, bias = example_inputs
        M = x.shape[0]
        K = x.shape[1]
        N = weight.shape[0]
        # Linear: 2 * M * K * N FLOPs
        flops = 2.0 * M * K * N
        tflops = flops / metrics.latency / 1e12
        return (
            tflops,
            flops / metrics.latency.max / 1e12,
            flops / metrics.latency.min / 1e12,
        )

    def get_input_iter(self) -> Generator:
        M = self.batch_size * self.seq_len

        if self.in_features and self.out_features:
            configs = [(self.in_features, self.out_features)]
        else:
            # Common transformer FFN sizes: (hidden, 4*hidden) expansion
            configs = [
                (1024, 4096),
                (2048, 8192),
                (4096, 16384),
                (5120, 20480),
                (8192, 32768),
            ]

        for K, N in configs:
            x = torch.randn(M, K, device=self.device, dtype=self.dtype)
            weight = torch.randn(N, K, device=self.device, dtype=self.dtype)
            bias = torch.randn(N, device=self.device, dtype=self.dtype)
            yield x, weight, bias
