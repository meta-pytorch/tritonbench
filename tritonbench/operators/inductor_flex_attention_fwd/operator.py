import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from tritonbench.operators.flex_attention.mods import causal_mask
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)


torch._dynamo.config.automatic_dynamic_shapes = False


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--n-heads", type=int, default=16, help="Number of heads")
    parser.add_argument("--d-head", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--seq-len", type=int, default=None, help="Fixed sequence length"
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
        self.num_heads = args.n_heads
        self.head_dim = args.d_head
        self.seq_len = args.seq_len

    @register_x_val(label="(B, H, S, D)")
    def get_x_val(self, example_inputs) -> str:
        q, k, v, block_mask = example_inputs
        B, H, S, D = q.shape
        return f"({B}, {H}, {S}, {D})"

    @register_benchmark(baseline=True)
    def aten(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: Optional[BlockMask],
    ) -> Callable:
        return lambda: flex_attention(q, k, v, block_mask=block_mask)

    @register_benchmark()
    def inductor(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: Optional[BlockMask],
    ) -> Callable:
        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        return lambda: compiled_fn(q, k, v, block_mask=block_mask)

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Tuple, metrics: BenchmarkOperatorMetrics
    ):
        q, k, v, block_mask = example_inputs
        B, H, S, D = q.shape

        # QK matmul + OV matmul: 2 * B * H * S^2 * D * 2
        flops = 2.0 * B * H * S * S * D * 2

        # Adjust for block sparsity
        if block_mask is not None:
            sparsity = block_mask.sparsity() / 100.0
            flops *= 1 - sparsity

        tflops = flops / metrics.latency / 1e12
        return (
            tflops,
            flops / metrics.latency.max / 1e12,
            flops / metrics.latency.min / 1e12,
        )

    def get_input_iter(self) -> Generator:
        B = self.batch_size
        H = self.num_heads
        D = self.head_dim

        if self.seq_len:
            seq_lens = [self.seq_len]
        else:
            seq_lens = [2**i for i in range(7, 15)]  # 128 to 16384

        compiled_block_mask = torch.compile(create_block_mask)

        for S in seq_lens:
            q = torch.randn(
                B, H, S, D, device=self.device, dtype=self.dtype, requires_grad=False
            )
            k = torch.randn(
                B, H, S, D, device=self.device, dtype=self.dtype, requires_grad=False
            )
            v = torch.randn(
                B, H, S, D, device=self.device, dtype=self.dtype, requires_grad=False
            )
            block_mask = compiled_block_mask(causal_mask, 1, 1, S, S, device=self.device)
            yield q, k, v, block_mask
