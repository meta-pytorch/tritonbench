import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="[Optional] Number of attention heads (integer)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="[Optional] Per-head dimension (integer)",
    )
    return parser.parse_args(args)


def multi_head_rmsnorm(
    query: torch.Tensor,
    key: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-head RMSNorm for both query and key tensors.

    Args:
        query: Query tensor [batch_size, seq_len, num_heads, head_dim]
        key: Key tensor [batch_size, seq_len, num_heads, head_dim]
        weight_q: Per-head scale for query [num_heads, head_dim]
        weight_k: Per-head scale for key [num_heads, head_dim]
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (normalized_query, normalized_key)
    """
    input_dtype = query.dtype

    # Compute RMS for query per head (normalize over head_dim dimension)
    # variance shape: [batch, seq_len, num_heads, 1]
    q_variance = query.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    query_norm = query * torch.rsqrt(q_variance + eps)

    # Compute RMS for key per head
    k_variance = key.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    key_norm = key * torch.rsqrt(k_variance + eps)

    # Apply learnable scales
    # weight shape: [num_heads, head_dim]
    # Broadcast to [batch, seq_len, num_heads, head_dim]
    query_norm = query_norm * weight_q.unsqueeze(0).unsqueeze(0)
    key_norm = key_norm * weight_k.unsqueeze(0).unsqueeze(0)

    return query_norm.to(input_dtype), key_norm.to(input_dtype)


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.eps = 1e-6
        if self.tb_args.rtol is None:
            self.tb_args.rtol = 1e-5
        if self.tb_args.atol is None:
            self.tb_args.atol = 1e-4

    def get_input_iter(self) -> Generator:
        # (num_heads, head_dim, batch_size, seq_len)
        shapes = [
            (48, 128, 2, 128),
            (48, 128, 4, 1657),
            (48, 128, 4, 1024),
            (48, 128, 8, 773),
        ]

        for num_heads, head_dim, batch_size, seq_len in shapes:
            # Allow overriding the head config from the command line.
            if self.num_heads is not None:
                num_heads = self.num_heads
            if self.head_dim is not None:
                head_dim = self.head_dim

            query = torch.randn(
                (batch_size, seq_len, num_heads, head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            key = torch.randn(
                (batch_size, seq_len, num_heads, head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            weight_q = torch.randn(
                (num_heads, head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            weight_k = torch.randn(
                (num_heads, head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            yield query, key, weight_q, weight_k

    @register_benchmark(baseline=True)
    def torch_multi_head_rmsnorm(self, query, key, weight_q, weight_k) -> Callable:
        return lambda: multi_head_rmsnorm(query, key, weight_q, weight_k, self.eps)

    @register_benchmark()
    def torch_compile_multi_head_rmsnorm(
        self, query, key, weight_q, weight_k
    ) -> Callable:
        compiled = torch.compile(multi_head_rmsnorm, mode="max-autotune-no-cudagraphs")
        return lambda: compiled(query, key, weight_q, weight_k, self.eps)

    @register_x_val(label="(B, T, num_heads, head_dim)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int, int]:
        query = example_inputs[0]
        batch_size, seq_len, num_heads, head_dim = query.shape
        return (batch_size, seq_len, num_heads, head_dim)
