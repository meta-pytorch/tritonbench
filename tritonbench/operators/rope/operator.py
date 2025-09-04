import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
)

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)

try:
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
except ModuleNotFoundError:
    liger_rotary_pos_emb = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_rope.py


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        # they are generated later
        self.baseline_op = None
        self.liger_op = None
        self.num_q_heads = 32
        self.num_kv_heads = 32  # should be 8
        self.hidden_dim = 128

    def get_input_iter(self) -> Generator:
        batch_size = 1
        for seq_length in [2**i for i in range(0, 19)]:
            yield batch_size, seq_length

        seq_length = 1
        for batch_size in [2**i for i in range(0, 11)]:
            yield batch_size, seq_length

    def prepare_input(self, batch_size, seq_length):
        rotary_emb = LlamaRotaryEmbedding(self.hidden_dim, device=self.device)
        q = torch.randn(
            (batch_size, self.num_q_heads, seq_length, self.hidden_dim),
            device=self.device,
            requires_grad=True,
            dtype=self.dtype,
        )
        k = torch.randn(
            (batch_size, self.num_kv_heads, seq_length, self.hidden_dim),
            device=self.device,
            requires_grad=True,
            dtype=self.dtype,
        )
        dq, dk = (
            torch.randn_like(q, device=self.device, dtype=self.dtype),
            torch.randn_like(k, device=self.device),
        )
        pos_ids = torch.arange(
            seq_length, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        cos, sin = rotary_emb(k, pos_ids)
        # save q,k to self for grad_to_none
        self.q = q
        self.k = k
        # save dq,dk to self for backward
        self.dq = dq
        self.dk = dk
        return q, k, cos, sin, pos_ids

    @register_benchmark(baseline=True)
    def apply_rotary_pos_emb(self, batch_size, seq_length) -> Callable:
        q, k, cos, sin, pos_ids = self.prepare_input(batch_size, seq_length)
        return lambda: apply_rotary_pos_emb(q, k, cos, sin, pos_ids)

    @register_benchmark()
    def liger_rotary_pos_emb(self, batch_size, seq_length) -> Callable:
        q, k, cos, sin, pos_ids = self.prepare_input(batch_size, seq_length)
        return lambda: liger_rotary_pos_emb(q, k, cos, sin, pos_ids)

    @register_benchmark()
    def inductor_rotary_pos_emb_full_op(self, batch_size, seq_length) -> Callable:
        q, k, cos, sin, pos_ids = self.prepare_input(batch_size, seq_length)
        get_rotary_embedding = LlamaRotaryEmbedding(self.hidden_dim, device=self.device)
        cos, sin = get_rotary_embedding(k, pos_ids)
        compiled_func = torch.compile(
            apply_rotary_pos_emb, mode="max-autotune-no-cudagraphs"
        )
        return lambda: compiled_func(q, k, cos, sin, pos_ids)

    @register_x_val(label="(B, S)")
    def get_x_val(self, example_inputs) -> Tuple[int, int]:
        return (example_inputs[0], example_inputs[1])

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        q_out, k_out = fwd_fn()
        return lambda: torch.autograd.grad(
            (q_out, k_out),
            (self.q, self.k),
            (self.dq, self.dk),
            allow_unused=True,
            retain_graph=True,
        )

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        return [self.q, self.k]
