import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
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
        self.num_kv_heads = 8

    def _shape_iter(self) -> Generator:
        hidden_size = 8192
        for seq_length in [2**i for i in range(10, 15)]:
            yield hidden_size, seq_length

        seq_length = 2048
        for hidden_size in [32 * (2**i) for i in range(4, 10, 2)]:
            yield hidden_size, seq_length

    def get_input_iter(self) -> Generator:
        for hidden_size, seq_length in self._shape_iter():
            yield self._make_inputs(hidden_size, seq_length)

    def get_available_num_inputs(self) -> int:
        return sum(1 for _ in self._shape_iter())

    def _make_inputs(self, hidden_size, seq_length):
        head_dim = hidden_size // self.num_q_heads
        llama_config = LlamaConfig(
            head_dim=head_dim,
        )
        rotary_emb = LlamaRotaryEmbedding(llama_config, device=self.device)
        q = (
            torch.randn(
                (1, seq_length, self.num_q_heads, head_dim),
                device=self.device,
                requires_grad=True,
                dtype=self.dtype,
            )
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            torch.randn(
                (1, seq_length, self.num_kv_heads, head_dim),
                device=self.device,
                requires_grad=True,
                dtype=self.dtype,
            )
            .transpose(1, 2)
            .contiguous()
        )
        pos_ids = torch.arange(
            seq_length, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        cos, sin = rotary_emb(k, pos_ids)
        return q, k, cos, sin, pos_ids

    def _save_for_backward(self, q, k):
        self.q = q
        self.k = k
        self.dq = torch.randn_like(q, device=self.device, dtype=self.dtype)
        self.dk = torch.randn_like(k, device=self.device)

    @register_benchmark(baseline=True)
    def apply_rotary_pos_emb(self, q, k, cos, sin, pos_ids) -> Callable:
        self._save_for_backward(q, k)
        return lambda: apply_rotary_pos_emb(q, k, cos, sin)

    @register_benchmark()
    def liger_rotary_pos_emb(self, q, k, cos, sin, pos_ids) -> Callable:
        self._save_for_backward(q, k)
        return lambda: liger_rotary_pos_emb(q, k, cos, sin, pos_ids)

    @register_benchmark()
    def torch_compile_rotary_pos_emb_full_op(self, q, k, cos, sin, pos_ids) -> Callable:
        self._save_for_backward(q, k)
        head_dim = q.shape[-1]
        llama_config = LlamaConfig(
            head_dim=head_dim,
        )
        compiled = torch.compile(
            LlamaRotaryEmbedding(llama_config, device=self.device),
            mode="max-autotune-no-cudagraphs",
        )
        cos, sin = compiled(k, pos_ids)
        compiled_func = torch.compile(
            apply_rotary_pos_emb, mode="max-autotune-no-cudagraphs"
        )
        return lambda: compiled_func(q, k, cos, sin)

    @register_x_val(label="(H, T)")
    def get_x_val(self, example_inputs) -> Tuple[int, int]:
        q = example_inputs[0]
        _, num_q_heads, seq_length, head_dim = q.shape
        return (head_dim * num_q_heads, seq_length)

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
