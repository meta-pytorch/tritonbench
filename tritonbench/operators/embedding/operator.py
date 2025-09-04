import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from torch.nn import Embedding

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)

try:
    from liger_kernel.transformers.experimental.embedding import LigerEmbedding
except ModuleNotFoundError:
    LigerEmbedding = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_embedding.py


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        # they are generated later
        self.baseline_op = None
        self.liger_op = None
        self.reset_dynamo = True

    def get_input_iter(self) -> Generator:
        for B, T, D in [(32, 512, 768), (8, 2048, 4096)]:
            for V in [2**i for i in range(10, 18)]:
                _input = torch.randint(0, V, (B, T), device=self.device)
                tmp_embed = Embedding(V, D).to(self.device).to(self.dtype)
                shared_weight = tmp_embed.weight.data
                yield V, D, _input, shared_weight

    @register_benchmark(baseline=True)
    def torch_embedding(self, V, D, input, shared_weight) -> Callable:
        self.baseline_op = Embedding(V, D).to(self.device).to(self.dtype)
        self.baseline_op.weight.data.copy_(shared_weight)
        return lambda: self.baseline_op(input)

    @register_benchmark()
    def liger_embedding(self, V, D, input, shared_weight) -> Callable:
        self.liger_op = LigerEmbedding(V, D).to(self.device).to(self.dtype)
        self.liger_op.weight.data.copy_(shared_weight)
        return lambda: self.liger_op(input)

    @register_benchmark()
    def torch_compile_embedding(self, V, D, input, shared_weight) -> Callable:
        self.baseline_op = Embedding(V, D).to(self.device).to(self.dtype)
        self.baseline_op.weight.data.copy_(shared_weight)
        compiled = torch.compile(self.baseline_op, mode="max-autotune-no-cudagraphs")
        return lambda: compiled(input)

    @register_x_val(label="(B, T, D, V)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        V, D, input_tensor, _ = example_inputs
        return (input_tensor.size(0), input_tensor.size(1), D, V)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        do = torch.randn_like(y)
        return lambda: y.backward(do, retain_graph=True)
