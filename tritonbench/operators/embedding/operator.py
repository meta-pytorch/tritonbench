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


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    # When any of --B/--T/--D is given, a single (B, T, D) shape is used instead
    # of the default two; --v-range overrides the vocab sweep. Defaults preserve
    # the original behavior.
    parser.add_argument("--B", type=int, default=None, help="Batch size")
    parser.add_argument("--T", type=int, default=None, help="Sequence length")
    parser.add_argument("--D", type=int, default=None, help="Embedding dim")
    parser.add_argument(
        "--v-range",
        type=str,
        default="10,18",
        help="Vocab size range 'start,end' -> V=2^start..2^(end-1)",
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        # they are generated later
        self.baseline_op = None
        self.liger_op = None
        self.reset_dynamo = True
        args = parse_op_args(self.extra_args)
        start, end = map(int, args.v_range.split(","))
        self._v_range = range(start, end)
        # Override the default (B, T, D) sweep only when explicitly requested.
        if args.B is not None or args.T is not None or args.D is not None:
            self._btd_shapes = [
                (
                    args.B if args.B is not None else 32,
                    args.T if args.T is not None else 512,
                    args.D if args.D is not None else 768,
                )
            ]
        else:
            self._btd_shapes = [(32, 512, 768), (8, 2048, 4096)]

    def get_input_iter(self) -> Generator:
        for B, T, D in self._btd_shapes:
            for V in [2**i for i in self._v_range]:
                # Pallas/TPU does not support int64 tensors; use int32 indices there.
                idx_dtype = torch.int32 if self.device == "tpu" else torch.int64
                _input = torch.randint(
                    0, V, (B, T), device=self.device, dtype=idx_dtype
                )
                tmp_embed = Embedding(V, D).to(self.device).to(self.dtype)
                shared_weight = tmp_embed.weight.data
                yield V, D, _input, shared_weight

    @register_benchmark(baseline=True)
    def torch_embedding(self, V, D, input, shared_weight) -> Callable:
        self.baseline_op = Embedding(V, D).to(self.device).to(self.dtype)
        self.baseline_op.weight.data.copy_(shared_weight)
        return lambda: self.baseline_op(input)

    @register_benchmark(enabled=LigerEmbedding is not None)
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
