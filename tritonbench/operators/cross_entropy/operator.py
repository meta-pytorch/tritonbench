import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch

from torch.nn import CrossEntropyLoss

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
except ModuleNotFoundError:
    LigerCrossEntropyLoss = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_cross_entropy.py


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8, help="Batch size")
    parser.add_argument("--T", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--v-range",
        type=str,
        default="12,18",
        help="Vocabulary size range as 'start,end' (e.g., '10,15' for 2^10 to 2^14)",
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.B = args.B
        self.T = args.T
        start, end = map(int, args.v_range.split(","))
        self.v_range = range(start, end)
        self.baseline_model = CrossEntropyLoss()
        self.liger_model = LigerCrossEntropyLoss()

    def get_input_iter(self) -> Generator:
        for V in [2**i for i in self.v_range]:
            _input = torch.randn(
                self.B * self.T,
                V,
                requires_grad=True,
                device=self.device,
            )
            target = torch.randint(V, (self.B * self.T, 1), device=self.device).squeeze(
                1
            )
            yield _input, target

    @register_benchmark(baseline=True)
    def cross_entropy_loss(self, input, target) -> Callable:
        return lambda: self.baseline_model(input, target)

    @register_benchmark()
    def liger_cross_entropy_loss(self, input, target) -> Callable:
        return lambda: self.liger_model(input, target)

    @register_benchmark()
    def torch_compile_cross_entropy_loss(self, input, target) -> Callable:
        compiled = torch.compile(
            self.baseline_model, dynamic=False, mode="max-autotune-no-cudagraphs"
        )
        return lambda: compiled(input, target)

    @register_x_val(label="(B, T, V)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        v = example_inputs[0].size(-1)
        return (self.B, self.T, v)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        # TODO: how to pass grad_to_none=[_input]?
        return lambda: y.backward(retain_graph=True)

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        x = args[0]
        return [x]
