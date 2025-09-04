import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except ModuleNotFoundError:
    LigerFusedLinearCrossEntropyLoss = None

# Reference: https://github.com/linkedin/Liger-Kernel/blob/\
# 3d0653b035222cbb845435a1994854e4fd219107/benchmark/scripts/benchmark_fused_linear_cross_entropy.py


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096, help="hidden size")
    parser.add_argument("--vocab-size", type=int, default=128256, help="vocab size")
    return parser.parse_args(args)


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, input, weight, target):
        logits = torch.nn.functional.linear(input, weight)
        return self.ce_loss(logits, target)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, input, weight, target):
        return self.ce_loss(weight, input, target)


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        op_args = parse_op_args(self.extra_args)
        self.hidden_size = op_args.hidden_size
        self.vocab_size = op_args.vocab_size
        # Create the shared weight tensor
        self.weight = torch.randn(
            self.vocab_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )

        self.baseline_model = TorchLMHeadCE().to(self.device)
        self.liger_model = LigerLMHeadCE().to(self.device)

    def get_input_iter(self) -> Generator:
        for BT in [2**i for i in range(12, 16)]:
            _input = torch.randn(
                BT,
                self.hidden_size,
                requires_grad=True,
                dtype=self.dtype,
                device=self.device,
            )
            target = torch.randint(
                self.vocab_size, (BT, 1), dtype=torch.long, device=self.device
            ).squeeze(1)
            yield _input, self.weight, target

    @register_benchmark(baseline=True)
    def torch_lm_head_ce(self, input, weight, target) -> Callable:
        return lambda: self.baseline_model(input, weight, target)

    @register_benchmark()
    def liger_lm_head_ce(self, input, weight, target) -> Callable:
        return lambda: self.liger_model(input, weight, target)

    @register_benchmark()
    def torch_compile_fused_linear_cross_entropy(
        self, input, weight, target
    ) -> Callable:
        compiled = torch.compile(self.baseline_model, mode="max-autotune-no-cudagraphs")
        return lambda: compiled(input, weight, target)

    @register_x_val(label="(B*T, H)")
    def get_x_val(self, example_inputs) -> Tuple[int, int]:
        return (example_inputs[0].size(0), example_inputs[0].size(1))

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        return lambda: y.backward(retain_graph=True)
