import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch

from tritonbench.utils.env_utils import is_hip

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
except ModuleNotFoundError:
    LigerRMSNorm = None

try:
    from .aiter import AITerRMSNorm

    HAS_AITER = True
except ModuleNotFoundError:
    HAS_AITER = False

try:
    from .quack import QuackRMSNorm
except ModuleNotFoundError:
    QuackRMSNorm = None


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--M",
        type=int,
        default=2048,
        help="[Optional] Size of dimension 0 in input shape (integer), default: 2048",
    )
    parser.add_argument(
        "--H",
        type=int,
        help="[Optional] Hidden size dimension (integer)",
    )
    return parser.parse_args(args)


# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_rms_norm.py


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.M = args.M
        self.H = args.H
        self.eps = 1e-6
        # they are generated later
        self.llama_rms_op = None
        self.liger_rms_op = None

    def get_input_iter(self) -> Generator:
        # If H is provided, use only that value; otherwise use the default range
        if self.H is not None:
            H_values = [self.H]
        else:
            H_values = [2**i for i in range(10, 16)]

        for H in H_values:
            x_shape = (self.M, H)
            _input = torch.randn(x_shape, dtype=self.dtype, device=self.device)
            yield H, _input

    @register_benchmark(baseline=True)
    def llama_rms(self, H, input) -> Callable:
        self.llama_rms_op = LlamaRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        return lambda: self.llama_rms_op(input)

    @register_benchmark()
    def liger_rms(self, H, input) -> Callable:
        self.liger_rms_op = LigerRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        return lambda: self.liger_rms_op(input)

    @register_benchmark(enabled=QuackRMSNorm)
    def quack_rms(self, H, input) -> Callable:
        self.quack_rms_op = QuackRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        return lambda: self.quack_rms_op(input)

    @register_benchmark()
    def torch_compile_rms(self, H, input) -> Callable:
        if self.llama_rms_op is None:
            self.llama_rms_op = LlamaRMSNorm(hidden_size=H, eps=self.eps).to(
                self.device
            )
        compiled = torch.compile(self.llama_rms_op, mode="max-autotune-no-cudagraphs")
        return lambda: compiled(input)

    @register_benchmark(enabled=is_hip() and HAS_AITER)
    def aiter(self, H, input) -> Callable:
        self.aiter_rms_op = AITerRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        return lambda: self.aiter_rms_op(input)

    @register_x_val(label="(M, H)")
    def get_x_val(self, example_inputs) -> Tuple[int, int]:
        H = example_inputs[0]
        return (self.M, H)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        do = torch.randn_like(y)
        return lambda: y.backward(do, retain_graph=True)
