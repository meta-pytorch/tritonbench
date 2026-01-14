import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from tritonbench.utils.env_utils import is_hip
from tritonbench.utils.python_utils import try_import
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    Mode,
    register_benchmark,
    register_x_val,
)

from . import fused_triton

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

with try_import("HAS_TILELANG"):
    from .tilelang import TileLangRMSNorm


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
        if self.tb_args.rtol is None:
            self.tb_args.rtol = 1e-5
        if self.tb_args.atol is None:
            self.tb_args.atol = 1e-4

    def get_input_iter(self) -> Generator:
        # If H is provided, use only that value; otherwise use the default range
        if self.H is not None:
            H_values = [self.H]
        else:
            H_values = [2**i for i in range(10, 16)]

        requires_grad = self.mode in (Mode.BWD, Mode.FWD_BWD)

        for H in H_values:
            x_shape = (self.M, H)
            _input = torch.randn(
                x_shape,
                dtype=self.dtype,
                device=self.device,
                requires_grad=requires_grad,
            )
            weight = torch.nn.Parameter(
                torch.ones(H, dtype=self.dtype, device=self.device),
                requires_grad=requires_grad,
            )
            yield H, _input, weight

    @register_benchmark(baseline=True)
    def llama_rms(self, H, input, weight) -> Callable:
        module = LlamaRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        module.weight = weight
        self.llama_rms_op = module
        return lambda: module(input)

    @register_benchmark(enabled=LigerRMSNorm is not None)
    def liger_rms(self, H, input, weight) -> Callable:
        module = LigerRMSNorm(
            hidden_size=H,
            eps=self.eps,
            in_place=False,
        ).to(self.device)
        module.weight = weight
        self.liger_rms_op = module
        return lambda: module(input)

    @register_benchmark(enabled=QuackRMSNorm)
    def quack_rms(self, H, input, weight) -> Callable:
        module = QuackRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        module.weight = weight
        self.quack_rms_op = module
        return lambda: module(input)

    @register_benchmark()
    def torch_compile_rms(self, H, input, weight) -> Callable:
        module = LlamaRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        module.weight = weight
        self.llama_rms_op = module
        compiled = torch.compile(module, mode="max-autotune-no-cudagraphs")
        return lambda: compiled(input)

    @register_benchmark()
    def triton_fused_rmsnorm(self, H, input, weight) -> Callable:
        return lambda: fused_triton.rms_norm(input, H, weight, self.eps)

    @register_benchmark(enabled=is_hip() and HAS_AITER)
    def aiter(self, H, input, weight) -> Callable:
        module = AITerRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        module.weight = weight
        self.aiter_rms_op = module
        return lambda: module(input)

    @register_benchmark(enabled=HAS_TILELANG)
    def tilelang(self, H, input, weight) -> Callable:
        module = TileLangRMSNorm(hidden_size=H, eps=self.eps).to(self.device)
        module.weight = weight
        return module(input)

    @register_x_val(label="(M, H)")
    def get_x_val(self, example_inputs) -> Tuple[int, int]:
        H = example_inputs[0]
        return (self.M, H)
