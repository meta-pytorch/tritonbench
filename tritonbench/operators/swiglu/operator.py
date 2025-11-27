import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    Mode,
    register_benchmark,
    register_x_val,
)

try:
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except ModuleNotFoundError:
    LigerSwiGLUMLP = None

# Reference: https://github.com/linkedin/Liger-Kernel/
# blob/main/benchmark/scripts/benchmark_swiglu.py


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.B = 4
        self.hidden_size = 4096
        self.dtype = torch.bfloat16
        self.intermediate_size = 11008
        self.hidden_act = "silu"
        self.llama_config = LlamaConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
        )
        self.baseline_op = LlamaMLP(self.llama_config).to(self.device).to(self.dtype)
        if LigerSwiGLUMLP is not None:
            self.liger_op = LigerSwiGLUMLP(self.llama_config).to(self.device).to(self.dtype)
            # Copy weights from baseline to liger model for fair accuracy comparison
            with torch.no_grad():
                self.liger_op.gate_proj.weight.data.copy_(
                    self.baseline_op.gate_proj.weight.data
                )
                self.liger_op.up_proj.weight.data.copy_(
                    self.baseline_op.up_proj.weight.data
                )
                self.liger_op.down_proj.weight.data.copy_(
                    self.baseline_op.down_proj.weight.data
                )

    def get_input_iter(self) -> Generator:
        for seq_len in [2**i for i in range(10, 14)]:
            x_shape = (self.B, seq_len, self.hidden_size)
            x = torch.randn(
                *x_shape, device=self.device, dtype=self.dtype, requires_grad=True
            )

            yield (x,)

    @register_benchmark(baseline=True)
    def torch_swiglu(self, input) -> Callable:
        return lambda: self.baseline_op(input)

    @register_benchmark(enabled=LigerSwiGLUMLP is not None)
    def liger_swiglu(self, input) -> Callable:
        return lambda: self.liger_op(input)

    @register_benchmark()
    def torch_compile_swiglu(self, input) -> Callable:
        # TODO: remove this once we have a better way to handle backward benchmarking
        # We need to run backward multiple times for proper benchmarking
        # so donated buffer have to be disabled
        if self.mode == Mode.BWD or self.mode == Mode.FWD_BWD:
            from torch._functorch import config as functorch_config

            functorch_config.donated_buffer = False

        compiled = torch.compile(self.baseline_op, mode="max-autotune-no-cudagraphs")
        return lambda: compiled(input)

    @register_x_val(label="(B, T, H)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        return (self.B, example_inputs[0].size(1), example_inputs[0].size(2))

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        return [args[0]]

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        # Override default tolerances for bfloat16
        output = fn()
        baseline_output = baseline_fn()
        rtol = self.tb_args.rtol if self.tb_args.rtol is not None else 0.05
        atol = self.tb_args.atol if self.tb_args.atol is not None else 0.005

        try:
            torch.testing.assert_close(output, baseline_output, rtol=rtol, atol=atol)
            return True
        except AssertionError:
            return False
