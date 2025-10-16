import argparse
from typing import Any, Callable, Generator, List, Optional

import torch
from torch._dynamo.testing import rand_strided

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_metric,
)

from .triton_welford import (
    fused_native_layer_norm as triton_welford,
    fused_native_layer_norm_no_welford as triton_no_welford,
)


BUILDIN_SHAPES = [
    (262144, 1024),
    (262144, 1536),
    (262144, 2048),
    (262144, 2560),
    (262144, 3072),
    (262144, 4096),
    (262144, 5120),
    (262144, 6144),
    (262144, 7168),
    (262144, 8192),
]


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "speedup", "accuracy"]
    FWD_ONLY = True

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.shapes = BUILDIN_SHAPES

    @register_benchmark()
    def triton_welford(self, p1, p2, p3) -> Callable:
        return lambda: triton_welford(p1, p2, p3)

    @register_benchmark()
    def test_no_welford(self, p1, p2, p3) -> Callable:
        return lambda: triton_no_welford(p1, p2, p3)

    @register_benchmark(baseline=True)
    def eager_layer_norm(self, p1, p2, p3) -> Callable:
        # p1 is weight, p2 is bias, p3 is input
        return lambda: torch.nn.functional.layer_norm(
            p3, normalized_shape=(p3.shape[-1],), weight=p1, bias=p2, eps=1e-05
        )

    @register_benchmark()
    def torch_compile_layer_norm(self, p1, p2, p3) -> Callable:
        return torch.compile(
            self.eager_layer_norm(p1, p2, p3), mode="max-autotune-no-cudagraphs"
        )

    def get_x_val(self, example_inputs) -> float:
        p1, p2, p3 = example_inputs
        s, d = p3.size()
        return d

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            s, d = shape
            p1 = rand_strided((d,), (1,), device="cuda:0", dtype=torch.bfloat16)
            p2 = rand_strided((d,), (1,), device="cuda:0", dtype=torch.bfloat16)
            p3 = rand_strided((s, d), (d, 1), device="cuda:0", dtype=torch.bfloat16)
            yield p1, p2, p3

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()

        # The triton_welford functions return a tuple (output, input, mean, rsqrt)
        # while eager_layer_norm returns just the output tensor
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(baseline_output, tuple):
            baseline_output = baseline_output[0]

        rtol = self.tb_args.rtol if self.tb_args.rtol is not None else 1e-2
        atol = self.tb_args.atol if self.tb_args.atol is not None else 1e-2

        try:
            torch.testing.assert_close(
                output,
                baseline_output,
                rtol=rtol,
                atol=atol,
            )
            return True
        except AssertionError:
            return False
