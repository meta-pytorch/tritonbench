# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)

from .quantize import quantize_bf16_to_int2

W2A16_ENABLED = False
try:
    torch.ops.load_library(
        "//pytorch/tritonbench/tritonbench/operators/mixed_gemm/kernels:w2a16_gemm_lib"
    )
    W2A16_ENABLED = True
except Exception:
    W2A16_ENABLED = False


try:
    from marlin.quantize import marlin_quantize

    torch.ops.load_library("//ai_codesign/gen_ai/marlin:marlin_ops")
    MARLIN_ENABLED = True
except ImportError:
    MARLIN_ENABLED = False

try:
    from machete.machete import machete_gemm
    from machete.quantize import machete_quantize_and_pack

    MACHETE_ENABLED = True
except ImportError:
    MACHETE_ENABLED = False

try:
    from tinygemm.utils import group_quantize_tensor

    torch.ops.load_library("//tinygemm:tinygemm")
    TINYGEMM_ENABLED = True
except ImportError:
    TINYGEMM_ENABLED = False


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, help="M dimension")
    parser.add_argument("--n", type=int, default=256, help="N dimension")
    parser.add_argument("--k", type=int, default=512, help="K dimension")
    return parser.parse_args(args)


from dataclasses import astuple, dataclass


@dataclass
class _Shape:
    m: int
    n: int
    k: int

    def unpack(self):
        return astuple(self)


def _generate_default_shapes():
    return [_Shape(1, 13312, 16384), _Shape(4096, 8192, 4096)]


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    FWD_ONLY = True
    DEFAULT_METRICS = ["latency", "speedup"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        if args.m:
            self.shapes = [
                _Shape(
                    args.m,
                    args.n,
                    args.k,
                )
            ]
        else:
            self.shapes = _generate_default_shapes()

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            m, n, k = shape.unpack()

            a = torch.randn((m, k), device=self.device, dtype=self.dtype)
            w = torch.randn((k, n), device=self.device, dtype=self.dtype)

            yield (a, w)

    @register_x_val(label="(MNK)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        a, w = example_inputs
        m, k = a.shape
        _, n = w.shape
        return (m, n, k)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()

        try:
            torch.testing.assert_close(output, baseline_output, atol=1e-3, rtol=0.5)
            return True
        except Exception:
            return False

    @register_benchmark(baseline=True)
    def aten_bf16_bf16(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
    ) -> Callable:
        return lambda: torch.matmul(a, w)

    @register_benchmark(enabled=W2A16_ENABLED)
    def cutlass_w2a16(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
    ) -> Callable:
        wq = quantize_bf16_to_int2(w)
        return lambda: torch.ops.mixed_gemm.w2a16_gemm(a, wq)

    @register_benchmark(enabled=MACHETE_ENABLED)
    def machete_w4a16(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
    ) -> Callable:
        _, wq, scale, _ = machete_quantize_and_pack(w, bits=4, groupsize=128)
        return lambda: machete_gemm(a, wq, bits=4, groupsize=128, scales=scale)

    @register_benchmark(enabled=MARLIN_ENABLED)
    def marlin_w4a16(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
    ) -> Callable:
        _, wq, scale = marlin_quantize(w, 128)
        return lambda: torch.ops.marlin.marlin_gemm(a, wq, scale)

    @register_benchmark(enabled=TINYGEMM_ENABLED)
    def tinny_w4a16(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
    ) -> Callable:
        w_t = w.t().contiguous()
        w_int32, w_scales_and_zeros = group_quantize_tensor(
            w_t, n_bit=4, q_group_size=128
        )
        wq = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, 4)
        return lambda: torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_int4TC(
            wq, a, 128, w_scales_and_zeros, False
        )
