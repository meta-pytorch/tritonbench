# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch

from tritonbench.utils.env_utils import is_blackwell

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore
    from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent_mxfp8 import (
        attention as tlx_mxfp8_attention,
        generate_attention_inputs,
    )

    HAS_TLX_MXFP8 = True
except (ImportError, AttributeError):
    HAS_TLX_MXFP8 = False

IS_BLACKWELL = is_blackwell()

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
    register_x_val,
)


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument("--n-heads", type=int, default=48, help="Number of heads")
    parser.add_argument(
        "--d-head", type=int, default=128, help="Head dimension"
    )
    parser.add_argument(
        "--causal", action="store_true", help="Enable causal masking"
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["latency", "tflops"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.BATCH = args.batch
        self.SEQ_LEN = args.seq_len
        self.H = args.n_heads
        self.D_HEAD = args.d_head
        self.causal = args.causal
        self.sm_scale = 1.0 / math.sqrt(self.D_HEAD)

    def get_input_iter(self) -> Generator:
        SEQ_LEN_LOG2 = 7
        if self.SEQ_LEN is not None:
            seq_lens = [self.SEQ_LEN]
        else:
            seq_lens = [2**i for i in range(SEQ_LEN_LOG2, 16)]

        for seq_len in seq_lens:
            shape = (self.BATCH, self.H, seq_len, self.D_HEAD)
            (q_data, q_scale, _), (k_data, k_scale, _), (v_data, v_scale, _) = (
                generate_attention_inputs(shape, self.device, torch.float8_e4m3fn)
            )
            yield (q_data, k_data, v_data, q_scale, k_scale, v_scale)

    @register_benchmark(enabled=HAS_TLX_MXFP8 and IS_BLACKWELL)
    def tlx_mxfp8_persistent(
        self, q, k, v, q_scale, k_scale, v_scale
    ) -> Callable:
        def fn():
            return tlx_mxfp8_attention(
                q, k, v, q_scale, k_scale, v_scale, self.sm_scale, self.causal
            )

        return fn

    @register_x_val(label="(B, H, SeqLen, D)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int, int]:
        q = example_inputs[0]
        return (q.shape[0], q.shape[1], q.shape[2], q.shape[3])

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        q = example_inputs[0]
        k = example_inputs[1]
        BATCH, H, N_CTX, D_HEAD = q.shape
        N_CTX_KV = k.shape[2]

        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX_KV * D_HEAD
        flops = 2 * flops_per_matmul
        if self.causal:
            flops *= 0.5

        if self.mode == BenchmarkMode.BWD:
            flops *= 2.5
        elif self.mode == BenchmarkMode.FWD_BWD:
            flops *= 3.5
        return flops
