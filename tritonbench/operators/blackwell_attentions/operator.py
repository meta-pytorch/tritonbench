# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import math
import os
from contextlib import nullcontext

from typing import Callable, Optional

import torch

from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention as sdpa

from tritonbench.kernels.attention_utils import SUPPORT_GLUON

from tritonbench.kernels.triton_fused_attention import (
    attention_opt as triton_tutorial_FA2_opt,
)

if SUPPORT_GLUON:
    from tritonbench.kernels.gluon_attention_forward import (
        attention_forward as gluon_blackwell_fwd,
    )
    from tritonbench.kernels.gluon_attention_persistent_forward import (
        attention_forward as gluon_blackwell_persistent_fwd,
    )

from tritonbench.utils.env_utils import get_nvidia_gpu_model, is_cuda

# [Optional] flash_attn v2
try:
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func as flash_attn_func,
    )

    from ..flash_attention.test_fmha_utils import make_packed_qkv

    HAS_FLASH_V2 = True
except (ImportError, IOError, AttributeError):
    HAS_FLASH_V2 = False

# [Optional] CuTe
try:
    from flash_attn.cute.interface import flash_attn_func as facute_flash_attn_func

    HAS_FLASH_CUTE = True
except (ImportError, IOError, AttributeError):
    HAS_FLASH_CUTE = False

# [Optional] xformers backend
try:
    import xformers  # @manual=//fair/xformers:xformers
    import xformers.ops.fmha as xformers_fmha  # @manual=//fair/xformers:xformers

    from ..flash_attention.test_fmha_utils import permute_qkv

    HAS_XFORMERS = True
except (ImportError, IOError, AttributeError, TypeError):
    HAS_XFORMERS = False

from typing import Any, Generator, List

from tritonbench.utils.input import input_filter

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
    register_x_val,
)

from .generate_inputs import customized_inputs, fa3_paper_inputs

HAS_CUDA_124 = (
    torch.cuda.is_available() and torch.version.cuda and torch.version.cuda >= "12.4"
)

IS_B200 = is_cuda() and get_nvidia_gpu_model() == "NVIDIA B200"


CUSTOMIZED_SHAPES = "CUSTOMIZED_SHAPES"
FA3_PAPER_SHAPES = "FA3_PAPER_SHAPES"


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length q")
    parser.add_argument(
        "--seq-len-kv", type=int, default=None, help="Sequence length kv"
    )
    parser.add_argument("--n-heads", type=int, default=48, help="Number of heads")
    parser.add_argument(
        "--n-heads-kv", type=int, default=None, help="Number of heads kv"
    )
    parser.add_argument("--d-head", type=int, default=64, help="specify head dimension")
    parser.add_argument(
        "--causal",
        action="store_true",
        help="enable causal",
    )
    parser.add_argument(
        "--native-sdpa", action="store_true", help="Use SDPA native choice."
    )
    parser.add_argument(
        "--pt2-sdpa", action="store_true", help="Compile SDPA with PT2."
    )
    parser.add_argument(
        "--sm-scale", type=Optional[float], default=None, help="softmax scale"
    )
    parser.add_argument(
        "--input-types",
        type=str,
        default=CUSTOMIZED_SHAPES,
        choices=[CUSTOMIZED_SHAPES, FA3_PAPER_SHAPES],
        help="specify input types",
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.BATCH = args.batch
        self.SEQ_LEN = args.seq_len
        self.SEQ_LEN_KV = (
            args.seq_len_kv if args.seq_len_kv is not None else args.seq_len
        )
        self.N_HEAD_KV = (
            args.n_heads_kv if args.n_heads_kv is not None else args.n_heads
        )
        self.H = args.n_heads
        self.D_HEAD = args.d_head
        self.causal = args.causal
        self.native_sdpa = args.native_sdpa
        self.pt2_sdpa = args.pt2_sdpa
        self.input_types = args.input_types
        self.sm_scale = args.sm_scale if args.sm_scale else 1.0 / math.sqrt(self.D_HEAD)

    @register_benchmark()
    def aten(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        def _inner():
            N_CTX = q.shape[2]
            M = torch.tril(torch.ones((N_CTX, N_CTX), device=self.device))
            p = torch.matmul(q, k.transpose(2, 3)) * self.sm_scale
            if self.causal:
                p[:, :, M == 0] = float("-inf")
            p = torch.softmax(p.float(), dim=-1).to(q.dtype)
            # p = torch.exp(p)
            ref_out = torch.matmul(p, v)
            return ref_out

        return _inner

    @register_benchmark()
    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        def sdpa_flash_attention(q, k, v):
            cxt = (
                nullcontext()
                if self.native_sdpa
                else sdpa_kernel([SDPBackend.FLASH_ATTENTION])
            )
            with cxt:
                sdpa_impl = (
                    torch.compile(
                        sdpa,
                        fullgraph=True,
                        backend="inductor",
                        mode="max-autotune",
                    )
                    if self.pt2_sdpa
                    else sdpa
                )
                return sdpa_impl(
                    q,
                    k,
                    v,
                    is_causal=self.causal,
                    scale=self.sm_scale,
                )

        return lambda: sdpa_flash_attention(
            q,
            k,
            v,
        )

    @register_benchmark(enabled=HAS_FLASH_V2)
    def flash_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        qkv = make_packed_qkv(q, k, v)
        fn = lambda: flash_attn_func(
            qkv, softmax_scale=self.sm_scale, causal=self.causal
        )
        return fn

    def xformers_preprocess(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        q_1, k_1, v_1 = permute_qkv(q, k, v, perm=(0, 2, 1, 3))
        attn_bias = xformers.ops.LowerTriangularMask() if self.causal else None
        fhma_input = xformers_fmha.Inputs(
            query=q_1, key=k_1, value=v_1, attn_bias=attn_bias, scale=self.sm_scale
        )
        return fhma_input

    @register_benchmark(enabled=HAS_XFORMERS)
    def xformers(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        need_gradient = not (self.mode == BenchmarkMode.FWD_NO_GRAD)
        fhma_input = self.xformers_preprocess(q, k, v)
        xformers_cutlass_fhma = xformers.ops.fmha.cutlass.FwOp
        return lambda: xformers_cutlass_fhma().apply(
            fhma_input, needs_gradient=need_gradient
        )

    @register_benchmark(enabled=HAS_XFORMERS, fwd_only=True)
    def xformers_splitk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        need_gradient = not (self.mode == BenchmarkMode.FWD_NO_GRAD)
        fhma_input = self.xformers_preprocess(q, k, v)
        xformers_splitk_fhma = xformers_fmha.triton_splitk.FwOp
        return lambda: xformers_splitk_fhma().apply(
            fhma_input, needs_gradient=need_gradient
        )

    @register_benchmark(
        enabled=IS_B200, label=f"cudnn-sdpa-{torch.backends.cudnn.version()}"
    )
    def cudnn_sdpa(self, q, k, v):
        os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

        def sdpa_flash_attention(q, k, v):
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                return sdpa(
                    q,
                    k,
                    v,
                    is_causal=self.causal,
                    scale=self.sm_scale,
                )

        return lambda: sdpa_flash_attention(
            q,
            k,
            v,
        )

    @register_benchmark(
        enabled=(IS_B200 and HAS_FLASH_CUTE), label=f"cutedsl-blackwell", fwd_only=True
    )
    def cutedsl_blackwell(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Callable:
        # [B, H, S, D] -> [B, S, H, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        return lambda: facute_flash_attn_func(
            q, k, v, softmax_scale=self.sm_scale, causal=self.causal
        )

    @register_benchmark()
    def flex_attention(self, q, k, v):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        flex_attention = torch.compile(flex_attention, dynamic=False)

        if self.causal:
            B, H, S, D = q.shape
            block_mask = create_block_mask(
                causal_mask, B=None, H=None, Q_LEN=S, KV_LEN=S
            )
        else:
            block_mask = None

        return lambda: flex_attention(q, k, v, block_mask=block_mask)

    @register_benchmark(enabled=False)
    def triton_tutorial_flash_v2_tma_ws_persistent_blackwell(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "tma_ws_persistent_blackwell"
        )

    # Only works with triton main, forward only.
    @register_benchmark(enabled=False)
    def gluon_blackwell_tutorial_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: gluon_blackwell_fwd(q, k, v, self.causal, self.sm_scale)

    # Only works with triton main, forward only.
    @register_benchmark(enabled=False)
    def gluon_blackwell_tutorial_persistent_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: gluon_blackwell_persistent_fwd(
            q, k, v, self.causal, self.sm_scale
        )

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        q, k, v = example_inputs
        BATCH, H, N_CTX, D_HEAD = q.shape
        _, _, N_CTX_KV, _ = k.shape
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX_KV * D_HEAD
        flops = 2 * flops_per_matmul
        if self.causal:
            flops *= 0.5
        if self.mode == BenchmarkMode.BWD:
            flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        elif self.mode == BenchmarkMode.FWD_BWD:
            flops *= 3.5  # 1.0(fwd) + 2.0(bwd) + 0.5(recompute)
        return flops

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        o = fwd_fn()
        o_tensor = input_filter(
            lambda x: isinstance(x, torch.Tensor),
            o,
        )
        do = torch.rand_like(o_tensor)
        fn = lambda: o_tensor.backward(do, retain_graph=True)
        return fn

    def get_input_iter(self) -> Generator:
        if self.input_types == "CUSTOMIZED_SHAPES":
            return customized_inputs(
                shape=(
                    self.BATCH,
                    self.H,
                    self.N_HEAD_KV,
                    self.SEQ_LEN,
                    self.SEQ_LEN_KV,
                    self.D_HEAD,
                ),
                num_inputs=self.tb_args.num_inputs,
                dtype=self.dtype,
                device=self.device,
            )
        elif self.input_types == "FA3_PAPER_SHAPES":
            return fa3_paper_inputs(
                dtype=self.dtype,
                device=self.device,
            )

        else:
            raise AssertionError(f"Unknown input type {self.input_types}")

    @register_x_val(label="(Batch, Heads, Heads_KV, SeqLen, SeqLen_KV, Dhead)")
    def get_x_val(self, example_inputs) -> float:
        q, k, v = example_inputs
        B, H, S, D = q.shape
        _, H_KV, S_KV, _ = k.shape
        return (B, H, H_KV, S, S_KV, D)
