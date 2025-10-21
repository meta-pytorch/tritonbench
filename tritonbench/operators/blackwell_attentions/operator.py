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
from tritonbench.kernels.blackwell_attention_utils import is_blackwell

try:
    from tritonbench.kernels.blackwell_triton_fused_attention import (
        attention_opt as blackwell_triton_tutorial_FA2_opt,
    )
    from tritonbench.kernels.blackwell_triton_fused_attention_dp import (
        attention_opt as blackwell_triton_tutorial_FA2_dp,
    )

    HAS_BLACKWELL_AUTOWS = True
except (ImportError, IOError, AttributeError, TypeError):
    # Needs compiler that supports autoWS
    HAS_BLACKWELL_AUTOWS = False

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

import logging

from tritonbench.utils.env_utils import get_nvidia_gpu_model, is_cuda

logger = logging.getLogger(__name__)

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
    from xformers.ops.fmha import MemoryEfficientAttentionCutlassBlackwellOp

    from ..flash_attention.test_fmha_utils import permute_qkv

    HAS_XFORMERS = True
except (ImportError, IOError, AttributeError, TypeError):
    HAS_XFORMERS = False


try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore

    HAS_TLX = True
except ImportError:
    # suppress type checking errors
    tlx = None

    HAS_TLX = False

if HAS_TLX:
    from tritonbench.kernels.tlx_attention_ws_pipelined import (
        attention as tlx_blackwell_fwd,
    )


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

from .generate_inputs import customized_inputs, fa3_paper_inputs, sweep_inputs

HAS_CUDA_124 = (
    torch.cuda.is_available() and torch.version.cuda and torch.version.cuda >= "12.4"
)

IS_B200 = is_cuda() and "B200" in get_nvidia_gpu_model()


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
        "--window-size",
        type=lambda x: tuple(map(int, x.split(","))),
        default=(-1, -1),
        help="sliding window size as (left_window, right_window). Use (-1, -1) to disable sliding window",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enable deterministic algorithms by calling torch.use_deterministic_algorithms(True)",
    )
    parser.add_argument(
        "--native-sdpa", action="store_true", help="Use SDPA native choice."
    )
    parser.add_argument(
        "--pt2-sdpa", action="store_true", help="Compile SDPA with PT2."
    )
    parser.add_argument("--sm-scale", type=float, default=None, help="softmax scale")
    parser.add_argument(
        "--input-types",
        type=str,
        default="CUSTOMIZED_SHAPES",
        choices=["CUSTOMIZED_SHAPES", "FA3_PAPER_SHAPES", "SWEEP_SHAPES"],
        help="specify input types",
    )
    return parser.parse_args(args)


def _sdpa_cudnn_attention(q, k, v, is_causal=False, scale=False):
    os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        return sdpa(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=scale,
        )


def _is_sdpa_cudnn_attention_available():
    q = torch.randn(1, 4, 8, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    try:
        _sdpa_cudnn_attention(q, k, v)
        return True
    except RuntimeError as e:
        return False


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["latency", "tflops", "tbps"]

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
        self.window_size = args.window_size
        self.local = self.window_size != (-1, -1)

        # Prioritize sliding window over causal when both are specified
        if self.causal and self.local:
            self.causal = False

        # Enable deterministic algorithms if requested
        if args.deterministic:
            torch.use_deterministic_algorithms(True)
            logger.warning(
                "--deterministic is on. Some operators might not support "
                "deterministic runs (we guarantee that Flash Attention v2 "
                "Cutlass Attention support this mode)"
            )
        else:
            torch.use_deterministic_algorithms(False)

        self.native_sdpa = args.native_sdpa
        self.pt2_sdpa = args.pt2_sdpa
        self.input_types = args.input_types
        self.sm_scale = args.sm_scale if args.sm_scale else 1.0 / math.sqrt(self.D_HEAD)
        self.deterministic = args.deterministic

    @register_benchmark()
    def aten(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        def _inner():
            N_CTX = q.shape[2]
            N_CTX_KV = k.shape[2]
            p = torch.matmul(q, k.transpose(2, 3)) * self.sm_scale

            if self.causal:
                M = torch.tril(torch.ones((N_CTX, N_CTX_KV), device=self.device))
                p[:, :, M == 0] = float("-inf")
            elif self.local:
                # Create sliding window mask
                i = torch.arange(N_CTX, device=self.device).unsqueeze(1)
                j = torch.arange(N_CTX_KV, device=self.device).unsqueeze(0)
                # Allow attention if within window (both left and right)
                left_window, right_window = self.window_size
                window_mask = (i - j) <= left_window & ((j - i) <= right_window)
                # Note: causal is already handled separately above and should not be true when sliding_window is true
                p[:, :, ~window_mask] = float("-inf")

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
        if self.local:
            # sdpa with flash attention backend doesn't support non-null attn_mask
            raise NotImplementedError("Skip")

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
            qkv,
            softmax_scale=self.sm_scale,
            causal=self.causal,
            window_size=self.window_size,
            deterministic=self.deterministic,
        )
        return fn

    def xformers_preprocess(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        q_1, k_1, v_1 = permute_qkv(q, k, v, perm=(0, 2, 1, 3))
        # Make sure that inputs are contiguous
        q_1 = q_1.contiguous()
        k_1 = k_1.contiguous()
        v_1 = v_1.contiguous()

        # Create attention bias based on settings
        attn_bias = None
        if self.causal:
            attn_bias = xformers.ops.LowerTriangularMask()
        elif self.local:
            attn_bias = xformers.ops.fmha.attn_bias.LocalAttentionFromBottomRightMask(
                window_left=self.window_size[0],
                window_right=self.window_size[1],
            )

        fhma_input = xformers_fmha.Inputs(
            query=q_1, key=k_1, value=v_1, attn_bias=attn_bias, scale=self.sm_scale
        )
        return fhma_input

    @register_benchmark(enabled=HAS_XFORMERS, label="cutlass-blackwell")
    def cutlass_blackwell(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        fhma_input = self.xformers_preprocess(q, k, v)

        return lambda: xformers.ops.fmha._memory_efficient_attention(
            fhma_input,
            op=MemoryEfficientAttentionCutlassBlackwellOp,
        )

    @register_benchmark(enabled=HAS_XFORMERS, fwd_only=True)
    def xformers_splitk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        if self.local or self.causal:
            # SplitK doesn't support local attention yet
            raise NotImplementedError("Skip")
        need_gradient = not (self.mode == BenchmarkMode.FWD_NO_GRAD)
        fhma_input = self.xformers_preprocess(q, k, v)
        xformers_splitk_fhma = xformers_fmha.triton_splitk.FwOp
        return lambda: xformers_splitk_fhma().apply(
            fhma_input, needs_gradient=need_gradient
        )

    @register_benchmark(
        enabled=IS_B200 and _is_sdpa_cudnn_attention_available(),
        label=f"cudnn-sdpa-{torch.backends.cudnn.version()}",
    )
    def cudnn_sdpa(self, q, k, v):
        if self.local:
            # Skip CUDNN SDPA for local attention for now
            raise NotImplementedError("Skip")

        return lambda: _sdpa_cudnn_attention(
            q, k, v, is_causal=self.causal, scale=self.sm_scale
        )

    @register_benchmark(
        enabled=(IS_B200 and HAS_FLASH_CUTE), label="FAv4", fwd_only=True
    )
    def cutedsl_blackwell(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Callable:
        # [B, H, S, D] -> [B, S, H, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        return lambda: facute_flash_attn_func(
            q,
            k,
            v,
            softmax_scale=self.sm_scale,
            causal=self.causal,
            window_size=self.window_size if self.local else (None, None),
        )

    @register_benchmark()
    def flex_attention(self, q, k, v):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def local_mask(b, h, q_idx, kv_idx):
            # Left window check: allow tokens within left_window_size lookback
            left_ok = q_idx - kv_idx <= self.window_size[0]
            # Right window check: allow tokens within right_window_size lookahead
            right_ok = kv_idx - q_idx <= self.window_size[1]
            return left_ok & right_ok

        flex_attention = torch.compile(flex_attention, dynamic=False)

        B, H, S, D = q.shape
        _, _, S_KV, _ = k.shape

        mask_mod = None
        if self.causal:
            mask_mod = causal_mask
        elif self.local:
            mask_mod = local_mask

        if mask_mod:
            block_mask = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S_KV
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

    @register_benchmark(enabled=False)
    def triton_tutorial_flash_v2_blackwell(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: blackwell_triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "ws"
        )

    @register_benchmark(enabled=False)
    def triton_tutorial_flash_v2_persistent_blackwell(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: blackwell_triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "ws_persistent"
        )

    @register_benchmark(enabled=False)
    def triton_tutorial_flash_dp_blackwell(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: blackwell_triton_tutorial_FA2_dp(
            q, k, v, self.causal, self.sm_scale, "ws"
        )

    @register_benchmark(enabled=is_blackwell() and HAS_BLACKWELL_AUTOWS)
    def triton_tutorial_flash_dp_persistent_blackwell(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: blackwell_triton_tutorial_FA2_dp(
            q, k, v, self.causal, self.sm_scale, "ws_persistent"
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
    @register_benchmark(enabled=SUPPORT_GLUON)
    def gluon_blackwell_tutorial_persistent_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: gluon_blackwell_persistent_fwd(
            q, k, v, self.causal, self.sm_scale
        )

    # Only works with triton beta, forward only.
    @register_benchmark(enabled=HAS_TLX)
    def tlx_blackwell_ws_pipelined_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: tlx_blackwell_fwd(q, k, v, self.sm_scale, False)

    # Only works with triton beta, forward only.
    @register_benchmark(enabled=HAS_TLX)
    def tlx_blackwell_ws_pipelined_persistent_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        return lambda: tlx_blackwell_fwd(q, k, v, self.sm_scale, True)

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        q, k, v = example_inputs
        BATCH, H, N_CTX, D_HEAD = q.shape
        _, _, N_CTX_KV, _ = k.shape

        if not self.local:
            flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX_KV * D_HEAD
            flops = 2 * flops_per_matmul
            if self.causal:
                flops *= 0.5
        else:
            row_idx = torch.arange(N_CTX, device="cuda")
            col_left = torch.maximum(
                row_idx + N_CTX_KV - N_CTX - self.window_size[0], torch.tensor(0)
            )
            col_right = torch.minimum(
                row_idx + N_CTX_KV - N_CTX + self.window_size[1],
                torch.tensor(N_CTX_KV - 1),
            )
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
            flops = 2 * 2.0 * BATCH * H * N_CTX * avg_seqlen * D_HEAD

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
        elif self.input_types == "SWEEP_SHAPES":
            return sweep_inputs(
                dtype=self.dtype,
                device=self.device,
            )
        else:
            raise AssertionError(f"Unknown input type {self.input_types}")

    @register_x_val(label="(Batch, Heads, Heads_KV, SeqLen, SeqLen_KV, Dhead)")
    def get_x_val(self, example_inputs) -> str:
        q, k, v = example_inputs
        B, H, S, D = q.shape
        _, H_KV, S_KV, _ = k.shape

        # Add local mask info to the label if enabled
        base_info = f"({B}, {H}, {H_KV}, {S}, {S_KV}, {D})"
        if self.local:
            base_info += f" Local {self.window_size[0]},{self.window_size[1]}"
        if self.causal:
            base_info += " Causal"
        return base_info
