# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import math
import os
from contextlib import nullcontext
from functools import partial
from typing import Callable, Optional, Tuple

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention as sdpa
from tritonbench.kernels.attention_utils import SUPPORT_GLUON

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

from tritonbench.utils.env_utils import IS_BLACKWELL

logger = logging.getLogger(__name__)

# [Optional] flash_attn v2
try:
    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func

    HAS_FLASH_V2 = True
except (ImportError, IOError, AttributeError):
    HAS_FLASH_V2 = False

# [Optional] CuTe
try:
    from mslk.attention.flash_attn.interface import (
        flash_attn_func as facute_flash_attn_func,
    )

    HAS_FLASH_CUTE = True
except (ImportError, IOError, AttributeError):
    HAS_FLASH_CUTE = False
except SystemError as e:
    HAS_FLASH_CUTE = False
    import traceback

    print(f"SystemError resulted from importing FA4: {e.__class__.__name__}: {e}")
    traceback.print_exc()

# [Optional] Vanilla Flash Attention v4
try:
    from flash_attn.cute import flash_attn_func as upstream_fa4_flash_attn_func

    HAS_VANILLA_FA4 = True
except (ImportError, IOError, AttributeError):
    HAS_VANILLA_FA4 = False
except SystemError as e:
    HAS_VANILLA_FA4 = False
    import traceback

    print(
        f"SystemError resulted from importing vanilla FA4: {e.__class__.__name__}: {e}"
    )
    traceback.print_exc()

from ..flash_attention.test_fmha_utils import permute_qkv

# [Optional] xformers backend
try:
    import xformers  # @manual=//fair/xformers:xformers
    import xformers.ops.fmha as xformers_fmha  # @manual=//fair/xformers:xformers

    HAS_XFORMERS = True
except (ImportError, IOError, AttributeError, TypeError):
    HAS_XFORMERS = False

try:
    from mslk.attention.cutlass_blackwell_fmha import cutlass_blackwell_fmha_func

    HAS_CUTLASS_BLACKWELL = True
except (ImportError, IOError, AttributeError, TypeError):
    HAS_CUTLASS_BLACKWELL = False


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
        attention as tlx_blackwell,
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
    parser.add_argument(
        "--n-heads-q-per-kv",
        type=int,
        default=1,
        help="Number of heads per KV group for GQA",
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
    parser.add_argument(
        "--gen-cache-size-inputs",
        action="store_true",
        help="Generate inputs as large as the GPU L2 cache size",
    )
    parser.add_argument(
        "--max-inputs-per-iter",
        type=int,
        default=0,
        help="Max inputs per iteration. This is used when --gen-cache-size-inputs is on.",
    )
    return parser.parse_args(args)


def unpack_inputs(*args):
    inputs = args
    if len(args) == 1 and isinstance(args[0], xformers_fmha.Inputs):
        inp = args[0]
        inputs = (inp.query, inp.key, inp.value)
    return (t.detach() for t in inputs)


def multi_input_wrapper(fn):
    def wrapper(self, *args):
        preproc_fn, benchmark_fn = fn(self, *args)
        arg_len = len(args)
        assert arg_len % 3 == 0
        inputs = []
        all_inputs = []
        for i in range(0, arg_len, 3):
            q, k, v = args[i : i + 3]
            inp = preproc_fn(q, k, v)
            all_inputs += [*unpack_inputs(*inp)]
            inputs.append(inp)

        def multi_input_fn():
            outputs = []
            for i in inputs:
                outputs.append(benchmark_fn(*i))
            return outputs

        self.optims[multi_input_fn] = torch.optim.SGD(all_inputs)

        return multi_input_fn

    wrapper.__name__ = fn.__name__
    return wrapper


def preproc_noop(*args):
    return args


def preproc_permute(q, k, v):
    return [t.contiguous() for t in permute_qkv(q, k, v, perm=(0, 2, 1, 3))]


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
        self.N_HEADS_Q_PER_KV = args.n_heads_q_per_kv
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
        self.gen_cache_size_inputs = args.gen_cache_size_inputs
        self.max_inputs_per_iter = args.max_inputs_per_iter
        self.optims = {}

    @register_benchmark()
    @multi_input_wrapper
    def aten(self, *args) -> Tuple[Callable, Callable]:
        def _inner(q, k, v):
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

        return preproc_noop, _inner

    @register_benchmark()
    @multi_input_wrapper
    def sdpa(self, *args) -> Tuple[Callable, Callable]:
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

        return preproc_noop, sdpa_flash_attention

    @register_benchmark(enabled=HAS_FLASH_V2)
    @multi_input_wrapper
    def flash_v2(self, *args) -> Tuple[Callable, Callable]:
        fn = partial(
            flash_attn_func,
            softmax_scale=self.sm_scale,
            causal=self.causal,
            window_size=self.window_size,
            deterministic=self.deterministic,
        )
        return preproc_permute, fn

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
        return (fhma_input,)

    @register_benchmark(enabled=HAS_CUTLASS_BLACKWELL, label="cutlass-blackwell")
    @multi_input_wrapper
    def cutlass_blackwell(self, *args) -> Tuple[Callable, Callable]:
        fn = partial(
            cutlass_blackwell_fmha_func,
            softmax_scale=self.sm_scale,
            causal=self.causal,
            window_size=self.window_size if self.local else (-1, -1),
            deterministic=self.deterministic,
            bottom_right=False,
        )
        return preproc_permute, fn

    @register_benchmark(enabled=HAS_XFORMERS, fwd_only=True)
    @multi_input_wrapper
    def xformers_splitk(self, *args) -> Tuple[Callable, Callable]:
        if self.local or self.causal:
            # SplitK doesn't support local attention yet
            raise NotImplementedError("Skip")
        need_gradient = not (self.mode == BenchmarkMode.FWD_NO_GRAD)
        xformers_splitk_fhma = xformers_fmha.triton_splitk.FwOp
        fn = partial(xformers_splitk_fhma().apply, needs_gradient=need_gradient)
        return self.xformers_preprocess, fn

    @register_benchmark(
        enabled=IS_BLACKWELL and _is_sdpa_cudnn_attention_available(),
        label=f"cudnn-sdpa-{torch.backends.cudnn.version()}",
    )
    @multi_input_wrapper
    def cudnn_sdpa(self, *args) -> Tuple[Callable, Callable]:
        if self.local:
            # Skip CUDNN SDPA for local attention for now
            raise NotImplementedError("Skip")

        fn = partial(
            _sdpa_cudnn_attention,
            is_causal=self.causal,
            scale=self.sm_scale,
        )
        return preproc_noop, fn

    @register_benchmark(enabled=(IS_BLACKWELL and HAS_FLASH_CUTE), label="FAv4")
    @multi_input_wrapper
    def cutedsl_blackwell(self, *args) -> Tuple[Callable, Callable]:
        fn = partial(
            facute_flash_attn_func,
            softmax_scale=self.sm_scale,
            causal=self.causal,
            window_size=self.window_size if self.local else (None, None),
            deterministic=self.deterministic,
        )
        return preproc_permute, fn

    @register_benchmark(
        enabled=(IS_BLACKWELL and HAS_VANILLA_FA4), label="vanilla-FAv4"
    )
    @multi_input_wrapper
    def vanilla_fa4(self, *args) -> Tuple[Callable, Callable]:
        fn = partial(
            upstream_fa4_flash_attn_func,
            softmax_scale=self.sm_scale,
            causal=self.causal,
            window_size=self.window_size if self.local else (None, None),
        )
        return preproc_permute, fn

    @register_benchmark()
    @multi_input_wrapper
    def flex_attention(self, *args) -> Tuple[Callable, Callable]:
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

        assert len(args) % 3 == 0
        q, k = args[0:2]

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

        fn = partial(flex_attention, block_mask=block_mask)
        return preproc_noop, fn

    # Disable for now due to the smem size problem
    @register_benchmark(
        enabled=False and is_blackwell() and HAS_BLACKWELL_AUTOWS, fwd_only=True
    )
    @multi_input_wrapper
    def triton_tutorial_flash_dp_persistent_blackwell(
        self, *args
    ) -> Tuple[Callable, Callable]:
        def fn(q, k, v):
            return blackwell_triton_tutorial_FA2_dp(
                q,
                k,
                v,
                self.causal,
                self.sm_scale,
                "ws_persistent",
            )

        return preproc_noop, fn

    # Only works with triton main, forward only.
    @register_benchmark(enabled=SUPPORT_GLUON)
    @multi_input_wrapper
    def gluon_blackwell_tutorial_persistent_fwd(
        self, *args
    ) -> Tuple[Callable, Callable]:
        fn = partial(
            gluon_blackwell_persistent_fwd,
            causal=self.causal,
            sm_scale=self.sm_scale,
        )
        return preproc_noop, fn

    # Only works with triton beta, forward only.
    @register_benchmark(enabled=HAS_TLX)
    @multi_input_wrapper
    def tlx_blackwell_ws_pipelined_fwd(self, *args) -> Tuple[Callable, Callable]:
        def fn(q, k, v):
            return tlx_blackwell(
                q,
                k,
                v,
                self.causal,
                self.sm_scale,
                False,
            )

        return preproc_noop, fn

    # Only works with triton beta.
    @register_benchmark(enabled=HAS_TLX)
    @multi_input_wrapper
    def tlx_blackwell_ws_pipelined_persistent(self, *args) -> Tuple[Callable, Callable]:
        def fn(q, k, v):
            return tlx_blackwell(
                q,
                k,
                v,
                self.causal,
                self.sm_scale,
                True,
            )

        return preproc_noop, fn

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        assert len(example_inputs) % 3 == 0
        q, k, v = example_inputs[0:3]
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
        outputs = [input_filter(lambda x: isinstance(x, torch.Tensor), o_) for o_ in o]
        dOs = [torch.rand_like(o_).detach() for o_ in outputs]
        zero_grad = (
            self.optims[fwd_fn].zero_grad
            if fwd_fn in self.optims
            else lambda set_to_none: None
        )

        def fn():
            zero_grad(set_to_none=True)
            for (
                o_tensor,
                do,
            ) in zip(outputs, dOs):
                o_tensor.backward(do, retain_graph=True)

        return fn

    def get_input_iter(self) -> Generator:
        common_kwargs = {
            "dtype": self.dtype,
            "device": self.device,
            "gen_cache_size_inputs": self.gen_cache_size_inputs,
            "max_inputs_per_iter": self.max_inputs_per_iter,
        }
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
                **common_kwargs,
            )
        elif self.input_types == "FA3_PAPER_SHAPES":
            return fa3_paper_inputs(**common_kwargs)
        elif self.input_types == "SWEEP_SHAPES":
            return sweep_inputs(
                D=self.D_HEAD,
                num_heads_q_per_kv=self.N_HEADS_Q_PER_KV,
                **common_kwargs,
            )
        else:
            raise AssertionError(f"Unknown input type {self.input_types}")

    @register_x_val(label="(Batch, Heads, Heads_KV, SeqLen, SeqLen_KV, Dhead)")
    def get_x_val(self, example_inputs) -> str:
        assert len(example_inputs) % 3 == 0
        q, k, v = example_inputs[0:3]
        B, H, S, D = q.shape
        _, H_KV, S_KV, _ = k.shape

        # Add local mask info to the label if enabled
        base_info = f"({B}, {H}, {H_KV}, {S}, {S_KV}, {D})"
        if self.local:
            base_info += f" Local {self.window_size[0]},{self.window_size[1]}"
        if self.causal:
            base_info += " Causal"
        if self.mode in (BenchmarkMode.FWD, BenchmarkMode.FWD_NO_GRAD):
            base_info += f" {BenchmarkMode.FWD.value}"
        else:
            base_info += f" {self.mode.value}"
        return base_info

    def get_latency_scale(self, example_inputs):
        assert len(example_inputs) % 3 == 0
        return len(example_inputs) // 3
