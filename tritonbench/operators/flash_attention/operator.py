# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This benchmark script is based on the benchmark code from:
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

It benchmarks the following FMHA kernels:

* Triton-Flash-V2: the triton version of FA-V2:

  https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

* SDPA: the torch.nn.attention version of FA-V2

* [optional] Flash-V2: the FA-V2 from //ai_codesign/gen_ai/flash_attention_v2:flash_attention_v2,
  which was imported from https://github.com/Dao-AILab/flash-attention

* [optional] Xformers: the memory-efficient attention from xformers:

  https://fburl.com/code/cuorcm9h

* [optional] Xformers-Splitk: the triton-splitk FMHA kernel from xformers:

  https://fburl.com/code/awt36vjj
  Disabled by default because it failed with some configs. Note that
  the relevant benchmark only works with causal = False at the moment.
  Known to work with "--batch=8 --n-heads=8 --xformers-splitk"
"""

import argparse
import os
from contextlib import nullcontext
from itertools import chain

from typing import Callable, Optional

import torch
import triton  # @manual=//triton:triton

from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention as sdpa

from tritonbench.kernels.triton_fused_attention import (
    attention_opt as triton_tutorial_FA2_opt,
)

from tritonbench.utils.path_utils import add_ld_library_path
from tritonbench.utils.triton_op import is_fbcode


# [Optional] flash_attn v2
try:
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func as flash_attn_func,
    )

    from .test_fmha_utils import make_packed_qkv

    HAS_FLASH_V2 = True
except (ImportError, IOError, AttributeError):
    HAS_FLASH_V2 = False

HAS_CUDA_124 = (
    torch.cuda.is_available() and torch.version.cuda and torch.version.cuda >= "12.4"
)

# [Optional] flash_attn v3
try:
    torch_lib_path = os.path.join(os.path.dirname(__file__), "lib")
    with add_ld_library_path(torch_lib_path):
        from flash_attn_interface import flash_attn_func as flash_attn_v3
    HAS_FLASH_V3 = True
except (ImportError, IOError, AttributeError):
    try:
        from fa3.hopper.flash_attn_interface import flash_attn_func as flash_attn_v3

        HAS_FLASH_V3 = True
    except (ImportError, IOError, AttributeError):
        HAS_FLASH_V3 = False

# [Optional] xformers backend
try:
    import xformers  # @manual=//fair/xformers:xformers
    import xformers.ops.fmha as xformers_fmha  # @manual=//fair/xformers:xformers

    from .test_fmha_utils import permute_qkv

    HAS_XFORMERS = True
except (ImportError, IOError, AttributeError, TypeError):
    HAS_XFORMERS = False

try:
    import tilelang

    from .tilelang_mha import tilelang_mha

    HAS_TILELANG = True
except (ImportError, IOError, AttributeError, TypeError):
    HAS_TILELANG = False

# [Optional] ThunderKittens backend
try:
    from .tk import tk_attn

    HAS_TK = True
except (ImportError, IOError, AttributeError):
    HAS_TK = False

# [Optional] JAX Pallas backend
try:
    import jax

    from tritonbench.utils.jax_utils import torch_to_jax_tensor

    from .pallas import mha as pallas_mha

    HAS_PALLAS = True
except (ImportError, IOError, AttributeError):
    HAS_PALLAS = False

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
from tritonbench.utils.triton_utils import has_warp_spec


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument("--n-heads", type=int, default=48, help="Number of heads")
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
        "--additional-inputs", action="store_true", help="enable additional inputs"
    )
    parser.add_argument(
        "--ragged-shapes", action="store_true", help="enable additional inputs"
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
        self.H = args.n_heads
        self.D_HEAD = args.d_head
        self.N_CTX = None
        self.causal = args.causal
        self.native_sdpa = args.native_sdpa
        self.pt2_sdpa = args.pt2_sdpa
        self.additional_inputs = args.additional_inputs
        self.ragged_shapes = args.ragged_shapes
        self.sm_scale = 1.3

    @register_benchmark()
    def aten(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        def _inner():
            M = torch.tril(torch.ones((self.N_CTX, self.N_CTX), device=self.device))
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

    @register_benchmark(enabled=HAS_FLASH_V3, baseline=True)
    def flash_v3(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # [B, H, S, D] -> [B, S, H, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        fn = lambda: flash_attn_v3(q, k, v, self.sm_scale, self.causal)
        return fn

    @register_benchmark()
    def triton_tutorial_flash_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # base: do not enable TMA/WarpSpec/CompPipe
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "base"
        )

    @register_benchmark(enabled=HAS_CUDA_124)
    def triton_tutorial_flash_v2_opt(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # autotune CompPipe
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "opt"
        )

    @register_benchmark(enabled=HAS_CUDA_124)
    def triton_tutorial_flash_v2_tma(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # autotune TMA/CompPipe
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "tma"
        )

    @register_benchmark(enabled=HAS_CUDA_124 and has_warp_spec())
    def triton_tutorial_flash_v2_ws(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # autotune WarpSpec/CompPipe
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "ws"
        )

    @register_benchmark(enabled=HAS_CUDA_124 and has_warp_spec())
    def triton_tutorial_flash_v2_tma_ws(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # autotune TMA/WarpSpec/CompPipe
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "tma_ws"
        )

    @register_benchmark(enabled=HAS_CUDA_124 and has_warp_spec())
    def triton_tutorial_flash_v2_tma_ws_persistent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # autotune TMA/WarpSpec/CompPipe/Persistent
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "tma_ws_persistent"
        )

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

    @register_benchmark(enabled=not is_fbcode() and HAS_TK)
    def tk(self, q, k, v):
        def _inner():
            out = tk_attn(q, k, v, self.causal)
            return out[0]

        return _inner

    @register_benchmark(enabled=HAS_PALLAS)
    def pallas(self, q, k, v):
        q = torch_to_jax_tensor(q)
        k = torch_to_jax_tensor(k)
        v = torch_to_jax_tensor(v)

        def _inner():
            pallas_mha(q, k, v, segment_ids=None)
            jax.device_put(0.0).block_until_ready()

        return _inner

    @register_benchmark(enabled=HAS_TILELANG)
    def tile(self, q, k, v):
        # [B, H, S, D] -> [B, S, H, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        best_config = tilelang_mha(
            self.BATCH,
            self.H,
            self.N_CTX,
            self.D_HEAD,
            self.causal,
            self.dtype,
            tune=True,
        )[1]
        func = tilelang_mha(
            self.BATCH,
            self.H,
            self.N_CTX,
            self.D_HEAD,
            self.causal,
            self.dtype,
        )(*best_config)
        jit_kernel = tilelang.compile(func, out_idx=[3])

        def _inner():
            o = jit_kernel(q, k, v)
            return o

        return _inner

    @register_benchmark(enabled=False, label=f"cudnn-{torch.backends.cudnn.version()}")
    def cudnn(self, q, k, v):
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

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        q, k, v = example_inputs
        BATCH, H, N_CTX, D_HEAD = q.shape
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
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
        D_HEAD = self.D_HEAD
        BATCH = self.BATCH
        H = self.H
        SEQ_LEN_LOG2 = 7

        def get_ctx_vals():
            if self.SEQ_LEN:
                SEQ_LEN = self.SEQ_LEN
                for _i in range(self.tb_args.num_inputs):
                    yield (BATCH, self.H, SEQ_LEN, self.D_HEAD)
                    SEQ_LEN *= 2
                return
            for i in range(SEQ_LEN_LOG2, 15):
                N_CTX = 2**i
                # BATCH = 16384 // N_CTX
                # H = 2048 // D_HEAD
                yield (BATCH, H, N_CTX, D_HEAD)

        ctx_vals = get_ctx_vals()

        if self.ragged_shapes:
            shapes = self.__ragged_shapes()
        elif self.additional_inputs:
            shapes = self.__additional_example_input(ctx_vals)
        else:
            shapes = ctx_vals
        requires_grad = True
        for shape in shapes:
            BATCH, H, N_CTX, D_HEAD = shape
            q = torch.randn(
                (BATCH, H, N_CTX, D_HEAD),
                dtype=self.dtype,
                device=self.device,
                requires_grad=requires_grad,
            )
            k = torch.randn(
                (BATCH, H, N_CTX, D_HEAD),
                dtype=self.dtype,
                device=self.device,
                requires_grad=requires_grad,
            )
            v = torch.randn(
                (BATCH, H, N_CTX, D_HEAD),
                dtype=self.dtype,
                device=self.device,
                requires_grad=requires_grad,
            )
            self.N_CTX = N_CTX
            yield (q, k, v)

    def __additional_example_input(self, standard_shapes: Generator) -> Generator:
        llama_shapes = [
            (4, 32, 19, 128),
            (4, 32, 1, 128),
            # currently we are only able to use the same shape for q, k, v but in prod q shape is (4, 32, 1, 128) here
            (4, 32, 511, 128),
        ]
        shapes = chain(standard_shapes, llama_shapes)
        if self.add_production_shapes:
            from ...utils.fb.durin_data import productionDataLoader

            shapes = chain(
                shapes,
                productionDataLoader.get_shapes_from_frozen_durin(
                    self.name, "attention", shuffle_shapes=self.tb_args.shuffle_shapes
                ),
            )
        return shapes

    def __ragged_shapes(self) -> Generator:
        additional_shapes = [
            (1024, 4, 1024, 128),
            (256, 4, 256, 128),
            (256, 4, 512, 128),
            (256, 4, 1024, 128),
            (256, 4, 2048, 128),
            (256, 4, 4096, 128),
            (256, 4, 8192, 128),
            (256, 4, 16384, 128),
        ]
        return chain(additional_shapes)

    @register_x_val(label="(Batch, Heads, SeqLen, Dhead)")
    def get_x_val(self, example_inputs) -> float:
        q, k, v = example_inputs
        B, H, S, D = q.shape
        return (B, H, S, D)

    def plot(self):
        y_metric_name = "tflops"

        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["N_CTX"],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "aten",
                    "sdpa",
                    "flash_v2",
                    "triton_tutorial_flash_v2",
                    "xformers",
                    "hw_roofline",
                ],  # possible values for `line_arg``
                line_names=[
                    "ATen",
                    "SDPA",
                    "Flash V2",
                    "Triton Tutorial Flash V2",
                    "XFormers",
                    "Hardware Roofline",
                ],  # label name for the lines
                styles=[
                    ("blue", "-"),
                    ("yellow", "-"),
                    ("green", "-"),
                    ("red", "-"),
                    ("brown", "-"),
                    ("purple", "-"),
                    ("black", "dashed"),
                ],  # line styles
                ylabel=y_metric_name,  # label name for the y-axis
                plot_name="flashattention-tflops",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(N_CTX, provider):
            tflops = self.output.get_y_vals(N_CTX, provider, y_metric_name)
            return tflops

        _plot.run(
            show_plots=True, print_data=False, save_path="/tmp/test_flashattention"
        )
