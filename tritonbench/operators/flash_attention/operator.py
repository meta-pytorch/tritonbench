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
import logging
import os
from contextlib import nullcontext
from functools import partial

from typing import Callable, Optional

import torch
import triton  # @manual=//triton:triton

from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention as sdpa

from tritonbench.kernels.proton_blackwell_ws_fused_attention import (
    attention_opt as proton_blackwell_ws_FA2_opt,
)

from tritonbench.kernels.proton_fused_attention import (
    attention_opt as proton_tutorial_FA2_opt,
)

from tritonbench.kernels.triton_fused_attention import (
    attention_opt as triton_tutorial_FA2_opt,
)

from tritonbench.utils.env_utils import get_nvidia_gpu_model, is_cuda, is_hip
from tritonbench.utils.path_utils import add_ld_library_path
from tritonbench.utils.python_utils import try_import
from tritonbench.utils.triton_op import is_fbcode

from .generate_inputs import (
    additional_inputs,
    ragged_inputs,
    standard_inputs,
    sweep_inputs,
)

logger = logging.getLogger(__name__)

# [Optional] flash_attn v2
with try_import("HAS_FLASH_V2"):
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func as flash_attn_func,
    )

    from .test_fmha_utils import make_packed_qkv

HAS_CUDA_124 = (
    torch.cuda.is_available() and torch.version.cuda and torch.version.cuda >= "12.4"
)

IS_B200 = is_cuda() and get_nvidia_gpu_model() == "NVIDIA B200"

# only enabling the variants known to be working on B200 (trunk).
if not IS_B200:
    # [Optional] flash_attn v3
    with try_import("HAS_FLASH_V3"):
        try:
            torch_lib_path = os.path.join(os.path.dirname(__file__), "lib")
            with add_ld_library_path(torch_lib_path):
                from flash_attn_interface import flash_attn_func as flash_attn_v3
        except (ImportError, IOError, AttributeError):
            from fa3.hopper.flash_attn_interface import flash_attn_func as flash_attn_v3

    with try_import("HAS_TILELANG"):
        import tilelang

        from .tilelang_mha import tilelang_mha

    # [Optional] ThunderKittens backend
    with try_import("HAS_TK"):
        from .tk import tk_attn

    # [Optional] JAX Pallas backend
    with try_import("HAS_PALLAS"):
        import jax
        from tritonbench.utils.jax_utils import torch_to_jax_tensor

        from .pallas import mha as pallas_mha

# [Optional] xformers backend
with try_import("HAS_XFORMERS"):
    import xformers  # @manual=//fair/xformers:xformers
    import xformers.ops.fmha as xformers_fmha  # @manual=//fair/xformers:xformers

    from .test_fmha_utils import permute_qkv

from typing import Any, Generator, List, Tuple

from tritonbench.utils.input import input_filter

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
    register_x_val,
)
from tritonbench.utils.triton_utils import has_new_tma, has_warp_spec


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length q")
    parser.add_argument(
        "--seq-len-kv", type=int, default=None, help="Sequence length kv"
    )
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
        "--input-types",
        type=str,
        default="STANDARD_SHAPES",
        choices=(
            "STANDARD_SHAPES",
            "RAGGED_SHAPES",
            "ADDITIONAL_SHAPES",
            "SWEEP_SHAPES",
        ),
        help="specify input types",
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Run with deterministic mode."
    )
    parser.add_argument(
        "--gen-cache-size-inputs",
        action="store_true",
        help="Generate inputs as large as the GPU L2 cache size",
    )
    return parser.parse_args(args)


def multi_input_wrapper(fn):
    def wrapper(self, *args):
        preproc_fn, benchmark_fn = fn(self, *args)
        arg_len = len(args)
        assert arg_len % 3 == 0
        inputs = []
        for i in range(0, arg_len, 3):
            q, k, v = args[i : i + 3]
            inp = preproc_fn(q, k, v)
            inputs.append(inp)

        def multi_input_fn():
            outputs = []
            for i in inputs:
                outputs.append(benchmark_fn(*i))
            return outputs

        return multi_input_fn

    wrapper.__name__ = fn.__name__
    return wrapper


def preproc_noop(*args):
    return args


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
        self.H = args.n_heads
        self.D_HEAD = args.d_head
        self.N_CTX = None
        self.causal = args.causal
        self.native_sdpa = args.native_sdpa
        self.pt2_sdpa = args.pt2_sdpa
        # Use standard scale factor: 1/sqrt(head_dim)
        self.sm_scale = 1.0 / (self.D_HEAD**0.5)
        self.input_types = args.input_types
        self.deterministic = args.deterministic
        if self.deterministic:
            logger.warning(
                "--deterministic is on. Some operators might not support "
                "deterministic runs (we guarantee that Flash Attention v2 and "
                "v3 support this mode)"
            )
            torch.use_deterministic_algorithms(True)
        self.gen_cache_size_inputs = args.gen_cache_size_inputs

    @register_benchmark(baseline=True)
    @multi_input_wrapper
    def aten(self, *args) -> Tuple[Callable, Callable]:
        def _inner(q, k, v):
            seq_len = q.shape[2]
            M = torch.tril(torch.ones((seq_len, seq_len), device=self.device))
            p = torch.matmul(q, k.transpose(2, 3)) * self.sm_scale
            if self.causal:
                p[:, :, M == 0] = float("-inf")
            p = torch.softmax(p.float(), dim=-1).to(q.dtype)
            # p = torch.exp(p)
            ref_out = torch.matmul(p, v)
            return ref_out

        return preproc_noop, _inner

    @register_benchmark()
    @multi_input_wrapper
    def sdpa(self, *args) -> Tuple[Callable, Callable]:
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

    @register_benchmark(enabled=HAS_FLASH_V2)  # noqa
    @multi_input_wrapper
    def flash_v2(self, *args) -> Tuple[Callable, Callable]:
        def preproc(q, k, v):
            return (make_packed_qkv(q, k, v),)

        fn = partial(
            flash_attn_func,
            softmax_scale=self.sm_scale,
            causal=self.causal,
            deterministic=self.deterministic,
        )
        return preproc, fn

    @register_benchmark()
    @multi_input_wrapper
    def triton_tutorial_flash_v2(self, *args) -> Tuple[Callable, Callable]:
        def fn(q, k, v):
            # includes base (default scheduling) + opt (optimized loop scheduling based on heuristics)
            return triton_tutorial_FA2_opt(
                q, k, v, self.causal, self.sm_scale, "base_opt"
            )

        return preproc_noop, fn

    @register_benchmark(enabled=HAS_CUDA_124 and has_new_tma())
    @multi_input_wrapper
    def triton_tutorial_flash_v2_tma(self, *args) -> Tuple[Callable, Callable]:
        def fn(q, k, v):
            # autotune TMA/CompPipe
            return triton_tutorial_FA2_opt(q, k, v, self.causal, self.sm_scale, "tma")

        return preproc_noop, fn

    def xformers_preprocess(self, q, k, v):
        q_1, k_1, v_1 = permute_qkv(q, k, v, perm=(0, 2, 1, 3))
        attn_bias = xformers.ops.LowerTriangularMask() if self.causal else None
        fhma_input = xformers_fmha.Inputs(
            query=q_1, key=k_1, value=v_1, attn_bias=attn_bias, scale=self.sm_scale
        )
        return fhma_input

    # Cutlass implementation is not supported on AMD GPUs.
    @register_benchmark(enabled=HAS_XFORMERS and not is_hip())  # noqa
    @multi_input_wrapper
    def xformers(self, *args) -> Tuple[Callable, Callable]:
        need_gradient = not (self.mode == BenchmarkMode.FWD_NO_GRAD)
        xformers_cutlass_fhma = xformers.ops.fmha.cutlass.FwOp

        def preproc(q, k, v):
            return (self.xformers_preprocess(q, k, v),)

        fn = partial(
            xformers_cutlass_fhma().apply,
            needs_gradient=need_gradient,
        )
        return preproc, fn

    @register_benchmark(enabled=HAS_XFORMERS, fwd_only=True)  # noqa
    @multi_input_wrapper
    def xformers_splitk(self, *args):
        need_gradient = not (self.mode == BenchmarkMode.FWD_NO_GRAD)
        xformers_splitk_fhma = xformers_fmha.triton_splitk.FwOp

        def preproc(q, k, v):
            return (self.xformers_preprocess(q, k, v),)

        fn = partial(xformers_splitk_fhma().apply, needs_gradient=need_gradient)
        return preproc, fn

    @register_benchmark(enabled=False, label=f"cudnn-{torch.backends.cudnn.version()}")
    @multi_input_wrapper
    def cudnn(self, *args):
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

        return preproc_noop, sdpa_flash_attention

    if IS_B200:
        # Only enable calling this benchmark directly.
        @register_benchmark(enabled=False)
        @multi_input_wrapper
        def proton_tutorial_flash_v2(self, *args) -> Tuple[Callable, Callable]:
            # includes base (default scheduling) + opt (optimized loop scheduling based on heuristics)
            # Also allows for TMA via WITH_TMA=1
            def fn(q, k, v):
                return proton_tutorial_FA2_opt(
                    q, k, v, self.causal, self.sm_scale, "base_opt"
                )

            return preproc_noop, fn

        # Only enable calling this benchmark directly.
        @register_benchmark(enabled=False)
        @multi_input_wrapper
        def proton_blackwell_tutorial_flash_v2(
            self, *args
        ) -> Tuple[Callable, Callable]:
            # Calls the Triton Tutorial from OAI without modification
            # without using the warp spec path.
            def fn(q, k, v):
                return proton_blackwell_ws_FA2_opt(
                    q, k, v, self.causal, self.sm_scale, False
                )

            return preproc_noop, fn

        # Only enable calling this benchmark directly.
        @register_benchmark(enabled=False)
        @multi_input_wrapper
        def proton_blackwell_tutorial_flash_v2_ws(
            self, *args
        ) -> Tuple[Callable, Callable]:
            # Calls the Triton Tutorial from OAI without modification
            # using the warp spec path.
            def fn(q, k, v):
                return proton_blackwell_ws_FA2_opt(
                    q,
                    k,
                    v,
                    self.causal,
                    self.sm_scale,
                    True,
                )

            return preproc_noop, fn

    if not IS_B200:

        @register_benchmark(enabled=HAS_FLASH_V3)  # noqa
        @multi_input_wrapper
        def flash_v3(self, *args) -> Tuple[Callable, Callable]:
            def preproc(q, k, v):
                # [B, H, S, D] -> [B, S, H, D]
                q = q.transpose(1, 2).contiguous().detach()
                k = k.transpose(1, 2).contiguous().detach()
                v = v.transpose(1, 2).contiguous().detach()
                q.requires_grad_()
                k.requires_grad_()
                v.requires_grad_()
                return q, k, v

            def fn(q, k, v):
                return flash_attn_v3(  # noqa
                    q,
                    k,
                    v,
                    self.sm_scale,
                    self.causal,
                    deterministic=self.deterministic,
                )

            return preproc, fn

        @register_benchmark(enabled=HAS_CUDA_124 and has_warp_spec())
        @multi_input_wrapper
        def triton_tutorial_flash_v2_ws(self, *args) -> Tuple[Callable, Callable]:
            # autotune WarpSpec/CompPipe
            def fn(q, k, v):
                return triton_tutorial_FA2_opt(
                    q, k, v, self.causal, self.sm_scale, "ws"
                )

            return preproc_noop, fn

        @register_benchmark(enabled=HAS_CUDA_124 and has_warp_spec() and has_new_tma())
        @multi_input_wrapper
        def triton_tutorial_flash_v2_tma_ws(self, *args) -> Tuple[Callable, Callable]:
            # autotune TMA/WarpSpec/CompPipe
            def fn(q, k, v):
                return triton_tutorial_FA2_opt(
                    q, k, v, self.causal, self.sm_scale, "tma_ws"
                )

            return preproc_noop, fn

        @register_benchmark(enabled=HAS_CUDA_124 and has_warp_spec() and has_new_tma())
        @multi_input_wrapper
        def triton_tutorial_flash_v2_tma_ws_persistent(
            self, *args
        ) -> Tuple[Callable, Callable]:
            # autotune TMA/WarpSpec/CompPipe/Persistent
            def fn(q, k, v):
                return triton_tutorial_FA2_opt(
                    q, k, v, self.causal, self.sm_scale, "tma_ws_persistent"
                )

            return preproc_noop, fn

        @register_benchmark(enabled=not is_fbcode() and HAS_TK)  # noqa
        @multi_input_wrapper
        def tk(self, *args):
            def _inner(q, k, v):
                out = tk_attn(q, k, v, self.causal)
                return out[0]

            return preproc_noop, _inner

        @register_benchmark(enabled=HAS_PALLAS)  # noqa
        @multi_input_wrapper
        def pallas(self, *args):
            def preproc(q, k, v):
                q = torch_to_jax_tensor(q)
                k = torch_to_jax_tensor(k)
                v = torch_to_jax_tensor(v)
                return q, k, v

            def _inner(q, k, v):
                pallas_mha(q, k, v, segment_ids=None)
                jax.device_put(0.0).block_until_ready()

            return preproc, _inner

        @register_benchmark(enabled=HAS_TILELANG)  # noqa
        @multi_input_wrapper
        def tile(self, *args):
            def preproc(q, k, v):
                # [B, H, S, D] -> [B, S, H, D]
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                return q, k, v

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

            def _inner(q, k, v):
                o = jit_kernel(q, k, v)
                return o

            return preproc, _inner

    @register_benchmark()
    @multi_input_wrapper
    def flex_attention(self, *args):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        flex_attention = torch.compile(flex_attention, dynamic=False)

        if self.causal:
            B, H, S, D = args[0].shape
            block_mask = create_block_mask(
                causal_mask, B=None, H=None, Q_LEN=S, KV_LEN=S
            )
        else:
            block_mask = None

        fn = partial(
            flex_attention,
            block_mask=block_mask,
        )
        return preproc_noop, fn

    def accuracy(self, fn, baseline_fn):
        """Override accuracy to use relaxed tolerance for bfloat16."""
        output_list = fn()
        baseline_output_list = baseline_fn()

        for output, baseline_output in zip(output_list, baseline_output_list):
            # Check for NaN values
            if torch.isnan(output).any():
                return False

            if output.dtype in [torch.bfloat16, torch.float16]:
                default_rtol = 1e-2
                default_atol = 2e-2
            else:
                default_rtol = 1e-5
                default_atol = 1e-8

            rtol = self.tb_args.rtol if self.tb_args.rtol is not None else default_rtol
            atol = self.tb_args.atol if self.tb_args.atol is not None else default_atol

            try:
                torch.testing.assert_close(
                    output,
                    baseline_output,
                    rtol=rtol,
                    atol=atol,
                )
                return True
            except Exception:
                return False

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        assert len(example_inputs) % 3 == 0
        q, k, v = example_inputs[0:3]

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
        outputs = [input_filter(lambda x: isinstance(x, torch.Tensor), o_) for o_ in o]
        dOs = [torch.rand_like(o_).detach() for o_ in outputs]

        def fn():
            for o_tensor, do in zip(outputs, dOs):
                o_tensor.backward(do, retain_graph=True)

        return fn

    def get_input_iter(self) -> Generator:
        if self.input_types == "RAGGED_SHAPES":
            return ragged_inputs(
                self.dtype,
                self.device,
                gen_cache_size_inputs=self.gen_cache_size_inputs,
            )
        elif self.input_types == "ADDITIONAL_SHAPES":
            return additional_inputs(
                shape=(self.BATCH, self.H, self.SEQ_LEN, self.SEQ_LEN_KV, self.D_HEAD),
                num_inputs=self.tb_args.num_inputs,
                dtype=self.dtype,
                device=self.device,
                add_production_shapes=self.add_production_shapes,
                name=self.name,
                shuffle_shapes=self.tb_args.shuffle_shapes,
                gen_cache_size_inputs=self.gen_cache_size_inputs,
            )
        elif self.input_types == "STANDARD_SHAPES":
            return standard_inputs(
                shape=(self.BATCH, self.H, self.SEQ_LEN, self.SEQ_LEN_KV, self.D_HEAD),
                num_inputs=self.tb_args.num_inputs,
                dtype=self.dtype,
                device=self.device,
                gen_cache_size_inputs=self.gen_cache_size_inputs,
            )
        elif self.input_types == "SWEEP_SHAPES":
            return sweep_inputs(
                self.dtype,
                self.device,
                gen_cache_size_inputs=self.gen_cache_size_inputs,
            )
        else:
            raise AssertionError(f"Unknown input type {self.input_types}")

    @register_x_val(label="(Batch, Heads, SeqLen, SeqLen_KV, Dhead)")
    def get_x_val(self, example_inputs) -> float:
        q, k, v = example_inputs[0:3]
        B, H, S, D = q.shape
        _, _, S_KV, _ = k.shape
        return (B, H, S, S_KV, D)

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
        def _plot(N_CTX, N_CTX_KV, provider):
            tflops = self.output.get_y_vals(N_CTX, N_CTX_KV, provider, y_metric_name)
            return tflops

        _plot.run(
            show_plots=True, print_data=False, save_path="/tmp/test_flashattention"
        )

    def get_latency_scale(self, example_inputs):
        assert len(example_inputs) % 3 == 0
        return len(example_inputs) // 3
