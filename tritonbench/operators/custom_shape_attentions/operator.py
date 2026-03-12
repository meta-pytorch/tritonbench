# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
import math
import traceback
from functools import partial
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
from tritonbench.utils.env_utils import IS_BLACKWELL

logger = logging.getLogger(__name__)

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
    print(f"SystemError resulted from importing FA4: {e.__class__.__name__}: {e}")
    traceback.print_exc()

# [Optional] OSS Flash Attention v4
try:
    from flash_attn.cute import (
        flash_attn_func as oss_fa4_flash_attn_func,
        flash_attn_varlen_func as oss_fa4_flash_attn_varlen_func,
    )

    HAS_OSS_FA4 = True
except (ImportError, IOError, AttributeError):
    HAS_OSS_FA4 = False
except SystemError as e:
    HAS_OSS_FA4 = False
    print(f"SystemError resulted from importing OSS FA4: {e.__class__.__name__}: {e}")
    traceback.print_exc()

from tritonbench.utils.input import input_filter
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
    register_x_val,
)

from ..flash_attention.test_fmha_utils import permute_qkv
from .generate_inputs import AttentionShape, customized_inputs


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enable deterministic algorithms by calling torch.use_deterministic_algorithms(True)",
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
    parser.add_argument(
        "--custom-shapes-file",
        type=str,
        default=None,
        help="Absolute path to a Python file containing custom attention shapes. "
        "The file should define a list of shape dictionaries. "
        "Example: /data/users/.../attention_shapes.py",
    )
    parser.add_argument(
        "--custom-shapes-attr",
        type=str,
        default="ATTENTION_SHAPES",
        help="Name of the attribute in the custom shapes file containing the shapes list. "
        "Default: ATTENTION_SHAPES",
    )
    return parser.parse_args(args)


def unpack_inputs(*args):
    return (t.detach() if isinstance(t, torch.Tensor) else t for t in args)


def detach_and_requires_grad(t):
    if not isinstance(t, torch.Tensor):
        return t
    if t.is_floating_point():
        return t.detach().requires_grad_(True)
    return t.detach()


def detach_inputs(*args):
    return [detach_and_requires_grad(t) for t in args]


def multi_input_wrapper(fn):
    def wrapper(self, *args):
        preproc_fn, benchmark_fn = fn(self, *args)
        arg_len = len(args)
        # Determine input group size:
        # - 8 for paged attention (q, k_cache, v_cache, cu_seqlens_q, max_seqlen_q, max_seqlen_k, page_table, seqused_k)
        # - 3 for regular attention (q, k, v)
        if self._is_paged_attention():
            group_size = 8
            assert arg_len % group_size == 0, (
                f"Expected {group_size} inputs per group for paged attention, got {arg_len} total"
            )
        else:
            group_size = 3
            assert arg_len % group_size == 0, (
                f"Expected {group_size} inputs per group for regular attention, got {arg_len} total"
            )
        inputs = []
        all_inputs = []
        for i in range(0, arg_len, group_size):
            group_args = args[i : i + group_size]
            inp = preproc_fn(*group_args)
            inp = detach_inputs(*inp)
            all_inputs += [*unpack_inputs(*inp)]
            inputs.append(inp)

        def multi_input_fn():
            outputs = []
            for i in inputs:
                outputs.append(benchmark_fn(*i))
            return outputs

        # Filter out non-tensor inputs (e.g., max_seqlen_q, max_seqlen_k are integers)
        tensor_inputs = [
            t for t in all_inputs if isinstance(t, torch.Tensor) and t.requires_grad
        ]
        if tensor_inputs:
            self.optims[multi_input_fn] = torch.optim.SGD(tensor_inputs)

        return multi_input_fn

    wrapper.__name__ = fn.__name__
    return wrapper


def preproc_permute(*args):
    # Regular attention: q, k, v (BHSD -> BSHD for flash_attn_func)
    q, k, v = args
    return [t.contiguous() for t in permute_qkv(q, k, v, perm=(0, 2, 1, 3))]


def preproc_paged_attention(
    q, k_cache, v_cache, cu_seqlens_q, max_seqlen_q, max_seqlen_k, page_table, seqused_k
):
    """
    Preprocess paged attention inputs for flash_attn_varlen_func.

    Input shapes:
    - q: [total_q, num_heads, head_dim] - already in correct format
    - k_cache: [total_blocks, block_size, num_heads_kv, head_dim]
    - v_cache: [total_blocks, block_size, num_heads_kv, head_dim]
    - cu_seqlens_q: [batch + 1]
    - max_seqlen_q: int
    - max_seqlen_k: int
    - page_table: [batch, max_num_blocks_per_seq]
    - seqused_k: [batch]

    Returns tuple ready for flash_attn_varlen_func.
    """
    return (
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        seqused_k,
    )


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["latency", "tflops", "tbps"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)

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

        self.deterministic = args.deterministic
        self.gen_cache_size_inputs = args.gen_cache_size_inputs
        self.max_inputs_per_iter = args.max_inputs_per_iter
        self.custom_shapes_file = args.custom_shapes_file
        self.custom_shapes_attr = args.custom_shapes_attr
        self.optims = {}
        # Per-input shape metadata (set during input iteration)
        self.current_shape: Optional[AttentionShape] = None

    def _get_causal(self) -> bool:
        """Get causal setting from current shape."""
        if self.current_shape is None:
            return True
        return self.current_shape.causal

    def _get_window_size(self) -> Tuple[int, int]:
        """Get window size from current shape."""
        if self.current_shape is None:
            return (-1, -1)
        return self.current_shape.window_size

    def _get_local(self) -> bool:
        """Check if sliding window is enabled."""
        return self._get_window_size() != (-1, -1)

    def _get_sm_scale(self) -> float:
        """Get softmax scale from current shape's d_head."""
        if self.current_shape is None:
            return 1.0 / math.sqrt(64)  # default
        return 1.0 / math.sqrt(self.current_shape.d_head)

    def _is_paged_attention(self) -> bool:
        """Check if current shape uses paged attention."""
        if self.current_shape is None:
            return False
        return self.current_shape.paged_attention

    def _get_block_size(self) -> int:
        """Get block size for paged attention."""
        if self.current_shape is None:
            return 16
        return self.current_shape.block_size

    def _get_max_model_seq_len(self) -> int:
        """Get max model sequence length for paged attention."""
        if self.current_shape is None:
            return 32768
        return self.current_shape.max_model_seq_len

    def _get_return_lse(self) -> bool:
        """Get return_lse setting from current shape."""
        if self.current_shape is None:
            return False
        return self.current_shape.return_lse

    @register_benchmark(enabled=(IS_BLACKWELL and HAS_FLASH_CUTE), label="FAv4")
    def cutedsl_blackwell(self, *args) -> Callable:
        if self._is_paged_attention():
            return self._cutedsl_blackwell_paged(*args)
        return self._cutedsl_blackwell(*args)

    @multi_input_wrapper
    def _cutedsl_blackwell(self, *args) -> Tuple[Callable, Callable]:
        causal = self._get_causal()
        local = self._get_local()
        window_size = self._get_window_size()
        fn = partial(
            facute_flash_attn_func,
            softmax_scale=self._get_sm_scale(),
            causal=causal,
            window_size=window_size if local else (None, None),
            deterministic=self.deterministic,
            bottom_right=True,
        )
        return preproc_permute, fn

    @multi_input_wrapper
    def _cutedsl_blackwell_paged(self, *args) -> Tuple[Callable, Callable]:
        causal = self._get_causal()
        window_size = self._get_window_size()
        local = self._get_local()

        def paged_attn_fn(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            page_table,
            seqused_k,
        ):
            return facute_flash_attn_func(
                q,
                k_cache,
                v_cache,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=None,
                seqlen_kv=seqused_k,
                max_seq_len_q=max_seqlen_q,
                max_seq_len_k=max_seqlen_k,
                softmax_scale=self._get_sm_scale(),
                causal=causal,
                window_size=window_size if local else (None, None),
                page_table=page_table,
                deterministic=self.deterministic,
                bottom_right=True,
            )

        return preproc_paged_attention, paged_attn_fn

    @register_benchmark(enabled=(IS_BLACKWELL and HAS_OSS_FA4), label="OSS-FAv4")
    def oss_fa4(self, *args) -> Callable:
        if self._is_paged_attention():
            return self._oss_fa4_paged(*args)
        return self._oss_fa4(*args)

    @multi_input_wrapper
    def _oss_fa4(self, *args) -> Tuple[Callable, Callable]:
        causal = self._get_causal()
        local = self._get_local()
        window_size = self._get_window_size()
        # OSS FA4 may not support bottom_right or return_lse parameters
        fn = partial(
            oss_fa4_flash_attn_func,
            softmax_scale=self._get_sm_scale(),
            causal=causal,
            window_size=window_size if local else (None, None),
            deterministic=self.deterministic,
        )
        return preproc_permute, fn

    @multi_input_wrapper
    def _oss_fa4_paged(self, *args) -> Tuple[Callable, Callable]:
        # Paged attention uses flash_attn_varlen_func with page_table
        causal = self._get_causal()
        window_size = self._get_window_size()
        local = self._get_local()

        def paged_attn_fn(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            page_table,
            seqused_k,
        ):
            return oss_fa4_flash_attn_varlen_func(
                q,
                k_cache,
                v_cache,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=None,  # Not needed with page_table
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                seqused_q=None,
                seqused_k=seqused_k,
                page_table=page_table,
                softmax_scale=self._get_sm_scale(),
                causal=causal,
                window_size=window_size if local else (None, None),
                deterministic=self.deterministic,
            )

        return preproc_paged_attention, paged_attn_fn

    @register_x_val(
        label="(name, B, H, H_KV, S, S_KV, D, causal, window, paged, blk_sz)"
    )
    def get_x_val(
        self, example_inputs
    ) -> Tuple[str, int, int, int, int, Any, int, bool, str, str, int]:
        """Return all shape parameters as a tuple for separate columns."""
        if self.current_shape is None:
            return ("unknown", 0, 0, 0, 0, 0, 0, True, "(-1,-1)", "False", 0)

        ws = self.current_shape.window_size
        # Format paged attention status with shuffle info
        if self.current_shape.paged_attention:
            if self.current_shape.page_shuffle:
                paged_str = "True(shuffled)"
            else:
                paged_str = "True(contiguous)"
        else:
            paged_str = "False"

        # Format seq_len_kv: show sum for variable lengths
        seq_len_kv = self.current_shape.seq_len_kv
        if isinstance(seq_len_kv, list):
            seq_len_kv_display = sum(seq_len_kv)
        else:
            seq_len_kv_display = seq_len_kv

        return (
            self.current_shape.name,
            self.current_shape.batch,
            self.current_shape.n_heads,
            self.current_shape.n_heads_kv,
            self.current_shape.seq_len,
            seq_len_kv_display,
            self.current_shape.d_head,
            self.current_shape.causal,
            f"({ws[0]},{ws[1]})",
            paged_str,
            self.current_shape.block_size if self.current_shape.paged_attention else 0,
        )

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        if self._is_paged_attention():
            # Paged attention: q, k_cache, v_cache, cu_seqlens_q, max_seqlen_q, max_seqlen_k, page_table, seqused_k
            assert len(example_inputs) % 8 == 0
            q = example_inputs[0]
            # q shape: [total_q, num_heads, head_dim]
            total_q, H, D_HEAD = q.shape
            BATCH = self.current_shape.batch if self.current_shape else 1
            N_CTX = total_q // BATCH
            N_CTX_KV = (
                self.current_shape.seq_len_kv
                if self.current_shape
                else example_inputs[5]
            )  # max_seqlen_k
        else:
            assert len(example_inputs) % 3 == 0
            q, k, v = example_inputs[0:3]
            BATCH, H, N_CTX, D_HEAD = q.shape
            _, _, N_CTX_KV, _ = k.shape

        local = self._get_local()
        causal = self._get_causal()
        window_size = self._get_window_size()

        # Normalize N_CTX_KV to a list for variable-length support
        if isinstance(N_CTX_KV, list):
            kv_lens = N_CTX_KV
        else:
            kv_lens = [N_CTX_KV] * BATCH

        if not local:
            flops = sum(2.0 * 2 * H * N_CTX * kv_len * D_HEAD for kv_len in kv_lens)
            if causal:
                flops *= 0.5
        else:
            flops = 0
            for kv_len in kv_lens:
                row_idx = torch.arange(N_CTX, device="cuda")
                col_left = torch.maximum(
                    row_idx + kv_len - N_CTX - window_size[0], torch.tensor(0)
                )
                col_right = torch.minimum(
                    row_idx + kv_len - N_CTX + window_size[1],
                    torch.tensor(kv_len - 1),
                )
                avg_seqlen = (col_right - col_left + 1).float().mean().item()
                flops += 2 * 2.0 * H * N_CTX * avg_seqlen * D_HEAD

        if self.mode == BenchmarkMode.BWD:
            flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        elif self.mode == BenchmarkMode.FWD_BWD:
            flops *= 3.5  # 1.0(fwd) + 2.0(bwd) + 0.5(recompute)
        return flops

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        if self.use_cuda_graphs:
            stream = self.get_cudagraph_stream()
            stream.wait_stream(torch.cuda.current_stream())
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            o = fwd_fn()
            outputs = [
                input_filter(lambda x: isinstance(x, torch.Tensor), o_) for o_ in o
            ]
            dOs = [torch.rand_like(o_).detach() for o_ in outputs]
            zero_grad = (
                self.optims[fwd_fn].zero_grad
                if fwd_fn in self.optims
                else lambda set_to_none: None
            )

        if self.use_cuda_graphs:
            torch.cuda.current_stream().wait_stream(stream)

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
            "custom_shapes_file": self.custom_shapes_file,
            "custom_shapes_attr": self.custom_shapes_attr,
        }
        for tensors, shape in customized_inputs(**common_kwargs):
            self.current_shape = shape
            yield tensors

    def get_num_inputs_per_iter(self, example_inputs) -> int:
        if self._is_paged_attention():
            assert len(example_inputs) % 8 == 0
            return len(example_inputs) // 8
        else:
            assert len(example_inputs) % 3 == 0
            return len(example_inputs) // 3

    def get_latency_scale(self, example_inputs):
        return self.get_num_inputs_per_iter(example_inputs)
