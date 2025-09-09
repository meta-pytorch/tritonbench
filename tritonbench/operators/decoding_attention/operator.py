# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os

from typing import Callable, Optional, Tuple

import torch
from gen_ai.llm_inference.fb.llm.quantization.kv_quantize import quantize_kv_fp8

from tritonbench.utils.path_utils import add_ld_library_path

# [Optional] flash_attn v2
HAS_FLASH_V2 = True
try:
    from flash_attn import flash_attn_interface as flash_attn_v2
except (ImportError, IOError, AttributeError):
    HAS_FLASH_V2 = False

HAS_CUDA_124 = (
    torch.cuda.is_available() and torch.version.cuda and torch.version.cuda >= "12.4"
)

# [Optional] flash_attn v3
HAS_FLASH_V3 = True
try:
    torch_lib_path = os.path.join(os.path.dirname(__file__), "lib")
    with add_ld_library_path(torch_lib_path):
        import flash_attn_interface as flash_attn_v3
except (ImportError, IOError, AttributeError):
    try:
        from ai_codesign.gen_ai.flash_attention_v2.hopper import (
            flash_attn_interface as flash_attn_v3,
        )
    except (ImportError, IOError, AttributeError):
        HAS_FLASH_V3 = False

# [Optional] xformers backend
# try:
# import xformers  # @manual=//fair/xformers:xformers

from typing import Generator, List

# [Optional] xformers backend
HAS_XFORMERS = True
try:
    import xformers.ops.fmha as fmha  # @manual=//fair/xformers:xformers
except (ImportError, IOError, AttributeError):
    HAS_XFORMERS = False


torch.ops.load_library(
    "//deeplearning/fbgemm/fbgemm_gpu/experimental:gen_ai_attention_ops"
)

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)

# [AMD only] aiter backend
HAS_AITER = True
try:
    import aiter_
except (ImportError, IOError, AttributeError):
    HAS_AITER = False


# [Optional] flash_fwd cute-DSL backend
HAS_FLASH_CUTE = True
try:
    from flash_attn.cute.interface import flash_attn_func as flash_attn_cute_func
except (ImportError, IOError, AttributeError):
    HAS_FLASH_CUTE = False
    flash_attn_cute_func = None  # Define it as None to avoid NameError


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--seq-len-q", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len-kv", type=int, default=4096, help="Batch size")
    parser.add_argument("--max-len-kv", type=int, default=8192, help="Batch size")
    parser.add_argument("--head-q", type=int, default=8, help="Number of Q heads")
    parser.add_argument("--head-kv", type=int, default=1, help="Number of KV heads")
    parser.add_argument(
        "--head-d", type=int, default=128, help="specify head dimension"
    )
    parser.add_argument("--page-size", type=int, default=256, help="Page Size")
    return parser.parse_args(args)


from dataclasses import astuple, dataclass


"""
- Runbook
- Nvidia:
buck2 run @mode/opt @mode/inplace -c fbcode.enable_gpu_sections=true -c fbcode.nvcc_arch=h100a -c fbcode.platform010_cuda_version=12.4 //pytorch/tritonbench:run -- --op decoding_attention --cudagraph --csv
- AMD:
buck2 run @mode/opt-amd-gpu @mode/inplace -c fbcode.enable_gpu_sections=true -c fbcode.rocm_arch=mi300 //pytorch/tritonbench:run -- --op decoding_attention --cudagraph --csv
"""


@dataclass
class _Shape:
    batch: int
    seq_len_q: int
    seq_len_kv: int
    max_len_kv: int
    head_q: int
    head_kv: int
    head_d: int

    def unpack(self):
        return astuple(self)


def _generate_shapes():
    # llama4 128e: head_q = 5
    HEAD_Q = 5
    HEAD_KV = 1
    HEAD_D = 128
    max_len_kv = 32768

    SEQ_LEN_KVs = [1024, 2048, 4096, 8190, 32760]
    SEQ_LEN_Qs = [1, 4]
    BATCHs = [16, 32, 64, 128]

    return [
        _Shape(
            batch=batch,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            max_len_kv=max_len_kv,
            head_q=HEAD_Q,
            head_kv=HEAD_KV,
            head_d=HEAD_D,
        )
        for seq_len_q in SEQ_LEN_Qs
        for batch in BATCHs
        for seq_len_kv in SEQ_LEN_KVs
    ]


CAUSAL = True


def _pack_xformer_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
]:
    batch, seq_len_q, head_q, head_d = q.shape
    _, max_len_kv, head_kv, _ = k.shape

    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[seq_len_q] * batch,
        kv_seqlen=cache_seqlens.tolist(),
        kv_padding=max_len_kv,
    )

    q = q.view(1, -1, head_q, head_d)
    k = k.expand(-1, -1, head_q, -1).view(1, -1, head_q, k.shape[-1])
    v = v.expand(-1, -1, head_q, -1).view(1, -1, head_q, v.shape[-1])
    return q, k, v, attn_bias


def get_dtype_max(dtype):
    try:
        dtypeMax = torch.finfo(dtype).max
    except:
        dtypeMax = torch.iinfo(dtype).max
    return dtypeMax


def pertoken_quant(x, y_scale_dtype=torch.float, x_scale=None, quant_dtype=torch.int8):
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x.to(x_scale) * x_scale
    # [m, 1]
    per_token_amax, _ = torch.max(input=torch.abs(hidden_states), dim=-1, keepdim=True)

    dtypeMax = get_dtype_max(quant_dtype)

    per_token_scale = per_token_amax.to(dtype=torch.float32) / dtypeMax
    per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(y_scale_dtype)
    return y, y_scale


def pertoken_quant_kvcache_symm(
    # [num_blocks, num_heads, head_size // x, block_size, x]
    k_cache: torch.Tensor,
    # [num_blocks, num_heads, head_size, block_size]
    v_cache: torch.Tensor,
    quant_dtype: torch.dtype,  # e.g. torch.float8_e4m3fnuz
    scale_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = k_cache.shape[0]
    num_heads = k_cache.shape[1]
    head_dim = v_cache.shape[2]
    block_size = v_cache.shape[3]
    total_tokens = num_blocks * block_size

    k_cache_permute = (
        k_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    v_cache_permute = (
        v_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )

    k_quant, k_scale_asm = pertoken_quant(
        k_cache_permute, scale_dtype, quant_dtype=quant_dtype
    )
    v_quant, v_scale_asm = pertoken_quant(
        v_cache_permute, scale_dtype, quant_dtype=quant_dtype
    )

    # NOTE: quant_x and original x could be different
    quant_x = 16 // quant_dtype.itemsize

    k_quant = (
        k_quant.view(num_blocks, num_heads, block_size, head_dim // quant_x, quant_x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    k_scale = k_scale_asm.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    v_quant = (
        v_quant.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    v_scale = v_scale_asm.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)

    return k_quant, k_scale, v_quant, v_scale, k_scale_asm, v_scale_asm


def asm_V_shuffle(VC):
    # [num_blocks, num_kv_heads, head_size, block_size]
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    # [num_blocks, num_kv_heads, block_size/X, head_size, X]
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    return VC


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"

    DEFAULT_METRICS = ["latency", "speedup"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        if args.batch:
            self.shapes = [
                _Shape(
                    args.batch,
                    args.seq_len_q,
                    args.seq_len_kv,
                    args.max_len_kv,
                    args.head_q,
                    args.head_kv,
                    args.head_d,
                )
            ]
        else:
            self.shapes = _generate_shapes()

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            batch, seq_len_q, seq_len_kv, max_len_kv, head_q, head_kv, head_d = (
                shape.unpack()
            )
            q = torch.randn(
                (batch, seq_len_q, head_q, head_d),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
            k_cache = torch.randn(
                (batch, max_len_kv, head_kv, head_d),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
            v_cache = torch.randn(
                (batch, max_len_kv, head_kv, head_d),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
            cache_seqlens = torch.tensor(
                [seq_len_kv] * batch, dtype=torch.int32, device=self.device
            )

            yield (q, k_cache, v_cache, cache_seqlens)

    @register_x_val(label="(Batch, SeqLenQ, SeqLenKV, MaxLenKV, HeadQ, HeadKV, HeadD)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int, int, int, int, int]:
        q, k_cache, v_cache, cache_seqlens = example_inputs
        batch, seq_len_q, head_q, head_d = q.shape
        _, max_len_kv, head_kv, _ = k_cache.shape
        seq_len_kv = cache_seqlens.max().item()
        return (batch, seq_len_q, seq_len_kv, max_len_kv, head_q, head_kv, head_d)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        accuracy = True

        try:
            torch.testing.assert_close(output, baseline_output, atol=1e-3, rtol=0.5)
        except Exception:
            accuracy = False
        finally:
            return accuracy

    @register_metric()
    def gbps(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics) -> float:
        # fp16/bf16 by default
        Q_ELEMENT_SIZE = 2
        K_ELEMENT_SIZE = 2
        if "fp8qkv" in str(fn):
            Q_ELEMENT_SIZE = 1
            K_ELEMENT_SIZE = 1
        elif "fp8kv" in str(fn):
            K_ELEMENT_SIZE = 1

        def nbytes(t):
            return t.numel() * Q_ELEMENT_SIZE

        q, k_cache, v_cache, cache_seqlens = example_inputs

        q_bytes = nbytes(q)
        # k_cache: B, max_len_kv, head_kv, head_d
        k_cache_bytes = (
            k_cache.shape[-1]
            * k_cache.shape[-2]
            * int(cache_seqlens.sum().item())
            * K_ELEMENT_SIZE
        )
        v_cache_bytes = k_cache_bytes
        gb = (q_bytes + k_cache_bytes + v_cache_bytes) * 1e-9
        return gb / metrics.latency * 1e3

    @register_metric()
    def flops(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics) -> float:
        q, k_cache, v_cache, cache_seqlens = example_inputs

        batch, seq_len_q, head_q, head_d = q.shape
        total_kv_length = int(cache_seqlens.sum().item())

        qk_gemm = 2 * total_kv_length * seq_len_q * head_q * head_d

        attention_flops = qk_gemm * 2

        return attention_flops

    @register_benchmark(enabled=HAS_FLASH_V2)
    def fa2_kvcache(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        return lambda: flash_attn_v2.flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            causal=CAUSAL,
        )

    @register_benchmark(enabled=HAS_FLASH_V3)
    def fa3_kvcache_gqa_heuristic(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        return lambda: flash_attn_v3.flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            causal=CAUSAL,
            pack_gqa=True,
            num_splits=0,
        )

    @register_benchmark(enabled=HAS_XFORMERS and HAS_FLASH_V3)
    def fa3_mha_varlen_fwd(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        _q, _k, _v, attn_bias = _pack_xformer_input(q, k_cache, v_cache, cache_seqlens)
        return lambda: fmha.memory_efficient_attention_forward(
            _q,
            _k,
            _v,
            attn_bias,
            op=fmha.flash3.FwOp,
        ).view(q.shape)

    @register_benchmark(enabled=HAS_FLASH_V2 and HAS_XFORMERS)
    def fa2_mha_varlen_fwd(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        _q, _k, _v, attn_bias = _pack_xformer_input(q, k_cache, v_cache, cache_seqlens)
        return lambda: fmha.memory_efficient_attention_forward(
            _q,
            _k,
            _v,
            attn_bias,
            op=fmha.flash.FwOp,
        ).view(q.shape)

    @register_benchmark(baseline=True, enabled=HAS_XFORMERS)
    def triton_splitk(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        _q, _k, _v, attn_bias = _pack_xformer_input(q, k_cache, v_cache, cache_seqlens)

        return lambda: fmha.memory_efficient_attention_forward(
            _q,
            _k,
            _v,
            attn_bias,
            op=fmha.triton_splitk.FwOp,
        ).view(q.shape)

    @register_benchmark(enabled=HAS_XFORMERS)
    def triton_splitk_fp8kv(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        _, _, num_q_heads, _ = q.shape
        batch_size, max_sequence_length, _, _ = k_cache.shape
        _q, _k, _v, attn_bias = _pack_xformer_input(q, k_cache, v_cache, cache_seqlens)

        _k = _k.to(torch.uint8).view(torch.int32)
        _v = _v.to(torch.uint8).view(torch.int32)

        k_fp8_scales_shifts = torch.zeros(
            batch_size * max_sequence_length,
            dtype=torch.int32,
            device="cuda",
        )
        v_fp8_scales_shifts = torch.zeros(
            batch_size * max_sequence_length,
            dtype=torch.int32,
            device="cuda",
        )

        def _to_expanded_shape(x: torch.Tensor) -> torch.Tensor:
            return x.view(1, batch_size * max_sequence_length, 1, -1).expand(
                1, batch_size * max_sequence_length, num_q_heads, -1
            )

        packed_k_fp8_scales_shifts = _to_expanded_shape(k_fp8_scales_shifts).squeeze(-1)
        packed_v_fp8_scales_shifts = _to_expanded_shape(v_fp8_scales_shifts).squeeze(-1)

        inp = fmha.triton_splitk.InputsFp8(
            query=_q,
            key=_k,
            value=_v,
            attn_bias=attn_bias,
            k_fp8_scale_shift=packed_k_fp8_scales_shifts,
            v_fp8_scale_shift=packed_v_fp8_scales_shifts,
        )

        return lambda: fmha._memory_efficient_attention_forward(
            inp,
            op=fmha.triton_splitk.FwOp,
        ).view(q.shape)

    @register_benchmark(enabled=HAS_FLASH_V3)
    def fa3_kvcache_fp8qkv(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        batch, _, head_kv, _ = k_cache.shape
        _q = q.to(torch.float8_e4m3fn)
        _k_cache = k_cache.to(torch.float8_e4m3fn)
        _v_cache = v_cache.to(torch.float8_e4m3fn)

        _dummy_descale = torch.ones([batch, head_kv], device=q.device)

        return lambda: flash_attn_v3.flash_attn_with_kvcache(
            q=_q,
            k_cache=_k_cache,
            v_cache=_v_cache,
            cache_seqlens=cache_seqlens,
            causal=CAUSAL,
            q_descale=torch.ones_like(_dummy_descale),
            k_descale=torch.ones_like(_dummy_descale),
            v_descale=torch.ones_like(_dummy_descale),
            pack_gqa=True,
            num_splits=0,
        )

    @register_benchmark(enabled=False)
    def fbgemm_gqa_fp8kv(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        kv_cache_quant_num_groups = 1

        k_fp8 = quantize_kv_fp8(k_cache, kv_cache_quant_num_groups).view(torch.uint8)
        v_fp8 = quantize_kv_fp8(v_cache, kv_cache_quant_num_groups).view(torch.uint8)
        return lambda: torch.ops.fbgemm.gqa_attn_splitk(
            q,
            k_fp8,
            v_fp8,
            cache_seqlens,
            1.0,  # qk_scale
            num_split_ks=16,
            kv_cache_quant_num_groups=kv_cache_quant_num_groups,
            use_tensor_cores=True,
            cache_logical_dtype_int=1,  # FP8 = 1
        )

    @register_benchmark(enabled=HAS_FLASH_CUTE)
    def flash_cute_dsl(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        """Flash Attention implementation using cute-DSL backend."""
        # For GQA, cute-DSL handles the head expansion internally
        # We pass the original KV tensors without manual expansion
        q_heads = q.shape[2]
        kv_heads = k_cache.shape[2]
        return lambda: flash_attn_cute_func(
            q, k_cache, v_cache, causal=CAUSAL, pack_gqa=(q_heads != kv_heads)
        )

    @register_benchmark(enabled=HAS_AITER)
    def aiter_paged_fp8kv(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        ori_dtype = q.dtype
        dtype = torch.float8_e4m3fnuz

        num_seqs = k_cache.shape[0]
        max_seq_len = k_cache.shape[1]
        block_size = 16
        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        num_blocks = max_num_blocks_per_seq * num_seqs
        head_size = k_cache.shape[3]
        num_heads = k_cache.shape[2]

        x = 16 // ori_dtype.itemsize
        k_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
        v_cache_shape = (num_blocks, num_heads, head_size, block_size)
        _k_cache = torch.rand(k_cache_shape, dtype=ori_dtype, device=self.device)
        _v_cache = torch.rand(v_cache_shape, dtype=ori_dtype, device=self.device)

        k_quant, k_scale, v_quant, v_scale, k_scale_asm, v_scale_asm = (
            pertoken_quant_kvcache_symm(
                _k_cache, _v_cache, dtype, scale_dtype=torch.float32
            )
        )

        # total_tokens = num_blocks * block_size
        # k_scale = torch.ones(
        #     (num_heads, total_tokens), dtype=torch.float32, device=self.device
        # )
        # v_scale = torch.ones_like(k_scale)

        available_blocks = list(range(num_blocks))  # Blocks 0 to num_blocks-1
        # available_blocks = [0] * num_blocks
        block_tables_list = []
        for _ in range(num_seqs):
            block_tables = available_blocks[:max_num_blocks_per_seq]
            available_blocks = available_blocks[max_num_blocks_per_seq:]
            block_tables_list.append(block_tables)

        block_tables = torch.tensor(
            block_tables_list, dtype=torch.int, device=self.device
        )

        num_query_heads = q.shape[2]
        num_kv_heads = num_heads
        uniform_range = (-1, 1)
        query = torch.empty_strided(
            (num_seqs, num_query_heads, head_size),
            ((num_query_heads + 2 * num_kv_heads) * head_size, head_size, 1),
            dtype=ori_dtype,
            device=self.device,
        )
        query.uniform_(*uniform_range)

        return lambda: aiter_.pa_fwd_asm(
            query.contiguous(),
            k_quant,
            asm_V_shuffle(v_quant),
            block_tables,
            cache_seqlens,
            max_num_blocks_per_seq,
            k_scale_asm,
            v_scale_asm,
        )
