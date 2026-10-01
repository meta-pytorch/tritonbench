# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 variant of the blackwell_attentions operator.

This is a thin operator that reuses everything from `blackwell_attentions`
(input generation, CLI args, the `flops` metric, the bf16 baselines such as
`cudnn_sdpa`/`cutedsl_blackwell`/`tlx_blackwell_ws_pipelined_persistent`) and
adds one prequantized MXFP8 backend for kernel-only Blackwell flash-attention
forward and backward benchmarks.

Benchmark registration keys on the defining module path, so subclassing the
parent operator does not by itself expose the parent's backends/metrics under
this op's name. We therefore clone the parent registry buckets into this op's
name at import time (see the bottom of this file).
"""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import triton
from tritonbench.operators.blackwell_attentions.operator import (
    multi_input_wrapper,
    Operator as BlackwellAttentionsOperator,
)
from tritonbench.utils.triton_op import (
    BASELINE_BENCHMARKS,
    Mode as BenchmarkMode,
    OVERRIDDEN_METRICS,
    register_benchmark,
    REGISTERED_BENCHMARKS,
    REGISTERED_METRICS,
    REGISTERED_X_VALS,
)

HAS_TLX_MXFP8 = False
try:
    from torchao.prototype.mx_formats.mx_tensor import (
        MXTensor as _MXTensor,
        ScaleCalculationMode as _ScaleCalculationMode,
    )

    # @manual=//triton:triton
    from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent_mxfp8 import (
        _attn_fwd_mxf8_ws as _tlx_mxfp8_attn_fwd,
        _mxf8_host_descriptor_pre_hook as _tlx_mxfp8_fwd_pre_hook,
        swizzled_to_tma_preshuffled as _tlx_swizzled_to_tma_preshuffled,
    )
    from triton.tlx.ops.kernels.flash_attn_mxfp8 import sm100 as _tlx_ops_mxfp8_sm100
    from triton.tools.tensor_descriptor import TensorDescriptor as _TensorDescriptor

    HAS_TLX_MXFP8 = True
except (ImportError, IOError, AttributeError, TypeError):
    HAS_TLX_MXFP8 = False

_HAS_TLX_MXFP8_2CTA = HAS_TLX_MXFP8 and all(
    hasattr(_tlx_ops_mxfp8_sm100, name)
    for name in (
        "_MXFP8_BWD_2CTA_HEAD_DIM",
        "_MXFP8_BWD_2CTA_N_CTXS",
        "_MXFP8_BWD_2CTA_PIPELINE_READY",
    )
)


# Forward config for the MXFP8 Blackwell FA kernel, matching the config the
# reference correctness/perf harness uses to drive _attn_fwd_mxf8_ws.fn directly
# (third_party/tlx/tutorials/testing/test_correctness.py:FlashAttention.CONFIGS).
_MXFP8_FWD_CONFIG = {
    "BLOCK_M": 256,
    "BLOCK_N": 128,
    "NUM_BUFFERS_Q": 1,
    "NUM_BUFFERS_KV": 3,
    "NUM_BUFFERS_QK": 1,
    "NUM_MMA_GROUPS": 2,
    "NUM_Q_SCALE_TMEM_BUFFERS": 1,
    "NUM_KV_SCALE_TMEM_BUFFERS": 2,
    "GROUP_SIZE_N": 1,
    "RESCALE_OPT": True,
}


def _mxfp8_quantize_operand(ref, dtype, transpose_for_reduction=False):
    # Quantize a [Z, H, N_CTX, HEAD_DIM] bf16 tensor to MXFP8 (E4M3 data + E8M0
    # block scales in TMA-preshuffled 5D layout). transpose_for_reduction blocks
    # the scales along N_CTX instead of HEAD_DIM (needed for V in the forward and
    # for the reduction-axis-swapped operands in the backward).
    Z, H, N_CTX, HEAD_DIM = ref.shape
    flat = ref.reshape(Z * H * N_CTX, HEAD_DIM).contiguous()
    quant_input = flat.t().contiguous() if transpose_for_reduction else flat
    mx = _MXTensor.to_mx(
        quant_input,
        dtype,
        scaling_mode=_ScaleCalculationMode.RCEIL,
        is_swizzled_scales=True,
    )
    if transpose_for_reduction:
        data = mx.qdata.t().reshape_as(ref).contiguous()
        scale = _tlx_swizzled_to_tma_preshuffled(mx.scale, HEAD_DIM, N_CTX, 32, Z * H)
    else:
        data = mx.qdata.reshape_as(ref).contiguous()
        scale = _tlx_swizzled_to_tma_preshuffled(mx.scale, N_CTX, HEAD_DIM, 32, Z * H)
    return data, scale


def _mxfp8_forward_with_lse(q, k, v, q_scale, k_scale, v_scale, sm_scale, causal):
    # Launch the forward kernel directly (rather than the public attention()
    # wrapper) so we can capture the logsumexp M, which the backward needs.
    Z, H, N_CTX, HEAD_DIM = q.shape
    y_dim = Z * H * N_CTX
    o = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    dummy_block = [1, 1]
    dummy_5d = [1, 1, 1, 1, 1]

    desc_q = _TensorDescriptor(
        q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block
    )
    desc_k = _TensorDescriptor(
        k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block
    )
    desc_v = _TensorDescriptor(
        v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block
    )
    desc_o = _TensorDescriptor(
        o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block
    )
    desc_m = _TensorDescriptor(M, shape=[y_dim], strides=[1], block_shape=[1])
    desc_q_scale = _TensorDescriptor.from_tensor(q_scale, block_shape=dummy_5d)
    desc_k_scale = _TensorDescriptor.from_tensor(k_scale, block_shape=dummy_5d)
    desc_v_scale = _TensorDescriptor.from_tensor(v_scale, block_shape=dummy_5d)

    nargs = {
        **_MXFP8_FWD_CONFIG,
        "HEAD_DIM": HEAD_DIM,
        "desc_q": desc_q,
        "desc_k": desc_k,
        "desc_v": desc_v,
        "desc_o": desc_o,
        "desc_m": desc_m,
        "desc_q_scale": desc_q_scale,
        "desc_k_scale": desc_k_scale,
        "desc_v_scale": desc_v_scale,
    }
    _tlx_mxfp8_fwd_pre_hook(nargs)

    def alloc_fn(size, align, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    grid = (
        triton.cdiv(N_CTX, _MXFP8_FWD_CONFIG["BLOCK_M"]) * Z * H,
        1,
        1,
    )
    _tlx_mxfp8_attn_fwd.fn[grid](
        sm_scale,
        desc_m,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        desc_q_scale,
        desc_k_scale,
        desc_v_scale,
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM,
        STAGE=3 if causal else 1,
        num_stages=1,
        num_warps=4,
        **_MXFP8_FWD_CONFIG,
    )
    return o, M


@dataclass
class _MXFP8PreparedState:
    q_fp8: torch.Tensor
    k_fp8: torch.Tensor
    v_fwd: torch.Tensor
    q_scale: torch.Tensor
    q_scale_dk: torch.Tensor
    k_scale: torch.Tensor
    k_scale_dq: torch.Tensor
    v_fwd_scale: torch.Tensor
    v_bwd: torch.Tensor
    v_bwd_scale: torch.Tensor


class _TLXBlackwellMXFP8Attention(torch.autograd.Function):
    """Connect prequantized MXFP8 kernels to TritonBench's autograd harness."""

    @staticmethod
    def forward(ctx, q, k, v, prepared, sm_scale, causal):
        del k, v
        o, M = _mxfp8_forward_with_lse(
            prepared.q_fp8,
            prepared.k_fp8,
            prepared.v_fwd,
            prepared.q_scale,
            prepared.k_scale,
            prepared.v_fwd_scale,
            sm_scale,
            causal,
        )
        ctx.save_for_backward(o, M)
        ctx.prepared = prepared
        ctx.input_dtype = q.dtype
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.do_cache = {}
        return o

    @staticmethod
    def backward(ctx, do):
        key = do.data_ptr()
        if key not in ctx.do_cache:
            do_bf16 = do.to(torch.bfloat16).contiguous()
            ctx.do_cache = {
                key: (
                    do_bf16,
                    *_tlx_ops_mxfp8_sm100._quantize_mxfp8_32x32_operand(do_bf16),
                )
            }
        do_bf16, do_fp8, do_scale, do_scale_dv = ctx.do_cache[key]
        o, M = ctx.saved_tensors
        prepared = ctx.prepared

        dq, dk, dv = _tlx_ops_mxfp8_sm100.attention_bwd(
            do_fp8,
            prepared.q_fp8,
            prepared.k_fp8,
            prepared.v_bwd,
            o,
            M,
            prepared.q_scale,
            prepared.q_scale_dk,
            prepared.k_scale,
            prepared.k_scale_dq,
            prepared.v_bwd_scale,
            do_scale,
            do_scale_dv,
            ctx.sm_scale,
            do_bf16=do_bf16,
            causal=ctx.causal,
        )
        return (
            dq.to(ctx.input_dtype),
            dk.to(ctx.input_dtype),
            dv.to(ctx.input_dtype),
            None,
            None,
            None,
        )


class Operator(BlackwellAttentionsOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[list[str]] = None
    ) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--mxfp8-bwd-num-ctas",
            type=int,
            choices=(1, 2),
            default=1,
            help="Select the CTA count for the MXFP8 backward kernel.",
        )
        mxfp8_args, parent_args = parser.parse_known_args(extra_args or [])
        super().__init__(tb_args, parent_args)
        self.mxfp8_bwd_num_ctas = mxfp8_args.mxfp8_bwd_num_ctas

        if self.mxfp8_bwd_num_ctas == 2:
            if not _HAS_TLX_MXFP8_2CTA:
                raise ValueError(
                    "the installed TLX MXFP8 backward kernel does not support two CTAs"
                )
            if self.mode not in (BenchmarkMode.BWD, BenchmarkMode.FWD_BWD):
                raise ValueError("two-CTA MXFP8 routing is supported only for backward")
            if self.causal:
                raise ValueError(
                    "two-CTA MXFP8 backward does not support causal attention"
                )
            if self.D_HEAD != _tlx_ops_mxfp8_sm100._MXFP8_BWD_2CTA_HEAD_DIM:
                raise ValueError(
                    "two-CTA MXFP8 backward requires head dimension "
                    f"{_tlx_ops_mxfp8_sm100._MXFP8_BWD_2CTA_HEAD_DIM}"
                )

        if _HAS_TLX_MXFP8_2CTA:
            _tlx_ops_mxfp8_sm100._MXFP8_BWD_2CTA_PIPELINE_READY = (
                self.mxfp8_bwd_num_ctas == 2
            )

    @multi_input_wrapper
    def _tlx_blackwell_mxfp8_with_backward(self, *args):
        if self.varlen or self.local:
            raise NotImplementedError(
                "MXFP8 attention supports only dense, non-local inputs"
            )

        def preproc(q, k, v):
            if q.shape != k.shape or q.shape != v.shape:
                raise ValueError("MXFP8 backward requires self-attention shapes")
            if q.dtype != torch.bfloat16:
                raise ValueError("MXFP8 backward requires BF16 inputs")
            if q.shape[3] != 128 or q.shape[2] % 256 != 0:
                raise ValueError(
                    "MXFP8 backward requires head dimension 128 and a sequence "
                    "length divisible by 256"
                )
            if (
                self.mxfp8_bwd_num_ctas == 2
                and q.shape[2] not in _tlx_ops_mxfp8_sm100._MXFP8_BWD_2CTA_N_CTXS
            ):
                supported_n_ctxs = _tlx_ops_mxfp8_sm100._MXFP8_BWD_2CTA_N_CTXS
                raise ValueError(
                    "unsupported two-CTA MXFP8 backward sequence length "
                    f"{q.shape[2]}; expected one of {supported_n_ctxs}"
                )

            q_fp8, q_scale, q_scale_dk = (
                _tlx_ops_mxfp8_sm100._quantize_mxfp8_32x32_operand(q)
            )
            k_fp8, k_scale, k_scale_dq = (
                _tlx_ops_mxfp8_sm100._quantize_mxfp8_32x32_operand(k)
            )
            v_fwd, v_fwd_scale = _mxfp8_quantize_operand(
                v, torch.float8_e4m3fn, transpose_for_reduction=True
            )
            if self.mxfp8_bwd_num_ctas == 2:
                v_bwd, v_bwd_scale, _ = (
                    _tlx_ops_mxfp8_sm100._quantize_mxfp8_32x32_operand(v)
                )
            else:
                v_bwd, v_bwd_scale = _mxfp8_quantize_operand(v, torch.float8_e4m3fn)

            return (
                q,
                k,
                v,
                _MXFP8PreparedState(
                    q_fp8=q_fp8,
                    k_fp8=k_fp8,
                    v_fwd=v_fwd,
                    q_scale=q_scale,
                    q_scale_dk=q_scale_dk,
                    k_scale=k_scale,
                    k_scale_dq=k_scale_dq,
                    v_fwd_scale=v_fwd_scale,
                    v_bwd=v_bwd,
                    v_bwd_scale=v_bwd_scale,
                ),
            )

        def fn(q, k, v, prepared):
            return _TLXBlackwellMXFP8Attention.apply(
                q, k, v, prepared, self.sm_scale, self.causal
            )

        return preproc, fn

    # Only works with triton beta. Quantization happens during benchmark setup;
    # timed iterations measure only the MXFP8 Blackwell FA kernels.
    @register_benchmark(enabled=HAS_TLX_MXFP8, label="tlx-mxfp8")
    def tlx_blackwell_mxfp8(self, *args) -> Callable:
        if self.mode in (BenchmarkMode.BWD, BenchmarkMode.FWD_BWD):
            return self._tlx_blackwell_mxfp8_with_backward(*args)

        self.optims.clear()
        assert len(args) % 3 == 0
        dtype = torch.float8_e4m3fn
        quantized_inputs = []
        for i in range(0, len(args), 3):
            q, k, v = args[i : i + 3]
            q_fp8, q_scale = _mxfp8_quantize_operand(q, dtype)
            k_fp8, k_scale = _mxfp8_quantize_operand(k, dtype)
            v_fp8, v_scale = _mxfp8_quantize_operand(
                v, dtype, transpose_for_reduction=True
            )
            quantized_inputs.append((q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale))

        def fn():
            outputs = []
            for q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale in quantized_inputs:
                output, _ = _mxfp8_forward_with_lse(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    q_scale,
                    k_scale,
                    v_scale,
                    self.sm_scale,
                    self.causal,
                )
                outputs.append(output)
            return outputs

        return fn


# Clone the parent operator's registry buckets into this op's name so the bf16
# baselines and the `flops` metric remain available for comparison under
# `--op blackwell_attentions_mxfp8`. Registration keys on the defining module
# path, so the inherited methods exist on the subclass but their configs live
# under the parent's op name until copied here.
_PARENT_OP = "blackwell_attentions"
_OP = "blackwell_attentions_mxfp8"

_dst = REGISTERED_BENCHMARKS.setdefault(_OP, OrderedDict())
for _name, _cfg in REGISTERED_BENCHMARKS.get(_PARENT_OP, {}).items():
    _dst.setdefault(_name, _cfg)

for _metric in REGISTERED_METRICS.get(_PARENT_OP, []):
    if _metric not in REGISTERED_METRICS[_OP]:
        REGISTERED_METRICS[_OP].append(_metric)

for _metric in OVERRIDDEN_METRICS.get(_PARENT_OP, []):
    if _metric not in OVERRIDDEN_METRICS[_OP]:
        OVERRIDDEN_METRICS[_OP].append(_metric)

for _name in BASELINE_BENCHMARKS.get(_PARENT_OP, []):
    BASELINE_BENCHMARKS.setdefault(_OP, [])
    if _name not in BASELINE_BENCHMARKS[_OP]:
        BASELINE_BENCHMARKS[_OP].append(_name)

if _PARENT_OP in REGISTERED_X_VALS:
    REGISTERED_X_VALS.setdefault(_OP, REGISTERED_X_VALS[_PARENT_OP])
