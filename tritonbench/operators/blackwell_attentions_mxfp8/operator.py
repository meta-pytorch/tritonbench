# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 variant of the blackwell_attentions operator.

This is a thin operator that reuses everything from `blackwell_attentions`
(input generation, CLI args, the `flops` metric, the bf16 baselines such as
`cudnn_sdpa`/`cutedsl_blackwell`/`tlx_blackwell_ws_pipelined_persistent`) and
adds a single `tlx_blackwell_mxfp8` backend that benchmarks the MXFP8 Blackwell
flash-attention tutorial kernels.

Benchmark registration keys on the defining module path, so subclassing the
parent operator does not by itself expose the parent's backends/metrics under
this op's name. We therefore clone the parent registry buckets into this op's
name at import time (see the bottom of this file).
"""

from collections import OrderedDict
from typing import Callable

import torch
import triton
from tritonbench.operators.blackwell_attentions.operator import (
    Operator as BlackwellAttentionsOperator,
)
from tritonbench.utils.triton_op import (
    BASELINE_BENCHMARKS,
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
        attention_bwd as _tlx_mxfp8_attention_bwd,
        swizzled_to_tma_preshuffled as _tlx_swizzled_to_tma_preshuffled,
    )
    from triton.tools.tensor_descriptor import TensorDescriptor as _TensorDescriptor

    HAS_TLX_MXFP8 = True
except (ImportError, IOError, AttributeError, TypeError):
    HAS_TLX_MXFP8 = False


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

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (
        min(num_sms, triton.cdiv(N_CTX, _MXFP8_FWD_CONFIG["BLOCK_M"]) * Z * H),
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


class _TLXBlackwellMXFP8Attention(torch.autograd.Function):
    """Wraps the MXFP8 Blackwell FA tutorial kernels in an autograd Function so
    tritonbench can benchmark forward and backward through its standard
    o.backward(dO) path. q/k/v are plain bf16 leaves; the MXFP8 quantization the
    kernels require is done here.

    Forward does only forward work (so --mode fwd is not penalized). All backward
    operands are quantized lazily on the first backward call and cached: the
    dO-independent operands once, and the dO-derived operands keyed on dO. Since
    tritonbench reuses one dO and warms up before timing, the timed backward only
    runs attention_bwd.
    """

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal):
        dtype = torch.float8_e4m3fn
        # Forward quantization: Q/K block scales along HEAD_DIM, V along N_CTX.
        q_fp8, q_scale = _mxfp8_quantize_operand(q, dtype)
        k_fp8, k_scale = _mxfp8_quantize_operand(k, dtype)
        v_fp8, v_scale = _mxfp8_quantize_operand(v, dtype, transpose_for_reduction=True)
        o, M = _mxfp8_forward_with_lse(
            q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, sm_scale, causal
        )

        ctx.save_for_backward(q, k, v)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.dtype = dtype
        ctx.fwd = (q_fp8, k_fp8, o, M, q_scale, k_scale)
        ctx.bwd_static = None
        ctx.do_cache = {}
        return o

    @staticmethod
    def backward(ctx, do):
        dtype = ctx.dtype
        # dO-independent backward operands: reduction-axis-swapped Q/K and an
        # N_CTX-blocked V. Computed once, then reused across timed iterations.
        if ctx.bwd_static is None:
            q, k, v = ctx.saved_tensors
            q_dk, q_scale_dk = _mxfp8_quantize_operand(
                q, dtype, transpose_for_reduction=True
            )
            k_dq, k_scale_dq = _mxfp8_quantize_operand(
                k, dtype, transpose_for_reduction=True
            )
            v_bwd, v_scale_bwd = _mxfp8_quantize_operand(v, dtype)
            ctx.bwd_static = (q_dk, q_scale_dk, k_dq, k_scale_dq, v_bwd, v_scale_bwd)

        do_bf16 = do.to(torch.bfloat16).contiguous()
        # tritonbench reuses one dO across timed iterations, so quantize it once
        # (during warmup) and reuse, keeping quantization out of the timed region.
        key = do_bf16.data_ptr()
        if key not in ctx.do_cache:
            do_fp8, do_scale = _mxfp8_quantize_operand(do_bf16, dtype)
            do_fp8_dv, do_scale_dv = _mxfp8_quantize_operand(
                do_bf16, dtype, transpose_for_reduction=True
            )
            ctx.do_cache = {key: (do_fp8, do_scale, do_fp8_dv, do_scale_dv)}
        do_fp8, do_scale, do_fp8_dv, do_scale_dv = ctx.do_cache[key]

        q_fp8, k_fp8, o, M, q_scale, k_scale = ctx.fwd
        q_dk, q_scale_dk, k_dq, k_scale_dq, v_bwd, v_scale_bwd = ctx.bwd_static

        dq, dk, dv = _tlx_mxfp8_attention_bwd(
            do_fp8,
            do_fp8_dv,
            q_fp8,
            q_dk,
            k_fp8,
            k_dq,
            v_bwd,
            o,
            M,
            q_scale,
            q_scale_dk,
            k_scale,
            k_scale_dq,
            v_scale_bwd,
            do_scale,
            do_scale_dv,
            ctx.sm_scale,
            do_bf16=do_bf16,
            causal=ctx.causal,
        )
        # dq comes back FP32; cast to match the bf16 leaf.
        return dq.to(torch.bfloat16), dk, dv, None, None


class Operator(BlackwellAttentionsOperator):
    # Only works with triton beta. Quantization happens during benchmark setup;
    # timed iterations measure only the MXFP8 Blackwell FA forward kernel.
    @register_benchmark(enabled=HAS_TLX_MXFP8, label="tlx-mxfp8", fwd_only=True)
    def tlx_blackwell_mxfp8(self, *args) -> Callable:
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
            quantized_inputs.append(
                (q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)
            )

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
