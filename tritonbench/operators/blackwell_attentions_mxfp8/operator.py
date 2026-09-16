# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 variant of the Blackwell attention operator.

This operator reuses the inputs, metrics, and BF16 baselines from
``blackwell_attentions`` and adds the TLX Ops MXFP8 forward kernel.

Benchmark registration keys on the defining module path, so subclassing the
parent operator does not expose the parent's backends and metrics under this
operator's name. The parent registry buckets are therefore cloned below.
"""

from collections import OrderedDict
from typing import Callable

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
    # This benchmark intentionally uses the SM100 backend's prepared-input path
    # so timed iterations exclude MXFP8 quantization, matching the prior
    # tutorial benchmark's kernel-only measurement.
    # @manual=//triton:triton
    from triton.tlx.ops.kernels.flash_attn_mxfp8.sm100 import (
        _forward_prequantized_with_lse,
        _quantize_mxfp8_operand,
    )

    HAS_TLX_MXFP8 = True
except (ImportError, IOError, AttributeError, TypeError):
    HAS_TLX_MXFP8 = False


class Operator(BlackwellAttentionsOperator):
    @register_benchmark(enabled=HAS_TLX_MXFP8, label="tlx-mxfp8", fwd_only=True)
    def tlx_blackwell_mxfp8(self, *args) -> Callable:
        self.optims.clear()
        assert len(args) % 3 == 0
        prepared = []
        for i in range(0, len(args), 3):
            q, k, v = args[i : i + 3]
            q_fp8, q_scale = _quantize_mxfp8_operand(q)
            k_fp8, k_scale = _quantize_mxfp8_operand(k)
            v_fp8, v_scale = _quantize_mxfp8_operand(v, transpose_for_reduction=True)
            prepared.append((q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale))

        def fn():
            outputs = []
            for q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale in prepared:
                output, _ = _forward_prequantized_with_lse(
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
