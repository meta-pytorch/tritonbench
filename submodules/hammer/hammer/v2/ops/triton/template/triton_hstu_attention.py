# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

#!/usr/bin/env python3

from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from generative_recommenders.common import (
    autotune_max_seq_len,
    prev_power_of_2,
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from generative_recommenders.ops.triton.triton_attention_utils import acc_dq
from triton.language.extra.libdevice import fast_dividef  # @manual=//triton:triton


def _get_fw_configs() -> List[triton.Config]:  # noqa: C901
    configs = []
    if torch.version.hip:
        for BLOCK_M in [32, 64, 128]:
            for BLOCK_N in [32, 64]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": BLOCK_M,
                                        "BLOCK_N": BLOCK_N,
                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                        "waves_per_eu": 0,
                                        "kpack": 2,
                                    },
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                )
                            )
    else:
        configs = [
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
            ),
        ]
    return configs


def _bwd_pre_hook(nargs):
    nargs["DQ"].zero_()
    if nargs["SEQUENCE_PARALLEL"] is True:
        nargs["LOCK"].zero_()


def _get_bw_configs() -> List[triton.Config]:
    if torch.version.hip:
        configs = []
        for BLOCK_M in [32, 64]:
            for BLOCK_N in [32, 64]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            for waves_per_eu in [0, 2, 4]:
                                for sp in [True, False]:
                                    configs.append(
                                        triton.Config(
                                            {
                                                "BLOCK_M": BLOCK_M,
                                                "BLOCK_N": BLOCK_N,
                                                "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                                "waves_per_eu": waves_per_eu,
                                                "SEQUENCE_PARALLEL": sp,
                                                "UNROLL": 1,
                                            },
                                            num_stages=num_stages,
                                            num_warps=num_warps,
                                            pre_hook=_bwd_pre_hook,
                                        )
                                    )
        return configs

    configs = [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 16, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=3,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 4},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
    ]
    if torch.cuda.is_available():
        configs += [
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
                num_stages=1,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=1,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=3,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=2,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {
                    "BLOCK_M": 32,
                    "BLOCK_N": 128,
                    "SEQUENCE_PARALLEL": False,
                    "UNROLL": 2,
                },
                num_stages=2,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
        ]
    else:
        print("WARNING: temporarily disabled some autotune configs for CUDA 12.8+")
    return configs


@triton.jit
def backward_activation(qk_trans, alpha, scale, valid_mask_trans, k):
    qk_trans = qk_trans * alpha
    sig_trans = fast_dividef(1.0, 1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * scale
    act_qk_trans = tl.where(valid_mask_trans, silu_trans, 0)
    act_qk_trans = act_qk_trans.to(k.dtype)
    return qk_trans, sig_trans, act_qk_trans


@triton.jit
def backward_d_activation(dact_qk_trans, sig_trans, qk_trans, scale, valid_mask_trans):
    dqk_trans = dact_qk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * scale
    dqk_trans = tl.where(valid_mask_trans, dqk_trans, 0)
    return dqk_trans


@triton.jit
def backward_off_common_preprocess(
    seq_len_q,
    contextual_seq_len,
    n_targets,
    offs_n,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
):
    max_ids = seq_len_q
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_n = offs_n - contextual_seq_len + 1
        pos_offs_n = tl.where(
            pos_offs_n > 0,
            pos_offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    else:
        pos_offs_n = offs_n
    if HAS_NUM_TARGETS:
        max_ids = max_ids - n_targets
        pos_offs_n = tl.where(
            pos_offs_n < max_ids,
            pos_offs_n,
            max_ids,
        )
    return max_ids, pos_offs_n


@triton.jit
def backward_valid_mask(
    offs_m,
    pos_offs_n,
    offs_n,
    max_ids,
    contextual_seq_len,
    max_attn_len,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    valid_mask_trans = offs_m[None, :] == offs_n[:, None]
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(
            offs_m > 0,
            offs_m,
            0,
        )
    if HAS_NUM_TARGETS:
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        pos_offs_n = tl.where(
            pos_offs_n < max_ids,
            pos_offs_n,
            max_ids,
        )
    pos_offs_m_minus_n = offs_m[None, :] - pos_offs_n[:, None]
    valid_mask_trans = valid_mask_trans | (pos_offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        valid_mask_trans = valid_mask_trans & (pos_offs_m_minus_n <= max_attn_len)
    if HAS_CONTEXTUAL_SEQ_LEN:
        valid_mask_trans = valid_mask_trans | (
            (offs_m[None, :] == 0) & (pos_offs_n[:, None] < max_ids)
        )
    return valid_mask_trans


@triton.jit
def forward_activation(qk, alpha, scale, valid_mask):
    qk = qk * alpha
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * scale
    act_qk = tl.where(valid_mask, silu, 0)
    return act_qk


@triton.jit
def forward_uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS: tl.constexpr):
    if HAS_NUM_TARGETS:
        uih_end = seq_len_q - n_targets
    else:
        uih_end = seq_len_q
    return uih_end


@triton.jit
def forward_valid_mask(
    offs_m,
    offs_n,
    seq_len_q,
    contextual_seq_len,
    max_attn_len,
    n_targets,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    valid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len_q
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(
            offs_m > 0,
            offs_m,
            0,
        )
        offs_n = offs_n - contextual_seq_len + 1
        offs_n = tl.where(
            offs_n > 0,
            offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    if HAS_NUM_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        offs_n = tl.where(
            offs_n < max_ids,
            offs_n,
            max_ids,
        )
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    valid_mask = valid_mask | (offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        valid_mask = valid_mask & (offs_m_minus_n <= max_attn_len)
    if HAS_CONTEXTUAL_SEQ_LEN:
        valid_mask = valid_mask | ((offs_m[:, None] == 0) & (offs_n[None, :] < max_ids))
    return valid_mask


@triton.jit
def target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS: tl.constexpr):
    if HAS_NUM_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = 0
    return n_targets


@triton.jit
def _hstu_attn_fwd_one_block_0(  # noqa: C901
    start_n,
    seq_len_q,
    seq_len_kv,
    offs_m,
    offs_n,
    off_h,
    q,
    K,
    V,
    acc,
    K_block_ptr,
    V_block_ptr,
    device_desc_k,
    device_desc_v,
    offset_kh,
    offset_vh,
    seq_start_q,
    seq_start_kv,
    alpha,
    scale,
    num_targets,
    max_attn_len,
    contextual_seq_len,
    n_targets,
    uih_end,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = None
    qk = None
    if ENABLE_TMA:
        k = tl._experimental_descriptor_load(
            device_desc_k,
            [(seq_start_kv + start_n).to(tl.int32), offset_kh.to(tl.int32)],
            [BLOCK_N, BLOCK_D_Q],
            K.dtype.element_ty,
        )
        # tma can only be loaded in one order, use trans afterwards
        qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32)
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k, allow_tf32=ALLOW_TF32)
    valid_mask = forward_valid_mask(
        offs_m,
        offs_n,
        seq_len_q,
        contextual_seq_len,
        max_attn_len,
        n_targets,
        HAS_CONTEXTUAL_SEQ_LEN,
        HAS_NUM_TARGETS,
        HAS_MAX_ATTN_LEN,
    )
    act_qk = forward_activation(qk, alpha, scale, valid_mask)
    v = None
    if ENABLE_TMA:
        v = tl._experimental_descriptor_load(
            device_desc_v,
            [(seq_start_kv + start_n).to(tl.int32), offset_vh.to(tl.int32)],
            [BLOCK_N, BLOCK_D_V],
            V.dtype.element_ty,
        )
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    act_qk = act_qk.to(v.dtype)
    acc += tl.dot(act_qk, v, allow_tf32=ALLOW_TF32)
    return acc


@triton.jit
def _hstu_attn_bwd_one_block_0(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs_trans,
    dq_ptrs_trans,
    do_ptrs,
    device_desc_q,
    device_desc_do,
    dk,
    dv,
    k,
    v,
    seq_len_q,
    seq_len_kv,
    LOCK,
    off_h,
    stride_qh,
    stride_doh,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    attn_scale,
    max_q_len,
    num_targets,
    max_attn_len,
    contextual_seq_len,
    n_targets,
    max_ids,
    pos_offs_n,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
):
    offs_m = offs_m + start_m
    mask_m = offs_m < seq_len_q
    if ATTN_SCALE_TYPE == "scalar":
        scale = tl.load(attn_scale).to(tl.float32)
    else:
        tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
        scale = tl.load(attn_scale + offs_m, mask=mask_m).to(tl.float32)
    # recompute qk and silu
    if ENABLE_TMA:
        q = tl._experimental_descriptor_load(
            device_desc_q,
            [start_m, (off_h * stride_qh).to(tl.int32)],
            [BLOCK_M, BLOCK_D_Q],
            k.dtype,
        )
        q_trans = tl.trans(q)
    else:
        q_trans = tl.load(
            q_ptrs_trans + start_m * stride_qm,
            mask=mask_m[None, :],
            other=0.0,
        )
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32)
    valid_mask_trans = backward_valid_mask(
        offs_m,
        pos_offs_n,
        offs_n,
        max_ids,
        contextual_seq_len,
        max_attn_len,
        HAS_CONTEXTUAL_SEQ_LEN,
        HAS_NUM_TARGETS,
        HAS_MAX_ATTN_LEN,
    )
    qk_trans, sig_trans, act_qk_trans = backward_activation(
        qk_trans, alpha, scale, valid_mask_trans, k
    )
    # compute dv
    if ENABLE_TMA:
        do = tl._experimental_descriptor_load(
            device_desc_do,
            [start_m, (off_h * stride_doh).to(tl.int32)],
            [BLOCK_M, BLOCK_D_V],
            k.dtype,
        )
    else:
        do = tl.load(
            do_ptrs + start_m * stride_dom,
            mask=mask_m[:, None],
            other=0.0,
        )
    dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)

    # compute dk and dq
    dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
    dqk_trans = backward_d_activation(
        dact_qk_trans, sig_trans, qk_trans, scale, valid_mask_trans
    )
    dqk_trans = dqk_trans.to(k.dtype)

    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    dk += tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32)
    acc_dq(
        dq_ptrs_trans=dq_ptrs_trans,
        start_m=start_m,
        stride_dqm=stride_dqm,
        k=k,
        dqk_trans=dqk_trans,
        alpha=alpha,
        mask_m=mask_m,
        MAX_SEQ_LEN=max_q_len,
        LOCK=LOCK,
        BLOCK_M=BLOCK_M,
        ATOMIC_ADD=ATOMIC_ADD,
        ALLOW_TF32=ALLOW_TF32,
    )
    return dk, dv


@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
    Q,
    K,
    V,
    H,
    DimQ,
    DimV,
    workspace_ptr,
    seq_offsets,
    seq_offsets_q,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    attn_scale,
    off_z,
    off_h,
    pid,
    num_targets,
    max_attn_len,
    contextual_seq_len,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
):
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    device_desc_q = None
    device_desc_k = None
    device_desc_v = None
    device_desc_o = None
    if ENABLE_TMA:
        workspace_base = workspace_ptr + TMA_DESC_SIZE * 4 * (
            tl.program_id(1) + tl.program_id(0) * tl.num_programs(1)
        )
        device_desc_q = workspace_base
        device_desc_k = workspace_base + 1 * TMA_DESC_SIZE
        device_desc_v = workspace_base + 2 * TMA_DESC_SIZE
        device_desc_o = workspace_base + 3 * TMA_DESC_SIZE

        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_k,
            global_address=K,
            load_size=[
                BLOCK_N,
                BLOCK_D_Q,
            ],
            global_size=[seq_end_kv.to(tl.int32), H * DimQ],
            element_ty=K.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_v,
            global_address=V,
            load_size=[
                BLOCK_N,
                BLOCK_D_V,
            ],
            global_size=[seq_end_kv.to(tl.int32), H * DimV],
            element_ty=V.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_k)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_v)

    start_m = pid * BLOCK_M
    if start_m < seq_len_q:
        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        if ATTN_SCALE_TYPE == "scalar":
            scale = tl.load(attn_scale).to(tl.float32)
        else:
            tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
            scale = tl.load(
                attn_scale + seq_start_q + offs_m, mask=offs_m < seq_len_q
            ).to(tl.float32)

        Q_block_ptr = None
        K_block_ptr = None
        V_block_ptr = None
        if not ENABLE_TMA:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + off_h * stride_qh + seq_start_q * stride_qm,
                shape=(seq_len_q, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

            K_block_ptr = tl.make_block_ptr(
                base=K + off_h * stride_kh + seq_start_kv * stride_kn,
                shape=(BLOCK_D_Q, seq_len_kv),
                strides=(1, stride_kn),
                offsets=(0, 0),
                block_shape=(BLOCK_D_Q, BLOCK_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=V + off_h * stride_vh + seq_start_kv * stride_vn,
                shape=(seq_len_kv, BLOCK_D_V),
                strides=(stride_vn, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D_V),
                order=(1, 0),
            )
        else:
            # pyre-ignore [20]
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=device_desc_q,
                global_address=Q,
                load_size=[
                    BLOCK_M,
                    BLOCK_D_Q,
                ],
                global_size=[seq_end_q.to(tl.int32), H * DimQ],
                element_ty=Q.dtype.element_ty,
            )
            # pyre-ignore [20]
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_q)

            q = tl._experimental_descriptor_load(
                device_desc_q,
                [
                    (seq_start_q + start_m).to(tl.int32),
                    (off_h * stride_qh).to(tl.int32),
                ],
                [
                    BLOCK_M,
                    BLOCK_D_Q,
                ],
                Q.dtype.element_ty,
            )
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        end_n = 0
        n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
        uih_end = forward_uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
        if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
            low = 0
            high = seq_len_q
        else:
            low = 0
            high = start_m + BLOCK_M
            if HAS_MAX_ATTN_LEN:
                if start_m > uih_end:
                    low = uih_end - max_attn_len
                else:
                    low = start_m - max_attn_len
                if HAS_CONTEXTUAL_SEQ_LEN:
                    low = low if low > contextual_seq_len else 0
                else:
                    low = low if low > 0 else 0
            if HAS_NUM_TARGETS:
                uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
                if uih_end < start_m:
                    high = seq_len_q - n_targets
        offset = low
        # single loop compute
        if offset > 0:
            if not ENABLE_TMA:
                K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
        end_n = low
        for start_n in tl.range(low, high, BLOCK_N):
            acc = _hstu_attn_fwd_one_block_0(
                start_n=start_n,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                offs_m=offs_m,
                offs_n=offs_n + start_n,
                off_h=off_h,
                q=q,
                K=K,
                V=V,
                acc=acc,
                K_block_ptr=K_block_ptr,
                V_block_ptr=V_block_ptr,
                device_desc_k=device_desc_k,
                device_desc_v=device_desc_v,
                offset_kh=off_h * stride_kh,
                offset_vh=off_h * stride_vh,
                seq_start_q=seq_start_q,
                seq_start_kv=seq_start_kv,
                alpha=alpha,
                scale=scale,
                num_targets=num_targets,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                n_targets=n_targets,
                uih_end=uih_end,
                HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_N=BLOCK_N,
                ENABLE_TMA=ENABLE_TMA,
            )
            if not ENABLE_TMA:
                K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            end_n += BLOCK_N
        if HAS_NUM_TARGETS:
            if uih_end < start_m:
                low = start_m
                high = start_m + BLOCK_M
                offset = (low - end_n).to(tl.int32)
                # single loop compute
                if offset > 0:
                    if not ENABLE_TMA:
                        K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                        V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
                end_n = low
                for start_n in tl.range(low, high, BLOCK_N, num_stages=0):
                    acc = _hstu_attn_fwd_one_block_0(
                        start_n=start_n,
                        seq_len_q=seq_len_q,
                        seq_len_kv=seq_len_kv,
                        offs_m=offs_m,
                        offs_n=offs_n + start_n,
                        off_h=off_h,
                        q=q,
                        K=K,
                        V=V,
                        acc=acc,
                        K_block_ptr=K_block_ptr,
                        V_block_ptr=V_block_ptr,
                        device_desc_k=device_desc_k,
                        device_desc_v=device_desc_v,
                        offset_kh=off_h * stride_kh,
                        offset_vh=off_h * stride_vh,
                        seq_start_q=seq_start_q,
                        seq_start_kv=seq_start_kv,
                        alpha=alpha,
                        scale=scale,
                        num_targets=num_targets,
                        max_attn_len=max_attn_len,
                        contextual_seq_len=contextual_seq_len,
                        n_targets=n_targets,
                        uih_end=uih_end,
                        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                        BLOCK_N=BLOCK_N,
                        ENABLE_TMA=ENABLE_TMA,
                    )
                    if not ENABLE_TMA:
                        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
                    end_n += BLOCK_N

        if not ENABLE_TMA:
            # rematerialize offsets to save registers
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + seq_start_q * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[:, None])
        else:
            # Important: must cast to proper dtype. If acc is float32, but
            # TMA descriptor specifies float16, the program will run
            # without crashes but produce wrong results.
            acc = acc.to(Out.dtype.element_ty)
            # pyre-ignore [20]
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=device_desc_o,
                global_address=Out,
                load_size=[BLOCK_M, BLOCK_D_V],
                global_size=[seq_end_q.to(tl.int32), H * DimV],
                element_ty=Out.dtype.element_ty,
            )
            # pyre-ignore [20]
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_o)
            tl._experimental_descriptor_store(
                device_desc_o,
                acc,
                [
                    (seq_start_q + pid * BLOCK_M).to(tl.int32),
                    (off_h * stride_oh).to(tl.int32),
                ],
            )


@triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
    ],
)
@triton.jit
def _hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    workspace_ptr,
    sort_by_length_indices,
    seq_offsets,
    seq_offsets_q,
    Out,
    alpha,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    Z,
    AUTOTUNE_Z,
    H,
    attn_scale,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    num_targets,
    max_attn_len,
    contextual_seq_len,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    pid = tl.program_id(0)
    _hstu_attn_fwd_compute(
        Q=Q,
        K=K,
        V=V,
        H=H,
        DimQ=DimQ,
        DimV=DimV,
        workspace_ptr=workspace_ptr,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        Out=Out,
        stride_qm=stride_qm,
        stride_qh=stride_qh,
        stride_kn=stride_kn,
        stride_kh=stride_kh,
        stride_vn=stride_vn,
        stride_vh=stride_vh,
        stride_om=stride_om,
        stride_oh=stride_oh,
        alpha=alpha,
        attn_scale=attn_scale,
        off_z=off_z,
        off_h=off_h,
        pid=pid,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
        ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_D_Q=BLOCK_D_Q,
        BLOCK_D_V=BLOCK_D_V,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        ENABLE_TMA=ENABLE_TMA,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
    )


@triton.jit
def _hstu_attn_bwd_one_col_block(  # noqa C901
    start_n,
    seq_len_q,
    seq_len_kv,
    Q,
    K,
    V,
    DOut,
    DQ,
    DK,
    DV,
    device_desc_q,
    device_desc_k,
    device_desc_v,
    device_desc_do,
    device_desc_dk,
    device_desc_dv,
    LOCK,
    off_h,
    off_z,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dkh,
    stride_dvh,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    alpha,
    attn_scale,
    max_q_len,
    seq_start_q,
    num_targets,
    max_attn_len,
    contextual_seq_len,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dq_ptrs_trans = DQ + (offs_m[None, :] * stride_dqm + offs_qk_d[:, None])
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    if ENABLE_TMA:
        q_ptrs_trans = None
        do_ptrs = None
        k = tl._experimental_descriptor_load(
            device_desc_k,
            [start_n, (off_h * stride_kh).to(tl.int32)],
            [BLOCK_N, BLOCK_D_Q],
            K.dtype.element_ty,
        )
        v = tl._experimental_descriptor_load(
            device_desc_v,
            [start_n, (off_h * stride_vh).to(tl.int32)],
            [BLOCK_N, BLOCK_D_V],
            V.dtype.element_ty,
        )
    else:
        mask_n = offs_n < seq_len_kv
        q_ptrs_trans = Q + (offs_m[None, :] * stride_qm + offs_qk_d[:, None])
        do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
    max_ids, pos_offs_n = backward_off_common_preprocess(
        seq_len_q,
        contextual_seq_len,
        n_targets,
        offs_n,
        HAS_CONTEXTUAL_SEQ_LEN,
        HAS_NUM_TARGETS,
    )
    if HAS_CONTEXTUAL_SEQ_LEN:
        low = 0
        high = contextual_seq_len
        for start_m in tl.range(low, high, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _hstu_attn_bwd_one_block_0(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans,
                do_ptrs=do_ptrs,
                device_desc_q=device_desc_q,
                device_desc_do=device_desc_do,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                LOCK=LOCK,
                off_h=off_h,
                stride_qh=stride_qh,
                stride_doh=stride_doh,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                attn_scale=attn_scale,
                max_q_len=max_q_len,
                num_targets=num_targets,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                n_targets=n_targets,
                max_ids=max_ids,
                pos_offs_n=pos_offs_n,
                HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                ATOMIC_ADD=ATOMIC_ADD,
                ENABLE_TMA=ENABLE_TMA,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
            )
    if HAS_NUM_TARGETS:
        low = start_n
        if HAS_MAX_ATTN_LEN:
            high = start_n + max_attn_len + BLOCK_N
            high = high if high + n_targets < seq_len_q else seq_len_q
        else:
            high = seq_len_q
    else:
        low = start_n
        if HAS_MAX_ATTN_LEN:
            high = start_n + max_attn_len + BLOCK_N
            high = high if high < seq_len_q else seq_len_q
        else:
            high = seq_len_q
    if HAS_CONTEXTUAL_SEQ_LEN:
        contextual_block_end = tl.cdiv(contextual_seq_len, BLOCK_M) * BLOCK_M
        if low < contextual_block_end:
            low = contextual_block_end
    for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _hstu_attn_bwd_one_block_0(
            start_m=start_m,
            offs_n=offs_n,
            offs_m=offs_m,
            q_ptrs_trans=q_ptrs_trans,
            dq_ptrs_trans=dq_ptrs_trans,
            do_ptrs=do_ptrs,
            device_desc_q=device_desc_q,
            device_desc_do=device_desc_do,
            dk=dk,
            dv=dv,
            k=k,
            v=v,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            LOCK=LOCK,
            off_h=off_h,
            stride_qh=stride_qh,
            stride_doh=stride_doh,
            stride_qm=stride_qm,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            alpha=alpha,
            attn_scale=attn_scale,
            max_q_len=max_q_len,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            n_targets=n_targets,
            max_ids=max_ids,
            pos_offs_n=pos_offs_n,
            HAS_NUM_TARGETS=HAS_NUM_TARGETS,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_M=BLOCK_M,
            ATOMIC_ADD=ATOMIC_ADD,
            ENABLE_TMA=ENABLE_TMA,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
        )
    # write-back
    dk = dk * alpha
    if ENABLE_TMA:
        tl._experimental_descriptor_store(
            device_desc_dv,
            dv.to(k.dtype),
            [start_n, (off_h * stride_dvh).to(tl.int32)],
        )
        tl._experimental_descriptor_store(
            device_desc_dk,
            dk.to(k.dtype),
            [start_n, (off_h * stride_dkh).to(tl.int32)],
        )
    else:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
        tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])  # pyre-ignore[61]
        tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])  # pyre-ignore[61]


@triton_autotune(
    configs=_get_bw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
    ],
)
@triton.jit
def _hstu_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    tma_workspace_ptr,
    sort_by_length_indices,
    seq_offsets,
    seq_offsets_q,
    DOut,
    DQ,
    DK,
    DV,
    LOCK,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    attn_scale,
    Z,
    AUTOTUNE_Z,
    H,
    max_q_len,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    num_targets,
    max_attn_len,
    contextual_seq_len,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h = off_h.to(tl.int64)
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)
    # offset pointers for batch/head
    Q = Q + seq_start_q * stride_qm
    K = K + seq_start_kv * stride_kn
    V = V + seq_start_kv * stride_vn
    DOut = DOut + seq_start_q * stride_dom
    DQ = DQ + seq_start_q * stride_dqm + off_h * stride_dqh
    DK = DK + seq_start_kv * stride_dkn
    DV = DV + seq_start_kv * stride_dvn
    device_desc_q = None
    device_desc_k = None
    device_desc_v = None
    device_desc_do = None
    device_desc_dk = None
    device_desc_dv = None
    if ENABLE_TMA:
        workspace_base = tma_workspace_ptr + TMA_DESC_SIZE * 6 * (
            tl.program_id(1) + tl.program_id(0) * tl.num_programs(1)
        )
        device_desc_q = workspace_base
        device_desc_k = workspace_base + 1 * TMA_DESC_SIZE
        device_desc_v = workspace_base + 2 * TMA_DESC_SIZE
        device_desc_do = workspace_base + 3 * TMA_DESC_SIZE
        device_desc_dk = workspace_base + 4 * TMA_DESC_SIZE
        device_desc_dv = workspace_base + 5 * TMA_DESC_SIZE

        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_q,
            global_address=Q,
            load_size=[
                BLOCK_M,
                BLOCK_D_Q,
            ],
            global_size=[seq_len_q, H * DimQ],
            element_ty=Q.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_q)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_do,
            global_address=DOut,
            load_size=[
                BLOCK_M,
                BLOCK_D_V,
            ],
            global_size=[seq_len_q, H * DimV],
            element_ty=DOut.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_do)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_k,
            global_address=K,
            load_size=[
                BLOCK_N,
                BLOCK_D_Q,
            ],
            global_size=[seq_len_kv, H * DimQ],
            element_ty=K.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_k)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_dk,
            global_address=DK,
            load_size=[
                BLOCK_N,
                BLOCK_D_Q,
            ],
            global_size=[seq_len_kv, H * DimQ],
            element_ty=DK.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_dk)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_v,
            global_address=V,
            load_size=[
                BLOCK_N,
                BLOCK_D_V,
            ],
            global_size=[seq_len_kv, H * DimV],
            element_ty=V.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_v)
        # pyre-ignore [20]
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=device_desc_dv,
            global_address=DV,
            load_size=[
                BLOCK_N,
                BLOCK_D_V,
            ],
            global_size=[seq_len_kv, H * DimV],
            element_ty=DV.dtype.element_ty,
        )
        # pyre-ignore [20]
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_dv)
    else:
        Q += off_h * stride_qh
        K += off_h * stride_kh
        V += off_h * stride_vh
        DOut += off_h * stride_doh
        DK += off_h * stride_dkh
        DV += off_h * stride_dvh
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len_kv:
            return
        _hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            Q=Q,
            K=K,
            V=V,
            DOut=DOut,
            DQ=DQ,
            DK=DK,
            DV=DV,
            device_desc_q=device_desc_q,
            device_desc_k=device_desc_k,
            device_desc_v=device_desc_v,
            device_desc_do=device_desc_do,
            device_desc_dk=device_desc_dk,
            device_desc_dv=device_desc_dv,
            LOCK=LOCK,
            off_h=off_h,
            off_z=off_z,
            stride_qh=stride_qh,
            stride_kh=stride_kh,
            stride_vh=stride_vh,
            stride_doh=stride_doh,
            stride_dkh=stride_dkh,
            stride_dvh=stride_dvh,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            stride_dkn=stride_dkn,
            stride_dvn=stride_dvn,
            alpha=alpha,
            attn_scale=attn_scale,
            max_q_len=max_q_len,
            seq_start_q=seq_start_q,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            HAS_NUM_TARGETS=HAS_NUM_TARGETS,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            UNROLL=UNROLL,
            ATOMIC_ADD=True,
            ENABLE_TMA=ENABLE_TMA,
        )
    else:
        for start_n in range(0, seq_len_kv, BLOCK_N):
            _hstu_attn_bwd_one_col_block(
                start_n=start_n,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                Q=Q,
                K=K,
                V=V,
                DOut=DOut,
                DQ=DQ,
                DK=DK,
                DV=DV,
                device_desc_q=device_desc_q,
                device_desc_k=device_desc_k,
                device_desc_v=device_desc_v,
                device_desc_do=device_desc_do,
                device_desc_dk=device_desc_dk,
                device_desc_dv=device_desc_dv,
                LOCK=LOCK,
                off_h=off_h,
                off_z=off_z,
                stride_qh=stride_qh,
                stride_kh=stride_kh,
                stride_vh=stride_vh,
                stride_doh=stride_doh,
                stride_dkh=stride_dkh,
                stride_dvh=stride_dvh,
                stride_qm=stride_qm,
                stride_kn=stride_kn,
                stride_vn=stride_vn,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                stride_dkn=stride_dkn,
                stride_dvn=stride_dvn,
                alpha=alpha,
                attn_scale=attn_scale,
                max_q_len=max_q_len,
                seq_start_q=seq_start_q,
                num_targets=num_targets,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                ATTN_SCALE_TYPE=ATTN_SCALE_TYPE,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                ATOMIC_ADD=False,
                ENABLE_TMA=ENABLE_TMA,
            )


def triton_hstu_attention_fwd(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    sort_by_length_indices: Optional[torch.Tensor],
    enable_tma: bool,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
) -> torch.Tensor:
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets
    total_seq_len_q, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
    if total_seq_len_q == 0:
        return out
    TMA_DESC_SIZE = 128
    workspace = None
    if enable_tma:
        MIN_BLOCK_M = 16
        workspace = torch.empty(
            4 * TMA_DESC_SIZE * (triton.cdiv(max_q_len, MIN_BLOCK_M) * Z * H),
            dtype=torch.uint8,
            device="cuda",
        )
    if attn_scale.ndim == 0:
        attn_scale_type = "scalar"
    else:
        attn_scale_type = "dynamic"
    grid = lambda meta: (  # noqa E731
        triton.cdiv(max_q_len, meta["BLOCK_M"]),
        Z * H,
    )
    HAS_NUM_TARGETS = num_targets is not None
    HAS_MAX_ATTN_LEN = max_attn_len != 0
    HAS_CONTEXTUAL_SEQ_LEN = contextual_seq_len != 0
    _hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        workspace_ptr=workspace,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        Out=out,
        alpha=alpha,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        attn_scale=attn_scale,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        DimQ=DimQ,
        DimV=DimV,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
        ATTN_SCALE_TYPE=attn_scale_type,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
    )
    return out


def triton_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    sort_by_length_indices: Optional[torch.Tensor],
    enable_tma: bool,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    dv = switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets
    _, H, DimQ = q.shape
    _, _, DimV = v.shape
    if attn_scale.ndim == 0:
        attn_scale_type = "scalar"
    else:
        attn_scale_type = "dynamic"
    grid = lambda meta: (  # noqa E731
        Z * H,
        (triton.cdiv(max_seq_len, meta["BLOCK_N"]) if meta["SEQUENCE_PARALLEL"] else 1),
    )
    # The minimum size of BLOCK_M used in `_get_bw_configs`.
    # TODO (linjianma): avoid hardcoding the value.
    MIN_BLOCK_M = 16
    lock = torch.empty(
        (Z * H, triton.cdiv(max_q_len, MIN_BLOCK_M)),
        dtype=torch.int32,
        device=q.device,
    )
    AUTOTUNE_Z = prev_power_of_2(Z)
    TMA_DESC_SIZE = 128
    tma_workspace = None
    if enable_tma:
        MIN_BLOCK_N = 16
        tma_workspace = torch.empty(
            6 * TMA_DESC_SIZE * (triton.cdiv(max_seq_len, MIN_BLOCK_N) * Z * H),
            dtype=torch.uint8,
            device="cuda",
        )
    HAS_NUM_TARGETS = num_targets is not None
    HAS_MAX_ATTN_LEN = max_attn_len != 0
    HAS_CONTEXTUAL_SEQ_LEN = contextual_seq_len != 0
    _hstu_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        tma_workspace_ptr=tma_workspace,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        DOut=dout,
        DQ=dq,
        DK=dk,
        DV=dv,
        LOCK=lock,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        attn_scale=attn_scale,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        max_q_len=max_q_len,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        DimQ=DimQ,
        DimV=DimV,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
        ATTN_SCALE_TYPE=attn_scale_type,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
    )

    return dq, dk, dv


class _AttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        attn_scale: torch.Tensor,
        max_q_len: Optional[int],
        seq_offsets_q: Optional[torch.Tensor],
        sort_by_length: bool,
        enable_tma: bool,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets, attn_scale]
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        if seq_offsets_q is not None:
            saved_tensors.append(seq_offsets_q)
        if num_targets is not None:
            saved_tensors.append(num_targets)
        ctx.has_num_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.contextual_seq_len = contextual_seq_len

        ctx.alpha = alpha
        ctx.max_seq_len = max_seq_len
        ctx.max_q_len = max_q_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        out = triton_hstu_attention_fwd(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            attn_scale=attn_scale,
            max_q_len=max_q_len,
            seq_offsets_q=seq_offsets_q,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=enable_tma,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
        )
        saved_tensors.append(out)
        ctx.save_for_backward(*saved_tensors)
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        saved_tensors = ctx.saved_tensors
        q, k, v, seq_offsets, attn_scale = saved_tensors[:5]
        idx = 5
        if ctx.sort_by_length:
            sort_by_length_indices = saved_tensors[idx]
            idx += 1
        else:
            sort_by_length_indices = None
        if ctx.max_q_len is not None:
            seq_offsets_q = saved_tensors[idx]
            idx += 1
        else:
            seq_offsets_q = None
        if ctx.has_num_targets:
            num_targets = ctx.saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        max_attn_len = ctx.max_attn_len
        contextual_seq_len = ctx.contextual_seq_len

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq, dk, dv = triton_hstu_attention_bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            seq_offsets=seq_offsets,
            attn_scale=attn_scale,
            max_seq_len=ctx.max_seq_len,
            alpha=ctx.alpha,
            max_q_len=ctx.max_q_len,
            seq_offsets_q=seq_offsets_q,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=ctx.enable_tma,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
        )
        return (
            None,
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.fx.wrap
def triton_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    enable_tma: bool = False,
    sort_by_length: bool = False,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    return _AttentionFunction.apply(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        attn_scale,
        max_q_len,
        seq_offsets_q,
        sort_by_length,
        enable_tma,
        num_targets,
        max_attn_len,
        contextual_seq_len,
    )


def triton_hstu_mha_wrapper(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    enable_tma: bool = False,
    sort_by_length: bool = False,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    return triton_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        attn_scale=attn_scale,
        max_q_len=max_q_len,
        seq_offsets_q=seq_offsets_q,
        enable_tma=enable_tma,
        sort_by_length=sort_by_length,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
