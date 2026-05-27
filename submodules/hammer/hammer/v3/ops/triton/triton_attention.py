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


from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from generative_recommenders.common import (
    autotune_max_seq_len,
    prev_power_of_2,
    triton_autotune,
)
from hammer.v3.ops.pytorch.pt_attention import MaskType
from hammer.v3.ops.triton.vararg_kernel import unroll_varargs, VarargMode
from triton.language.extra.libdevice import (  # @manual=//triton:triton
    fast_dividef,
    fast_expf,
)

VAR_ARGS_ARRAY_Q = List[Any]  # noqa: F841
VAR_ARGS_ARRAY_KV = List[Any]  # noqa: F841

MASK_CAUSAL = MaskType.CAUSAL.value
MASK_ALL = MaskType.ALL.value
MASK_DIAGONAL = MaskType.DIAGONAL.value
MASK_NULL = MaskType.NULL.value
MASK_LOCAL = MaskType.LOCAL.value

# Check if hardware fast tanh instruction is available (H100+)
HAS_FAST_TANH_INSTRUCTION = (
    torch.version.cuda is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9  # >= H100
)

if HAS_FAST_TANH_INSTRUCTION:

    @triton.jit
    def tanh_approx_fp32(x):
        output = tl.inline_asm_elementwise(
            asm="""
            tanh.approx.f32 $0, $1;
            """,
            constraints="=r,r",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return output

    @triton.jit
    def fast_silu(x):
        # Replace divf(x, 1 + expf(-x)) with x * 0.5 * (tanh(x/2) + 1)
        # Uses hardware tanh.approx.f32 instruction on H100+
        x = x * 0.5
        return x * (tanh_approx_fp32(x) + 1)

    @triton.jit
    def fast_sigmoid(x):
        # sigmoid(x) = 0.5 * (tanh(x/2) + 1)
        # Uses hardware tanh.approx.f32 instruction on H100+
        return 0.5 * (tanh_approx_fp32(x * 0.5) + 1.0)

else:

    @triton.jit
    def fast_silu(x):
        return fast_dividef(x, 1.0 + fast_expf(-x))

    @triton.jit
    def fast_sigmoid(x):
        return fast_dividef(1.0, 1.0 + fast_expf(-x))


def _flatten_mask_matrix(mask_matrix: List[List[MaskType]]) -> Tuple[int, ...]:
    return tuple(m.value for row in mask_matrix for m in row)


def _create_mask_tensor(
    mask_matrix: List[List[MaskType]], device: torch.device
) -> torch.Tensor:
    flattened = [m.value for row in mask_matrix for m in row]
    return torch.tensor(flattened, dtype=torch.int32, device=device)


def _get_fw_configs() -> List[triton.Config]:
    """Get autotuning configurations for forward pass."""
    if torch.version.hip:
        configs = []
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


@lru_cache(maxsize=None)
def _get_autotune_kernel_mha(kernel: Callable) -> Callable:
    return triton_autotune(
        configs=_get_fw_configs(),
        key=[
            "AUTOTUNE_Z",
            "H",
            "AUTOTUNE_MAX_SEQ_LEN",
            "DimQ",
            "DimV",
        ],
    )(kernel)


def _get_bw_configs() -> List[triton.Config]:
    """Get autotuning configurations for backward pass."""
    if torch.version.hip:
        configs = []
        for BLOCK_M in [32, 64]:
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
                {"BLOCK_M": 16, "BLOCK_N": 16},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 32},
                num_stages=1,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 64},
                num_stages=1,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=1,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=1,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=1,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=1,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=1,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=3,
                num_warps=8,
            ),
        ]

    return configs


@lru_cache(maxsize=None)
def _get_autotune_kernel_mha_bwd(kernel: Callable) -> Callable:
    return triton_autotune(
        configs=_get_bw_configs(),
        key=[
            "AUTOTUNE_Z",
            "H",
            "AUTOTUNE_MAX_SEQ_LEN",
            "DimQ",
            "DimV",
        ],
    )(kernel)


@triton.jit
def _mha_fwd_compute_list_varargs(  # noqa: C901
    Q_ptrs: "VAR_ARGS_ARRAY_Q",
    K_ptrs: "VAR_ARGS_ARRAY_KV",
    V_ptrs: "VAR_ARGS_ARRAY_KV",
    Out_ptrs: "VAR_ARGS_ARRAY_Q",
    AttnScale_ptrs: "VAR_ARGS_ARRAY_Q",
    q_seq_offsets_tensor,
    kv_seq_offsets_tensor,
    q_cumsum_lengths,
    kv_cumsum_lengths,
    max_q_len_tensor,
    mask_tensor,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    stride_q_cumsum_t,
    stride_q_cumsum_z,
    stride_kv_cumsum_t,
    stride_kv_cumsum_z,
    stride_q_so_t,
    stride_q_so_b,
    stride_kv_so_t,
    stride_kv_so_b,
    alpha,
    max_attn_len,
    workspace_ptr,
    AUTOTUNE_Z,
    H,
    AUTOTUNE_MAX_SEQ_LEN,
    DimQ,
    DimV,
    NUM_Q: tl.constexpr,
    NUM_KV: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    # pyre-ignore[9]
    SINGLE_BLOCK: tl.constexpr = False,
    # pyre-ignore[9]
    ENABLE_TMA: tl.constexpr = False,
    # pyre-ignore[9]
    TMA_DESC_SIZE: tl.constexpr = 128,
):
    if SINGLE_BLOCK:
        t_q = 0
        t_kv = 0
        pid = tl.program_id(0)
        off_hz = tl.program_id(1)

        off_z = off_hz // H
        off_h = off_hz % H
        off_h = off_h.to(tl.int64)
        off_z_i64 = off_z.to(tl.int64)

        t_seq_start = tl.load(q_seq_offsets_tensor + off_z_i64 * stride_q_so_b).to(
            tl.int64
        )
        t_seq_end = tl.load(q_seq_offsets_tensor + (off_z_i64 + 1) * stride_q_so_b).to(
            tl.int64
        )
        q_seq_len = (t_seq_end - t_seq_start).to(tl.int32)

        kv_t_seq_start = tl.load(kv_seq_offsets_tensor + off_z_i64 * stride_kv_so_b).to(
            tl.int64
        )
        kv_t_seq_end = tl.load(
            kv_seq_offsets_tensor + (off_z_i64 + 1) * stride_kv_so_b
        ).to(tl.int64)
        kv_seq_len = (kv_t_seq_end - kv_t_seq_start).to(tl.int32)

        start_m_local = pid * BLOCK_M
        if start_m_local >= q_seq_len:
            return

        offs_m_local = start_m_local + tl.arange(0, BLOCK_M)
        rows_valid = offs_m_local < q_seq_len

        # TMA descriptor setup
        device_desc_k = None
        device_desc_v = None
        if ENABLE_TMA:
            tma_base = workspace_ptr + TMA_DESC_SIZE * 3 * (
                pid + off_hz * tl.num_programs(0)
            )
            device_desc_q = tma_base
            device_desc_k = tma_base + 1 * TMA_DESC_SIZE
            device_desc_v = tma_base + 2 * TMA_DESC_SIZE

            Q_base_t = Q_ptrs[t_q]
            K_base_t = K_ptrs[t_kv]
            V_base_t = V_ptrs[t_kv]

            # pyre-ignore [20]
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=device_desc_q,
                global_address=Q_base_t,
                load_size=[BLOCK_M, BLOCK_D_Q],
                global_size=[t_seq_end.to(tl.int32), H * DimQ],
                element_ty=Q_base_t.dtype.element_ty,
            )
            # pyre-ignore [20]
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=device_desc_k,
                global_address=K_base_t,
                load_size=[BLOCK_N, BLOCK_D_Q],
                global_size=[kv_t_seq_end.to(tl.int32), H * DimQ],
                element_ty=K_base_t.dtype.element_ty,
            )
            # pyre-ignore [20]
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=device_desc_v,
                global_address=V_base_t,
                load_size=[BLOCK_N, BLOCK_D_V],
                global_size=[kv_t_seq_end.to(tl.int32), H * DimV],
                element_ty=V_base_t.dtype.element_ty,
            )
            # pyre-ignore [20]
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_q)
            # pyre-ignore [20]
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_k)
            # pyre-ignore [20]
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_v)

            offset_kh = (off_h * stride_kh).to(tl.int32)
            offset_vh = (off_h * stride_vh).to(tl.int32)

            q = tl._experimental_descriptor_load(
                device_desc_q,
                [
                    (t_seq_start + start_m_local).to(tl.int32),
                    (off_h * stride_qh).to(tl.int32),
                ],
                [BLOCK_M, BLOCK_D_Q],
                Q_base_t.dtype.element_ty,
            ).to(tl.bfloat16)
        else:
            Q_base_t = Q_ptrs[t_q]
            Q_block_ptr = tl.make_block_ptr(
                base=Q_base_t + off_h * stride_qh + t_seq_start * stride_qm,
                shape=(q_seq_len, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m_local, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero").to(
                tl.bfloat16
            )

        AttnScale_base_t = AttnScale_ptrs[t_q]
        scale_ptrs_t = AttnScale_base_t + t_seq_start + offs_m_local
        attn_scale = tl.load(scale_ptrs_t, mask=rows_valid, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)

        K_base_t = K_ptrs[t_kv]
        V_base_t = V_ptrs[t_kv]
        if not ENABLE_TMA:
            K_base_offset = K_base_t + off_h * stride_kh + kv_t_seq_start * stride_kn
            V_base_offset = V_base_t + off_h * stride_vh + kv_t_seq_start * stride_vn

        cur_mask = tl.load(mask_tensor).to(tl.int32)

        # Compute loop bounds
        delta = kv_seq_len - q_seq_len
        if cur_mask == MASK_CAUSAL:
            kv_loop_end = tl.minimum(
                kv_seq_len,
                ((start_m_local + BLOCK_M + delta + BLOCK_N) // BLOCK_N) * BLOCK_N,
            )
            unmasked_end = start_m_local + delta - BLOCK_N + 1
            unmasked_end = (unmasked_end // BLOCK_N) * BLOCK_N
            unmasked_end = tl.maximum(unmasked_end, 0)
            unmasked_end = tl.minimum(unmasked_end, kv_loop_end)
            loop_start = 0
        elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
            min_valid_k_pos = start_m_local + delta - max_attn_len
            min_valid_k_pos = tl.maximum(min_valid_k_pos, 0)
            loop_start = (min_valid_k_pos // BLOCK_N) * BLOCK_N
            kv_loop_end = tl.minimum(
                kv_seq_len,
                ((start_m_local + BLOCK_M + delta + BLOCK_N) // BLOCK_N) * BLOCK_N,
            )
            unmasked_end = loop_start
        elif cur_mask == MASK_ALL:
            kv_loop_end = kv_seq_len
            unmasked_end = (kv_seq_len // BLOCK_N) * BLOCK_N
            loop_start = 0
        elif cur_mask == MASK_DIAGONAL:
            loop_start = (start_m_local // BLOCK_N) * BLOCK_N
            loop_start = tl.maximum(0, loop_start - BLOCK_N)
            kv_loop_end = tl.minimum(
                kv_seq_len,
                ((start_m_local + BLOCK_M + BLOCK_N) // BLOCK_N) * BLOCK_N,
            )
            unmasked_end = loop_start
        else:
            kv_loop_end = kv_seq_len
            unmasked_end = (kv_seq_len // BLOCK_N) * BLOCK_N
            loop_start = 0

        # Ensure unmasked_end only covers full blocks
        seq_aligned = (kv_seq_len // BLOCK_N) * BLOCK_N
        unmasked_end = tl.minimum(unmasked_end, seq_aligned)

        # Create K/V block pointers
        K_block_ptr = None
        V_block_ptr = None
        if not ENABLE_TMA:
            K_block_ptr = tl.make_block_ptr(
                # pyre-ignore[61]
                base=K_base_offset,
                shape=(BLOCK_D_Q, kv_seq_len),
                strides=(1, stride_kn),
                offsets=(0, loop_start),
                block_shape=(BLOCK_D_Q, BLOCK_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                # pyre-ignore[61]
                base=V_base_offset,
                shape=(kv_seq_len, BLOCK_D_V),
                strides=(stride_vn, 1),
                offsets=(loop_start, 0),
                block_shape=(BLOCK_N, BLOCK_D_V),
                order=(1, 0),
            )

        # Unmasked region - no mask needed
        for start_n in range(loop_start, unmasked_end, BLOCK_N):
            if ENABLE_TMA:
                k = tl._experimental_descriptor_load(
                    device_desc_k,
                    # pyre-ignore[61]
                    [(kv_t_seq_start + start_n).to(tl.int32), offset_kh],
                    [BLOCK_N, BLOCK_D_Q],
                    K_base_t.dtype.element_ty,
                ).to(tl.bfloat16)
                qk = tl.dot(q, tl.trans(k)).to(tl.float32) * alpha
            else:
                k = tl.load(K_block_ptr).to(tl.bfloat16)
                qk = tl.dot(q, k).to(tl.float32) * alpha

            silu = fast_silu(qk) * attn_scale[:, None]

            if ENABLE_TMA:
                v = tl._experimental_descriptor_load(
                    device_desc_v,
                    # pyre-ignore[61]
                    [(kv_t_seq_start + start_n).to(tl.int32), offset_vh],
                    [BLOCK_N, BLOCK_D_V],
                    V_base_t.dtype.element_ty,
                ).to(tl.bfloat16)
            else:
                v = tl.load(V_block_ptr).to(tl.bfloat16)

            silu = silu.to(v.dtype)
            acc += tl.dot(silu, v)

            if not ENABLE_TMA:
                K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        # Masked region
        for start_n in range(unmasked_end, kv_loop_end, BLOCK_N):
            if ENABLE_TMA:
                k = tl._experimental_descriptor_load(
                    device_desc_k,
                    # pyre-ignore[61]
                    [(kv_t_seq_start + start_n).to(tl.int32), offset_kh],
                    [BLOCK_N, BLOCK_D_Q],
                    K_base_t.dtype.element_ty,
                ).to(tl.bfloat16)
                qk = tl.dot(q, tl.trans(k)).to(tl.float32) * alpha
            else:
                k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero").to(
                    tl.bfloat16
                )
                qk = tl.dot(q, k).to(tl.float32) * alpha

            offs_n = tl.arange(0, BLOCK_N)
            k_local_pos = start_n + offs_n
            cols_valid = k_local_pos < kv_seq_len

            if cur_mask == MASK_CAUSAL:
                # pyre-ignore[16]
                causal_mask = (offs_m_local[:, None] + delta) >= k_local_pos[None, :]
                # pyre-ignore[16]
                valid_mask = rows_valid[:, None] & cols_valid[None, :] & causal_mask
            elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
                q_shifted = offs_m_local[:, None] + delta
                causal_mask = q_shifted >= k_local_pos[None, :]
                local_mask = (q_shifted - k_local_pos[None, :]) < max_attn_len
                valid_mask = (
                    rows_valid[:, None] & cols_valid[None, :] & causal_mask & local_mask
                )
            elif cur_mask == MASK_ALL:
                valid_mask = rows_valid[:, None] & cols_valid[None, :]
            elif cur_mask == MASK_DIAGONAL:
                diag_mask = offs_m_local[:, None] == k_local_pos[None, :]
                valid_mask = rows_valid[:, None] & cols_valid[None, :] & diag_mask
            else:
                valid_mask = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)

            scale = tl.where(valid_mask, attn_scale[:, None], 0.0)
            silu = fast_silu(qk) * scale

            if ENABLE_TMA:
                v = tl._experimental_descriptor_load(
                    device_desc_v,
                    # pyre-ignore[61]
                    [(kv_t_seq_start + start_n).to(tl.int32), offset_vh],
                    [BLOCK_N, BLOCK_D_V],
                    V_base_t.dtype.element_ty,
                ).to(tl.bfloat16)
            else:
                v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero").to(
                    tl.bfloat16
                )

            silu = silu.to(v.dtype)
            acc += tl.dot(silu, v)

            if not ENABLE_TMA:
                K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        # Output store
        offs_v_d = tl.arange(0, BLOCK_D_V)
        Out_base_t = Out_ptrs[t_q]
        out_ptrs_t = (
            Out_base_t
            + off_h * stride_oh
            + (t_seq_start + offs_m_local)[:, None] * stride_om
            + offs_v_d[None, :]
        )
        tl.store(out_ptrs_t, acc, mask=rows_valid[:, None])
    else:
        t_q = tl.program_id(0)
        pid = tl.program_id(1)
        off_hz = tl.program_id(2)

        off_z = off_hz // H
        off_h = off_hz % H
        off_h = off_h.to(tl.int64)
        off_z_i64 = off_z.to(tl.int64)
        q_cumsum_ptr = q_cumsum_lengths + off_z_i64 * stride_q_cumsum_z
        kv_cumsum_ptr = kv_cumsum_lengths + off_z_i64 * stride_kv_cumsum_z

        max_q_len = tl.load(max_q_len_tensor + t_q).to(tl.int32)

        start_m_local = pid * BLOCK_M
        if start_m_local >= max_q_len:
            return

        t_q_cumsum_start = tl.load(q_cumsum_ptr + t_q * stride_q_cumsum_t)
        t_q_cumsum_end = tl.load(q_cumsum_ptr + (t_q + 1) * stride_q_cumsum_t)
        q_block_len = t_q_cumsum_end - t_q_cumsum_start

        if start_m_local >= q_block_len:
            return

        start_m = t_q_cumsum_start + start_m_local
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_m_local = start_m_local + tl.arange(0, BLOCK_M)

        rows_valid = offs_m_local < q_block_len

        t_seq_start = tl.load(
            q_seq_offsets_tensor + t_q * stride_q_so_t + off_z_i64 * stride_q_so_b
        ).to(tl.int64)

        # TMA descriptor setup
        device_desc_k = None
        device_desc_v = None
        if ENABLE_TMA:
            tma_base = workspace_ptr + TMA_DESC_SIZE * 3 * (
                (t_q * tl.num_programs(1) + pid) * tl.num_programs(2) + off_hz
            )
            device_desc_q = tma_base
            device_desc_k = tma_base + 1 * TMA_DESC_SIZE
            device_desc_v = tma_base + 2 * TMA_DESC_SIZE

            Q_base_t = Q_ptrs[t_q]
            q_t_seq_end_tma = tl.load(
                q_seq_offsets_tensor
                + t_q * stride_q_so_t
                + (off_z_i64 + 1) * stride_q_so_b
            ).to(tl.int32)

            # pyre-ignore [20]
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=device_desc_q,
                global_address=Q_base_t,
                load_size=[BLOCK_M, BLOCK_D_Q],
                global_size=[q_t_seq_end_tma, H * DimQ],
                element_ty=Q_base_t.dtype.element_ty,
            )
            # pyre-ignore [20]
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_q)

            offset_kh = (off_h * stride_kh).to(tl.int32)
            offset_vh = (off_h * stride_vh).to(tl.int32)

            q = tl._experimental_descriptor_load(
                device_desc_q,
                [
                    (t_seq_start + start_m_local).to(tl.int32),
                    (off_h * stride_qh).to(tl.int32),
                ],
                [BLOCK_M, BLOCK_D_Q],
                Q_base_t.dtype.element_ty,
            ).to(tl.bfloat16)
        else:
            Q_base_t = Q_ptrs[t_q]
            Q_block_ptr = tl.make_block_ptr(
                base=Q_base_t + off_h * stride_qh + t_seq_start * stride_qm,
                shape=(q_block_len.to(tl.int32), BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m_local, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )
            # Use bf16 to rely on tensor core ops
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero").to(
                tl.bfloat16
            )

        AttnScale_base_t = AttnScale_ptrs[t_q]
        scale_ptrs_t = AttnScale_base_t + t_seq_start + offs_m_local
        attn_scale = tl.load(scale_ptrs_t, mask=rows_valid, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)

        q_t_seq_start = t_seq_start.to(tl.int32)
        q_t_seq_end = tl.load(
            q_seq_offsets_tensor + t_q * stride_q_so_t + (off_z_i64 + 1) * stride_q_so_b
        ).to(tl.int32)
        q_t_seq_len = q_t_seq_end - q_t_seq_start

        for t_kv in tl.static_range(NUM_KV):
            cur_mask = tl.load(mask_tensor + t_q * NUM_KV + t_kv)
            if cur_mask != MASK_NULL:
                t_kv_cumsum_start = tl.load(kv_cumsum_ptr + t_kv * stride_kv_cumsum_t)
                t_kv_cumsum_end = tl.load(
                    kv_cumsum_ptr + (t_kv + 1) * stride_kv_cumsum_t
                )
                kv_t_seq_start = tl.load(
                    kv_seq_offsets_tensor
                    + t_kv * stride_kv_so_t
                    + off_z_i64 * stride_kv_so_b
                ).to(tl.int32)
                kv_t_seq_end = tl.load(
                    kv_seq_offsets_tensor
                    + t_kv * stride_kv_so_t
                    + (off_z_i64 + 1) * stride_kv_so_b
                ).to(tl.int32)
                kv_t_seq_len = kv_t_seq_end - kv_t_seq_start
                kv_block_len = t_kv_cumsum_end - t_kv_cumsum_start

                # Compute loop bounds
                kv_loop_start = 0
                kv_loop_end = kv_block_len

                if cur_mask == MASK_CAUSAL:
                    delta = kv_t_seq_len - q_t_seq_len
                    max_valid_k_pos = start_m_local + BLOCK_M + delta
                    kv_loop_end = tl.minimum(
                        kv_block_len,
                        ((max_valid_k_pos + BLOCK_N) // BLOCK_N) * BLOCK_N,
                    )
                elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
                    delta = kv_t_seq_len - q_t_seq_len
                    min_valid_k_pos = start_m_local + delta - max_attn_len
                    min_valid_k_pos = tl.maximum(min_valid_k_pos, 0)
                    kv_loop_start = (min_valid_k_pos // BLOCK_N) * BLOCK_N
                    max_valid_k_pos = start_m_local + BLOCK_M + delta
                    kv_loop_end = tl.minimum(
                        kv_block_len,
                        ((max_valid_k_pos + BLOCK_N) // BLOCK_N) * BLOCK_N,
                    )
                elif cur_mask == MASK_DIAGONAL:
                    kv_loop_start = (start_m_local // BLOCK_N) * BLOCK_N
                    kv_loop_start = tl.maximum(0, kv_loop_start - BLOCK_N)
                    kv_loop_end = tl.minimum(
                        kv_block_len,
                        ((start_m_local + BLOCK_M + BLOCK_N) // BLOCK_N) * BLOCK_N,
                    )

                t_kv_seq_start_i64 = kv_t_seq_start.to(tl.int64)
                kv_seq_len_i32 = kv_t_seq_len.to(tl.int32)
                K_base_t = K_ptrs[t_kv]
                V_base_t = V_ptrs[t_kv]

                # Create TMA descriptors
                if ENABLE_TMA:
                    # pyre-ignore [20]
                    tl.extra.cuda.experimental_device_tensormap_create2d(
                        desc_ptr=device_desc_k,
                        global_address=K_base_t,
                        load_size=[BLOCK_N, BLOCK_D_Q],
                        global_size=[kv_t_seq_end, H * DimQ],
                        element_ty=K_base_t.dtype.element_ty,
                    )
                    # pyre-ignore [20]
                    tl.extra.cuda.experimental_device_tensormap_create2d(
                        desc_ptr=device_desc_v,
                        global_address=V_base_t,
                        load_size=[BLOCK_N, BLOCK_D_V],
                        global_size=[kv_t_seq_end, H * DimV],
                        element_ty=V_base_t.dtype.element_ty,
                    )
                    # pyre-ignore [20]
                    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(
                        device_desc_k
                    )
                    # pyre-ignore [20]
                    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(
                        device_desc_v
                    )
                else:
                    K_base_offset = (
                        K_base_t + off_h * stride_kh + t_kv_seq_start_i64 * stride_kn
                    )
                    V_base_offset = (
                        V_base_t + off_h * stride_vh + t_kv_seq_start_i64 * stride_vn
                    )

                # Compute boundary
                if cur_mask == MASK_CAUSAL:
                    delta = kv_t_seq_len - q_t_seq_len
                    unmasked_end = (start_m_local + delta - BLOCK_N + 1).to(tl.int32)
                    unmasked_end = (unmasked_end // BLOCK_N) * BLOCK_N
                    unmasked_end = tl.maximum(unmasked_end, kv_loop_start.to(tl.int32))
                    unmasked_end = tl.minimum(unmasked_end, kv_loop_end.to(tl.int32))
                elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
                    unmasked_end = kv_loop_start.to(tl.int32)
                elif cur_mask == MASK_DIAGONAL:
                    unmasked_end = kv_loop_start.to(tl.int32)
                else:
                    unmasked_end = kv_loop_end.to(tl.int32)

                # Ensure unmasked_end only covers full blocks
                kv_seq_aligned = (kv_seq_len_i32 // BLOCK_N) * BLOCK_N
                unmasked_end = tl.minimum(unmasked_end, kv_seq_aligned)

                # Create K/V block pointers
                K_block_ptr = None
                V_block_ptr = None
                if not ENABLE_TMA:
                    K_block_ptr = tl.make_block_ptr(
                        # pyre-ignore[61]
                        base=K_base_offset,
                        shape=(BLOCK_D_Q, kv_seq_len_i32),
                        strides=(1, stride_kn),
                        offsets=(0, kv_loop_start.to(tl.int32)),
                        block_shape=(BLOCK_D_Q, BLOCK_N),
                        order=(0, 1),
                    )
                    V_block_ptr = tl.make_block_ptr(
                        # pyre-ignore[61]
                        base=V_base_offset,
                        shape=(kv_seq_len_i32, BLOCK_D_V),
                        strides=(stride_vn, 1),
                        offsets=(kv_loop_start.to(tl.int32), 0),
                        block_shape=(BLOCK_N, BLOCK_D_V),
                        order=(1, 0),
                    )

                # Unmasked region
                for start_n in range(kv_loop_start, unmasked_end, BLOCK_N):
                    if ENABLE_TMA:
                        k = tl._experimental_descriptor_load(
                            device_desc_k,
                            # pyre-ignore[61]
                            [(kv_t_seq_start + start_n).to(tl.int32), offset_kh],
                            [BLOCK_N, BLOCK_D_Q],
                            K_base_t.dtype.element_ty,
                        ).to(tl.bfloat16)
                        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * alpha
                    else:
                        k = tl.load(K_block_ptr).to(tl.bfloat16)
                        qk = tl.dot(q, k).to(tl.float32) * alpha

                    silu = fast_silu(qk) * attn_scale[:, None]

                    if ENABLE_TMA:
                        v = tl._experimental_descriptor_load(
                            device_desc_v,
                            # pyre-ignore[61]
                            [(kv_t_seq_start + start_n).to(tl.int32), offset_vh],
                            [BLOCK_N, BLOCK_D_V],
                            V_base_t.dtype.element_ty,
                        ).to(tl.bfloat16)
                    else:
                        v = tl.load(V_block_ptr).to(tl.bfloat16)

                    silu = silu.to(v.dtype)
                    acc += tl.dot(silu, v)

                    if not ENABLE_TMA:
                        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

                # Masked region
                for start_n in range(unmasked_end, kv_loop_end, BLOCK_N):
                    if ENABLE_TMA:
                        k = tl._experimental_descriptor_load(
                            device_desc_k,
                            # pyre-ignore[61]
                            [(kv_t_seq_start + start_n).to(tl.int32), offset_kh],
                            [BLOCK_N, BLOCK_D_Q],
                            K_base_t.dtype.element_ty,
                        ).to(tl.bfloat16)
                        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * alpha
                    else:
                        k = tl.load(
                            K_block_ptr, boundary_check=(1,), padding_option="zero"
                        ).to(tl.bfloat16)
                        qk = tl.dot(q, k).to(tl.float32) * alpha

                    offs_n = tl.arange(0, BLOCK_N)
                    k_local_pos = start_n + offs_n
                    cols_valid = k_local_pos < kv_block_len

                    if cur_mask == MASK_CAUSAL:
                        delta = kv_t_seq_len - q_t_seq_len
                        causal_mask = (offs_m_local[:, None] + delta) >= k_local_pos[
                            None, :
                        ]
                        valid_mask = (
                            rows_valid[:, None] & cols_valid[None, :] & causal_mask
                        )
                    elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
                        delta = kv_t_seq_len - q_t_seq_len
                        q_shifted = offs_m_local[:, None] + delta
                        causal_mask = q_shifted >= k_local_pos[None, :]
                        local_mask = (q_shifted - k_local_pos[None, :]) < max_attn_len
                        valid_mask = (
                            rows_valid[:, None]
                            & cols_valid[None, :]
                            & causal_mask
                            & local_mask
                        )
                    elif cur_mask == MASK_ALL:
                        valid_mask = rows_valid[:, None] & cols_valid[None, :]
                    elif cur_mask == MASK_DIAGONAL:
                        diag_mask = offs_m_local[:, None] == k_local_pos[None, :]
                        valid_mask = (
                            rows_valid[:, None] & cols_valid[None, :] & diag_mask
                        )
                    else:
                        valid_mask = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)

                    scale = tl.where(valid_mask, attn_scale[:, None], 0.0)
                    silu = fast_silu(qk) * scale

                    if ENABLE_TMA:
                        v = tl._experimental_descriptor_load(
                            device_desc_v,
                            # pyre-ignore[61]
                            [(kv_t_seq_start + start_n).to(tl.int32), offset_vh],
                            [BLOCK_N, BLOCK_D_V],
                            V_base_t.dtype.element_ty,
                        ).to(tl.bfloat16)
                    else:
                        v = tl.load(
                            V_block_ptr, boundary_check=(0,), padding_option="zero"
                        ).to(tl.bfloat16)

                    silu = silu.to(v.dtype)
                    acc += tl.dot(silu, v)

                    if not ENABLE_TMA:
                        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        # Output store
        offs_v_d = tl.arange(0, BLOCK_D_V)
        Out_base_t = Out_ptrs[t_q]
        out_ptrs_t = (
            Out_base_t
            + off_h * stride_oh
            + (t_seq_start + offs_m_local)[:, None] * stride_om
            + offs_v_d[None, :]
        )
        tl.store(out_ptrs_t, acc, mask=rows_valid[:, None])


@triton.jit
def _mha_bwd_compute_list_varargs(  # noqa: C901
    Q_ptrs: "VAR_ARGS_ARRAY_Q",
    K_ptrs: "VAR_ARGS_ARRAY_KV",
    V_ptrs: "VAR_ARGS_ARRAY_KV",
    AttnScale_ptrs: "VAR_ARGS_ARRAY_Q",
    DOut_ptrs: "VAR_ARGS_ARRAY_Q",
    DQ_ptrs: "VAR_ARGS_ARRAY_Q",
    DK_ptrs: "VAR_ARGS_ARRAY_KV",
    DV_ptrs: "VAR_ARGS_ARRAY_KV",
    q_seq_offsets_tensor,
    kv_seq_offsets_tensor,
    q_cumsum_lengths,
    kv_cumsum_lengths,
    max_kv_len_tensor,
    mask_tensor,
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
    stride_q_cumsum_t,
    stride_q_cumsum_z,
    stride_kv_cumsum_t,
    stride_kv_cumsum_z,
    stride_q_so_t,
    stride_q_so_b,
    stride_kv_so_t,
    stride_kv_so_b,
    alpha,
    max_attn_len,
    AUTOTUNE_Z,
    H,
    AUTOTUNE_MAX_SEQ_LEN,
    DimQ,
    DimV,
    NUM_Q: tl.constexpr,
    NUM_KV: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    t_kv = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_hz = tl.program_id(2)

    off_z = off_hz // H
    off_h = off_hz % H
    off_h_i64 = off_h.to(tl.int64)
    off_z_i64 = off_z.to(tl.int64)

    max_kv_len = tl.load(max_kv_len_tensor + t_kv).to(tl.int32)

    start_n = pid_n * BLOCK_N
    if start_n >= max_kv_len:
        return

    # Get KV block boundaries
    kv_t_seq_start = tl.load(
        kv_seq_offsets_tensor + t_kv * stride_kv_so_t + off_z_i64 * stride_kv_so_b
    ).to(tl.int64)
    kv_t_seq_end = tl.load(
        kv_seq_offsets_tensor + t_kv * stride_kv_so_t + (off_z_i64 + 1) * stride_kv_so_b
    ).to(tl.int64)
    kv_t_seq_len = (kv_t_seq_end - kv_t_seq_start).to(tl.int32)

    if start_n >= kv_t_seq_len:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    cols_valid = offs_n < kv_t_seq_len

    # Load K, V for this KV-block
    K_base = K_ptrs[t_kv]
    V_base = V_ptrs[t_kv]

    offs_d_q = tl.arange(0, BLOCK_D_Q)
    offs_d_v = tl.arange(0, BLOCK_D_V)

    # Compute base offsets
    K_base_offset = K_base + off_h_i64 * stride_kh + kv_t_seq_start * stride_kn
    V_base_offset = V_base + off_h_i64 * stride_vh + kv_t_seq_start * stride_vn

    kv_seq_len_i32 = kv_t_seq_len.to(tl.int32)
    start_n_i32 = start_n.to(tl.int32)

    K_block_ptr = tl.make_block_ptr(
        base=K_base_offset,
        shape=(BLOCK_D_Q, kv_seq_len_i32),
        strides=(1, stride_kn),
        offsets=(0, start_n_i32),
        block_shape=(BLOCK_D_Q, BLOCK_N),
        order=(0, 1),
    )
    k_trans = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero").to(
        tl.bfloat16
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_base_offset,
        shape=(kv_seq_len_i32, BLOCK_D_V),
        strides=(stride_vn, 1),
        offsets=(start_n_i32, 0),
        block_shape=(BLOCK_N, BLOCK_D_V),
        order=(1, 0),
    )
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.bfloat16)

    K_block_ptr_nt = tl.make_block_ptr(
        base=K_base_offset,
        shape=(kv_seq_len_i32, BLOCK_D_Q),
        strides=(stride_kn, 1),
        offsets=(start_n_i32, 0),
        block_shape=(BLOCK_N, BLOCK_D_Q),
        order=(1, 0),
    )
    k = tl.load(K_block_ptr_nt, boundary_check=(0,), padding_option="zero").to(
        tl.bfloat16
    )

    v_trans = tl.trans(v)

    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)

    # Iterate over Q-blocks
    for t_q in tl.static_range(NUM_Q):
        cur_mask = tl.load(mask_tensor + t_q * NUM_KV + t_kv)

        if cur_mask != MASK_NULL:
            # Get Q block boundaries
            q_t_seq_start = tl.load(
                q_seq_offsets_tensor + t_q * stride_q_so_t + off_z_i64 * stride_q_so_b
            ).to(tl.int64)
            q_t_seq_end = tl.load(
                q_seq_offsets_tensor
                + t_q * stride_q_so_t
                + (off_z_i64 + 1) * stride_q_so_b
            ).to(tl.int64)
            q_t_seq_len = (q_t_seq_end - q_t_seq_start).to(tl.int32)

            Q_base = Q_ptrs[t_q]
            DOut_base = DOut_ptrs[t_q]
            DQ_base = DQ_ptrs[t_q]
            Scale_base = AttnScale_ptrs[t_q]

            # Compute loop bounds based on mask type
            delta = kv_t_seq_len - q_t_seq_len
            if cur_mask == MASK_CAUSAL:
                # Minimum Q position that can attend to this KV block
                low_m = start_n - delta - BLOCK_M + 1
                low_m = low_m if low_m > 0 else 0
                high_m = q_t_seq_len
                # Boundary where masking is no longer needed
                unmasked_start = start_n + BLOCK_N - 1 - delta
                unmasked_start = ((unmasked_start + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
                unmasked_start = unmasked_start if unmasked_start > low_m else low_m
            elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
                # LOCAL: Q at position m attends to KV at [m + delta - max_attn_len, m + delta]
                low_m = start_n - delta - BLOCK_M + 1
                low_m = low_m if low_m > 0 else 0
                # Q at position m attends to KV at m + delta - max_attn_len
                high_m = start_n + BLOCK_N - delta + max_attn_len
                high_m = high_m if high_m < q_t_seq_len else q_t_seq_len
                unmasked_start = high_m
            elif cur_mask == MASK_DIAGONAL:
                # Diagonal: Q pos m can ONLY attend to KV pos m
                low_m = start_n
                low_m = low_m if low_m > 0 else 0
                # Only Q positions up to start_n + BLOCK_N
                high_m = start_n + BLOCK_N
                high_m = high_m if high_m < q_t_seq_len else q_t_seq_len
                unmasked_start = high_m
            else:
                # MASK_ALL: iterate over all Q positions
                low_m = 0
                high_m = q_t_seq_len
                unmasked_start = low_m

            # Iterate over Q blocks that may have partial masking with the current KV block
            for start_m in range(low_m, unmasked_start, BLOCK_M):
                start_m = tl.multiple_of(start_m, BLOCK_M)
                offs_m = start_m + tl.arange(0, BLOCK_M)
                rows_valid = offs_m < q_t_seq_len

                q_ptrs = (
                    Q_base
                    + off_h_i64 * stride_qh
                    + (q_t_seq_start + offs_m)[:, None] * stride_qm
                    + offs_d_q[None, :]
                )
                q = tl.load(q_ptrs, mask=rows_valid[:, None], other=0.0).to(tl.bfloat16)

                do_ptrs = (
                    DOut_base
                    + off_h_i64 * stride_doh
                    + (q_t_seq_start + offs_m)[:, None] * stride_dom
                    + offs_d_v[None, :]
                )
                do = tl.load(do_ptrs, mask=rows_valid[:, None], other=0.0).to(
                    tl.bfloat16
                )

                scale_ptrs = Scale_base + q_t_seq_start + offs_m
                attn_scale = tl.load(scale_ptrs, mask=rows_valid, other=0.0).to(
                    tl.float32
                )

                # Recompute QK = Q @ K^T [BLOCK_M, BLOCK_N]
                qk = tl.dot(q, k_trans).to(tl.float32) * alpha

                # Compute attention mask based on mask type
                if cur_mask == MASK_CAUSAL:
                    q_shifted = offs_m[:, None] + delta
                    causal_mask = q_shifted >= offs_n[None, :]
                    valid_mask = rows_valid[:, None] & cols_valid[None, :] & causal_mask
                elif HAS_MAX_ATTN_LEN and cur_mask == MASK_LOCAL:
                    q_shifted = offs_m[:, None] + delta
                    causal_mask = q_shifted >= offs_n[None, :]
                    local_mask = (q_shifted - offs_n[None, :]) < max_attn_len
                    valid_mask = (
                        rows_valid[:, None]
                        & cols_valid[None, :]
                        & causal_mask
                        & local_mask
                    )
                elif cur_mask == MASK_DIAGONAL:
                    diag_mask = offs_m[:, None] == offs_n[None, :]
                    valid_mask = rows_valid[:, None] & cols_valid[None, :] & diag_mask
                else:
                    valid_mask = rows_valid[:, None] & cols_valid[None, :]

                # Recompute forward: silu_scaled = silu(qk) * scale * mask
                sig = fast_sigmoid(qk)
                silu_out = qk * sig
                scale_broadcast = tl.where(valid_mask, attn_scale[:, None], 0.0)
                silu_scaled = silu_out * scale_broadcast

                silu_scaled_bf16 = silu_scaled.to(tl.bfloat16)
                dv += tl.dot(tl.trans(silu_scaled_bf16), do)

                dsilu = tl.dot(do, v_trans)
                dqk = dsilu * sig * (1.0 + qk - qk * sig) * scale_broadcast
                dqk = tl.where(valid_mask, dqk, 0.0)

                dqk_bf16 = dqk.to(tl.bfloat16)
                dk += tl.dot(tl.trans(dqk_bf16), q)

                dq_contrib = tl.dot(dqk_bf16, k)

                dq_ptrs = (
                    DQ_base
                    + off_h_i64 * stride_dqh
                    + (q_t_seq_start + offs_m)[:, None] * stride_dqm
                    + offs_d_q[None, :]
                )
                dq_contrib_scaled = (dq_contrib * alpha).to(k.dtype)
                tl.atomic_add(
                    dq_ptrs, dq_contrib_scaled, mask=rows_valid[:, None], sem="relaxed"
                )

            # Unmasked region: All Q positions can attend to all KV positions
            # Skip mask computation since all elements are valid
            for start_m in range(unmasked_start, high_m, BLOCK_M):
                start_m = tl.multiple_of(start_m, BLOCK_M)
                offs_m = start_m + tl.arange(0, BLOCK_M)
                rows_valid = offs_m < q_t_seq_len

                # Load Q block
                q_ptrs = (
                    Q_base
                    + off_h_i64 * stride_qh
                    + (q_t_seq_start + offs_m)[:, None] * stride_qm
                    + offs_d_q[None, :]
                )
                q = tl.load(q_ptrs, mask=rows_valid[:, None], other=0.0).to(tl.bfloat16)

                do_ptrs = (
                    DOut_base
                    + off_h_i64 * stride_doh
                    + (q_t_seq_start + offs_m)[:, None] * stride_dom
                    + offs_d_v[None, :]
                )
                do = tl.load(do_ptrs, mask=rows_valid[:, None], other=0.0).to(
                    tl.bfloat16
                )

                scale_ptrs = Scale_base + q_t_seq_start + offs_m
                attn_scale = tl.load(scale_ptrs, mask=rows_valid, other=0.0).to(
                    tl.float32
                )

                # Recompute QK = Q @ K^T [BLOCK_M, BLOCK_N]
                qk = tl.dot(q, k_trans).to(tl.float32) * alpha

                valid_mask = rows_valid[:, None] & cols_valid[None, :]

                # Recompute forward: silu_scaled = silu(qk) * scale
                sig = fast_sigmoid(qk)
                silu_out = qk * sig
                scale_broadcast = tl.where(valid_mask, attn_scale[:, None], 0.0)
                silu_scaled = silu_out * scale_broadcast

                silu_scaled_bf16 = silu_scaled.to(tl.bfloat16)
                dv += tl.dot(tl.trans(silu_scaled_bf16), do)

                dsilu = tl.dot(do, v_trans)
                dqk = dsilu * sig * (1.0 + qk - qk * sig) * scale_broadcast
                dqk = tl.where(valid_mask, dqk, 0.0)

                dqk_bf16 = dqk.to(tl.bfloat16)
                dk += tl.dot(tl.trans(dqk_bf16), q)

                dq_contrib = tl.dot(dqk_bf16, k)

                dq_ptrs = (
                    DQ_base
                    + off_h_i64 * stride_dqh
                    + (q_t_seq_start + offs_m)[:, None] * stride_dqm
                    + offs_d_q[None, :]
                )
                dq_contrib_scaled = (dq_contrib * alpha).to(k.dtype)
                tl.atomic_add(
                    dq_ptrs, dq_contrib_scaled, mask=rows_valid[:, None], sem="relaxed"
                )

    dk = dk * alpha
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)

    DK_base = DK_ptrs[t_kv]
    DV_base = DV_ptrs[t_kv]

    dk_ptrs = (
        DK_base
        + off_h_i64 * stride_dkh
        + (kv_t_seq_start + offs_n)[:, None] * stride_dkn
        + offs_d_q[None, :]
    )
    dv_ptrs = (
        DV_base
        + off_h_i64 * stride_dvh
        + (kv_t_seq_start + offs_n)[:, None] * stride_dvn
        + offs_d_v[None, :]
    )

    tl.store(dk_ptrs, dk, mask=cols_valid[:, None])
    tl.store(dv_ptrs, dv, mask=cols_valid[:, None])


def triton_mha_fwd(
    alpha: float,
    q_list: List[torch.Tensor],
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
    q_seq_offsets_list: List[torch.Tensor],
    mask_matrix: List[List[MaskType]],
    attn_scale_list: List[torch.Tensor],
    kv_seq_offsets_list: Optional[List[torch.Tensor]] = None,
    max_attn_len: int = 0,
) -> List[torch.Tensor]:
    """Forward pass for blocked MHA."""
    if kv_seq_offsets_list is None:
        kv_seq_offsets_list = q_seq_offsets_list

    num_q_tensors = len(q_list)
    num_kv_tensors = len(k_list)
    device = q_list[0].device
    B = q_seq_offsets_list[0].numel() - 1
    H = q_list[0].shape[1]
    DimQ = q_list[0].shape[2]
    DimV = v_list[0].shape[2]
    out_list = [
        torch.empty((q.shape[0], q.shape[1], DimV), dtype=q.dtype, device=device)
        for q in q_list
    ]
    is_single_block = num_q_tensors == 1 and num_kv_tensors == 1

    if is_single_block:
        # Fast path: skip computations not needed by single block kernel
        q_seq_offsets_tensor = q_seq_offsets_list[0].unsqueeze(0)
        kv_seq_offsets_tensor = kv_seq_offsets_list[0].unsqueeze(0)

        q_lengths = q_seq_offsets_list[0][1:] - q_seq_offsets_list[0][:-1]
        global_max_q_len = int(q_lengths.max().item())
        if global_max_q_len == 0:
            return out_list

        q_cumsum_lengths = q_seq_offsets_tensor[:1, :1]  # dummy
        kv_cumsum_lengths = kv_seq_offsets_tensor[:1, :1]  # dummy
        max_q_len_tensor = torch.empty(1, dtype=torch.int32, device=device)  # dummy
        mask_tensor = torch.tensor(
            [mask_matrix[0][0].value], dtype=torch.int32, device=device
        )
        grid = lambda meta: (  # noqa E731
            triton.cdiv(global_max_q_len, meta["BLOCK_M"]),
            B * H,
        )
    else:
        q_seq_offsets_tensor = torch.stack(q_seq_offsets_list, dim=0)
        kv_seq_offsets_tensor = torch.stack(kv_seq_offsets_list, dim=0)
        q_lengths_per_batch = torch.zeros(
            (num_q_tensors, B), dtype=torch.int64, device=device
        )
        for t, seq_offsets in enumerate(q_seq_offsets_list):
            q_lengths_per_batch[t] = seq_offsets[1:] - seq_offsets[:-1]
        q_cumsum_lengths = torch.zeros(
            (num_q_tensors + 1, B), dtype=torch.int64, device=device
        )
        q_cumsum_lengths[1:] = torch.cumsum(q_lengths_per_batch, dim=0)
        kv_lengths_per_batch = torch.zeros(
            (num_kv_tensors, B), dtype=torch.int64, device=device
        )
        for t, seq_offsets in enumerate(kv_seq_offsets_list):
            kv_lengths_per_batch[t] = seq_offsets[1:] - seq_offsets[:-1]
        kv_cumsum_lengths = torch.zeros(
            (num_kv_tensors + 1, B), dtype=torch.int64, device=device
        )
        kv_cumsum_lengths[1:] = torch.cumsum(kv_lengths_per_batch, dim=0)

        max_q_len_tensor = q_lengths_per_batch.max(dim=1).values.to(torch.int32)
        global_max_q_len = int(max_q_len_tensor.max().item())

        if global_max_q_len == 0:
            return out_list

        mask_tensor = _create_mask_tensor(mask_matrix, device)

        grid = lambda meta: (  # noqa E731
            num_q_tensors,
            triton.cdiv(global_max_q_len, meta["BLOCK_M"]),
            B * H,
        )

    TMA_DESC_SIZE = 128
    enable_tma = (
        not torch.version.hip and torch.cuda.get_device_capability(device)[0] >= 9
    )

    workspace = None
    if enable_tma:
        MIN_BLOCK_M = 16
        num_m_blocks = triton.cdiv(global_max_q_len, MIN_BLOCK_M)
        if is_single_block:
            # 2D grid: (num_m_blocks, B*H)
            total_programs = num_m_blocks * B * H
        else:
            # 3D grid: (num_q_tensors, num_m_blocks, B*H)
            total_programs = num_q_tensors * num_m_blocks * B * H
        workspace = torch.empty(
            3 * TMA_DESC_SIZE * total_programs,
            dtype=torch.uint8,
            device=device,
        )

    unrolled_kernel = unroll_varargs(
        _mha_fwd_compute_list_varargs,
        N={"Q": num_q_tensors, "KV": num_kv_tensors},
        mode=VarargMode.CONDITIONAL,
    )
    autotuned_kernel = _get_autotune_kernel_mha(unrolled_kernel)
    autotuned_kernel[grid](  # pyre-ignore [16]
        *q_list,
        *k_list,
        *v_list,
        *out_list,
        *attn_scale_list,
        q_seq_offsets_tensor,
        kv_seq_offsets_tensor,
        q_cumsum_lengths,
        kv_cumsum_lengths,
        max_q_len_tensor,
        mask_tensor,
        q_list[0].stride(0),
        q_list[0].stride(1),
        k_list[0].stride(0),
        k_list[0].stride(1),
        v_list[0].stride(0),
        v_list[0].stride(1),
        out_list[0].stride(0),
        out_list[0].stride(1),
        q_cumsum_lengths.stride(0),
        q_cumsum_lengths.stride(1),
        kv_cumsum_lengths.stride(0),
        kv_cumsum_lengths.stride(1),
        q_seq_offsets_tensor.stride(0),
        q_seq_offsets_tensor.stride(1),
        kv_seq_offsets_tensor.stride(0),
        kv_seq_offsets_tensor.stride(1),
        alpha,
        max_attn_len,
        workspace,
        prev_power_of_2(B),
        H,
        autotune_max_seq_len(global_max_q_len),
        DimQ,
        DimV,
        NUM_Q=num_q_tensors,
        NUM_KV=num_kv_tensors,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_MAX_ATTN_LEN=max_attn_len > 0,
        SINGLE_BLOCK=is_single_block,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
    )

    return out_list


def triton_mha_bwd(
    dout_list: List[torch.Tensor],
    q_list: List[torch.Tensor],
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
    q_seq_offsets_list: List[torch.Tensor],
    mask_matrix: List[List[MaskType]],
    attn_scale_list: List[torch.Tensor],
    alpha: float,
    kv_seq_offsets_list: Optional[List[torch.Tensor]] = None,
    max_attn_len: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Backward pass for blocked MHA."""
    if kv_seq_offsets_list is None:
        kv_seq_offsets_list = q_seq_offsets_list

    num_q_tensors = len(q_list)
    num_kv_tensors = len(k_list)
    device = q_list[0].device
    B = q_seq_offsets_list[0].numel() - 1
    H = q_list[0].shape[1]
    DimQ = q_list[0].shape[2]
    DimV = v_list[0].shape[2]

    dq_list = [torch.zeros_like(q) for q in q_list]
    dk_list = [torch.zeros_like(k) for k in k_list]
    dv_list = [torch.zeros_like(v) for v in v_list]

    q_seq_offsets_tensor = torch.stack(q_seq_offsets_list, dim=0)
    kv_seq_offsets_tensor = torch.stack(kv_seq_offsets_list, dim=0)

    q_lengths_per_batch = torch.zeros(
        (num_q_tensors, B), dtype=torch.int64, device=device
    )
    for t, seq_offsets in enumerate(q_seq_offsets_list):
        q_lengths_per_batch[t] = seq_offsets[1:] - seq_offsets[:-1]
    q_cumsum_lengths = torch.zeros(
        (num_q_tensors + 1, B), dtype=torch.int64, device=device
    )
    q_cumsum_lengths[1:] = torch.cumsum(q_lengths_per_batch, dim=0)

    kv_lengths_per_batch = torch.zeros(
        (num_kv_tensors, B), dtype=torch.int64, device=device
    )
    for t, seq_offsets in enumerate(kv_seq_offsets_list):
        kv_lengths_per_batch[t] = seq_offsets[1:] - seq_offsets[:-1]
    kv_cumsum_lengths = torch.zeros(
        (num_kv_tensors + 1, B), dtype=torch.int64, device=device
    )
    kv_cumsum_lengths[1:] = torch.cumsum(kv_lengths_per_batch, dim=0)

    max_kv_len_tensor = kv_lengths_per_batch.max(dim=1).values.to(torch.int32)
    global_max_kv_len = int(max_kv_len_tensor.max().item())
    global_max_q_len = int(q_lengths_per_batch.max().item())

    if global_max_kv_len == 0:
        return dq_list, dk_list, dv_list

    mask_tensor = _create_mask_tensor(mask_matrix, device)

    grid = lambda meta: (  # noqa E731
        num_kv_tensors,
        triton.cdiv(global_max_kv_len, meta["BLOCK_N"]),
        B * H,
    )

    unrolled_bwd_kernel = unroll_varargs(
        _mha_bwd_compute_list_varargs,
        N={"Q": num_q_tensors, "KV": num_kv_tensors},
        mode=VarargMode.CONDITIONAL,
    )
    autotuned_bwd_kernel = _get_autotune_kernel_mha_bwd(unrolled_bwd_kernel)

    autotuned_bwd_kernel[grid](  # pyre-ignore [16]
        *q_list,
        *k_list,
        *v_list,
        *attn_scale_list,
        *dout_list,
        *dq_list,
        *dk_list,
        *dv_list,
        q_seq_offsets_tensor,
        kv_seq_offsets_tensor,
        q_cumsum_lengths,
        kv_cumsum_lengths,
        max_kv_len_tensor,
        mask_tensor,
        q_list[0].stride(0),
        q_list[0].stride(1),
        k_list[0].stride(0),
        k_list[0].stride(1),
        v_list[0].stride(0),
        v_list[0].stride(1),
        dout_list[0].stride(0),
        dout_list[0].stride(1),
        dq_list[0].stride(0),
        dq_list[0].stride(1),
        dk_list[0].stride(0),
        dk_list[0].stride(1),
        dv_list[0].stride(0),
        dv_list[0].stride(1),
        q_cumsum_lengths.stride(0),
        q_cumsum_lengths.stride(1),
        kv_cumsum_lengths.stride(0),
        kv_cumsum_lengths.stride(1),
        q_seq_offsets_tensor.stride(0),
        q_seq_offsets_tensor.stride(1),
        kv_seq_offsets_tensor.stride(0),
        kv_seq_offsets_tensor.stride(1),
        alpha,
        max_attn_len,
        prev_power_of_2(B),
        H,
        autotune_max_seq_len(global_max_q_len),
        DimQ,
        DimV,
        NUM_Q=num_q_tensors,
        NUM_KV=num_kv_tensors,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_MAX_ATTN_LEN=max_attn_len > 0,
    )

    return dq_list, dk_list, dv_list


class _BlockedMHAFunction(torch.autograd.Function):
    """Autograd function for blocked MHA."""

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        alpha: float,
        max_attn_len: int,
        mask_matrix_tuple: Tuple[int, ...],
        num_q_tensors: int,
        num_kv_tensors: int,
        *tensors,
    ) -> Tuple[torch.Tensor, ...]:
        q_list = list(tensors[:num_q_tensors])
        k_list = list(tensors[num_q_tensors : num_q_tensors + num_kv_tensors])
        v_list = list(
            tensors[num_q_tensors + num_kv_tensors : num_q_tensors + 2 * num_kv_tensors]
        )
        attn_scale_list = list(
            tensors[
                num_q_tensors + 2 * num_kv_tensors : 2 * num_q_tensors
                + 2 * num_kv_tensors
            ]
        )
        q_seq_offsets_list = list(
            tensors[
                2 * num_q_tensors + 2 * num_kv_tensors : 3 * num_q_tensors
                + 2 * num_kv_tensors
            ]
        )
        kv_seq_offsets_list = list(tensors[3 * num_q_tensors + 2 * num_kv_tensors :])

        mask_matrix = [
            [MaskType(m) for m in row]
            for row in [
                mask_matrix_tuple[i * num_kv_tensors : (i + 1) * num_kv_tensors]
                for i in range(num_q_tensors)
            ]
        ]

        out_list = triton_mha_fwd(
            alpha=alpha,
            q_list=q_list,
            k_list=k_list,
            v_list=v_list,
            q_seq_offsets_list=q_seq_offsets_list,
            mask_matrix=mask_matrix,
            attn_scale_list=attn_scale_list,
            kv_seq_offsets_list=kv_seq_offsets_list,
            max_attn_len=max_attn_len,
        )

        # Save for backward
        ctx.save_for_backward(
            *q_list,
            *k_list,
            *v_list,
            *attn_scale_list,
            *q_seq_offsets_list,
            *kv_seq_offsets_list,
        )
        ctx.alpha = alpha
        ctx.max_attn_len = max_attn_len
        ctx.mask_matrix_tuple = mask_matrix_tuple
        ctx.num_q_tensors = num_q_tensors
        ctx.num_kv_tensors = num_kv_tensors

        return tuple(out_list)

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        num_q = ctx.num_q_tensors
        num_kv = ctx.num_kv_tensors

        # Unpack saved tensors
        saved = ctx.saved_tensors
        q_list = list(saved[:num_q])
        k_list = list(saved[num_q : num_q + num_kv])
        v_list = list(saved[num_q + num_kv : num_q + 2 * num_kv])
        attn_scale_list = list(saved[num_q + 2 * num_kv : 2 * num_q + 2 * num_kv])
        q_seq_offsets_list = list(
            saved[2 * num_q + 2 * num_kv : 3 * num_q + 2 * num_kv]
        )
        kv_seq_offsets_list = list(saved[3 * num_q + 2 * num_kv :])

        mask_matrix = [
            [MaskType(m) for m in row]
            for row in [
                ctx.mask_matrix_tuple[i * num_kv : (i + 1) * num_kv]
                for i in range(num_q)
            ]
        ]

        dout_list = list(grad_outputs)

        dq_list, dk_list, dv_list = triton_mha_bwd(
            dout_list=dout_list,
            q_list=q_list,
            k_list=k_list,
            v_list=v_list,
            q_seq_offsets_list=q_seq_offsets_list,
            mask_matrix=mask_matrix,
            attn_scale_list=attn_scale_list,
            alpha=ctx.alpha,
            kv_seq_offsets_list=kv_seq_offsets_list,
            max_attn_len=ctx.max_attn_len,
        )

        # pyre-ignore[60]
        return (
            None,  # alpha
            None,  # max_attn_len
            None,  # mask_matrix_tuple
            None,  # num_q_tensors
            None,  # num_kv_tensors
            *dq_list,  # dQ
            *dk_list,  # dK
            *dv_list,  # dV
            *[None] * num_q,  # attn_scale (no grad)
            *[None] * num_q,  # q_seq_offsets (no grad)
            *[None] * num_kv,  # kv_seq_offsets (no grad)
        )


@torch.fx.wrap
def triton_mha(
    alpha: float,
    q_list: List[torch.Tensor],
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
    q_seq_offsets_list: List[torch.Tensor],
    mask_matrix: List[List[MaskType]],
    attn_scale_list: List[torch.Tensor],
    kv_seq_offsets_list: Optional[List[torch.Tensor]] = None,
    max_attn_len: int = 0,
) -> List[torch.Tensor]:
    if kv_seq_offsets_list is None:
        kv_seq_offsets_list = q_seq_offsets_list

    requires_grad = any(t.requires_grad for t in q_list + k_list + v_list)

    if requires_grad:
        num_q_tensors = len(q_list)
        num_kv_tensors = len(k_list)

        # Convert mask_matrix to a flat tuple for autograd
        mask_matrix_tuple = tuple(m.value for row in mask_matrix for m in row)

        # pyre-ignore[60]
        tensors = (
            *q_list,
            *k_list,
            *v_list,
            *attn_scale_list,
            *q_seq_offsets_list,
            *kv_seq_offsets_list,
        )

        out_tuple = _BlockedMHAFunction.apply(
            alpha,
            max_attn_len,
            mask_matrix_tuple,
            num_q_tensors,
            num_kv_tensors,
            *tensors,
        )
        return list(out_tuple)
    else:
        return triton_mha_fwd(
            alpha=alpha,
            q_list=q_list,
            k_list=k_list,
            v_list=v_list,
            q_seq_offsets_list=q_seq_offsets_list,
            mask_matrix=mask_matrix,
            attn_scale_list=attn_scale_list,
            kv_seq_offsets_list=kv_seq_offsets_list,
            max_attn_len=max_attn_len,
        )
