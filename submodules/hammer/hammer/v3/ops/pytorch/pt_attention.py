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

#!/usr/bin/env python3

# pyre-strict

from enum import Enum, unique
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


def _pad_qkv_cross(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    seq_offsets_kv: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad Q, K, V tensors with different max sequence lengths."""
    total_seq_len_q, H, D = q.shape
    total_seq_len_kv = k.shape[0]
    batch_size = seq_offsets_q.shape[0] - 1
    V = v.shape[2]

    padded_q = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=q.reshape(total_seq_len_q, H * D),
            offsets=[seq_offsets_q],
            max_lengths=[max_seq_len_q],
            padding_value=0.0,
        )
        .view(batch_size, max_seq_len_q, H, D)
        .transpose(1, 2)
    )  # [B, H, N_q, D]

    padded_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(total_seq_len_kv, H * D),
            offsets=[seq_offsets_kv],
            max_lengths=[max_seq_len_kv],
            padding_value=0.0,
        )
        .view(batch_size, max_seq_len_kv, H, D)
        .transpose(1, 2)
    )  # [B, H, N_kv, D]

    padded_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(total_seq_len_kv, H * V),
            offsets=[seq_offsets_kv],
            max_lengths=[max_seq_len_kv],
            padding_value=0.0,
        )
        .view(batch_size, max_seq_len_kv, H, V)
        .transpose(1, 2)
    )  # [B, H, N_kv, V]

    return padded_q, padded_k, padded_v


@unique
class MaskType(Enum):
    CAUSAL = 0
    ALL = 1
    DIAGONAL = 2
    NULL = 3
    LOCAL = 4


@torch.fx.wrap
def _valid_block_attn_mask(
    device: torch.device,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    seq_lengths_q: torch.Tensor,
    seq_lengths_kv: torch.Tensor,
    mask_type: MaskType,
    max_attn_len: int = 0,
) -> torch.Tensor:
    col_ids = torch.arange(0, max_seq_len_q, device=device).view(1, max_seq_len_q, 1)
    row_ids = torch.arange(0, max_seq_len_kv, device=device).view(1, 1, max_seq_len_kv)
    in_boundary_valid_attn_mask = torch.logical_and(
        row_ids < seq_lengths_kv.view(-1, 1, 1), col_ids < seq_lengths_q.view(-1, 1, 1)
    )
    if mask_type == MaskType.CAUSAL:
        delta_col_ids = seq_lengths_kv - seq_lengths_q
        shifted_col_ids = col_ids + delta_col_ids.view(-1, 1, 1)
        causal_mask = shifted_col_ids >= row_ids
        return torch.logical_and(in_boundary_valid_attn_mask, causal_mask)
    elif mask_type == MaskType.DIAGONAL:
        return torch.logical_and(in_boundary_valid_attn_mask, col_ids == row_ids)
    elif mask_type == MaskType.LOCAL:
        delta_col_ids = seq_lengths_kv - seq_lengths_q
        shifted_col_ids = col_ids + delta_col_ids.view(-1, 1, 1)
        causal_mask = shifted_col_ids >= row_ids
        local_mask = (shifted_col_ids - row_ids) < max_attn_len
        return torch.logical_and(
            in_boundary_valid_attn_mask, torch.logical_and(causal_mask, local_mask)
        )
    else:
        assert mask_type == MaskType.ALL
        return in_boundary_valid_attn_mask


def pytorch_mha_block(
    alpha: float,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: Optional[torch.Tensor],
    mask_type: MaskType,
    attn_scale: torch.Tensor,
    max_attn_len: int = 0,
) -> torch.Tensor:
    total_seq_len_q, H, _ = q.shape
    V = v.shape[2]
    if mask_type == MaskType.NULL:
        return torch.zeros((total_seq_len_q, H, V), dtype=q.dtype, device=q.device)
    if kv_offsets is None:
        kv_offsets = q_offsets
    q, k, v = _pad_qkv_cross(
        q=q,
        k=k,
        v=v,
        seq_offsets_q=q_offsets,
        seq_offsets_kv=kv_offsets,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_kv=max_seq_len_kv,
    )
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    if attn_scale.ndim > 0:
        attn_scale = (
            torch.ops.fbgemm.jagged_to_padded_dense(
                values=attn_scale.unsqueeze(-1),
                offsets=[q_offsets],
                max_lengths=[max_seq_len_q],
                padding_value=0.0,
            )
            .unsqueeze(1)
            .to(q.dtype)
        )
    valid_attn_mask = _valid_block_attn_mask(
        device=q.device,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_kv=max_seq_len_kv,
        seq_lengths_q=q_offsets[1:] - q_offsets[:-1],
        seq_lengths_kv=kv_offsets[1:] - kv_offsets[:-1],
        mask_type=mask_type,
        max_attn_len=max_attn_len,
    ).unsqueeze(1)
    qk_attn_silu = F.silu(qk_attn) * attn_scale
    qk_attn_silu = torch.where(valid_attn_mask, qk_attn_silu, 0.0)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn_silu, v)  # [B, H, N, V]
    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, N, H, V]->[B, N, H * V]
        [q_offsets],
        total_seq_len_q,
    )[0].view(total_seq_len_q, H, V)


@torch.fx.wrap
def pytorch_mha(
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
    assert len(q_list) == len(q_seq_offsets_list) == len(attn_scale_list)
    assert len(kv_seq_offsets_list) == len(k_list) == len(v_list)
    assert len(q_list) > 0 and len(k_list) > 0, "At least one tensor required"
    out_block_list = []
    for i, q_block in enumerate(q_list):
        out_block_acc = None
        offsets_q = q_seq_offsets_list[i]
        lengths_q = offsets_q[1:] - offsets_q[:-1]
        attn_scale = attn_scale_list[i]
        for j in range(len(k_list)):
            k_block = k_list[j]
            v_block = v_list[j]
            mask_type = mask_matrix[i][j]
            offsets_kv = kv_seq_offsets_list[j]
            lengths_kv = offsets_kv[1:] - offsets_kv[:-1]
            out_block = pytorch_mha_block(
                alpha=alpha,
                max_seq_len_q=int(lengths_q.max().item()),
                max_seq_len_kv=int(lengths_kv.max().item()),
                q=q_block,
                k=k_block,
                v=v_block,
                q_offsets=offsets_q,
                kv_offsets=offsets_kv,
                mask_type=mask_type,
                attn_scale=attn_scale,
                max_attn_len=max_attn_len,
            ).to(torch.float32)
            if out_block_acc is None:
                out_block_acc = out_block
            else:
                out_block_acc += out_block
        assert out_block_acc is not None
        out_block_list.append(out_block_acc.to(q_list[0].dtype))
    return out_block_list
