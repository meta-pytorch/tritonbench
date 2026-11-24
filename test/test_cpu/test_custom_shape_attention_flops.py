# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tritonbench.operators.custom_shape_attentions.generate_inputs import (
    ATTENTION_SHAPES,
    AttentionShape,
)
from tritonbench.operators.custom_shape_attentions.operator import Operator
from tritonbench.utils.triton_op import Mode


def test_paged_decode_flops():
    shape = next(
        shape for shape in ATTENTION_SHAPES if shape.name == "example_paged_attn"
    )
    op = Operator.__new__(Operator)
    op.current_shape = shape
    q = torch.empty((shape.batch, shape.n_heads, shape.d_head), device="meta")
    inputs = (q, None, None, None, 1, shape.seq_len_kv, None, None)
    # One query per request can attend to every key, including the current token.
    assert op.flops("oss_fa4", inputs, None) == 4 * 32 * 128 * 4 * 4096


@pytest.mark.parametrize(
    "q_len,kv_lengths,causal,window",
    [
        (5, [3, 8], True, (-1, -1)),
        (5, [3, 8], False, (-1, -1)),
        (5, [3, 8], False, (2, -1)),
        (5, [3, 8], False, (-1, 1)),
        (5, [3, 8], True, (2, 3)),
        (8, [1, 3], False, (1, 1)),
    ],
)
def test_masked_flops(q_len, kv_lengths, causal, window):
    heads, dim = 4, 16
    op = Operator.__new__(Operator)
    op.current_shape = AttentionShape(
        "test",
        len(kv_lengths),
        heads,
        2,
        q_len,
        kv_lengths,
        dim,
        causal=causal,
        window_size=window,
        paged_attention=True,
    )
    q = torch.empty((q_len * len(kv_lengths), heads, dim), device="meta")
    inputs = (q, None, None, None, q_len, max(kv_lengths), None, None)

    visible = 0
    left, right = window
    for k_len in kv_lengths:
        row = torch.arange(q_len)[:, None] + k_len - q_len
        col = torch.arange(k_len)[None, :]
        mask = torch.ones((q_len, k_len), dtype=torch.bool)
        if left >= 0:
            mask &= col >= row - left
        if right >= 0:
            mask &= col <= row + right
        if causal:
            mask &= col <= row
        visible += mask.sum().item()

    assert op.flops("oss_fa4", inputs, None) == 4 * heads * dim * visible


@pytest.mark.parametrize(
    "mode,scale", [(Mode.FWD, 1), (Mode.BWD, 2.5), (Mode.FWD_BWD, 3.5)]
)
def test_dense_causal_chunk_flops(mode, scale):
    op = Operator.__new__(Operator)
    op.current_shape = AttentionShape("chunk", 2, 4, 2, 3, 5, 16)
    op.mode = mode
    q = torch.empty((2, 4, 3, 16), device="meta")
    k = torch.empty((2, 2, 5, 16), device="meta")
    # The three queries can see 3, 4, and 5 keys respectively.
    assert op.flops("oss_fa4", (q, k, k), None) == scale * 4 * 4 * 16 * 2 * 12
