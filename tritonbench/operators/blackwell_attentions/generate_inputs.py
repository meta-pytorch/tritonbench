# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Generator, Tuple

import torch


def _generated_qkv_inputs(
    shape, dtype, device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    requires_grad = True

    BATCH, H, N_HEADS_KV, N_CTX, N_CTX_KV, D_HEAD = shape

    q = torch.randn(
        (BATCH, H, N_CTX, D_HEAD),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        (BATCH, N_HEADS_KV, N_CTX_KV, D_HEAD),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    v = torch.randn(
        (BATCH, N_HEADS_KV, N_CTX_KV, D_HEAD),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return (q, k, v)


# Config input tensors.
# We can add more shapes, such as training, prefill, decoding, etc.


def customized_inputs(shape, num_inputs, dtype, device) -> Generator:
    BATCH, H, N_HEADS_KV, SEQ_LEN, SEQ_LEN_KV, D_HEAD = shape

    SEQ_LEN_LOG2 = 7

    if SEQ_LEN is not None:
        SEQ_LEN_KV = SEQ_LEN if SEQ_LEN_KV is None else SEQ_LEN_KV
        if num_inputs is None:
            yield _generated_qkv_inputs(
                (BATCH, H, N_HEADS_KV, SEQ_LEN, SEQ_LEN_KV, D_HEAD),
                dtype=dtype,
                device=device,
            )
        else:
            for _i in range(num_inputs):
                yield _generated_qkv_inputs(
                    (BATCH, H, N_HEADS_KV, SEQ_LEN, SEQ_LEN, D_HEAD),
                    dtype=dtype,
                    device=device,
                )
                SEQ_LEN *= 2
        return
    for i in range(SEQ_LEN_LOG2, 15):
        SEQ_LEN = 2**i
        yield _generated_qkv_inputs(
            (BATCH, H, H, SEQ_LEN, SEQ_LEN, D_HEAD), dtype=dtype, device=device
        )


def fa3_paper_inputs(dtype, device) -> Generator:
    D_HEAD = 128
    H = 2048 // D_HEAD
    for BATCH in [32, 16, 8, 4, 2, 1]:
        N_CTX = 16384 // BATCH
        yield _generated_qkv_inputs(
            shape=(BATCH, H, N_CTX, N_CTX, D_HEAD), dtype=dtype, device=device
        )
