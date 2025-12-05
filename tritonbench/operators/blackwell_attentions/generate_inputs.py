# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Generator, Tuple

import torch


def get_bytes(x):
    return x.numel() * x.element_size()


def _generated_qkv_inputs(
    shape, dtype, device, gen_cache_size_inputs
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
    inputs = [q, k, v]
    if gen_cache_size_inputs:
        q_bytes = get_bytes(q)
        k_bytes = get_bytes(k)
        v_bytes = get_bytes(v)
        total_bytes = q_bytes + k_bytes + v_bytes
        # Fix to 128 MB for now
        min_bytes = 128 * 1024 * 1024
        num_inputs = math.ceil(min_bytes / total_bytes)
        for _ in range(num_inputs - 1):
            inputs.append(q.clone().detach())
            inputs.append(k.clone().detach())
            inputs.append(v.clone().detach())
    assert len(inputs) % 3 == 0
    return tuple(inputs)


# Config input tensors.
# We can add more shapes, such as training, prefill, decoding, etc.


def customized_inputs(
    shape, num_inputs, dtype, device, gen_cache_size_inputs
) -> Generator:
    BATCH, H, N_HEADS_KV, SEQ_LEN, SEQ_LEN_KV, D_HEAD = shape

    SEQ_LEN_LOG2 = 7

    if SEQ_LEN is not None:
        SEQ_LEN_KV = SEQ_LEN if SEQ_LEN_KV is None else SEQ_LEN_KV
        if num_inputs is None:
            yield _generated_qkv_inputs(
                (BATCH, H, N_HEADS_KV, SEQ_LEN, SEQ_LEN_KV, D_HEAD),
                dtype=dtype,
                device=device,
                gen_cache_size_inputs=gen_cache_size_inputs,
            )
        else:
            for _i in range(num_inputs):
                yield _generated_qkv_inputs(
                    (BATCH, H, N_HEADS_KV, SEQ_LEN, SEQ_LEN, D_HEAD),
                    dtype=dtype,
                    device=device,
                    gen_cache_size_inputs=gen_cache_size_inputs,
                )
                SEQ_LEN *= 2
        return
    for i in range(SEQ_LEN_LOG2, 15):
        SEQ_LEN = 2**i
        yield _generated_qkv_inputs(
            (BATCH, H, H, SEQ_LEN, SEQ_LEN, D_HEAD),
            dtype=dtype,
            device=device,
            gen_cache_size_inputs=gen_cache_size_inputs,
        )


def fa3_paper_inputs(dtype, device, gen_cache_size_inputs) -> Generator:
    D_HEAD = 128
    H = 2048 // D_HEAD
    for BATCH in [32, 16, 8, 4, 2, 1]:
        N_CTX = 16384 // BATCH
        yield _generated_qkv_inputs(
            shape=(BATCH, H, H, N_CTX, N_CTX, D_HEAD),
            dtype=dtype,
            device=device,
            gen_cache_size_inputs=gen_cache_size_inputs,
        )


def sweep_inputs(dtype, device, gen_cache_size_inputs) -> Generator:
    D = 128
    batch_sizes = [2**i for i in range(6)]
    num_heads = [1, 4, 8, 16]
    seqlen = [512 * (2**i) for i in range(6)]
    for B in batch_sizes:
        for H in num_heads:
            for S in seqlen:
                yield _generated_qkv_inputs(
                    shape=(B, H, H, S, S, D),
                    dtype=dtype,
                    device=device,
                    gen_cache_size_inputs=gen_cache_size_inputs,
                )
