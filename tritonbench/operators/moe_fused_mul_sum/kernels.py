# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl


@triton.jit
def moe_fused_mul_sum_kernel(
    inputs_ptr,
    topk_weights_ptr,
    outputs_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    num_tokens,
    stride_m,
    has_expert_map: tl.constexpr,
    top_k: tl.constexpr,
    size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused weighted reduction of top-k expert outputs.

    Adapted from vLLM's ``moe_fused_mul_sum_kernel``:
    https://github.com/vllm-project/vllm/blob/1baf372bfc14d739860b4a7122877e22ef0dcbf1/vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py
    """
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m < num_tokens
    k_mask = offs_k < size
    mask = m_mask[:, None] & k_mask[None, :]

    a_base = inputs_ptr + (offs_m * stride_m)[:, None] + offs_k[None, :]
    b_base = topk_weights_ptr + offs_m * top_k
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in tl.static_range(top_k):
        b_val = tl.load(b_base + n, mask=m_mask, other=0.0).to(tl.float32)
        if has_expert_map:
            id_val = tl.load(topk_ids_ptr + offs_m * top_k + n, mask=m_mask, other=0)
            expert_mask = tl.load(expert_map_ptr + id_val) >= 0
            a_vec = tl.load(
                a_base + n * size,
                mask=mask & expert_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            a_vec = tl.load(
                a_base + n * size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        acc += a_vec * b_val[:, None]

    out_ptrs = outputs_ptr + (offs_m * size)[:, None] + offs_k[None, :]
    tl.store(out_ptrs, acc.to(outputs_ptr.dtype.element_ty), mask=mask)


def _heuristic_config(
    num_tokens: int,
    size: int,
    element_size: int,
) -> tuple[int, int, int, int]:
    is_fp32 = element_size > 2
    if torch.version.cuda is not None:
        major, minor = torch.cuda.get_device_capability()
        capability = major * 10 + minor
    else:
        # vLLM's numeric device-capability branches are NVIDIA-specific.
        capability = 80
    is_sm90_plus = capability >= 90
    is_before_sm80 = capability < 80

    if is_sm90_plus:
        if is_fp32:
            block_m = 1 if num_tokens <= 4 else 2
        elif num_tokens <= 4:
            block_m = 1
        elif num_tokens <= 128:
            block_m = 2
        else:
            block_m = 4
    elif is_fp32:
        if num_tokens <= 4:
            block_m = 1
        elif num_tokens <= 32:
            block_m = 2
        else:
            block_m = 4
    elif num_tokens <= 4:
        block_m = 1
    elif num_tokens <= 32:
        block_m = 2
    elif num_tokens <= 128:
        block_m = 4
    elif num_tokens <= 1024:
        block_m = 16
    else:
        block_m = 8

    if is_fp32:
        max_block_k = 256
    elif is_before_sm80 or is_sm90_plus:
        max_block_k = 512
    else:
        max_block_k = 1024
    block_k = max(256, min(triton.next_power_of_2(size), max_block_k))

    total = block_m * block_k
    if is_fp32:
        num_warps = max(8, min(16, total // 64))
    else:
        num_warps = max(4, min(16, total // 256))

    if is_before_sm80:
        num_warps = min(num_warps, 8)
        num_stages = 2
    elif is_sm90_plus:
        num_warps = min(num_warps, 8)
        num_stages = 4 if total <= 2048 else 2
    else:
        num_stages = 4 if total <= 2048 else 2

    return block_m, block_k, num_warps, num_stages


def moe_fused_mul_sum(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, top_k, size = inputs.shape
    output = torch.empty((num_tokens, size), dtype=inputs.dtype, device=inputs.device)
    block_m, block_k, num_warps, num_stages = _heuristic_config(
        num_tokens, size, inputs.element_size()
    )
    grid = (triton.cdiv(size, block_k), triton.cdiv(num_tokens, block_m))
    moe_fused_mul_sum_kernel[grid](
        inputs,
        topk_weights,
        output,
        topk_ids,
        expert_map,
        num_tokens,
        top_k * size,
        expert_map is not None,
        top_k,
        size,
        block_m,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
