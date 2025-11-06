# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TRTLLM FMHA utility functions for handling tensor conversion and kernel preparation.
"""

import torch


def trtllm_paged_attention_decode_func(q, k_cache, v_cache, cache_seqlens):
    """
    TRTLLM FMHA paged attention decode function that prepares inputs for the
    FlashInfer fmha_gen library's trtllm_paged_attention_decode kernel.

    This function converts standard KV cache tensors to paged format and prepares
    all necessary parameters for the TRTLLM kernel.

    Args:
        q: Query tensor [batch, seq_len_q, num_qo_heads, head_dim]
        k_cache: Key cache tensor [batch, max_seq_len_kv, num_kv_heads, head_dim]
        v_cache: Value cache tensor [batch, max_seq_len_kv, num_kv_heads, head_dim]
        cache_seqlens: Sequence lengths tensor [batch]

    Returns:
        Tuple of arguments for torch.ops.fmha_gen.trtllm_paged_attention_decode:
        (out, out_scale_factor, query, key_cache, value_cache, workspace_buffer,
         block_tables, seq_lens, max_kv_len, bmm1_scale, bmm2_scale, o_sf_scale,
         o_sf_vec_size, o_sf_start_index, window_left, sm_count, enable_pdl,
         workspace_size, attention_sinks)
    """

    device = q.device
    # Convert input tensors to paged format for TRTLLM FMHA
    batch_size, seq_len_q, num_qo_heads, head_dim = q.shape
    _, max_seq_len_kv, num_kv_heads, _ = k_cache.shape

    # Use page size of 16 for TRTLLM FMHA
    page_size = 16
    max_num_blocks_per_seq = (max_seq_len_kv + page_size - 1) // page_size
    total_pages = batch_size * max_num_blocks_per_seq

    # Reshape k_cache and v_cache to paged format [total_pages, num_kv_heads, page_size, head_dim]
    k_cache_paged = k_cache.view(
        batch_size, max_num_blocks_per_seq, page_size, num_kv_heads, head_dim
    )
    k_cache_paged = k_cache_paged.permute(0, 1, 3, 2, 4).contiguous()
    k_cache_paged = k_cache_paged.view(total_pages, num_kv_heads, page_size, head_dim)

    v_cache_paged = v_cache.view(
        batch_size, max_num_blocks_per_seq, page_size, num_kv_heads, head_dim
    )
    v_cache_paged = v_cache_paged.permute(0, 1, 3, 2, 4).contiguous()
    v_cache_paged = v_cache_paged.view(total_pages, num_kv_heads, page_size, head_dim)

    # Create block tables
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int32, device=device
    )
    for i in range(batch_size):
        for j in range(max_num_blocks_per_seq):
            block_tables[i, j] = i * max_num_blocks_per_seq + j

    # Create output tensor
    out = torch.zeros_like(q)

    # Create workspace buffer
    workspace_size = 128 * 1024 * 1024  # 128MB
    workspace_buffer = torch.zeros(workspace_size, dtype=torch.uint8, device=device)

    # Attention parameters
    max_seq_len = cache_seqlens.max().item()
    bmm1_scale = 1.0 / (head_dim**0.5)
    bmm2_scale = 1.0

    # Output scale factor parameters (not used for non-FP8)
    out_scale_factor = None  # Optional tensor for FP8 output scaling
    o_sf_scale = -1.0  # Output scale factor scale (disabled when -1)
    o_sf_vec_size = -1  # Output scale factor vector size (disabled when -1)
    o_sf_start_index = -1  # Output scale factor start index (disabled when -1)

    # Attention window settings
    window_left = -1  # No sliding window (disabled when -1)

    # Device settings
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    # PDL (Programmatic Dependent Launch) settings
    enable_pdl = False

    # Attention sinks (optional)
    attention_sinks = None

    # Return tuple matching trtllm_paged_attention_decode signature:
    # void trtllm_paged_attention_decode(
    #     at::Tensor out,
    #     std::optional<at::Tensor> out_scale_factor,
    #     at::Tensor query,
    #     at::Tensor key_cache,
    #     at::Tensor value_cache,
    #     at::Tensor workspace_buffer,
    #     at::Tensor block_tables,
    #     at::Tensor seq_lens,
    #     int64_t max_kv_len,
    #     double bmm1_scale,
    #     double bmm2_scale,
    #     double o_sf_scale,
    #     int64_t o_sf_vec_size,
    #     int64_t o_sf_start_index,
    #     int64_t window_left,
    #     int64_t sm_count,
    #     bool enable_pdl,
    #     int64_t workspace_size,
    #     std::optional<at::Tensor> attention_sinks
    # )

    args = (
        out,  # out
        out_scale_factor,  # out_scale_factor (optional)
        q,  # query
        k_cache_paged,  # key_cache
        v_cache_paged,  # value_cache
        workspace_buffer,  # workspace_buffer
        block_tables,  # block_tables
        cache_seqlens,  # seq_lens
        max_seq_len,  # max_kv_len
        bmm1_scale,  # bmm1_scale
        bmm2_scale,  # bmm2_scale
        o_sf_scale,  # o_sf_scale
        o_sf_vec_size,  # o_sf_vec_size
        o_sf_start_index,  # o_sf_start_index
        window_left,  # window_left
        sm_count,  # sm_count
        enable_pdl,  # enable_pdl
        workspace_size,  # workspace_size
        attention_sinks,  # attention_sinks (optional)
    )
    return args
