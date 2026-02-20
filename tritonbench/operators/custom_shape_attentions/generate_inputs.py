# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import math
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import torch


@dataclass
class AttentionShape:
    """Defines the shape and attention parameters for a benchmark input."""

    name: str
    batch: int
    n_heads: int
    n_heads_kv: int
    seq_len: int
    seq_len_kv: int
    d_head: int
    causal: bool = True
    window_size: Tuple[int, int] = (-1, -1)  # (-1, -1) means no sliding window
    # Page attention fields
    paged_attention: bool = False
    block_size: int = 16  # Default block size for paged attention
    max_model_seq_len: int = 32768  # Maximum model sequence length
    page_shuffle: bool = False  # Whether to randomly shuffle page blocks


# Default placeholder shapes for benchmarking (non-confidential)
ATTENTION_SHAPES: List[AttentionShape] = [
    # Example: Standard self-attention shape
    AttentionShape(
        name="example_self_attn",
        batch=1,
        n_heads=32,
        n_heads_kv=8,
        seq_len=2048,
        seq_len_kv=2048,
        d_head=128,
        causal=True,
        window_size=(-1, -1),
    ),
    # Example: Cross-attention shape
    AttentionShape(
        name="example_cross_attn",
        batch=1,
        n_heads=32,
        n_heads_kv=8,
        seq_len=1024,
        seq_len_kv=4096,
        d_head=64,
        causal=False,
        window_size=(-1, -1),
    ),
    # Example: Sliding window attention shape
    AttentionShape(
        name="example_swa_attn",
        batch=1,
        n_heads=32,
        n_heads_kv=8,
        seq_len=2048,
        seq_len_kv=8192,
        d_head=64,
        causal=False,
        window_size=(1024, 1024),
    ),
    # Example: Paged attention shape for blocked KV cache
    # KV cache blocks: [B * max_model_seq_len // block_size, block_size, n_kv_h, h_dim]
    # Block table: [B, max_model_seq_len // block_size] maps positions to cache blocks
    # For batch i, block_table values are [0, 1, ..., seq_len_kv//block_size-1, 0, ...] + i * (max_model_seq_len // block_size)
    AttentionShape(
        name="example_paged_attn",
        batch=4,
        n_heads=32,
        n_heads_kv=8,
        seq_len=1,  # Decode phase: single query token
        seq_len_kv=4096,  # Context length (how many KV tokens to attend to)
        d_head=128,
        causal=True,
        window_size=(-1, -1),
        paged_attention=True,
        block_size=16,  # Block size for KV cache
        max_model_seq_len=8192,  # Maximum model sequence length
    ),
]


def _convert_to_attention_shapes(shapes_data: List) -> List[AttentionShape]:
    """
    Convert a list of shape data (either dicts or AttentionShape objects) to AttentionShape objects.

    Args:
        shapes_data: List of dictionaries or AttentionShape objects

    Returns:
        List of AttentionShape objects
    """
    shapes = []
    for shape_item in shapes_data:
        if isinstance(shape_item, AttentionShape):
            shapes.append(shape_item)
        elif isinstance(shape_item, dict):
            shapes.append(
                AttentionShape(
                    name=shape_item["name"],
                    batch=shape_item["batch"],
                    n_heads=shape_item["n_heads"],
                    n_heads_kv=shape_item["n_heads_kv"],
                    seq_len=shape_item["seq_len"],
                    seq_len_kv=shape_item["seq_len_kv"],
                    d_head=shape_item["d_head"],
                    causal=shape_item.get("causal", True),
                    window_size=tuple(shape_item.get("window_size", (-1, -1))),
                    paged_attention=shape_item.get("paged_attention", False),
                    block_size=shape_item.get("block_size", 16),
                    max_model_seq_len=shape_item.get("max_model_seq_len", 32768),
                    page_shuffle=shape_item.get("page_shuffle", False),
                )
            )
        else:
            raise TypeError(
                f"Expected AttentionShape or dict, got {type(shape_item).__name__}"
            )
    return shapes


def load_shapes_from_file(
    file_path: str, attr_name: str = "ATTENTION_SHAPES"
) -> List[AttentionShape]:
    """
    Load attention shapes from an external Python file.

    The file should contain a list (default: ATTENTION_SHAPES) with dictionaries
    or AttentionShape objects defining each shape. Each dictionary should have
    the following keys:
    - name: str
    - batch: int
    - n_heads: int
    - n_heads_kv: int
    - seq_len: int
    - seq_len_kv: int
    - d_head: int
    - causal: bool (optional, default: True)
    - window_size: Tuple[int, int] (optional, default: (-1, -1))

    Args:
        file_path: Absolute path to the Python file containing the shapes list
        attr_name: Name of the attribute in the file (default: ATTENTION_SHAPES)

    Returns:
        List of AttentionShape objects
    """
    spec = importlib.util.spec_from_file_location("custom_shapes", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shapes from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, attr_name):
        raise AttributeError(f"File {file_path} does not contain '{attr_name}' list")

    shapes_data = getattr(module, attr_name)
    if not isinstance(shapes_data, list):
        raise TypeError(
            f"Expected '{attr_name}' to be a list, got {type(shapes_data).__name__}"
        )

    return _convert_to_attention_shapes(shapes_data)


def get_bytes(x):
    return x.numel() * x.element_size()


def _generate_qkv_inputs(
    shape: AttentionShape, dtype, device, gen_cache_size_inputs, max_inputs_per_iter
) -> Tuple[Tuple[torch.Tensor, ...], AttentionShape]:
    """Generate QKV tensors and return them along with the shape metadata."""
    requires_grad = True

    q = torch.randn(
        (shape.batch, shape.n_heads, shape.seq_len, shape.d_head),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        (shape.batch, shape.n_heads_kv, shape.seq_len_kv, shape.d_head),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    v = torch.randn(
        (shape.batch, shape.n_heads_kv, shape.seq_len_kv, shape.d_head),
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
        if max_inputs_per_iter > 0:
            num_inputs = min(num_inputs, max_inputs_per_iter)
        for _ in range(num_inputs - 1):
            for t in (q, k, v):
                t = t.clone().detach()
                t.requires_grad = True
                inputs.append(t)
    assert len(inputs) % 3 == 0
    return tuple(inputs), shape


def _generate_paged_attention_inputs(
    shape: AttentionShape, dtype, device, gen_cache_size_inputs, max_inputs_per_iter
) -> Tuple[Tuple[torch.Tensor, ...], AttentionShape]:
    """
    Generate paged attention inputs for flash_attn_varlen_func.

    Paged attention uses KV cache blocks and a page table for efficient memory management.
    This follows the flash_attn_varlen_func API which expects:
    - q: [total_q, num_heads, head_dim] - Unpadded query tensor
    - k_cache: [total_blocks, block_size, num_heads_kv, head_dim] - Paged KV cache for keys
    - v_cache: [total_blocks, block_size, num_heads_kv, head_dim] - Paged KV cache for values
    - cu_seqlens_q: [batch + 1] - Cumulative sequence lengths for queries
    - cu_seqlens_k: [batch + 1] - Cumulative sequence lengths for keys (optional with page_table)
    - max_seqlen_q: int - Maximum query sequence length
    - max_seqlen_k: int - Maximum key sequence length
    - page_table: [batch, max_num_blocks_per_seq] - Maps sequence positions to cache block indices
    - seqused_k: [batch] - Actual KV sequence lengths for each request

    Page table values:
    - For request i: [0, 1, 2, ..., num_blocks_used-1, 0, 0, ...] + offset
    - where offset = i * max_num_blocks_per_seq
    """
    batch = shape.batch
    block_size = shape.block_size
    max_model_seq_len = shape.max_model_seq_len
    n_heads = shape.n_heads
    n_kv_h = shape.n_heads_kv
    h_dim = shape.d_head
    seq_len_q = shape.seq_len
    seq_len_kv = shape.seq_len_kv

    max_num_blocks_per_seq = max_model_seq_len // block_size
    total_blocks = batch * max_num_blocks_per_seq
    total_q = batch * seq_len_q

    # Q in unpadded format: [total_q, num_heads, head_dim]
    q = torch.randn(
        (total_q, n_heads, h_dim),
        dtype=dtype,
        device=device,
    )

    # Paged KV cache: [total_blocks, block_size, num_heads_kv, head_dim]
    k_cache = torch.randn(
        (total_blocks, block_size, n_kv_h, h_dim),
        dtype=dtype,
        device=device,
    )

    v_cache = torch.randn(
        (total_blocks, block_size, n_kv_h, h_dim),
        dtype=dtype,
        device=device,
    )

    # Cumulative sequence lengths for queries: [batch + 1]
    cu_seqlens_q = torch.arange(
        0, (batch + 1) * seq_len_q, seq_len_q, dtype=torch.int32, device=device
    )

    # Page table: [batch, max_num_blocks_per_seq]
    page_table = torch.zeros(
        (batch, max_num_blocks_per_seq), dtype=torch.int32, device=device
    )

    num_blocks_used = (seq_len_kv + block_size - 1) // block_size

    for i in range(batch):
        offset = i * max_num_blocks_per_seq
        if shape.page_shuffle:
            # Shuffle all blocks and take the first num_blocks_used
            shuffled_indices = torch.randperm(max_num_blocks_per_seq, device=device)
            page_table[i, :num_blocks_used] = (
                offset + shuffled_indices[:num_blocks_used]
            )
        else:
            # Continuous page block assignment
            for j in range(num_blocks_used):
                page_table[i, j] = offset + j

    # Actual KV sequence lengths: [batch]
    seqused_k = torch.full((batch,), seq_len_kv, dtype=torch.int32, device=device)

    # max sequence lengths
    max_seqlen_q = seq_len_q
    max_seqlen_k = seq_len_kv

    inputs = [
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        seqused_k,
    ]

    if gen_cache_size_inputs:
        q_bytes = get_bytes(q)
        k_bytes = get_bytes(k_cache)
        v_bytes = get_bytes(v_cache)
        total_bytes = q_bytes + k_bytes + v_bytes
        min_bytes = 128 * 1024 * 1024
        num_inputs = math.ceil(min_bytes / total_bytes)
        if max_inputs_per_iter > 0:
            num_inputs = min(num_inputs, max_inputs_per_iter)
        for _ in range(num_inputs - 1):
            q_clone = q.clone().detach()
            q_clone.requires_grad = True
            k_clone = k_cache.clone().detach()
            k_clone.requires_grad = True
            v_clone = v_cache.clone().detach()
            v_clone.requires_grad = True
            inputs.extend(
                [
                    q_clone,
                    k_clone,
                    v_clone,
                    cu_seqlens_q.clone(),
                    max_seqlen_q,
                    max_seqlen_k,
                    page_table.clone(),
                    seqused_k.clone(),
                ]
            )

    return tuple(inputs), shape


def customized_inputs(
    custom_shapes_file: Optional[str] = None,
    custom_shapes_attr: str = "ATTENTION_SHAPES",
    **kwargs,
) -> Generator:
    """
    Generate QKV inputs for each shape.

    If custom_shapes_file is provided, shapes are loaded from that file.
    Otherwise, the default CUSTOMIZED_SHAPES from this module are used.

    For paged attention shapes, generates:
    - q: [B, n_heads, seq_len, d_head]
    - k_cache: [B * max_model_seq_len // block_size, block_size, n_kv_h, h_dim]
    - v_cache: [B * max_model_seq_len // block_size, block_size, n_kv_h, h_dim]
    - block_table: [B, max_model_seq_len // block_size]
    - seq_lens: [B]

    For regular attention shapes, generates:
    - q: [B, n_heads, seq_len, d_head]
    - k: [B, n_kv_h, seq_len_kv, d_head]
    - v: [B, n_kv_h, seq_len_kv, d_head]

    Args:
        custom_shapes_file: Optional absolute path to a Python file containing
                           a list of attention shapes.
        custom_shapes_attr: Name of the attribute in the file containing the shapes
                           list (default: ATTENTION_SHAPES).
        **kwargs: Additional arguments passed to input generators

    Yields:
        Tuple of (tensors, shape) for each attention shape
    """
    if custom_shapes_file:
        shapes = load_shapes_from_file(custom_shapes_file, custom_shapes_attr)
    else:
        shapes = ATTENTION_SHAPES

    for shape in shapes:
        if shape.paged_attention:
            yield _generate_paged_attention_inputs(shape, **kwargs)
        else:
            yield _generate_qkv_inputs(shape, **kwargs)
