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


def customized_inputs(
    custom_shapes_file: Optional[str] = None,
    custom_shapes_attr: str = "ATTENTION_SHAPES",
    **kwargs,
) -> Generator:
    """
    Generate QKV inputs for each shape.

    If custom_shapes_file is provided, shapes are loaded from that file.
    Otherwise, the default CUSTOMIZED_SHAPES from this module are used.

    Args:
        custom_shapes_file: Optional absolute path to a Python file containing
                           a list of attention shapes.
        custom_shapes_attr: Name of the attribute in the file containing the shapes
                           list (default: ATTENTION_SHAPES).
        **kwargs: Additional arguments passed to _generate_qkv_inputs

    Yields:
        Tuple of (tensors, shape) for each attention shape
    """
    if custom_shapes_file:
        shapes = load_shapes_from_file(custom_shapes_file, custom_shapes_attr)
    else:
        shapes = ATTENTION_SHAPES

    for shape in shapes:
        yield _generate_qkv_inputs(shape, **kwargs)
