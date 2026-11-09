# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseAttentionShape:
    batch: int
    seqlen_q: int
    seqlen_kv: int
    num_heads: int
    head_dim: int
    causal: bool


@dataclass(frozen=True)
class VarlenAttentionShape:
    total_q: int
    total_kv: int
    batch: int
    num_heads: int
    head_dim: int
    max_seqlen_q: int
    max_seqlen_kv: int
    causal: bool


@dataclass(frozen=True)
class SwiGLUShape:
    tokens: int
    input_dim: int
    hidden_dim: int
    experts: int | None = None


@dataclass(frozen=True)
class LinearResidualShape:
    tokens: int
    input_dim: int
    output_dim: int


@dataclass(frozen=True)
class JaggedDenseBmmShape:
    total_tokens: int
    batch: int
    max_seqlen: int
    n: int
    k: int


@dataclass(frozen=True)
class GemmShape:
    op: str
    m: int
    n: int
    k: int
    layout_a: str
    layout_b: str


DENSE_SDPA_SHAPES: dict[str, DenseAttentionShape] = {
    "dense_sdpa_min": DenseAttentionShape(3139, 10, 10, 8, 128, False),
    "dense_sdpa_p50": DenseAttentionShape(4045, 10, 10, 8, 128, False),
    "dense_sdpa_p90": DenseAttentionShape(4433, 10, 10, 8, 128, False),
    "dense_sdpa_max": DenseAttentionShape(5095, 10, 10, 8, 128, False),
}


VARLEN_CROSS_ATTENTION_SHAPES: dict[str, VarlenAttentionShape] = {
    "varlen_cross_min": VarlenAttentionShape(
        62780, 946279, 768, 4, 128, 300, 3200, False
    ),
    "varlen_cross_p50": VarlenAttentionShape(
        80900, 940737, 768, 4, 128, 300, 3200, False
    ),
    "varlen_cross_p90": VarlenAttentionShape(
        88660, 944840, 768, 4, 128, 300, 3200, False
    ),
    "varlen_cross_max": VarlenAttentionShape(
        101900, 946104, 768, 4, 128, 300, 3200, False
    ),
    "varlen_cross_max_q": VarlenAttentionShape(
        84880, 945242, 768, 4, 128, 400, 3200, False
    ),
}


VARLEN_SELF_ATTENTION_SHAPES: dict[str, VarlenAttentionShape] = {
    "varlen_self_min": VarlenAttentionShape(
        925210, 925210, 768, 4, 128, 3200, 3200, True
    ),
    "varlen_self_p50": VarlenAttentionShape(
        944906, 944906, 768, 4, 128, 3200, 3200, True
    ),
    "varlen_self_p90": VarlenAttentionShape(
        946581, 946581, 768, 4, 128, 3200, 3200, True
    ),
    "varlen_self_max": VarlenAttentionShape(
        948456, 948456, 768, 4, 128, 3200, 3200, True
    ),
}


DENSE_SWIGLU_SHAPES: dict[str, SwiGLUShape] = {
    "dense_swiglu_min": SwiGLUShape(925210, 512, 2048),
    "dense_swiglu_p50": SwiGLUShape(944906, 512, 2048),
    "dense_swiglu_p90": SwiGLUShape(946581, 512, 2048),
    "dense_swiglu_max": SwiGLUShape(948456, 512, 2048),
}

MOE_SWIGLU_SHAPES: dict[str, SwiGLUShape] = {
    "moe_swiglu_min": SwiGLUShape(29011, 1024, 2736, 40),
    "moe_swiglu_p50": SwiGLUShape(30296, 1024, 2736, 40),
    "moe_swiglu_p90": SwiGLUShape(30767, 1024, 2736, 40),
    "moe_swiglu_max": SwiGLUShape(31144, 1024, 2736, 40),
}

LINEAR_RESIDUAL_SHAPES: dict[str, LinearResidualShape] = {
    f"linear_{input_dim}x512_{percentile}": LinearResidualShape(
        tokens, input_dim, 512
    )
    for input_dim in (512, 2048)
    for percentile, tokens in (
        ("min", 925210),
        ("p50", 944906),
        ("p90", 946581),
        ("max", 948456),
    )
}

JAGGED_DENSE_BMM_SHAPES: dict[str, JaggedDenseBmmShape] = {
    f"jagged_bmm_n{n}_{percentile}": JaggedDenseBmmShape(
        total_tokens, 768, max_seqlen, n, 512
    )
    for n in (224, 1664)
    for percentile, total_tokens, max_seqlen in (
        ("min", 830389, 3087),
        ("p50", 850554, 3082),
        ("p90", 852404, 3087),
        ("max", 854091, 3086),
    )
}

GEMM_SHAPES: dict[str, GemmShape] = {
    "gemm_addmm_768x851968x256": GemmShape(
        "addmm", 768, 851968, 256, "row", "column"
    ),
    "gemm_768x256x851968": GemmShape("mm", 768, 256, 851968, "row", "row"),
    "gemm_851968x256x768": GemmShape("mm", 851968, 256, 768, "column", "row"),
    "gemm_512x256x98304": GemmShape("mm", 512, 256, 98304, "column", "row"),
    "gemm_98304x256x512": GemmShape("mm", 98304, 256, 512, "row", "row"),
    "gemm_768x256x114688": GemmShape("mm", 768, 256, 114688, "row", "row"),
    "gemm_addmm_768x114688x256": GemmShape(
        "addmm", 768, 114688, 256, "row", "column"
    ),
    "gemm_114688x256x768": GemmShape("mm", 114688, 256, 768, "column", "row"),
    "gemm_addmm_98304x512x256": GemmShape(
        "addmm", 98304, 512, 256, "row", "column"
    ),
    "gemm_43500x1024x1024": GemmShape("mm", 43500, 1024, 1024, "row", "row"),
    "gemm_1024x1024x43500": GemmShape("mm", 1024, 1024, 43500, "column", "row"),
}


DENSE_SDPA_GROUPS: dict[str, tuple[str, ...]] = {
    "dense_sdpa": tuple(DENSE_SDPA_SHAPES),
}

VARLEN_ATTENTION_GROUPS: dict[str, tuple[str, ...]] = {
    "varlen_cross": tuple(VARLEN_CROSS_ATTENTION_SHAPES),
    "varlen_self": tuple(VARLEN_SELF_ATTENTION_SHAPES),
    "varlen_all": (
        *VARLEN_CROSS_ATTENTION_SHAPES,
        *VARLEN_SELF_ATTENTION_SHAPES,
    ),
}

MMA_GROUPS: dict[str, tuple[str, ...]] = {
    "dense_swiglu": tuple(DENSE_SWIGLU_SHAPES),
    "moe_swiglu": tuple(MOE_SWIGLU_SHAPES),
    "linear_residual": tuple(LINEAR_RESIDUAL_SHAPES),
    "jagged_bmm": tuple(JAGGED_DENSE_BMM_SHAPES),
    "gemm": tuple(GEMM_SHAPES),
}


def expand_shape_names(
    value: str,
    shapes: dict[str, object],
    groups: dict[str, tuple[str, ...]],
) -> list[str]:
    names: list[str] = []
    for item in value.split(","):
        name = item.strip()
        if name in groups:
            names.extend(groups[name])
        elif name in shapes:
            names.append(name)
        else:
            known = ",".join((*groups, *shapes))
            raise ValueError(f"unknown --config {name!r}; known: {known}")
    return names


def make_balanced_lengths(
    total_tokens: int,
    batch: int,
    max_seqlen: int,
) -> list[int]:
    if batch <= 0 or max_seqlen <= 0:
        raise ValueError("batch and max_seqlen must be positive")
    if total_tokens < batch or total_tokens > batch * max_seqlen:
        raise ValueError(
            f"total_tokens={total_tokens} is outside [{batch}, {batch * max_seqlen}]"
        )
    if batch == 1:
        if total_tokens != max_seqlen:
            raise ValueError("a single sequence must have total_tokens == max_seqlen")
        return [max_seqlen]

    remaining = total_tokens - max_seqlen
    base, extra = divmod(remaining, batch - 1)
    if base < 1 or base + (1 if extra else 0) > max_seqlen:
        raise ValueError("cannot preserve both total_tokens and max_seqlen")
    return [max_seqlen, *([base + 1] * extra), *([base] * (batch - 1 - extra))]
