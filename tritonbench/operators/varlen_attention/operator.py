# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

"""Variable-length attention benchmarks with synthetic inputs."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Callable, Generator

import torch
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
    register_x_val,
)
from tritonbench.utils.workload_shapes import (
    VARLEN_ATTENTION_GROUPS,
    VARLEN_CROSS_ATTENTION_SHAPES,
    VARLEN_SELF_ATTENTION_SHAPES,
    VarlenAttentionShape,
    expand_shape_names,
    make_balanced_lengths,
)

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    HAS_FLASH_ATTN = True
except ImportError as error:
    logging.getLogger(__name__).warning("flash_ck is unavailable: %s", error)
    HAS_FLASH_ATTN = False


SHAPES: dict[str, VarlenAttentionShape] = {
    **VARLEN_CROSS_ATTENTION_SHAPES,
    **VARLEN_SELF_ATTENTION_SHAPES,
}


def parse_op_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="varlen_cross",
        help=(
            "Comma-separated config or group names: "
            f"{','.join((*VARLEN_ATTENTION_GROUPS, *SHAPES))}"
        ),
    )
    return parser.parse_args(args)


def _make_offsets(
    total_tokens: int,
    batch: int,
    max_seqlen: int,
    device: torch.device | str,
) -> torch.Tensor:
    lengths = make_balanced_lengths(total_tokens, batch, max_seqlen)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
    return torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), lengths_tensor.cumsum(0)]
    )


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["latency", "tflops", "accuracy"]

    def __init__(
        self,
        tb_args: argparse.Namespace,
        extra_args: list[str] | None = None,
    ) -> None:
        super().__init__(tb_args, extra_args=extra_args)
        args = parse_op_args(self.extra_args)
        self.config_names = expand_shape_names(
            args.config, SHAPES, VARLEN_ATTENTION_GROUPS
        )

    @register_benchmark(
        enabled=HAS_FLASH_ATTN,
        baseline=True,
        label="FlashAttention varlen (CK on AMD)",
    )
    def flash_ck(
        self,
        _config_name: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
    ) -> Callable[[], torch.Tensor]:
        def fn() -> torch.Tensor:
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=q.shape[-1] ** -0.5,
                causal=causal,
            )

        return fn

    def get_input_iter(self) -> Generator:
        requires_grad = self.mode != Mode.FWD_NO_GRAD
        for name in self.config_names:
            shape = SHAPES[name]
            cu_seqlens_q = _make_offsets(
                shape.total_q, shape.batch, shape.max_seqlen_q, self.device
            )
            cu_seqlens_k = _make_offsets(
                shape.total_kv, shape.batch, shape.max_seqlen_kv, self.device
            )

            q = torch.randn(
                shape.total_q,
                shape.num_heads,
                shape.head_dim,
                device=self.device,
                dtype=self.dtype,
                requires_grad=requires_grad,
            )
            k = torch.randn(
                shape.total_kv,
                shape.num_heads,
                shape.head_dim,
                device=self.device,
                dtype=self.dtype,
                requires_grad=requires_grad,
            )
            if shape.causal:
                v_storage = torch.randn(
                    shape.total_kv,
                    3,
                    shape.num_heads,
                    shape.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                v = v_storage[:, 0].detach().requires_grad_(requires_grad)
            else:
                v = torch.randn(
                    shape.total_kv,
                    shape.num_heads,
                    shape.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=requires_grad,
                )
            yield (
                name,
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                shape.max_seqlen_q,
                shape.max_seqlen_kv,
                shape.causal,
            )

    @register_x_val(
        label="(name, total_q, total_kv, B, H, D, max_q, max_kv, causal)"
    )
    def get_x_val(self, example_inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        name, q, k, _v, cu_q, _cu_k, max_q, max_kv, causal = example_inputs
        return (
            name,
            q.shape[0],
            k.shape[0],
            cu_q.numel() - 1,
            q.shape[1],
            q.shape[2],
            max_q,
            max_kv,
            causal,
        )

    @register_metric(x_only=True)
    def flops(
        self,
        fn_name: str,
        example_inputs: tuple[Any, ...],
        metrics: BenchmarkOperatorMetrics,
    ) -> float:
        del fn_name, metrics
        name = example_inputs[0]
        shape = SHAPES[name]
        q_lengths = make_balanced_lengths(
            shape.total_q, shape.batch, shape.max_seqlen_q
        )
        kv_lengths = make_balanced_lengths(
            shape.total_kv, shape.batch, shape.max_seqlen_kv
        )
        if shape.causal:
            attention_pairs = sum(length * (length + 1) // 2 for length in q_lengths)
        else:
            attention_pairs = sum(
                q_len * kv_len for q_len, kv_len in zip(q_lengths, kv_lengths)
            )
        flops = 4.0 * shape.num_heads * shape.head_dim * attention_pairs
        if self.mode == Mode.BWD:
            flops *= 2.5
        elif self.mode == Mode.FWD_BWD:
            flops *= 3.5
        return flops

    def get_grad_to_none(self, args: tuple[Any, ...]) -> list[torch.Tensor]:
        return [args[1], args[2], args[3]]
