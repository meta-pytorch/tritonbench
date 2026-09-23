# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest

from tritonbench.utils.workload_shapes import (
    DENSE_SWIGLU_SHAPES,
    DENSE_SDPA_GROUPS,
    DENSE_SDPA_SHAPES,
    GEMM_SHAPES,
    JAGGED_DENSE_BMM_SHAPES,
    LINEAR_RESIDUAL_SHAPES,
    MMA_GROUPS,
    MOE_SWIGLU_SHAPES,
    VARLEN_ATTENTION_GROUPS,
    VARLEN_CROSS_ATTENTION_SHAPES,
    VARLEN_SELF_ATTENTION_SHAPES,
    expand_shape_names,
    make_balanced_lengths,
)


class ExpandShapeNamesTest(unittest.TestCase):
    def test_expands_dense_group(self) -> None:
        self.assertEqual(
            list(DENSE_SDPA_SHAPES),
            expand_shape_names("dense_sdpa", DENSE_SDPA_SHAPES, DENSE_SDPA_GROUPS),
        )

    def test_expands_mixed_varlen_selection(self) -> None:
        selected = expand_shape_names(
            "varlen_cross,varlen_self_max",
            {**VARLEN_CROSS_ATTENTION_SHAPES, **VARLEN_SELF_ATTENTION_SHAPES},
            VARLEN_ATTENTION_GROUPS,
        )
        self.assertEqual(
            [*VARLEN_CROSS_ATTENTION_SHAPES, "varlen_self_max"], selected
        )

    def test_rejects_unknown_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown --config"):
            expand_shape_names("missing", DENSE_SDPA_SHAPES, DENSE_SDPA_GROUPS)

    def test_non_attention_catalogs_cover_mma_shapes(self) -> None:
        self.assertEqual(4, len(DENSE_SWIGLU_SHAPES))
        self.assertEqual(4, len(MOE_SWIGLU_SHAPES))
        self.assertEqual(8, len(LINEAR_RESIDUAL_SHAPES))
        self.assertEqual(8, len(JAGGED_DENSE_BMM_SHAPES))
        self.assertEqual(11, len(GEMM_SHAPES))
        self.assertEqual(tuple(GEMM_SHAPES), MMA_GROUPS["gemm"])


class MakeBalancedLengthsTest(unittest.TestCase):
    def test_preserves_shape(self) -> None:
        cases = [
            ("cross", 80900, 768, 300),
            ("self", 944906, 768, 3200),
            ("single", 16, 1, 16),
        ]
        for name, total_tokens, batch, max_seqlen in cases:
            with self.subTest(name=name):
                lengths = make_balanced_lengths(total_tokens, batch, max_seqlen)

                self.assertEqual(batch, len(lengths))
                self.assertEqual(total_tokens, sum(lengths))
                self.assertEqual(max_seqlen, max(lengths))
                self.assertGreaterEqual(min(lengths), 1)

    def test_rejects_invalid_shape(self) -> None:
        cases = [
            ("zero_batch", 10, 0, 10),
            ("too_few_tokens", 3, 4, 4),
            ("too_many_tokens", 17, 4, 4),
            ("single_mismatch", 15, 1, 16),
        ]
        for name, total_tokens, batch, max_seqlen in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                make_balanced_lengths(total_tokens, batch, max_seqlen)
