# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

from tritonbench.operators import load_opbench_by_name
from tritonbench.utils.parser import get_parser


class KimiDeltaAttentionTest(unittest.TestCase):
    def _run_accuracy(self, mode: str, tolerance: float) -> None:
        args = [
            "--op",
            "kimi_delta_attention",
            "--mode",
            mode,
            "--metrics",
            "accuracy",
            "--only",
            "chunk_kda,kda_ws",
            "--test-only",
            "--batch",
            "1",
            "--seq-lens",
            "256",
            "--heads",
            "3",
            "--head-dim",
            "128",
            "--atol",
            str(tolerance),
            "--rtol",
            str(tolerance),
        ]
        parser = get_parser()
        tb_args, extra_args = parser.parse_known_args(args)
        operator_class = load_opbench_by_name(tb_args.op)
        operator = operator_class(tb_args=tb_args, extra_args=extra_args)
        operator.run()

        headers, rows = operator.output._table()
        accuracy_columns = [
            index for index, header in enumerate(headers) if "accuracy" in header.lower()
        ]
        self.assertTrue(accuracy_columns)
        self.assertTrue(rows)
        for row in rows:
            for index in accuracy_columns:
                if row[index] is not None:
                    self.assertEqual(int(row[index]), 1)

    def test_forward_accuracy(self) -> None:
        self._run_accuracy("fwd", 3e-2)

    def test_backward_accuracy(self) -> None:
        self._run_accuracy("bwd", 5e-2)


if __name__ == "__main__":
    unittest.main()

