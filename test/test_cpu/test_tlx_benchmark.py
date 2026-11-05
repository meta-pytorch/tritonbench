import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from parameterized import parameterized
from tritonbench.utils.tlx_benchmark import plot_results
from tritonbench.utils.triton_op import (
    BenchmarkOperatorMetrics,
    BenchmarkOperatorResult,
)


class TLXBenchmarkTest(unittest.TestCase):
    def _make_operator(self, temp_dir, input_ids, result_count, kernel_ids):
        output = BenchmarkOperatorResult(
            benchmark_name="tlx-report-test",
            op_name="flex_attention",
            op_mode="fwd",
            metrics=["speedup", "accuracy"],
            simple_mode=True,
            result=[
                (
                    "same shape",
                    {
                        "torch_tlx_flex_attention": BenchmarkOperatorMetrics(
                            speedup=2**i, accuracy=True
                        )
                    },
                )
                for i in range(result_count)
            ],
            expected_num_inputs=len(input_ids),
        )
        op = SimpleNamespace(
            name="flex_attention",
            torch_tlx_config=None,
            tb_args=SimpleNamespace(output_dir=temp_dir),
            output=output,
            _input_ids=input_ids,
            _tlx_contracts=[{"shape": i} for i in range(7)],
            _tlx_evidence=[],
        )
        if kernel_ids is not None:
            op._tlx_kernel_inputs = kernel_ids
        return op

    @parameterized.expand(
        [
            ("zero_based", [0, 1], 2, {0}, [True, False], "1.000x over 1/2"),
            ("nonzero", [4], 1, {4}, [True], "1.000x over 1/1"),
            (
                "noncontiguous",
                [1, 4, 6],
                3,
                {4, 6},
                [False, True, True],
                "2.828x over 2/3",
            ),
            ("partial", [1, 4, 6], 2, {1, 6}, [True, False], "1.000x over 1/2"),
            ("empty", [1, 4], 0, {4}, [], "no TLX kernel speedup on any of 0"),
            ("fallback", [1, 4], 2, set(), [False, False], "no TLX kernel speedup"),
            ("unrecorded", [1, 4], 2, None, [None, None], None),
        ]
    )
    def test_selected_input_reporting(
        self, _name, input_ids, result_count, kernel_ids, expected_coverage, summary
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            op = self._make_operator(temp_dir, input_ids, result_count, kernel_ids)
            with patch("tritonbench.utils.plot_utils.plot_benchmark_results") as plot:
                plot_results(op)
            rows = json.loads((Path(temp_dir) / "results.json").read_text())
        self.assertEqual([row["tlx_kernel"] for row in rows], expected_coverage)
        completed_ids = input_ids[:result_count]
        self.assertEqual([row["input_id"] for row in rows], completed_ids)
        self.assertEqual(
            [row["input_contract"] for row in rows],
            [{"shape": i} for i in completed_ids],
        )
        self.assertEqual(plot.call_count, 3)
        self.assertEqual(
            [label for label, _ in plot.call_args_list[0].args[0].result],
            [f"{i}: ({i},)" for i in completed_ids],
        )
        expected_fallback = (
            None if kernel_ids is None else [not used for used in expected_coverage]
        )
        for call in plot.call_args_list:
            self.assertEqual(call.kwargs["tlx_fallback"], expected_fallback)
        title = plot.call_args_list[0].kwargs["plot_title"]
        if summary is None:
            self.assertNotIn("torchTLX geomean", title)
        else:
            self.assertIn(summary, title)
