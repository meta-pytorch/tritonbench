"""Tests for the A/B mode ``--output-json`` report."""

import json
import random
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

from tritonbench.components.do_bench.run import Latency
from tritonbench.utils.ab_test import (
    AB_COMPARISON_KEY,
    compare_ab_results,
    SIDE_A_KEY,
    SIDE_B_KEY,
    write_ab_json,
)
from tritonbench.utils.triton_op import (
    BenchmarkOperatorBackend,
    BenchmarkOperatorMetrics,
    BenchmarkOperatorResult,
    REGISTERED_BENCHMARKS,
    REGISTERED_X_VALS,
)

OP_NAME = "_ab_json_test_op"
X_VAL_NAME = "(M, N)"
BACKENDS = ("triton", "torch")
X_VALS = (1024, 2048)


def _samples(seed: int, mean: float) -> Latency:
    rng = random.Random(seed)
    # Outlier removal keeps the sample count comfortably above MIN_SAMPLE.
    return Latency([rng.gauss(mean, 0.01 * mean) for _ in range(200)])


def _make_result(seed_offset: int, scale: float) -> BenchmarkOperatorResult:
    result = []
    for i, x_val in enumerate(X_VALS):
        backend_metrics = {}
        for j, backend in enumerate(BACKENDS):
            seed = seed_offset + 10 * i + j
            backend_metrics[backend] = BenchmarkOperatorMetrics(
                latency=_samples(seed, scale * (1.0 + i)),
                speedup=1.0 + 0.1 * j,
                extra_metrics={},
            )
        result.append((x_val, backend_metrics))
    return BenchmarkOperatorResult(
        benchmark_name=f"{OP_NAME}_fwd",
        op_name=OP_NAME,
        op_mode="fwd",
        metrics=["latency", "speedup"],
        simple_mode=False,
        result=result,
        expected_num_inputs=len(X_VALS),
    )


class AbTestJsonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        REGISTERED_X_VALS[OP_NAME] = X_VAL_NAME
        REGISTERED_BENCHMARKS[OP_NAME] = OrderedDict(
            (b, BenchmarkOperatorBackend(name=b, label=b)) for b in BACKENDS
        )

    @classmethod
    def tearDownClass(cls):
        REGISTERED_X_VALS.pop(OP_NAME, None)
        REGISTERED_BENCHMARKS.pop(OP_NAME, None)

    def test_single_side_report_has_only_side_a(self):
        report = compare_ab_results(_make_result(0, 1.0), None, ["--rep", "1000"])
        self.assertEqual(list(report.keys()), [SIDE_A_KEY])

        side_a = report[SIDE_A_KEY]
        self.assertEqual(side_a["config"], ["--rep", "1000"])
        self.assertEqual(side_a["global_args"][-2:], ["--rep", "1000"])
        self.assertEqual(side_a["op_name"], OP_NAME)

        # One metrics entry per (backend, x_val) cell.
        metrics = side_a["metrics"]
        self.assertEqual(len(metrics), len(BACKENDS) * len(X_VALS))
        cell = metrics[f"tritonbench_{OP_NAME}[triton-1024]"]
        # The collected metrics, ...
        self.assertGreater(cell["latency"], 0)
        self.assertEqual(cell["speedup"], 1.0)
        # ... then the statistics of the raw latency samples behind them.
        self.assertGreater(cell["n"], 0)
        self.assertAlmostEqual(cell["mean"], cell["latency"], delta=0.1)
        self.assertGreater(cell["stddev"], 0)
        self.assertEqual(len(cell["mean_ci"]), 2)

    def test_two_sided_report_has_all_three_sections(self):
        report = compare_ab_results(
            _make_result(0, 1.0),
            _make_result(100, 1.2),
            ["--rep", "1000"],
            ["--rep", "2000"],
        )
        self.assertEqual(
            sorted(report.keys()), sorted([SIDE_A_KEY, SIDE_B_KEY, AB_COMPARISON_KEY])
        )
        self.assertEqual(report[SIDE_B_KEY]["config"], ["--rep", "2000"])
        # Side B is 20% slower by construction.
        key = f"tritonbench_{OP_NAME}[triton-1024]"
        self.assertAlmostEqual(
            report[SIDE_B_KEY]["metrics"][key]["mean"],
            1.2 * report[SIDE_A_KEY]["metrics"][key]["mean"],
            delta=0.01,
        )

        comparison = report[AB_COMPARISON_KEY]
        self.assertEqual(comparison["op_name"], OP_NAME)
        self.assertEqual(comparison["backends"], sorted(BACKENDS))
        self.assertEqual(
            comparison["config_differences"],
            {"rep": {SIDE_A_KEY: "1000", SIDE_B_KEY: "2000"}},
        )
        self.assertIn("latency", comparison["performance_summary"]["triton"])
        self.assertTrue(comparison["detailed_comparison"])

        # Side B is 20% slower by construction, so every cell should say so.
        latency_comparison = comparison["latency_comparison"]
        self.assertEqual(len(latency_comparison), len(BACKENDS) * len(X_VALS))
        for entry in latency_comparison:
            self.assertTrue(entry["comparison"]["significant"])
            self.assertAlmostEqual(entry["comparison"]["pct_change"], 20.0, delta=1.0)

    def test_written_json_is_valid_and_round_trips(self):
        # parse_constant fires on NaN/Infinity, which are not valid JSON.
        def _reject(constant):
            raise AssertionError(f"invalid JSON constant: {constant}")

        report = compare_ab_results(
            _make_result(0, 1.0), _make_result(100, 1.2), [], []
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ab.json"
            write_ab_json(str(path), report)
            with open(path) as f:
                loaded = json.load(f, parse_constant=_reject)
        self.assertEqual(loaded, report)


if __name__ == "__main__":
    unittest.main()
