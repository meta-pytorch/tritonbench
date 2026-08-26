"""Tests for the A/B mode ``--output-json`` report."""

import json
import os
import random
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import tritonbench.utils.run_utils as run_utils
from tritonbench.components.do_bench.run import Latency
from tritonbench.utils.ab_test import (
    AB_COMPARISON_KEY,
    compare_ab_results,
    merge_ab_reports,
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


# The command line a side's `config` is reported relative to. Patched in so the
# reports do not pick up whatever argv the test runner was started with.
BASE_ARGV = ["run.py", "--op", "softmax", "--num-inputs", "2", "--M", "4096"]


class AbTestJsonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        REGISTERED_X_VALS[OP_NAME] = X_VAL_NAME
        REGISTERED_BENCHMARKS[OP_NAME] = OrderedDict(
            (b, BenchmarkOperatorBackend(name=b, label=b)) for b in BACKENDS
        )

    def setUp(self):
        patcher = patch.object(sys, "argv", list(BASE_ARGV))
        patcher.start()
        self.addCleanup(patcher.stop)

    @classmethod
    def tearDownClass(cls):
        REGISTERED_X_VALS.pop(OP_NAME, None)
        REGISTERED_BENCHMARKS.pop(OP_NAME, None)

    def test_single_side_report_has_only_side_a(self):
        report = compare_ab_results(_make_result(0, 1.0), None, ["--rep", "1000"])
        self.assertEqual(list(report.keys()), [SIDE_A_KEY])

        side_a = report[SIDE_A_KEY]
        # The command line, plus the side's own args, globals then operator
        # args -- one string. --M 4096 is an operator arg, so it comes last.
        self.assertEqual(
            side_a["config"], "--op softmax --num-inputs 2 --rep 1000 --M 4096"
        )
        self.assertEqual(side_a["op_name"], OP_NAME)

        # One metrics entry per (backend, x_val) cell.
        metrics = side_a["metrics"]
        self.assertEqual(len(metrics), len(BACKENDS) * len(X_VALS))
        cell = metrics[f"tritonbench_{OP_NAME}_fwd[x_1024-triton]"]
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
        self.assertEqual(
            report[SIDE_B_KEY]["config"],
            "--op softmax --num-inputs 2 --rep 2000 --M 4096",
        )
        # Side B is 20% slower by construction.
        key = f"tritonbench_{OP_NAME}_fwd[x_1024-triton]"
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

    def test_merge_makes_every_measurement_a_list_per_repeat(self):
        key = f"tritonbench_{OP_NAME}_fwd[x_1024-triton]"
        repeats = [
            compare_ab_results(
                _make_result(seed, 1.0), _make_result(seed + 100, 1.2), [], []
            )
            for seed in (0, 1000, 2000)
        ]
        merged = merge_ab_reports(repeats)

        cell = merged[SIDE_A_KEY]["metrics"][key]
        self.assertEqual(
            cell["latency"], [r[SIDE_A_KEY]["metrics"][key]["latency"] for r in repeats]
        )
        self.assertEqual(len(cell["mean"]), 3)
        # A value that was already a list becomes a list per repeat.
        self.assertEqual(len(cell["mean_ci"]), 3)
        self.assertEqual(len(cell["mean_ci"][0]), 2)

        # What a number describes stays scalar; only measurements become lists.
        side_a = merged[SIDE_A_KEY]
        self.assertEqual(side_a["op_name"], OP_NAME)
        self.assertEqual(side_a["config"], "--op softmax --num-inputs 2 --M 4096")
        comparison = merged[AB_COMPARISON_KEY]
        self.assertEqual(comparison["backends"], sorted(BACKENDS))
        self.assertEqual(comparison["metrics"], ["latency", "speedup"])
        self.assertEqual(comparison["config_differences"], {})

        # Rows keep one entry per repeat, lined up by what they describe.
        row = comparison["detailed_comparison"][0]
        self.assertIn(row["backend"], BACKENDS)
        self.assertEqual(len(row["pct_change"]), 3)
        verdict = comparison["latency_comparison"][0]["comparison"]
        self.assertEqual(len(verdict["pct_change"]), 3)
        self.assertEqual(verdict["significant"], [True, True, True])

    def test_merge_of_a_single_run_still_yields_lists(self):
        report = compare_ab_results(_make_result(0, 1.0), None, [])
        merged = merge_ab_reports([report])
        key = f"tritonbench_{OP_NAME}_fwd[x_1024-triton]"
        cell = merged[SIDE_A_KEY]["metrics"][key]
        self.assertEqual(len(cell["latency"]), 1)
        self.assertEqual(len(cell["mean"]), 1)

    def test_merge_pads_a_cell_missing_from_one_repeat(self):
        full = compare_ab_results(_make_result(0, 1.0), None, [])
        partial_result = _make_result(0, 1.0)
        # Drop one backend from the second repeat.
        partial_result.result = [
            (x_val, {b: m for b, m in backends.items() if b != "torch"})
            for x_val, backends in partial_result.result
        ]
        partial = compare_ab_results(partial_result, None, [])

        merged = merge_ab_reports([full, partial])
        dropped = merged[SIDE_A_KEY]["metrics"][
            f"tritonbench_{OP_NAME}_fwd[x_1024-torch]"
        ]
        self.assertEqual(len(dropped["latency"]), 2)
        self.assertIsNone(dropped["latency"][1])

    def test_written_json_is_valid_and_round_trips(self):
        # parse_constant fires on NaN/Infinity, which are not valid JSON.
        def _reject(constant):
            raise AssertionError(f"invalid JSON constant: {constant}")

        report = merge_ab_reports(
            [compare_ab_results(_make_result(0, 1.0), _make_result(100, 1.2), [], [])]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ab.json"
            write_ab_json(str(path), report)
            with open(path) as f:
                loaded = json.load(f, parse_constant=_reject)
        self.assertEqual(loaded, report)


class AbPowerDirTest(unittest.TestCase):
    """The per-side, per-repeat directories a --power-chart A/B run writes to."""

    def test_layout_is_side_then_repeat(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = run_utils._ab_power_dirs(root, None, 1, has_side_b=True)
            self.assertEqual(
                dirs,
                {
                    "a": os.path.join(root, "side_a_repeat_1"),
                    "b": os.path.join(root, "side_b_repeat_1"),
                },
            )
            # The directories exist: the power manager and the per-op json dump
            # both write into them without creating them first.
            for path in dirs.values():
                self.assertTrue(Path(path).is_dir())

    def test_single_side_and_multi_mode(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = run_utils._ab_power_dirs(root, "bwd", 7, has_side_b=False)
            self.assertEqual(dirs, {"a": os.path.join(root, "bwd", "side_a_repeat_7")})

    def test_no_power_chart_leaves_output_dir_alone(self):
        self.assertIsNone(run_utils._ab_power_dirs(None, None, 1, has_side_b=True))


if __name__ == "__main__":
    unittest.main()
