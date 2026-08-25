"""A KeyboardInterrupt must end an A/B run, not roll on to the next repeat."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import tritonbench.utils.run_utils as run_utils


def _fake_report(latency: float):
    return {"side-a": {"config": "", "metrics": {"cell": {"latency": latency}}}}


class AbTestInterruptTest(unittest.TestCase):
    def _run(self, interrupt_on: int, repeats: int, output_json: str):
        """Run the A/B loop with the Nth repeat raising KeyboardInterrupt."""
        started = []

        def fake_run_ab_test(args, extra_args, run_func):
            started.append(len(started) + 1)
            if len(started) == interrupt_on:
                raise KeyboardInterrupt
            return MagicMock(), None

        with (
            patch.object(run_utils, "run_ab_test", fake_run_ab_test),
            patch.object(
                run_utils,
                "compare_ab_results",
                lambda *a, **k: _fake_report(float(len(started))),
            ),
            patch.object(sys, "argv", ["run.py"]),
        ):
            with self.assertRaises(SystemExit) as caught:
                run_utils.tritonbench_run(
                    [
                        "--op",
                        "vector_add",
                        "--side-a=",
                        "--ab-repeat",
                        str(repeats),
                        "--output-json",
                        output_json,
                    ]
                )
        return started, caught.exception

    def test_interrupt_stops_the_remaining_repeats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ab.json")
            started, exc = self._run(interrupt_on=3, repeats=10, output_json=path)

            # Repeats 4..10 never start.
            self.assertEqual(started, [1, 2, 3])
            self.assertEqual(exc.code, 1)

            # The two repeats that did finish are still reported.
            with open(path) as f:
                report = json.load(f)
            self.assertEqual(report["side-a"]["metrics"]["cell"]["latency"], [1.0, 2.0])

    def test_interrupt_on_the_first_repeat_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ab.json")
            started, exc = self._run(interrupt_on=1, repeats=4, output_json=path)
            self.assertEqual(started, [1])
            self.assertEqual(exc.code, 1)
            self.assertFalse(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
