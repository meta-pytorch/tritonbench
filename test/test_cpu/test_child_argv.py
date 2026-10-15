"""Child benchmark processes must not inherit the parent's script path."""

import argparse
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import tritonbench.utils.run_utils as run_utils

SCRIPT = "/path/to/tritonbench/run.py"


class ChildArgvTest(unittest.TestCase):
    def test_run_in_task_drops_script_path(self):
        argv = [SCRIPT, "--device", "cpu", "--op", "layer_norm,softmax"]
        with (
            patch.object(sys, "argv", argv),
            patch.object(run_utils.subprocess, "check_call") as check_call,
        ):
            run_utils.run_in_task("softmax")
        cmd = check_call.call_args.args[0]
        self.assertEqual(
            cmd, [sys.executable, "run.py", "--device", "cpu", "--op", "softmax"]
        )

    def test_run_multi_device_drops_script_path(self):
        argv = [SCRIPT, "--op", "softmax", "--devices", "0,1", "--shard-by-inputs"]
        args = argparse.Namespace(
            op="softmax",
            devices="0,1",
            shard_by_inputs=True,
            csv=False,
            output=None,
            output_json=None,
            output_dir=None,
        )
        proc = MagicMock(returncode=0)
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(sys, "argv", argv),
            patch("tritonbench.utils.device_utils.validate_device_ids"),
            patch.object(
                run_utils, "_get_total_available_inputs", return_value=(0, 20)
            ),
            patch.object(run_utils.tempfile, "mkdtemp", return_value=tmp_dir),
            patch.object(run_utils.subprocess, "Popen", return_value=proc) as popen,
        ):
            run_utils.run_multi_device(args, [])
        self.assertEqual(popen.call_count, 2)
        for call in popen.call_args_list:
            cmd = call.args[0]
            self.assertEqual(cmd[:4], [sys.executable, "run.py", "--op", "softmax"])
            self.assertNotIn(SCRIPT, cmd)


if __name__ == "__main__":
    unittest.main()
