"""
Tritonbench benchmark runner.

Note: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
      has been installed. This script intentionally does not automate or enforce setup steps.
"""

import importlib
import sys
from typing import List, Optional

from tritonbench.utils.run_utils import tritonbench_run


def run(args: Optional[List[str]] = None):
    if args is None:
        args = sys.argv[1:]

    if "--launch" in args:
        launch_idx = args.index("--launch")
        if launch_idx + 1 >= len(args):
            raise ValueError("--launch requires a module path argument")
        launch_module = args[launch_idx + 1]
        extra_args = args[:launch_idx] + args[launch_idx + 2 :]
        module = importlib.import_module(launch_module)
        if not hasattr(module, "run"):
            raise ValueError(f"Module {launch_module} has no 'run' function")
        module.run(extra_args)
        return

    tritonbench_run(args)


if __name__ == "__main__":
    run()
