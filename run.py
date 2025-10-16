"""
Tritonbench benchmark runner.

Note: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
      has been installed. This script intentionally does not automate or enforce setup steps.
"""
from tritonbench.utils.run_utils import tritonbench_run


if __name__ == "__main__":
    tritonbench_run()
