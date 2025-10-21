"""
Perform power and performance analysis on a Triton kernel.
"""

import argparse
import logging
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_tritonbench_cwd():
    original_dir = os.path.abspath(os.getcwd())

    for tritonbench_dir in (
        ".",
        "../../../tritonbench",
    ):
        if os.path.exists(tritonbench_dir):
            break

    if os.path.exists(tritonbench_dir):
        tritonbench_dir = os.path.abspath(tritonbench_dir)
        os.chdir(tritonbench_dir)
        sys.path.append(tritonbench_dir)
    return original_dir


setup_tritonbench_cwd()

from tritonbench.utils.run_utils import load_operator_by_args


if __name__ == "__main__":
    args = ["--op", "gemm", "--num-inputs", "1", "--only", "triton_tutorial_matmul"]
    opbench = load_operator_by_args(args)
    for x_val, inputs, bm_name, bm in opbench.run(ret_mode="yield"):
        print("x_val: ", x_val)
        print("bm_name", bm_name)
        bm()
