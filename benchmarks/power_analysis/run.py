"""
Perform power and performance analysis on a Triton kernel.
"""

import argparse
import logging
import os
import sys

import torch

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

REPCNT = 2000


workloads = [
    # gemm
    [
        "--op",
        "gemm",
        "--only",
        "aten_matmul,triton_tutorial_matmul,triton_blackwell_warpspec_persistent_matmul",
        "--m",
        "4096",
        "--n",
        "4096",
        "--k",
        "4096",
        "--repcnt",
        "2000",
        "--force",
    ],
    # blackwell_attention
    # rms norm
    [
        "--op",
        "rms_norm",
        "--only",
        "triton_tutorial_rms_norm",
        "--repcnt",
        "2000",
        "--force",
    ],
    #
]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repcnt", type=int, default=REPCNT)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    tb_args = [
        "--op",
        "gemm",
        "--num-inputs",
        "1",
        "--only",
        "triton_tutorial_matmul",
        "--repcnt",
        "2000",
        "--power-chart",
    ]
    opbench = load_operator_by_args(tb_args)
    opbench.run()
