"""
Upload result json file to scribe.
"""

import argparse
import json
import os
import sys
from os.path import abspath, exists


def setup_tritonbench_cwd():
    original_dir = abspath(os.getcwd())

    for tritonbench_dir in (
        ".",
        "../../tritonbench",
    ):
        if exists(tritonbench_dir):
            break

    if exists(tritonbench_dir):
        tritonbench_dir = abspath(tritonbench_dir)
        os.chdir(tritonbench_dir)
        sys.path.append(tritonbench_dir)
    return original_dir


setup_tritonbench_cwd()
from tritonbench.utils.scuba_utils import log_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", required=True, type=argparse.FileType("r"), help="Userbenchmark json"
    )
    args = parser.parse_args()
    benchmark_data = json.load(args.json)
    log_benchmark(benchmark_data)
