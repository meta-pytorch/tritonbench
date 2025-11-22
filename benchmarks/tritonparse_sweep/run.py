"""
Validate TritonParse across all Triton kernels in TritonBench.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from os.path import abspath, exists
from pathlib import Path
from typing import Any, Dict

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_tritonbench_cwd():
    original_dir = abspath(os.getcwd())

    for tritonbench_dir in (
        ".",
        "../../../tritonbench",
    ):
        if exists(tritonbench_dir):
            break

    if exists(tritonbench_dir):
        tritonbench_dir = abspath(tritonbench_dir)
        os.chdir(tritonbench_dir)
        sys.path.append(tritonbench_dir)
    return original_dir

setup_tritonbench_cwd()

import tritonparse
from tritonbench.operators_collection import list_operators_by_collection
from tritonbench.utils.run_utils import run_in_task, setup_output_dir

NOT_WORKING_OPS = ["tritonparse_softmax_triton_softmax"]

def run_tritonparse(op: str, backend: str, output_dir: str):
    tritonparse_log_dir = os.path.join(output_dir, f"tritonparse_{op}_{backend}")
    run_args = [
        "--op",
        op,
        "--only",
        backend,
        "--num-inputs",
        "1",
        "--tritonparse",
        tritonparse_log_dir,
    ]
    run_in_task(op_args=run_args)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, help="Operator to benchmark and apply tritonparse.")
    parser.add_argument(
        "--reproduce",
        type=str,
        default=None,
        help="Reproduce the results from a previous run.",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run the tritonparse reproducing script."
    )
    return parser

def find_ndjson_files(log_dir):
    ndjson_files = []
    for root, dirs, files in os.walk(log_dir):
        for name in files:
            if name.endswith(".ndjson"):
                ndjson_files.append(os.path.abspath(os.path.join(root, name)))
    return ndjson_files

def find_reproducer_script(repro_dir):
    pass

def run_repro_script(repro_script):
    pass

if __name__ == "__main__":
    args = get_parser().parse_args()
    triton_workloads = list_operators_by_collection("triton")
    run_timestamp, output_dir = setup_output_dir("tritonparse_sweep", ci=False)
    # Run the reproducer mode
    if args.reproduce:
        directory = args.reproduce
        tritonparse_dir = os.path.dirname(os.path.dirname(tritonparse.__file__))
        ndjson_files = find_ndjson_files(args.reproduce)
        print("Found", len(ndjson_files), "ndjson files in", args.reproduce)
        for ndjson_file in ndjson_files:
            ndjson_dir = os.path.dirname(ndjson_file)
            repro_dir = os.path.join(ndjson_dir, "repro_tritonbench")
            if os.path.exists(repro_dir):
                shutil.rmtree(repro_dir)
            reproduce_cmd = [
                sys.executable,
                "run.py",
                "reproduce",
                "--out-dir",
                repro_dir,
                "--line",
                "2",
                "--template",
                "tritonbench",
                "--kernel-import",
                "copy",
                ndjson_file,
            ]
            subprocess.check_call(
                reproduce_cmd,
                cwd=tritonparse_dir,
            )
            if args.test_run:
                repro_script = find_reproducer_script(repro_dir)
                run_repro_script(repro_script)
            # reproduce without tritonbench
            repro_dir = os.path.join(ndjson_dir, "repro_kernel")
            if os.path.exists(repro_dir):
                shutil.rmtree(repro_dir)
            reproduce_cmd = [
                sys.executable,
                "run.py",
                "reproduce",
                "--out-dir",
                os.path.join(ndjson_dir, "repro_kernel"),
                "--line",
                "2",
                "--kernel-import",
                "copy",
                ndjson_file,
            ]
            subprocess.check_call(
                reproduce_cmd,
                cwd=tritonparse_dir,
            )
            if args.test_run:
                repro_script = find_reproducer_script(repro_dir)
                run_repro_script(repro_script)
    # Run the tracing mode
    if args.op:
        triton_workloads = {args.op: triton_workloads[args.op]}

    for op in triton_workloads:
        for backend in triton_workloads[op]:
            print("Running tritonparse on", op, "with backend: ", backend)
            run_tritonparse(op, backend, output_dir)
