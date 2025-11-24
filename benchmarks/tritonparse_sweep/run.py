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
from tritonparse.reproducer.orchestrator import reproduce as tritonparse_reproduce
from tritonparse.reproducer.types import KernelImportMode
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
    parser.add_argument(
        "--op", type=str, help="Operator to benchmark and apply tritonparse."
    )
    parser.add_argument(
        "--reproduce",
        type=str,
        default=None,
        help="Reproduce the results from a previous run.",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run the tritonparse reproducing script.",
    )
    return parser


def find_ndjson_files(log_dir):
    ndjson_files = []
    for root, dirs, files in os.walk(log_dir):
        for name in files:
            if name.endswith(".ndjson"):
                ndjson_files.append(os.path.abspath(os.path.join(root, name)))
    return ndjson_files


def find_reproducer_script(output: str):
    output_line: list[str] = [ x for x in output.splitlines() if "repro_script" in x ]
    if len(output_line) == 0:
        return None
    output_line = output_line[0][output_line[0].find("{"):].strip()
    output_dict = eval(output_line)
    return output_dict['repro_script']


def run_repro_script(repro_script):
    cmd = [sys.executable, repro_script]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    args = get_parser().parse_args()
    triton_workloads = list_operators_by_collection("triton")
    run_timestamp, output_dir = setup_output_dir("tritonparse_sweep", ci=False)
    # Run the reproducer mode
    if args.reproduce:
        directory = args.reproduce
        tritonparse_dir = os.path.dirname(os.path.dirname(tritonparse.__file__))
        ndjson_files = find_ndjson_files(args.reproduce)
        result = {}
        print("Found", len(ndjson_files), "ndjson files in", args.reproduce)
        for ndjson_id, ndjson_file in enumerate(ndjson_files):
            result[ndjson_file] = "success"
            ndjson_dir = os.path.dirname(ndjson_file)
            repro_dir = os.path.join(ndjson_dir, "repro_tritonbench")
            if os.path.exists(repro_dir):
                shutil.rmtree(repro_dir)
            try:
                reproducer_output = tritonparse_reproduce(
                    input_path=ndjson_file,
                    line_index=2,
                    out_dir=repro_dir,
                    template="tritonbench",
                    kernel_import=KernelImportMode.COPY,
                )
            except Exception as e:
                result[ndjson_file] = "fail-gen-repro"
                continue
            if args.test_run:
                try:
                    run_repro_script(reproducer_output["repro_script"])
                except subprocess.CalledProcessError as e:
                    result[ndjson_file] = "fail-run-repro"
                    continue
            # reproduce without tritonbench
            repro_dir = os.path.join(ndjson_dir, "repro_kernel")
            if os.path.exists(repro_dir):
                shutil.rmtree(repro_dir)
            reproducer_output = tritonparse_reproduce(
                input_path=ndjson_file,
                line_index=2,
                out_dir=repro_dir,
                template="example",
                kernel_import=KernelImportMode.COPY,
            )
            if args.test_run:
                try:
                    run_repro_script(reproducer_output["repro_script"])
                except subprocess.CalledProcessError as e:
                    result[ndjson_file] = "fail-run-repro"
                    continue
        print(result)
        sys.exit(0)
    # Run the tracing mode
    if args.op:
        triton_workloads = {args.op: triton_workloads[args.op]}

    for op in triton_workloads:
        for backend in triton_workloads[op]:
            print("Running tritonparse on", op, "with backend: ", backend)
            run_tritonparse(op, backend, output_dir)
