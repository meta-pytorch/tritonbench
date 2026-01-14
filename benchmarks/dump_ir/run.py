"""
Run all available custom Triton operators and save their IRs to a directory.
For autotuned operators, we save the IRs of the best kernels.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

from libfb.py import parutil
from tritonbench.operators import list_custom_triton_operators
from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.run_utils import run_in_task, run_one_operator

METADATA_DIR = (
    parutil.get_file_path("tritonbench/metadata")
    if is_fbcode()
    else Path(__file__).parent.parent.parent.joinpath("tritonbench/metadata")
)

OSS_CUSTOM_TRITON_YAML = os.path.join(METADATA_DIR, "oss_triton_operators.yaml")
INTERNAL_CUSTOM_TRITON_YAML = os.path.join(
    METADATA_DIR, "fb/internal_triton_operators.yaml"
)


def get_parser():
    parser = argparse.ArgumentParser(description="Dump Triton IRs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory to save the IRs",
    )
    parser.add_argument(
        "--run-in-task", action="store_true", help="indicate running in task."
    )
    return parser


def run_operator(op: str, subop: List[str], output_dir: str):
    """Run a Tritonbench operator and save its IR to the specified directory"""
    opbench_args = [
        "--run-in-task",
        "--op",
        op,
        "--only",
        ",".join(subop),
        "--dump-ir",
        output_dir,
    ]
    run_in_task(op, opbench_args)


if __name__ == "__main__":
    parser = get_parser()
    args, extra_args = parser.parse_known_args()
    if args.run_in_task:
        run_one_operator(extra_args, with_bwd=True)
        exit(0)
    custom_triton_op_yamls = (
        [OSS_CUSTOM_TRITON_YAML, INTERNAL_CUSTOM_TRITON_YAML]
        if is_fbcode()
        else [OSS_CUSTOM_TRITON_YAML]
    )
    operators: Dict[str, List[str]] = list_custom_triton_operators(
        custom_triton_op_yamls
    )
    [run_operator(op, operators[op].keys(), args.output_dir) for op in operators]
    print(f"[tritonbench][dump_ir] Result saved to {args.output_dir}")
