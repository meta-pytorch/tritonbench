"""
Trace op backends to generate tags.
"""

import argparse
import logging
import os
import sys
from os.path import abspath, exists
from pathlib import Path

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

import inspect

from tritonbench.operators import list_operators
from tritonbench.utils.operator_utils import get_backends_for_operator
from tritonbench.utils.run_utils import load_operator_by_args

try:
    from ast_analyzer import build_backend_callees, trace_callees
except ImportError:
    from .ast_analyzer import build_backend_callees, trace_callees


def get_parser():
    parser = argparse.ArgumentParser(description="Trace op backends to generate tags.")
    parser.add_argument(
        "--op",
        type=str,
        help="Op name to trace. If unspecified, trace all ops.",
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Only trace the specified backend. If unspecified, trace all backends.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output file path. If none, print to stdout.",
    )
    return parser


def prevalidate_backends(backend_edges):
    op_with_tags = {}
    # heuristic: do not search torch.nn, torch.compile, and xformers backends
    for backend, callees in backend_edges.items():
        if "torch.compile" in callees or any(
            ["torch._inductor" in callee for callee in callees]
        ):
            op_with_tags[backend] = {"tags": ["pt2"]}
        elif any(["torch.nn" in callee for callee in callees]):
            op_with_tags[backend] = {"tags": ["aten"]}
        elif any(["xformers" in callee for callee in callees]):
            op_with_tags[backend] = {"tags": ["xformers"]}
        elif any([callee.startswith("torch.ops.") for callee in callees]):
            custom_op_category = [
                callee[callee.rfind(".") + 1 :]
                for callee in callees
                if callee.startswith("torch.ops.")
            ]
            op_with_tags[backend] = {"tags": custom_op_category + ["native_custom_ops"]}

    return op_with_tags


def trace_op(op):
    op_with_tags = {op: {}}
    opbench = load_operator_by_args(task_args=["--op", op])
    opbench_file = inspect.getfile(opbench.__class__)
    opbench_file_name = Path(opbench_file).name
    module_name = opbench.__module__
    with open(opbench_file, "r") as f:
        source = f.read()
    backends = (
        get_backends_for_operator(opbench.name)
        if not args.only
        else args.only.split(",")
    )
    backend_edges = build_backend_callees(
        source=source,
        filename=opbench_file_name,
        module_name=module_name,
        backends=backends,
    )
    assert len(backend_edges) == len(backends)
    op_with_tags[op] = prevalidate_backends(backend_edges)
    remaining_backends = [
        backend for backend in backends if backend not in op_with_tags[op]
    ]
    # for backends without tags, we need to trace their callees to find tags
    # trace the callees of each backend, and return their tags
    for backend in remaining_backends:
        # special case for torch.compile
        callees = backend_edges[backend]
        base_module_name = module_name[: module_name.rfind(".")]
        callees_with_module: list[tuple[Unknown, Unknown]] = [
            (callee, base_module_name) for callee in callees
        ]
        op_with_tags[op][backend] = trace_callees(callees_with_module)
        # postprocess: add human heuristics
        if "liger" in backend:
            if not op_with_tags[op][backend]:
                op_with_tags[op][backend] = {"tags": []}
            op_with_tags[op][backend]["tags"].extend(["liger"])
            if "triton" not in op_with_tags[op][backend]["tags"]:
                op_with_tags[op][backend]["tags"].append("triton")
        if "tlx_" in backend:
            if not op_with_tags[op][backend]:
                op_with_tags[op][backend] = {"tags": []}
            op_with_tags[op][backend]["tags"].extend(["tlx"])
        if "eager" in backend or "aten" in backend:
            if not op_with_tags[op][backend]:
                op_with_tags[op][backend] = {"tags": []}
            op_with_tags[op][backend]["tags"].append("aten")
    return op_with_tags


UNSUPPORTED_OPS = [
    "fp8_fused_quant_gemm_rowwise",
    "fp32_to_mx4",
    "flex_attention",
    "mx4_to_fp32",
]

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if not args.op:
        ops = list_operators()
    else:
        ops = [args.op]
    print(f"Running tagging test on ops: {ops}...")
    results = {}
    for op in ops:
        # deadloop on flex_attention
        if op in UNSUPPORTED_OPS:
            continue
        results.update(trace_op(op))
    if not args.output:
        print(results)
    else:
        with open(args.output, "w") as f:
            f.write(yaml.safe_dump(results))
        print("success!")
