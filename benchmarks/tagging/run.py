"""
Trace op backends to generate tags.
"""

import argparse
import logging
import os
import sys
from os.path import abspath, exists
from pathlib import Path

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

from tritonbench.operators import list_operators
from tritonbench.utils.run_utils import load_operator_by_args
from tritonbench.utils.operator_utils import get_backends_for_operator
import inspect
from .ast_analyzer import build_backend_callees, trace_callees

def get_parser():
    parser = argparse.ArgumentParser(
        description="Trace op backends to generate tags."
    )
    parser.add_argument(
        "--op",
        type=str,
        help="Op name to trace. If unspecified, trace all ops.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output file path.",
    )
    return parser

def prevalidate_backends(backend_edges):
    op_with_tags = {}
    # heuristic: do not search torch.nn, torch.compile, and xformers backends
    for backend, callees in backend_edges.items():
        if "torch.compile" in callees or any(["torch._inductor" in callee for callee in callees]):
            op_with_tags[backend] = {"tags": ["pt2"]}
        elif any(["torch.nn" in callee for callee in callees]):
            op_with_tags[backend] = {"tags": ["aten"]}
        elif any(["xformers" in callee for callee in callees]):
            op_with_tags[backend] = {"tags": ["xformers"]}
    return op_with_tags


def trace_op(op):
    op_with_tags = {op: {}}
    opbench = load_operator_by_args(task_args=["--op", op])
    opbench_file = inspect.getfile(opbench.__class__)
    opbench_file_name = Path(opbench_file).name
    module_name = opbench.__module__
    with open(opbench_file, "r") as f:
        source = f.read()
    backends = get_backends_for_operator(opbench.name)
    backend_edges = build_backend_callees(
        source=source,
        filename=opbench_file_name,
        module_name=module_name,
        backends=backends,
    )
    assert len(backend_edges) == len(backends)
    op_with_tags[op] = prevalidate_backends(backend_edges)
    remaining_backends = [backend for backend in backends if backend not in op_with_tags[op]]
    # for backends without tags, we need to trace their callees to find tags
    # trace the callees of each backend, and return their tags
    for backend in remaining_backends:
        # special case for torch.compile
        callees = backend_edges[backend]
        base_module_name = module_name[:module_name.rfind(".")]
        callees_with_module = [(callee, base_module_name) for callee in callees]
        op_with_tags[op][backend] = trace_callees(callees_with_module)
        # postprocess: add human heuristics
        if "liger" in backend:
            if not op_with_tags[op][backend]:
                op_with_tags[op][backend] = {"tags": []}
            op_with_tags[op][backend]["tags"].append("liger")
        if "eager" in backend or "aten" in backend:
            if not op_with_tags[op][backend]:
                op_with_tags[op][backend] = {"tags": []}
            op_with_tags[op][backend]["tags"].append("aten")
    return op_with_tags


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if not args.op:
        ops = list_operators()
    else:
        ops = [args.op]
    results = {}
    for op in ops:
        results.update(trace_op(op))
    print(results)
