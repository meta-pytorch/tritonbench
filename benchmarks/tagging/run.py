"""
Trace op backends to generate tags.
"""

import argparse
import logging
import os
import sys
from os.path import abspath, exists
from pathlib import Path
from typing import Any

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
from tritonbench.utils.triton_op import REGISTERED_BENCHMARKS

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


def apply_name_based_heuristics(backend_name, tags_dict):
    """
    Apply name-based heuristics to add tags based on backend name.

    Args:
        backend_name: The name of the backend
        tags_dict: Dictionary with 'tags' key, e.g., {"tags": ["pt2"]}
                   If None, will be created.

    Returns:
        Updated tags_dict
    """
    if not tags_dict:
        tags_dict = {"tags": []}
    if "tags" not in tags_dict:
        tags_dict["tags"] = []

    # Liger backends are based on Triton
    if "liger" in backend_name:
        if "liger" not in tags_dict["tags"]:
            tags_dict["tags"].append("liger")
        if "triton" not in tags_dict["tags"]:
            tags_dict["tags"].append("triton")

    # CUTLASS backends
    if "cutlass" in backend_name.lower():
        if "cutlass" not in tags_dict["tags"]:
            tags_dict["tags"].append("cutlass")

    # TLX backends
    if "tlx_" in backend_name:
        if "tlx" not in tags_dict["tags"]:
            tags_dict["tags"].append("tlx")

    # Eager/Aten backends
    if "eager" in backend_name or "aten" in backend_name:
        if "aten" not in tags_dict["tags"]:
            tags_dict["tags"].append("aten")

    return tags_dict


def merge_decorator_tags(op_name, backend_name, tags_dict):
    """
    Merge tags from @register_benchmark decorator with auto-detected tags.

    Args:
        op_name: The operator name
        backend_name: The backend name
        tags_dict: Dictionary with auto-detected tags, e.g., {"tags": ["pt2"]}
                   If None, will be created.

    Returns:
        Updated tags_dict with decorator tags merged
    """
    if not tags_dict:
        tags_dict = {"tags": []}
    if "tags" not in tags_dict:
        tags_dict["tags"] = []

    # Get decorator tags if they exist
    backend_config = REGISTERED_BENCHMARKS.get(op_name, {}).get(backend_name)
    decorator_tags = backend_config.tags if (backend_config and backend_config.tags) else []
    if decorator_tags:
        # Merge decorator tags with auto-detected tags (remove duplicates)
        all_tags = list(set(decorator_tags + tags_dict["tags"]))
        tags_dict["tags"] = all_tags

    return tags_dict


def prevalidate_backends(backend_edges, op_name=None):
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
                callee for callee in callees if callee.startswith("torch.ops.")
            ]
            op_with_tags[backend] = {
                "tags": ["native_custom_ops"],
                "kernels": custom_op_category,
            }
            if any(["fbgemm" in callee for callee in callees]):
                op_with_tags[backend]["tags"].append("fbgemm")

    # Apply name-based heuristics for all prevalidated backends
    for backend in op_with_tags.keys():
        op_with_tags[backend] = apply_name_based_heuristics(
            backend, op_with_tags[backend]
        )
        # Merge with decorator tags if available
        if op_name:
            op_with_tags[backend] = merge_decorator_tags(
                op_name, backend, op_with_tags[backend]
            )

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
    op_with_tags[op] = prevalidate_backends(backend_edges, op_name=op)
    remaining_backends = [
        backend for backend in backends if backend not in op_with_tags[op]
    ]
    # for backends without tags, we need to trace their callees to find tags
    # trace the callees of each backend, and return their tags
    for backend in remaining_backends:
        # special case for torch.compile
        callees = backend_edges[backend]
        callees_with_module: list[tuple[Any, Any]] = [
            (callee, module_name) for callee in callees
        ]
        op_with_tags[op][backend] = trace_callees(callees_with_module)
        # Apply name-based heuristics
        op_with_tags[op][backend] = apply_name_based_heuristics(
            backend, op_with_tags[op][backend]
        )
        # Merge with decorator tags
        op_with_tags[op][backend] = merge_decorator_tags(
            op, backend, op_with_tags[op][backend]
        )
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
