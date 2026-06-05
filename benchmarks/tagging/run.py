"""
Trace op backends to generate tags.
"""

import argparse
import gzip
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from ..common import setup_tritonbench_cwd

setup_tritonbench_cwd()

import inspect

from tritonbench.components.do_bench.run import _is_cache_clear_kernel
from tritonbench.operators import list_operators
from tritonbench.utils.operator_utils import get_backends_for_operator
from tritonbench.utils.run_utils import load_operator_by_args, run_in_task
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
        "--static-analysis-result",
        type=str,
        default="",
        help="Path to an existing static analysis result YAML file. Used as "
        "input when --bypass-run is set to skip recomputing static analysis.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output file path. With --kineto-validation, the validation "
        "results are written here. Without it, the static analysis results "
        "are written here. If unspecified, results are printed to stdout.",
    )
    parser.add_argument(
        "--kineto-validation",
        action="store_true",
        default=False,
        help="For each operator backend, run one input with kineto tracing "
        "and compare observed GPU kernel names against static analysis results.",
    )
    parser.add_argument(
        "--kineto-trace-dir",
        type=str,
        default=None,
        help="Directory to save kineto trace files when --kineto-validation is set.",
    )
    parser.add_argument(
        "--kineto-dir",
        type=str,
        default=None,
        help="Manifold URL (manifold://<bucket>/<path>) or local directory "
        "containing pre-collected kineto traces. Requires --kineto-validation. "
        "If a manifold URL, traces are downloaded to --kineto-trace-dir "
        "(default /tmp/kineto_traces). If a local directory, it is used "
        "directly as the trace dir. Existing traces are reused without re-running.",
    )
    parser.add_argument(
        "--bypass-run",
        action="store_true",
        default=False,
        help="When used with --kineto-validation, skip both static analysis "
        "and GPU trace collection. Static analysis results are loaded from "
        "the file specified by --static-analysis-result instead of being recomputed. Kineto "
        "traces that are not already available are skipped. If --kineto-dir "
        "is not set, defaults to "
        "manifold://tc_bench_ci/tree/tritonbench/kineto_traces.",
    )
    parser.add_argument(
        "--kineto-collect",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


_NON_KERNEL_EVENT_NAMES = frozenset(
    {
        "cudaMemcpyAsync",
        "cudaMemsetAsync",
        "cudaMemcpy",
        "cudaMemset",
        "Memcpy HtoD",
        "Memcpy DtoH",
        "Memcpy DtoD",
        "Memset",
    }
)


def extract_kernel_names(trace_data: Dict[str, Any]) -> List[str]:
    """Extract deduplicated GPU kernel names from a kineto trace dict."""
    events = trace_data.get("traceEvents", [])
    kernel_names: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        cat = event.get("cat", "")
        name = event.get("name", "")
        if (
            cat == "kernel"
            and name
            and name not in _NON_KERNEL_EVENT_NAMES
            and not _is_cache_clear_kernel(name)
        ):
            kernel_names.add(name)
    return sorted(kernel_names)


def parse_kineto_trace(trace_path: str, output_dir: str | None = None) -> List[str]:
    """Load a kineto trace from a local path and return GPU kernel names.

    Args:
        trace_path: Return value of ``opbench.kineto_trace()`` — a local file
            path or a URL.  Local ``.json`` and ``.json.gz`` files are read
            directly.
        output_dir: Fallback directory to scan for trace files when
            *trace_path* does not point to a readable file (e.g. when
            ``kineto_trace`` returns a remote URL).

    Returns:
        Sorted, deduplicated list of GPU kernel names.
    """
    trace_data = None

    # 1. Try trace_path as a local file.
    if os.path.isfile(trace_path):
        if trace_path.endswith(".gz"):
            with gzip.open(trace_path, "rt") as f:
                trace_data = json.load(f)
        else:
            with open(trace_path, "r") as f:
                trace_data = json.load(f)

    # 2. Fallback: scan output_dir for the latest trace file.
    if trace_data is None and output_dir and os.path.isdir(output_dir):
        candidates = sorted(
            Path(output_dir).glob("*.json*"),
            key=lambda p: p.stat().st_ctime,
            reverse=True,
        )
        for candidate in candidates:
            if candidate.name.endswith(".json.gz"):
                with gzip.open(candidate, "rt") as f:
                    trace_data = json.load(f)
                break
            elif candidate.name.endswith(".json"):
                with open(candidate, "r") as f:
                    trace_data = json.load(f)
                break

    if trace_data is None:
        logger.warning(f"Could not load kineto trace from {trace_path}")
        return []

    return extract_kernel_names(trace_data)


def _find_cached_trace(trace_dir: str, op_name: str, backend_name: str) -> str | None:
    """Return the path of an existing trace in the trace dir, or None."""
    trace_dir_path = Path(trace_dir)
    if not trace_dir_path.is_dir():
        return None
    prefix = f"{op_name}-{backend_name}"
    matching_dirs = sorted(
        (
            d
            for d in trace_dir_path.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ),
        key=lambda d: d.stat().st_ctime,
        reverse=True,
    )
    if not matching_dirs:
        return None
    candidates = sorted(
        matching_dirs[0].glob("*.json*"),
        key=lambda p: p.stat().st_ctime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def collect_kineto_kernel_names(
    op_name: str, backend_name: str, trace_dir: str | None = None
) -> List[str]:
    """Run one input of *backend_name* under kineto and return kernel names."""
    if trace_dir:
        cached = _find_cached_trace(trace_dir, op_name, backend_name)
        if cached:
            logger.info(
                f"Using cached kineto trace for {op_name}/{backend_name}: {cached}"
            )
            return parse_kineto_trace(cached)

    import torch
    from tritonbench.utils.input import input_cast

    opbench = load_operator_by_args(
        task_args=[
            "--op",
            op_name,
            "--num-inputs",
            "1",
            "--only",
            backend_name,
            "--warmup",
            "0",
            "--rep",
            "0",
        ]
    )
    opbench._cur_input_id = 0
    opbench.example_inputs = opbench.get_example_inputs()
    if opbench.example_inputs is None:
        logger.warning(f"No inputs available for {op_name}/{backend_name}. Skipping.")
        return []
    opbench.example_inputs = input_cast(
        lambda x: isinstance(x, torch.Tensor),
        lambda x: x.to(opbench.device),
        opbench.example_inputs,
    )
    fn = opbench._get_bm_func(backend_name)
    x_val = opbench.get_x_val(opbench.example_inputs)
    output_dir = None
    if trace_dir:
        output_dir = str(Path(trace_dir) / f"{op_name}-{backend_name}-x_{x_val}")
    trace_path = opbench.kineto_trace(input_id=0, fn=fn, output_dir=output_dir)
    return parse_kineto_trace(trace_path, output_dir=output_dir)


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
    decorator_tags = (
        backend_config.tags if (backend_config and backend_config.tags) else []
    )
    if decorator_tags:
        # Merge decorator tags with auto-detected tags (remove duplicates)
        all_tags = list(set(decorator_tags + tags_dict["tags"]))
        tags_dict["tags"] = all_tags

    return tags_dict


def prevalidate_backends(backend_edges, op_name=None, opbench_file=None):
    if opbench_file:
        from .ast_analyzer import _strip_link_tree_prefix

        opbench_file = _strip_link_tree_prefix(opbench_file)
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
            if opbench_file:
                op_with_tags[backend]["files"] = [opbench_file]
            if any(["fbgemm" in callee for callee in callees]):
                op_with_tags[backend]["tags"].append("fbgemm")
            if any(["mslk" in callee for callee in callees]):
                op_with_tags[backend]["tags"].append("mslk")

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
    try:
        opbench = load_operator_by_args(task_args=["--op", op])
    except Exception as e:
        logger.warning(f"Failed to load operator '{op}': {e}. Skipping.")
        return op_with_tags
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
    op_with_tags[op] = prevalidate_backends(
        backend_edges, op_name=op, opbench_file=opbench_file
    )
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


_PT2_TRITON_PREFIXES = (
    "triton_poi_",
    "triton_red_",
    "triton_tem_",
    "triton_per_",
    "triton_fused_",
)
_NON_TRITON_SUBSTRINGS = ("nvjet", "xmma", "cublas", "cublaslt", "cutlass", "cudnn_")


def _is_handwritten_triton_kernel(name: str) -> bool:
    if any(name.startswith(p) for p in _PT2_TRITON_PREFIXES):
        return False
    if any(s in name.lower() for s in _NON_TRITON_SUBSTRINGS):
        return False
    if name.startswith("void "):
        return False
    return True


def collect_kineto_kernel_names_subprocess(
    op_name: str,
    backend_name: str,
    trace_dir: str | None = None,
    bypass_run: bool = False,
) -> tuple[List[str], str]:
    """Run kineto collection in a separate process to isolate CUDA crashes."""
    if trace_dir:
        cached = _find_cached_trace(trace_dir, op_name, backend_name)
        if cached:
            logger.info(
                f"Using cached kineto trace for {op_name}/{backend_name}: {cached}"
            )
            return parse_kineto_trace(cached), "ok"

    if bypass_run:
        logger.info(
            f"No cached trace for {op_name}/{backend_name}, skipping (--bypass-run)"
        )
        return [], "skipped"

    fd, result_file = tempfile.mkstemp(suffix=".json", prefix="kineto_result_")
    os.close(fd)
    try:
        with tempfile.TemporaryDirectory(prefix="kineto_log_") as log_dir:
            op_args = [
                "--op",
                op_name,
                "--only",
                backend_name,
                "--kineto-collect",
                result_file,
            ]
            if trace_dir:
                op_args += ["--kineto-trace-dir", trace_dir]
            returncode = run_in_task(
                op_args=op_args,
                benchmark_name=f"kineto_{op_name}_{backend_name}",
                capture_output=log_dir,
                timeout_s=300,
            )
        if returncode != 0:
            logger.warning(
                f"Kineto subprocess crashed for {op_name}/{backend_name} "
                f"(exit code {returncode})"
            )
            if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                with open(result_file, "r") as f:
                    data = json.load(f)
                return data.get("kernels", []), "error"
            return [], "error"
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            with open(result_file, "r") as f:
                data = json.load(f)
            if data.get("status") == "error":
                logger.warning(
                    f"Kineto error for {op_name}/{backend_name}: "
                    f"{data.get('error', 'unknown')}"
                )
            return data.get("kernels", []), data.get("status", "error")
        return [], "error"
    finally:
        if os.path.exists(result_file):
            os.unlink(result_file)


def run_kineto_validation(
    ops, results, kineto_trace_dir=None, output_file=None, bypass_run=False
):
    """For each op/backend, collect kineto kernel names and compare with
    static analysis results.  When *output_file* is set the yaml is flushed
    after every op so partial results survive crashes."""
    validation = {}
    for op in ops:
        op_results = results.get(op, {})
        validation[op] = {}
        for backend, static_info in op_results.items():
            static_kernels = static_info.get("kernels", []) if static_info else []
            kineto_kernels, status = collect_kineto_kernel_names_subprocess(
                op, backend, trace_dir=kineto_trace_dir, bypass_run=bypass_run
            )
            handwritten = [
                k for k in kineto_kernels if _is_handwritten_triton_kernel(k)
            ]
            if status == "skipped":
                status = "kineto_missing"
            elif status == "ok" and handwritten:
                if not set(handwritten).issubset(set(static_kernels)):
                    status = "mismatch"
            print(
                f"[{op}/{backend}] status={status} "
                f"static_kernels={static_kernels} "
                f"kineto_kernels={kineto_kernels} "
                f"handwritten_triton={handwritten}"
            )
            if handwritten or status == "kineto_missing":
                validation[op][backend] = {
                    "static_kernels": static_kernels,
                    "kineto_kernels": handwritten,
                    "status": status,
                }
            if output_file:
                to_write = {k: v for k, v in validation.items() if v}
                with open(output_file, "w") as f:
                    f.write(yaml.safe_dump(to_write))
        if not validation[op]:
            del validation[op]
    return validation


def _download_manifold_traces(manifold_url: str, local_dir: str) -> None:
    """Download kineto traces from a manifold URL to a local directory."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    url = manifold_url.removeprefix("manifold://")
    cmd = ["manifold", "getr", url, local_dir, "--skip_root_dir"]
    logger.info(f"Downloading kineto traces: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    logger.info(f"Downloaded kineto traces to {local_dir}")


_DEFAULT_MANIFOLD_TRACE_URL = "manifold://tc_bench_ci/tree/tritonbench/kineto_traces"


def _resolve_kineto_dir(args) -> str | None:
    """Resolve --kineto-dir and --kineto-trace-dir into the local trace dir."""
    kineto_dir = args.kineto_dir
    if not kineto_dir and args.bypass_run:
        kineto_dir = _DEFAULT_MANIFOLD_TRACE_URL
    if kineto_dir:
        if kineto_dir.startswith("manifold://"):
            local_dir = args.kineto_trace_dir or "/tmp/kineto_traces"
            if not os.path.isdir(local_dir) or not any(Path(local_dir).iterdir()):
                _download_manifold_traces(kineto_dir, local_dir)
            else:
                logger.info(f"Using existing traces in {local_dir}")
            return local_dir
        else:
            return kineto_dir
    return args.kineto_trace_dir


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.kineto_collect:
        try:
            kernels = collect_kineto_kernel_names(
                args.op, args.only, trace_dir=args.kineto_trace_dir
            )
            with open(args.kineto_collect, "w") as f:
                json.dump({"kernels": kernels, "status": "ok"}, f)
        except Exception as e:
            with open(args.kineto_collect, "w") as f:
                json.dump({"kernels": [], "status": "error", "error": str(e)}, f)
        sys.exit(0)

    results = {}
    if args.bypass_run:
        if not args.static_analysis_result or not os.path.isfile(
            args.static_analysis_result
        ):
            parser.error(
                "--bypass-run requires --static-analysis-result pointing to an "
                "existing YAML file with static analysis results"
            )
        logger.info(
            f"Loading static analysis results from {args.static_analysis_result} (--bypass-run)"
        )
        with open(args.static_analysis_result, "r") as f:
            results = yaml.safe_load(f) or {}
        if args.op:
            ops = [args.op]
        else:
            ops = list(results.keys())
    else:
        if not args.op:
            ops = list_operators()
        else:
            ops = [args.op]
        for op in ops:
            results.update(trace_op(op))
    print(f"Running tagging test on ops: {ops}...")

    if args.kineto_validation:
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        kineto_trace_dir = _resolve_kineto_dir(args)
        validation = run_kineto_validation(
            ops,
            results,
            kineto_trace_dir=kineto_trace_dir,
            output_file=args.output,
            bypass_run=args.bypass_run,
        )
        if args.output:
            print("Kineto validation written to", args.output)
        else:
            print(yaml.safe_dump(validation))
    else:
        if not args.output:
            print(results)
        else:
            with open(args.output, "w") as f:
                f.write(yaml.safe_dump(results))
            print("success!")
