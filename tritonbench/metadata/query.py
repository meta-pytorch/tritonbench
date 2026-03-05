import os
from typing import Any, Dict, List, Optional

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

KERNEL_METADATA_PATH = os.path.join(CURRENT_DIR, "oss_cuda_kernels.yaml")
BACKWARD_METADATA_PATH = os.path.join(CURRENT_DIR, "backward_operators.yaml")
DTYPE_METADATA_PATH = os.path.join(CURRENT_DIR, "dtype_operators.yaml")
TFLOPS_OPS_PATH = os.path.join(CURRENT_DIR, "tflops_operators.yaml")
BASELINE_OPS_PATH: Dict[str, str] = os.path.join(CURRENT_DIR, "baseline_operators.yaml")

SKIP_DTYPE = ["bypass", "fp8", "int4", "bf16xint16"]


def load_metadata(metadata_path: str) -> Any:
    with open(metadata_path, "r") as f:
        return yaml.safe_load(f)


def _has_meaningful_baseline(
    op: str, backends: List[str], baseline_metadata: Dict[str, Any]
) -> bool:
    return op in baseline_metadata and not (
        baseline_metadata[op] in backends and len(backends) == 1
    )


def get_metric_args(op: str, backends: List[str], required_metrics: List[str]) -> str:
    special_metrics = ["flops", "tflops", "speedup"]
    valid_metrics = [m for m in required_metrics if m not in special_metrics]
    # do basic sanity checks
    # only add tflops/flops/speedup if the op supports it
    baseline_prefix = ""
    if "tflops" in required_metrics or "flops" in required_metrics:
        tflops_ops = load_metadata(TFLOPS_OPS_PATH)
        if op in tflops_ops:
            valid_metrics.append("tflops")
    if "speedup" in required_metrics:
        baseline_metadata = load_metadata(BASELINE_OPS_PATH)
        if _has_meaningful_baseline(op, backends, baseline_metadata):
            valid_metrics.append("speedup")
            baseline_prefix = f"--baseline {baseline_metadata[op]} "
    return baseline_prefix + "--metrics " + ",".join(valid_metrics)


def get_benchmark_dtype(op_name: str, runtime_dtype: str | None = None):
    if runtime_dtype:
        return runtime_dtype
    with open(DTYPE_METADATA_PATH, "r") as f:
        dtype_mapping = yaml.safe_load(f)
    return dtype_mapping[op_name]


def _merge_skip_tests(skip_tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged_skip_tests = {}
    for skip_test in skip_tests:
        for skip_test_op in skip_test:
            if skip_test_op not in merged_skip_tests:
                merged_skip_tests[skip_test_op] = skip_test[skip_test_op]
            else:
                merged_skip_tests[skip_test_op].update(skip_test[skip_test_op])
    return merged_skip_tests


def _update_benchmark_by_skip_tests(
    op: str, benchmark_config: Dict[str, Any], skip_tests: Dict[str, Any]
) -> Dict[str, Any]:
    if op not in skip_tests:
        return benchmark_config
    if "devices" in skip_tests[op]:
        benchmark_config["devices"] = skip_tests[op]["devices"].copy()
    if "channels" in skip_tests[op]:
        benchmark_config["channels"] = skip_tests[op]["channels"].copy()
    if "extra_args" in skip_tests[op]:
        benchmark_config["args"] += " " + skip_tests[op]["extra_args"]
    if "extra_bwd_args" in skip_tests[op] and "--bwd" in benchmark_config["args"]:
        benchmark_config["args"] += " " + skip_tests[op]["extra_bwd_args"]
    return benchmark_config


def _need_skip_by_skip_test(op: str, skip_tests: Dict[str, Any]) -> bool:
    if op in skip_tests and skip_tests[op] == None:
        return True
    return False


def get_benchmark_config_with_tags(
    tags: List[str],
    runtime_metadata: Optional[Dict[str, Any]] = None,
    runtime_only: bool = False,
    per_backend: bool = False,
    with_backwards: bool = True,
    metrics: Optional[List[str]] = None,
    skip_tests: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return benchmark config dict with any of these tags.
    runtime_metadata: runtime metadata to override the default metadata
    runtime_only: whether to only include runtime metadata, not built-in metadata
    with_backwards: whether to include backward run
    metrics: list of metrics to include in the benchmark config with best efforts
    """
    if runtime_only:
        operators = runtime_metadata
    else:
        with open(KERNEL_METADATA_PATH, "r") as f:
            operators = yaml.safe_load(f)
            if runtime_metadata is not None:
                for op in runtime_metadata:
                    if op not in operators:
                        operators.update({op: runtime_metadata[op]})
                    else:
                        operators[op].update(runtime_metadata[op])

    if with_backwards:
        with open(BACKWARD_METADATA_PATH, "r") as f:
            backwards = yaml.safe_load(f)
    with open(DTYPE_METADATA_PATH, "r") as f:
        dtype = yaml.safe_load(f)

    skip_tests = _merge_skip_tests(skip_tests or [])

    result_dict = {}
    for op, backend in operators.items():
        if _need_skip_by_skip_test(op, skip_tests):
            continue
        backend_names_with_tags = {
            b
            for b in backend
            if "tags" in backend[b] and any(t in backend[b]["tags"] for t in tags)
        }
        if not backend_names_with_tags:
            continue
        dtype_prefix = (
            f"{dtype[op]}_" if op in dtype and dtype[op] not in SKIP_DTYPE else ""
        )
        metric_args = (
            get_metric_args(op, backend_names_with_tags, metrics) if metrics else ""
        )
        if per_backend:
            for backend_name in backend_names_with_tags:
                benchmark_prefix = f"{dtype_prefix}{op}_{backend_name}"
                benchmark_name = f"{benchmark_prefix}_fwd"
                result_dict[benchmark_name] = {}
                result_dict[benchmark_name]["args"] = " ".join(
                    ["--op", op, "--only", backend_name]
                ) + [metric_args]
                result_dict[benchmark_name] = _update_benchmark_by_skip_tests(
                    op, result_dict[benchmark_name], skip_tests
                )
                if with_backwards and op in backwards:
                    benchmark_name = f"{benchmark_prefix}_bwd"
                    result_dict[benchmark_name] = {}
                    result_dict[benchmark_name]["args"] = " ".join(
                        ["--op", op, "--only", backend_name, "--bwd"] + [metric_args]
                    )
                    result_dict[benchmark_name] = _update_benchmark_by_skip_tests(
                        op, result_dict[benchmark_name], skip_tests
                    )
        else:
            benchmark_prefix = f"{dtype_prefix}{op}"
            benchmark_name = f"{benchmark_prefix}_fwd"
            result_dict[benchmark_name] = {}
            result_dict[benchmark_name]["args"] = " ".join(
                ["--op", op, "--only"]
                + [",".join(backend_names_with_tags)]
                + [metric_args]
            )
            result_dict[benchmark_name] = _update_benchmark_by_skip_tests(
                op, result_dict[benchmark_name], skip_tests
            )
            if with_backwards and op in backwards:
                benchmark_name = f"{benchmark_prefix}_bwd"
                result_dict[benchmark_name] = {}
                result_dict[benchmark_name]["args"] = " ".join(
                    ["--op", op, "--only"]
                    + [",".join(backend_names_with_tags), "--bwd"]
                    + [metric_args]
                )
                result_dict[benchmark_name] = _update_benchmark_by_skip_tests(
                    op, result_dict[benchmark_name], skip_tests
                )
    return result_dict
