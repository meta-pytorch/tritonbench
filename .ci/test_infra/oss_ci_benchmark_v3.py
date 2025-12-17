"""
Convert Tritonbench json to ClickHouse oss_ci_benchmark_v3 schema.
https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_benchmark_v3/schema.sql
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from os.path import abspath, exists
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

RUNNER_TYPE_MAPPING = {
    "gcp-h100-runner": {
        "name": "gcp-h100-runner",
        "gpu_count": 1,
        "avail_gpu_mem_in_gb": 80,
    },
    "amd-mi350-runner": {
        "name": "amd-mi350-runner",
        "gpu_count": 1,
        "avail_gpu_mem_in_gb": 288,
    },
    "linux.dgx.b200": {
        "name": "linux.dgx.b200",
        "gpu_count": 1,
        "avail_gpu_mem_in_gb": 192,
    },
}

DTYPE_PREFIXES = ["fp16", "fp32", "bf16", "int8", "int4", "fp8"]
BUILTIN_DTYPE_PREFIXES = ["fp8", "int8", "int4"]


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

from tritonbench.utils.scuba_utils import get_github_env


def parse_runners(
    runner_name: str, runner_type: str, envs: Dict[str, str]
) -> List[Dict[str, Any]]:
    runner_mapping = RUNNER_TYPE_MAPPING.get(runner_type, {}).copy()
    runner_mapping["name"] = runner_name
    runner_mapping["gpu_info"] = envs["device"]
    runner_mapping["extra_info"] = {}
    runner_mapping["extra_info"]["cuda_version"] = envs["cuda_version"]
    return [runner_mapping]


def parse_dependencies(envs: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    dependencies = {
        "pytorch": "pytorch/pytorch",
        "triton": "triton-lang/triton",
        "tritonbench": "meta-pytorch/tritonbench",
    }
    out = {}
    for dep in dependencies:
        out[dep] = {}
        out[dep]["repo"] = dependencies[dep]
        out[dep]["branch"] = envs[f"{dep}_branch"]
        out[dep]["sha"] = envs[f"{dep}_commit"]
        out[dep]["extra_info"] = {}
        out[dep]["extra_info"]["commit_time"] = envs[f"{dep}_commit_time"]
    return out


@dataclass
class TritonBenchMetricRow:
    op: str
    mode: str
    metric_name: str
    dtype: str = "unknown"
    backend: Optional[str] = None
    input: Optional[str] = None
    aggregation: Optional[str] = None


def get_dtype_from_op(op: str) -> Tuple[str, str]:
    for dtype_prefix in DTYPE_PREFIXES:
        if op.startswith(f"{dtype_prefix}_"):
            if dtype_prefix in BUILTIN_DTYPE_PREFIXES:
                return dtype_prefix, op
            else:
                return dtype_prefix, op[len(dtype_prefix) + 1 :]
    return "unknown", op


def parse_metric_id(metric_id: str) -> TritonBenchMetricRow:
    print(metric_id)
    # per-input metric
    if "[x_" in metric_id:
        # ignore x_average input rows
        if "[x_average" in metric_id:
            return None
        metric_id_regex = (
            r"tritonbench_([0-9a-z_]+)_([a-z_]+)\[x_(.*)-([0-9a-z_]+)\]_([a-z_]+)"
        )
        op, mode, input, backend, metric = re.match(metric_id_regex, metric_id).groups()
        dtype, op = get_dtype_from_op(op)
        assert not op.startswith("_"), f"Invalid op {op} with dtype {dtype}."
        # by default, aggregation for latency is p50
        aggregation = "p50" if metric == "latency" else None
        # individual input metric signal
        return TritonBenchMetricRow(
            op=op,
            mode=mode,
            metric_name=metric,
            dtype=dtype,
            backend=backend,
            input=input.strip(),
            aggregation=aggregation,
        )
    elif metric_id.endswith("-pass"):  # pass/fail metric
        metric_id_regex = r"tritonbench_([0-9a-z_]+)_([a-z_]+)-pass"
        op, mode = re.match(metric_id_regex, metric_id).groups()
        dtype, op = get_dtype_from_op(op)
        if not mode == "fwd" and not mode == "bwd":
            op = f"{op}_{mode}"
            mode = "fwd"
        # benchmark pass/fail signal
        return TritonBenchMetricRow(
            op=op,
            mode=mode,
            metric_name="pass",
            dtype=dtype,
            backend=None,
        )
    # aggregated metric
    input = None
    metric_id_regex = r"tritonbench_([0-9a-z_]+)_([a-z_]+)\[([0-9a-z_]+)\]-(.+)"
    op, mode, backend, metric = re.search(metric_id_regex, metric_id).groups()
    dtype, op = get_dtype_from_op(op)
    return TritonBenchMetricRow(
        op=op,
        mode=mode,
        metric_name=metric,
        dtype=dtype,
        backend=backend,
    )


def generate_oss_ci_benchmark_v3_json(
    benchmark_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Parse Benchmark Json and return a list of entries
    """
    common = {}
    out = []
    for metric_id in benchmark_result["metrics"]:
        # bypass if the metric is a target value
        if metric_id.endswith("-target"):
            continue
        entry = common.copy()
        entry["runners"] = parse_runners(
            benchmark_result["github"]["RUNNER_NAME"],
            benchmark_result["github"]["RUNNER_TYPE"],
            benchmark_result["env"],
        )
        entry["dependencies"] = parse_dependencies(benchmark_result["env"])
        metric_row: TritonBenchMetricRow = parse_metric_id(metric_id)
        if metric_row is None:
            continue
        try:
            metric_value = benchmark_result["metrics"][metric_id]
            metric_value = float(metric_value) if metric_value else 0.0
        except ValueError:
            # If value error (e.g., "CUDA OOM"), override the field value to 0.0
            metric_value = 0.0
        entry["benchmark"] = {
            "name": benchmark_result["name"],
            "mode": metric_row.mode,
            "dtype": metric_row.dtype,
            "extra_info": {},
        }
        # We use the model field for operator
        entry["model"] = {
            "name": metric_row.op,
            "type": "tritonbench-oss",
            "backend": metric_row.backend,
        }
        entry["metric"] = {
            "name": metric_row.metric_name,
            "benchmark_values": [metric_value],
            "extra_info": {},
        }
        # add input shape if applicable
        if metric_row.input:
            entry["metric"]["extra_info"]["input_shape"] = metric_row.input
        # add aggregation if applicable
        if metric_row.aggregation:
            entry["metric"]["extra_info"]["aggregation"] = metric_row.aggregation
        out.append(entry)
    return out


def v3_json_to_str(v3_json: List[Dict[str, Any]], to_lines: bool = True) -> str:
    if to_lines:
        entry_list = [json.dumps(entry) for entry in v3_json]
        return "\n".join(entry_list)
    else:
        return json.dumps(v3_json, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        required=True,
        help="Upload benchmark result json file.",
    )
    parser.add_argument(
        "--add-github-env",
        action="store_true",
        help="Add github env to the result json file.",
    )
    parser.add_argument("--output", required=True, help="output json.")
    args = parser.parse_args()
    upload_file_path = Path(args.json)
    assert (
        upload_file_path.exists()
    ), f"Specified result json path {args.json} does not exist."
    with open(upload_file_path, "r") as fp:
        benchmark_result = json.load(fp)
    if args.add_github_env:
        github_env = get_github_env()
        benchmark_result["github"] = github_env
        out_str = v3_json_to_str(benchmark_result, to_lines=False)
    else:
        oss_ci_v3_json = generate_oss_ci_benchmark_v3_json(benchmark_result)
        out_str = v3_json_to_str(oss_ci_v3_json)
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fp:
        fp.write(out_str)
    print(f"[oss_ci_benchmark_v3] Successfully saved to {args.output}")
