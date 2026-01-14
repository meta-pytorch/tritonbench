"""
Log benchmark results to scuba table (Requires Scuba token stored in TRITONBENCH_SCRIBE_GRAPHQL_ACCESS_TOKEN)
"""

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests
from tritonbench.utils.gpu_utils import get_nvidia_gpu_states, has_nvidia_smi
from tritonbench.utils.path_utils import REPO_PATH

CATEGORY_NAME = "perfpipe_pytorch_user_benchmarks"

BENCHMARK_SCHEMA = {
    "int": ["time"],
    "normal": [
        "benchmark_date",
        "unix_user",
        "submission_group_id",
        "cuda_version",
        "device",
        "conda_env",
        "pytorch_commit",
        "triton_commit",
        "tritonbench_commit",
        "triton_branch",
        "pytorch_branch",
        "tritonbench_branch",
        "triton_commit_time",
        "pytorch_commit_time",
        "tritonbench_commit_time",
        "github_action",
        "github_actor",
        "github_base_ref",
        "github_ref",
        "github_ref_protected",
        "github_repository",
        "github_run_attempt",
        "github_run_id",
        "github_run_number",
        "github_workflow",
        "github_workflow_ref",
        "github_workflow_sha",
        "job_name",
        "runner_arch",
        "runner_name",
        "runner_type",
        "runner_os",
        "metric_id",
    ],
    "float": ["metric_value"],
}


def get_github_env() -> Dict[str, str]:
    if "GITHUB_RUN_ID" not in os.environ:
        return {}
    out = {}
    out["GITHUB_ACTION"] = os.environ["GITHUB_ACTION"]
    out["GITHUB_ACTOR"] = os.environ["GITHUB_ACTOR"]
    out["GITHUB_BASE_REF"] = os.environ["GITHUB_BASE_REF"]
    out["GITHUB_REF"] = os.environ["GITHUB_REF"]
    out["GITHUB_REF_PROTECTED"] = os.environ["GITHUB_REF_PROTECTED"]
    out["GITHUB_REPOSITORY"] = os.environ["GITHUB_REPOSITORY"]
    out["GITHUB_RUN_ATTEMPT"] = os.environ["GITHUB_RUN_ATTEMPT"]
    out["GITHUB_RUN_ID"] = os.environ["GITHUB_RUN_ID"]
    out["GITHUB_RUN_NUMBER"] = os.environ["GITHUB_RUN_NUMBER"]
    out["GITHUB_WORKFLOW"] = os.environ["GITHUB_WORKFLOW"]
    out["GITHUB_WORKFLOW_REF"] = os.environ["GITHUB_WORKFLOW_REF"]
    out["GITHUB_WORKFLOW_SHA"] = os.environ["GITHUB_WORKFLOW_SHA"]
    out["JOB_NAME"] = os.environ["JOB_NAME"]
    out["RUNNER_ARCH"] = os.environ["RUNNER_ARCH"]
    out["RUNNER_TYPE"] = os.environ["RUNNER_TYPE"]
    out["RUNNER_NAME"] = os.environ["RUNNER_NAME"]
    out["RUNNER_OS"] = os.environ["RUNNER_OS"]
    return out


class ScribeUploader:
    def __init__(self, category, schema):
        self.category = category
        self.schema = schema

    def _format_message(self, field_dict):
        assert "time" in field_dict, "Missing required Scribe field 'time'"
        message = defaultdict(dict)
        for field, value in field_dict.items():
            field = field.lower()
            if value is None:
                continue
            if field in self.schema["normal"]:
                message["normal"][field] = str(value)
            elif field in self.schema["int"]:
                message["int"][field] = int(value)
            elif field in self.schema["float"]:
                try:
                    message["float"][field] = float(value)
                except ValueError:
                    # If value error (e.g., "CUDA OOM"), override the field value to 0.0
                    message["float"][field] = 0.0
            else:
                raise ValueError(
                    "Field {} is not currently used, "
                    "be intentional about adding new fields to schema".format(field)
                )
        return message

    def _upload(self, messages: list):
        access_token = os.environ.get("TRITONBENCH_SCRIBE_GRAPHQL_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("Can't find access token from environment variable")
        url = "https://graph.facebook.com/scribe_logs"
        r = requests.post(
            url,
            data={
                "access_token": access_token,
                "logs": json.dumps(
                    [
                        {
                            "category": self.category,
                            "message": json.dumps(message),
                            "line_escape": False,
                        }
                        for message in messages
                    ]
                ),
            },
        )
        print(r.text)
        r.raise_for_status()

    def post_benchmark_results(self, bm_data):
        messages = []
        base_message = {
            "time": int(time.time()),
        }
        base_message.update(bm_data["env"])
        base_message.update(bm_data.get("github", {}))
        base_message["submission_group_id"] = f"tritonbench.{bm_data['name']}"
        base_message["unix_user"] = bm_data["env"].get("unix_user", "tritonbench_ci")
        for metric in bm_data["metrics"]:
            msg = base_message.copy()
            msg["metric_id"] = metric
            msg["metric_value"] = bm_data["metrics"][metric]
            formatted_msg = self._format_message(msg)
            messages.append(formatted_msg)
        self._upload(messages)


def decorate_benchmark_data(
    name, run_timestamp, ci: bool, benchmark_data: List[Dict[str, Any]]
):
    """aggregate benchmark_data into a single object"""
    from tritonbench.utils.run_utils import get_run_env

    repo_locs = {
        "tritonbench": REPO_PATH,
        "triton": os.environ.get("TRITONBENCH_TRITON_INSTALL_DIR", "unknown"),
        "pytorch": os.environ.get("TRITONBENCH_PYTORCH_REPO_PATH", "unknown"),
    }
    aggregated_obj = {
        "name": name,
        "env": get_run_env(run_timestamp, repo_locs),
        "metrics": {},
    }
    if has_nvidia_smi():
        aggregated_obj.update(
            {
                "nvidia_gpu_states": get_nvidia_gpu_states(),
            }
        )

    # Collecting GitHub environment variables when running in CI environment
    if ci:
        aggregated_obj["github"] = get_github_env()
    else:
        aggregated_obj["env"]["unix_user"] = os.environ.get("USER", "unknown")

    for data in benchmark_data:
        aggregated_obj["metrics"].update(data)

    return aggregated_obj


def log_benchmark(
    benchmark_data, run_timestamp: Optional[str] = None, opbench: Optional[Any] = None
):
    if opbench:
        assert benchmark_data is None, (
            "Only one of opbench or benchmark_data can be specified"
        )
        benchmark_data = decorate_benchmark_data(
            name=opbench.logging_group
            if opbench.logging_group
            else opbench.benchmark_name,
            run_timestamp=run_timestamp,
            ci=False,
            benchmark_data=[opbench.output.userbenchmark_dict],
        )
    uploader = ScribeUploader(category=CATEGORY_NAME, schema=BENCHMARK_SCHEMA)
    uploader.post_benchmark_results(benchmark_data)
