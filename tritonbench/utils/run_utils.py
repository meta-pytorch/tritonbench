import copy
import logging
import os
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path

from typing import Dict, List, Optional

import yaml

from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.git_utils import get_branch, get_commit_time, get_current_hash
from tritonbench.utils.path_utils import (
    add_cmd_parameter,
    remove_cmd_parameter,
    REPO_PATH,
)

BENCHMARKS_OUTPUT_DIR = REPO_PATH.joinpath(".benchmarks")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_run_env(
    run_timestamp: str, repo_locs: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Gather environment of the benchmark.
    repo_locs: Git repository dict of the repositories.
    """
    import torch

    run_env = {}
    run_env["benchmark_date"] = run_timestamp
    run_env["cuda_version"] = torch.version.cuda if torch.version.cuda else "unknown"
    try:
        run_env["device"] = torch.cuda.get_device_name()
    except AssertionError:
        run_env["device"] = "unknown"
    run_env["conda_env"] = os.environ.get("CONDA_ENV", "unknown")
    run_env["pytorch_commit"] = torch.version.git_version
    # we assume Tritonbench CI will properly set Triton commit hash in env
    run_env["triton_commit"] = os.environ.get(
        "TRITONBENCH_TRITON_MAIN_COMMIT", "unknown"
    )
    run_env["tritonbench_commit"] = get_current_hash(REPO_PATH)
    for repo in ["triton", "pytorch", "tritonbench"]:
        repo_loc = repo_locs.get(repo, None)
        if not run_env[f"{repo}_commit"] == "unknown" and repo_loc:
            run_env[f"{repo}_branch"] = get_branch(repo_loc, run_env[f"{repo}_commit"])
            run_env[f"{repo}_commit_time"] = get_commit_time(
                repo_loc, run_env[f"{repo}_commit"]
            )
        else:
            run_env[f"{repo}_branch"] = "unknown"
            run_env[f"{repo}_commit_time"] = "unknown"
    return run_env


def get_github_env() -> Dict[str, str]:
    assert (
        "GITHUB_RUN_ID" in os.environ
    ), "GITHUB_RUN_ID environ must exist to obtain GitHub env"
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


def run_config(config_file: str):
    assert Path(config_file).exists(), f"Config file {config_file} must exist."
    with open(config_file, "r") as fp:
        config = yaml.safe_load(fp)
    for benchmark_name in config:
        benchmark_config = config[benchmark_name]
        op_name = benchmark_config["op"]
        op_args = benchmark_config["args"].split(" ")
        env_string = benchmark_config.get("envs", None)
        extra_envs = {}
        if env_string:
            for env_part in env_string.split(" "):
                key, val = env_part.split("=")
                extra_envs[key] = val
        run_in_task(
            op=op_name,
            op_args=op_args,
            benchmark_name=benchmark_name,
            extra_envs=extra_envs,
        )


def run_in_task(
    op: Optional[str],
    op_args: Optional[List[str]] = None,
    benchmark_name: Optional[str] = None,
    extra_envs: Optional[Dict[str, str]] = None,
) -> None:
    op_task_cmd = [] if is_fbcode() else [sys.executable]
    if not op_args:
        assert op, "If op_args is none, op must not be None."
        copy_sys_argv = copy.deepcopy(sys.argv)
        copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--op")
        copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--isolate")
        copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--op-collection")
        add_cmd_parameter(copy_sys_argv, "--op", op)
        op_task_cmd.extend(copy_sys_argv)
    else:
        op_task_cmd.extend(op_args)
    if benchmark_name:
        op_args.extend(["--benchmark-name", benchmark_name])
    else:
        benchmark_name = op

    # Remove "TRITONBENCH_RUN_CONFIG" env
    if "TRITONBENCH_RUN_CONFIG" in os.environ:
        del os.environ["TRITONBENCH_RUN_CONFIG"]

    # In OSS, we assume always using the run.py benchmark driver
    if not is_fbcode() and not op_task_cmd[1] == "run.py":
        op_task_cmd.insert(1, "run.py")
    try:
        print(
            f"[tritonbench] Running {benchmark_name}: " + " ".join(op_task_cmd),
            flush=True,
        )
        subprocess_env = os.environ.copy()
        subprocess_env.update(extra_envs or {})
        subprocess.check_call(
            op_task_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=REPO_PATH,
            env=subprocess_env,
        )
    except subprocess.CalledProcessError:
        # By default, we will continue on the failed operators
        pass
    except KeyboardInterrupt:
        logger.warning("[tritonbench] KeyboardInterrupt received, exiting...")
        sys.exit(1)


def setup_output_dir(bm_name: str):
    current_timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    output_dir = BENCHMARKS_OUTPUT_DIR.joinpath(bm_name, f"run-{current_timestamp}")
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    return current_timestamp, output_dir.absolute()
