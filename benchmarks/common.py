import json
import logging
import os
import sys
import time
from datetime import datetime
from os.path import abspath, exists
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

from tritonbench.utils.env_utils import (
    is_blackwell,
    is_h100,
    is_hip_mi300,
    is_hip_mi350,
)
from tritonbench.utils.path_utils import REPO_PATH
from tritonbench.utils.run_utils import run_in_task

BENCHMARKS_OUTPUT_DIR = REPO_PATH.joinpath(".benchmarks")

CI_SUPPORTED_DEVICES = {
    "b200": is_blackwell,
    "h100": is_h100,
    "mi300": is_hip_mi300,
    "mi350": is_hip_mi350,
}


def run_benchmark_config_ci(
    benchmark_group_name: str,
    benchmark_config_obj: Dict[str, Any],
    extra_args: List[str] | None = None,
    output_dir: str | None = None,
    op: str | None = None,
    ci: bool = False,
    log_scuba: bool = False,
):
    def _filter_benchmark_config_obj_by_device(benchmark_config_obj):
        ret = {}
        for benchmark in benchmark_config_obj:
            if (
                "disabled" in benchmark_config_obj[benchmark]
                and benchmark_config_obj[benchmark]["disabled"]
            ):
                continue
            if "device" in benchmark_config_obj[benchmark]:
                devices = benchmark_config_obj[benchmark]["device"].split()
                assert any(device in CI_SUPPORTED_DEVICES for device in devices), (
                    f"Found unsupported device: {devices}"
                )
                if any(CI_SUPPORTED_DEVICES[device]() for device in devices):
                    ret[benchmark] = benchmark_config_obj[benchmark]
                else:
                    print(f"skipping by device: {benchmark}")
            else:
                ret[benchmark] = benchmark_config_obj[benchmark]
        return ret

    from tritonbench.utils.scuba_utils import decorate_benchmark_data, log_benchmark

    output_files = []
    run_timestamp, output_dir = setup_output_dir(
        benchmark_group_name, ci=ci, output_dir=output_dir
    )
    benchmark_config_obj = _filter_benchmark_config_obj_by_device(benchmark_config_obj)
    for benchmark in benchmark_config_obj:
        if op and f"--op {op}" not in benchmark_config_obj[benchmark]["args"]:
            continue
        if isinstance(benchmark_config_obj[benchmark]["args"], str):
            op_args = benchmark_config_obj[benchmark]["args"].split(" ")
        elif isinstance(benchmark_config_obj[benchmark]["args"], List):
            op_args = benchmark_config_obj[benchmark]["args"]
        else:
            raise RuntimeError(
                f"Unexpected benchmark args type: {type(benchmark_config_obj[benchmark]['args'])}"
            )
        if extra_args:
            op_args.extend(extra_args)
        output_file = output_dir.joinpath(f"{benchmark}.json")
        op_args.extend(["--output-json", str(output_file.absolute())])
        run_in_task(op_args=op_args, benchmark_name=benchmark)
        # write pass or fail to result json
        # todo: check every input shape has passed
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logger.warning(
                f"[{benchmark_group_name}] Failed to run benchmark {benchmark}."
            )
            with open(output_file, "w") as f:
                json.dump({f"tritonbench_{benchmark}-pass": 0}, f)
        else:
            with open(output_file, "r") as f:
                obj = json.load(f)
            obj[f"tritonbench_{benchmark}-pass"] = 1
            with open(output_file, "w") as f:
                json.dump(obj, f, indent=4)
        output_files.append(output_file)
    # Reduce all operator CSV outputs to a single output json
    benchmark_data = [json.load(open(f, "r")) for f in output_files]
    aggregated_obj = decorate_benchmark_data(
        benchmark_group_name, run_timestamp, ci, benchmark_data
    )
    result_json_file = os.path.join(output_dir, "result.json")
    with open(result_json_file, "w") as fp:
        json.dump(aggregated_obj, fp, indent=4)
    logger.info(
        f"[{benchmark_group_name}] logging result json file to {result_json_file}."
    )
    if log_scuba:
        log_benchmark(aggregated_obj)
        logger.info(
            f"[{benchmark_group_name}] logging results to scuba table pytorch_user_benchmarks."
        )


def setup_output_dir(bm_name: str, ci: bool = False, output_dir: str | None = None):
    current_timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    if output_dir:
        return current_timestamp, output_dir
    output_dir = BENCHMARKS_OUTPUT_DIR.joinpath(bm_name, f"run-{current_timestamp}")
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    # set writable permission for all users (used by the ci env)
    if ci:
        output_dir.chmod(0o777)
    return current_timestamp, output_dir.absolute()
