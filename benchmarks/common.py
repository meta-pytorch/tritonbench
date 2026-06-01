import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from functools import partial
from os.path import abspath, exists
from pathlib import Path
from typing import Callable, List

import yaml

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


from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.path_utils import REPO_PATH

BENCHMARKS_OUTPUT_DIR = REPO_PATH.joinpath(".benchmarks")


def post_run_callback(
    logger, benchmark_group_name, benchmark, output_file, output_files, disabled
):
    if disabled:
        return
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        logger.warning(f"[{benchmark_group_name}] Failed to run benchmark {benchmark}.")
        with open(output_file, "w") as f:
            json.dump({f"tritonbench_{benchmark}-pass": 0}, f)
    else:
        with open(output_file, "r") as f:
            obj = json.load(f)
        obj[f"tritonbench_{benchmark}-pass"] = 1
        with open(output_file, "w") as f:
            json.dump(obj, f, indent=4)
    output_files.append(output_file)


def run_benchmark_config_ci(
    benchmark_group_name: str,
    benchmark_config_file: str,
    extra_args: List[str] | None = None,
    transform_func: Callable | None = None,
    output_dir: str | None = None,
    op: str | None = None,
    ci: bool = False,
    log_scuba: bool = False,
    logging_group: str | None = None,
):
    from tritonbench.utils.run_utils import run_config, SPECIAL_CONFIG_FIELDS
    from tritonbench.utils.scuba_utils import decorate_benchmark_data, log_benchmark

    benchmark_start_time = time.time()
    output_files = []
    run_timestamp, output_dir = setup_output_dir(
        benchmark_group_name, ci=ci, output_dir=output_dir
    )
    if transform_func is None:
        transform_func = lambda x: x
    with open(benchmark_config_file, "r") as f:
        benchmark_config_obj = yaml.safe_load(f)
    per_benchmark_map = {}
    for benchmark in benchmark_config_obj:
        if benchmark in SPECIAL_CONFIG_FIELDS:
            continue
        per_benchmark_extra_args = [] if extra_args is None else extra_args.copy()
        output_file = output_dir.joinpath(f"{benchmark}.json")
        per_benchmark_extra_args.extend(["--output-json", str(output_file.absolute())])
        per_benchmark_map[benchmark] = {
            "extra_args": per_benchmark_extra_args,
            "enable_condition": lambda op_name: op == None or op_name == op,
            "callback": partial(
                post_run_callback,
                logger,
                benchmark_group_name,
                benchmark,
                output_file,
                output_files,
            ),
        }
    # Run the config file w/per-benchmark extra args and callback
    maybe_scuba_args = ["--log-scuba"] if log_scuba and is_fbcode() else []
    maybe_logging_group = ["--logging-group", logging_group] if logging_group else []
    run_config(
        config_file=benchmark_config_file,
        args=["--worker-mode"] + maybe_scuba_args + maybe_logging_group,
        per_config_entry=per_benchmark_map,
        benchmark_group_name=benchmark_group_name,
    )
    # Reduce all operator CSV outputs to a single output json
    benchmark_data = [transform_func(json.load(open(f, "r"))) for f in output_files]
    aggregated_obj = decorate_benchmark_data(
        benchmark_group_name, run_timestamp, ci, benchmark_data
    )
    result_json_file = os.path.join(output_dir, "result.json")
    with open(result_json_file, "w") as fp:
        json.dump(aggregated_obj, fp, indent=4)
    logger.info(
        f"[{benchmark_group_name}] logging result json file to {result_json_file}."
    )
    # when in oss, log to scuba at the end of the run
    if log_scuba and not is_fbcode():
        log_benchmark(aggregated_obj)
        logger.info(f"[{benchmark_group_name}] logging results to scuba table.")

    if is_fbcode():
        from tritonbench.utils.fb.utils import log_job_summary_to_scuba

        metrics = aggregated_obj.get("metrics", {})
        failed_benchmarks = []
        num_success = 0
        for benchmark in per_benchmark_map:
            pass_key = f"tritonbench_{benchmark}-pass"
            if metrics.get(pass_key) == 1:
                num_success += 1
            else:
                failed_benchmarks.append(benchmark)
        num_total = len(per_benchmark_map)
        num_failed = len(failed_benchmarks)
        pass_rate = num_success / num_total if num_total > 0 else 0.0
        log_job_summary_to_scuba(
            benchmark_group_name,
            benchmark_start_time,
            num_success=num_success,
            num_failed=num_failed,
            num_total=num_total,
            pass_rate=pass_rate,
            failed_benchmarks=failed_benchmarks,
        )


def setup_output_dir(bm_name: str, ci: bool = False, output_dir: str | None = None):
    current_timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    if output_dir:
        return current_timestamp, output_dir
    if is_fbcode():
        base_dir = Path(tempfile.mkdtemp(prefix="tritonbench_benchmarks_"))
    else:
        base_dir = BENCHMARKS_OUTPUT_DIR
    output_dir = base_dir.joinpath(bm_name, f"run-{current_timestamp}")
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    # set writable permission for all users (used by the ci env)
    if ci:
        output_dir.chmod(0o777)
    return current_timestamp, output_dir.absolute()
