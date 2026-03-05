"""
Tritonbench nightly run, dashboard: https://hud.pytorch.org/tritonbench/commit_view
Run all operators in nightly/autogen.yaml.
Requires the operator to support the speedup metric.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from ..common import setup_output_dir, setup_tritonbench_cwd


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="nightly", help="Benchmark name.")
    parser.add_argument(
        "--ci", action="store_true", help="Running in GitHub Actions CI mode."
    )
    parser.add_argument(
        "--log-scuba", action="store_true", help="Upload results to Scuba."
    )
    args = parser.parse_args()
    setup_tritonbench_cwd()
    from tritonbench.utils.run_utils import run_in_task
    from tritonbench.utils.scuba_utils import decorate_benchmark_data, log_benchmark

    run_timestamp, output_dir = setup_output_dir("nightly", ci=args.ci)
    # Run each operator
    output_files = []
    with open(os.path.join(CURRENT_DIR, "ci.yaml"), "r") as f:
        operator_benchmarks = yaml.safe_load(f)
    for op_bench in operator_benchmarks:
        benchmark_config = operator_benchmarks[op_bench]
        disabled = (
            benchmark_config.get("disabled", False)
            and _device_env_check(benchmark_config)
            and _triton_env_check(benchmark_config)
        )
        if disabled:
            logger.info(f"[nightly] Skipping disabled benchmark {benchmark_name}.")
            continue
        output_file = output_dir.joinpath(f"{op_bench}.json")
        benchmark_config["args"] += " " + " ".join(
            ["--output-json", str(output_file.absolute())]
        )
        run_in_task(
            op_args=benchmark_config["args"].split(" "), benchmark_name=op_bench
        )
        # write pass or fail to result json
        # todo: check every input shape has passed
        output_file_name = Path(output_file).stem
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logger.warning(f"[nightly] Failed to run {output_file_name}.")
            with open(output_file, "w") as f:
                json.dump({f"tritonbench_{output_file_name}-pass": 0}, f)
        else:
            with open(output_file, "r") as f:
                obj = json.load(f)
            obj[f"tritonbench_{output_file_name}-pass"] = 1
            with open(output_file, "w") as f:
                json.dump(obj, f, indent=4)
        output_files.append(output_file)
    # Reduce all operator CSV outputs to a single output json
    benchmark_data = [json.load(open(f, "r")) for f in output_files]
    aggregated_obj = decorate_benchmark_data(
        args.name, run_timestamp, args.ci, benchmark_data
    )
    result_json_file = os.path.join(output_dir, "result.json")
    with open(result_json_file, "w") as fp:
        json.dump(aggregated_obj, fp, indent=4)
    logger.info(f"[nightly] logging result json file to {result_json_file}.")
    if args.log_scuba:
        log_benchmark(aggregated_obj)
        logger.info(f"[nightly] logging results to scuba.")


if __name__ == "__main__":
    run()
