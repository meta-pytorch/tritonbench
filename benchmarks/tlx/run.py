"""
Tritonbench nightly run, dashboard: https://hud.pytorch.org/tritonbench/commit_view
Run all operators in nightly/autogen.yaml.
Requires the operator to support the speedup metric.
"""

import argparse
import json
import logging
import os
import sys
from os.path import abspath, exists
from pathlib import Path
from typing import Any, Dict

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

def gen_tlx_benchmark_config() -> Dict[str, Any]:
    from tritonbench.metadata.query import get_benchmark_config_with_tags
    def _load_benchmarks(config_path: str) -> Dict[str, Any]:
        out = {}
        with open(config_path, "r") as f:
            obj = yaml.safe_load(f)
        if not obj:
            return out
        for benchmark_name in obj:
            # bypass disabled benchmarks
            if obj[benchmark_name].get("disabled", False):
                continue
        return out

    out = _load_benchmarks(os.path.join(CURRENT_DIR, "tlx_benchmarks_manual.yaml"))
    metadata_benchmarks = get_benchmark_config_with_tags(tags=["tlx"])
    out.update(metadata_benchmarks)
    return out


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="tlx", help="Benchmark name.")
    parser.add_argument("--generate-config", action="store_true", help="Generate Tritonbench run config file.")
    parser.add_argument(
        "--ci", action="store_true", help="Running in GitHub Actions CI mode."
    )
    parser.add_argument(
        "--log-scuba", action="store_true", help="Upload results to Scuba."
    )
    args = parser.parse_args()
    setup_tritonbench_cwd()
    from tritonbench.utils.run_utils import run_in_task, setup_output_dir
    from tritonbench.utils.scuba_utils import decorate_benchmark_data, log_benchmark

    run_timestamp, output_dir = setup_output_dir("tlx", ci=args.ci)
    # Run each operator
    output_files = []
    tlx_benchmarks = gen_tlx_benchmark_config()
    print(yaml.dump(tlx_benchmarks))
    if args.generate_config:
        with open(os.path.join(output_dir, "tlx_benchmarks_autogen.yaml"), "w") as f:
            yaml.dump(tlx_benchmarks, f)
        logger.info(f"[tlx benchmark] Generated config file to {output_dir}.")
        return
    for tlx_bench in tlx_benchmarks:
        op_args = tlx_benchmarks[tlx_bench].split(" ")
        output_file = output_dir.joinpath(f"{tlx_bench}.json")
        op_args.extend(["--output-json", str(output_file.absolute())])
        run_in_task(op_args=op_args, benchmark_name=tlx_bench)
        # write pass or fail to result json
        # todo: check every input shape has passed
        output_file_name = Path(output_file).stem
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logger.warning(f"[tlx benchmark] Failed to run {output_file_name}.")
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
    logger.info(f"[tlx benchmark] logging result json file to {result_json_file}.")
    if args.log_scuba:
        log_benchmark(aggregated_obj)
        logger.info(f"[tlx benchmark] logging results to scuba.")


if __name__ == "__main__":
    run()