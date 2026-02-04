"""
Tritonbench nightly run on TLX
Run all operator backends with tlx tags, plus tlx/tlx_benchmarks.yaml.
Output default metrics.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from common import setup_tritonbench_cwd

setup_tritonbench_cwd()

from tritonbench.utils.run_utils import run_benchmark_config_ci


def gen_tlx_benchmark_config() -> Dict[str, Any]:
    from tlx_tutorial_plugin import load_tlx_tutorial_backends
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

    out = _load_benchmarks(os.path.join(CURRENT_DIR, "tlx_benchmarks.yaml"))
    tlx_tutorial_benchmark_metadata = load_tlx_tutorial_backends()
    # only load benchmarks from runtime metadata
    metadata_benchmarks = get_benchmark_config_with_tags(
        tags=["tlx"],
        runtime_metadata=tlx_tutorial_benchmark_metadata,
        runtime_only=True,
    )
    out.update(metadata_benchmarks)
    return out


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="tlx", help="Benchmark name.")
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate Tritonbench run config file.",
    )
    parser.add_argument("--op", help="only run specified operator.")
    parser.add_argument(
        "--ci", action="store_true", help="Running in GitHub Actions CI mode."
    )
    parser.add_argument(
        "--log-scuba", action="store_true", help="Upload results to Scuba."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir, default to .benchmark/tlx/run-<timestamp>",
    )
    args = parser.parse_args()

    # Run each operator
    tlx_benchmarks = gen_tlx_benchmark_config()
    print(yaml.dump(tlx_benchmarks))
    if args.generate_config:
        with open(os.path.join(output_dir, "tlx_benchmarks_autogen.yaml"), "w") as f:
            yaml.dump(tlx_benchmarks, f)
        logger.info(f"[tlx benchmark] Generated config file to {output_dir}.")
        return
    run_benchmark_config_ci(
        args.name,
        tlx_benchmarks,
        extra_args=[
            "--plugin",
            "benchmarks.tlx.tlx_tutorial_plugin.load_tlx_tutorial_backends",
        ],
        output_dir=args.output_dir,
        op=args.op,
        ci=args.ci,
        log_scuba=args.log_scuba,
    )


if __name__ == "__main__":
    run()
