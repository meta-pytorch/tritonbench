"""
Tritonbench nightly run on TLX
Run benchmarks/tlx/ci.yaml.
Output op default metrics.
"""

import argparse
import logging
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from ..common import run_benchmark_config_ci


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="tlx", help="Benchmark name.")
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

    run_benchmark_config_ci(
        args.name,
        os.path.join(CURRENT_DIR, "ci.yaml"),
        output_dir=args.output_dir,
        op=args.op,
        ci=args.ci,
        log_scuba=args.log_scuba,
    )


if __name__ == "__main__":
    run()
