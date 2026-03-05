"""
Tritonbench nightly run, dashboard: https://hud.pytorch.org/tritonbench/commit_view
Run all operators in nightly/autogen.yaml.
Requires the operator to support the speedup metric.
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
    parser.add_argument("--name", default="nightly", help="Benchmark name.")
    parser.add_argument("--op", type=str, default=None, help="Running on a single op.")
    parser.add_argument(
        "--ci", action="store_true", help="Running in GitHub Actions CI mode."
    )
    parser.add_argument(
        "--log-scuba", action="store_true", help="Upload results to Scuba."
    )
    args = parser.parse_args()

    run_benchmark_config_ci(
        args.name,
        os.path.join(CURRENT_DIR, "ci.yaml"),
        op=args.op,
        ci=args.ci,
        log_scuba=args.log_scuba,
    )


if __name__ == "__main__":
    run()
