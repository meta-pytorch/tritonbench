"""
Measure and collect compile time for operators.
"""

import argparse
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from ..common import run_benchmark_config_ci

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="compile_time", help="Benchmark name.")
    parser.add_argument(
        "--ci", action="store_true", help="Running in GitHub Actions CI mode."
    )
    parser.add_argument(
        "--op", required=False, default=None, help="Run a single operator."
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
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
