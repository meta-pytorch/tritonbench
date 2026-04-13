"""
Tritonbench sweep runner.

Generates a TRITONBENCH_RUN_CONFIG from a config file and sweep target,
then launches tritonbench with the generated config.

Example config file format (YAML):
```yaml
run_configs:
  perf:
    with_backwards: true
    tags:
      - triton
    metrics:
      - latency
  compile_time:
    tags:
      - triton
    metrics:
      - compile_time
```

Usage:
    python -m benchmarks.sweep_runner.run \
        --sweep-config-file triton.yaml --sweep-target timing_accuracy \
        --separate-backends
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..common import REPO_PATH, setup_output_dir, setup_tritonbench_cwd


setup_tritonbench_cwd()

from tritonbench.metadata.query import get_benchmark_config_with_tags

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.absolute()


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and run tritonbench sweep configurations"
    )
    parser.add_argument(
        "--sweep-config-file",
        type=str,
        required=True,
        default="triton.yaml",
        help="Path to the YAML config file specifying operators and backends",
    )
    parser.add_argument(
        "--sweep-target",
        type=str,
        required=True,
        help="The sweep target to run (e.g., timing_accuracy)",
    )
    parser.add_argument(
        "--separate-backends",
        action="store_true",
        help="Whether to separate benchmark entries for each backend",
    )
    parser.add_argument(
        "--sweep-run",
        action="store_true",
        help="Run the benchmark after generating the config file",
    )
    parser.add_argument(
        "--sweep-output-file",
        type=str,
        default=None,
        help="Output generated config file.",
    )
    parser.add_argument(
        "--sweep-num-configs",
        default=None,
        type=int,
        help="Maximum number of configs to generate.",
    )
    parser.add_argument(
        "--attach-launch",
        action="store_true",
        help="Attach launch argument in common_args.",
    )

    parsed_args, extra_args = parser.parse_known_args(args)
    parsed_args.extra_args = extra_args
    return parsed_args


def load_config(config_file: str, base_dir=CURRENT_DIR) -> Dict[str, Any]:
    """Load and parse the YAML config file."""
    config_path = Path(config_file)
    if not config_path.exists():
        config_path = Path(base_dir).joinpath(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def generate_run_config(
    sweep_runner_config: Dict[str, Any],
    target: str,
    extra_args: List[str],
    separate_backends: bool = False,
    num_configs: Optional[int] = None,
    attach_launch: bool = False,
) -> Dict[str, Any]:
    """
    Generate a TRITONBENCH_RUN_CONFIG file from the sweep runner config and target.

    Args:
        config: The loaded YAML config containing operators and backends
        sweep_target: The target metric/mode to sweep
        extra_args: Additional arguments to append to each benchmark
        separate_backends: If True, generate separate entries for each backend

    Returns:
        A dictionary in TRITONBENCH_RUN_CONFIG format
    """
    result_configs = {}
    if attach_launch:
        result_configs["common_args"] = f"--launch benchmarks.{target}.run"
    if extra_args:
        if "common_args" not in result_configs:
            result_configs["common_args"] = ""
        if len(result_configs["common_args"]):
            result_configs["common_args"] += " "
        result_configs["common_args"] += " ".join(extra_args)

    disabled_benchmarks = sweep_runner_config.get("disabled", {})
    override_benchmarks = sweep_runner_config.get("overrides", {})

    config_count = 0
    for _run_config_name, run_config_value in sweep_runner_config["run_configs"].items():
        if skip_tests := run_config_value.get("skip_tests", None):
            skip_tests = [
                load_config(skip_test, base_dir=REPO_PATH) for skip_test in skip_tests
            ]
        run_config = get_benchmark_config_with_tags(
            tags=run_config_value["tags"],
            per_backend=separate_backends,
            with_backwards=run_config_value.get("with_backwards", False),
            metrics=run_config_value.get("metrics", None),
            skip_tests=skip_tests,
        )
        for c in run_config:
            if num_configs is not None and config_count >= num_configs:
                break
            if c in disabled_benchmarks:
                continue
            if c in override_benchmarks:
                result_configs[c] = override_benchmarks[c].copy()
            else:
                result_configs[c] = run_config[c].copy()
            config_count += 1
    return result_configs


def write_run_config(run_config: Dict[str, Any], output_path: str) -> None:
    """Write the run config to a YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)


def run(args: Optional[List[str]] = None) -> None:
    """Main entry point for the sweep runner."""
    parsed_args = parse_args(args)

    if not parsed_args.sweep_output_file:
        default_output_name = f"sweep_{parsed_args.sweep_target}.yaml"
        timestamp, output_dir = setup_output_dir(bm_name=parsed_args.target)
        parsed_args.sweep_output_file = output_dir.join(default_output_name)

    run_config = generate_run_config(
        sweep_runner_config=load_config(parsed_args.sweep_config_file),
        target=parsed_args.sweep_target,
        extra_args=parsed_args.extra_args,
        separate_backends=parsed_args.separate_backends,
        num_configs=parsed_args.sweep_num_configs,
        attach_launch=parsed_args.attach_launch,
    )

    write_run_config(run_config, parsed_args.sweep_output_file)
    logger.info(f"Generated config written to: {parsed_args.sweep_output_file}")

    if not parsed_args.sweep_run:
        return

    env = os.environ.copy()
    env["TRITONBENCH_RUN_CONFIG"] = parsed_args.sweep_output_file

    run_py_path = REPO_PATH / "run.py"

    cmd = [sys.executable, str(run_py_path)]

    logger.info(
        f"Running tritonbench with generated config: {parsed_args.sweep_output_file}"
    )
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"TRITONBENCH_RUN_CONFIG={parsed_args.sweep_output_file}")

    subprocess.run(cmd, env=env, cwd=str(REPO_PATH), check=True)


if __name__ == "__main__":
    run()
