"""
Perform power and performance analysis on a Triton kernel.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from ..common import setup_tritonbench_cwd, strip_torchrun_env

setup_tritonbench_cwd()

from tritonbench.utils.run_utils import (
    load_operator_by_args,
    run_config,
    tritonbench_run,
)

MANIFOLD_SCHEME = "manifold://"

# Manifold bucket path (without the ``manifold://`` scheme) that receives the
# uploaded power-analysis output directories.
MANIFOLD_DEST = "tc_bench_ci/tree/power_analysis"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tritonbench-config",
        type=str,
        required=True,
        default=None,
        help="Path to a tritonbench config file (e.g. benchmarks/run_config/*.yaml), "
        "or a manifold:// URL to one (e.g. "
        "manifold://tc_bench_ci/tree/reactor_ci_benchmark/B200_ci.yaml). "
        "The config is rewritten with power-analysis flags appended to its "
        "common args and then run.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of A/B repeats. Fills the --ab-repeat value appended to "
        "the config's common args.",
    )
    return parser


def build_power_common_args(output_dir, repeat):
    """Power-analysis flags appended to the config's common args."""
    result_json = os.path.join(output_dir, "result.json")
    # NOTE: --side-a must use the bare `--side-a=` form, not `--side-a=""`.
    # Common args are split on spaces with no shell involved, so quotes would
    # survive literally: argparse would see the value '""', which parses to a
    # phantom empty-string arg that operators reject with
    # "run.py: error: unrecognized arguments: ".
    return (
        "--power-chart "
        f"--ab-repeat={repeat} "
        "--side-a= "
        f"--output-json {result_json} "
        f"--output-dir={output_dir}"
    )


def get_output_dir():
    """Tmp dir holding this run's rewritten config and outputs.

    Under MAST (``MAST_HPC_JOB_NAME`` set) the dir is named after the MAST
    job ID so runs are identifiable; otherwise a unique
    ``tritonbench_power_analysis_<XXXXX>`` dir is created.
    """
    mast_job_id = os.environ.get("MAST_HPC_JOB_NAME")
    if mast_job_id:
        output_dir = os.path.join(
            tempfile.gettempdir(),
            re.sub(r"[^A-Za-z0-9_.-]", "_", mast_job_id),
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    return tempfile.mkdtemp(prefix="tritonbench_power_analysis_")


def use_do_bench_latency(args):
    """Replace profiler latency measurement with `triton_do_bench` in `args`."""
    return re.sub(
        r"--latency-measure-mode[= ]profiler",
        "--latency-measure-mode=triton_do_bench",
        args,
    )


def rewrite_config_with_power_args(config_path, output_dir, repeat):
    """Copy `config_path` into `output_dir` with power flags in common args.

    Benchmarks that measure latency with the profiler are switched to
    `triton_do_bench`.

    Returns the path of the rewritten config file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    common_args = use_do_bench_latency((config.get("common_args") or "").strip())
    extra_args = build_power_common_args(output_dir, repeat)
    config["common_args"] = f"{common_args} {extra_args}".strip()
    for entry in config.values():
        if isinstance(entry, dict) and entry.get("args"):
            entry["args"] = use_do_bench_latency(entry["args"])
    rewritten_path = os.path.join(output_dir, os.path.basename(config_path))
    with open(rewritten_path, "w") as f:
        # A wide width keeps each arg string on a single line. The default
        # width (80) folds long scalars into continuation lines, turning them
        # into multi-line strings in the rewritten config.
        yaml.safe_dump(config, f, width=4096)
    return rewritten_path


def unset_nccl_envs():
    """Remove NCCL_* variables from the environment.

    Stale NCCL plugin/config vars (e.g. from the launcher environment) can
    break the benchmark subprocesses, which inherit os.environ. Returns the
    removed vars so the caller can restore them afterwards.
    """
    removed = {}
    for key in [key for key in os.environ if key.startswith("NCCL_")]:
        removed[key] = os.environ.pop(key)
    if removed:
        logger.info(f"Unset NCCL env vars: {sorted(removed)}")
    return removed


def fetch_config_from_manifold(config_url):
    """Download the config at `config_url` into a tmp dir, returning its path.

    The basename is preserved so the rewritten config keeps the same name.
    """
    local_dir = tempfile.mkdtemp(prefix="tritonbench_power_analysis_config_")
    local_path = os.path.join(local_dir, os.path.basename(config_url))
    cmd = ["manifold", "get", config_url[len(MANIFOLD_SCHEME) :], local_path]
    logger.info(f"Downloading tritonbench config {config_url} to {local_path}")
    subprocess.run(cmd, check=True)
    return local_path


def upload_to_manifold(local_dir):
    """Recursively upload `local_dir` under MANIFOLD_DEST."""
    dest = f"{MANIFOLD_DEST}/{os.path.basename(local_dir)}"
    cmd = ["manifold", "putr", local_dir, dest]
    logger.info(f"Uploading {local_dir} to manifold://{dest}")
    subprocess.run(cmd, check=True)
    logger.info(f"Upload complete: manifold://{dest}")


def run_with_config(config_path, repeat):
    """Run power analysis for a tritonbench config file.

    `config_path` is a local path or a ``manifold://`` URL. Rewrites the config
    with power-analysis flags, runs tritonbench with it, then uploads the output
    dir to manifold. Returns the output dir.
    """
    if config_path.startswith(MANIFOLD_SCHEME):
        config_path = fetch_config_from_manifold(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Tritonbench config file not found: {config_path}")
    output_dir = get_output_dir()
    logger.info(f"Power analysis output dir: {output_dir}")
    rewritten_config = rewrite_config_with_power_args(config_path, output_dir, repeat)
    logger.info(f"Rewritten tritonbench config: {rewritten_config}")
    cmd_env = strip_torchrun_env(os.environ.copy())
    run_config(
        rewritten_config, ["--worker-mode"], extra_envs=cmd_env, override_envs=True
    )
    upload_to_manifold(output_dir)
    return output_dir


def run(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    # run_config spawns one subprocess per benchmark, re-invoking this same
    # binary with the operator's args plus --worker-mode. Those runs belong to
    # the main tritonbench runner, not to this driver's parser.
    if "--worker-mode" in argv:
        tritonbench_run(argv)
        return None
    parser = get_parser()
    args = parser.parse_args(argv)
    return run_with_config(args.tritonbench_config, args.repeat)


if __name__ == "__main__":
    run(sys.argv[1:])
