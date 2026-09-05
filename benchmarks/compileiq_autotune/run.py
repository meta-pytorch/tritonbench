import argparse
import functools
import logging
import os
import subprocess
import sys
from typing import List, Optional

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import LocalSearchSpaceBin
from compileiq.types import SearchConfiguration, WorkerTypes

from ..common import setup_tritonbench_cwd
from .extract_config import extract_best_configs
from .objective import (
    DEFAULT_CONFIG_FILE,
    KNOBS_FILENAME,
    objective_func,
    run_tritonbench_in_temp_dir,
    TRITONBENCH_CONFIGS_DIR,
)
from .shim import (
    CONTEXT_FILE,
    get_date_string,
    get_metric_name_from_config,
    REPO_WORK_DIR,
    save_context,
)

setup_tritonbench_cwd()

MANIFOLD_BUCKET = "tc_bench_ci"
MANIFOLD_URI_PREFIX = f"manifold://{MANIFOLD_BUCKET}/tree/compileiq"

MAX_METRIC_NAME = ["tflops"]
MIN_METRIC_NAME = ["latency"]

# The pre-built PTXAS search space. Triton dispatches to ptxas-blackwell (13.3)
# for arch >= 100, see get_ptxas() in the Triton nvidia backend compiler.
DEFAULT_SEARCH_SPACE = f"{MANIFOLD_URI_PREFIX}/ptxas_knobs/ptxas13.3_search_space.bin"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mount_manifold_bucket(bucket=MANIFOLD_BUCKET, uri_prefix=MANIFOLD_URI_PREFIX):
    mast_job_name = os.environ.get("MAST_HPC_JOB_NAME")
    mast_job_version = os.environ.get("MAST_HPC_JOB_VERSION")
    mast_job_attempt = os.environ.get("MAST_HPC_JOB_ATTEMPT_INDEX")
    env_vars = {
        "MAST_HPC_JOB_NAME": mast_job_name,
        "MAST_HPC_JOB_VERSION": mast_job_version,
        "MAST_HPC_JOB_ATTEMPT_INDEX": mast_job_attempt,
    }
    missing = [k for k, v in env_vars.items() if not v]
    if missing:
        if len(missing) < len(env_vars):
            logger.warning(
                f"Skipping manifold mount: missing env vars {missing}. "
                f"All of {list(env_vars.keys())} are required."
            )
        return False
    target_path = f"/mnt/{bucket}"
    subprocess.run(
        ["mkdir", "-p", target_path],
        check=True,
    )
    # /packages/oil.oilfs/oilfs-wrapper --profile=manifold --log-level debug --user="${AI_RM_ATTRIBUTION-}" ${extra_flags} "$manifold_uri" "$mount_dst"
    subprocess.run(
        [
            "/packages/oil.oilfs/oilfs-wrapper",
            "--profile=manifold",
            "--log-level",
            "debug",
            f"--user={os.environ.get('AI_RM_ATTRIBUTION', '')}",
            uri_prefix,
            target_path,
        ],
        check=True,
    )
    logger.info(
        f"[tritonbench_compileiq] Mounted {bucket}/{uri_prefix} to {target_path}"
    )
    return True


def resolve_search_space(search_space):
    """Resolve the search space to a local binary CompileIQ can sample from.

    `search_space` is either a local path or a manifold:// URI, in which case it
    is downloaded into REPO_WORK_DIR once and reused on later runs.
    """
    if search_space.startswith("manifold://"):
        manifold_path = search_space[len("manifold://") :]
        local_path = REPO_WORK_DIR.joinpath(os.path.basename(manifold_path))
        if not local_path.exists():
            logger.info(
                f"[tritonbench_compileiq] Downloading search space {search_space} to {local_path}"
            )
            subprocess.run(
                ["manifold", "get", manifold_path, str(local_path)],
                check=True,
            )
    else:
        local_path = search_space
    return LocalSearchSpaceBin(path=local_path)


def write_validate_script(output_dir, tritonbench_config, acf_file):
    """Write a validate.sh into output_dir that re-applies an ACF config and re-runs
    the ptxas check for the search's tritonbench config.

    TRITONBENCH_ROOT and CIQ_ACF can be overridden via env vars; they default to a
    local tritonbench checkout and the best extracted ACF file respectively.
    """
    script = f"""CURDIR=$PWD

if [ -z ${{TRITONBENCH_ROOT:-}} ]; then
  TRITONBENCH_ROOT=$HOME/local/tritonbench
fi

TRITONBENCH_CONFIGS_DIR=$TRITONBENCH_ROOT/benchmarks/run_config
TRITONBENCH_CONFIG_FILE={tritonbench_config}

if [ ! -f $CURDIR/$CIQ_ACF ]; then
  CIQ_ACF="{acf_file}"
fi

cd $TRITONBENCH_ROOT

PTXAS_OPTIONS="--apply-controls=$CURDIR/$CIQ_ACF" TRITONBENCH_RUN_CONFIG="$TRITONBENCH_CONFIGS_DIR/$TRITONBENCH_CONFIG_FILE" python -m benchmarks.ptxas_check.run

cd -
"""
    script_path = os.path.join(output_dir, "validate.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def search(
    results_csv,
    generations,
    short,
    metric_name,
    tritonbench_config,
    search_space,
    has_manifold=False,
    verbose=False,
):
    # Remove the .yaml extension
    tritonbench_config_name = os.path.splitext(tritonbench_config)[0]

    if short:
        generations = 1
        pool_size = 8
        cull_size = 4
    else:
        # CompileIQ derives pool_size/cull_size from num_objectives when unset.
        pool_size, cull_size = None, None

    run_name = f"{tritonbench_config_name}-{get_date_string()}"
    search_space_bin = resolve_search_space(search_space)
    search_context = {
        "short": short,
        "name": run_name,
        "metric_name": metric_name,
        "search_space": search_space,
    }

    search_context["generations"] = generations
    search_context["pool_size"] = pool_size
    logger.info(f"[tritonbench_compileiq] Search starts with context: {search_context}")

    save_context(
        path=os.path.join(REPO_WORK_DIR, CONTEXT_FILE),
        run_name=run_name,
        metric_name=metric_name,
        tritonbench_config=tritonbench_config,
        tritonbench_config_name=tritonbench_config_name,
    )

    if metric_name in MAX_METRIC_NAME:
        problem_type = "max"
    elif metric_name in MIN_METRIC_NAME:
        problem_type = "min"
    else:
        raise RuntimeError(f"Unknown metric: {metric_name}")

    main_config = SearchConfiguration(
        normalize=False,
        pool_size=pool_size,
        cull_size=cull_size,
        generations=generations,
        mutate_rate=0.25,
        problem_type=problem_type,
        num_objectives=1,
    )
    # The RAY workers are separate processes, so the flag has to travel with the
    # objective function itself rather than through module state.
    objective = (
        functools.partial(objective_func, verbose=True) if verbose else objective_func
    )
    tuner = Search(
        objective_function=objective,
        search_space=search_space_bin,
        search_config=main_config,
        worker_type=WorkerTypes.RAY,
        dump_results=results_csv,
        debug=False,
    )
    # NOTE: num_cpus=1 and num_gpus=1 in RAY means:
    #       Use as many CPUs and GPUs
    #       as there are available in the system.
    num_cpus, num_gpus = 1, 1
    results = tuner.start(num_cpus=num_cpus, num_gpus=num_gpus)

    logger.info(f"[tritonbench_compileiq] Best result: {results.get_best_result()}")
    logger.info(f"[tritonbench_compileiq] Search ends, result saves to {results_csv}")

    # Upload results to Manifold if running in a MAST job
    if has_manifold:
        mast_job_name = os.environ.get("MAST_HPC_JOB_NAME")
        mast_job_version = os.environ.get("MAST_HPC_JOB_VERSION")
        mast_job_attempt = os.environ.get("MAST_HPC_JOB_ATTEMPT_INDEX")
        manifold_path = f"{mast_job_name}_v{mast_job_version}_attempt{mast_job_attempt}"
        target_path = f"/mnt/{MANIFOLD_BUCKET}/{manifold_path}"
        logger.info(
            f"[tritonbench_compileiq] Uploading {results_csv} to manifold mount: {target_path}"
        )
        try:
            subprocess.run(
                ["mkdir", target_path],
                check=True,
            )
            subprocess.run(
                ["cp", results_csv, target_path],
                check=True,
            )
            logger.info(
                f"[tritonbench_compileiq] Upload complete: {MANIFOLD_URI_PREFIX}/{manifold_path}"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"[tritonbench_compileiq] Failed to upload to manifold: {e}")

        # Extract the best 3 configs and a validate.sh, then save them to the manifold mount
        try:
            best_configs = extract_best_configs(
                results=results,
                output_dir=target_path,
                best_number=3,
                file_prefix=tritonbench_config_name,
                higher_is_better=metric_name in MAX_METRIC_NAME,
            )
            logger.info(
                f"[tritonbench_compileiq] Saved best {len(best_configs)} configs to manifold mount: {best_configs}"
            )
            if best_configs:
                validate_script = write_validate_script(
                    output_dir=target_path,
                    tritonbench_config=tritonbench_config,
                    acf_file=os.path.basename(best_configs[0]),
                )
                logger.info(
                    f"[tritonbench_compileiq] Wrote validation script to manifold mount: {validate_script}"
                )
        except Exception as e:
            logger.error(
                f"[tritonbench_compileiq] Failed to extract best configs or write validation script: {e}"
            )


def get_parser():
    parser = argparse.ArgumentParser(description="Top level for the CompileIQ Search.")
    parser.add_argument("--test", action="store_true", help="Run in test mode.")
    parser.add_argument(
        "--results-csv",
        type=str,
        default=str(REPO_WORK_DIR.joinpath("result-compileiq.csv").absolute()),
        help="The name of the csv file to store results",
    )
    parser.add_argument("--short", action="store_true", help="Short run for testing")
    parser.add_argument(
        "--generations",
        type=int,
        default=110,
        help="Generations to tune, default to 110.",
    )
    parser.add_argument(
        "--tritonbench-config",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="The Tritonbench config file to use. This is a file name in the Tritonbench config directory.",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default=DEFAULT_SEARCH_SPACE,
        help="The CompileIQ ptxas search space. Either a local path or a manifold:// URI.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=f"Save the full stdout/stderr of every Tritonbench run to {REPO_WORK_DIR}.",
    )
    return parser


def run(args: Optional[List[str]] = None):
    parser = get_parser()
    args = parser.parse_args(args)
    REPO_WORK_DIR.mkdir(parents=True, exist_ok=True)

    local_rank = os.environ.get("LOCAL_RANK")
    if (
        os.environ.get("MAST_HPC_JOB_NAME")
        and os.environ.get("MAST_HPC_JOB_VERSION")
        and os.environ.get("MAST_HPC_JOB_ATTEMPT_INDEX")
    ):
        # running on MAST, check local rank
        if not local_rank:
            logger.info(
                "[tritonbench_compileiq] LOCAL_RANK env not set. Exiting the search."
            )
            exit(1)
        if local_rank and local_rank != "0":
            logger.info(
                f"[tritonbench_compileiq] Skipping search for non-zero local rank: {local_rank}"
            )
            return
        logger.info(
            "[tritonbench_compileiq] Running in MAST environment. Mounting manifold bucket..."
        )
        has_manifold = mount_manifold_bucket()
    else:
        has_manifold = False

    # Check that the tritonbench_config file exists
    if not os.path.exists(
        os.path.join(TRITONBENCH_CONFIGS_DIR, args.tritonbench_config)
    ):
        logger.error(
            f"Error: Tritonbench config needs to point to a file name in the Tritonbench config directory.\n{args.tritonbench_config} does not exist in {TRITONBENCH_CONFIGS_DIR}"
        )
        exit(1)

    metric_name = get_metric_name_from_config(
        os.path.join(TRITONBENCH_CONFIGS_DIR, args.tritonbench_config)
    )

    # This is a test run to check that the objective function is working correctly
    if args.test:
        logger.info("[tritonbench_compileiq] [test] Starting test run...")
        config = None
        try:
            results = run_tritonbench_in_temp_dir(
                config,
                metric_name=metric_name,
                knobs_file=KNOBS_FILENAME,
                iterations=1,
                tritonbench_config=args.tritonbench_config,
                verbose=True,
            )
        except Exception as e:
            exit_code = 1
            logger.error(f"Error running tritonbench: {e}")
        else:
            exit_code = 0
            logger.info(f"[tritonbench_compileiq] [test] {results}")
        finally:
            exit(exit_code)

    # This is the full search run
    else:
        logger.info(
            "[tritonbench_compileiq] Starting CompileIQ search. CUDA VISIBLE DEVICES: "
            + os.environ.get("CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES not set")
        )
        import torch

        cuda_version = (
            torch.version.cuda
            if hasattr(torch, "version") and hasattr(torch.version, "cuda")
            else "cuda not set"
        )
        logger.info(
            "[tritonbench_compileiq] torch version: "
            + torch.__version__
            + " cuda version: "
            + cuda_version
            + " cuda devices available: "
            + str(torch.cuda.device_count())
        )
        search(
            results_csv=args.results_csv,
            generations=args.generations,
            short=args.short,
            metric_name=metric_name,
            tritonbench_config=args.tritonbench_config,
            search_space=args.search_space,
            has_manifold=has_manifold,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    run(sys.argv[1:])
