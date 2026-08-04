import argparse
import os
import subprocess
import sys
import logging

from pathlib import Path

import evo_solar.evo
from evo_solar.evo import EvoSearch
from evo_solar.config.types import EvoConfiguration, WorkerTypes

from .shim import get_metric_name_from_config, get_date_string, save_context, REPO_WORK_DIR, CONTEXT_FILE
from .objective import objective_func, KNOBS_FILENAME, run_tritonbench_in_temp_dir, DEFAULT_CONFIG_FILE, TRITONBENCH_CONFIGS_DIR
from .extract_config import extract_best_configs

from ...common import setup_tritonbench_cwd

from typing import Optional, List

setup_tritonbench_cwd()

MANIFOLD_BUCKET = "tc_bench_ci"
MANIFOLD_URI_PREFIX = f"manifold://{MANIFOLD_BUCKET}/tree/compileiq"

MAX_METRIC_NAME = ["tflops"]
MIN_METRIC_NAME = ["latency"]
EVO_SOLAR_PATH = Path(evo_solar.evo.__file__).parent.absolute()
PTXAS_CONFIG_FILE = "cuda-12.8-ptxas-p2.config"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_sizes(num_objectives):
    # when num_objectives=1:
    # pollsize = 12
    # cullsize = 10
    cull = 0.75
    target = (2 * num_objectives) + 1
    poolsize = int(target / (1 - cull))
    poolsize = max(poolsize, 32)
    poolsize = poolsize if poolsize % 2 == 0 else poolsize + 1
    cullsize = int(poolsize * cull)
    cullsize = cullsize if cullsize % 2 == 0 else cullsize - 1

    return poolsize, cullsize

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
    logger.info(f"Mounted {bucket}/{uri_prefix} to {target_path}")
    return True


def write_validate_script(output_dir, tritonbench_config, evo_acf_file):
    """Write a validate.sh into output_dir that re-applies an ACF config and re-runs
    the ptxas check for the search's tritonbench config.

    TRITONBENCH_ROOT and EVO_ACF can be overridden via env vars; they default to a
    local fbsource-evo checkout and the best extracted ACF file respectively.
    """
    script = f"""CURDIR=$PWD

if [ -z ${{TRITONBENCH_ROOT:-}} ]; then
  TRITONBENCH_ROOT=$HOME/local/fbsource-evo/fbcode/pytorch/tritonbench
fi

EVO_AUTOTUNE_ROOT=$TRITONBENCH_ROOT/benchmarks/fb/evo_autotune
EVO_AUTOTUNE_FILE={tritonbench_config}

if [ ! -f $CURDIR/$EVO_ACF ]; then
  EVO_ACF="{evo_acf_file}"
fi

cd $TRITONBENCH_ROOT

PTXAS_OPTIONS="--apply-controls=$CURDIR/$EVO_ACF" TRITONBENCH_RUN_CONFIG="$EVO_AUTOTUNE_ROOT/$EVO_AUTOTUNE_FILE" python -m benchmarks.ptxas_check.run

cd -
"""
    script_path = os.path.join(output_dir, "validate.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def search(dbname, generations, short, metric_name, tritonbench_config, has_manifold=False):
    # Remove the .yaml extension
    tritonbench_config_name = os.path.splitext(tritonbench_config)[0]

    if short:
        generations = 1
        pool_size = 8
        cull_size = 4
    else:
        pool_size, cull_size = get_sizes(1)

    run_name = f"{tritonbench_config_name}-{get_date_string()}"
    dna_config = os.path.join(EVO_SOLAR_PATH, PTXAS_CONFIG_FILE)
    search_context = {
        "short": short,
        "name": run_name,
        "metric_name": metric_name,
        "dna_config": dna_config,
    }

    logger.info(f"[tritonbench_evo] Search starts with context: {search_context}")

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

    main_config = EvoConfiguration(
        qualitative=True,
        pool_size=pool_size,
        cull_size=cull_size,
        generations=generations,
        mutate_rate=0.25,
        problem_type=problem_type,
        num_objectives=1,
        enable_db=True,
        results_database=dbname,
        search_max_time=864000,
    )
    tuner = EvoSearch(
        objective_function=objective_func,
        search_space=dna_config,
        evo_config=main_config,
        worker_type=WorkerTypes.RAY,
        debug=False,
    )
    # NOTE: num_cpus=1 and num_gpus=1 in RAY means:
    #       Use as many CPUs and GPUs
    #       as there are available in the system.
    num_cpus, num_gpus = 1, 1
    results = tuner.start(num_cpus=num_cpus, num_gpus=num_gpus)

    logger.info(results.get_best_result())
    logger.info(f"[tritonbench_evo] Search ends, result saves to {dbname}")

    # Upload results to Manifold if running in a MAST job
    if has_manifold:
        mast_job_name = os.environ.get("MAST_HPC_JOB_NAME")
        mast_job_version = os.environ.get("MAST_HPC_JOB_VERSION")
        mast_job_attempt = os.environ.get("MAST_HPC_JOB_ATTEMPT_INDEX")
        manifold_path = f"{mast_job_name}_v{mast_job_version}_attempt{mast_job_attempt}"
        target_path = f"/mnt/{MANIFOLD_BUCKET}/{manifold_path}"
        logger.info(f"[tritonbench_evo] Uploading {dbname} to manifold mount: {target_path}")
        try:
            subprocess.run(
                ["mkdir", target_path],
                check=True,
            )
            subprocess.run(
                ["cp", dbname, target_path],
                check=True,
            )
            logger.info(f"[tritonbench_evo] Upload complete: {MANIFOLD_URI_PREFIX}/{manifold_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"[tritonbench_evo] Failed to upload to manifold: {e}")

        # Extract the best 3 configs and a validate.sh, then save them to the manifold mount
        try:
            best_configs = extract_best_configs(
                database=dbname,
                output_dir=target_path,
                best_number=3,
                file_prefix=tritonbench_config_name,
                higher_is_better=metric_name in MAX_METRIC_NAME,
            )
            logger.info(f"[tritonbench_evo] Saved best {len(best_configs)} configs to manifold mount: {best_configs}")
            if best_configs:
                validate_script = write_validate_script(
                    output_dir=target_path,
                    tritonbench_config=tritonbench_config,
                    evo_acf_file=os.path.basename(best_configs[0]),
                )
                logger.info(f"[tritonbench_evo] Wrote validation script to manifold mount: {validate_script}")
        except Exception as e:
            logger.error(f"[tritonbench_evo] Failed to extract best configs or write validation script: {e}")


def get_parser():
    parser = argparse.ArgumentParser(description="Top level for the Evo Search.")
    parser.add_argument("--test", action="store_true", help="Run in test mode.")
    parser.add_argument("--dbname", type=str, default=str(REPO_WORK_DIR.joinpath("result-evo.sqlite").absolute()), help="The name of the db file to store results")
    parser.add_argument('--short', action='store_true', help="Short run for testing")
    parser.add_argument('--generations', type=int, default=110, help="Generations to tune, default to 110.")
    parser.add_argument('--tritonbench-config', type=str, default=DEFAULT_CONFIG_FILE, help="The Tritonbench config file to use. This is a file name in the Tritonbench config directory.")
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
            logger.info("LOCAL_RANK env not set. Exiting the search.")
            exit(1)
        if local_rank and local_rank != "0":
            logger.info(f"Skipping search for non-zero local rank: {local_rank}")
            return
        logger.info("Running in MAST environment. Mounting manifold bucket...")
        has_manifold = mount_manifold_bucket()
    else:
        has_manifold = False

    # Check that the tritonbench_config file exists
    if not os.path.exists(os.path.join(TRITONBENCH_CONFIGS_DIR, args.tritonbench_config)):
        logger.error(f"Error: Tritonbench config needs to point to a file name in the Tritonbench config directory.\n{args.tritonbench_config} does not exist in {TRITONBENCH_CONFIGS_DIR}")
        exit(1)

    metric_name = get_metric_name_from_config(os.path.join(TRITONBENCH_CONFIGS_DIR, args.tritonbench_config))

    # This is a test run to check that the objective function is working correctly
    if args.test:
        logger.info("[test] Starting test run...")
        config = None
        try:
            results = run_tritonbench_in_temp_dir(config, metric_name=metric_name, knobs_file=KNOBS_FILENAME, iterations=1, tritonbench_config=args.tritonbench_config, verbose=True)
        except Exception as e:
            exit_code = 1
            logger.error(f"Error running tritonbench: {e}")
        else:
            exit_code = 0
            logger.info(results)
        finally:
            exit(exit_code)

    # This is the full search run
    else:
        logger.info(f"Starting evo search with generations: {args.generations}. CUDA VISIBLE DEVICES: " + os.environ.get("CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES not set"))
        import torch
        cuda_version = torch.version.cuda if hasattr(torch, "version") and hasattr(torch.version, "cuda") else "cuda not set"
        logger.info("torch version: " + torch.__version__ + " cuda version: " + cuda_version + " cuda devices available: " + str(torch.cuda.device_count()))
        search(dbname=args.dbname, generations=args.generations, short=args.short, metric_name=metric_name, tritonbench_config=args.tritonbench_config, has_manifold=has_manifold)


if __name__ == "__main__":
    run(sys.argv[1:])
