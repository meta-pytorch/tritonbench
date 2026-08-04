import os
import tempfile
import shutil

from evo_solar.evo import BASELINE_DNA

from .shim import (
    get_mean_and_std,
    get_metric_from_df,
    run_tritonbench,
    write_encrypted_knobs,
    load_context,
    REPO_WORK_DIR,
    DEFAULT_CONFIG_FILE,
    CONTEXT_FILE,
)

from ...common import REPO_PATH

KNOBS_FILENAME="knobs.bin"
INVALID_SCORE = '*'

TRITONBENCH_CONFIGS_DIR=os.path.join(REPO_PATH, "benchmarks", "fb", "evo_autotune")

def run_tritonbench_in_temp_dir(config:dict|bytes, knobs_file:str|None, iterations:int, metric_name: str, tritonbench_config:str=DEFAULT_CONFIG_FILE, verbose:bool=False):
    # Create a temporary directory for the run in the current working directory
    # This function assumes that tritonbench_config is a file in the TRITONBENCH_CONFIGS_DIR
    with tempfile.TemporaryDirectory(dir=REPO_WORK_DIR) as temp_dir:
        os.chmod(temp_dir, 0o777)

        config_file_temp = os.path.join(temp_dir, tritonbench_config)
        shutil.copy(os.path.join(TRITONBENCH_CONFIGS_DIR, tritonbench_config), config_file_temp)
        os.chmod(config_file_temp, 0o777)

        if knobs_file and isinstance(config, str):
            knobs_file_temp = os.path.join(temp_dir, knobs_file)
            write_encrypted_knobs(config, knobs_file_temp)
        else:
            knobs_file = None
        results = run_tritonbench(iterations, temp_dir, metric_name=metric_name, knobs_file=knobs_file, config_file=tritonbench_config, mock=False)
    return results

def print_bm_error(e):
    print("ERROR: Benchmark failed")
    print("       Returning INVALID_SCORE")
    print(f"{str(e)[-256:]}")


def objective_func(config, verbose=False):
    """
    Objective function.

    Tries to log as much information as possible in advance to
    running the objective function.
    """
    GPU_ID = os.getenv("CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES not set")

    if config == BASELINE_DNA:
        run_type = "baseline"
        iterations = 10
        knobs_file = None
    else:
        run_type = "evaluation"
        iterations = 1
        knobs_file = KNOBS_FILENAME

    context = load_context(os.path.join(REPO_WORK_DIR, CONTEXT_FILE))
    context["baseline"] = True if config == BASELINE_DNA else False

    # print(f"[xzhao9] Worker Start GPUID: {GPU_ID}, context: {context}.", flush=True)

    metric_name = context["metric_name"]

    try:
        results = run_tritonbench_in_temp_dir(config, knobs_file, iterations, metric_name=metric_name, tritonbench_config=context["tritonbench_config"], verbose=verbose)
        # For baseline, we want to get the mean of the metric name
        if run_type == "baseline":
            score , bs_std = get_mean_and_std(results, metric_name=metric_name)
        # For evaluation, we want to get the first row
        else:
            bs_std = None
            if results is not None and len(results) > 0:
                score = get_metric_from_df(results, 0, metric_name)
            else:
                score = INVALID_SCORE
    except Exception as e:
        print_bm_error(e)
        score = INVALID_SCORE
    finally:
        pass

    # FIXME: This does not work
    # if run_type == "baseline" and score != BADSCORE:
    #     with get_run(run_id=mlflow_context["run_id"]) as run:
    #         mlf_log_run_params(baseline_tflops=score, baseline_std=bs_std)

    print(f"[xzhao9] Worker End GPUID: {GPU_ID}, context: {context}, returning score: {score}.", flush=True)
    return score