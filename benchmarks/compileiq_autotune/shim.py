import subprocess
import sys
import pandas as pd
from datetime import datetime
import yaml
import os
import json
from pathlib import Path
import logging

from compileiq.utils.helpers import save_compiler_config

from ..common import REPO_PATH

REPO_WORK_DIR = Path("/tmp/tritonbench_compileiq_search")
DEFAULT_CONFIG_FILE = "gemm_config_3.yaml"
CONTEXT_FILE = "context.json"
# Per-benchmark timeout. One evaluation of a `--rep 3000` config takes minutes on
# its own, and the search runs one per GPU concurrently, so this needs headroom.
RUN_TIMEOUT_SEC = int(os.environ.get("TRITONBENCH_COMPILEIQ_TIMEOUT", "1800"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_repo_root():
    return REPO_PATH

def get_gpu_info():
    """
    Uses Nvidia-smi to get the GPU information.
    """
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    except Exception as e:
        gpu_info = f"Error getting GPU info: {e}"
    else:
        if result.returncode != 0:
            gpu_info = f"Error getting GPU info: {result.stderr}"
        else:
            gpu_info = result.stdout
    finally:
        return gpu_info

# FIXME: This will break for multi-GPU systems
def get_gpu_name():
    # Get the name of the GPU
    return subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode("utf-8").strip()

def get_mean_and_std(df, metric_name):
    # Get the mean and standard deviation of the performance metrics
    return df[metric_name].mean(), df[metric_name].std()

def get_metric_from_df(df: pd.DataFrame, index: int, metric_name):
    return df.iloc[index][metric_name]

def get_date_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_performance_metric(output):
    # Extract the performance metric from the output
    # Hack: In MAST environment, the job will print something like
    # "NCCL INFO ENV/Plugin: Could not find: libnccl-env.so"
    # "NCCL INFO ENV/Plugin: Closing env plugin ncclEnvDefault"
    # Error messages. These are not related to benchmarking and is safe to bypass.
    last_line = [x for x in output.stdout.splitlines() if "NCCL INFO ENV/Plugin" not in x and "ncclEnvDefault" not in x][-1]
    # return the first metric
    if "," in last_line:
        return float(last_line.split(",")[0].strip())
    return float(last_line.split()[-1])

def truncate_output(output, limit=128):
    # subprocess.TimeoutExpired may carry stdout/stderr as None.
    if not output:
        return ""
    return output[-min(limit, len(output)):]

def write_encrypted_knobs(config: dict|bytes, knobs_file: str):
    """
    To be used if encrypted knobs are provided.
    """
    if isinstance(config, dict) or config == None:
        with open(knobs_file, "w") as f:
            f.write(yaml.dump(config))
        return
    # CompileIQ hands out a file-backed search space candidate as a hex string,
    # which decodes into the binary Advanced Control File (ACF).
    save_compiler_config(knobs_file, config)

def run_tritonbench(num_runs, workdir, metric_name="tflops", mock=False, knobs_file=None, config_file="example_config.yaml", csv_file=None, verbose=False):
    """
    Runs the tritonbench locally  and returns a dataframe with the performance metrics.

    Args:
        num_runs (int): The number of runs to perform.
        workdir (str): The TritonBench workdir to use.
        mock (bool): Whether to mock the run.
        knobs_file (str): The path to the knobs file to use in the docker container.
    """
    if knobs_file:
        knobs_env = f"--apply-controls={workdir}/{knobs_file}"
    else:
        knobs_env = ""


    cmd_env = os.environ.copy()
    assert os.path.exists(f"{workdir}/{config_file}")
    assert os.path.exists(REPO_PATH.joinpath("run.py")), f"run.py not found in {REPO_PATH}"

    cmd_env["TRITONBENCH_RUN_CONFIG"] = f"{workdir}/{config_file}"
    cmd_env["TRITON_ALWAYS_COMPILE"] = "1"
    cmd_env["TRITON_CACHE_DIR"] = f"{workdir}/.triton"
    if knobs_env:
        cmd_env["PTXAS_OPTIONS"] = knobs_env

    if mock:
        cmd = "echo blablabla && echo 1.234567890 && echo blablabla"
    else:
        cmd = f"{sys.executable} run.py"

    df = pd.DataFrame(columns=["timestamp", metric_name])
    for i in range(num_runs):
        if verbose:
            print(f"Running run {i+1} of {num_runs}, cmd: {cmd}")
        GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        # Run the docker container
        try:
            output = subprocess.run(
                cmd,
                timeout=RUN_TIMEOUT_SEC,
                shell=True,
                capture_output=True,
                text=True,
                env=cmd_env,
                cwd=REPO_PATH,
            )
            truncated_output_stdout = truncate_output(output.stdout)
            truncated_output_stderr = truncate_output(output.stderr)
        except subprocess.TimeoutExpired as e:
            truncated_output_stdout = truncate_output(e.stdout)
            truncated_output_stderr = truncate_output(e.stderr)
            error_msg = f"[GPU ID {GPU_ID}] Timeout ({RUN_TIMEOUT_SEC}s) running Tritonbench. stdout: {truncated_output_stdout}, stderr: {truncated_output_stderr}"
            print(error_msg)
            raise RuntimeError(error_msg)
        else:
            if output.returncode != 0:
                error_msg = f"[GPU ID {GPU_ID}] Error running Tritonbench. stdout: {truncated_output_stdout}, stderr: {truncated_output_stderr}"
                print(error_msg)
                raise RuntimeError(error_msg)
            else:
                # Extract the performance metric
                try:
                    performance_metric = extract_performance_metric(output)
                except Exception as e:
                    print(f"[GPU ID {GPU_ID}] Error extracting performance metric: {e}\nstdout:\n{output.stdout[-200:]} stderr:\n{output.stderr[-200:]}\n")
                    raise RuntimeError(f"[GPU ID {GPU_ID}]Error extracting performance metric: {e}\n{output.stdout}\n{output.stderr}")
                else:
                    # Store the performance metric in the dataframe
                    df.loc[len(df)] = {"timestamp": get_date_string(), metric_name: performance_metric}
                    if csv_file:
                        df.to_csv(csv_file, mode="w", header=True)
    return df

def get_metric_name_from_config(config_file):
    """
    Get the metric name from the config file.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    first_benchmark_args = list(config.items())[0][1]["args"].split(" ")
    assert "--metrics" in first_benchmark_args, "No metrics found in config file."
    metric_index = first_benchmark_args.index("--metrics")
    logger.info("[tritonbench_compileiq] Found tritonbench metrics: {}".format(first_benchmark_args[metric_index + 1]))
    metric_name = first_benchmark_args[metric_index + 1]
    if "," in metric_name:
        metric_name = metric_name.split(",")[0]
    return metric_name


def save_context(path=CONTEXT_FILE, **kwargs):
    obj = {}
    for key, item in kwargs.items():
        obj[key] = item
    with open(path, "w") as f:
        json.dump(obj, f)

def load_context(path: str = CONTEXT_FILE):
    with open(path, "r") as f:
        return json.load(f)
