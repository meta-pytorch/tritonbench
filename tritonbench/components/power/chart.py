import csv
import logging
import os
import signal
import subprocess
import time

import matplotlib.pyplot as plt
import torch

# query every 10 ms
QUERY_FREQUENCY = 10
QUERY_STDOUT_FILE = "power.csv"
QUERY_STDERR_FILE = "power.log"
QUERY_COMMAND = """nvidia-smi -lms {QUERY_FREQUENCY} -i {QUERY_DEVICE} --query-gpu=power.draw.average,power.draw.instant,power.max_limit,temperature.gpu,temperature.memory,clocks.current.sm,clocks.current.memory,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,nounits"""
global QUERY_PROC
global POWER_OUTPUT_DIR

QUERY_PROC = None
POWER_OUTPUT_DIR = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_cuda_device_id():
    return torch.cuda.current_device()


def _gen_power_charts(benchmark_name: str, device_name: str, power_csv_file: str):
    # Read CSV
    with open(power_csv_file) as f:
        reader = csv.reader(f)
        header = next(reader)  # first row as header
        header = [col.strip() for col in header]
        data = {col: [] for col in header}

        for row in reader:
            for col, value in zip(header, row):
                if value == "[N/A]":
                    logger.warning(
                        f"[tritonbench][power] {col} is not available, skipping"
                    )
                    value = 0.0
                else:
                    value = (
                        float(value)
                        if col
                        not in [
                            "clocks_event_reasons.hw_thermal_slowdown",
                            "clocks_event_reasons.sw_thermal_slowdown",
                        ]
                        else value
                    )
                data[col].append(value)

    # Generate synthetic time axis (100 ms per sample)
    n_samples = len(next(iter(data.values())))
    time = [i * 0.1 for i in range(n_samples)]  # seconds (0.1s = 100 ms)

    # Plot power chart
    plt.figure(figsize=(10, 6))
    for power_col in header[:3]:
        plt.plot(time, data[power_col], label=power_col)
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.title(
        f"[tritonbench] {benchmark_name} power consumption over time on {device_name}"
    )
    plt.savefig(
        os.path.join(POWER_OUTPUT_DIR, "power.png"), dpi=300, bbox_inches="tight"
    )
    # Plot temp chart
    plt.figure(figsize=(10, 6))
    for temp_col in header[3:5]:
        plt.plot(time, data[temp_col], label=temp_col)
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (C)")
    plt.legend()
    plt.title(f"[tritonbench] {benchmark_name} temperature over time on {device_name}")
    plt.savefig(
        os.path.join(POWER_OUTPUT_DIR, "temp.png"), dpi=300, bbox_inches="tight"
    )
    # Plot frequency chart
    plt.figure(figsize=(10, 6))
    for temp_col in header[5:7]:
        plt.plot(time, data[temp_col], label=temp_col)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (MHz)")
    plt.legend()
    plt.title(f"[tritonbench] {benchmark_name} frequency over time on {device_name}")
    plt.savefig(
        os.path.join(POWER_OUTPUT_DIR, "freq.png"), dpi=300, bbox_inches="tight"
    )


def power_chart_begin(benchmark_name, output_dir):
    # check no other proc is running
    global QUERY_PROC, POWER_OUTPUT_DIR
    assert QUERY_PROC is None, "Power query process must be None to start a new one"
    # clean up the directory
    POWER_OUTPUT_DIR = os.path.join(output_dir, benchmark_name)
    if not os.path.exists(POWER_OUTPUT_DIR):
        os.mkdir(POWER_OUTPUT_DIR)
    stdout_file_path = os.path.join(POWER_OUTPUT_DIR, QUERY_STDOUT_FILE)
    stderr_file_path = os.path.join(POWER_OUTPUT_DIR, QUERY_STDERR_FILE)
    # Run the command
    query_cmd = QUERY_COMMAND.format(
        QUERY_FREQUENCY=QUERY_FREQUENCY, QUERY_DEVICE=_get_cuda_device_id()
    ).split(" ")
    with open(stdout_file_path, "w") as stdout_file, open(
        stderr_file_path, "w"
    ) as stderr_file:
        QUERY_PROC = subprocess.Popen(
            query_cmd, stdout=stdout_file, stderr=stderr_file, start_new_session=True
        )


def power_chart_end():
    global QUERY_PROC, POWER_OUTPUT_DIR
    assert QUERY_PROC is not None, "Power query process cannot be None"
    # Kill the process
    QUERY_PROC.send_signal(signal.SIGINT)
    time.sleep(0.2)
    assert (
        QUERY_PROC.poll() is not None
    ), "Power query process must be killed to proceed"
    # generate the chart based on csv
    stdout_file_path = os.path.join(POWER_OUTPUT_DIR, QUERY_STDOUT_FILE)
    benchmark_name = os.path.basename(POWER_OUTPUT_DIR)
    device_name = torch.cuda.get_device_name(_get_cuda_device_id())
    _gen_power_charts(benchmark_name, device_name, stdout_file_path)
    logger.warning(f"[tritonbench][power] Power chart saved to {POWER_OUTPUT_DIR}.")
