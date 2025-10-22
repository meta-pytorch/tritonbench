import csv
import logging
import os
import signal
import subprocess
import time

import matplotlib.pyplot as plt
import torch

from tritonbench.components.power.power_manager import (
    DEFAULT_QUERY_INTERVAL,
    PowerManagerTask,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_cuda_device_id():
    return torch.cuda.current_device()


def gen_power_charts(benchmark_name: str, device_name: str, power_csv_file: str):
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


def gen_metrics_charts(metrics):
    for x_val in metrics:
        for backend in metrics[x_val]:
            # Generate charts for each metric and backend
            # Generate synthetic time axis (100 ms per sample)
            n_samples = len(next(iter(metrics[x_val][backend].values())))
            x_val = [i for i in range(n_samples)]  # seconds (0.1s = 100 ms)
            plt.figure(figsize=(10, 6))
            plt.plot(x_val, metrics[x_val][backend], label=backend)
            plt.title(
                f"[tritonbench] {metrics.name} {backend} latency on input {x_val} over time"
            )


def power_chart_end(power_manager_task):
    assert power_manager_task is not None, "Power manager task cannot be None"
    power_manager_task.stop_monitor()
    # generate the chart based on csv
    metrics = power_manager_task.metrics
    power_output_dir = power_manager_task.power_output_dir
    power_csv_path = power_manager_task.power_csv_path
    benchmark_name = os.path.basename(power_output_dir)
    device_name = torch.cuda.get_device_name(_get_cuda_device_id())
    _gen_power_charts(benchmark_name, device_name, power_csv_path)
    logger.warning(f"[tritonbench][power] Power chart saved to {power_output_dir}.")
