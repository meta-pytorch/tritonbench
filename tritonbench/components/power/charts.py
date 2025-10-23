import csv
import logging
import os
import signal
import subprocess
import time

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_power_charts(
    benchmark_name: str, gpu_id: int, output_dir: str, power_csv_file: str
):
    # Read CSV
    with open(power_csv_file) as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)  # first row as header
        header = [col.strip() for col in header]
        data = {col: [] for col in header}

        for row in reader:
            for col, value in zip(header, row):
                value = float(value)
                data[col].append(value)

    # Generate synthetic time axis (100 ms per sample)
    n_samples = len(next(iter(data.values())))
    time = [
        (data["timestamp"][i] - data["timestamp"][0]) / 1000.0 for i in range(n_samples)
    ]  # seconds (0.1s = 100 ms)

    # Plot power chart
    plt.figure(figsize=(10, 6))
    for power_col in header[3:5]:
        plt.plot(time, data[power_col], label=power_col)
    plt.xlabel("Time (ms)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.title(
        f"[tritonbench] {benchmark_name} power consumption over time on device {gpu_id}"
    )
    plt.savefig(
        os.path.join(output_dir, f"{benchmark_name}-power.png"),
        dpi=300,
        bbox_inches="tight",
    )
    # Plot temp chart
    plt.figure(figsize=(10, 6))
    for temp_col in header[5:]:
        plt.plot(time, data[temp_col], label=temp_col)
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature (C)")
    plt.legend()
    plt.title(
        f"[tritonbench] {benchmark_name} temperature over time on device {gpu_id}"
    )
    plt.savefig(
        os.path.join(output_dir, f"{benchmark_name}-temp.png"),
        dpi=300,
        bbox_inches="tight",
    )
    # Plot frequency chart
    plt.figure(figsize=(10, 6))
    for temp_col in header[1:3]:
        plt.plot(time, data[temp_col], label=temp_col)
        plt.xlabel("Time (ms)")
        plt.ylabel("Frequency (MHz)")
    plt.legend()
    plt.title(f"[tritonbench] {benchmark_name} frequency over time on device {gpu_id}")
    plt.savefig(
        os.path.join(output_dir, f"{benchmark_name}-freq.png"),
        dpi=300,
        bbox_inches="tight",
    )


def plot_latencies(output_dir, gpu_id, metrics: "BenchmarkOperatorResult") -> None:
    op_name = metrics.op_name
    result_dict = metrics._get_result_dict()

    for x_val in result_dict:
        plt.figure(figsize=(10, 6))
        for backend in result_dict[x_val]:
            # ditch the first data points as it might be compiling and tuning the kernel
            latency_times = result_dict[x_val][backend].latency.times[1:]
            n_samples = len(latency_times)
            x = [i for i in range(n_samples)]  # seconds (0.1s = 100 ms)
            plt.scatter(x, latency_times, label=backend, s=10)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Latency (ms)")
        plt.title(
            f"[tritonbench] latency over time chart on device {gpu_id} for {op_name} input {x_val}"
        )
        plt.savefig(
            os.path.join(output_dir, f"{op_name}_input_{x_val}_latency.png"),
            dpi=300,
            bbox_inches="tight",
        )
