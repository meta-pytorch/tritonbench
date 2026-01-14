import csv
import os
import subprocess
from typing import Dict, List

# The nsys metrics to the reports. The value is the list of reports of nsys.
nsys_metrics_to_reports = {
    # the sum of kernel execution time
    "nsys_gpu_kernel_sum": ["nvtx_kern_sum", "nvtx_sum"],
    # the overhead of kernel launch
    "nsys_launch_overhead": ["nvtx_kern_sum", "nvtx_sum"],
    # the names of kernels
    "nsys_kernel_names": ["nvtx_kern_sum"],
    # the durations of kernels
    "nsys_kernel_durations": ["nvtx_kern_sum"],
    # the duration of nvtx range
    "nsys_nvtx_range_duration": ["nvtx_sum"],
    # the number of kernels
    "nsys_num_of_kernels": ["nvtx_kern_sum"],
}


def get_nsys_metrics(metrics: List[str]) -> List[str]:
    nsys_metrics = []
    for metric_name in nsys_metrics_to_reports.keys():
        if metric_name in metrics:
            nsys_metrics.append(metric_name)
    return nsys_metrics


def read_nsys_report(
    report_path: str, required_metrics: List[str]
) -> Dict[str, List[float]]:
    assert os.path.exists(report_path), (
        f"The nsys report at {report_path} does not exist."
    )
    reports_required = []
    for metric in required_metrics:
        if metric in nsys_metrics_to_reports:
            reports_required.extend(nsys_metrics_to_reports[metric])
    reports_required = list(set(reports_required))
    assert reports_required, "No nsys reports required"
    cmd = f"nsys stats --report {','.join(reports_required)} --timeunit ns --force-export=true --format csv --output . --force-overwrite=true {report_path}"
    try:
        subprocess.check_call(
            cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to run nsys command: {cmd}\nError: {e}")
        raise e
    # Get the base path and filename without extension
    base_path = os.path.dirname(report_path)
    base_name = os.path.splitext(os.path.basename(report_path))[0]

    results = {}
    csv_contents = {}

    for report in reports_required:
        csv_path = os.path.join(base_path, f"{base_name}_{report}.csv")
        if not os.path.exists(csv_path):
            raise RuntimeError(f"Expected CSV report not found at {csv_path}")

        # Read CSV using DictReader
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            csv_contents[report] = list(reader)
    kernel_duration = []
    kernel_names = []
    sum_kernel_duration = 0
    nvtx_range_duration = 0
    if "nvtx_kern_sum" in csv_contents:
        # gpu kernel execution time summary
        for row in csv_contents["nvtx_kern_sum"]:
            # use ms as the unit
            kernel_duration.append(float(row["Total Time (ns)"]) / 1_000_000)
            kernel_names.append(row["Kernel Name"])
        sum_kernel_duration = sum(kernel_duration)
    if "nvtx_sum" in csv_contents:
        # It is supposed to be only one row. The nvtx range is `:tritonbench_range`
        assert len(csv_contents["nvtx_sum"]) == 1
        nvtx_range_duration = (
            float(csv_contents["nvtx_sum"][0]["Total Time (ns)"]) / 1_000_000
        )

    # Define mapping of metrics to their values. The keys must be in nsys_bench_metrics.
    metrics_map = {
        # Because tritonbench takes the median of numerical values, we need to convert
        # the list of floats to a list of strings.
        "nsys_kernel_durations": [str(duration) for duration in kernel_duration],
        "nsys_kernel_names": kernel_names,
        "nsys_gpu_kernel_sum": sum_kernel_duration,
        "nsys_nvtx_range_duration": nvtx_range_duration,
        "nsys_launch_overhead": nvtx_range_duration - sum_kernel_duration,
        "nsys_num_of_kernels": len(kernel_names),
    }
    # Verify that metrics_map keys match nsys_metrics_to_reports keys
    assert set(metrics_map.keys()) == set(nsys_metrics_to_reports.keys()), (
        f"Mismatch between metrics_map keys and nsys_metrics_to_reports keys.\n"
        f"metrics_map keys: {set(metrics_map.keys())}\n"
        f"nsys_metrics_to_reports keys: {set(nsys_metrics_to_reports.keys())}"
    )
    # Add only requested metrics to results
    results.update(
        {
            metric: metrics_map[metric]
            for metric in required_metrics
            if metric in metrics_map
        }
    )

    return results
