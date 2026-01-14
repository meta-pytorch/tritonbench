import os
import shutil
import sys
from collections import defaultdict
from typing import List

"""
A dictionary mapping short metric names to their corresponding NVIDIA Nsight Compute
(NCU) metric names. Don't directly use the NCU metric names in the code, use these short
names instead. This mapping can help us manage the metrics we use in the benchmark.
"""
short_ncu_metric_name = {
    "inst_executed_ffma_peak": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "inst_executed_dfma_peak": "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained",
    "inst_executed_fadd": "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
    "inst_executed_fmul": "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
    "inst_executed_ffma": "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed",
    "inst_executed_dadd": "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed",
    "inst_executed_dmul": "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed",
    "inst_executed_dfma": "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed",
    "dram_bytes_write": "dram__bytes_write.sum",
    "dram_bytes_read": "dram__bytes_read.sum",
    "dram_bytes_per_second": "dram__bytes.sum.per_second",
    "dram_bytes": "dram__bytes.sum",
    "sm_freq": "smsp__cycles_elapsed.avg.per_second",
    "dram_bandwidth": "dram__bytes.sum.per_second",
    "duration": "gpu__time_duration.sum",
}
# A dictionary mapping benchmark metric names to their corresponding short NCU metric
# names.
bench_metric_to_short_ncu_metric = {
    "memory_traffic": ["dram_bytes_write", "dram_bytes_read"],
    "arithmetic_intensity": [
        "inst_executed_ffma_peak",
        "inst_executed_dfma_peak",
        "inst_executed_fadd",
        "inst_executed_fmul",
        "inst_executed_ffma",
        "inst_executed_dadd",
        "inst_executed_dmul",
        "inst_executed_dfma",
        "dram_bytes_write",
        "dram_bytes_read",
        "dram_bytes",
        "sm_freq",
        "dram_bandwidth",
        "duration",
    ],
    "ncu_tflops": [
        "inst_executed_fadd",
        "inst_executed_fmul",
        "inst_executed_ffma",
        "inst_executed_dadd",
        "inst_executed_dmul",
        "inst_executed_dfma",
        "duration",
        "sm_freq",
    ],
}


def get_ncu_metrics(metrics: List[str]) -> List[str]:
    """
    This function returns a list of all the NCU metrics used in the benchmark.

    Returns:
        list: A list of all the NCU metrics used in the benchmark.
    """
    ncu_metrics = []
    for (
        bench_metric,
        short_ncu_metrics,
    ) in bench_metric_to_short_ncu_metric.items():
        # Only process metrics that are required
        if bench_metric in metrics:
            # For each short metric name in the list of metrics for this benchmark metric
            for short_ncu_metric in short_ncu_metrics:
                # Get the full NCU metric name and add it to our list
                full_metric_name = short_ncu_metric_name[short_ncu_metric]
                ncu_metrics.append(full_metric_name)
    return ncu_metrics


def _import_ncu_python_path():
    """
    This function modifies the Python path to include the NVIDIA Nsight Compute (NCU) Python modules.
    It searches for the 'ncu' command in the system PATH, determines its location, and appends the
    'extras/python' directory to the Python path.

    Raises:
        FileNotFoundError: If the 'ncu' command is not found in the system PATH.
        FileNotFoundError: If the 'extras/python' directory does not exist in the determined NCU path.
    """
    ncu_path = shutil.which("ncu")
    if not ncu_path:
        raise FileNotFoundError("Could not find 'ncu' command in PATH.")
    ncu_path = os.path.dirname(ncu_path)
    if not os.path.exists(os.path.join(ncu_path, "extras/python")):
        raise FileNotFoundError(
            f"'extras/python' does not exist in the provided ncu_path: {ncu_path}"
        )
    sys.path.append(os.path.join(ncu_path, "extras/python"))


def get_mem_traffic(kernel):
    return (
        kernel.metric_by_name(short_ncu_metric_name["dram_bytes_read"]).value(),
        kernel.metric_by_name(short_ncu_metric_name["dram_bytes_write"]).value(),
    )


def get_duration(kernel):
    return kernel.metric_by_name(short_ncu_metric_name["duration"]).value()


def get_flops(kernel):
    """
    Calculate the achieved floating point operations per second (FLOPS) for both FP32 and FP64 operations.

    This function calculates FLOPS by:
    1. Summing up the achieved ADD, MUL and FMA operations (FMA counts as 2 operations)
    2. Multiplying by the SM frequency to get operations per second

    Args:
        kernel: An NCU kernel object containing the profiling metrics

    Returns:
        tuple: A pair of (fp32_flops, fp64_flops) containing:
            - fp32_flops: Achieved single precision (FP32) FLOPS
            - fp64_flops: Achieved double precision (FP64) FLOPS

    Reference:
        Implementation based on NVIDIA Nsight Compute's SpeedOfLight_Roofline.py and
        SpeedOfLight_RooflineChart.section

    TODO: Add Tensor FLOPS and Half Precision FLOPS
    """
    fp32_add_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_fadd"]
    ).value()
    fp32_mul_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_fmul"]
    ).value()
    fp32_fma_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_ffma"]
    ).value()
    fp32_achieved = fp32_add_achieved + fp32_mul_achieved + 2 * fp32_fma_achieved
    fp64_add_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_dadd"]
    ).value()
    fp64_mul_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_dmul"]
    ).value()
    fp64_fma_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_dfma"]
    ).value()
    fp64_achieved = fp64_add_achieved + fp64_mul_achieved + 2 * fp64_fma_achieved
    sm_freq = kernel.metric_by_name(short_ncu_metric_name["sm_freq"]).value()
    fp32_flops = fp32_achieved * sm_freq
    fp64_flops = fp64_achieved * sm_freq
    return fp32_flops, fp64_flops


def get_arithmetic_intensity(kernel):
    dram_bandwidth = kernel.metric_by_name(
        short_ncu_metric_name["dram_bandwidth"]
    ).value()
    fp32_flops, fp64_flops = get_flops(kernel)
    fp32_arithmetic_intensity = fp32_flops / dram_bandwidth
    fp64_arithmetic_intensity = fp64_flops / dram_bandwidth
    return fp32_arithmetic_intensity, fp64_arithmetic_intensity


def read_ncu_report(report_path: str, required_metrics: List[str]):
    assert os.path.exists(report_path), (
        f"The NCU report at {report_path} does not exist."
    )
    _import_ncu_python_path()
    import ncu_report

    # save all kernels' metrics. {metric_name: [kernel1_metric_value, kernel2_metric_value, ...]}
    results = defaultdict(list)
    test_report = ncu_report.load_report(report_path)
    assert test_report.num_ranges() > 0, (
        f"No profile data found in the NCU report at {report_path}"
    )
    default_range = test_report.range_by_idx(0)
    assert default_range.num_actions() > 0, (
        f"No profile data found in the default range of the NCU report at {report_path}"
    )
    total_duration = 0
    total_dram_bytes = 0
    weighted_fp32_ai_sum = 0
    weighted_fp64_ai_sum = 0
    for i in range(default_range.num_actions()):
        kernel = default_range.action_by_idx(i)
        if set(required_metrics) & {"arithmetic_intensity", "ncu_tflops"}:
            duration = get_duration(kernel)
            results["durations"].append(duration)
            total_duration += duration
        if "memory_traffic" in required_metrics:
            results["memory_traffic_raw"].append(get_mem_traffic(kernel))
        if "arithmetic_intensity" in required_metrics:
            dram_bytes = kernel.metric_by_name(
                short_ncu_metric_name["dram_bytes"]
            ).value()
            fp32_ai, fp64_ai = get_arithmetic_intensity(kernel)
            weighted_fp32_ai_sum += fp32_ai * dram_bytes
            weighted_fp64_ai_sum += fp64_ai * dram_bytes
            # do not use the arithmetic_intensity_raw in benchmark metric argument
            # because metric printer will only print the first element of the list
            results["arithmetic_intensity_raw"].append((fp32_ai, fp64_ai))
            total_dram_bytes += dram_bytes
        if "ncu_tflops" in required_metrics:
            results["ncu_tflops_raw"].append(get_flops(kernel))

    if "memory_traffic" in required_metrics:
        memory_traffic_read = [item[0] for item in results["memory_traffic_raw"]]
        memory_traffic_write = [item[1] for item in results["memory_traffic_raw"]]
        results["memory_traffic_read_sum"] = sum(memory_traffic_read)
        results["memory_traffic_write_sum"] = sum(memory_traffic_write)
        results["memory_traffic"] = (
            results["memory_traffic_read_sum"],
            results["memory_traffic_write_sum"],
        )
    if "arithmetic_intensity" in required_metrics:
        results["weighted_fp32_arithmetic_intensity"] = (
            weighted_fp32_ai_sum / total_dram_bytes
        )
        results["weighted_fp64_arithmetic_intensity"] = (
            weighted_fp64_ai_sum / total_dram_bytes
        )
        results["arithmetic_intensity"] = (
            results["weighted_fp32_arithmetic_intensity"],
            results["weighted_fp64_arithmetic_intensity"],
        )
    if "ncu_tflops" in required_metrics:
        assert results["durations"], "No kernel durations found in the NCU report."
        weighted_fp32_flops_sum = sum(
            flop[0] * dur
            for flop, dur in zip(results["ncu_tflops_raw"], results["durations"])
        )
        weighted_fp64_flops_sum = sum(
            flop[1] * dur
            for flop, dur in zip(results["ncu_tflops_raw"], results["durations"])
        )
        weighted_fp32_tflops_sum = weighted_fp32_flops_sum / (
            10**12
        )  # Convert to TFLOPS
        weighted_fp64_tflops_sum = weighted_fp64_flops_sum / (
            10**12
        )  # Convert to TFLOPS
        results["ncu_tflops"] = (
            weighted_fp32_tflops_sum / total_duration,
            weighted_fp64_tflops_sum / total_duration,
        )
    return results
