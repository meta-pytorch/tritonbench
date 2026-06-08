import logging
import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tritonbench.utils.triton_op import BenchmarkOperatorMetrics

logger: logging.Logger = logging.getLogger(__name__)

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
    assert os.path.exists(
        report_path
    ), f"The NCU report at {report_path} does not exist."
    _import_ncu_python_path()
    import ncu_report

    # save all kernels' metrics. {metric_name: [kernel1_metric_value, kernel2_metric_value, ...]}
    results = defaultdict(list)
    test_report = ncu_report.load_report(report_path)
    assert (
        test_report.num_ranges() > 0
    ), f"No profile data found in the NCU report at {report_path}"
    default_range = test_report.range_by_idx(0)
    assert (
        default_range.num_actions() > 0
    ), f"No profile data found in the default range of the NCU report at {report_path}"
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


def ncu_trace(
    op_task_args: List[str],
    output_dir: Path,
    range_name: str,
    replay: bool = False,
    profile_ir: bool = False,
    extend_ncu_args: Optional[List[str]] = None,
) -> str:
    """
    Run NCU on a single operator backend/input subprocess and return the path to
    the generated report.

    Args:
        op_task_args: The command line that runs a single operator backend/input
            in a subprocess.
        output_dir: The directory the report is written to.
        range_name: The NVTX range to profile (``range_start``/``range_end``).
        replay: Whether to generate a ``.ncu-rep`` report (vs. a ``.csv`` log).
        profile_ir: Whether to profile with TTGIR source locations.
        extend_ncu_args: Extra ``--metrics`` to collect; defaults to ``--set full``.
    """
    extend_ncu_args = (
        ["--metrics", ",".join(extend_ncu_args)]
        if extend_ncu_args
        else [
            "--set",
            "full",
        ]
    )
    # Disable DCGM
    disable_dyno_dcgm = [
        "sudo",
        "dyno",
        "dcgm_profiling",
        "--mute=true",
        "--duration=100000_s",
    ]
    disable_dcgm_service = [
        "sudo",
        "systemctl",
        "stop",
        "nvidia-dcgm",
    ]

    def service_exists(service_name):
        try:
            result = subprocess.run(
                ["systemctl", "status", service_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    if shutil.which("dyno") or service_exists("nvidia-dcgm"):
        dyno_result = subprocess.run(disable_dyno_dcgm).returncode
        systemctl_result = subprocess.run(disable_dcgm_service).returncode
        if dyno_result != 0 and systemctl_result != 0:
            logger.warning(
                "DCGM may not have been successfully disabled. Proceeding to collect NCU trace anyway..."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = ".csv" if not replay else ".ncu-rep"
    ncu_output_file = output_dir.joinpath(
        f"ncu_rep{'_ir' if profile_ir else ''}{ext}"
    ).resolve()
    ncu_args = [
        "ncu",
        "--nvtx",
        "--nvtx-include",
        # it is for range_start and range_end. no ending /.
        f"{range_name}",
        "--pm-sampling-max-passes",
        "4",
        "--warp-sampling-max-passes",
        "4",
        "--target-processes",
        "all",
        "--import-source",
        "yes",
    ]
    ncu_args.extend(extend_ncu_args)
    if replay:
        ncu_args.extend(
            [
                "-f",
                "-o",
                str(ncu_output_file.resolve()),
            ]
        )
    else:
        ncu_args.extend(
            [
                "--csv",
                "-f",
                "--log-file",
                str(ncu_output_file.resolve()),
            ]
        )
    ncu_args.extend(op_task_args)
    logger.info("Running NCU: %s", shlex.join(ncu_args))
    # Sometimes, `ncu --target-processes all` will fail with the message "Failed to connect to process". Setting
    # CUDA_INJECTION64_PATH=none seems to fix this issue.
    env = {**os.environ, "CUDA_INJECTION64_PATH": "none"}
    if profile_ir:
        env["USE_TTGIR_LOC"] = "1"
    subprocess.check_call(ncu_args, env=env)
    return str(ncu_output_file.resolve())


def analyze_ncu_metrics(
    required_metrics: List[str],
    op_task_args: List[str],
    output_dir: Path,
    range_name: str,
    metrics: "BenchmarkOperatorMetrics",
) -> None:
    """
    Collect NCU metrics (ncu_rep, ncu_rep_ir, or ncu_analyzer metrics) and
    populate `metrics` in place.

    Args:
        required_metrics: The metrics requested for this benchmark run.
        op_task_args: The command line that runs a single operator backend/input
            in a subprocess, forwarded to ``ncu_trace``.
        output_dir: The directory the NCU report is written to.
        range_name: The NVTX range to profile, forwarded to ``ncu_trace``.
        metrics: The ``BenchmarkOperatorMetrics`` instance to populate.
    """
    # ncu metrics (ncu_rep, ncu_rep_ir, or ncu_analyzer metrics)
    ncu_metrics = get_ncu_metrics(required_metrics)
    out = None
    if ncu_metrics or "ncu_rep" in required_metrics or "ncu_rep_ir" in required_metrics:
        profile_ir = "ncu_rep_ir" in required_metrics
        out = ncu_trace(
            op_task_args,
            output_dir,
            range_name,
            replay=True,
            extend_ncu_args=ncu_metrics,
            profile_ir=profile_ir,
        )
    # Read and update NCU metrics if any required metrics match the NCU metrics
    if ncu_metrics:
        ncu_analyzer_results = read_ncu_report(out, required_metrics)
        for metric_name, metric_value in ncu_analyzer_results.items():
            metrics.extra_metrics[metric_name] = metric_value
        if "arithmetic_intensity" in required_metrics:
            logger.warning("Arithmetic intensity only supports FP32 and FP64 for now.")
    if "ncu_rep" in required_metrics:
        metrics.ncu_rep = out
    if "ncu_rep_ir" in required_metrics:
        metrics.ncu_rep_ir = out
