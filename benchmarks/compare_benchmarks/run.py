"""
Compare TritonBench benchmarks across operators and workloads.

Usage:
    buck2 run @mode/opt //pytorch/tritonbench/benchmarks:compare_benchmarks -- \
        --ops gemm,addmm --workloads cmf,igctr,omnifm

Assumes 1 GPU type (e.g. H100, MI350). GPU type defined by torchx job.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from pytorch.tritonbench.benchmarks.compare_benchmarks.utils import (
    BenchmarkConfig,
    DEFAULT_METRICS,
    DEFAULT_OPS,
    DEFAULT_WORKLOADS,
    DiodeBenchmarkConfig,
    detect_gpu,
    log_benchmark,
)
from pytorch.tritonbench.tools.fb.inductor_analyzer.autotune_parser import (
    compare_benchmark_results,
    parse_benchmark_results,
)
from tritonbench.utils.env_utils import is_fbcode

import pandas as pd
from tritonbench.utils.path_utils import REPO_PATH
from tritonbench.utils.run_utils import run_in_task, run_one_operator


def build_op_args(
    op: str,
    config: BenchmarkConfig,
    benchmark_name: str,
    input_loader: Optional[str] = None,
) -> List[str]:
    """Build command-line arguments for a single operator benchmark."""
    args = [
        "--op",
        op,
        "--metrics",
        ",".join(config.metrics),
        "--only",
        benchmark_name,
        "--force",
        "--allow-tf32",
        "True",
        "--input-loader",
        input_loader,
    ]

    if config.custom_bench == "diode" and "diode" in benchmark_name:
        args.extend(
            [
                "--diode-version",
                config.diode_version,
                "--diode-topk",
                str(config.diode_topk),
            ]
        )
        if config.diode_model_config is not None:
            args.extend(["--diode-model-config", config.diode_model_config])

    return args


def get_input_loader(
    gpu: str, workloads: List[str], op: str
) -> List[tuple[str, str]]:
    """
    Get input loader paths for the given GPU type and workloads.

    Args:
        gpu: GPU type (e.g., "H100", "MI350") - will be lowercased
        workloads: List of workload names (e.g., ["cmf", "omnifm_v4"]).
        op: Operator name to find shape files for (e.g., "gemm", "addmm")

    Returns:
        List of (workload, input_loader_path) tuples
        (e.g., [("cmf", "fb/cmf/h100/shapes_mm.json")])
    """
    gpu_lower = gpu.lower()
    input_configs_dir = Path(REPO_PATH) / "tritonbench" / "data" / "input_configs" / "fb"

    workloads = workloads if workloads else DEFAULT_WORKLOADS

    input_loaders: List[tuple[str, str]] = []

    for workload in workloads:
        workload_dir = input_configs_dir / workload / gpu_lower

        if not workload_dir.exists():
            print(
                f"[Compare Benchmarks] WARNING: No eval shapes for workload={workload}, gpu={gpu_lower}"
            )
            continue

        op_pattern = f"shapes_{op}.json" if op != "gemm" else "shapes_mm.json"
        shape_file = workload_dir / op_pattern
        if shape_file.exists():
            relative_path = f"fb/{workload}/{gpu_lower}/{op_pattern}"
            input_loaders.append((workload, relative_path))
            print(f"[Compare Benchmarks] Found input config: {relative_path}")
        else:
            print(
                f"[Compare Benchmarks] WARNING: No shape file {op_pattern} for op={op} in {workload}/{gpu_lower}"
            )

    return input_loaders


def run_benchmark_with_logs(
    op: str,
    benchmark_name: str,
    config: BenchmarkConfig,
    output_dir: Path,
    workload: str,
    input_loader: str,
) -> Optional[Path]:
    """
    Run a benchmark in a subprocess and capture autotune logs for parsing.
    Uses run_in_task to isolate each operator in its own subprocess.
    """
    log_file = output_dir / f"{op}_{benchmark_name}_{workload}.log"
    op_args = build_op_args(op, config, benchmark_name, input_loader)

    print(f"[Compare Benchmarks] Running {benchmark_name} on {op}")
    print(f"[Compare Benchmarks] Args: {' '.join(str(arg) for arg in op_args)}")

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(log_file, "w") as log_f:
            log_fd = log_f.fileno()
            os.dup2(log_fd, stdout_fd)
            os.dup2(log_fd, stderr_fd)

            try:
                # Only add --launch if running via MAST launcher (not direct compare_benchmarks binary)
                if "compare_benchmarks" not in sys.argv[0]:
                    op_args.extend([
                        "--launch",
                        "pytorch.tritonbench.benchmarks.compare_benchmarks.run",
                    ])
                op_args.append("--run-in-task")
                run_in_task(
                    op=op,
                    op_args=op_args,
                    benchmark_name=benchmark_name,
                )
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_stdout_fd, stdout_fd)
                os.dup2(saved_stderr_fd, stderr_fd)

        # Print log contents to stdout for MAST visibility
        if log_file.exists():
            with open(log_file, "r") as f:
                print(f.read())

    except Exception as e:
        print(f"[Compare Benchmarks] WARNING: Benchmark {op} {benchmark_name} failed: {e}")
    finally:
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

    if log_file.exists() and log_file.stat().st_size > 0:
        return log_file

    return None


def compare_results(
    lhs_log: Path,
    rhs_log: Path,
) -> pd.DataFrame:
    """
    Compare LHS vs. RHS benchmark results using autotune parser.

    Args:
        lhs_log: Path to combined LHS benchmark autotune log
        rhs_log: Path to combined RHS benchmark autotune log

    Returns a DataFrame with comparison results.
    """
    lhs_ops = parse_benchmark_results(str(lhs_log))
    rhs_ops = parse_benchmark_results(str(rhs_log))

    if not lhs_ops or not rhs_ops:
        print("[Compare Benchmarks] No valid operations to compare")
        return pd.DataFrame()

    print(f"[Compare Benchmarks] Parsed {len(lhs_ops)} LHS benchmark, {len(rhs_ops)} RHS benchmark operations")
    print("[Compare Benchmarks] Generating comparison between LHS and RHS benchmarks")

    return compare_benchmark_results(lhs_ops, rhs_ops)


def log_scuba(df: pd.DataFrame, config: BenchmarkConfig) -> None:
    if not config.scuba_eval_id:
        config.scuba_eval_id = f"{df["gpu"]}_{int(time.time())}"
    print(f"[Compare Benchmarks] Logging comparison results to Scuba table triton_multi_operator_benchmark_comparisons with eval_id={config.scuba_eval_id}")
    for op in df["op"].unique():
        op_df = df[df["op"] == op]
        log_benchmark(
            df=op_df,
            config=config,
        )


def run_benchmarks(config: BenchmarkConfig) -> None:
    """Main benchmark runner."""
    gpu = config.gpu if config.gpu else detect_gpu()

    print(f"[Compare Benchmarks] GPU: {gpu}")
    print(f"[Compare Benchmarks] Ops: {config.ops}")
    print(f"[Compare Benchmarks] Workloads: {config.workloads or 'all'}")

    all_dfs: List[pd.DataFrame] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        for op in config.ops:
            if op not in config.benchmark_map:
                print(f"[Compare Benchmarks] WARNING: Unknown op: {op}, skipping")
                continue

            lhs_benchmark, rhs_benchmark = config.benchmark_map[op]
            input_loaders = get_input_loader(gpu, config.workloads, op)

            if not input_loaders:
                print(f"[Compare Benchmarks] WARNING: No input configs found for op={op}, skipping")
                continue

            for workload, input_loader in input_loaders:
                print(
                    f"[Compare Benchmarks] Running {op} with workload={workload}, input_loader=tritonbench/data/input_configs/{input_loader}, LHS benchmark={lhs_benchmark}, RHS benchmark={rhs_benchmark}"
                )

                lhs_log = run_benchmark_with_logs(
                    op, lhs_benchmark, config, output_dir, workload, input_loader
                )
                rhs_log = run_benchmark_with_logs(
                    op, rhs_benchmark, config, output_dir, workload, input_loader
                )

                if not lhs_log or not rhs_log:
                    print(
                        f"[Compare Benchmarks] WARNING: Either lhs_log (exists = {lhs_log is not None}) or rhs_log (exists = {rhs_log is not None}) does not exist"
                    )
                    continue

                if config.parse_autotune_logs:
                    print(f"[Compare Benchmarks] Parsing LHS and RHS logs with autotune parser")
                    comparison_df = compare_results(lhs_log, rhs_log)

                    if not comparison_df.empty:
                        comparison_df["workload"] = workload
                        comparison_df["gpu"] = gpu
                        comparison_df["op"] = op
                        comparison_df["lhs_benchmark_name"] = lhs_benchmark
                        comparison_df["rhs_benchmark_name"] = rhs_benchmark
                        all_dfs.append(comparison_df)

    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if not combined_df.empty:  # reorder cols to have identifying information at the front
        priority_cols = ["gpu", "workload", "op", "lhs_benchmark_name", "rhs_benchmark_name"]
        other_cols = [c for c in combined_df.columns if c not in priority_cols]
        combined_df = combined_df[priority_cols + other_cols]

        if config.parse_autotune_logs and config.log_scuba:
            log_scuba(combined_df, config)


def parse_args(args: List[str] = None) -> BenchmarkConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare benchmarks across operators, metrics, and workloads in TritonBench.")

    parser.add_argument(
        "--custom-bench",
        type=str,
        default=None,
        help=f"Custom benchmarking framework to use (e.g. diode). Default: None"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help=f"GPU type override (e.g. h100). Auto-detected if not provided.",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=",".join(DEFAULT_OPS),
        help=f"Comma-separated list of operators. Default: {','.join(DEFAULT_OPS)}",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help=f"Comma-separated list of metrics. Default: {','.join(DEFAULT_METRICS)}",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default=None,
        help=f"Comma-separated list of workloads representing shapes to evaluate. Default: {','.join(DEFAULT_WORKLOADS)}",
    )
    parser.add_argument(
        "--benchmarks-lhs",
        type=str,
        default=None,
        help=f"Comma-separated list of benchmarks to run on the left-hand side. Default: None",
    )
    parser.add_argument(
        "--benchmarks-rhs",
        type=str,
        default=None,
        help=f"Comma-separated list of benchmarks to run on the right-hand side. Default: None",
    )
    parser.add_argument(
        "--parse-autotune-logs",
        action="store_true",
        default=False,
        help="Parse autotune logs and print comparison results to stdout. Omit to skip parsing.",
    )
    parser.add_argument(
        "--log-scuba",
        action="store_true",
        default=False,
        help="Log comparison results to TritonMultiOperatorBenchmarkComparisons Scuba table. Omit to skip logging.",
    )
    parser.add_argument(
        "--scuba-eval-id",
        type=str,
        default=None,
        help=f"Custom experiment name to log to Scuba. Default: gpu_timestamp (printed at the end of the run)",
    )

    # Diode-specific arguments
    parser.add_argument(
        "--diode-version",
        type=str,
        default="recommended",
        help="Diode model version to use. Default: recommended",
    )
    parser.add_argument(
        "--diode-model-config",
        type=str,
        default=None,
        help="JSON-serialized Diode ModelConfig. Overrides --diode-version.",
    )
    parser.add_argument(
        "--diode-topk",
        type=int,
        default=1,
        help="Top K kernel configs to return from Diode. Default: 1",
    )

    args = parser.parse_args(args)

    workloads = args.workloads.split(",") if args.workloads else None

    base_configs = {
        "gpu": args.gpu,
        "ops": args.ops.split(","),
        "metrics": args.metrics.split(","),
        "workloads": workloads,
        "parse_autotune_logs": args.parse_autotune_logs,
        "log_scuba": args.log_scuba,
        "scuba_eval_id": args.scuba_eval_id,
    }

    if args.custom_bench == "diode":
        if not is_fbcode():
            raise RuntimeError("Diode benchmarking is only supported in fbcode")

        benchmark_map = {
            "gemm": ("pt2_matmul_maxautotune", "pt2_matmul_maxautotune_diode"),
            "addmm": ("pt2_addmm_maxautotune", "pt2_addmm_maxautotune_diode"),
            "bmm": ("pt2_bmm_maxautotune", "pt2_bmm_maxautotune_diode"),
            "scaled_mm": ("pt2_fp8_gemm", "pt2_fp8_gemm_maxautotune_diode"),
        }
        return DiodeBenchmarkConfig(
            **base_configs,
            benchmark_map=benchmark_map,
            diode_version=args.diode_version,
            diode_model_config=args.diode_model_config,
            diode_topk=args.diode_topk,
        )

    if len(args.ops.split(",")) != len(args.benchmarks_lhs.split(",")) or len(args.ops.split(",")) != len(args.benchmarks_rhs.split(",")):
        raise ValueError("Number of ops, benchmarks_lhs, and benchmarks_rhs must be equal")

    benchmark_map = {
        op: (lhs, rhs) for op, lhs, rhs in zip(args.ops.split(","), args.benchmarks_lhs.split(","), args.benchmarks_rhs.split(","))
    }

    return BenchmarkConfig(
        **base_configs,
        benchmark_map=benchmark_map,
    )


def run(args: List[str] = None) -> None:
    """Entry point for running compare_benchmarks."""
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--run-in-task", action="store_true")
    _args, extra_args = _parser.parse_known_args(args)

    if _args.run_in_task:
        run_one_operator(extra_args)
        exit(0)

    config = parse_args(args)
    run_benchmarks(config)


if __name__ == "__main__":
    run()
