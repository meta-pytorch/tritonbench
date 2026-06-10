"""
Compare TritonBench benchmarks across operators.

Runs a pair of benchmarks (LHS vs RHS) on each operator and compares
kernel selection and runtime. Results can be logged to the
TritonMultiOperatorBenchmarkComparisons Scuba table.

Shape source (priority order):
  1. --input-loader <file.json>    Single JSON shape file.
  2. --input-loader <dir/>         Directory of JSON shape files. Use
                                   --input-filter to select by filename.
  3. ai_infra.inductor_mm_shapes   Hive table with all MM shapes logged
                                   during torch.compile() (fbcode only).
                                   Use --hive-job-filter to narrow by
                                   MAST job name.

Examples:
    # Diode eval on all ops, shapes from Hive (default):
    buck2 run @mode/opt //pytorch/tritonbench/benchmarks:compare_benchmarks -- \
        --custom-bench diode --ops gemm,addmm,bmm,scaled_mm \
        --parse-autotune-logs --log-scuba --scuba-eval-id my-experiment

    # Diode eval, filter Hive shapes to a specific MAST job:
    buck2 run @mode/opt //pytorch/tritonbench/benchmarks:compare_benchmarks -- \
        --custom-bench diode --ops gemm --hive-job-filter <mast_job_name>

    # Custom shape file instead of Hive:
    buck2 run @mode/opt //pytorch/tritonbench/benchmarks:compare_benchmarks -- \
        --custom-bench diode --ops gemm \
        --input-loader fb/cmf/h100/shapes_mm.json

    # Directory of shape files, filtered by substring:
    buck2 run @mode/opt //pytorch/tritonbench/benchmarks:compare_benchmarks -- \
        --custom-bench diode --ops gemm \
        --input-loader fb/ads_omnifm_v4/ --input-filter layers_0

    # Non-Diode: explicit LHS/RHS benchmarks with custom shapes:
    buck2 run @mode/opt //pytorch/tritonbench/benchmarks:compare_benchmarks -- \
        --ops gemm --benchmarks-lhs pt2_matmul_maxautotune \
        --benchmarks-rhs pt2_matmul_maxautotune_v2 \
        --input-loader my_shapes.json

Assumes 1 GPU type (e.g. H100, MI350). GPU type defined by torchx job.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pytorch.tritonbench.benchmarks.compare_benchmarks.utils import (
    BenchmarkConfig,
    DEFAULT_METRICS,
    DEFAULT_OPS,
    detect_gpu,
    DiodeBenchmarkConfig,
)
from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.run_utils import run_in_task, run_one_operator

OP_TO_TRITONBENCH_OP: dict[str, str] = {
    "scaled_mm": "fp8_gemm",
}

if is_fbcode():
    from pytorch.tritonbench.benchmarks.compare_benchmarks.utils import log_benchmark
    from pytorch.tritonbench.tools.fb.inductor_analyzer.autotune_parser import (
        compare_benchmark_results,
        parse_benchmark_results,
    )


def build_op_args(
    op: str,
    config: BenchmarkConfig,
    benchmark_name: str,
    input_loader: Optional[str] = None,
) -> List[str]:
    """Build command-line arguments for a single operator benchmark."""
    tritonbench_op = OP_TO_TRITONBENCH_OP.get(op, op)
    args = [
        "--op",
        tritonbench_op,
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
        if config.diode_model_config is not None:
            args.extend(["--diode-model-config", config.diode_model_config])
        else:
            args.extend(["--diode-version", config.diode_version])
        args.extend(["--diode-topk", str(config.diode_topk)])

    return args


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
                    op_args.extend(
                        [
                            "--launch",
                            "pytorch.tritonbench.benchmarks.compare_benchmarks.run",
                        ]
                    )
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
        print(
            f"[Compare Benchmarks] WARNING: Benchmark {op} {benchmark_name} failed: {e}"
        )
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

    print(
        f"[Compare Benchmarks] Parsed {len(lhs_ops)} LHS benchmark, {len(rhs_ops)} RHS benchmark operations"
    )
    print("[Compare Benchmarks] Generating comparison between LHS and RHS benchmarks")

    return compare_benchmark_results(lhs_ops, rhs_ops)


def log_scuba(df: pd.DataFrame, config: BenchmarkConfig) -> None:
    if not config.scuba_eval_id:
        config.scuba_eval_id = f"{df['gpu']}_{int(time.time())}"
    print(
        f"[Compare Benchmarks] Logging comparison results to Scuba table triton_multi_operator_benchmark_comparisons with eval_id={config.scuba_eval_id}"
    )
    for op in df["op"].unique():
        op_df = df[df["op"] == op]
        log_benchmark(
            df=op_df,
            config=config,
        )


def _resolve_input_loaders(
    config: BenchmarkConfig,
    op: str,
    gpu: str,
    output_dir: Path,
) -> list[tuple[str, str]]:
    """Resolve input loader paths for a given op.

    Returns a list of (label, path) tuples. The label is used for the
    workload column in Scuba logging and log file naming.

    Priority:
      1. --input-loader <file>  -> single file
      2. --input-loader <dir>   -> scan directory, optionally filter by --input-filter
      3. ai_infra.inductor_mm_shapes Hive table (fbcode only)
    """
    if config.input_loader:
        loader_path = Path(config.input_loader)

        if loader_path.is_file():
            print(f"[Compare Benchmarks] Shape source: file '{config.input_loader}'")
            return [(loader_path.stem, config.input_loader)]

        if loader_path.is_dir():
            results = []
            for json_file in sorted(loader_path.glob("*.json")):
                if config.input_filter and config.input_filter not in json_file.name:
                    continue
                results.append((json_file.stem, str(json_file)))
            if results:
                print(
                    f"[Compare Benchmarks] Shape source: directory '{config.input_loader}' "
                    f"({len(results)} files"
                    f"{', filter=' + repr(config.input_filter) if config.input_filter else ''})"
                )
            else:
                print(
                    f"[Compare Benchmarks] WARNING: No matching JSON files in '{config.input_loader}'"
                    f"{' with filter=' + repr(config.input_filter) if config.input_filter else ''}"
                )
            return results

        print(
            f"[Compare Benchmarks] ERROR: --input-loader path '{config.input_loader}' "
            "does not exist"
        )
        return []

    if not is_fbcode():
        print(
            "[Compare Benchmarks] ERROR: No --input-loader provided, and Hive shape "
            "source is only available in fbcode"
        )
        return []

    print("[Compare Benchmarks] Shape source: ai_infra.inductor_mm_shapes (Hive)")
    from pytorch.tritonbench.benchmarks.compare_benchmarks.fb.hive_shapes import (
        get_shapes_from_hive,
    )

    hive_path = get_shapes_from_hive(
        gpu, op, output_dir, config.hive_job_filter, config.hive_max_shapes
    )
    if not hive_path:
        return []
    label = config.hive_job_filter or "inductor_mm_shapes"
    return [(label, hive_path)]


def run_benchmarks(config: BenchmarkConfig) -> None:
    """Main benchmark runner."""
    gpu = config.gpu if config.gpu else detect_gpu()

    print(f"[Compare Benchmarks] GPU: {gpu}")
    print(f"[Compare Benchmarks] Ops: {config.ops}")

    all_dfs: List[pd.DataFrame] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        for op in config.ops:
            if op not in config.benchmark_map:
                print(f"[Compare Benchmarks] WARNING: Unknown op: {op}, skipping")
                continue

            lhs_benchmark, rhs_benchmark = config.benchmark_map[op]

            input_loaders = _resolve_input_loaders(config, op, gpu, output_dir)
            if not input_loaders:
                print(
                    f"[Compare Benchmarks] WARNING: No shapes for op={op}, gpu={gpu}, skipping"
                )
                continue

            for label, input_loader in input_loaders:
                print(
                    f"[Compare Benchmarks] Running {op} ({label}): "
                    f"LHS={lhs_benchmark}, RHS={rhs_benchmark}"
                )

                lhs_log = run_benchmark_with_logs(
                    op, lhs_benchmark, config, output_dir, label, input_loader
                )
                rhs_log = run_benchmark_with_logs(
                    op, rhs_benchmark, config, output_dir, label, input_loader
                )

                if not lhs_log or not rhs_log:
                    print(
                        f"[Compare Benchmarks] WARNING: Either lhs_log (exists = {lhs_log is not None}) "
                        f"or rhs_log (exists = {rhs_log is not None}) does not exist"
                    )
                    continue

                if config.parse_autotune_logs and is_fbcode():
                    print(
                        "[Compare Benchmarks] Parsing LHS and RHS logs with autotune parser"
                    )
                    comparison_df = compare_results(lhs_log, rhs_log)

                    if not comparison_df.empty:
                        comparison_df["workload"] = label
                        comparison_df["gpu"] = gpu
                        comparison_df["op"] = op
                        comparison_df["lhs_benchmark_name"] = lhs_benchmark
                        comparison_df["rhs_benchmark_name"] = rhs_benchmark
                        all_dfs.append(comparison_df)

    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if not combined_df.empty:
        priority_cols = [
            "gpu",
            "workload",
            "op",
            "lhs_benchmark_name",
            "rhs_benchmark_name",
        ]
        other_cols = [c for c in combined_df.columns if c not in priority_cols]
        combined_df = combined_df[priority_cols + other_cols]

        if config.parse_autotune_logs and config.log_scuba and is_fbcode():
            log_scuba(combined_df, config)


def parse_args(args: List[str] = None) -> BenchmarkConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare benchmarks across operators, metrics, and workloads in TritonBench."
    )

    parser.add_argument(
        "--custom-bench",
        type=str,
        default=None,
        help=f"Custom benchmarking framework to use (e.g. diode). Default: None",
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
        "--input-loader",
        type=str,
        default=None,
        help="Path to a JSON shape file or a directory of JSON shape files. "
        "Overrides the default Hive shape source. Use --input-filter to "
        "select specific files when pointing to a directory.",
    )
    parser.add_argument(
        "--input-filter",
        type=str,
        default=None,
        help="Substring filter on filenames when --input-loader is a directory. "
        "Only JSON files whose name contains this string will be used.",
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

    if is_fbcode():
        parser.add_argument(
            "--hive-job-filter",
            type=str,
            default=None,
            help="Optional mast_job_name substring filter for Hive shape query. "
            "When set, only shapes from matching MAST jobs are used. "
            "Ignored when --input-loader is provided.",
        )
        parser.add_argument(
            "--hive-max-shapes",
            type=int,
            default=1500,
            help="Maximum number of shapes to evaluate per op when using Hive. "
            "Shapes are ranked by frequency (most common first). Default: 1500. "
            "Ignored when --input-loader is provided.",
        )
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

    base_configs = {
        "gpu": args.gpu,
        "ops": args.ops.split(","),
        "metrics": args.metrics.split(","),
        "input_loader": args.input_loader,
        "input_filter": args.input_filter,
        "parse_autotune_logs": args.parse_autotune_logs,
        "log_scuba": args.log_scuba,
        "scuba_eval_id": args.scuba_eval_id,
    }

    if is_fbcode():
        base_configs = {
            **base_configs,
            "hive_job_filter": args.hive_job_filter,
            "hive_max_shapes": args.hive_max_shapes,
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

    if len(args.ops.split(",")) != len(args.benchmarks_lhs.split(",")) or len(
        args.ops.split(",")
    ) != len(args.benchmarks_rhs.split(",")):
        raise ValueError(
            "Number of ops, benchmarks_lhs, and benchmarks_rhs must be equal"
        )

    benchmark_map = {
        op: (lhs, rhs)
        for op, lhs, rhs in zip(
            args.ops.split(","),
            args.benchmarks_lhs.split(","),
            args.benchmarks_rhs.split(","),
        )
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
