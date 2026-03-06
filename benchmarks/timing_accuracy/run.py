import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import triton

from ..common import setup_tritonbench_cwd, run_benchmark_config_ci

setup_tritonbench_cwd()

from tritonbench.utils.parser import get_parser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class MethodStats:
    """Statistics for a benchmarking method."""

    method_name: str
    n_tests: int
    warmup: int
    rep: int
    benchmark_time: float
    intra_test_medians: List[float] = field(default_factory=list)
    intra_test_stds: List[float] = field(default_factory=list)
    intra_test_cvs: List[float] = field(default_factory=list)
    intra_test_mins: List[float] = field(default_factory=list)
    intra_test_maxs: List[float] = field(default_factory=list)
    all_samples: List[List[float]] = field(default_factory=list)

    @property
    def inter_test_median(self) -> float:
        return (
            statistics.median(self.intra_test_medians)
            if self.intra_test_medians
            else 0.0
        )

    @property
    def inter_test_std(self) -> float:
        return (
            statistics.stdev(self.intra_test_medians)
            if len(self.intra_test_medians) > 1
            else 0.0
        )

    @property
    def inter_test_cv(self) -> float:
        median = self.inter_test_median
        return self.inter_test_std / median if median > 0 else 0.0

    @property
    def avg_intra_test_cv(self) -> float:
        return statistics.mean(self.intra_test_cvs) if self.intra_test_cvs else 0.0

    @property
    def avg_intra_test_std(self) -> float:
        return statistics.mean(self.intra_test_stds) if self.intra_test_stds else 0.0

    @property
    def inter_test_min(self) -> float:
        return min(self.intra_test_mins) if self.intra_test_mins else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "n_tests": self.n_tests,
            "warmup": self.warmup,
            "rep": self.rep,
            "benchmark_time_s": self.benchmark_time,
            "intra_test": {
                "avg_median_ms": (
                    statistics.mean(self.intra_test_medians)
                    if self.intra_test_medians
                    else 0
                ),
                "avg_std_ms": (
                    statistics.mean(self.intra_test_stds) if self.intra_test_stds else 0
                ),
                "avg_cv": self.avg_intra_test_cv,
                "avg_min_ms": (
                    statistics.mean(self.intra_test_mins) if self.intra_test_mins else 0
                ),
                "avg_max_ms": (
                    statistics.mean(self.intra_test_maxs) if self.intra_test_maxs else 0
                ),
            },
            "inter_test": {
                "median_ms": self.inter_test_median,
                "std_ms": self.inter_test_std,
                "cv": self.inter_test_cv,
                "min_ms": self.inter_test_min,
            },
            "intra_test_medians": self.intra_test_medians,
            "intra_test_stds": self.intra_test_stds,
            "intra_test_cvs": self.intra_test_cvs,
            "intra_test_mins": self.intra_test_mins,
            "intra_test_maxs": self.intra_test_maxs,
            "all_samples": self.all_samples,
        }


def run_do_bench_standard(fn: Callable, warmup: int, rep: int) -> List[float]:
    return triton.runtime.driver.active.get_benchmarker()(
        fn, warmup=warmup, rep=rep, return_mode="all"
    )


def run_do_bench_profiler(fn: Callable, warmup: int, rep: int) -> List[float]:
    from tritonbench.components.do_bench.run import _do_bench_profiler

    return _do_bench_profiler(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="all",
        use_cudagraph=False,
        skip_cache_clearing=False,
    )


def run_do_bench_cudagraph(fn: Callable, warmup: int, rep: int) -> List[float]:
    from tritonbench.components.do_bench.run import _do_bench_cudagraph_with_cache_clear

    return _do_bench_cudagraph_with_cache_clear(
        fn, rep=rep, return_mode="all", skip_cache_clearing=False
    )


def run_do_bench_entropy(fn: Callable, warmup: int, rep: int) -> List[float]:
    from tritonbench.components.do_bench.run import _do_bench_entropy

    return _do_bench_entropy(fn, warmup=warmup, rep=rep, return_mode="all", repcnt=rep)


def run_do_bench_profiler_cudagraph(fn: Callable, warmup: int, rep: int) -> List[float]:
    from tritonbench.components.do_bench.run import _do_bench_profiler

    return _do_bench_profiler(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="all",
        use_cudagraph=True,
        skip_cache_clearing=False,
    )


def run_do_bench_gpu_events(fn: Callable, warmup: int, rep: int) -> List[float]:
    from tritonbench.components.do_bench.gpu_events import do_bench_events

    return do_bench_events(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="all",
        skip_cache_clearing=False,
    )


BENCHMARK_METHODS = {
    "triton_do_bench": ("triton do_bench (standard)", run_do_bench_standard),
    "profiler": ("profiler", run_do_bench_profiler),
    "triton_do_bench_cudagraph": ("Triton do_bench CUDA graph", run_do_bench_cudagraph),
    "entropy": ("entropy-based", run_do_bench_entropy),
    "profiler_cudagraph": ("profiler + CUDA graph", run_do_bench_profiler_cudagraph),
    "gpu_events": ("GPU events", run_do_bench_gpu_events),
}


def benchmark_method(
    method_name: str,
    method_fn: Callable,
    kernel_fn: Callable,
    n_tests: int,
    warmup: int,
    rep: int,
    sleep_between_tests: float = 0.5,
    verbose: bool = True,
) -> MethodStats:
    stats = MethodStats(
        method_name=method_name,
        benchmark_time=0.0,
        n_tests=n_tests,
        warmup=warmup,
        rep=rep,
    )

    start_ts = time.time_ns()
    for test_idx in range(n_tests):
        if verbose:
            logger.info(f"  Testing {test_idx + 1}/{n_tests}...")

        if test_idx > 0 and sleep_between_tests > 0:
            time.sleep(sleep_between_tests)

        try:
            samples = method_fn(kernel_fn, warmup=warmup, rep=rep)
            if not samples:
                logger.warn("WARNING: No samples returned!")
                continue

            median = statistics.median(samples)
            mean = statistics.mean(samples)
            std = statistics.stdev(samples) if len(samples) > 1 else 0.0
            cv = std / median if median > 0 else 0.0

            stats.intra_test_medians.append(median)
            stats.intra_test_stds.append(std)
            stats.intra_test_cvs.append(cv)
            stats.intra_test_mins.append(min(samples))
            stats.intra_test_maxs.append(max(samples))
            stats.all_samples.append(samples)

            if verbose:
                logger.info(
                    f"median={median:.4f}ms, mean={mean:.4f}ms, std={std:.4f}ms, cv={cv:.4f}"
                )

        except Exception as e:
            logger.error(f"ERROR: {e}")
            import traceback

            traceback.print_exc()
    end_ts = time.time_ns()
    stats.benchmark_time = (end_ts - start_ts) / 1e9

    return stats


def print_summary_table(results: Dict[str, MethodStats], operation_name: str):
    print("\n" + "=" * 120)
    print(f"SUMMARY: Latency Noise Comparison for '{operation_name}'")
    print("=" * 120)

    header = f"{'Method':<25} | {'Benchmark Time (s)':<25} | {'Min (ms)':<10} | {'Median (ms)':<12} | {'Intra-Std (ms)':<14} | {'Intra-CV':<10} | {'Inter-CV':<10} | {'Inter-Std (ms)':<14}"
    print(header)
    print("-" * 120)

    for method_name, stats in sorted(results.items(), key=lambda x: x[1].inter_test_cv):
        print(
            f"{method_name:<25} | "
            f"{stats.benchmark_time:<10.1f} | "
            f"{stats.inter_test_min:<10.4f} | "
            f"{stats.inter_test_median:<12.4f} | "
            f"{stats.avg_intra_test_std:<14.4f} | "
            f"{stats.avg_intra_test_cv:<10.4f} | "
            f"{stats.inter_test_cv:<10.4f} | "
            f"{stats.inter_test_std:<14.4f}"
        )

    print("=" * 120)
    print(
        "\nLegend: Intra-CV = noise within each run, Inter-CV = noise between runs. Lower = better.\n"
    )


def _run(
    args: argparse.Namespace,
    tb_args: argparse.Namespace,
    extra_args: List[str],
):
    device_name = torch.cuda.get_device_name()
    logger.info(f"Loading operator: {tb_args.op}")

    # Use existing tritonbench infrastructure to load operator
    from tritonbench.utils.run_utils import load_operator_by_args
    from tritonbench.utils.triton_op import _translate_mode

    _translate_mode(tb_args)

    tb_arg_list = [
        "--op",
        tb_args.op,
        "--mode",
        tb_args.mode,
        "--precision",
        tb_args.precision,
        "--device",
        tb_args.device,
        "--input-id",
        tb_args.input_id,
        "--num-inputs",
        "1",
    ]
    if tb_args.only:
        tb_arg_list.extend(["--only", tb_args.only])
    tb_arg_list.extend(extra_args)

    opbench = load_operator_by_args(tb_arg_list)
    tb_args = opbench.tb_args
    opbench.example_inputs = opbench.get_example_inputs()

    if opbench.example_inputs is None:
        logger.error(f"ERROR: No example inputs for operator '{args.op}'")
        sys.exit(1)

    # Get the benchmark function
    if tb_args.only:
        assert "," not in tb_args.only, "ERROR: Only one backend can be specified."
        backend_name = tb_args.only
        bench_fn_factory = getattr(opbench, backend_name, None)
        if bench_fn_factory is None:
            logger.error(f"ERROR: Backend '{backend_name}' not found")
            sys.exit(1)
    else:
        from tritonbench.utils.triton_op import REGISTERED_BENCHMARKS

        registered = REGISTERED_BENCHMARKS.get(opbench.name, {})
        if not registered:
            logger.error(f"ERROR: No benchmarks registered for '{args.op}'")
            sys.exit(1)
        backend_name = list(registered.keys())[0]
        bench_fn_factory = getattr(opbench, backend_name)

    example_inputs = opbench.example_inputs
    if isinstance(example_inputs, dict):
        kernel_fn = bench_fn_factory(**example_inputs)
    else:
        kernel_fn = bench_fn_factory(*example_inputs)

    operation_name = (
        f"{tb_args.op}_{tb_args.mode}:{backend_name}_input_id={tb_args.input_id}"
    )
    logger.info(
        f"[timing_accuracy] Device: {device_name}, Op: {tb_args.op}, Mode: {tb_args.mode}, Backend: {backend_name}, Tests: {args.n_tests}, Warmup: {tb_args.warmup}, Reps: {tb_args.rep}\n"
    )

    # Determine methods to run
    if args.methods == "all":
        methods_to_run = list(BENCHMARK_METHODS.keys())
    else:
        methods_to_run = [m.strip() for m in args.methods.split(",")]
        for m in methods_to_run:
            if m not in BENCHMARK_METHODS:
                logger.error(
                    f"ERROR: Unknown method '{m}'. Available: {', '.join(BENCHMARK_METHODS.keys())}"
                )
                sys.exit(1)

    # Run benchmarks
    results: Dict[str, MethodStats] = {}
    for method_key in methods_to_run:
        method_display_name, method_fn = BENCHMARK_METHODS[method_key]
        logger.info(f"\n{'=' * 60}\nBenchmarking: {method_display_name}\n{'=' * 60}")

        stats = benchmark_method(
            method_name=method_key,
            method_fn=method_fn,
            kernel_fn=kernel_fn,
            n_tests=args.n_tests,
            warmup=tb_args.warmup,
            rep=tb_args.rep,
            sleep_between_tests=args.sleep_between_tests,
            verbose=not args.quiet,
        )
        results[method_display_name] = stats

    print_summary_table(results, operation_name)

    if not args.output_json:
        return

    output_data = {
        "config": {
            "device": device_name,
            "operator": tb_args.op,
            "backend": backend_name,
            "input_id": tb_args.input_id,
            "mode": tb_args.mode,
            "precision": tb_args.precision,
            "n_tests": args.n_tests,
            "rep": tb_args.rep,
            "warmup": tb_args.warmup,
        },
        "results": {name: stats.to_dict() for name, stats in results.items()},
    }
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to: {args.output_json}")


def transform_single_result(result) -> Dict[str, Any]:
    out = {}
    op_name = result["config"]["operator"]
    mode = result["config"]["mode"]
    backend = result["config"]["backend"]
    input_id = result["config"]["input_id"]
    prefix = f"tritonbench_{op_name}_{mode}[input_id_{input_id}-{backend}]-"
    out[f"{prefix}precision"] = result["config"]["precision"]
    out[f"{prefix}warmup"] = result["config"]["warmup"]
    out[f"{prefix}ntests"] = result["config"]["n_tests"]
    out[f"{prefix}rep"] = result["config"]["rep"]
    for method_name, method_stats in result["results"].items():
        method_key = method_stats["method_name"]
        out[f"{prefix}{method_key}-benchmark_time_s"] = method_stats[
            "benchmark_time_s"
        ]
        out[f"{prefix}{method_key}-intra_test_avg_median_ms"] = method_stats[
            "intra_test"
        ]["avg_median_ms"]
        out[f"{prefix}{method_key}-intra_test_avg_std_ms"] = method_stats[
            "intra_test"
        ]["avg_std_ms"]
        out[f"{prefix}{method_key}-intra_test_avg_cv"] = method_stats["intra_test"][
            "avg_cv"
        ]
        out[f"{prefix}{method_key}-intra_test_avg_min_ms"] = method_stats[
            "intra_test"
        ]["avg_min_ms"]
        out[f"{prefix}{method_key}-intra_test_avg_max_ms"] = method_stats[
            "intra_test"
        ]["avg_max_ms"]
        out[f"{prefix}{method_key}-inter_test_median_ms"] = method_stats[
            "inter_test"
        ]["median_ms"]
        out[f"{prefix}{method_key}-inter_test_std_ms"] = method_stats["inter_test"][
            "std_ms"
        ]
        out[f"{prefix}{method_key}-inter_test_cv"] = method_stats["inter_test"][
            "cv"
        ]
        out[f"{prefix}{method_key}-inter_test_min_ms"] = method_stats["inter_test"][
            "min_ms"
        ]
    return out


def run(args: Optional[List[str]] = None):
    if not args:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Compare latency noise across benchmarking methods",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--n-tests", type=int, default=10, help="Benchmark runs per method"
    )
    parser.add_argument("--sleep-between-tests", type=float, default=0.5)
    parser.add_argument(
        "--bench-methods",
        type=str,
        default="all",
        dest="methods",
        help=f"Methods: {','.join(BENCHMARK_METHODS.keys())},all",
    )
    parser.add_argument(
        "--dump-json", action="store_true", help="Output results as JSON"
    )
    parser.add_argument(
        "--ci", action="store_true", help="Run in CI mode (run all tests in ci.yaml)"
    )
    parser.add_argument(
        "--test-op", type=str, help="(CI mode only) Only run specific op in ci."
    )
    parser.add_argument(
        "--log-scuba",
        action="store_true",
        help="Log results to Scuba",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON file")
    parser.add_argument("--quiet", action="store_true")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)

    args, extra_args = parser.parse_known_args(args)

    if args.ci:
        run_benchmark_config_ci(
            benchmark_group_name="timing_accuracy",
            benchmark_config_file=os.path.join(CURRENT_DIR, "ci.yaml"),
            transform_func=transform_single_result,
            op=args.test_op,
            ci=args.ci,
            log_scuba=args.log_scuba,
        )
        return

    tb_parser = get_parser()

    tb_args, extra_args = tb_parser.parse_known_args(extra_args)
    _run(args, tb_args, extra_args)



if __name__ == "__main__":
    run()
