import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import torch
import triton


@dataclass
class MethodStats:
    """Statistics for a benchmarking method."""

    method_name: str
    n_tests: int
    reps_per_test: int
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
            "reps_per_test": self.reps_per_test,
            "intra_test": {
                "avg_median_ms": statistics.mean(self.intra_test_medians)
                if self.intra_test_medians
                else 0,
                "avg_std_ms": statistics.mean(self.intra_test_stds)
                if self.intra_test_stds
                else 0,
                "avg_cv": self.avg_intra_test_cv,
                "avg_min_ms": statistics.mean(self.intra_test_mins)
                if self.intra_test_mins
                else 0,
                "avg_max_ms": statistics.mean(self.intra_test_maxs)
                if self.intra_test_maxs
                else 0,
            },
            "inter_test": {
                "median_ms": self.inter_test_median,
                "std_ms": self.inter_test_std,
                "cv": self.inter_test_cv,
            },
            "intra_test_medians": self.intra_test_medians,
            "intra_test_stds": self.intra_test_stds,
            "intra_test_cvs": self.intra_test_cvs,
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
    "standard": ("triton do_bench (standard)", run_do_bench_standard),
    "profiler": ("profiler", run_do_bench_profiler),
    "cudagraph": ("CUDA graph", run_do_bench_cudagraph),
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
    stats = MethodStats(method_name=method_name, n_tests=n_tests, reps_per_test=rep)

    for test_idx in range(n_tests):
        if verbose:
            print(f"  Test {test_idx + 1}/{n_tests}...", end=" ", flush=True)

        if test_idx > 0 and sleep_between_tests > 0:
            time.sleep(sleep_between_tests)

        try:
            samples = method_fn(kernel_fn, warmup=warmup, rep=rep)
            if not samples:
                print("WARNING: No samples returned!")
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
                print(
                    f"median={median:.4f}ms, mean={mean:.4f}ms, std={std:.4f}ms, cv={cv:.4f}"
                )

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()

    return stats


def print_summary_table(results: Dict[str, MethodStats], operation_name: str):
    print("\n" + "=" * 120)
    print(f"SUMMARY: Latency Noise Comparison for '{operation_name}'")
    print("=" * 120)

    header = f"{'Method':<25} | {'Min (ms)':<10} | {'Median (ms)':<12} | {'Intra-Std (ms)':<14} | {'Intra-CV':<10} | {'Inter-CV':<10} | {'Inter-Std (ms)':<14}"
    print(header)
    print("-" * 120)

    for method_name, stats in sorted(results.items(), key=lambda x: x[1].inter_test_cv):
        print(
            f"{method_name:<25} | "
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


def main():
    parser = argparse.ArgumentParser(
        description="Compare latency noise across benchmarking methods",
        allow_abbrev=False,
    )

    parser.add_argument("--op", type=str, required=True, help="TritonBench operator")
    parser.add_argument(
        "--only", type=str, default=None, help="Kernel implementation(s)"
    )
    parser.add_argument("--input-id", type=str, default="0", help="Input config ID")
    parser.add_argument(
        "--mode", choices=["fwd", "bwd", "fwd_bwd", "fwd_no_grad"], default="fwd"
    )
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument(
        "--n-tests", type=int, default=10, help="Benchmark runs per method"
    )
    parser.add_argument("--reps-per-test", type=int, default=100, help="Reps per run")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup (ms)")
    parser.add_argument("--sleep-between-tests", type=float, default=0.5)
    parser.add_argument(
        "--bench-methods",
        type=str,
        default="all",
        dest="methods",
        help=f"Methods: {','.join(BENCHMARK_METHODS.keys())},all",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--quiet", action="store_true")

    args, extra_args = parser.parse_known_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)

    device_name = torch.cuda.get_device_name()
    print(f"\nLoading operator: {args.op}")

    # Use existing tritonbench infrastructure to load operator
    from tritonbench.utils.run_utils import load_operator_by_args

    tb_arg_list = [
        "--op",
        args.op,
        "--mode",
        args.mode,
        "--precision",
        args.precision,
        "--device",
        "cuda",
        "--input-id",
        args.input_id,
        "--num-inputs",
        "1",
        "--test-only",
    ]
    if args.only:
        tb_arg_list.extend(["--only", args.only])
    tb_arg_list.extend(extra_args)

    opbench = load_operator_by_args(tb_arg_list)
    opbench.example_inputs = opbench.get_example_inputs()

    if opbench.example_inputs is None:
        print(f"ERROR: No example inputs for operator '{args.op}'")
        sys.exit(1)

    # Get the benchmark function
    if args.only:
        backend_name = args.only.split(",")[0]
        bench_fn_factory = getattr(opbench, backend_name, None)
        if bench_fn_factory is None:
            print(f"ERROR: Backend '{backend_name}' not found")
            sys.exit(1)
    else:
        from tritonbench.utils.triton_op import REGISTERED_BENCHMARKS

        registered = REGISTERED_BENCHMARKS.get(opbench.name, {})
        if not registered:
            print(f"ERROR: No benchmarks registered for '{args.op}'")
            sys.exit(1)
        backend_name = list(registered.keys())[0]
        bench_fn_factory = getattr(opbench, backend_name)

    example_inputs = opbench.example_inputs
    if isinstance(example_inputs, dict):
        kernel_fn = bench_fn_factory(**example_inputs)
    else:
        kernel_fn = bench_fn_factory(*example_inputs)

    operation_name = f"{args.op}:{backend_name} (input_id={args.input_id})"
    print(
        f"Device: {device_name}, Backend: {backend_name}, Tests: {args.n_tests}, Reps: {args.reps_per_test}\n"
    )

    # Determine methods to run
    if args.methods == "all":
        methods_to_run = list(BENCHMARK_METHODS.keys())
    else:
        methods_to_run = [m.strip() for m in args.methods.split(",")]
        for m in methods_to_run:
            if m not in BENCHMARK_METHODS:
                print(
                    f"ERROR: Unknown method '{m}'. Available: {', '.join(BENCHMARK_METHODS.keys())}"
                )
                sys.exit(1)

    # Warmup
    print("GPU warmup...")
    for _ in range(10):
        kernel_fn()
    torch.cuda.synchronize()

    # Run benchmarks
    results: Dict[str, MethodStats] = {}
    for method_key in methods_to_run:
        method_display_name, method_fn = BENCHMARK_METHODS[method_key]
        print(f"\n{'=' * 60}\nBenchmarking: {method_display_name}\n{'=' * 60}")

        stats = benchmark_method(
            method_name=method_display_name,
            method_fn=method_fn,
            kernel_fn=kernel_fn,
            n_tests=args.n_tests,
            warmup=args.warmup,
            rep=args.reps_per_test,
            sleep_between_tests=args.sleep_between_tests,
            verbose=not args.quiet,
        )
        results[method_display_name] = stats

    print_summary_table(results, operation_name)

    if args.output:
        output_data = {
            "config": {
                "device": device_name,
                "operator": args.op,
                "backend": backend_name,
                "input_id": args.input_id,
                "mode": args.mode,
                "precision": args.precision,
                "n_tests": args.n_tests,
                "reps_per_test": args.reps_per_test,
                "warmup": args.warmup,
            },
            "results": {name: stats.to_dict() for name, stats in results.items()},
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
