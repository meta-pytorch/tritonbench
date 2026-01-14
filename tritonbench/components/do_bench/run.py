import logging
import statistics
import time
from functools import partial
from typing import List, Optional

import torch
import triton
from torch._inductor.runtime.benchmarking import benchmarker
from tritonbench.components.do_bench.entropy.entropy_criterion import EntropyCriterion
from tritonbench.utils.constants import DEFAULT_N_REP, DEFAULT_N_WARMUP

from .common import summarize_statistics
from .gpu_events import do_bench_events
from .power import do_bench_power

NS_TO_MS = 1e-6
logger = logging.getLogger(__name__)
# Kernel name for L2 cache clearing - we want to exclude this from latency measurements
CACHE_CLEAR_KERNEL = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)"


class Latency:
    times: List[float]

    def __init__(self, times, remove_outliers=True):
        self.times = self._remove_outliers_iqr(times) if remove_outliers else times

    def __str__(self):
        """By default, use p50"""
        return self.to_str()

    def _remove_outliers_iqr(self, data):
        """
        Removes outliers from a list of floats using the IQR method.

        Args:
            data: A list of floats.

        Returns:
            A new list with outliers removed, preserving original order.
        """
        starting_length = len(data)
        if starting_length <= 3:
            return data
        if not data:
            return []

        # create a copy to calculate quantiles, preserving original order
        sorted_data = sorted(data)
        quantiles = statistics.quantiles(sorted_data, n=100)
        q1 = quantiles[25]
        q3 = quantiles[75]
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        # Filter while preserving original temporal order
        filtered_data = [x for x in data if lower_bound <= x and x <= upper_bound]
        end_len = len(filtered_data)
        if end_len != starting_length:
            logger.debug(
                f"Removed {starting_length - end_len} outliers from {starting_length} samples"
            )
        return filtered_data

    @property
    def p50(self):
        return statistics.median_low(self.times)

    @property
    def min(self):
        return min(self.times)

    @property
    def max(self):
        return max(self.times)

    def __add__(self, other):
        return self.p50 + other.p50 if isinstance(other, Latency) else self.p50 + other

    def __radd__(self, other):
        return other.p50 + self.p50 if isinstance(other, Latency) else other + self.p50

    def __sub__(self, other):
        return self.p50 - other.p50 if isinstance(other, Latency) else self.p50 - other

    def __rsub__(self, other):
        return other.p50 - self.p50 if isinstance(other, Latency) else other - self.p50

    def __mul__(self, other):
        return self.p50 * other.p50 if isinstance(other, Latency) else self.p50 * other

    def __rmul__(self, other):
        return other.p50 * self.p50 if isinstance(other, Latency) else other * self.p50

    def __truediv__(self, other):
        return self.p50 / other.p50 if isinstance(other, Latency) else self.p50 / other

    def __rtruediv__(self, other):
        return other.p50 / self.p50 if isinstance(other, Latency) else other / self.p50

    def __floordiv__(self, other):
        return (
            self.p50 // other.p50 if isinstance(other, Latency) else self.p50 // other
        )

    def __rfloordiv__(self, other):
        return (
            other.p50 // self.p50 if isinstance(other, Latency) else other // self.p50
        )

    def to_str(self, mode="p50") -> str:
        if mode == "p50":
            return str(self.p50)
        elif mode == "with_variance":
            max_variance = max((self.max - self.p50), (self.p50 - self.min)) / self.p50
            return f"{self.p50:6f} (±{max_variance * 100:.2f}%)"
        elif mode == "max":
            return str(self.max)
        elif mode == "min":
            return str(self.max)
        elif mode == "mean":
            return str(statistics.mean(self.times))
        else:
            raise ValueError(f"Unsupported latency output mode: {mode}")

    def to_float(self) -> float:
        return float(self.to_str())

    def scale(self, scale):
        for i in range(len(self.times)):
            self.times[i] /= scale


def _do_bench_inductor(fn, warmup, rep, return_mode="all", grad_to_none=None):
    """Measure latency using inductor benchmarker.

    Args:
        warmup: Target warmup time in milliseconds (matches triton.testing.do_bench)
        rep: Target total measurement time in milliseconds (matches triton.testing.do_bench)
        return_mode: "all" for list of measurements, other modes for single values
        grad_to_none: Tensors whose gradients should be cleared before each measurement

    Returns:
        List of measured times in milliseconds (if return_mode="all") or single value.
    """
    # First, estimate the runtime with a single measurement
    estimate_ms = benchmarker.benchmark_gpu(fn, estimation_iters=5, benchmark_iters=10)

    # Calculate number of iterations based on target rep time
    # Similar to how triton.testing.do_bench calculates iterations
    if estimate_ms == 0:
        n_repeat = 1000  # Default if function is very fast
    else:
        n_repeat = max(1, int(rep / estimate_ms))

    # Collect multiple measurements like triton.testing.do_bench with return_mode='all'
    times_ms = []
    for _ in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None

        # Measure only the function execution time
        ms_time = benchmarker.benchmark_gpu(fn)
        times_ms.append(ms_time)

    times = torch.tensor(times_ms, dtype=torch.float)
    return summarize_statistics(times, quantiles=None, return_mode=return_mode)


def _do_bench_cudagraph_with_cache_clear(
    fn,
    rep=20,
    grad_to_none=None,
    quantiles=None,
    return_mode="mean",
    skip_cache_clearing=False,
):
    """Clone of triton.testing.do_bench_cudagraph with explicit L2 cache clearing."""
    assert return_mode in ["min", "max", "mean", "median", "all"]

    cache = (
        triton.runtime.driver.active.get_empty_cache_for_benchmark()
        if not skip_cache_clearing
        else None
    )
    clear_cache_fn = cache.zero_ if not skip_cache_clearing else lambda *args: None

    with torch.cuda.stream(torch.cuda.Stream()):
        clear_cache_fn()
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            clear_cache_fn()
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        n_repeat = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                clear_cache_fn()
                fn()
        torch.cuda.synchronize()

        cache_clear_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cache_clear_graph):
            for _ in range(n_repeat):
                clear_cache_fn()
        torch.cuda.synchronize()

        n_retries = 10
        cache_clear_times = []
        total_times = []
        for _ in range(n_retries):
            cache_clear_start_event = torch.cuda.Event(enable_timing=True)
            cache_clear_end_event = torch.cuda.Event(enable_timing=True)
            cache_clear_start_event.record()
            cache_clear_graph.replay()
            cache_clear_end_event.record()
            torch.cuda.synchronize()
            cache_clear_times.append(
                cache_clear_start_event.elapsed_time(cache_clear_end_event) / n_repeat
            )

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            total_times.append(start_event.elapsed_time(end_event) / n_repeat)

    all_kernel_times = []
    for total_time, cache_clear_time in zip(total_times, cache_clear_times):
        kernel_time = total_time - cache_clear_time
        all_kernel_times.append(kernel_time)

    times = torch.tensor(all_kernel_times, dtype=torch.float)
    return summarize_statistics(times, quantiles, return_mode)


def _do_bench_profiler(
    fn,
    warmup,
    rep,
    return_mode="all",
    grad_to_none=None,
    use_cudagraph=False,
    skip_cache_clearing=False,
):
    """Measure GPU kernel execution time using PyTorch profiler.

    This method profiles the function and extracts the actual GPU kernel execution
    time by summing up all CUDA kernel durations (excluding overlaps) from the profiler trace.

    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds (matches triton.testing.do_bench)
        rep: Target total measurement time in milliseconds (matches triton.testing.do_bench)
        return_mode: "all" for list of measurements, other modes for single values
        grad_to_none: Tensors whose gradients should be cleared before each measurement
        use_cudagraph: Whether to use CUDA graphs for benchmarking

    Returns:
        List of measured kernel times in milliseconds (if return_mode="all") or single value.
    """
    # Get cache for L2 cache clearing
    cache = (
        triton.runtime.driver.active.get_empty_cache_for_benchmark()
        if not skip_cache_clearing
        else None
    )

    # First, estimate the runtime to calculate iterations
    estimate_ms = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        grad_to_none=grad_to_none,
        return_mode="mean",
    )

    # Calculate number of iterations based on target rep time
    if estimate_ms == 0:
        n_repeat = DEFAULT_N_REP  # Default if function is very fast
    else:
        n_repeat = max(1, int(rep / estimate_ms))

    clear_cache_fn = cache.zero_ if not skip_cache_clearing else lambda *args: None

    # Helper function to execute one iteration
    def run_iteration():
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        clear_cache_fn()
        fn()

    if use_cudagraph:
        # Create CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                run_iteration()
        torch.cuda.synchronize()
    else:
        # Regular mode warmup
        n_warmup = (
            max(1, int(warmup / estimate_ms)) if estimate_ms > 0 else DEFAULT_N_WARMUP
        )

        torch.cuda.synchronize()
        for _ in range(n_warmup):
            run_iteration()
        torch.cuda.synchronize()

    iterations_per_profiler_run = n_repeat

    # Benchmark phase - collect kernel times for each iteration
    all_kernel_times = []
    profiler_config = {
        "activities": [
            torch.profiler.ProfilerActivity.CUDA,
        ],
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
    }

    def _trace_handler(prof: torch.profiler.profile) -> None:
        # Collect all kernel execution intervals
        kernel_intervals = []

        # Get raw function events and collect time intervals
        for evt in prof.events():
            # Check for CUDA kernel events, excluding cache clear kernel
            if (
                evt.device_type == torch.autograd.DeviceType.CUDA
                and hasattr(evt, "time_range")
                and evt.name != CACHE_CLEAR_KERNEL
            ):
                # time_range has start and end attributes in microseconds
                start_us = evt.time_range.start
                end_us = evt.time_range.end
                if start_us < end_us:  # Valid interval
                    kernel_intervals.append((start_us, end_us))

        # Merge overlapping intervals to get actual GPU busy time
        # This algorithm handles concurrent kernels across multiple streams by:
        # 1. Sorting all kernel intervals by start time
        # 2. Merging overlapping intervals to avoid double-counting concurrent execution
        # 3. Summing only the time when at least one kernel is running
        # This gives us the true GPU wall-clock time, excluding idle gaps between kernels
        if kernel_intervals:
            # Sort intervals by start time
            kernel_intervals.sort(key=lambda x: x[0])

            # Merge overlapping intervals using a sweep-line algorithm
            # Example: [(0,5), (3,8), (10,15)] -> [(0,8), (10,15)]
            merged_intervals = [kernel_intervals[0]]
            for start, end in kernel_intervals[1:]:
                last_start, last_end = merged_intervals[-1]

                if start <= last_end:
                    # Overlapping or adjacent intervals, merge them
                    # Take the max of end times to handle nested intervals
                    merged_intervals[-1] = (last_start, max(last_end, end))
                else:
                    # Non-overlapping interval, add as new
                    merged_intervals.append((start, end))

            # Calculate total GPU busy time by summing merged intervals
            total_kernel_time_us = sum(end - start for start, end in merged_intervals)
        else:
            # No kernel events found - this likely indicates an issue
            raise RuntimeError(
                "No CUDA kernel events found in profiler trace. "
                "This may indicate the function is not executing any GPU kernels, "
                "or there's an issue with profiler event collection."
            )

        # Convert to milliseconds and normalize by iterations
        kernel_time_per_iteration_ms = (
            total_kernel_time_us / 1000.0
        ) / iterations_per_profiler_run
        all_kernel_times.append(kernel_time_per_iteration_ms)

    # Profile execution
    with torch.profiler.profile(
        **profiler_config, on_trace_ready=_trace_handler
    ) as prof:
        if use_cudagraph:
            g.replay()
        else:
            # Execute multiple iterations for regular mode
            for _ in range(iterations_per_profiler_run):
                run_iteration()
        torch.cuda.synchronize()

    times = torch.tensor(all_kernel_times, dtype=torch.float)
    return summarize_statistics(times, quantiles=None, return_mode=return_mode)


def _do_bench_cpu(
    fn, warmup, rep=20, grad_to_none=None, quantiles=None, return_mode="mean"
):
    """Measure latency of a function on CPU."""
    assert return_mode in ["min", "max", "mean", "median", "all"]
    fn()
    # Estimate the runtime of the function
    t0 = time.time_ns()
    for _ in range(5):
        fn()
    t1 = time.time_ns()
    estimate_ms = (t1 - t0) * NS_TO_MS / 5

    # compute number of warmup and repeat
    if estimate_ms == 0:
        n_repeat = 1000
        n_warmup = 1000
    else:
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
    # Warm-up
    for _ in range(n_warmup):
        fn()
    times_ms = []
    # Benchmark
    for _i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # record time of `fn`
        t0 = time.time_ns()
        fn()
        t1 = time.time_ns()
        times_ms.append((t1 - t0) * NS_TO_MS)
    times = torch.tensor(times_ms, dtype=torch.float)
    return summarize_statistics(times, quantiles, return_mode)


def _do_bench_entropy(
    fn,
    warmup=25,
    rep=100,
    grad_to_none=None,
    quantiles=None,
    return_mode="mean",
    max_angle=0.048,
    min_r2=0.36,
    window_size=299,
    max_samples=10000,
    min_warmup_samples=20,
    repcnt=None,
):
    """
    Benchmark function using entropy-based adaptive warmup followed by fixed measurement window.

    This function uses entropy criterion for adaptive GPU warmup, then runs a traditional
    fixed-time measurement window for final benchmark results.

    Args:
        fn: Function to benchmark
        warmup: Unused (for API compatibility)
        rep: Target measurement time in ms for final benchmark (default: 100ms)
        grad_to_none: Gradients to reset between measurements
        quantiles: Quantiles to compute (e.g., [0.5, 0.95])
        return_mode: "min", "max", "mean", "median", or "all"
        max_angle: Maximum entropy slope angle for convergence (degrees)
        min_r2: Minimum R² for convergence
        window_size: Size of rolling window for entropy tracking
        max_samples: Maximum samples before stopping warmup (safety limit)
        min_warmup_samples: Minimum samples before checking convergence (ensures GPU warmup)
        repcnt: If provided, use this many iterations (skips calibration)

    Returns:
        Measurement statistic based on return_mode
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    # ENTROPY-BASED WARMUP
    entropy_criterion = EntropyCriterion(
        max_angle=max_angle,
        min_r2=min_r2,
        window_size=window_size,
        min_warmup_samples=min_warmup_samples,
    )
    entropy_criterion.reset()

    rounding_factor = 3
    BATCH_SIZE = 50
    last_batch = [-1.00] * BATCH_SIZE
    counter = 0
    converged = False
    precision_increase = False

    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    # Adaptive warmup loop with batched synchronization
    while True:
        remaining = max_samples - counter
        batch_size = min(BATCH_SIZE, remaining) if remaining > 0 else BATCH_SIZE

        batch_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(batch_size)
        ]
        batch_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(batch_size)
        ]

        for i in range(batch_size):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            triton.runtime.driver.active.clear_cache(cache)
            batch_start_events[i].record()
            fn()
            batch_end_events[i].record()

        for i in range(batch_size):
            batch_end_events[i].synchronize()
            v = round(
                batch_start_events[i].elapsed_time(batch_end_events[i]), rounding_factor
            )
            last_batch[i] = v

            entropy_criterion.add_measurement(v)

            if entropy_criterion.is_finished():
                converged = True
                break

        counter += batch_size

        if converged:
            break

        if counter >= max_samples:
            break

        if counter >= 200 and not precision_increase:
            stats = entropy_criterion.get_stats()
            unique_count = stats.get("unique_measurements", 0)

            # If we have < 20 unique values, this indicates quantization, increase rounding precision
            if unique_count < 20:
                rounding_factor = 4
                entropy_criterion.reset()
                entropy_criterion.entropy_window_size = 1000

                logger.info(
                    f"Quantization detected: only {unique_count} unique measurements. "
                )
                precision_increase = True

    # Log if warmup didn't converge
    if not converged:
        logger.warning(
            f"Warmup did not converge after {counter} samples "
            f"(max_samples={max_samples})"
        )

    # DETERMINE ITERATION COUNT
    if repcnt is not None:
        # Use fixed iteration count (skip calibration)
        n_iterations = int(repcnt)
    else:
        # CALIBRATION: Reuse mean of last 5 warmup samples
        CALIBRATION_SAMPLES = 5

        if counter >= CALIBRATION_SAMPLES:
            avg_kernel_time_ms = statistics.mean(last_batch)
        else:
            avg_kernel_time_ms = statistics.mean(last_batch) if last_batch else 0

        if avg_kernel_time_ms > 0:
            n_iterations = max(10, int(rep / avg_kernel_time_ms))
        else:
            n_iterations = 100

    # BENCHMARK PHASE
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iterations)]

    for i in range(n_iterations):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        triton.runtime.driver.active.clear_cache(cache)
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()

    benchmark_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    times = torch.tensor(benchmark_times, dtype=torch.float)
    return summarize_statistics(times, quantiles, return_mode)


def do_bench_wrapper(
    fn,
    warmup,
    rep,
    repcnt,
    grad_to_none,
    device: str = "cuda",
    use_cuda_graphs: bool = False,
    bypass_fail: bool = False,
    latency_measure_mode: str = "triton_do_bench",
    skip_cache_clearing: bool = False,
    entropy_criterion: bool = False,
    entropy_max_angle: float = 0.048,
    entropy_min_r2: float = 0.36,
    entropy_window_size: int = 299,
    entropy_max_samples: int = 10000,
    entropy_min_warmup_samples: int = 20,
) -> Optional[Latency]:
    """Wrapper to triton's do_bench to gain latency.

    Args:
        latency_measure_mode: Either "triton_do_bench" (default) or "inductor_benchmarker" or "profiler"
        entropy_criterion: Use entropy-based adaptive warmup + fixed measurement window
        entropy_max_angle: Maximum entropy slope angle for convergence (degrees)
        entropy_min_r2: Minimum R² for linear regression fit
        entropy_window_size: Size of rolling window for entropy tracking
        entropy_max_samples: Maximum samples before stopping warmup (safety limit)
    """
    try:
        if device == "cpu":
            return Latency(
                times=_do_bench_cpu(
                    fn,
                    warmup=warmup,
                    rep=rep,
                    return_mode="all",
                    grad_to_none=grad_to_none,
                )
            )
        elif entropy_criterion and not use_cuda_graphs:
            return Latency(
                times=_do_bench_entropy(
                    fn=fn,
                    warmup=warmup,
                    rep=rep,
                    grad_to_none=grad_to_none,
                    return_mode="all",
                    max_angle=entropy_max_angle,
                    min_r2=entropy_min_r2,
                    window_size=entropy_window_size,
                    max_samples=entropy_max_samples,
                    min_warmup_samples=entropy_min_warmup_samples,
                    repcnt=repcnt,
                )
            )
        elif use_cuda_graphs:
            with torch.cuda.stream(torch.cuda.Stream()):
                if latency_measure_mode == "profiler":
                    bench_fn = partial(_do_bench_profiler, warmup=1, use_cudagraph=True)
                else:
                    bench_fn = _do_bench_cudagraph_with_cache_clear

                return Latency(
                    times=bench_fn(
                        fn,
                        rep=rep,
                        return_mode="all",
                        grad_to_none=grad_to_none,
                        skip_cache_clearing=skip_cache_clearing,
                    )
                )
        elif repcnt:
            # benchmark using repcnt
            return Latency(
                times=do_bench_power(
                    fn,
                    repcnt=repcnt,
                    grad_to_none=grad_to_none,
                    skip_cache_clearing=skip_cache_clearing,
                    use_cuda_graphs=use_cuda_graphs,
                ),
                remove_outliers=False,
            )
        elif latency_measure_mode == "gpu_events":
            return Latency(
                times=do_bench_events(
                    fn,
                    warmup=warmup,
                    rep=rep,
                    return_mode="all",
                    grad_to_none=grad_to_none,
                    skip_cache_clearing=skip_cache_clearing,
                )
            )
        else:
            bench_fn = (
                partial(_do_bench_profiler, skip_cache_clearing=skip_cache_clearing)
                if latency_measure_mode == "profiler"
                else (
                    _do_bench_inductor
                    if latency_measure_mode == "inductor_benchmarker"
                    else triton.runtime.driver.active.get_benchmarker()
                )
            )

            return Latency(
                times=bench_fn(
                    fn,
                    warmup=warmup,
                    rep=rep,
                    return_mode="all",
                    grad_to_none=grad_to_none,
                )
            )
    except Exception as e:
        if not bypass_fail:
            raise e
        return None
