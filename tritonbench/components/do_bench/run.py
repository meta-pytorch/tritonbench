import statistics
import time
from functools import partial
from typing import List, Optional

import torch
import triton
from torch._inductor.runtime.benchmarking import benchmarker

NS_TO_MS = 1e-6

# Kernel name for L2 cache clearing - we want to exclude this from latency measurements
CACHE_CLEAR_KERNEL = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)"


class Latency:
    times: List[float]

    def __init__(self, times):
        self.times = self._remove_outliers_iqr(times)

    def __str__(self):
        """By default, use p50"""
        return self.to_str()

    def _remove_outliers_iqr(self, data):
        """
        Removes outliers from a list of floats using the IQR method.

        Args:
            data: A list of floats.

        Returns:
            A new list with outliers removed.
        """
        starting_length = len(data)
        if starting_length <= 3:
            return data
        if not data:
            return []

        data.sort()
        quantiles = statistics.quantiles(data, n=100)
        q1 = quantiles[25]
        q3 = quantiles[75]
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        filtered_data = [x for x in data if lower_bound <= x and x <= upper_bound]
        end_len = len(filtered_data)
        if end_len != starting_length:
            print(
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


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(torch, return_mode)(times).item()


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
    return _summarize_statistics(times, quantiles=None, return_mode=return_mode)


def _do_bench_profiler(
    fn, warmup, rep, return_mode="all", grad_to_none=None, use_cudagraph=False
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
    # we don't want any outside errors propagating into benchmarking
    torch.cuda.synchronize()

    # warmup `fn` (and catches any failures in the process)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    # Get cache for L2 cache clearing
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    # First, estimate the runtime to calculate iterations
    estimate_ms = benchmarker.benchmark_gpu(fn, estimation_iters=5, benchmark_iters=10)

    # Calculate number of iterations based on target rep time
    if estimate_ms == 0:
        n_repeat = 100  # Default if function is very fast
    else:
        n_repeat = max(1, int(rep / estimate_ms))

    # Helper function to execute one iteration
    def run_iteration():
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        cache.zero_()
        fn()

    if use_cudagraph:
        # Create CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                run_iteration()
        torch.cuda.synchronize()

    n_profiler_runs = 5

    # Benchmark phase - collect kernel times for each iteration
    all_kernel_times = []
    profiler_config = {
        "activities": [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
    }

    for _ in range(n_profiler_runs):
        # Profile execution
        with torch.profiler.profile(**profiler_config) as prof:
            if use_cudagraph:
                g.replay()
            else:
                # Execute multiple iterations for regular mode
                for _ in range(n_repeat):
                    run_iteration()
            torch.cuda.synchronize()

        # Collect all kernel execution intervals
        kernel_intervals = []

        # check the number of cache clear kernels.
        # we rely on hard-coded aten op name for excluding cache clear kernels.
        # this check ensures that pytorch does not dispatch to another kernel.
        num_cache_clear_kernels = len(
            [
                evt
                for evt in prof.events()
                if evt.device_type == torch.autograd.DeviceType.CUDA
                and evt.name == CACHE_CLEAR_KERNEL
            ]
        )
        assert (
            num_cache_clear_kernels == n_repeat
        ), f"Expected {n_repeat} cache clear kernels but found {num_cache_clear_kernels}"

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
        total_kernel_time_ms = (total_kernel_time_us / 1000.0) / n_repeat
        all_kernel_times.append(total_kernel_time_ms)

    times = torch.tensor(all_kernel_times, dtype=torch.float)
    return _summarize_statistics(times, quantiles=None, return_mode=return_mode)


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
    return _summarize_statistics(times, quantiles, return_mode)


def do_bench_wrapper(
    fn,
    warmup,
    rep,
    grad_to_none,
    device: str = "cuda",
    use_cuda_graphs: bool = False,
    bypass_fail: bool = False,
    latency_measure_mode: str = "triton_do_bench",
) -> Optional[Latency]:
    """Wrapper to triton's do_bench to gain latency.

    Args:
        latency_measure_mode: Either "triton_do_bench" (default) or "inductor_benchmarker" or "profiler"
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
        elif use_cuda_graphs:
            with torch.cuda.stream(torch.cuda.Stream()):
                if latency_measure_mode == "profiler":
                    bench_fn = partial(_do_bench_profiler, warmup=1, use_cudagraph=True)
                else:
                    bench_fn = triton.testing.do_bench_cudagraph

                return Latency(
                    times=bench_fn(
                        fn,
                        rep=rep,
                        return_mode="all",
                        grad_to_none=grad_to_none,
                    )
                )
        else:
            bench_fn = (
                _do_bench_profiler
                if latency_measure_mode == "profiler"
                else _do_bench_inductor
                if latency_measure_mode == "inductor_benchmarker"
                else triton.testing.do_bench
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
