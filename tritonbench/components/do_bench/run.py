import statistics
import time

from typing import List, Optional

import torch
import triton
from torch._inductor.runtime.benchmarking import benchmarker

NS_TO_MS = 1e-6


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


def _do_bench_inductor(fn, warmup, rep, grad_to_none=None):
    """Measure latency using inductor benchmarker.

    Args:
        warmup: Target warmup time in milliseconds (matches triton.testing.do_bench)
        rep: Target total measurement time in milliseconds (matches triton.testing.do_bench)
        grad_to_none: Tensors whose gradients should be cleared before each measurement

    Returns:
        List of measured times in milliseconds.
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

    return times_ms


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
        latency_measure_mode: Either "triton_do_bench" (default) or "inductor_benchmarker"
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
                return Latency(
                    times=triton.testing.do_bench_cudagraph(
                        fn,
                        rep=rep,
                        return_mode="all",
                        grad_to_none=grad_to_none,
                    )
                )
        else:
            if latency_measure_mode == "inductor_benchmarker":
                return Latency(
                    times=_do_bench_inductor(
                        fn,
                        warmup=warmup,
                        rep=rep,
                        grad_to_none=grad_to_none,
                    )
                )
            else:  # default to triton do_bench
                return Latency(
                    times=triton.testing.do_bench(
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
