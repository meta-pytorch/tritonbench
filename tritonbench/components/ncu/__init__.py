from typing import Callable

import torch
from tritonbench.components.do_bench.utils import (
    estimate_cuda_runtime_ms,
    resolve_warmup_and_rep,
)


class cuda_profiler_range:
    def __init__(self, use_cuda_profiler_range):
        self.use_cuda_profiler_range = use_cuda_profiler_range

    def __enter__(self):
        if self.use_cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStart()

    def __exit__(self, *exc_info):
        if self.use_cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()


def do_bench_in_task(
    fn: Callable,
    grad_to_none=None,
    range_name: str = "",
    warmup: bool = False,
    test_run: bool = False,
    use_cuda_profiler_range: bool = False,
) -> None:
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    """

    if test_run:
        fn()
    torch.cuda.synchronize()

    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    if warmup == True:
        estimate_ms = estimate_cuda_runtime_ms(fn, clear_cache_fn=cache.zero_)
        warmup, _ = resolve_warmup_and_rep(warmup, None, estimate_ms)

        # compute number of warmup and repeat
        n_warmup = 1 if estimate_ms == 0 else max(1, int(warmup / estimate_ms))
        # Warm-up
        for _ in range(n_warmup):
            fn()

    # we don't want `fn` to accumulate gradient values
    # if it contains a backward pass. So we clear the
    # provided gradients
    if grad_to_none is not None:
        for x in grad_to_none:
            x.grad = None
    # we clear the L2 cache before run
    cache.zero_()
    with cuda_profiler_range(use_cuda_profiler_range):
        nvtx_range_id = torch.cuda.nvtx.range_start(range_name)
        fn()
        torch.cuda.nvtx.range_end(nvtx_range_id)
