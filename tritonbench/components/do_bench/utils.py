import logging
from typing import Callable, Iterable, Optional, Tuple

import torch
from tritonbench.utils.constants import DEFAULT_WARMUP_REP_BY_ESTIMATED_KERNEL_MS
from tritonbench.utils.env_utils import get_device_module

logger = logging.getLogger(__name__)

_cache_clear_buffer_primed = False


def prime_cache_clear_buffer() -> None:
    """Allocate and first-touch the L2 cache-clear buffer, once per process.

    Benchmarkers flush the L2 by zeroing the ~256MB scratch buffer returned by
    ``get_empty_cache_for_benchmark()``. Zeroing a freshly allocated buffer
    pays a one-off page-mapping cost (measured ~6ms, versus ~0.04ms once the
    pages are resident). Triton's ``do_bench`` runs its 5-iteration runtime
    estimate with no prior flush, so whichever backend happens to be measured
    first in a process folds that one-off into its ``estimate_ms`` -- inflating
    it by two orders of magnitude and shrinking the sample count it derives,
    ``n_repeat = rep / estimate_ms``, by the same factor.

    One throwaway flush here keeps that cost out of every estimate. The buffer
    is not retained: it goes back to the caching allocator's pool, so later
    ``get_empty_cache_for_benchmark()`` calls reuse the same resident pages.

    Idempotent, and a no-op on backends with no benchmark cache buffer.
    """
    global _cache_clear_buffer_primed
    if _cache_clear_buffer_primed:
        return
    # Mark upfront so an unsupported backend is not retried on every call.
    _cache_clear_buffer_primed = True
    try:
        import triton

        driver = triton.runtime.driver.active
        driver.clear_cache(driver.get_empty_cache_for_benchmark())
        get_device_module().synchronize()
    except Exception as e:
        logger.warning("Could not prime the cache-clear buffer: %s", e)


def resolve_warmup_and_rep(
    warmup: Optional[int], rep: Optional[int], estimate_ms: float
) -> Tuple[int, int]:
    if estimate_ms <= 1:
        default_warmup, default_rep = DEFAULT_WARMUP_REP_BY_ESTIMATED_KERNEL_MS["1"]
    elif estimate_ms <= 10:
        default_warmup, default_rep = DEFAULT_WARMUP_REP_BY_ESTIMATED_KERNEL_MS["10"]
    else:
        default_warmup, default_rep = DEFAULT_WARMUP_REP_BY_ESTIMATED_KERNEL_MS["100"]
    return (
        default_warmup if warmup is None else warmup,
        default_rep if rep is None else rep,
    )


def estimate_gpu_runtime_ms(
    fn: Callable,
    grad_to_none: Optional[Iterable[torch.Tensor]] = None,
    clear_cache_fn: Optional[Callable[[], None]] = None,
    iters: int = 5,
    prime: bool = True,
) -> float:
    """Estimate the per-iteration runtime of ``fn`` using GPU events."""
    clear_cache_fn = clear_cache_fn or (lambda: None)
    device_module = get_device_module()

    def run_once() -> None:
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        clear_cache_fn()
        fn()

    if prime:
        run_once()
        device_module.synchronize()

    start_event = device_module.Event(enable_timing=True)
    end_event = device_module.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        run_once()
    end_event.record()
    device_module.synchronize()
    return start_event.elapsed_time(end_event) / iters


# Deprecated alias: this helper is no longer CUDA-specific. Kept so out-of-tree
# callers keep working; prefer `estimate_gpu_runtime_ms`.
estimate_cuda_runtime_ms = estimate_gpu_runtime_ms
