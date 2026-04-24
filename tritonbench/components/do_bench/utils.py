from typing import Callable, Iterable, Optional, Tuple

import torch
from tritonbench.utils.constants import DEFAULT_WARMUP_REP_BY_ESTIMATED_KERNEL_MS


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


def estimate_cuda_runtime_ms(
    fn: Callable,
    grad_to_none: Optional[Iterable[torch.Tensor]] = None,
    clear_cache_fn: Optional[Callable[[], None]] = None,
    iters: int = 5,
    prime: bool = True,
) -> float:
    clear_cache_fn = clear_cache_fn or (lambda: None)

    def run_once() -> None:
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        clear_cache_fn()
        fn()

    if prime:
        run_once()
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        run_once()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters
