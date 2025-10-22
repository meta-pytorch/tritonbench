from typing import List

import triton


def do_bench_power(fn: callable, repcnt: int, grad_to_none) -> List[float]:
    di = triton.runtime.driver.active.get_device_interface()

    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    n_repeat = int(repcnt)

    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]

    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        triton.runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks

    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return times
