from typing import List

import triton
from tritonbench.utils.env_utils import get_device_module, get_graph_cls


def do_bench_power(
    fn: callable,
    repcnt: int,
    grad_to_none,
    skip_cache_clearing: bool = False,
    use_cuda_graphs: bool = False,
) -> List[float]:
    di = triton.runtime.driver.active.get_device_interface()

    n_repeat = int(repcnt)

    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]

    if skip_cache_clearing:
        # skip both cache clearing and gradient clearing
        cache = None
        cache_clear = lambda *args, **kwargs: None
        grad_to_none = None
    else:
        cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
        cache_clear = triton.runtime.driver.active.clear_cache

    if use_cuda_graphs:
        # Create device graph
        device_module = get_device_module()
        with device_module.stream(device_module.Stream()):
            g = get_graph_cls()()
            with device_module.graph(g):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                cache_clear(cache)
                fn()
            device_module.synchronize()
            # record cache clear graph
            if not skip_cache_clearing:
                cache_start_event = [
                    di.Event(enable_timing=True) for i in range(n_repeat)
                ]
                cache_end_event = [
                    di.Event(enable_timing=True) for i in range(n_repeat)
                ]
                cache_clear_graph = get_graph_cls()()
                with device_module.graph(cache_clear_graph):
                    cache_clear(cache)
                device_module.synchronize()
            for i in range(n_repeat):
                if not skip_cache_clearing:
                    cache_start_event[i].record()
                    cache_clear_graph.replay()
                    cache_end_event[i].record()
                start_event[i].record()
                g.replay()
                end_event[i].record()
            device_module.synchronize()
        times = []
        for i in range(n_repeat):
            cache_clear_time = (
                0
                if skip_cache_clearing
                else cache_end_event[i].elapsed_time(cache_start_event[i])
            )
            graph_time = end_event[i].elapsed_time(start_event[i])
            times.append(graph_time - cache_clear_time)
        return times

    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache_clear(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks

    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return times
