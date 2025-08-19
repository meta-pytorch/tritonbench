from typing import Any

import torch

import triton
from torch._C._autograd import DeviceType
from torch.profiler import profile, ProfilerActivity

from .run import Latency


CACHE_CLEAR_KERNEL = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)"


def _kineto_events_to_latency(prof, n_repeat):
    prof_averages = prof.key_averages(group_by_input_shape=False)
    cuda_event_names = [
        event.key
        for event in prof_averages
        if event.device_time > 0 and event.key != CACHE_CLEAR_KERNEL
    ]
    events = prof.profiler.kineto_results.events()
    # remove all zero cuda time events and cache clear kernel
    kineto_events = [
        event
        for event in events
        if event.name() in cuda_event_names and event.device_type() == DeviceType.CUDA
    ]

    kernel_duration_name_map = {}
    for event in kineto_events:
        if event.name() not in kernel_duration_name_map:
            kernel_duration_name_map[event.name()] = []
        kernel_duration_name_map[event.name()].append(event.duration_ns() / 1e6)

    op_time = 0.0
    for name in kernel_duration_name_map:
        op_time += sum(kernel_duration_name_map[name])

    op_time = op_time / n_repeat

    print(
        prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total")
    )
    return op_time


def _do_bench_cuda_time_cudagraph(
    synchronize_with_timing: Any,
    cache: Any,
    fn: Any,
    n_warmup: int,
    n_repeat: int,
    grad_to_none: bool,
    bypass_fail: bool = False,
) -> float:
    with torch.cuda.stream(torch.cuda.Stream()):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                # we clear the L2 cache before each run
                cache.zero_()
                fn()
        synchronize_with_timing()

        # measure time and return
        with profile(
            schedule=torch.profiler.schedule(
                wait=0, warmup=n_warmup, active=n_repeat, repeat=1
            ),
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            activities=[ProfilerActivity.CUDA],
            with_stack=False,
            record_shapes=False,
        ) as prof:
            for _iters in range(n_warmup + n_repeat):
                g.replay()
                prof.step()
            synchronize_with_timing()

    return _kineto_events_to_latency(prof, n_repeat)


def do_bench_cuda_time(
    fn: Any,
    warmup: int,
    rep: int,
    grad_to_none: bool,
    use_cuda_graphs: bool = False,
    bypass_fail: bool = False,
) -> float:
    """
    Return the aggregated CUDA time of a benchmarked operator backend.
    """

    di = triton.runtime.driver.active.get_device_interface()

    def synchronize_with_timing():
        di.synchronize()

    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    synchronize_with_timing()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    if use_cuda_graphs:
        return _do_bench_cuda_time_cudagraph(
            synchronize_with_timing,
            cache,
            fn,
            n_warmup,
            n_repeat,
            grad_to_none,
            bypass_fail,
        )

    with profile(
        schedule=torch.profiler.schedule(
            wait=0, warmup=n_warmup, active=n_repeat, repeat=1
        ),
        # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
        activities=[ProfilerActivity.CUDA],
        with_stack=False,
        record_shapes=False,
    ) as prof:
        for _iters in range(n_warmup + n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            cache.zero_()
            fn()
            prof.step()
        synchronize_with_timing()

    return _kineto_events_to_latency(prof, n_repeat)
