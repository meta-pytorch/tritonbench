import random
import string
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.profiler as profiler
from tritonbench.utils.constants import DEFAULT_N_REP, DEFAULT_N_WARMUP
from tritonbench.utils.env_utils import has_manifold

DEFAULT_PROFILE_OPTS = {
    "record_shapes": True,
    "profile_memory": True,
    "with_stack": True,
    "with_flops": True,
    "with_modules": True,
}

if not hasattr(torch.version, "git_version"):
    from .fb.run_utils import trace_handler


def _find_the_latest_file(output_dir, recursive: bool = False, glob_pattern="*"):
    iterator = (
        Path(output_dir).rglob(glob_pattern)
        if recursive
        else Path(output_dir).glob(glob_pattern)
    )

    latest_path: Optional[Path] = None
    latest_ctime: float = float("-inf")

    for p in iterator:
        if p.is_file():
            try:
                ctime = p.stat().st_ctime
            except OSError:
                # Skip unreadable entries
                continue

            if ctime > latest_ctime:
                latest_ctime = ctime
                latest_path = p

    return latest_path.name if latest_path else None


def post_process(output_dir, name) -> str:
    if not hasattr(torch.version, "git_version"):
        return f"https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/tritonbench/{name}.gz&bucket=pyper_traces"
    elif has_manifold():
        lastest_json_file = _find_the_latest_file(output_dir, glob_pattern="*.json.gz")
        assert lastest_json_file is not None, "No trace file found"
        cmd = [
            "manifold",
            "put",
            lastest_json_file,
            f"pyper_traces/tree/traces/tritonbench/{lastest_json_file}",
        ]
        subprocess.check_call(cmd, cwd=output_dir)
        return f"https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/tritonbench/{lastest_json_file}&bucket=pyper_traces"
    else:
        return f"{output_dir}/{name}"


def do_bench_kineto_cudagraph(
    fn,
    clear_cache,
    n_warmup,
    n_repeat,
    grad_to_none,
    profile_opts,
    output_dir,
) -> str:
    activity_groups = [
        profiler.ProfilerActivity.CUDA,
        profiler.ProfilerActivity.CPU,
    ]
    with torch.cuda.stream(torch.cuda.Stream()):
        # step 1 - construct a cuda graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            clear_cache()
            fn()
        torch.cuda.synchronize()
        prefix = f"tritonbench_cudagraph_{fn._name}"
        name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{''.join(random.choices(string.digits, k=10))}.json"
        # step 2 - profile cuda graph launch with kineto
        with profiler.profile(
            schedule=profiler.schedule(
                wait=0, warmup=n_warmup + n_repeat - 1, active=1, repeat=1
            ),
            activities=activity_groups,
            record_shapes=profile_opts["record_shapes"],
            profile_memory=profile_opts["profile_memory"],
            with_stack=profile_opts["with_stack"],
            with_flops=profile_opts["with_flops"],
            with_modules=profile_opts["with_modules"],
            on_trace_ready=(
                partial(trace_handler, name)
                if not hasattr(torch.version, "git_version")
                else profiler.tensorboard_trace_handler(output_dir, use_gzip=True)
            ),
        ) as prof:
            for _i in range(n_warmup + n_repeat):
                # we don't want `fn` to accumulate gradient values
                # if it contains a backward pass. So we clear the
                # provided gradients
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                g.replay()
                prof.step()
    return post_process(output_dir, name)


def do_bench_kineto(
    fn: Callable,
    warmup: int,
    rep: int,
    grad_to_none=None,
    fast_flush=True,
    profile_opts=None,
    output_dir=None,
    use_cuda_graphs: bool = False,
    skip_cache_clearing: bool = False,
) -> str:
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    :param profile_opts: Options to pass into profiler.profile
    :type profile_opts: dict, optional
    :param output_dir: Output directory to store the trace
    :type output_dir: str, optional
    """
    if profile_opts is None:
        profile_opts = DEFAULT_PROFILE_OPTS
    import torch

    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if not skip_cache_clearing:
        if fast_flush:
            cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
        else:
            cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
        clear_cache = cache.zero_
    else:
        clear_cache = lambda *args: None

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        clear_cache()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # Calculate number of iterations based on target rep time
    if estimate_ms == 0:
        n_warmup = DEFAULT_N_WARMUP
        n_repeat = DEFAULT_N_REP  # Default if function is very fast
    else:
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))

    if use_cuda_graphs:
        return do_bench_kineto_cudagraph(
            fn, clear_cache, n_warmup, n_repeat, grad_to_none, profile_opts, output_dir
        )

    activity_groups = [
        profiler.ProfilerActivity.CUDA,
        profiler.ProfilerActivity.CPU,
    ]
    prefix = f"tritonbench_{fn._name}"
    name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{''.join(random.choices(string.digits, k=10))}.json"
    with profiler.profile(
        schedule=profiler.schedule(
            wait=0, warmup=n_warmup + n_repeat - 1, active=1, repeat=1
        ),
        activities=activity_groups,
        record_shapes=profile_opts["record_shapes"],
        profile_memory=profile_opts["profile_memory"],
        with_stack=profile_opts["with_stack"],
        with_flops=profile_opts["with_flops"],
        with_modules=profile_opts["with_modules"],
        on_trace_ready=(
            partial(trace_handler, name)
            if not hasattr(torch.version, "git_version")
            else profiler.tensorboard_trace_handler(output_dir, use_gzip=True)
        ),
    ) as prof:
        for i in range(n_warmup + n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            # we clear the L2 cache before run
            clear_cache()
            fn()
            prof.step()
    return post_process(output_dir, name)


def do_bench_kineto_walltime(fn, repcnt=5, profile_opts=None, output_dir=None):
    if profile_opts is None:
        profile_opts = DEFAULT_PROFILE_OPTS
    import torch

    fn()
    torch.cuda.synchronize()

    activity_groups = [
        profiler.ProfilerActivity.CUDA,
        profiler.ProfilerActivity.CPU,
    ]
    prefix = f"tritonbench_{fn._name}"
    name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{''.join(random.choices(string.digits, k=10))}.json"
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=repcnt - 1, active=1, repeat=1),
        activities=activity_groups,
        record_shapes=profile_opts["record_shapes"],
        profile_memory=profile_opts["profile_memory"],
        with_stack=profile_opts["with_stack"],
        with_flops=profile_opts["with_flops"],
        with_modules=profile_opts["with_modules"],
        on_trace_ready=(
            partial(trace_handler, name)
            if not hasattr(torch.version, "git_version")
            else profiler.tensorboard_trace_handler(output_dir, use_gzip=True)
        ),
    ) as prof:
        for i in range(repcnt):
            fn()
            torch.cuda.synchronize()
            prof.step()
    return post_process(output_dir, name)
