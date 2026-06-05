import random
import string
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.profiler as profiler
from tritonbench.components.do_bench.utils import (
    estimate_cuda_runtime_ms,
    resolve_warmup_and_rep,
)
from tritonbench.utils.constants import DEFAULT_N_REP, DEFAULT_N_WARMUP
from tritonbench.utils.env_utils import has_manifold

if not hasattr(torch.version, "git_version"):
    from .fb.run_utils import trace_handler

DEFAULT_PROFILE_OPTS = {
    "record_shapes": True,
    "profile_memory": True,
    "with_stack": True,
    "with_flops": True,
    "with_modules": True,
}


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


def post_process(output_dir, name, skip_manifold: bool = False) -> str:
    if not output_dir:
        return f"https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/tritonbench/{name}.gz&bucket=pyper_traces"
    latest = _find_the_latest_file(output_dir, glob_pattern="*.json.gz")
    trace_file = latest or name
    if not skip_manifold and has_manifold():
        cmd = [
            "manifold",
            "put",
            trace_file,
            f"pyper_traces/tree/traces/tritonbench/{trace_file}",
        ]
        subprocess.check_call(cmd, cwd=output_dir)
        return f"https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/tritonbench/{trace_file}&bucket=pyper_traces"
    return str(Path(output_dir) / trace_file)


def do_bench_kineto_cudagraph(
    fn,
    clear_cache,
    n_warmup,
    n_repeat,
    grad_to_none,
    profile_opts,
    output_dir,
    skip_manifold: bool = False,
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
        fn_name = getattr(fn, "_name", getattr(fn, "__name__", "unknown"))
        prefix = f"tritonbench_cudagraph_{fn_name}"
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
            on_trace_ready=_get_on_trace_ready(output_dir, name),
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
    return post_process(output_dir, name, skip_manifold=skip_manifold)


def _get_on_trace_ready(output_dir, name):
    if not output_dir and not hasattr(torch.version, "git_version"):
        return partial(trace_handler, name)
    return profiler.tensorboard_trace_handler(output_dir, use_gzip=True)


def do_bench_kineto(
    fn: Callable,
    warmup: Optional[int],
    rep: Optional[int],
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
    :param output_dir: Output directory to store the trace locally. When set,
        the trace is saved as a gzipped JSON file in this directory and the
        return value is the local file path. When ``None``, the trace is
        uploaded to Manifold on internal builds.
    :type output_dir: str, optional
    """
    if profile_opts is None:
        profile_opts = DEFAULT_PROFILE_OPTS
    import shutil
    import tempfile

    import torch

    user_output_dir = output_dir is not None
    _temp_output_dir = None
    if output_dir is None and hasattr(torch.version, "git_version"):
        _temp_output_dir = tempfile.mkdtemp(prefix="tritonbench_kineto_")
        output_dir = _temp_output_dir

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

    estimate_ms = estimate_cuda_runtime_ms(fn, clear_cache_fn=clear_cache)
    warmup, rep = resolve_warmup_and_rep(warmup, rep, estimate_ms)

    # Calculate number of iterations based on target rep time
    if estimate_ms == 0:
        n_warmup = DEFAULT_N_WARMUP
        n_repeat = DEFAULT_N_REP  # Default if function is very fast
    else:
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))

    if use_cuda_graphs:
        return do_bench_kineto_cudagraph(
            fn,
            clear_cache,
            n_warmup,
            n_repeat,
            grad_to_none,
            profile_opts,
            output_dir,
            skip_manifold=user_output_dir,
        )

    activity_groups = [
        profiler.ProfilerActivity.CUDA,
        profiler.ProfilerActivity.CPU,
    ]
    fn_name = getattr(fn, "_name", getattr(fn, "__name__", "unknown"))
    prefix = f"tritonbench_{fn_name}"
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
        on_trace_ready=_get_on_trace_ready(output_dir, name),
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
    try:
        return post_process(output_dir, name, skip_manifold=user_output_dir)
    finally:
        if _temp_output_dir:
            shutil.rmtree(_temp_output_dir, ignore_errors=True)


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
    fn_name = getattr(fn, "_name", getattr(fn, "__name__", "unknown"))
    prefix = f"tritonbench_{fn_name}"
    name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{''.join(random.choices(string.digits, k=10))}.json"
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=repcnt - 1, active=1, repeat=1),
        activities=activity_groups,
        record_shapes=profile_opts["record_shapes"],
        profile_memory=profile_opts["profile_memory"],
        with_stack=profile_opts["with_stack"],
        with_flops=profile_opts["with_flops"],
        with_modules=profile_opts["with_modules"],
        on_trace_ready=_get_on_trace_ready(output_dir, name),
    ) as prof:
        for i in range(repcnt):
            fn()
            torch.cuda.synchronize()
            prof.step()
    return post_process(output_dir, name)
