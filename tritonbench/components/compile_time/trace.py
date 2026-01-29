import os
import random
import string
import time
from datetime import datetime
from functools import partial
from typing import Callable, Dict, Optional

import torch
import torch.profiler as profiler
from tritonbench.utils.env_utils import fresh_triton_cache, is_fbcode

if is_fbcode():
    from triton.fb.triton_util import triton_add_listener, TritonHook

DEFAULT_PROFILE_OPTS = {
    "record_shapes": True,
    "profile_memory": True,
    "with_stack": True,
    "with_flops": True,
    "with_modules": True,
}


def fbcode_do_compile_time_in_task(fn: Callable) -> Dict[str, float]:
    # not yet getting results that make sense to me
    detailed_data = {}
    with fresh_triton_cache():

        def _inner(**kwargs):
            stats = kwargs.get("stats", {})
            if not stats:
                return
            if "compile_time_stats" in stats:
                detailed_data["compile_time_stats"] = stats["compile_time_stats"]

        triton_add_listener(TritonHook.POST_COMPILE, _inner)
        fn()
    if "compile_time_stats" in detailed_data:
        return detailed_data["compile_time_stats"]
    return None


def do_compile_time_in_task(fn: Callable, cold_start: bool = False) -> float:
    with fresh_triton_cache():
        if not cold_start:
            # compile a dummy kernel to skip cold start overhead
            from tritonbench.kernels.nop import nop_kernel

            nop_kernel[1,]()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
    walltime_with_compile = (end_time - start_time) * 1e3
    return walltime_with_compile


def do_compile_kineto_trace_in_task(
    fn: Callable,
    profile_opts: Optional[Dict[str, bool]] = None,
    output_dir: Optional[str] = None,
    cold_start: bool = False,
) -> Optional[str]:
    """Profile compilation stage using Kineto."""
    activity_groups = [
        profiler.ProfilerActivity.CUDA,
        profiler.ProfilerActivity.CPU,
    ]
    if not profile_opts:
        profile_opts = DEFAULT_PROFILE_OPTS
    prefix = f"tritonbench_{fn._name}"
    name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{''.join(random.choices(string.digits, k=10))}.json"
    trace_path = os.path.join(output_dir, name)
    with fresh_triton_cache():
        if not cold_start:
            # compile a dummy kernel to skip cold start overhead
            from tritonbench.kernels.nop import nop_kernel

            nop_kernel[1,]()
        torch.cuda.synchronize()
        with profiler.profile(
            schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            activities=activity_groups,
            record_shapes=profile_opts["record_shapes"],
            profile_memory=profile_opts["profile_memory"],
            with_stack=profile_opts["with_stack"],
            with_flops=profile_opts["with_flops"],
            with_modules=profile_opts["with_modules"],
            on_trace_ready=(
                partial(lambda name, prof: prof.export_chrome_trace(name), trace_path)
            ),
        ) as prof:
            fn()
            prof.step()
        return trace_path
