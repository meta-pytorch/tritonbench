import os
import time

import torch
from torch import zeros
from torch._inductor.utils import triton_version_uses_attrs_dict
from triton.compiler import CompiledKernel
from tritonbench.utils.python_utils import try_import
from tritonbench.utils.triton_op import BenchmarkOperator, register_benchmark

with try_import("HAS_TILELANG"):
    import tilelang

    from .tilelang import tilelang_nop_kernel, tilelang_nop_with_args_kernel

with try_import("HAS_CUTEDSL"):
    import cutlass.cute as cute

    from .cutedsl import cutedsl_nop_kernel, cutedsl_nop_with_args_kernel

from .kernels import get_trivial_add_kernel, nop_kernel, nop_with_args_kernel


def _patch_triton_run_profiling():
    """Monkey-patch JITFunction.run to profile launch overhead breakdown."""
    from triton.runtime.jit import JITFunction

    if hasattr(JITFunction, "_original_run"):
        return  # already patched

    original_run = JITFunction.run

    def profiled_run(self_jit, *args, grid, warmup, **kwargs):
        import time as _time

        from triton.runtime.jit import compute_cache_key, driver, knobs

        _t0 = _time.perf_counter()

        kwargs["debug"] = kwargs.get("debug", self_jit.debug) or knobs.runtime.debug

        _t1 = _time.perf_counter()
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        _t2 = _time.perf_counter()

        for hook in self_jit.pre_run_hooks:
            hook(*args, **kwargs)

        _t3 = _time.perf_counter()
        kernel_cache, kernel_key_cache, target, backend, binder = (
            self_jit.device_caches[device]
        )
        bound_args, specialization, options = binder(*args, **kwargs)
        _t4 = _time.perf_counter()

        key = compute_cache_key(kernel_key_cache, specialization, options)
        kernel = kernel_cache.get(key, None)
        _t5 = _time.perf_counter()

        if kernel is None:
            # Fall back to original for compilation
            JITFunction.run = original_run
            result = original_run(self_jit, *args, grid=grid, warmup=warmup, **kwargs)
            JITFunction.run = profiled_run
            return result

        not_present = object()
        for (name, _), (val, globals_dict) in self_jit.used_global_vals.items():
            if globals_dict.get(name, not_present) != val:
                raise RuntimeError(f"Global variable {name} has changed")
        _t6 = _time.perf_counter()

        if not warmup:
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            if hasattr(kernel, "result"):
                kernel = kernel.result()

            _t7 = _time.perf_counter()
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(
                grid_0,
                grid_1,
                grid_2,
                stream,
                kernel.function,
                kernel.packed_metadata,
                launch_metadata,
                knobs.runtime.launch_enter_hook,
                knobs.runtime.launch_exit_hook,
                *bound_args.values(),
            )
            _t8 = _time.perf_counter()

            _profiled_run_samples.append(
                (
                    (_t1 - _t0) * 1e6,  # preamble
                    (_t2 - _t1) * 1e6,  # driver (device+stream)
                    (_t3 - _t2) * 1e6,  # hooks
                    (_t4 - _t3) * 1e6,  # binder
                    (_t5 - _t4) * 1e6,  # cache_key+lookup
                    (_t6 - _t5) * 1e6,  # global_check
                    (_t7 - _t6) * 1e6,  # grid
                    (_t8 - _t7) * 1e6,  # launch
                    (_t8 - _t0) * 1e6,  # TOTAL
                )
            )

        return kernel

    JITFunction._original_run = original_run
    JITFunction.run = profiled_run


_profiled_run_samples = []


def _print_profiling_summary():
    if not _profiled_run_samples:
        return
    labels = [
        "preamble",
        "driver",
        "hooks",
        "binder",
        "cache_key+lookup",
        "global_check",
        "grid",
        "launch",
        "TOTAL",
    ]
    n = len(_profiled_run_samples)
    # skip first 10% as warmup
    skip = max(1, n // 10)
    samples = _profiled_run_samples[skip:]
    if not samples:
        samples = _profiled_run_samples
    avgs = [sum(s[i] for s in samples) / len(samples) for i in range(len(labels))]
    print(
        "\n[triton-launch-profile] Average over %d samples (skipped %d warmup):"
        % (len(samples), skip)
    )
    for label, avg in zip(labels, avgs):
        print(f"  {label:20s}: {avg:8.2f} us")
    print()


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["walltime"]
    FWD_ONLY = True

    def get_input_iter(self):
        yield tuple()
        targs = [zeros(1, device="cuda") for _ in range(5)]
        iargs = [1 for _ in range(9)]
        cargs = [32 for _ in range(5)]
        yield tuple([*targs, *iargs, *cargs])

    def get_x_val(self, example_inputs) -> float:
        return len(example_inputs)

    @register_benchmark()
    def nop_triton_kernel(self, *args):
        if os.environ.get("TRITON_LAUNCH_LATENCY_PROFILE", "0") == "1":
            _patch_triton_run_profiling()
            _profiled_run_samples.clear()
            if len(args) == 0:
                fn = lambda: nop_kernel[1,]()
            else:
                fn = lambda: nop_with_args_kernel[1,](*args)

            def profiled_fn():
                fn()

            import atexit

            atexit.register(_print_profiling_summary)
            return profiled_fn
        if len(args) == 0:
            return lambda: nop_kernel[1,]()
        return lambda: nop_with_args_kernel[1,](*args)

    @register_benchmark()
    def nop_triton_compiled_kernel_run(self, *args):
        if len(args) == 0:
            bin = nop_kernel[1,]()

        else:
            bin = nop_with_args_kernel[1,](*args)
            # triton <= 3.3 does not include tl.constexpr args in call
            # but triton 3.4 does
            if not triton_version_uses_attrs_dict():
                args = args[:-5]
        function = bin.function
        metadata = (
            bin.packed_metadata if hasattr(bin, "packed_metadata") else bin.metadata
        )
        if hasattr(CompiledKernel, "launch_metadata"):
            return lambda: bin.run(
                1, 1, 1, 0, function, metadata, None, None, None, *args
            )
        else:
            return lambda: bin.run(
                1, 1, 1, 1, 1, 1, 1, 1, 0, 0, function, None, None, metadata, *args
            )

    @register_benchmark()
    def nop_inductor_kernel(self, *args):
        trivial_add_kernel = get_trivial_add_kernel()
        return lambda: trivial_add_kernel(*args)

    @register_benchmark(enabled=HAS_TILELANG)
    def nop_tilelang(self, *args):
        if len(args) == 0:
            kernel = tilelang_nop_kernel()
            return lambda: kernel()
        kernel = tilelang_nop_with_args_kernel()
        return lambda: kernel(*args)

    @register_benchmark(enabled=HAS_CUTEDSL)
    def nop_cutedsl(self, *args):
        if len(args) == 0:
            kernel = cute.compile(cutedsl_nop_kernel)
            return lambda: kernel()
        cute_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                cute_args.append(cute.runtime.from_dlpack(arg))
            else:
                cute_args.append(arg)
        kernel = cute.compile(cutedsl_nop_with_args_kernel, *cute_args)
        # remove constexpr args
        cute_args = cute_args[:-5]
        return lambda: kernel(*cute_args)

    @register_benchmark(enabled=HAS_CUTEDSL)
    def nop_cutedsl_tvm_ffi(self, *args):
        if len(args) == 0:
            kernel = cute.compile(cutedsl_nop_kernel)
            return lambda: kernel()
        cute_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                cute_args.append(cute.runtime.from_dlpack(arg, enable_tvm_ffi=True))
            else:
                cute_args.append(arg)
        kernel = cute.compile(
            cutedsl_nop_with_args_kernel, *cute_args, options="--enable-tvm-ffi"
        )
        # remove constexpr args
        cute_args = cute_args[:-5]
        return lambda: kernel(*cute_args)

    @register_benchmark(baseline=True)
    def nop_python_function(self, *args):
        # Dump JITFunction.run source on first call for profiling investigation
        if os.environ.get("TRITON_LAUNCH_LATENCY_PROFILE", "0") == "1":
            import inspect

            from triton.runtime.jit import JITFunction

            print("\n=== JITFunction members ===")
            print([m for m in dir(JITFunction) if not m.startswith("__")])
            print("\n=== JITFunction.run source ===")
            print(inspect.getsource(JITFunction.run))

        def nop():
            pass

        return nop
