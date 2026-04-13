import os
import time

import torch
from torch import zeros
from torch._inductor.utils import triton_version_uses_attrs_dict
from triton.compiler import CompiledKernel
from tritonbench.utils.env_utils import is_cuda
from tritonbench.utils.python_utils import try_import
from tritonbench.utils.triton_op import BenchmarkOperator, register_benchmark

with try_import("HAS_TILELANG"):
    import tilelang

    from .tilelang import tilelang_nop_kernel, tilelang_nop_with_args_kernel

with try_import("HAS_CUTEDSL"):
    import cutlass.cute as cute

    from .cutedsl import cutedsl_nop_kernel, cutedsl_nop_with_args_kernel

from .kernels import (
    get_trivial_add_kernel,
    nop_kernel,
    nop_with_args_kernel,
    nop_with_kwargs_kernel,
)


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


def _prepare_direct_culaunch(bin, args):
    """
    Pre-extract everything needed for a direct cuLaunchKernel call via ctypes.
    Simulates what TritonCC/AOT-T/AOTI do: all handles pre-cached,
    no Python binder, no PyArg_ParseTuple, no cuPointerGetAttribute.
    """
    import ctypes

    libcuda = ctypes.CDLL("libcuda.so.1")
    _cuLaunchKernel = libcuda.cuLaunchKernel
    _cuLaunchKernel.restype = ctypes.c_int

    cu_function = bin.function
    stream_handle = torch.cuda.current_stream().cuda_stream

    if hasattr(bin, "packed_metadata"):
        pm = bin.packed_metadata
        num_warps, shared_mem = pm[0], pm[2]
    else:
        num_warps = bin.metadata.num_warps
        shared_mem = bin.metadata.shared

    # Pre-cast all values to ctypes — no Python object creation per call
    _func = ctypes.c_void_p(cu_function)
    _g1 = ctypes.c_uint(1)
    _bx = ctypes.c_uint(32 * num_warps)
    _b1 = ctypes.c_uint(1)
    _shared = ctypes.c_uint(shared_mem)
    _stream = ctypes.c_void_p(stream_handle)
    _null = ctypes.c_void_p(0)

    if len(args) == 0:
        return lambda: _cuLaunchKernel(
            _func, _g1, _g1, _g1, _bx, _b1, _b1, _shared, _stream, _null, _null
        )
    else:
        # Pre-extract device pointers and scalars (one-time, like [3][4][5] build time)
        param_values = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                param_values.append(ctypes.c_uint64(arg.data_ptr()))
            elif isinstance(arg, int):
                param_values.append(ctypes.c_int32(arg))
            else:
                param_values.append(ctypes.c_int32(int(arg)))
        n = len(param_values)
        ParamsArray = ctypes.c_void_p * n
        param_ptrs = ParamsArray(
            *[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in param_values]
        )
        return lambda _pv=param_values: _cuLaunchKernel(
            _func, _g1, _g1, _g1, _bx, _b1, _b1, _shared, _stream, param_ptrs, _null
        )


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
    def nop_triton_kernel_kwargs(self, *args):
        """Same as nop_triton_kernel but passes constexpr params as kwargs."""
        if len(args) == 0:
            return lambda: nop_kernel[1,]()
        pos_args = args[:14]
        kw_vals = args[14:] if len(args) > 14 else (32, 32, 32, 32, 32)
        return lambda: nop_with_kwargs_kernel[1,](
            *pos_args,
            BLOCK_C1=kw_vals[0],
            BLOCK_C2=kw_vals[1],
            BLOCK_C3=kw_vals[2],
            BLOCK_C4=kw_vals[3],
            BLOCK_C5=kw_vals[4],
        )

    @register_benchmark()
    def nop_triton_kernel_new_tensors(self, *args):
        """Layer 1 misses (different tensor objects each call), Layer 2 should hit."""
        if len(args) == 0:
            return lambda: nop_kernel[1,]()
        # Pre-allocate N sets of cloned tensors to avoid allocation in the hot loop.
        N = 100
        targs = args[:5]
        rest_args = args[5:]
        tensor_sets = [tuple(t.clone() for t in targs) for _ in range(N)]
        idx = [0]

        def fn():
            i = idx[0] % N
            idx[0] += 1
            nop_with_args_kernel[1,](*tensor_sets[i], *rest_args)

        return fn

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

    @register_benchmark(enabled=is_cuda())
    def nop_triton_direct_culaunch(self, *args):
        """Simulate [3][4][5] (TritonCC/AOT-T/AOTI) style launch:
        pre-compile kernel, pre-extract all handles, call cuLaunchKernel
        directly via ctypes. No Python binder, no PyArg_ParseTuple, no
        cuPointerGetAttribute — just the raw CUDA driver call."""
        if len(args) == 0:
            bin = nop_kernel[1,]()
        else:
            bin = nop_with_args_kernel[1,](*args)
            if not triton_version_uses_attrs_dict():
                args = args[:-5]
        return _prepare_direct_culaunch(bin, args)

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
