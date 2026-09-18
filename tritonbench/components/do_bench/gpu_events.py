import threading
import time
from functools import lru_cache
from typing import Any, Callable, Optional

import torch
import triton
import triton.language as tl
from tritonbench.utils.constants import DEFAULT_N_REP, DEFAULT_N_WARMUP
from tritonbench.utils.cudagraph_utils import CudaGraphConfig, CudaGraphError
from tritonbench.utils.env_utils import get_current_device, get_device_module, is_hip

from .common import summarize_statistics
from .utils import resolve_warmup_and_rep

AMD_SLEEP_NS_PER_ITERATION = 3870

# Poll-iteration budget for _block_stream_kernel before giving up.
DEFAULT_BLOCK_STREAM_TIMEOUT_ITERS = 10000

# Host delay in _supports_stream_blocking between launching the spinner and
# releasing it. Long enough that the spinner is certainly already running.
_STREAM_BLOCKING_PROBE_DELAY_S = 0.05

_kernel_unblock_stream = None


@triton.jit
def sleep_amd(sleep_ns: tl.constexpr = 1000000):
    """
    AMD GPU sleep using s_sleep instruction.

    Each iteration of s_sleep 127 sleeps for ~127*64 = 8,128 clock cycles.
    On MI300X @ 2.1 GHz, this is approximately 3.87 μs per iteration.
    On MI350X @ 2.2 GHz, this is approximately 3.69 μs per iteration.

    Args:
        sleep_ns: Target sleep duration in nanoseconds.
                 Default 1000000 (1ms).

    Note:
        Timing is approximate and varies with GPU clock frequency.
    """
    # Calculate iterations: sleep_ns / 3870 ns per iteration
    num_iterations: tl.constexpr = max(1, sleep_ns // AMD_SLEEP_NS_PER_ITERATION)
    for _ in range(num_iterations):
        tl.inline_asm_elementwise(
            "s_sleep 127",
            "=r",
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


def _get_unblocking_stream(device: torch.device):
    """
    Get a new stream for the given device.
    """
    global _kernel_unblock_stream
    if _kernel_unblock_stream is None:
        _kernel_unblock_stream = get_device_module(device.type).Stream(device=device)
    return _kernel_unblock_stream


@triton.jit
def _block_stream_kernel(
    signal_ptr,
    timeout_ptr,
    sleep_ns: tl.constexpr = 1000000,
    signal: tl.constexpr = 1,
    device_type: tl.constexpr = "cuda",
    timeout: tl.constexpr = DEFAULT_BLOCK_STREAM_TIMEOUT_ITERS,
):
    """
    Sleep kernel that performs an iterative check on a single value
    global memory buffer using volatile memory access.

    Keeps checking until the value changes from 0 to nonzero.
    Once the value is nonzero, the kernel stops checking and returns.
    Sleeps for a few milliseconds between checks to reduce contention.

    Args:
        buffer_ptr: Pointer to a single-element buffer in global memory.
        sleep_ns: Sleep duration in nanoseconds between checks (default: 1ms).
        signal: The value to unblock the stream.
        device_type: Backend used to pick the in-kernel sleep instruction
            ("hip" -> s_sleep, "cuda" -> NVIDIA nanosleep; anything else
            busy-polls, e.g. Intel XPU has no sleep intrinsic exposed
            through Triton).
        timeout: Max poll iterations before giving up.
    """
    value = 0
    num_checks = 0
    while value != signal and num_checks <= timeout:
        # Read the value from global memory using volatile memory access
        value = tl.load(signal_ptr, volatile=True)

        # Sleep for a few milliseconds before checking again to reduce polling overhead
        if device_type == "hip":
            sleep_amd(sleep_ns)
        elif device_type == "cuda":
            # NVIDIA: CUDA PTX nanosleep instruction
            tl.inline_asm_elementwise(
                "nanosleep.u32 $1;",
                "=r, r",
                args=[sleep_ns],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        # Other backends (e.g. Intel XPU) have no sleep intrinsic exposed
        # through Triton, so they fall through to a plain busy-poll. Both
        # inline-asm variants above are backend-specific and will not compile
        # elsewhere; emitting them unconditionally is what made this mode
        # NVIDIA/AMD-only.
        num_checks += 1

    if value == signal:
        # Set the timeout buffer to 0 if the value is nonzero
        tl.atomic_xchg(timeout_ptr, 0)


def _block_stream(
    signal_buffer: torch.Tensor,
    timeout_buffer: torch.Tensor,
    sleep_ns: int = 1000000,
    signal: int = 1,
    device_type: str = "cuda",
):
    """
    Block stream function that calls the block_stream_kernel.

    Args:
        signal_buffer: Pointer to a single-element buffer in global memory.
        timeout_buffer: Pointer to a single-element buffer for indicating
            timeout in global memory. It's intalized with a non-zero value.
        sleep_ns: Sleep duration in nanoseconds between checks (default: 1ms = 1,000,000 ns).
        signal: The value to unblock the stream.
        device_type: Backend used to pick the in-kernel sleep instruction
            ("hip", "cuda", or anything else to busy-poll).
    """
    _block_stream_kernel[(1,)](
        signal_buffer,
        timeout_buffer,
        sleep_ns,
        signal,
        device_type,
        DEFAULT_BLOCK_STREAM_TIMEOUT_ITERS,
        num_warps=1,
    )


@triton.jit
def _unblock_stream_kernel(
    signal_ptr,
    signal: tl.constexpr = 1,
):
    """
    Unblock stream kernel that atomically sets a single value in global memory buffer.

    Args:
        signal_ptr: Pointer to a single-element buffer in global memory.
        signal: The value to atomically store in the buffer.
    """
    # Atomically exchange the buffer value with the input value
    tl.atomic_xchg(signal_ptr, signal)


def _unblock_stream(
    signal_buffer: torch.Tensor,
    signal: int = 1,
):
    """
    Unblock stream function that calls the unblock_stream_kernel.

    Args:
        signal_buffer: Pointer to a single-element buffer in global memory.
        signal: The value to atomically store in the buffer.
    """
    _unblock_stream_kernel[(1,)](signal_buffer, signal, num_warps=1)


@lru_cache(maxsize=None)
def _supports_stream_blocking(device_type: str) -> bool:
    """Can a spinning kernel observe a flag another stream writes mid-flight?

    The whole gpu_events design rests on this: a kernel parks on the compute
    stream polling a flag, the host enqueues the work behind it, then a kernel
    on a second stream releases it. That needs concurrent cross-stream
    execution *and* a device-side poll that actually re-reads memory each
    iteration. Both hold on NVIDIA/AMD, so the mode never checked; on Intel XPU
    ``tl.load(volatile=True)`` does not re-read, and the spinner exhausts its
    budget without ever seeing the update.

    Probed rather than hardcoded per backend, so this lights up on its own once
    a backend gains the guarantee. Costs one warm kernel launch when it works.
    """
    device = get_current_device()
    device_module = get_device_module(device)
    signal_buffer = torch.zeros(1, dtype=torch.int32, device=device)
    timeout_buffer = torch.ones(1, dtype=torch.int32, device=device)
    unblocking_stream = _get_unblocking_stream(signal_buffer.device)

    # Compile both kernels first so JIT time cannot be mistaken for a delay.
    _unblock_stream(signal_buffer=signal_buffer, signal=1)
    _block_stream(
        signal_buffer=signal_buffer,
        timeout_buffer=timeout_buffer,
        signal=1,
        device_type=device_type,
    )
    device_module.synchronize()

    signal_buffer.fill_(0)
    timeout_buffer.fill_(1)
    device_module.synchronize()
    _block_stream(
        signal_buffer=signal_buffer,
        timeout_buffer=timeout_buffer,
        signal=1,
        device_type=device_type,
    )
    # Signal only after the spinner is unambiguously already running, so a
    # backend that reads the flag once at entry cannot pass by accident.
    time.sleep(_STREAM_BLOCKING_PROBE_DELAY_S)
    with device_module.stream(unblocking_stream):
        _unblock_stream(signal_buffer=signal_buffer, signal=1)
    device_module.synchronize()
    return timeout_buffer.item() == 0


def _setup_stream_blocking(signal: int, device_type: str):
    """
    Allocate buffers and streams for stream blocking and warmup the
    blocking/unblocking stream kernels.
    """
    if not _supports_stream_blocking(device_type):
        raise NotImplementedError(
            f"latency_measure_mode='gpu_events' is not supported on device "
            f"'{get_current_device()}': a kernel spinning on the compute stream "
            f"never observes the release flag written from another stream, so "
            f"every measurement would time out. Use 'triton_do_bench', "
            f"'profiler', or --repcnt instead."
        )
    device = get_current_device()
    signal_buffer = torch.zeros(1, dtype=torch.int32, device=device)
    timeout_buffer = torch.ones(1, dtype=torch.int32, device=device)
    unblocking_stream = _get_unblocking_stream(signal_buffer.device)

    # Warm up block and unblock streams
    _unblock_stream(signal_buffer=signal_buffer, signal=signal)
    _block_stream(
        signal_buffer=signal_buffer,
        timeout_buffer=timeout_buffer,
        signal=signal,
        device_type=device_type,
    )
    get_device_module(device).synchronize()
    return signal_buffer, timeout_buffer, unblocking_stream


def _reset_stream_blocking_flags(
    signal_buffer: torch.Tensor,
    timeout_buffer: torch.Tensor,
):
    """
    Reset the blocking flags for the given signal buffer.
    """
    signal_buffer.fill_(0)
    timeout_buffer.fill_(1)


def _bench_with_stream_blocking(
    fn: Callable,
    compute_stream: torch.Stream,
    unblocking_stream: torch.Stream,
    signal_buffer: torch.Tensor,
    timeout_buffer: torch.Tensor,
    signal: int,
    n_repeat: int,
    device_type: str,
) -> Any:
    device_module = get_device_module(signal_buffer.device.type)
    to_bench = True
    while to_bench:
        # Reset the signal and timeout buffers
        _reset_stream_blocking_flags(signal_buffer, timeout_buffer)
        device_module.synchronize()

        with device_module.stream(compute_stream):
            # Start benchmarking
            # Block the stream until the kernel dispatching is complete
            _block_stream(
                signal_buffer=signal_buffer,
                timeout_buffer=timeout_buffer,
                signal=signal,
                device_type=device_type,
            )

            # Benchmark
            fn(n_repeat)

        # Unblock the stream to allow the benchmark to run
        with device_module.stream(unblocking_stream):
            _unblock_stream(signal_buffer=signal_buffer, signal=signal)

        # Wait for the events to complete
        device_module.synchronize()

        # Stop benchmarking even when fail when n_repeat is 1 since we cannot
        # futher reduce the number of iterations
        # Rerun the benchmark if timeout occurs in the previous run
        to_bench = n_repeat != 1 and timeout_buffer.item() != 0

        if to_bench:
            # Reduce the number of iterations
            n_repeat = max(1, n_repeat // 2)

    assert timeout_buffer.item() == 0, (
        "Failed to run the benchmark since the block_stream buffer runs into "
        "timeout even when n_repeat = 1. Consider reducing the number of kernels "
        "dispatched in a single iteration and run with CUDA_SCALE_LAUNCH_QUEUES=4x"
    )
    return n_repeat


def do_bench_events(
    fn,
    warmup,
    rep,
    return_mode="all",
    grad_to_none=None,
    use_cudagraph=False,
    skip_cache_clearing=False,
    cudagraph_config: Optional[CudaGraphConfig] = None,
):
    """Measure GPU kernel execution time using GPU events.

    This method profiles the function and extracts the actual GPU kernel execution
    time by summing up all CUDA kernel durations (excluding overlaps) from the profiler trace.

    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds (matches triton.testing.do_bench)
        rep: Target total measurement time in milliseconds (matches triton.testing.do_bench)
        return_mode: "all" for list of measurements, other modes for single values
        grad_to_none: Tensors whose gradients should be cleared before each measurement
        use_cudagraph: Whether to use CUDA graphs for benchmarking

    Returns:
        List of measured kernel times in milliseconds (if return_mode="all") or single value.
    """
    fn_only_bench = grad_to_none is None and skip_cache_clearing
    device = get_current_device()
    device_module = get_device_module(device)
    if use_cudagraph:
        assert (
            fn_only_bench
        ), "CUDA graphs only support grad_to_none=None and skip_cache_clearing=True"
        assert cudagraph_config is not None
        compute_stream = cudagraph_config.get_stream()
    else:
        compute_stream = device_module.current_stream()

    # Backend used to pick the in-kernel sleep instruction while the stream is
    # blocked ("hip", "cuda", or anything else to busy-poll).
    device_type = "hip" if is_hip() else device

    # Get cache for L2 cache clearing
    cache = (
        triton.runtime.driver.active.get_empty_cache_for_benchmark()
        if not skip_cache_clearing
        else None
    )

    clear_cache_fn = cache.zero_ if not skip_cache_clearing else lambda *args: None
    if grad_to_none is not None:

        def grad_to_none_fn():
            for x in grad_to_none:
                x.grad = None
    else:
        grad_to_none_fn = lambda *args: None

    # Setup buffer, and stream for blocking/unblocking the stream
    signal = 1
    signal_buffer, timeout_buffer, unblocking_stream = _setup_stream_blocking(
        signal, device_type=device_type
    )

    # Initial time events
    time_events = [device_module.Event(enable_timing=True) for _ in range(2)]

    # Estimate number of iterations based on target rep time
    if fn_only_bench:

        def _bench_loop_fn(n_repeat: int):
            time_events[0].record()
            for _ in range(n_repeat):
                fn()
            time_events[1].record()
    else:

        def _bench_loop_fn(n_repeat: int):
            time_events[0].record()
            for _ in range(n_repeat):
                grad_to_none_fn()
                fn()
            time_events[1].record()

    n_repeat = _bench_with_stream_blocking(
        _bench_loop_fn,
        compute_stream,
        unblocking_stream,
        signal_buffer,
        timeout_buffer,
        signal,
        n_repeat=10,
        device_type=device_type,
    )
    device_module.synchronize()

    estimate_ms = time_events[0].elapsed_time(time_events[1]) / n_repeat
    warmup, rep = resolve_warmup_and_rep(warmup, rep, estimate_ms)

    # Calculate number of warmup iterations based on target rep time
    if estimate_ms == 0:
        n_warmup = DEFAULT_N_WARMUP  # Default if function is very fast
    else:
        n_warmup = max(1, int(warmup / estimate_ms))

    # Regular mode warmup
    for _ in range(n_warmup):
        grad_to_none_fn()
        clear_cache_fn()
        fn()

    # Calculate number of iterations based on target rep time
    if estimate_ms == 0:
        n_repeat = DEFAULT_N_REP  # Default if function is very fast
    else:
        # Run at least 10 iterations to get a reasonable estimate
        n_repeat = max(10, int(rep / estimate_ms))

    if not fn_only_bench:
        additional_num_events = n_repeat * 2 - len(time_events)
        time_events += [
            device_module.Event(enable_timing=True)
            for _ in range(additional_num_events)
        ]

    # Run the benchmark
    if use_cudagraph:
        # Need to keep the rep count low for cudagraphs. So, we break the graph
        # into smaller chunks and replay many times otherwise the replay might
        # get stuck.
        max_num_kernels = 512
        num_kernels = cudagraph_config.get_num_kernels(fn)
        n_cudagraph_repeat = min(max(max_num_kernels // num_kernels, 1), n_repeat)

        n_replay = max(n_repeat // n_cudagraph_repeat, 1)
        device_module.synchronize()

        # Capture cudagraph
        fn_graph = cudagraph_config.get_graph()
        with device_module.graph(fn_graph, stream=cudagraph_config.get_stream()):
            for _ in range(n_cudagraph_repeat):
                fn()
        device_module.synchronize()

        def run_replay(event):
            try:
                with device_module.stream(cudagraph_config.get_stream()):
                    fn_graph.replay()
                device_module.synchronize()
                event.set()
            except Exception as e:
                print(f"An error occurred during CudaGraph replay: {e}", flush=True)

        # Warm up CudaGraph replay in another thread
        replay_event = threading.Event()
        thread = threading.Thread(target=run_replay, args=(replay_event,))
        thread.start()

        # Wait for the replay for 10 seconds
        is_done = replay_event.wait(10)
        if not is_done:
            # An attempt to send an interrupt to the replay thread by deleting
            # the graph when the replay is stuck or there is an error.  This is
            # unsafe and not gauranteed to work
            try:
                print("CudaGraph replay failed. Destroying graph", flush=True)
                cudagraph_config.reset_graph()
                thread.join()
                device_module.synchronize()
                print("CudaGraph replay thread cleanly terminated", flush=True)
            except Exception as e:
                # Raise
                raise CudaGraphError("CudaGraph capturing error: {}".format(e))
        else:
            thread.join()

        def _bench_loop_fn(n_replay: int):
            time_events[0].record()
            for i in range(n_replay):
                fn_graph.replay()
            time_events[1].record()

    elif fn_only_bench:

        def _bench_loop_fn(n_repeat: int):
            time_events[0].record()
            for i in range(n_repeat):
                fn()
            time_events[1].record()
    else:

        def _bench_loop_fn(n_repeat: int):
            for i in range(n_repeat):
                grad_to_none_fn()
                clear_cache_fn()
                time_events[i * 2].record()
                fn()
                time_events[i * 2 + 1].record()

    n_repeat = _bench_with_stream_blocking(
        _bench_loop_fn,
        compute_stream,
        unblocking_stream,
        signal_buffer,
        timeout_buffer,
        signal,
        n_repeat if not use_cudagraph else n_replay,
        device_type=device_type,
    )

    if use_cudagraph:
        n_repeat = n_repeat * n_cudagraph_repeat
        # Delete graph
        cudagraph_config.reset_graph()

    if fn_only_bench:
        kernel_time = time_events[0].elapsed_time(time_events[1]) / n_repeat
        assert kernel_time > 0, "Failed to run the benchmark since the kernel time is 0"
        all_kernel_times = [kernel_time] * n_repeat
    else:
        all_kernel_times = [
            time_events[i * 2].elapsed_time(time_events[i * 2 + 1])
            for i in range(n_repeat)
        ]
    times = torch.tensor(all_kernel_times, dtype=torch.float)
    return summarize_statistics(times, quantiles=None, return_mode=return_mode)
