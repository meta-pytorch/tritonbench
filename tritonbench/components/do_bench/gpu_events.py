import torch
import triton
import triton.language as tl
from tritonbench.utils.constants import DEFAULT_N_REP, DEFAULT_N_WARMUP
from tritonbench.utils.env_utils import is_hip
from tritonbench.utils.gpu_utils import sleep_amd

from .common import summarize_statistics

_kernel_unblock_stream = None


def _get_unblocking_stream(device: torch.device):
    """
    Get a new stream for the given device.
    """
    global _kernel_unblock_stream
    if _kernel_unblock_stream is None:
        _kernel_unblock_stream = torch.cuda.Stream(device=device)
    return _kernel_unblock_stream


@triton.jit
def _block_stream_kernel(
    signal_ptr,
    timeout_ptr,
    sleep_ns: tl.constexpr = 1000000,
    signal: tl.constexpr = 1,
    is_amd: tl.constexpr = False,
):
    """
    Sleep kernel that performs an iterative check on a single value
    global memory buffer using volatile memory access.

    Keeps checking until the value changes from 0 to nonzero.
    Once the value is nonzero, the kernel stops checking and returns.
    Sleeps for a few milliseconds between checks to reduce contention.

    Args:
        buffer_ptr: Pointer to a single-element buffer in global memory.
        sleep_ns: Sleep duration in nanoseconds between checks (default: 1ms = 1,000,000 ns).
        signal: The value to unblock the stream.
        is_amd: Whether to use AMD-specific sleep instruction.
    """
    value = 0
    timeout = 1000
    num_checks = 0
    while value != signal and num_checks <= timeout:
        # Read the value from global memory using volatile memory access
        value = tl.load(signal_ptr, volatile=True)

        # Sleep for a few milliseconds before checking again to reduce polling overhead
        if is_amd:
            sleep_amd(sleep_ns)
        else:
            # NVIDIA: CUDA PTX nanosleep instruction
            tl.inline_asm_elementwise(
                "nanosleep.u32 $1;",
                "=r, r",
                args=[sleep_ns],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        num_checks += 1

    if value == signal:
        # Set the timeout buffer to 0 if the value is nonzero
        tl.atomic_xchg(timeout_ptr, 0)


def _block_stream(
    signal_buffer: torch.Tensor,
    timeout_buffer: torch.Tensor,
    sleep_ns: int = 1000000,
    signal: int = 1,
    is_amd: bool = False,
):
    """
    Block stream function that calls the block_stream_kernel.

    Args:
        signal_buffer: Pointer to a single-element buffer in global memory.
        timeout_buffer: Pointer to a single-element buffer for indicating
            timeout in global memory. It's intalized with a non-zero value.
        sleep_ns: Sleep duration in nanoseconds between checks (default: 1ms = 1,000,000 ns).
        signal: The value to unblock the stream.
        is_amd: Whether to use AMD-specific sleep instruction.
    """
    _block_stream_kernel[(1,)](
        signal_buffer, timeout_buffer, sleep_ns, signal, is_amd, num_warps=1
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


def do_bench_events(
    fn,
    warmup,
    rep,
    return_mode="all",
    grad_to_none=None,
    use_cudagraph=False,
    skip_cache_clearing=False,
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
    assert not use_cudagraph, "CUDA graphs are not supported for the gpu_events mode"

    # Detect AMD for GPU sleep
    amd_device = is_hip()

    # Get cache for L2 cache clearing
    cache = (
        triton.runtime.driver.active.get_empty_cache_for_benchmark()
        if not skip_cache_clearing
        else None
    )

    # First, estimate the runtime to calculate iterations
    estimate_ms = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        grad_to_none=grad_to_none,
        return_mode="mean",
    )

    # Calculate number of iterations based on target rep time
    if estimate_ms == 0:
        n_repeat = DEFAULT_N_REP  # Default if function is very fast
    else:
        # Run at least 10 iterations to get a reasonable estimate
        # and maximum of 50 iterations to avoid filling up the kernel queue
        n_repeat = min(max(10, int(rep / estimate_ms)), 50)

    clear_cache_fn = cache.zero_ if not skip_cache_clearing else lambda *args: None
    if grad_to_none is not None:

        def grad_to_none_fn():
            for x in grad_to_none:
                x.grad = None
    else:
        grad_to_none_fn = lambda *args: None

    # Signal, buffer, and stream for blocking/unblocking the stream
    signal = 1
    signal_buffer = torch.zeros(1, dtype=torch.int32, device="cuda")
    timeout_buffer = torch.ones(1, dtype=torch.int32, device="cuda")
    unblocking_stream = _get_unblocking_stream(signal_buffer.device)

    # Warm up block and unblock streams
    _unblock_stream(signal_buffer=signal_buffer, signal=signal)
    _block_stream(
        signal_buffer=signal_buffer,
        timeout_buffer=timeout_buffer,
        signal=signal,
        is_amd=amd_device,
    )
    torch.cuda.synchronize()

    # Regular mode warmup
    n_warmup = (
        max(1, int(warmup / estimate_ms)) if estimate_ms > 0 else DEFAULT_N_WARMUP
    )
    for _ in range(n_warmup):
        grad_to_none_fn()
        clear_cache_fn()
        fn()
    torch.cuda.synchronize()
    fn_only_bench = grad_to_none is None and skip_cache_clearing

    # Time events
    num_events = n_repeat + 1 if fn_only_bench else n_repeat * 2
    time_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]

    if fn_only_bench:

        def _bench_fn(i):
            time_events[i].record()
            fn()
    else:

        def _bench_fn(i):
            grad_to_none_fn()
            clear_cache_fn()
            time_events[i * 2].record()
            fn()
            time_events[i * 2 + 1].record()

    to_bench = True
    while to_bench:
        # Reset the signal and timeout buffers
        signal_buffer.fill_(0)
        timeout_buffer.fill_(1)
        torch.cuda.synchronize()

        # Start benchmarking
        # Block the stream until the kernel dispatching is complete
        _block_stream(
            signal_buffer=signal_buffer,
            timeout_buffer=timeout_buffer,
            signal=signal,
            is_amd=amd_device,
        )

        # Benchmark
        for i in range(n_repeat):
            _bench_fn(i)
        if fn_only_bench:
            time_events[-1].record()

        # Unblock the stream to allow the benchmark to run
        with torch.cuda.stream(unblocking_stream):
            _unblock_stream(signal_buffer=signal_buffer, signal=signal)

        # Wait for the events to complete
        torch.cuda.synchronize()

        # Stop benchmarking even when fail when n_repeat is 1 since we cannot
        # futher reduce the number of iterations
        if n_repeat == 1:
            break

        # Reduce the number of iterations in case the benchmark has to be rerun
        n_repeat = max(1, n_repeat // 2)

        # Rerun the benchmark if timeout occurs in the previous run
        to_bench = timeout_buffer.item() != 0

    assert timeout_buffer.item() == 0, (
        "Failed to run the benchmark since the block_stream buffer runs into "
        "timeout even when n_repeat = 1. Consider reducing the number of kernels "
        "dispatched in a single iteration and run with CUDA_SCALE_LAUNCH_QUEUES=4x"
    )

    if fn_only_bench:
        all_kernel_times = [
            time_events[i].elapsed_time(time_events[i + 1]) for i in range(n_repeat)
        ]
    else:
        all_kernel_times = [
            time_events[i * 2].elapsed_time(time_events[i * 2 + 1])
            for i in range(n_repeat)
        ]
    times = torch.tensor(all_kernel_times, dtype=torch.float)
    return summarize_statistics(times, quantiles=None, return_mode=return_mode)
