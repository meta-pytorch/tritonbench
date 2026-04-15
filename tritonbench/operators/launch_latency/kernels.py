import torch
import triton
import triton.language as tl


@triton.jit
def nop_kernel():
    pass


@triton.jit
def nop_with_args_kernel(
    t1,
    t2,
    t3,
    t4,
    t5,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    c5: tl.constexpr,
):
    pass


@triton.jit
def nop_with_kwargs_kernel(
    t1,
    t2,
    t3,
    t4,
    t5,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    BLOCK_C1: tl.constexpr = 32,
    BLOCK_C2: tl.constexpr = 32,
    BLOCK_C3: tl.constexpr = 32,
    BLOCK_C4: tl.constexpr = 32,
    BLOCK_C5: tl.constexpr = 32,
):
    pass


def get_inductor_nop_kernel_0arg():
    """Minimal torch.compile'd function — 0 external args.

    Internally operates on a pre-allocated tensor to force exactly one kernel
    launch, but the caller invokes it with no arguments.
    """
    x = torch.zeros(1, device="cuda")

    @torch.compile
    def _nop_impl(x):
        x.add_(0)

    def nop_0arg():
        _nop_impl(x)

    return nop_0arg


def get_inductor_nop_kernel_19arg():
    """Minimal torch.compile'd function with 19 args matching the triton nop_with_args_kernel signature.

    Uses a fixed signature (not *args) so torch.compile doesn't need to handle
    variable-length args, and the compiled graph is stable.
    """

    @torch.compile
    def nop_19arg(
        t1, t2, t3, t4, t5, i1, i2, i3, i4, i5, i6, i7, i8, i9, c1, c2, c3, c4, c5
    ):
        t1.add_(0)

    return nop_19arg


def get_inductor_nop_kernel(tensor_args=None):
    """Extract a single CachingAutotuner.run() call from a compiled nop kernel.

    Compiles a nop kernel via torch.compile, intercepts CachingAutotuner.run()
    to capture the kernel instance and exact args, then returns a callable that
    replays kernel.run() once.

    This directly measures CachingAutotuner.run() (per-kernel overhead) without
    guard check, DeviceGuard, or assert_size_stride — i.e. the cost each kernel
    pays inside a multi-kernel compiled graph.

    When tensor_args has ≥2 tensors, compiles a multi-tensor element-wise sum
    so the CachingAutotuner.run() call receives more positional args, allowing
    measurement of per-kernel overhead scaling with arg count.
    """
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    targs = (
        [a for a in tensor_args if isinstance(a, torch.Tensor)] if tensor_args else []
    )

    # Monkey-patch CachingAutotuner.run at the class level to capture the
    # kernel instance + exact call args during the first compiled execution.
    captured = []
    original_run = CachingAutotuner.run

    def capturing_run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        captured.append((self, args, kwargs))
        return result

    CachingAutotuner.run = capturing_run
    try:
        if len(targs) < 2:
            x = torch.zeros(1, device="cuda")

            @torch.compile
            def _nop(t):
                t.add_(0)

            _nop(x)
        else:

            @torch.compile
            def _nop_multi(t1, t2, t3, t4, t5):
                return t1 + t2 + t3 + t4 + t5

            _nop_multi(*targs[:5])
    finally:
        CachingAutotuner.run = original_run

    if not captured:
        raise RuntimeError("No CachingAutotuner.run() calls captured")

    kernel, run_args, run_kwargs = captured[-1]

    def run():
        kernel.run(*run_args, **run_kwargs)

    return run
