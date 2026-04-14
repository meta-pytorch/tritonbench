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


def get_inductor_nop_multi_kernel(n_kernels=100):
    """Simulate a compiled graph with n_kernels separate triton kernel launches.

    Compiles a single nop kernel via torch.compile, extracts the CachingAutotuner
    instance from the generated code, then calls kernel.run() N times in a loop.

    This directly measures N × CachingAutotuner.run() (per-kernel overhead)
    without fighting inductor's fusion optimizer.
    """
    import glob
    import importlib.util
    import os

    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    x = torch.zeros(1, device="cuda")

    @torch.compile
    def _nop(t):
        t.add_(0)

    _nop(x)

    # Find the generated inductor code in the cache
    cache_dir = os.path.join(
        os.environ.get(
            "TORCHINDUCTOR_CACHE_DIR", f"/var/tmp/torchinductor_{os.environ['USER']}"
        ),
    )
    py_files = glob.glob(os.path.join(cache_dir, "**", "*.py"), recursive=True)
    candidates = []
    for f in py_files:
        try:
            with open(f) as fh:
                content = fh.read()
                if "fused_add_copy" in content and "def call(args)" in content:
                    candidates.append((os.path.getmtime(f), f))
        except (OSError, UnicodeDecodeError):
            pass
    if not candidates:
        raise RuntimeError("Could not find generated inductor code in cache")
    candidates.sort(reverse=True)

    spec = importlib.util.spec_from_file_location("generated_multi", candidates[0][1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the CachingAutotuner kernel instance
    kernel = None
    for name in dir(mod):
        attr = getattr(mod, name, None)
        if isinstance(attr, CachingAutotuner):
            kernel = attr
            break
    if kernel is None:
        raise RuntimeError("Could not find CachingAutotuner in generated module")

    stream = get_raw_stream(0)

    def run():
        for _ in range(n_kernels):
            kernel.run(x, x, 1, stream=stream)

    return run
