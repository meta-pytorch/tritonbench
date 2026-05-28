import inspect

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Compatibility: c_cache kwarg was added in a newer Triton version.
# When tritonbench pins an older Triton, fall back to plain @triton.jit.
_jit_supports_c_cache = "c_cache" in inspect.signature(triton.jit).parameters


def _jit(**kwargs):
    if not _jit_supports_c_cache:
        kwargs.pop("c_cache", None)
    return triton.jit(**kwargs)


@_jit(c_cache=True)
def nop_kernel():
    pass


@_jit(c_cache=True)
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


@_jit(c_cache=True)
def nop_hstu_args_kernel(
    # 14 pointer args (simulating Q, K, V, seq_offsets, TS, TW, PW, Bias,
    # seq2_offsets, delta_x_offsets, num_targets, attn_scale, Out, M_buffer)
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    # 8 stride args (int)
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_ts,
    stride_om,
    # 18 scalar args (int/float)
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,
    DimQ,
    DimV,
    DeltaSize,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    contextual_seq_len,
    max_attn_len,
    full_attn_size,
    num_softmax_heads,
    # 18 constexpr args
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
):
    pass


# Same kernels without C cache — for baseline comparison
@_jit(c_cache=False)
def nop_kernel_nocache():
    pass


@_jit(c_cache=False)
def nop_with_args_kernel_nocache(
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


@_jit(c_cache=False)
def nop_hstu_args_kernel_nocache(
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_ts,
    stride_om,
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,
    DimQ,
    DimV,
    DeltaSize,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    contextual_seq_len,
    max_attn_len,
    full_attn_size,
    num_softmax_heads,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
):
    pass


@_jit(c_cache=True)
def nop_tensordesc_kernel(
    out_ptr, desc, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr
):
    block = desc.load([0, 0])
    idx = tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :]
    tl.store(out_ptr + idx, block)


@_jit(c_cache=False)
def nop_tensordesc_kernel_nocache(
    out_ptr, desc, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr
):
    block = desc.load([0, 0])
    idx = tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :]
    tl.store(out_ptr + idx, block)


def make_tensordesc_inputs(m_block: int = 8, n_block: int = 32):
    M = m_block * 3
    N = n_block * 4
    t = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    out = torch.zeros((m_block, n_block), device="cuda", dtype=torch.float16)
    desc = TensorDescriptor(
        t, shape=t.shape, strides=t.stride(), block_shape=[m_block, n_block]
    )
    return out, desc, M, N, m_block, n_block


# --- Autotuned kernel for benchmarking c_cache + autotune interaction ---
# Uses num_ctas to exercise cluster dispatch path 1 (num_ctas > 1 → clusterDim.x = num_ctas).


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_C1": 32,
                "BLOCK_C2": 32,
                "BLOCK_C3": 32,
                "BLOCK_C4": 32,
                "BLOCK_C5": 32,
            },
            num_ctas=2,
        ),
        triton.Config(
            {
                "BLOCK_C1": 64,
                "BLOCK_C2": 64,
                "BLOCK_C3": 64,
                "BLOCK_C4": 64,
                "BLOCK_C5": 64,
            },
            num_ctas=1,
        ),
    ],
    key=[],
)
@_jit(c_cache=True)
def nop_autotuned_kernel(
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
    BLOCK_C1: tl.constexpr,
    BLOCK_C2: tl.constexpr,
    BLOCK_C3: tl.constexpr,
    BLOCK_C4: tl.constexpr,
    BLOCK_C5: tl.constexpr,
):
    pass


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_C1": 32,
                "BLOCK_C2": 32,
                "BLOCK_C3": 32,
                "BLOCK_C4": 32,
                "BLOCK_C5": 32,
            },
            num_ctas=2,
        ),
        triton.Config(
            {
                "BLOCK_C1": 64,
                "BLOCK_C2": 64,
                "BLOCK_C3": 64,
                "BLOCK_C4": 64,
                "BLOCK_C5": 64,
            },
            num_ctas=1,
        ),
    ],
    key=[],
)
@_jit(c_cache=False)
def nop_autotuned_kernel_nocache(
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
    BLOCK_C1: tl.constexpr,
    BLOCK_C2: tl.constexpr,
    BLOCK_C3: tl.constexpr,
    BLOCK_C4: tl.constexpr,
    BLOCK_C5: tl.constexpr,
):
    pass


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_C1": 32,
                "BLOCK_C2": 32,
            }
        ),
        triton.Config(
            {
                "BLOCK_C1": 64,
                "BLOCK_C2": 64,
            }
        ),
    ],
    key=[],
)
@_jit(c_cache=True)
def nop_autotuned_with_none_kernel(
    q,
    k,
    v,
    out,
    bias,
    mask,
    scale,
    N,
    H,
    BLOCK_C1: tl.constexpr,
    BLOCK_C2: tl.constexpr,
):
    pass


@_jit(c_cache=True)
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
