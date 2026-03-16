"""
Triton implementation by @jlebar: https://gist.github.com/jlebar/3435b2c00deea53258887ce37231e5e2
"""

import torch
import triton
import triton.language as tl

AUTOTUNE_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 32,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
        },
        num_stages=4,
        num_warps=8,
    ),
]


def _group_quantize_tensor(w, n_bit=4, q_group_size=16):
    assert w.dim() == 2
    w = w.transpose(0, 1).contiguous()
    assert q_group_size > 1
    assert w.shape[-1] % q_group_size == 0

    to_quant = w.reshape(-1, q_group_size)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    assert torch.isnan(scales).sum() == 0

    zeros = min_val + scales * (2 ** (n_bit - 1))
    assert torch.isnan(zeros).sum() == 0

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)
    assert torch.isnan(out).sum() == 0

    out = out.to(dtype=torch.int32).reshape(w.shape)
    out_uint8 = (out[::, ::2] << 4 | out[::, 1::2]).to(torch.uint8)

    # Scales and zeros for the same q-group should be contiguous, so we can
    # load as a 32-bit word
    scales = scales.view(w.shape[0], -1)
    zeros = zeros.view(w.shape[0], -1)
    scales_and_zeros = (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

    return out_uint8, scales_and_zeros


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions.
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    #
    # We assume `b` is packed with 2 `int4` elements per K, i.e. it's a
    # (K//2)xNx(2xint4) matrix, represented in Triton as (K//2)xNxi8.  If K
    # is the minor dimension, then stride_bk should logically be 0.5.  But
    # we don't want a fractional stride!  So let the given stride be the
    # stride per 2xint4.
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    tl.device_assert(K % BLOCK_SIZE_K == 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate.
    #
    # To avoid a layout-incompatible join+permute+reshape when unpacking
    # int4 weights, we load A's even and odd K-columns separately and
    # compute two half-K dot products:
    #   C += A_even @ B_lo + A_odd @ B_hi
    # This is equivalent to C += A @ unpack(B) but never materializes
    # the full (K, N) unpacked tensor.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_half_k = tl.arange(0, BLOCK_SIZE_K // 2)
    # a_even_ptrs loads A[:, 0], A[:, 2], A[:, 4], ... (even K-columns)
    # a_odd_ptrs  loads A[:, 1], A[:, 3], A[:, 5], ... (odd  K-columns)
    a_even_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + (2 * offs_half_k)[None, :] * stride_ak
    )
    a_odd_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + (2 * offs_half_k + 1)[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (offs_half_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        remaining = K - k * BLOCK_SIZE_K
        a_even = tl.load(
            a_even_ptrs, mask=(2 * offs_half_k)[None, :] < remaining, other=0.0
        )
        a_odd = tl.load(
            a_odd_ptrs, mask=(2 * offs_half_k + 1)[None, :] < remaining, other=0.0
        )
        b = tl.load(b_ptrs)
        tl.static_assert(b.dtype == tl.int8)

        # Unpack `b` into bf16 low/high nibbles with sign extension.
        _4_i8 = tl.full((1,), 4, dtype=tl.int8)
        b_lo = ((b << _4_i8) >> _4_i8).to(tl.bfloat16)
        b_hi = (b >> _4_i8).to(tl.bfloat16)

        # Two half-K dots instead of one full-K dot.
        # Equivalent to: accumulator += dot(a, join(b_lo, b_hi).permute().reshape())
        accumulator += tl.dot(a_even, b_lo) + tl.dot(a_odd, b_hi)
        a_even_ptrs += BLOCK_SIZE_K * stride_ak
        a_odd_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk // 2

    c = accumulator.to(tl.bfloat16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0] * 2, (
        f"Incompatible dimensions: {a.shape[1], b.shape[0] * 2}"
    )
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def pack_2xint4(t):
    # Packs a KxNxfp16 matrix into a (K//2)xNx(2xint4) matrix.
    t = t.to(torch.int8).reshape(t.shape[0] // 2, 2, t.shape[1]).permute(1, 0, 2)
    return (t[0] & 0xF) | (t[1] << 4)
