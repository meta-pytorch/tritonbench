"""Triton bf16 x int4 GEMM closely matching the Helion reference kernel."""

import torch
import triton
import triton.language as tl

AUTOTUNE_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
        },
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
        },
        num_stages=2,
        num_warps=8,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
        },
        num_stages=1,
        num_warps=4,
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


def quantize_int4_weights(w, q_group_size):
    """Quantize weights into packed int4 values with per-group metadata."""

    packed, scales_and_zeros = _group_quantize_tensor(
        w.to(torch.bfloat16), n_bit=4, q_group_size=q_group_size
    )

    packed = packed.transpose(0, 1).contiguous().to(torch.int8)
    scales = scales_and_zeros[..., 0].contiguous()
    zeros = scales_and_zeros[..., 1].contiguous()

    return packed, scales, zeros


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr,
    b_packed_ptr,
    c_ptr,
    M,
    N,
    K,
    K_HALF,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Compute C = A x B with B stored as packed int4 values."""

    tl.static_assert(BLOCK_SIZE_K % 2 == 0)
    tl.device_assert(K % 2 == 0)

    pid = tl.program_id(axis=0)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid % num_blocks_m
    pid_n = pid // num_blocks_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K // 2)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_packed_ptr + (
        offs_k_packed[:, None] * stride_bk + offs_n[None, :] * stride_bn
    )
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for k_iter in range(0, num_k_tiles):
        k_offset = k_iter * BLOCK_SIZE_K

        a_mask = mask_m & (offs_k[None, :] + k_offset < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_mask = (offs_k_packed[:, None] + (k_offset // 2) < K_HALF) & mask_n
        b_tile_packed = tl.load(b_ptrs, mask=b_mask, other=0)

        _4_i8 = tl.full((1,), 4, dtype=tl.int8)
        b_lo = (b_tile_packed << _4_i8) >> _4_i8
        b_hi = b_tile_packed >> _4_i8

        b_unpacked = (
            tl.join(b_lo.to(tl.float32), b_hi.to(tl.float32))
            .permute(0, 2, 1)
            .reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)
        )

        a_expanded = tl.expand_dims(a_tile.to(tl.float32), 2)
        b_expanded = tl.expand_dims(b_unpacked, 0)
        accumulator += tl.sum(a_expanded * b_expanded, axis=1)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    c_tile = accumulator.to(tl.bfloat16)
    tl.store(c_ptrs, c_tile, mask=mask_m & mask_n)


def matmul(a, b_packed):
    assert (
        a.shape[1] == b_packed.shape[0] * 2
    ), f"Incompatible dimensions: {(a.shape[1], b_packed.shape[0] * 2)}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    if not b_packed.is_contiguous():
        b_packed = b_packed.contiguous()

    M, K = a.shape
    K_half, N = b_packed.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b_packed,
        c,
        M,
        N,
        K,
        K_half,
        a.stride(0),
        a.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def pack_2xint4(t):
    """Pack a KxN matrix of int8 into (K//2)xN with low/high nibble pairing."""

    t_int8 = t.to(torch.int8)
    t_view = t_int8.reshape(t_int8.shape[0] // 2, 2, t_int8.shape[1]).permute(1, 0, 2)
    return (t_view[0] & 0xF) | (t_view[1] << 4)


__all__ = [
    "_group_quantize_tensor",
    "quantize_int4_weights",
    "matmul_kernel",
    "matmul",
    "pack_2xint4",
]
