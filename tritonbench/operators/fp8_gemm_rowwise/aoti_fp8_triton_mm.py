# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import triton
import triton.language as tl

from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
    get_fp8_constants as get_fp8_constants,
)
from triton import Config

FP8_DTYPE, _, _, _ = get_fp8_constants()
E4M3_MAX_POS: float = torch.finfo(FP8_DTYPE).max
EPS: float = 1e-12
FP16_MAX_POS: float = torch.finfo(torch.float16).max


# options from:
# https://www.internalfb.com/code/fbsource/[ac575a080071]/xplat/caffe2/torch/_inductor/template_heuristics.py?lines=164
@triton.autotune(
    configs=[
        Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16},
            num_stages=1,
            num_warps=2,
        ),
        Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128},
            num_stages=2,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=5,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=5,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 128},
            num_stages=5,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16},
            num_stages=2,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_stages=5,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=4,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=4,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=5,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"]) == 0,
    }
)
# mm implementation from:
# https://www.internalfb.com/code/fbsource/[a2d7fb686a88]/xplat/caffe2/torch/_inductor/kernel/mm.py?lines=78
@triton.jit
def _aoti_fp8_triton_mm(
    a_ptr,
    a_scales,
    b_ptr,
    b_scales,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    USE_FAST_ACCUM,
    ACC_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (
        (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)
    ) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if (
        (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)
    ) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        a_offsets = idx_m * stride_am + idx_n * stride_ak
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        b_offsets = idx_m * stride_bk + idx_n * stride_bn
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)

        if USE_FAST_ACCUM:
            acc = tl.dot(a, b, acc, allow_tf32=True, out_dtype=ACC_TYPE)
        else:
            acc += tl.dot(a, b, allow_tf32=True, out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_scales = tl.load(a_scales + rm, mask=rm < M)
    b_scales = tl.load(b_scales + rn, mask=rn < N)
    acc_scale = a_scales[:, None] * b_scales[None, :]
    acc = acc * acc_scale

    if USE_BIAS:
        bias = tl.load(bias_ptr + rn, mask=rn < N)
        acc = acc + bias

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    c_mask = (idx_m < M) & (idx_n < N)
    c_offsets = idx_m * stride_cm + idx_n * stride_cn
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)


def aoti_fp8_triton_mm(
    A: torch.Tensor,
    A_scales: torch.Tensor,
    B: torch.Tensor,
    B_scales: torch.Tensor,
    bias,
    fp8_fast_accum: bool = False,
) -> torch.Tensor:
    M, K = A.shape
    N, _ = B.shape

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    def grid(meta):  # non persistent
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _aoti_fp8_triton_mm[grid](
        a_ptr=A,
        a_scales=A_scales,
        b_ptr=B,
        b_scales=B_scales,
        c_ptr=C,
        bias_ptr=bias,
        M=M,
        N=N,
        K=K,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_bn=B.stride(0),
        stride_bk=B.stride(1),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        USE_FAST_ACCUM=True,
        ACC_TYPE=tl.float32,
        USE_BIAS=bias is not None,
        GROUP_M=1,
    )

    return C
