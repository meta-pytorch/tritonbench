# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TritonBench Conv1D Operator

Benchmarks Conv1D implementations:
- PyTorch reference (torch.nn.functional.conv1d)
- PyTorch Conv2D reference (unsqueeze + conv2d + squeeze)
- Triton kernel implementations
"""

import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
import torch.nn
import triton
import triton.language as tl

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)


def parse_op_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (if not specified, uses default shapes)",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=96,
        help="Number of input channels",
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=96,
        help="Number of output channels",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=200,
        help="Sequence length",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Convolution kernel size",
    )
    return parser.parse_args(args)


# ============================================================================
# Triton Kernel Implementation
# ============================================================================

# Autotune configurations
autotune_configs = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=8,
        num_stages=4,
    ),
]


@triton.jit
def pack_conv1d_weight_kernel(
    w_in_ptr,
    w_out_ptr,
    Cout,
    Cin_g,
    Ksz,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Pack weight from [Cout, Cin_g, Ksz] to [Cin_g*Ksz, Cout]."""
    R = Cin_g * Ksz
    C = Cout
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_r = offs_r < R
    mask_c = offs_c < C

    ldA = R
    a_ptrs = w_in_ptr + offs_c[:, None] * ldA + offs_r[None, :]
    a_mask = mask_c[:, None] & mask_r[None, :]
    a = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=".cg")

    ldB = C
    b_ptrs = w_out_ptr + offs_r[None, :] * ldB + offs_c[:, None]
    tl.store(b_ptrs, a, mask=a_mask)


@triton.autotune(
    configs=autotune_configs,
    key=["M", "N", "K", "stride", "dilation", "kernel_size", "Cin_g"],
)
@triton.jit
def conv1d_gemm_kernel(
    input_ptr,
    weight_packed_ptr,
    bias_ptr,
    output_ptr,
    B: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    Lin: tl.constexpr,
    Lout: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    Cin_g: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Conv1D GEMM-style kernel with packed weights."""
    tl.static_assert(BLOCK_K % 16 == 0)

    pid_m = tl.program_id(0)
    gid = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    batch_idx = offs_m // Lout
    out_pos = offs_m % Lout
    pos_base = out_pos * stride - padding

    in_ch_start = gid * Cin_g
    out_ch_start = gid * N

    batch_in_stride = Cin * Lin
    batch_out_stride = Cout * Lout

    batch_in_offs = batch_idx * batch_in_stride
    input_batch_group_base = input_ptr + batch_in_offs[:, None] + (in_ch_start * Lin)
    out_ch_idx = out_ch_start + offs_n

    w_col_base = weight_packed_ptr + out_ch_idx[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m_b = mask_m[:, None]
    mask_n_b = mask_n[None, :]

    for t in tl.static_range(0, kernel_size):
        tap_incr = t * dilation
        input_pos = pos_base[:, None] + tap_incr
        row_valid_mask = (input_pos >= 0) & (input_pos < Lin)
        row_valid_mask = row_valid_mask & mask_m_b

        for cstart in tl.static_range(0, Cin_g, BLOCK_K):
            cin_off = cstart + tl.arange(0, BLOCK_K)
            k_mask = cin_off < Cin_g

            cin_incr = cin_off * Lin
            in_ptrs = input_batch_group_base + cin_incr[None, :] + input_pos
            in_mask = row_valid_mask & k_mask[None, :]
            x = tl.load(in_ptrs, mask=in_mask, other=0.0, cache_modifier=".ca")

            k_linear = cin_off * kernel_size + t
            w_ptrs = w_col_base + k_linear[:, None] * Cout
            w_mask = k_mask[:, None] & mask_n_b
            w = tl.load(w_ptrs, mask=w_mask, other=0.0, cache_modifier=".cg")

            acc = tl.dot(x, w, acc)

    if has_bias:
        b = tl.load(bias_ptr + out_ch_idx, mask=mask_n, other=0.0).to(tl.float32)
        acc += b[None, :]

    batch_out_offs = batch_idx * batch_out_stride
    out_ptrs = (
        output_ptr
        + batch_out_offs[:, None]
        + out_ch_idx[None, :] * Lout
        + out_pos[:, None]
    )
    out_mask = mask_m_b & mask_n_b
    tl.store(out_ptrs, acc, mask=out_mask)


# Weight packing cache
_WEIGHT_PACK_CACHE = {}
_DISABLE_WEIGHT_CACHE = True


def _maybe_pack_weight(weight_tensor, in_channels_per_group, kernel_sz):
    """Pack weight from [Cout, Cin_g, Ksz] to [Cin_g*Ksz, Cout]."""
    Cout, Cin_g, Ksz = weight_tensor.shape
    assert Cin_g == in_channels_per_group and Ksz == kernel_sz

    if not _DISABLE_WEIGHT_CACHE:
        cache_key = (
            int(weight_tensor.data_ptr()),
            tuple(weight_tensor.shape),
            weight_tensor.device,
            weight_tensor.dtype,
        )
        packed = _WEIGHT_PACK_CACHE.get(cache_key, None)
        if packed is not None and packed.is_cuda and packed.dtype == weight_tensor.dtype:
            return packed

    packed = torch.empty(
        (Cin_g * Ksz, Cout), device=weight_tensor.device, dtype=weight_tensor.dtype
    )

    BR, BC = 128, 128

    def grid(meta):
        R = Cin_g * Ksz
        C = Cout
        gr = (R + BR - 1) // BR
        gc = (C + BC - 1) // BC
        return (gr, gc)

    pack_conv1d_weight_kernel[grid](
        weight_tensor,
        packed,
        Cout,
        Cin_g,
        Ksz,
        BLOCK_R=BR,
        BLOCK_C=BC,
    )

    if not _DISABLE_WEIGHT_CACHE:
        _WEIGHT_PACK_CACHE[cache_key] = packed
    return packed


def _triton_conv1d_impl(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Triton 1D convolution using GEMM tiling with packed weights."""
    assert input_tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert weight_tensor.dtype == input_tensor.dtype
    if bias_tensor is not None:
        assert bias_tensor.dtype == input_tensor.dtype

    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, in_channels_per_group, kernel_sz = weight_tensor.shape

    assert in_channels_per_group * groups == in_channels
    assert out_channels % groups == 0
    out_channels_per_group = out_channels // groups

    output_length = (
        (input_length + 2 * padding - dilation * (kernel_sz - 1) - 1) // stride
    ) + 1

    output = torch.empty(
        (batch_size, out_channels, output_length),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    has_bias = bias_tensor is not None
    if not has_bias:
        bias_tensor = torch.zeros(
            out_channels, device=input_tensor.device, dtype=input_tensor.dtype
        )

    weight_packed = _maybe_pack_weight(weight_tensor, in_channels_per_group, kernel_sz)

    M = batch_size * output_length
    N = out_channels_per_group
    K = in_channels_per_group * kernel_sz

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        gm = (M + BM - 1) // BM
        gn = (N + BN - 1) // BN
        return (gm, groups, gn)

    conv1d_gemm_kernel[grid](
        input_tensor,
        weight_packed,
        bias_tensor,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        groups,
        M,
        N,
        K,
        in_channels_per_group,
        kernel_sz,
        stride,
        padding,
        dilation,
        has_bias,
    )

    return output


# ============================================================================
# TritonBench Operator
# ============================================================================


class Operator(BenchmarkOperator):
    """TritonBench operator for Conv1D benchmarks."""
    
    DEFAULT_METRICS = ["latency", "speedup", "accuracy"]
    DEFAULT_PRECISION = "fp16"

    def __init__(
        self,
        tb_args: argparse.Namespace,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        super().__init__(tb_args, extra_args)
        self.args = parse_op_args(self.extra_args)

        # Conv parameters
        self.in_channels = self.args.in_channels
        self.out_channels = self.args.out_channels
        self.seq_length = self.args.seq_length
        self.kernel_size = self.args.kernel_size

    @register_x_val(label="(B, C, L)")
    def get_x_val(self, example_inputs: Tuple[torch.Tensor, ...]) -> str:
        """Extract x-axis value for plotting."""
        if not example_inputs or len(example_inputs) == 0:
            return "0"

        input_tensor = example_inputs[0]
        if isinstance(input_tensor, torch.Tensor):
            shape = input_tensor.shape
            if len(shape) >= 3:
                return f"{shape[0]}x{shape[1]}x{shape[2]}"
        return "Unknown"

    def get_input_iter(self) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """Generate test inputs for benchmarking."""
        torch.manual_seed(42)

        # Production data shape parameters
        max_batch_size = 2048
        in_channels = 96
        out_channels = 96
        seq_length = 200
        kernel_size = 3

        input_full = (
            torch.randn(max_batch_size, in_channels, seq_length, device=self.device, dtype=self.dtype)
            * 0.041105
            + 0.006413
        )

        weight_tensor = (
            torch.randn(out_channels, in_channels, kernel_size, device=self.device, dtype=self.dtype)
            * 0.047870
            - 0.000024
        )

        bias_tensor = (
            torch.randn(out_channels, device=self.device, dtype=self.dtype)
            * 0.033662
            - 0.003198
        )

        batch_sizes = [64, 128, 256, 512, 1024, 2048]
        for batch_size in batch_sizes:
            input_tensor = input_full[:batch_size, :, :].contiguous()
            yield (input_tensor, weight_tensor, bias_tensor)


        

    @register_benchmark(baseline=True)
    def pytorch_conv1d(
        self,
        input_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        bias_tensor: torch.Tensor,
    ) -> Callable[[], torch.Tensor]:
        """PyTorch Conv1D reference implementation (baseline)."""

        @torch.compile(mode="max-autotune-no-cudagraphs")
        def _impl():
            return torch.nn.functional.conv1d(
                input_tensor,
                weight_tensor,
                bias_tensor,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
            )

        return _impl

    @register_benchmark()
    def pytorch_conv2d(
        self,
        input_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        bias_tensor: torch.Tensor,
    ) -> Callable[[], torch.Tensor]:
        """PyTorch Conv2D reference (unsqueeze + conv2d + squeeze)."""

        @torch.compile(mode="max-autotune-no-cudagraphs")
        def _impl():
            # Unsqueeze to add height dimension
            input_4d = input_tensor.unsqueeze(2).to(memory_format=torch.channels_last)
            weight_4d = weight_tensor.unsqueeze(2)
            out_4d = torch.nn.functional.conv2d(
                input_4d,
                weight_4d,
                bias_tensor,
                stride=(1, 1),
                padding=(0, 1),
                dilation=1,
            )
            return out_4d.squeeze(2)

        return _impl

    @register_benchmark()
    def triton_conv1d(
        self,
        input_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        bias_tensor: torch.Tensor,
    ) -> Callable[[], torch.Tensor]:
        """Triton Conv1D kernel implementation."""

        def _impl():
            return _triton_conv1d_impl(
                input_tensor,
                weight_tensor,
                bias_tensor,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
            )

        return _impl
