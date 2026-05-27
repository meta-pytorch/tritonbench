# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

#!/usr/bin/env python3

from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from generative_recommenders.ops.utils import maybe_register_custom_op
from hammer.v3.ops.triton.vararg_kernel import unroll_varargs, VarargMode


try:
    # @manual=//triton:triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    TMA_AVAILABLE = True
except ImportError:
    TMA_AVAILABLE = False

try:
    from gwatch.cuda.trace.triton import (  # @manual  # pyre-ignore[21]
        scope_end as _gw_scope_end,
        scope_start as _gw_scope_start,
    )
except ImportError:

    @triton.jit
    def _gw_scope_start(tag_id: tl.constexpr):
        pass

    @triton.jit
    def _gw_scope_end(tag_id: tl.constexpr):
        pass


try:
    import importlib.util
    import os

    # The tutorials directory in triton.language.extra.tlx lacks __init__.py,
    # so we load hopper_gemm_ws.py via importlib from the file path directly.
    # Requires beta triton (build with -c triton.version=beta).
    import triton.language.extra.tlx as tlx

    _hopper_gemm_ws_path = os.path.join(
        os.path.dirname(tlx.__file__), "tutorials", "hopper_gemm_ws.py"
    )
    if os.path.exists(_hopper_gemm_ws_path):
        _spec = importlib.util.spec_from_file_location(
            "hopper_gemm_ws", _hopper_gemm_ws_path
        )
        if _spec is not None and _spec.loader is not None:
            _mod = importlib.util.module_from_spec(_spec)
            _loader = _spec.loader
            assert _loader is not None
            _loader.exec_module(_mod)
            _tlx_matmul = _mod.matmul
            TLX_AVAILABLE = True
        else:
            TLX_AVAILABLE = False
    else:
        TLX_AVAILABLE = False
except Exception:
    TLX_AVAILABLE = False

VAR_ARGS_ARRAY_A = List[Any]
VAR_ARGS_ARRAY_B = List[Any]
VAR_ARGS_ARRAY_C = List[Any]
VAR_ARGS_ARRAY_BIAS = List[Any]
VAR_ARGS_ARRAY_LENGTHS_M = List[Any]
VAR_ARGS_ARRAY_LENGTHS_N = List[Any]
VAR_ARGS_ARRAY_STRIDES_BK = List[Any]
VAR_ARGS_ARRAY_STRIDES_CM = List[Any]
VAR_ARGS_ARRAY_STRIDES_BIAS = List[Any]
VAR_ARGS_ARRAY_LENGTHS_K = List[Any]
VAR_ARGS_ARRAY_STRIDES_AM = List[Any]


def get_autotune_configs():
    """Generate autotuning configurations for different block sizes."""
    configs = [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ]
    return configs


def get_ws_autotune_configs():
    """Autotune configs for WS blocked GEMM."""
    return [
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": BN,
                "BLOCK_K": 64,
                "GROUP_M": g,
                "NUM_STAGES": s,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": epilogue,
            },
            num_stages=1,
            num_warps=4,
        )
        for BN in [128, 256]
        for s in [3, 4]
        for epilogue in [True, False]
        for g in [1, 8, 64]
    ]


def _ws_preprocess_configs(configs, named_args, **kwargs):
    """Prune WS configs based on M/N ratio for L2 cache locality."""
    M_max = named_args["M_max"]
    N_max = named_args["N_max"]
    K_max = named_args["K_max"]
    # Skip BLOCK_K configs larger than K_max (TMA descriptor requires K >= BLOCK_K)
    configs = [c for c in configs if c.kwargs["BLOCK_K"] <= K_max]
    IMBALANCE_THRESHOLD = 10
    if M_max > N_max * IMBALANCE_THRESHOLD:
        configs = [c for c in configs if c.kwargs["GROUP_M"] == 1]
    elif N_max > M_max * IMBALANCE_THRESHOLD:
        configs = [c for c in configs if c.kwargs["GROUP_M"] >= 32]
    else:
        configs = [c for c in configs if c.kwargs["GROUP_M"] == 8]
    return configs


@lru_cache(maxsize=None)
def _get_autotune_kernel_blocked_gemm(kernel: Callable) -> Callable:
    return triton.autotune(
        configs=get_autotune_configs(),
        key=["M_max", "N_max", "K_max"],
    )(kernel)


@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    GEMM kernel with optional bias: C = A @ B + bias.
    Uses grouped tiling (swizzle) for improved L2 cache reuse.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N

    A_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_am
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)

    B_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_bn
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        b = tl.load(b_ptrs, mask=mask_k & mask_n, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias_ptrs = bias_ptr + (pid_n * BLOCK_N + offs_n)[None, :] * stride_bias
        bias = tl.load(bias_ptrs, mask=mask_n)
        acc += bias.to(tl.float32)

    c_mask = mask_m & mask_n
    C_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_cm
    C_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_cn
    c_ptrs = C_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """
    Compute GEMM with optional bias: C = A @ B + bias using Triton.
    Wrapper that sets up the grid and launches the autotuned kernel.
    """
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    has_bias = bias is not None

    grid = lambda meta: (  # noqa E731
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    _gemm_kernel[grid](
        A,
        B,
        C,
        bias if has_bias else A,  # dummy pointer when no bias
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        bias.stride(0) if has_bias and bias is not None else 1,
        HAS_BIAS=has_bias,
    )

    return C


@triton.jit
def _blocked_gemm_kernel_ab_varargs(
    A_ptrs: "VAR_ARGS_ARRAY_A",
    B_ptrs: "VAR_ARGS_ARRAY_B",
    C_ptrs: "VAR_ARGS_ARRAY_C",
    BIAS_ptrs: "VAR_ARGS_ARRAY_BIAS",
    lengths_m: "VAR_ARGS_ARRAY_LENGTHS_M",
    lengths_n: "VAR_ARGS_ARRAY_LENGTHS_N",
    lengths_k: "VAR_ARGS_ARRAY_LENGTHS_K",
    strides_am: "VAR_ARGS_ARRAY_STRIDES_AM",
    strides_bk: "VAR_ARGS_ARRAY_STRIDES_BK",
    strides_cm: "VAR_ARGS_ARRAY_STRIDES_CM",
    strides_bias: "VAR_ARGS_ARRAY_STRIDES_BIAS",
    M_max,
    N_max,
    K_max,
    stride_ak,
    stride_bn,
    stride_cn,
    NUM_A: tl.constexpr,
    NUM_B: tl.constexpr,
    NUM_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Persistent blocked GEMM kernel with K blocking:
        C_list[i][j] = sum_k A_list[i][k] @ B_list[j][k] + bias[j]
    A_ptrs are flattened as A_ptrs[i * NUM_K + k].
    B_ptrs are flattened as B_ptrs[k * NUM_B + j].
    All K blocks are processed sequentially within each (i, j) output tile.
    TMA for all loads/stores.
    """
    sm_id = tl.program_id(0)

    num_pid_m_max = tl.cdiv(M_max, BLOCK_M)
    num_pid_n_max = tl.cdiv(N_max, BLOCK_N)
    tiles_per_pair = num_pid_m_max * num_pid_n_max
    num_tiles = NUM_A * NUM_B * tiles_per_pair

    tile_id = sm_id
    while tile_id < num_tiles:
        pid_ab = tile_id // tiles_per_pair
        pid_mn = tile_id % tiles_per_pair

        pid_a = pid_ab // NUM_B
        pid_b = pid_ab % NUM_B

        M_i = lengths_m[pid_a]
        N_j = lengths_n[pid_b]

        num_pid_m = tl.cdiv(M_i, BLOCK_M)
        num_pid_n = tl.cdiv(N_j, BLOCK_N)

        # Swizzle for L2 cache reuse
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid_mn // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        group_size_m = max(group_size_m, 1)
        pid_m = first_pid_m + (pid_mn % group_size_m)
        pid_n = (pid_mn % num_pid_in_group) // group_size_m

        if pid_m < num_pid_m and pid_n < num_pid_n:
            m_off = (pid_m * BLOCK_M).to(tl.int32)
            n_off = (pid_n * BLOCK_N).to(tl.int32)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_idx in range(NUM_K):
                K_k = lengths_k[k_idx]
                a_flat_idx = pid_a * NUM_K + k_idx
                b_flat_idx = k_idx * NUM_B + pid_b

                desc_a = tl.make_tensor_descriptor(
                    A_ptrs[a_flat_idx],
                    shape=[M_i, K_k],
                    strides=[strides_am[k_idx], stride_ak],
                    block_shape=[BLOCK_M, BLOCK_K],
                )
                desc_b = tl.make_tensor_descriptor(
                    B_ptrs[b_flat_idx],
                    shape=[K_k, N_j],
                    strides=[strides_bk[pid_b], stride_bn],
                    block_shape=[BLOCK_K, BLOCK_N],
                )

                for _k in range(0, tl.cdiv(K_k, BLOCK_K)):
                    k_off = (_k * BLOCK_K).to(tl.int32)
                    a = desc_a.load([m_off, k_off])
                    b = desc_b.load([k_off, n_off])
                    acc += tl.dot(a, b)

            if HAS_BIAS:
                offs_n = tl.arange(0, BLOCK_N)
                mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N_j
                stride_bias_j = strides_bias[pid_b]
                bias_ptrs = (
                    BIAS_ptrs[pid_b]
                    + (pid_n * BLOCK_N + offs_n[None, :]) * stride_bias_j
                )
                bias = tl.load(bias_ptrs, mask=mask_n)
                acc += bias.to(tl.float32)

            c_idx = pid_a * NUM_B + pid_b
            desc_c = tl.make_tensor_descriptor(
                C_ptrs[c_idx],
                shape=[M_i, N_j],
                strides=[strides_cm[pid_b], stride_cn],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            desc_c.store([m_off, n_off], acc.to(C_ptrs[c_idx].dtype.element_ty))

        tile_id += NUM_SMS


###############################################################################
# Blocked GEMM backward kernel (no TMA -- uses tl.load/tl.store)
###############################################################################


def get_bwd_autotune_configs():
    """Autotune configs optimized for backward pass on H100."""
    configs = [
        # BLOCK_K=64 configs
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        # num_stages=4 with 8 warps for deeper pipelining
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        # Small-block configs for cases with few output tiles (e.g., dW backward
        # where M_max, N_max are small but K is very large). More tiles = better
        # SM utilization on H100 (132 SMs).
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        # BLOCK_K=128 configs: fewer K-loop iterations for K-heavy workloads
        # (e.g., dW where K=65536 and M,N~1024).
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=2,
            num_warps=8,
        ),
        # 32-wide configs for maximum tile count (1024/32=32 tiles per dim)
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ]
    return configs


@triton.jit
def _blocked_gemm_bwd_kernel(
    A_ptrs: "VAR_ARGS_ARRAY_A",
    B_ptrs: "VAR_ARGS_ARRAY_B",
    C_ptrs: "VAR_ARGS_ARRAY_C",
    BIAS_ptrs: "VAR_ARGS_ARRAY_BIAS",
    lengths_m: "VAR_ARGS_ARRAY_LENGTHS_M",
    lengths_n: "VAR_ARGS_ARRAY_LENGTHS_N",
    lengths_k: "VAR_ARGS_ARRAY_LENGTHS_K",
    strides_am: "VAR_ARGS_ARRAY_STRIDES_AM",
    strides_bk: "VAR_ARGS_ARRAY_STRIDES_BK",
    strides_cm: "VAR_ARGS_ARRAY_STRIDES_CM",
    strides_bias: "VAR_ARGS_ARRAY_STRIDES_BIAS",
    M_max,
    N_max,
    K_max,
    stride_ak,
    stride_bn,
    stride_cn,
    NUM_A: tl.constexpr,
    NUM_B: tl.constexpr,
    NUM_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # pyre-ignore[9]: defaults are constexpr bools at Triton compile time
    EVEN_K: tl.constexpr = False,
    # pyre-ignore[9]
    EVEN_M: tl.constexpr = False,
    # pyre-ignore[9]
    EVEN_N: tl.constexpr = False,
):
    """
    Persistent blocked GEMM kernel for backward pass.
    Same computation as _blocked_gemm_kernel_ab_varargs but uses
    tl.load/tl.store with pointer arithmetic instead of TMA descriptors
    to avoid Triton compiler limitations with conditional vararg expansion.

    EVEN_K: when True, all K_k are divisible by BLOCK_K, so K-dim masking
    is skipped for better performance.
    EVEN_M: when True, all M_i are divisible by BLOCK_M, skip M masking.
    EVEN_N: when True, all N_j are divisible by BLOCK_N, skip N masking.
    """
    sm_id = tl.program_id(0)

    num_pid_m_max = tl.cdiv(M_max, BLOCK_M)
    num_pid_n_max = tl.cdiv(N_max, BLOCK_N)
    tiles_per_pair = num_pid_m_max * num_pid_n_max
    num_tiles = NUM_A * NUM_B * tiles_per_pair

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tile_id = sm_id
    while tile_id < num_tiles:
        pid_ab = tile_id // tiles_per_pair
        pid_mn = tile_id % tiles_per_pair

        pid_a = pid_ab // NUM_B
        pid_b = pid_ab % NUM_B

        M_i = lengths_m[pid_a]
        N_j = lengths_n[pid_b]

        num_pid_m = tl.cdiv(M_i, BLOCK_M)
        num_pid_n = tl.cdiv(N_j, BLOCK_N)

        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid_mn // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        group_size_m = max(group_size_m, 1)
        pid_m = first_pid_m + (pid_mn % group_size_m)
        pid_n = (pid_mn % num_pid_in_group) // group_size_m

        if pid_m < num_pid_m and pid_n < num_pid_n:
            m_off = pid_m * BLOCK_M
            n_off = pid_n * BLOCK_N

            # Pre-compute M and N offset arrays (hoisted out of K loop)
            m_idx = (m_off + offs_m[:, None]).to(tl.int64)
            n_idx = (n_off + offs_n[None, :]).to(tl.int64)

            # Build masks only when needed (EVEN_M/EVEN_N skip masking)
            if not EVEN_M:
                mask_m = (m_off + offs_m[:, None]) < M_i
            if not EVEN_N:
                mask_n = (n_off + offs_n[None, :]) < N_j

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_idx in range(NUM_K):
                K_k = lengths_k[k_idx]
                a_flat_idx = pid_a * NUM_K + k_idx
                b_flat_idx = k_idx * NUM_B + pid_b

                a_base = A_ptrs[a_flat_idx]
                b_base = B_ptrs[b_flat_idx]
                _stride_am = strides_am[k_idx]
                _stride_bk = strides_bk[pid_b]

                # Pre-compute row/col base pointers for this (a,b) pair
                a_row_ptrs = a_base + m_idx * _stride_am
                b_col_ptrs = b_base + n_idx * stride_bn

                for _k in range(0, tl.cdiv(K_k, BLOCK_K)):
                    k_off = _k * BLOCK_K
                    k_idx_a = (k_off + offs_k[None, :]).to(tl.int64)
                    k_idx_b = (k_off + offs_k[:, None]).to(tl.int64)

                    a_ptrs_block = a_row_ptrs + k_idx_a * stride_ak
                    b_ptrs_block = b_col_ptrs + k_idx_b * _stride_bk

                    if EVEN_K and EVEN_M and EVEN_N:
                        # No masking needed at all
                        a = tl.load(a_ptrs_block)
                        b = tl.load(b_ptrs_block)
                    elif EVEN_K:
                        # Only M/N masking
                        if EVEN_M:
                            a = tl.load(a_ptrs_block)
                        else:
                            a = tl.load(
                                a_ptrs_block,
                                mask=mask_m,  # pyre-ignore[61]
                                other=0.0,
                            )
                        if EVEN_N:
                            b = tl.load(b_ptrs_block)
                        else:
                            b = tl.load(
                                b_ptrs_block,
                                mask=mask_n,  # pyre-ignore[61]
                                other=0.0,
                            )
                    else:
                        if EVEN_M:
                            mask_ak = k_idx_a < K_k
                        else:
                            mask_ak = mask_m & (k_idx_a < K_k)  # pyre-ignore[61]
                        a = tl.load(a_ptrs_block, mask=mask_ak, other=0.0)
                        if EVEN_N:
                            mask_bk = k_idx_b < K_k
                        else:
                            mask_bk = (k_idx_b < K_k) & mask_n  # pyre-ignore[61]
                        b = tl.load(b_ptrs_block, mask=mask_bk, other=0.0)

                    acc += tl.dot(a, b)

            if HAS_BIAS:
                if EVEN_N:
                    stride_bias_j = strides_bias[pid_b]
                    bias_ptrs = (
                        BIAS_ptrs[pid_b]
                        + (n_off + offs_n[None, :]).to(tl.int64) * stride_bias_j
                    )
                    bias = tl.load(bias_ptrs)
                else:
                    bias_mask_n = (n_off + offs_n)[None, :] < N_j
                    stride_bias_j = strides_bias[pid_b]
                    bias_ptrs = (
                        BIAS_ptrs[pid_b]
                        + (n_off + offs_n[None, :]).to(tl.int64) * stride_bias_j
                    )
                    bias = tl.load(bias_ptrs, mask=bias_mask_n)
                acc += bias.to(tl.float32)

            c_idx = pid_a * NUM_B + pid_b
            c_base = C_ptrs[c_idx]
            _stride_cm = strides_cm[pid_b]
            c_ptrs_block = c_base + m_idx * _stride_cm + n_idx * stride_cn
            if EVEN_M and EVEN_N:
                tl.store(
                    c_ptrs_block,
                    acc.to(C_ptrs[c_idx].dtype.element_ty),
                )
            else:
                if EVEN_M:
                    c_mask = mask_n  # pyre-ignore[61]
                elif EVEN_N:
                    c_mask = mask_m  # pyre-ignore[61]
                else:
                    c_mask = mask_m & mask_n  # pyre-ignore[61]
                tl.store(
                    c_ptrs_block,
                    acc.to(C_ptrs[c_idx].dtype.element_ty),
                    mask=c_mask,
                )

        tile_id += NUM_SMS


@lru_cache(maxsize=None)
def _get_autotune_kernel_blocked_gemm_bwd(kernel: Callable) -> Callable:
    return triton.autotune(
        configs=get_bwd_autotune_configs(),
        key=["M_max", "N_max", "K_max"],
        use_cuda_graph=True,
    )(kernel)


###############################################################################
# Warp-Specialized GEMM with TLX on H100
###############################################################################


def _tlx_alloc_fn(size: int, align: int, stream: "int | None"):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def _ensure_triton_allocator() -> None:
    triton.set_allocator(_tlx_alloc_fn)  # pyre-ignore[6]


def _tlx_set_block_size_hook(nargs):
    BM = nargs["BM"]
    BN = nargs["BN"]
    BK = nargs["BK"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BM // NUM_MMA_GROUPS
    nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BK]
    nargs["b_desc"].block_shape = [BK, BN]
    if nargs.get("EPILOGUE_SUBTILE", False):
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BN // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BN]
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    nargs["NUM_SMS"] = NUM_SMS


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


###############################################################################
# Blocked GEMM: Warp-Specialized kernel (version="ws")
###############################################################################


@lru_cache(maxsize=None)
def _get_autotune_kernel_blocked_gemm_ws(kernel: Callable) -> Callable:
    return triton.autotune(
        configs=get_ws_autotune_configs(),
        key=["M_max", "N_max", "K_max"],
        use_cuda_graph=True,
        prune_configs_by={"early_config_prune": _ws_preprocess_configs},
    )(kernel)


@triton.jit
def _blocked_gemm_kernel_ws(
    A_ptrs: "VAR_ARGS_ARRAY_A",
    B_ptrs: "VAR_ARGS_ARRAY_B",
    C_ptrs: "VAR_ARGS_ARRAY_C",
    BIAS_ptrs: "VAR_ARGS_ARRAY_BIAS",
    lengths_m: "VAR_ARGS_ARRAY_LENGTHS_M",
    lengths_n: "VAR_ARGS_ARRAY_LENGTHS_N",
    lengths_k: "VAR_ARGS_ARRAY_LENGTHS_K",
    strides_am: "VAR_ARGS_ARRAY_STRIDES_AM",
    strides_bk: "VAR_ARGS_ARRAY_STRIDES_BK",
    strides_cm: "VAR_ARGS_ARRAY_STRIDES_CM",
    strides_bias: "VAR_ARGS_ARRAY_STRIDES_BIAS",
    M_max,
    N_max,
    K_max,
    stride_ak,
    stride_bn,
    stride_cn,
    NUM_A: tl.constexpr,
    NUM_B: tl.constexpr,
    NUM_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    INPUT_DTYPE: tl.constexpr = tl.bfloat16,  # pyre-ignore[9]
    # pyre-ignore[9]
    EPILOGUE_SUBTILE: tl.constexpr = False,
    # When True, allocate the MMA accumulator in TMEM and use Blackwell
    # tcgen05.mma semantics (use_acc=False on the first k-iter, tcgen05_commit
    # fence + local_load from TMEM at the end of each tile). When False, use
    # the Hopper register-accumulator + tlx.async_dot_wait pattern.
    IS_BLACKWELL: tl.constexpr = False,
):
    """WS persistent blocked GEMM with K blocking
    C_list[i][j] = sum_k A_list[i][k] @ B_list[j][k] + bias[j]"""
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    a_smem = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_K),
        INPUT_DTYPE,  # pyre-ignore[6]
        NUM_STAGES * NUM_MMA_GROUPS,
    )
    b_smem = tlx.local_alloc(
        (BLOCK_K, BLOCK_N),
        INPUT_DTYPE,  # pyre-ignore[6]
        NUM_STAGES,
    )

    bars_empty_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=tl.constexpr(1)
    )
    bars_full_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=tl.constexpr(1)
    )
    bars_empty_b = tlx.alloc_barriers(
        num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS
    )
    bars_full_b = tlx.alloc_barriers(
        num_barriers=NUM_STAGES, arrive_count=tl.constexpr(1)
    )

    # Blackwell: MMA accumulator lives in TMEM. tcgen05.mma writes to TMEM
    # asynchronously; we use one TMEM tile per MMA replica plus a dedicated
    # "MMA-done" mbarrier per replica that we arrive on via tcgen05_commit
    # after the last async_dot of each tile.
    if IS_BLACKWELL:
        acc_tmem = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_N),
            tl.float32,
            NUM_MMA_GROUPS,
            tlx.storage_kind.tmem,
        )
        bars_tmem_full = tlx.alloc_barriers(
            num_barriers=NUM_MMA_GROUPS, arrive_count=tl.constexpr(1)
        )

    with tlx.async_tasks():
        with tlx.async_task("default"):
            sm_id = tl.program_id(0)
            num_pid_m_max = tl.cdiv(M_max, BLOCK_M)
            num_pid_n_max = tl.cdiv(N_max, BLOCK_N)
            tiles_per_pair = num_pid_m_max * num_pid_n_max

            smem_accum_cnt = 0
            for pair_idx in range(NUM_A * NUM_B):
                pid_a = pair_idx // NUM_B
                pid_b = pair_idx % NUM_B  # pyre-ignore[58]
                M_i = lengths_m[pid_a]
                N_j = lengths_n[pid_b]
                num_pid_m = tl.cdiv(M_i, BLOCK_M)
                num_pid_n = tl.cdiv(N_j, BLOCK_N)

                # Create TMA descriptors ONCE per (a,b) pair, reuse
                # across all spatial tiles. With NUM_K constexpr, each
                # k_idx branch is unrolled and only valid branches exist.
                if NUM_K >= 1:
                    _ki0 = 0
                    _a0 = pid_a * NUM_K + _ki0
                    _b0 = _ki0 * NUM_B + pid_b
                    _K0 = lengths_k[_ki0]
                    _sam0 = strides_am[_ki0]
                    _gw_scope_start(1)
                    desc_a_0 = tl.make_tensor_descriptor(
                        A_ptrs[_a0],
                        shape=[M_i, _K0],
                        strides=[_sam0, stride_ak],
                        block_shape=[BLOCK_M_SPLIT, BLOCK_K],
                    )
                    desc_b_0 = tl.make_tensor_descriptor(
                        B_ptrs[_b0],
                        shape=[_K0, N_j],
                        strides=[strides_bk[pid_b], stride_bn],
                        block_shape=[BLOCK_K, BLOCK_N],
                    )
                    _gw_scope_end(1)
                if NUM_K >= 2:
                    _ki1 = 1
                    _a1 = pid_a * NUM_K + _ki1
                    _b1 = _ki1 * NUM_B + pid_b
                    _K1 = lengths_k[_ki1]
                    _sam1 = strides_am[_ki1]
                    _gw_scope_start(1)
                    desc_a_1 = tl.make_tensor_descriptor(
                        A_ptrs[_a1],
                        shape=[M_i, _K1],
                        strides=[_sam1, stride_ak],
                        block_shape=[BLOCK_M_SPLIT, BLOCK_K],
                    )
                    desc_b_1 = tl.make_tensor_descriptor(
                        B_ptrs[_b1],
                        shape=[_K1, N_j],
                        strides=[strides_bk[pid_b], stride_bn],
                        block_shape=[BLOCK_K, BLOCK_N],
                    )
                    _gw_scope_end(1)
                if NUM_K >= 3:
                    _ki2 = 2
                    _a2 = pid_a * NUM_K + _ki2
                    _b2 = _ki2 * NUM_B + pid_b
                    _K2 = lengths_k[_ki2]
                    _sam2 = strides_am[_ki2]
                    _gw_scope_start(1)
                    desc_a_2 = tl.make_tensor_descriptor(
                        A_ptrs[_a2],
                        shape=[M_i, _K2],
                        strides=[_sam2, stride_ak],
                        block_shape=[BLOCK_M_SPLIT, BLOCK_K],
                    )
                    desc_b_2 = tl.make_tensor_descriptor(
                        B_ptrs[_b2],
                        shape=[_K2, N_j],
                        strides=[strides_bk[pid_b], stride_bn],
                        block_shape=[BLOCK_K, BLOCK_N],
                    )
                    _gw_scope_end(1)
                if NUM_K >= 4:
                    _ki3 = 3
                    _a3 = pid_a * NUM_K + _ki3
                    _b3 = _ki3 * NUM_B + pid_b
                    _K3 = lengths_k[_ki3]
                    _sam3 = strides_am[_ki3]
                    _gw_scope_start(1)
                    desc_a_3 = tl.make_tensor_descriptor(
                        A_ptrs[_a3],
                        shape=[M_i, _K3],
                        strides=[_sam3, stride_ak],
                        block_shape=[BLOCK_M_SPLIT, BLOCK_K],
                    )
                    desc_b_3 = tl.make_tensor_descriptor(
                        B_ptrs[_b3],
                        shape=[_K3, N_j],
                        strides=[strides_bk[pid_b], stride_bn],
                        block_shape=[BLOCK_K, BLOCK_N],
                    )
                    _gw_scope_end(1)

                tile_id = sm_id
                while tile_id < tiles_per_pair:
                    pid_mn = tile_id
                    num_pid_in_group = GROUP_M * num_pid_n
                    group_id = pid_mn // num_pid_in_group
                    first_pid_m = group_id * GROUP_M
                    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
                    group_size_m = max(group_size_m, 1)
                    pid_m = first_pid_m + (pid_mn % group_size_m)
                    pid_n = (pid_mn % num_pid_in_group) // group_size_m

                    if pid_m < num_pid_m and pid_n < num_pid_n:
                        m_off = (pid_m * BLOCK_M).to(tl.int32)
                        n_off = (pid_n * BLOCK_N).to(tl.int32)

                        for k_idx in range(NUM_K):
                            # Select cached descriptor for this k_idx.
                            # Default to k=0, then override for higher indices.
                            # Separate if blocks with constexpr NUM_K guards
                            # prevent Triton from resolving undefined names.
                            K_k = lengths_k[k_idx]
                            desc_a = desc_a_0  # pyre-ignore[61]
                            desc_b = desc_b_0  # pyre-ignore[61]
                            if NUM_K >= 2 and k_idx == 1:
                                desc_a = desc_a_1  # pyre-ignore[61]
                                desc_b = desc_b_1  # pyre-ignore[61]
                            if NUM_K >= 3 and k_idx == 2:
                                desc_a = desc_a_2  # pyre-ignore[61]
                                desc_b = desc_b_2  # pyre-ignore[61]
                            if NUM_K >= 4 and k_idx == 3:
                                desc_a = desc_a_3  # pyre-ignore[61]
                                desc_b = desc_b_3  # pyre-ignore[61]

                            for _k in range(0, tl.cdiv(K_k, BLOCK_K)):
                                buf, p = _get_bufidx_phase(smem_accum_cnt, NUM_STAGES)
                                k_off = (_k * BLOCK_K).to(tl.int32)

                                _gw_scope_start(2)  # Producer: TMA async loads
                                empty_a_1st = tlx.local_view(bars_empty_a, buf)
                                full_a_1st = tlx.local_view(bars_full_a, buf)
                                tlx.barrier_wait(bar=empty_a_1st, phase=p ^ 1)
                                tlx.barrier_expect_bytes(
                                    full_a_1st, BLOCK_M_SPLIT * BLOCK_K * 2
                                )
                                data_a_1st = tlx.local_view(a_smem, buf)
                                tlx.async_descriptor_load(
                                    desc_a,
                                    data_a_1st,
                                    [m_off, k_off],
                                    full_a_1st,
                                    eviction_policy="evict_last",
                                )

                                empty_b = tlx.local_view(bars_empty_b, buf)
                                full_b = tlx.local_view(bars_full_b, buf)
                                tlx.barrier_wait(bar=empty_b, phase=p ^ 1)
                                tlx.barrier_expect_bytes(full_b, BLOCK_K * BLOCK_N * 2)
                                data_b = tlx.local_view(b_smem, buf)

                                tlx.async_descriptor_load(
                                    desc_b,
                                    data_b,
                                    [k_off, n_off],
                                    full_b,
                                    eviction_policy="evict_last",
                                )

                                empty_a_2nd = tlx.local_view(
                                    bars_empty_a, buf + NUM_STAGES
                                )
                                full_a_2nd = tlx.local_view(
                                    bars_full_a, buf + NUM_STAGES
                                )
                                tlx.barrier_wait(bar=empty_a_2nd, phase=p ^ 1)
                                tlx.barrier_expect_bytes(
                                    full_a_2nd, BLOCK_M_SPLIT * BLOCK_K * 2
                                )
                                data_a_2nd = tlx.local_view(a_smem, buf + NUM_STAGES)
                                tlx.async_descriptor_load(
                                    desc_a,
                                    data_a_2nd,
                                    [m_off + BLOCK_M_SPLIT, k_off],
                                    full_a_2nd,
                                    eviction_policy="evict_last",
                                )
                                _gw_scope_end(2)

                                smem_accum_cnt += 1

                    tile_id += NUM_SMS

        with tlx.async_task(num_warps=4, replicate=NUM_MMA_GROUPS):
            sm_id = tl.program_id(0)
            num_pid_m_max = tl.cdiv(M_max, BLOCK_M)
            num_pid_n_max = tl.cdiv(N_max, BLOCK_N)
            tiles_per_pair = num_pid_m_max * num_pid_n_max

            smem_accum_cnt = 0
            # Per-replica TMEM view and MMA-done barrier (Blackwell only).
            # tmem_phase_cnt advances once per actually-processed tile, used
            # as the wait phase for the MMA-done barrier.
            if IS_BLACKWELL:
                my_acc_tmem = tlx.local_view(acc_tmem, tlx.async_task_replica_id())
                my_tmem_full = tlx.local_view(
                    bars_tmem_full, tlx.async_task_replica_id()
                )
                tmem_phase_cnt = 0
            for pair_idx in range(NUM_A * NUM_B):
                pid_a = pair_idx // NUM_B
                pid_b = pair_idx % NUM_B  # pyre-ignore[58]
                M_i = lengths_m[pid_a]
                N_j = lengths_n[pid_b]
                num_pid_m = tl.cdiv(M_i, BLOCK_M)
                num_pid_n = tl.cdiv(N_j, BLOCK_N)

                tile_id = sm_id
                while tile_id < tiles_per_pair:
                    pid_mn = tile_id
                    num_pid_in_group = GROUP_M * num_pid_n
                    group_id = pid_mn // num_pid_in_group
                    first_pid_m = group_id * GROUP_M
                    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
                    group_size_m = max(group_size_m, 1)
                    pid_m = first_pid_m + (pid_mn % group_size_m)
                    pid_n = (pid_mn % num_pid_in_group) // group_size_m

                    if pid_m < num_pid_m and pid_n < num_pid_n:
                        m_off = (pid_m * BLOCK_M).to(tl.int32)
                        n_off = (pid_n * BLOCK_N).to(tl.int32)
                        if IS_BLACKWELL:
                            # use_acc switches from False on the first k-iter
                            # of this tile to True for the rest, accumulating
                            # into my_acc_tmem.
                            iter_in_tile = 0
                        else:
                            acc = tl.zeros([BLOCK_M_SPLIT, BLOCK_N], dtype=tl.float32)

                        for k_idx in range(NUM_K):
                            K_k = lengths_k[k_idx]

                            for _k in range(0, tl.cdiv(K_k, BLOCK_K)):
                                buf, p = _get_bufidx_phase(smem_accum_cnt, NUM_STAGES)

                                _gw_scope_start(3)  # Consumer: barrier wait + GEMM dot
                                full_a = tlx.local_view(
                                    bars_full_a,
                                    buf + NUM_STAGES * tlx.async_task_replica_id(),
                                )
                                full_b = tlx.local_view(bars_full_b, buf)
                                tlx.barrier_wait(bar=full_a, phase=p)
                                tlx.barrier_wait(bar=full_b, phase=p)

                                data_a = tlx.local_view(
                                    a_smem,
                                    buf + NUM_STAGES * tlx.async_task_replica_id(),
                                )
                                data_b = tlx.local_view(b_smem, buf)
                                empty_a = tlx.local_view(
                                    bars_empty_a,
                                    buf + NUM_STAGES * tlx.async_task_replica_id(),
                                )
                                empty_b = tlx.local_view(bars_empty_b, buf)
                                if IS_BLACKWELL:
                                    # tcgen05.mma writes to TMEM async; pass
                                    # empty_a / empty_b as mBarriers so the
                                    # producer is signaled when this MMA has
                                    # consumed the SMEM operands.
                                    tlx.async_dot(
                                        data_a,
                                        data_b,
                                        my_acc_tmem,
                                        use_acc=iter_in_tile != 0,
                                        mBarriers=[empty_a, empty_b],
                                    )
                                    iter_in_tile += 1
                                else:
                                    acc = tlx.async_dot(data_a, data_b, acc)
                                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)
                                _gw_scope_end(3)

                                if not IS_BLACKWELL:
                                    _gw_scope_start(
                                        4
                                    )  # Consumer: barrier arrive (signal producer)
                                    tlx.barrier_arrive(empty_a)
                                    tlx.barrier_arrive(empty_b)
                                    _gw_scope_end(4)

                                smem_accum_cnt += 1

                        if IS_BLACKWELL:
                            # Fence all preceding tcgen05.mma ops to the
                            # dedicated MMA-done barrier, wait, then load
                            # the accumulator out of TMEM into registers
                            # for the bias / store epilogue.
                            tlx.tcgen05_commit(my_tmem_full)
                            tlx.barrier_wait(my_tmem_full, tmem_phase_cnt & 1)
                            acc = tlx.local_load(my_acc_tmem)
                            tmem_phase_cnt += 1

                        _gw_scope_start(5)  # Consumer: bias epilogue
                        if HAS_BIAS:
                            offs_n = tl.arange(0, BLOCK_N)
                            mask_n = (pid_n * BLOCK_N + offs_n) < N_j
                            stride_bias_j = strides_bias[pid_b]
                            bias_ptrs = (
                                BIAS_ptrs[pid_b]
                                + (pid_n * BLOCK_N + offs_n) * stride_bias_j
                            )
                            bias = tl.load(bias_ptrs, mask=mask_n, other=0.0)
                            acc += bias[None, :].to(tl.float32)
                        _gw_scope_end(5)

                        _gw_scope_start(6)  # Consumer: TMA store
                        c_idx = pid_a * NUM_B + pid_b
                        c_m_off = m_off + BLOCK_M_SPLIT * tlx.async_task_replica_id()
                        if EPILOGUE_SUBTILE:
                            desc_c = tl.make_tensor_descriptor(
                                C_ptrs[c_idx],
                                shape=[M_i, N_j],
                                strides=[strides_cm[pid_b], stride_cn],
                                block_shape=[BLOCK_M_SPLIT, BLOCK_N // 2],
                            )
                            acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BLOCK_N // 2))
                            acc = tl.permute(acc, (0, 2, 1))
                            acc0, acc1 = tl.split(acc)
                            desc_c.store(
                                [c_m_off, n_off],
                                acc0.to(C_ptrs[c_idx].dtype.element_ty),
                            )
                            desc_c.store(
                                [c_m_off, n_off + BLOCK_N // 2],
                                acc1.to(C_ptrs[c_idx].dtype.element_ty),
                            )
                        else:
                            desc_c = tl.make_tensor_descriptor(
                                C_ptrs[c_idx],
                                shape=[M_i, N_j],
                                strides=[strides_cm[pid_b], stride_cn],
                                block_shape=[BLOCK_M_SPLIT, BLOCK_N],
                            )
                            desc_c.store(
                                [c_m_off, n_off], acc.to(C_ptrs[c_idx].dtype.element_ty)
                            )
                        _gw_scope_end(6)

                    tile_id += NUM_SMS


def _gemm_ws_preprocess_configs(configs, named_args, **kwargs):
    """Prune GEMM WS configs based on M/N ratio for L2 locality."""
    M = named_args["M"]
    N = named_args["N"]
    IMBALANCE_THRESHOLD = 10
    if M > N * IMBALANCE_THRESHOLD:
        configs = [c for c in configs if c.kwargs["GROUP_SIZE_M"] == 1]
    elif N > M * IMBALANCE_THRESHOLD:
        configs = [c for c in configs if c.kwargs["GROUP_SIZE_M"] >= 32]
    else:
        configs = [c for c in configs if c.kwargs["GROUP_SIZE_M"] == 8]
    return configs


def _get_gemm_ws_autotune_configs():
    return [
        triton.Config(
            {
                "BM": 128,
                "BN": 256,
                "BK": 64,
                "GROUP_SIZE_M": g,
                "NUM_STAGES": s,
                "NUM_MMA_WARPS": 8,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": epilogue,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=_tlx_set_block_size_hook,
        )
        for s in [3, 4]
        for epilogue in [True, False]
        for g in [1, 8, 64]
    ]


@triton.autotune(
    configs=_get_gemm_ws_autotune_configs(),
    key=["M", "N", "K"],
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": _gemm_ws_preprocess_configs},
)
@triton.jit
def _gemm_kernel_tlx_ws(
    a_desc,
    b_desc,
    c_desc,
    bias_ptr,
    M,
    N,
    K,
    stride_bias,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """TLX warp-specialized persistent GEMM: C = A @ B [+ bias]."""
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    a = tlx.local_alloc(
        (BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS
    )
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=tl.constexpr(1)
    )
    bars_full_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=tl.constexpr(1)
    )
    bars_empty_b = tlx.alloc_barriers(
        num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS
    )
    bars_full_b = tlx.alloc_barriers(
        num_barriers=NUM_STAGES, arrive_count=tl.constexpr(1)
    )

    with tlx.async_tasks():
        # Producer
        with tlx.async_task("default"):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            tile_id = sm_id
            smem_accum_cnt = 0
            while tile_id < num_tiles:
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                for k in range(0, tl.cdiv(K, BK)):
                    buf, p = _get_bufidx_phase(smem_accum_cnt, NUM_STAGES)
                    offset_k = k * BK

                    empty_a_1st = tlx.local_view(bars_empty_a, buf)
                    full_a_1st = tlx.local_view(bars_full_a, buf)
                    tlx.barrier_wait(bar=empty_a_1st, phase=p ^ 1)
                    tlx.barrier_expect_bytes(
                        full_a_1st,
                        BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)),
                    )
                    data_a_1st = tlx.local_view(a, buf)
                    tlx.async_descriptor_load(
                        a_desc, data_a_1st, [offset_am, offset_k], full_a_1st
                    )

                    empty_b = tlx.local_view(bars_empty_b, buf)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p ^ 1)
                    tlx.barrier_expect_bytes(
                        full_b, BN * BK * tlx.size_of(tlx.dtype_of(a_desc))
                    )
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(
                        b_desc, data_b, [offset_k, offset_bn], full_b
                    )

                    empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                    full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                    tlx.barrier_wait(bar=empty_a_2nd, phase=p ^ 1)
                    tlx.barrier_expect_bytes(
                        bar=full_a_2nd,
                        size=BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)),
                    )
                    data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)
                    tlx.async_descriptor_load(
                        a_desc,
                        data_a_2nd,
                        [offset_am + BLOCK_M_SPLIT, offset_k],
                        full_a_2nd,
                    )

                    smem_accum_cnt += 1

                tile_id += NUM_SMS

        # Consumer
        with tlx.async_task(num_warps=4, replicate=2):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            tile_id = sm_id
            smem_accum_cnt = 0
            while tile_id < num_tiles:
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                acc = tl.zeros([BM // 2, BN], dtype=tl.float32)
                for _k in range(0, tl.cdiv(K, BK)):
                    buf, p = _get_bufidx_phase(smem_accum_cnt, NUM_STAGES)

                    full_a = tlx.local_view(
                        bars_full_a,
                        buf + NUM_STAGES * tlx.async_task_replica_id(),
                    )
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    data_a = tlx.local_view(
                        a, buf + NUM_STAGES * tlx.async_task_replica_id()
                    )
                    data_b = tlx.local_view(b, buf)
                    acc = tlx.async_dot(data_a, data_b, acc)
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    empty_a = tlx.local_view(
                        bars_empty_a,
                        buf + NUM_STAGES * tlx.async_task_replica_id(),
                    )
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    tlx.barrier_arrive(empty_a)
                    tlx.barrier_arrive(empty_b)

                    smem_accum_cnt += 1

                if HAS_BIAS:
                    offs_n = tl.arange(0, BN)
                    bias_ptrs = bias_ptr + (offset_bn + offs_n) * stride_bias
                    mask_n = (offset_bn + offs_n) < N
                    bias_vals = tl.load(bias_ptrs, mask=mask_n, other=0.0)
                    acc = acc + bias_vals[None, :].to(tl.float32)

                offset_cm = offset_am + BLOCK_M_SPLIT * tlx.async_task_replica_id()
                if EPILOGUE_SUBTILE:
                    acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BN // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c0 = acc0.to(tlx.dtype_of(c_desc))
                    c_desc.store([offset_cm, offset_bn], c0)
                    c1 = acc1.to(tlx.dtype_of(c_desc))
                    c_desc.store([offset_cm, offset_bn + BN // 2], c1)
                else:
                    c_desc.store([offset_cm, offset_bn], acc.to(tlx.dtype_of(c_desc)))

                tile_id += NUM_SMS


WS_MIN_M_THRESHOLD = 8192


@torch.fx.wrap
def triton_gemm_ws(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Compute GEMM: C = A @ B [+ bias] using TLX warp-specialized kernel on H100."""
    M, K = A.shape
    _, N = B.shape

    if M < WS_MIN_M_THRESHOLD:
        return triton_gemm(A, B, bias=bias)

    assert TLX_AVAILABLE, "TLX GEMM requires triton.language.extra.tlx"
    assert TMA_AVAILABLE, "TLX GEMM requires TensorDescriptor"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(A, shape=[M, K], strides=[K, 1], block_shape=dummy_block)
    b_desc = TensorDescriptor(B, shape=[K, N], strides=[N, 1], block_shape=dummy_block)
    c_desc = TensorDescriptor(C, shape=[M, N], strides=[N, 1], block_shape=dummy_block)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = lambda META: (  # noqa E731
        min(NUM_SMS, triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"])),
    )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else A  # dummy pointer when no bias
    bias_stride = bias.stride(0) if bias is not None else 1

    _gemm_kernel_tlx_ws[grid](
        a_desc,
        b_desc,
        c_desc,
        bias_ptr,
        M,
        N,
        K,
        bias_stride,
        NUM_SMS=NUM_SMS,
        HAS_BIAS=has_bias,
    )

    return C


@torch.fx.wrap
def triton_gemm_tlx(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Compute GEMM: C = A @ B using TLX warp-specialized kernel on H100."""
    assert TLX_AVAILABLE, "TLX GEMM requires triton.language.extra.tlx"
    result = _tlx_matmul(A, B)
    if result.dtype != A.dtype:
        result = result.to(A.dtype)
    return result


@torch._dynamo.assume_constant_result
def _get_blocked_gemm_autotuned_kernel(
    version: str, num_a: int, num_b: int, num_k: int
):  # type: ignore[no-untyped-def]
    """Pre-compute unrolled + autotuned kernel outside Dynamo-traced region."""
    vararg_N = {
        "A": num_a * num_k,
        "B": num_k * num_b,
        "C": num_a * num_b,
        "BIAS": num_b,
        "LENGTHS_M": num_a,
        "LENGTHS_N": num_b,
        "LENGTHS_K": num_k,
        "STRIDES_AM": num_k,
        "STRIDES_BK": num_b,
        "STRIDES_CM": num_b,
        "STRIDES_BIAS": num_b,
    }
    if version == "ws":
        unrolled_kernel = unroll_varargs(
            _blocked_gemm_kernel_ws,
            N=vararg_N,
            mode=VarargMode.CONDITIONAL,
        )
        autotuned = _get_autotune_kernel_blocked_gemm_ws(unrolled_kernel)
    elif version == "bwd":
        unrolled_kernel = unroll_varargs(
            _blocked_gemm_bwd_kernel,
            N=vararg_N,
            mode=VarargMode.CONDITIONAL,
        )
        autotuned = _get_autotune_kernel_blocked_gemm_bwd(unrolled_kernel)
    else:
        unrolled_kernel = unroll_varargs(
            _blocked_gemm_kernel_ab_varargs,
            N=vararg_N,
            mode=VarargMode.CONDITIONAL,
        )
        autotuned = _get_autotune_kernel_blocked_gemm(unrolled_kernel)
    return autotuned


@maybe_register_custom_op("hammer::triton_blocked_gemm", mutates_args=())
def _triton_blocked_gemm_op(
    A_flat: List[torch.Tensor],
    B_flat: List[torch.Tensor],
    bias_flat: List[torch.Tensor],
    num_a: int,
    num_b: int,
    num_k: int,
    has_bias: bool,
    version: str,
) -> List[torch.Tensor]:
    """Flat custom-op that allocates C, launches the kernel, and returns C_flat."""
    device = A_flat[0].device
    dtype = A_flat[0].dtype

    # WS kernel only supports fp16/bf16 (shared memory and TMA byte sizes
    # are hardcoded for 2-byte types); fall back to default kernel otherwise.
    if version == "ws" and dtype not in (torch.float16, torch.bfloat16):
        version = ""

    lengths_m = [A_flat[i * num_k].shape[0] for i in range(num_a)]
    lengths_n = [B_flat[j].shape[1] for j in range(num_b)]
    lengths_k = [A_flat[k].shape[1] for k in range(num_k)]
    M_max = max(lengths_m) if lengths_m else 0
    N_max = max(lengths_n) if lengths_n else 0
    K_max = max(lengths_k) if lengths_k else 0

    C_flat = [
        torch.empty((lengths_m[i], lengths_n[j]), dtype=dtype, device=device)
        for i in range(num_a)
        for j in range(num_b)
    ]

    if M_max == 0 or N_max == 0:
        return C_flat

    strides_am = [A_flat[k].stride(0) for k in range(num_k)]
    strides_bk = [B_flat[j].stride(0) for j in range(num_b)]
    strides_cm = [C_flat[j].stride(0) for j in range(num_b)]
    strides_bias = [b.stride(0) for b in bias_flat[:num_b]] if has_bias else [1] * num_b

    autotuned_kernel = _get_blocked_gemm_autotuned_kernel(version, num_a, num_b, num_k)

    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count

    grid = lambda meta: (  # noqa E731
        min(
            NUM_SMS,
            num_a
            * num_b
            * triton.cdiv(M_max, meta["BLOCK_M"])
            * triton.cdiv(N_max, meta["BLOCK_N"]),
        ),
    )

    extra_kwargs = {}
    if version == "ws":
        extra_kwargs["INPUT_DTYPE"] = (
            tl.float16 if dtype == torch.float16 else tl.bfloat16
        )
        # Blackwell (sm_100+) requires the WS kernel to use a TMEM accumulator
        # because tlx.async_dot lowers to tcgen05.mma. Detect by device
        # capability and forward to the constexpr branch in the kernel.
        extra_kwargs["IS_BLACKWELL"] = (
            not torch.version.hip and torch.cuda.get_device_capability(device)[0] >= 10
        )
    elif version == "bwd":
        # EVEN_K/M/N: skip masking when all dimensions are BLOCK-aligned.
        # Use min for K (inner loop - only affects last iteration masking),
        # but max for M/N (tile boundary - must be safe for largest block).
        bwd_configs = get_bwd_autotune_configs()
        min_block_k = min(c.kwargs["BLOCK_K"] for c in bwd_configs)
        max_block_m = max(c.kwargs["BLOCK_M"] for c in bwd_configs)
        max_block_n = max(c.kwargs["BLOCK_N"] for c in bwd_configs)
        extra_kwargs["EVEN_K"] = all(k % min_block_k == 0 for k in lengths_k)
        extra_kwargs["EVEN_M"] = all(m % max_block_m == 0 for m in lengths_m)
        extra_kwargs["EVEN_N"] = all(n % max_block_n == 0 for n in lengths_n)

    _ensure_triton_allocator()

    # TMA requires contiguous inner dimension (stride=1).
    if version == "ws":
        A_flat = [a.contiguous() if a.stride(-1) != 1 else a for a in A_flat]
        B_flat = [b.contiguous() if b.stride(-1) != 1 else b for b in B_flat]
        # Recompute strides after making contiguous
        strides_am = [A_flat[k].stride(0) for k in range(num_k)]
        strides_bk = [B_flat[j].stride(0) for j in range(num_b)]

    autotuned_kernel[grid](
        *A_flat,
        *B_flat,
        *C_flat,
        *bias_flat,
        *lengths_m,
        *lengths_n,
        *lengths_k,
        *strides_am,
        *strides_bk,
        *strides_cm,
        *strides_bias,
        M_max,
        N_max,
        K_max,
        A_flat[0].stride(1),
        B_flat[0].stride(1),
        C_flat[0].stride(1),
        NUM_A=num_a,
        NUM_B=num_b,
        NUM_K=num_k,
        HAS_BIAS=has_bias,
        NUM_SMS=NUM_SMS,
        **extra_kwargs,
    )

    return C_flat


@_triton_blocked_gemm_op.register_fake
def _triton_blocked_gemm_op_fake(
    A_flat: List[torch.Tensor],
    B_flat: List[torch.Tensor],
    bias_flat: List[torch.Tensor],
    num_a: int,
    num_b: int,
    num_k: int,
    has_bias: bool,
    version: str,
) -> List[torch.Tensor]:
    """FakeTensor implementation: return empty C tensors with correct shapes."""
    device = A_flat[0].device
    dtype = A_flat[0].dtype
    return [
        torch.empty(
            (A_flat[i * num_k].shape[0], B_flat[j].shape[1]),
            dtype=dtype,
            device=device,
        )
        for i in range(num_a)
        for j in range(num_b)
    ]


def _triton_blocked_gemm_forward(
    A_list: List[List[torch.Tensor]],
    B_list: List[List[torch.Tensor]],
    bias_list: "List[torch.Tensor] | None" = None,
    version: str = "",
) -> List[List[torch.Tensor]]:
    """
    Internal forward-only blocked GEMM with K blocking (no autograd):
        C_list[i][j] = sum_k A_list[i][k] @ B_list[j][k] + bias_j

    A_list[i][k] has shape (M_i, K_k) - outer: M blocks, inner: K blocks.
    B_list[j][k] has shape (K_k, N_j) - outer: N blocks, inner: K blocks.
    Output C_list[i][j] has shape (M_i, N_j).
    """
    num_a = len(A_list)
    num_b = len(B_list)
    num_k = len(A_list[0])

    A_flat = [A_list[i][k] for i in range(num_a) for k in range(num_k)]
    B_flat = [B_list[j][k] for k in range(num_k) for j in range(num_b)]

    has_bias = bias_list is not None
    bias_flat = bias_list if has_bias else [B_list[j][0] for j in range(num_b)]

    with torch.no_grad():
        C_flat = _triton_blocked_gemm_op(
            A_flat, B_flat, bias_flat, num_a, num_b, num_k, has_bias, version
        )

    return [[C_flat[i * num_b + j] for j in range(num_b)] for i in range(num_a)]


# =============================================================================
# dW fused TN GEMM: dW[kb] = dC^T @ A[kb] with inner-loop dC sharing
# =============================================================================
#
# For block_K configs (num_k=3), fuses all K-blocks into one kernel.
# Each inner loop step loads dC ONCE and reuses it (via tl.trans) across
# all 3 K-blocks. Reduces HBM reads from 480MB to 224MB (2.14x).


@triton.jit
def _dw_tn_fused2_kernel(
    dC_ptr,
    A0_ptr,
    A1_ptr,
    p0_ptr,
    p1_ptr,
    M,
    N,
    K0,
    K1,
    K_max,
    stride_dc_m,
    stride_a0_m,
    stride_a1_m,
    stride_p0_s,
    stride_p0_n,
    stride_p1_s,
    stride_p1_n,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Fused 2x TN GEMM: dW[kb] = dC^T @ A[kb].
    dC loaded ONCE per reduction step, tl.trans amortized across 2 dots.
    K0, K1 may differ - tiles beyond each block's K are skipped.
    """
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K_max, BK)
    nk_tiles = n_tiles * k_tiles

    pid_sk = pid // nk_tiles
    pid_nk = pid % nk_tiles
    pid_n = pid_nk // k_tiles
    pid_k = pid_nk % k_tiles

    n_off = (pid_n * BN).to(tl.int32)
    k_off = (pid_k * BK).to(tl.int32)

    m_chunk = tl.cdiv(M, SPLIT_K)
    m_start = pid_sk * m_chunk
    m_steps = tl.cdiv(tl.minimum(m_chunk, M - m_start), BM)

    desc_dc = tl.make_tensor_descriptor(
        dC_ptr,
        shape=[M, N],
        strides=[stride_dc_m, 1],
        block_shape=[BM, BN],
    )
    desc_a0 = tl.make_tensor_descriptor(
        A0_ptr,
        shape=[M, K0],
        strides=[stride_a0_m, 1],
        block_shape=[BM, BK],
    )
    desc_a1 = tl.make_tensor_descriptor(
        A1_ptr,
        shape=[M, K1],
        strides=[stride_a1_m, 1],
        block_shape=[BM, BK],
    )

    active0 = k_off < K0
    active1 = k_off < K1

    acc0 = tl.zeros((BN, BK), dtype=tl.float32)
    acc1 = tl.zeros((BN, BK), dtype=tl.float32)

    for _m in range(m_steps):
        m_off = (m_start + _m * BM).to(tl.int32)
        dc = desc_dc.load([m_off, n_off])
        dc_t = tl.trans(dc)

        if active0:
            a0 = desc_a0.load([m_off, k_off])
            acc0 += tl.dot(dc_t, a0)

        if active1:
            a1 = desc_a1.load([m_off, k_off])
            acc1 += tl.dot(dc_t, a1)

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * BK + tl.arange(0, BK)

    if active0:
        mask0 = (offs_n[:, None] < N) & (offs_k[None, :] < K0)
        out0 = p0_ptr + pid_sk * stride_p0_s
        tl.store(
            out0 + offs_n[:, None] * stride_p0_n + offs_k[None, :],
            acc0,
            mask=mask0,
        )
    if active1:
        mask1 = (offs_n[:, None] < N) & (offs_k[None, :] < K1)
        out1 = p1_ptr + pid_sk * stride_p1_s
        tl.store(
            out1 + offs_n[:, None] * stride_p1_n + offs_k[None, :],
            acc1,
            mask=mask1,
        )


@triton.jit
def _dw_tn_fused3_kernel(
    dC_ptr,
    A0_ptr,
    A1_ptr,
    A2_ptr,
    p0_ptr,
    p1_ptr,
    p2_ptr,
    M,
    N,
    K_per,
    stride_dc_m,
    stride_a0_m,
    stride_a1_m,
    stride_a2_m,
    stride_p_s,
    stride_p_n,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Fused 3x TN GEMM: dW[kb] = dC^T @ A[kb].
    dC loaded ONCE per reduction step, tl.trans amortized across 3 dots.
    """
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K_per, BK)
    nk_tiles = n_tiles * k_tiles

    pid_sk = pid // nk_tiles
    pid_nk = pid % nk_tiles
    pid_n = pid_nk // k_tiles
    pid_k = pid_nk % k_tiles

    n_off = (pid_n * BN).to(tl.int32)
    k_off = (pid_k * BK).to(tl.int32)

    m_chunk = tl.cdiv(M, SPLIT_K)
    m_start = pid_sk * m_chunk
    m_steps = tl.cdiv(tl.minimum(m_chunk, M - m_start), BM)

    desc_dc = tl.make_tensor_descriptor(
        dC_ptr,
        shape=[M, N],
        strides=[stride_dc_m, 1],
        block_shape=[BM, BN],
    )
    desc_a0 = tl.make_tensor_descriptor(
        A0_ptr,
        shape=[M, K_per],
        strides=[stride_a0_m, 1],
        block_shape=[BM, BK],
    )
    desc_a1 = tl.make_tensor_descriptor(
        A1_ptr,
        shape=[M, K_per],
        strides=[stride_a1_m, 1],
        block_shape=[BM, BK],
    )
    desc_a2 = tl.make_tensor_descriptor(
        A2_ptr,
        shape=[M, K_per],
        strides=[stride_a2_m, 1],
        block_shape=[BM, BK],
    )

    acc0 = tl.zeros((BN, BK), dtype=tl.float32)
    acc1 = tl.zeros((BN, BK), dtype=tl.float32)
    acc2 = tl.zeros((BN, BK), dtype=tl.float32)

    for _m in range(m_steps):
        m_off = (m_start + _m * BM).to(tl.int32)
        dc = desc_dc.load([m_off, n_off])
        dc_t = tl.trans(dc)

        a0 = desc_a0.load([m_off, k_off])
        acc0 += tl.dot(dc_t, a0)

        a1 = desc_a1.load([m_off, k_off])
        acc1 += tl.dot(dc_t, a1)

        a2 = desc_a2.load([m_off, k_off])
        acc2 += tl.dot(dc_t, a2)

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * BK + tl.arange(0, BK)
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K_per)

    out0 = p0_ptr + pid_sk * stride_p_s
    tl.store(out0 + offs_n[:, None] * stride_p_n + offs_k[None, :], acc0, mask=mask)
    out1 = p1_ptr + pid_sk * stride_p_s
    tl.store(out1 + offs_n[:, None] * stride_p_n + offs_k[None, :], acc1, mask=mask)
    out2 = p2_ptr + pid_sk * stride_p_s
    tl.store(out2 + offs_n[:, None] * stride_p_n + offs_k[None, :], acc2, mask=mask)


@triton.jit
def _dw_reduce_kernel(
    partial_ptr,
    dW_ptr,
    R,
    C,
    stride_p_s,
    stride_p_r,
    stride_dw_r,
    BR: tl.constexpr,
    BC: tl.constexpr,
    SPLIT_K: tl.constexpr,
    # pyre-ignore[9]: defaults are constexpr at Triton compile time
    OUT_DTYPE: tl.constexpr = tl.float16,
):
    """Reduce fp32 split-K partials to output dtype."""
    pid = tl.program_id(0)
    c_tiles = tl.cdiv(C, BC)
    pid_r = pid // c_tiles
    pid_c = pid % c_tiles

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)
    mask = (offs_r[:, None] < R) & (offs_c[None, :] < C)

    acc = tl.zeros((BR, BC), dtype=tl.float32)
    for s in range(SPLIT_K):
        p = tl.load(
            partial_ptr
            + s * stride_p_s
            + offs_r[:, None] * stride_p_r
            + offs_c[None, :],
            mask=mask,
            other=0.0,
        )
        acc += p

    tl.store(
        dW_ptr + offs_r[:, None] * stride_dw_r + offs_c[None, :],
        acc.to(OUT_DTYPE),
        mask=mask,
    )


def _dw_fused_tn_2(
    dC: torch.Tensor,
    A_list: List[torch.Tensor],
    BN: int = 64,
    BK: int = 64,
    BM: int = 128,
    split_k: int = 2,
) -> List[torch.Tensor]:
    """Fused dW[k] = dC^T @ A[k] for 2 K-blocks using TN GEMM with dC sharing.

    Supports variable K dimensions across blocks.
    Returns list of 2 dW tensors, each (N, K_k).
    """
    M, N = dC.shape
    K0, K1 = A_list[0].shape[1], A_list[1].shape[1]
    K_max = max(K0, K1)

    n_tiles = triton.cdiv(N, BN)
    k_tiles = triton.cdiv(K_max, BK)

    if split_k == 1:
        dW_list = [
            torch.empty(N, Kk, device=dC.device, dtype=torch.float32) for Kk in (K0, K1)
        ]
        _dw_tn_fused2_kernel[(n_tiles * k_tiles,)](
            dC,
            A_list[0],
            A_list[1],
            dW_list[0],
            dW_list[1],
            M,
            N,
            K0,
            K1,
            K_max,
            dC.stride(0),
            A_list[0].stride(0),
            A_list[1].stride(0),
            0,
            dW_list[0].stride(0),
            0,
            dW_list[1].stride(0),
            BN=BN,  # pyre-ignore[6]
            BK=BK,  # pyre-ignore[6]
            BM=BM,  # pyre-ignore[6]
            SPLIT_K=1,  # pyre-ignore[6]
        )
        return [w.to(dC.dtype) for w in dW_list]
    else:
        partials = [
            torch.empty(split_k, N, Kk, device=dC.device, dtype=torch.float32)
            for Kk in (K0, K1)
        ]
        dW_list = [
            torch.empty(N, Kk, device=dC.device, dtype=dC.dtype) for Kk in (K0, K1)
        ]

        _dw_tn_fused2_kernel[(n_tiles * k_tiles * split_k,)](
            dC,
            A_list[0],
            A_list[1],
            partials[0],
            partials[1],
            M,
            N,
            K0,
            K1,
            K_max,
            dC.stride(0),
            A_list[0].stride(0),
            A_list[1].stride(0),
            partials[0].stride(0),
            partials[0].stride(1),
            partials[1].stride(0),
            partials[1].stride(1),
            BN=BN,  # pyre-ignore[6]
            BK=BK,  # pyre-ignore[6]
            BM=BM,  # pyre-ignore[6]
            SPLIT_K=split_k,  # pyre-ignore[6]
        )

        out_tl_dtype = tl.bfloat16 if dC.dtype == torch.bfloat16 else tl.float16
        Ks = (K0, K1)
        for i in range(2):
            reduce_tiles = triton.cdiv(N, BN) * triton.cdiv(Ks[i], BK)
            _dw_reduce_kernel[(reduce_tiles,)](
                partials[i],
                dW_list[i],
                N,
                Ks[i],
                partials[i].stride(0),
                partials[i].stride(1),
                dW_list[i].stride(0),
                BR=BN,  # pyre-ignore[6]
                BC=BK,  # pyre-ignore[6]
                SPLIT_K=split_k,  # pyre-ignore[6]
                OUT_DTYPE=out_tl_dtype,  # pyre-ignore[6]
            )
        return dW_list


def _dw_fused_tn_3(
    dC: torch.Tensor,
    A_list: List[torch.Tensor],
    BN: int = 64,
    BK: int = 64,
    BM: int = 128,
    split_k: int = 2,
) -> List[torch.Tensor]:
    """Fused dW[k] = dC^T @ A[k] for 3 K-blocks using TN GEMM with dC sharing.

    All K dimensions must be equal across blocks.
    Returns list of 3 dW tensors, each (N, K_per).
    """
    assert len(A_list) == 3
    M, N = dC.shape
    K_per = A_list[0].shape[1]

    n_tiles = triton.cdiv(N, BN)
    k_tiles = triton.cdiv(K_per, BK)

    if split_k == 1:
        dW_list = [
            torch.empty(N, K_per, device=dC.device, dtype=torch.float32)
            for _ in range(3)
        ]
        _dw_tn_fused3_kernel[(n_tiles * k_tiles,)](
            dC,
            A_list[0],
            A_list[1],
            A_list[2],
            dW_list[0],
            dW_list[1],
            dW_list[2],
            M,
            N,
            K_per,
            dC.stride(0),
            A_list[0].stride(0),
            A_list[1].stride(0),
            A_list[2].stride(0),
            0,
            dW_list[0].stride(0),
            BN=BN,  # pyre-ignore[6]
            BK=BK,  # pyre-ignore[6]
            BM=BM,  # pyre-ignore[6]
            SPLIT_K=1,  # pyre-ignore[6]
        )
        return [w.to(dC.dtype) for w in dW_list]
    else:
        partials = [
            torch.empty(split_k, N, K_per, device=dC.device, dtype=torch.float32)
            for _ in range(3)
        ]
        dW_list = [
            torch.empty(N, K_per, device=dC.device, dtype=dC.dtype) for _ in range(3)
        ]

        _dw_tn_fused3_kernel[(n_tiles * k_tiles * split_k,)](
            dC,
            A_list[0],
            A_list[1],
            A_list[2],
            partials[0],
            partials[1],
            partials[2],
            M,
            N,
            K_per,
            dC.stride(0),
            A_list[0].stride(0),
            A_list[1].stride(0),
            A_list[2].stride(0),
            partials[0].stride(0),
            partials[0].stride(1),
            BN=BN,  # pyre-ignore[6]
            BK=BK,  # pyre-ignore[6]
            BM=BM,  # pyre-ignore[6]
            SPLIT_K=split_k,  # pyre-ignore[6]
        )

        out_tl_dtype = tl.bfloat16 if dC.dtype == torch.bfloat16 else tl.float16
        for i in range(3):
            reduce_tiles = triton.cdiv(N, BN) * triton.cdiv(K_per, BK)
            _dw_reduce_kernel[(reduce_tiles,)](
                partials[i],
                dW_list[i],
                N,
                K_per,
                partials[i].stride(0),
                partials[i].stride(1),
                dW_list[i].stride(0),
                BR=BN,  # pyre-ignore[6]
                BC=BK,  # pyre-ignore[6]
                SPLIT_K=split_k,  # pyre-ignore[6]
                OUT_DTYPE=out_tl_dtype,  # pyre-ignore[6]
            )
        return dW_list


# =============================================================================
# Standalone backward pass for blocked GEMM (can be called outside autograd)
# =============================================================================


def triton_blocked_gemm_backward(  # noqa: C901
    dC_list: List[List[torch.Tensor]],
    A_list: List[List[torch.Tensor]],
    W_list: List[List[torch.Tensor]],
    has_bias: bool = False,
    version: str = "ws",
) -> Tuple[
    List[List[torch.Tensor]],
    List[List[torch.Tensor]],
    List[torch.Tensor],
]:
    """Compute gradients for blocked GEMM using Triton blocked GEMM kernels.

    Given forward: C[i][j] = sum_k A[i][k] @ W[j][k]^T + bias[j]

    Computes:
      dA[i][k] = sum_j dC[i][j] @ W[j][k]           (Triton blocked GEMM)
      dW[j][k] = sum_i dC[i][j]^T @ A[i][k]          (cuBLAS mm/addmm)
      dbias[j] = sum_i dC[i][j].sum(dim=0)            (simple reduction)
    """
    num_a = len(A_list)
    num_b = len(W_list)
    num_k = len(A_list[0])

    # dA uses forward ws kernel for fp16/bf16 - same dimension pattern as forward.
    dA_ver = "ws" if version == "ws" else "bwd"

    # --- dA[i][k] = sum_j dC[i][j] @ W[j][k] ---
    A_for_dA = [[dC_list[i][j] for j in range(num_b)] for i in range(num_a)]
    B_for_dA = [[W_list[j][k] for j in range(num_b)] for k in range(num_k)]
    dA_list = _triton_blocked_gemm_forward(A_for_dA, B_for_dA, version=dA_ver)

    # --- dW[j][k] = sum_i dC[i][j]^T @ A[i][k] ---
    # For block_K (num_k=2 or 3), use fused TN Triton kernel that shares
    # dC loads across K-blocks in the inner loop.
    # For other configs, use cuBLAS mm/addmm.
    _strides_ok = (
        TMA_AVAILABLE
        and all(
            A_list[i][k].stride(-1) == 1 for i in range(num_a) for k in range(num_k)
        )
        and all(
            dC_list[i][j].stride(-1) == 1 for i in range(num_a) for j in range(num_b)
        )
    )
    use_fused_tn_3 = num_k == 3 and _strides_ok

    dW_list: List[List[torch.Tensor]] = []
    if use_fused_tn_3:
        for j in range(num_b):
            dW_row = _dw_fused_tn_3(
                dC_list[0][j],
                [A_list[0][k] for k in range(3)],
            )
            for i in range(1, num_a):
                dW_partial = _dw_fused_tn_3(
                    dC_list[i][j],
                    [A_list[i][k] for k in range(3)],
                )
                for k in range(3):
                    dW_row[k] = dW_row[k] + dW_partial[k]
            dW_list.append(dW_row)
    else:
        concat_bytes = (
            sum(A_list[0][k].numel() for k in range(num_k))
            * A_list[0][0].element_size()
            * num_a
        )
        use_k_batch = num_k > 1 and concat_bytes < 100 * 1024 * 1024  # 100MB
        if use_k_batch:
            A_k_cats = [
                torch.cat([A_list[i][k] for k in range(num_k)], dim=1)
                for i in range(num_a)
            ]
            K_splits = [A_list[0][k].shape[1] for k in range(num_k)]
            for j in range(num_b):
                dW_concat = torch.mm(dC_list[0][j].t(), A_k_cats[0])
                for i in range(1, num_a):
                    torch.addmm(
                        dW_concat, dC_list[i][j].t(), A_k_cats[i], out=dW_concat
                    )
                dW_list.append(
                    [t.contiguous() for t in torch.split(dW_concat, K_splits, dim=1)]
                )
        else:
            for j in range(num_b):
                dW_row_cublas: List[torch.Tensor] = []
                for k in range(num_k):
                    dW_jk = torch.mm(dC_list[0][j].t(), A_list[0][k])
                    for i in range(1, num_a):
                        torch.addmm(dW_jk, dC_list[i][j].t(), A_list[i][k], out=dW_jk)
                    dW_row_cublas.append(dW_jk)
                dW_list.append(dW_row_cublas)

    # --- dbias[j] = sum_i dC[i][j].sum(dim=0) ---
    dbias_list: List[torch.Tensor] = []
    if has_bias:
        for j in range(num_b):
            dbias_j = torch.stack([dC_list[i][j].sum(dim=0) for i in range(num_a)]).sum(
                dim=0
            )
            dbias_list.append(dbias_j)

    return dA_list, dW_list, dbias_list


# =============================================================================
# Autograd support for blocked GEMM backward pass
# =============================================================================


class _BlockedGemmFunction(torch.autograd.Function):
    """Autograd function wrapping triton_blocked_gemm forward and backward.

    Forward: C[i][j] = sum_k A[i][k] @ W[j][k]^T + bias[j]
    Backward:
      dA[i][k] = sum_j dC[i][j] @ W[j][k]           (blocked GEMM, no bias)
      dW[j][k] = sum_i dC[i][j]^T @ A[i][k]          (blocked GEMM, no bias)
      dbias[j] = sum_i dC[i][j].sum(dim=0)            (simple reduction)

    Tensors layout in *tensors arg:
      [A_00, A_01, ..., A_{a-1,k-1},
       W_00, W_01, ..., W_{b-1,k-1},
       bias_0, ..., bias_{b-1}]            (if has_bias)

    where A[i][k] has shape (M_i, K_k) and W[j][k] has shape (N_j, K_k).
    """

    @staticmethod
    # pyre-ignore[14]: inconsistent override
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        has_bias: bool,
        version: str,
        num_a: int,
        num_b: int,
        num_k: int,
        *tensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        nA = num_a * num_k
        nW = num_b * num_k

        A_flat = list(tensors[:nA])
        W_flat = list(tensors[nA : nA + nW])
        bias_list: Optional[List[torch.Tensor]] = (
            list(tensors[nA + nW :]) if has_bias else None
        )

        A_list = [[A_flat[i * num_k + k] for k in range(num_k)] for i in range(num_a)]
        # B[j][k] = W[j][k]^T, shape (K_k, N_j)
        B_list = [
            [W_flat[j * num_k + k].t().contiguous() for k in range(num_k)]
            for j in range(num_b)
        ]

        C_list = _triton_blocked_gemm_forward(
            A_list, B_list, bias_list=bias_list, version=version
        )

        ctx.save_for_backward(*A_flat, *W_flat)
        ctx.has_bias = has_bias  # pyre-ignore[16]
        ctx.num_a = num_a  # pyre-ignore[16]
        ctx.num_b = num_b  # pyre-ignore[16]
        ctx.num_k = num_k  # pyre-ignore[16]
        ctx.version = version  # pyre-ignore[16]

        C_flat = [C_list[i][j] for i in range(num_a) for j in range(num_b)]
        return tuple(C_flat)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        num_a: int = ctx.num_a  # pyre-ignore[16]
        num_b: int = ctx.num_b  # pyre-ignore[16]
        num_k: int = ctx.num_k  # pyre-ignore[16]
        has_bias: bool = ctx.has_bias  # pyre-ignore[16]
        version: str = ctx.version  # pyre-ignore[16]
        nA = num_a * num_k
        nW = num_b * num_k

        saved = ctx.saved_tensors  # pyre-ignore[16]
        A_flat = list(saved[:nA])
        W_flat = list(saved[nA : nA + nW])

        A_list = [[A_flat[i * num_k + k] for k in range(num_k)] for i in range(num_a)]
        W_list = [[W_flat[j * num_k + k] for k in range(num_k)] for j in range(num_b)]
        dC_list = [
            [grad_outputs[i * num_b + j] for j in range(num_b)] for i in range(num_a)
        ]

        dA_list, dW_list, dbias_list = triton_blocked_gemm_backward(
            dC_list, A_list, W_list, has_bias=has_bias, version=version
        )

        # Pack gradients matching forward(*tensors) order:
        # [A_flat..., W_flat..., bias...] -> [dA_flat..., dW_flat..., dbias...]
        dA_flat = [dA_list[i][k] for i in range(num_a) for k in range(num_k)]
        dW_flat = [dW_list[j][k] for j in range(num_b) for k in range(num_k)]

        grad_list: List[Optional[torch.Tensor]] = [
            None,  # has_bias
            None,  # version
            None,  # num_a
            None,  # num_b
            None,  # num_k
        ]
        grad_list.extend(dA_flat)
        grad_list.extend(dW_flat)
        grad_list.extend(dbias_list)
        return tuple(grad_list)


def triton_blocked_gemm(
    A_list: List[List[torch.Tensor]],
    W_list: List[List[torch.Tensor]],
    bias_list: Optional[List[torch.Tensor]] = None,
    version: str = "ws",
) -> List[List[torch.Tensor]]:
    """Blocked GEMM with autograd backward support.

    C[i][j] = sum_k A[i][k] @ W[j][k]^T + bias[j]

    Takes W_list (original weight matrices, shape N_j x K_k) and handles
    the transpose internally.

    Args:
        A_list: Input activations. A_list[i][k] has shape (M_i, K_k).
        W_list: Weight matrices. W_list[j][k] has shape (N_j, K_k).
        bias_list: Optional bias vectors. bias_list[j] has shape (N_j,).
        version: Kernel version ("ws" for warp-specialized, "" for default).

    Returns:
        C_list where C_list[i][j] has shape (M_i, N_j).
    """
    num_a = len(A_list)
    num_k = len(A_list[0])
    num_b = len(W_list)
    has_bias = bias_list is not None

    A_flat = [A_list[i][k] for i in range(num_a) for k in range(num_k)]
    W_flat = [W_list[j][k] for j in range(num_b) for k in range(num_k)]
    flat_tensors: List[torch.Tensor] = A_flat + W_flat
    if has_bias:
        assert bias_list is not None
        flat_tensors = flat_tensors + list(bias_list)

    C_flat = _BlockedGemmFunction.apply(
        has_bias, version, num_a, num_b, num_k, *flat_tensors
    )

    return [[C_flat[i * num_b + j] for j in range(num_b)] for i in range(num_a)]
