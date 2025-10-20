from functools import lru_cache

from typing import Optional

import torch
import triton
import triton.language as tl

from torch._inductor.kernel.mm import ScalingType

from tritonbench.utils.env_utils import is_cuda
from tritonbench.utils.triton_utils import has_experimental_descriptor

if has_experimental_descriptor():
    import triton.tools.experimental_descriptor

cublas = None
if is_cuda():
    try:
        from triton._C.libtriton import nvidia

        cublas_workspace = torch.empty(
            32 * 1024 * 1024, device="cuda", dtype=torch.uint8
        )
        cublas = nvidia.cublas.CublasLt(cublas_workspace)
    except (ImportError, IOError, AttributeError):
        pass


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    ret["flops8"] = 2.0 * M * N * K
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret["bytes"] = bytes_per_elem * (M * K + N * K)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M - start_m, offs_am, 0)
            offs_bn = tl.where(offs_bn < N - start_n, offs_bn, 0)
            offs_am = tl.max_contiguous(
                tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M
            )
            offs_bn = tl.max_contiguous(
                tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N
            )
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(
            a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
        )
        b = tl.load(
            b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
        )
        accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            if c_ptr.dtype == tl.float8e4nv:
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_persistent(a, b):
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_stages": 4,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,  #
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_SMS: tl.constexpr,
):  #
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
        )
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_configs():
    # Autotuner does not work with TMA. Use manual config.
    return {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_stages": 4,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }


def matmul_configs_blackwell():
    # Autotuner does not work with TMA. Use manual config.
    return {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 4,
            "num_warps": 4,  # Note: num_warps >= 4 required for TMA
            "WARP_SPECIALIZE": True,
            "EPILOGUE_SUBTILE": True,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 2,
            "num_warps": 2,
        },
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 2,
            "num_warps": 2,
        },
    }


def allocate_matmul_tma(a, b):
    configs = matmul_configs()

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    desc_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        a.data_ptr(),
        M,
        K,
        configs[dtype]["BLOCK_SIZE_M"],
        configs[dtype]["BLOCK_SIZE_K"],
        a.element_size(),
    )
    desc_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        b.data_ptr(),
        N,
        K,
        configs[dtype]["BLOCK_SIZE_N"],
        configs[dtype]["BLOCK_SIZE_K"],
        b.element_size(),
    )
    desc_c = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        c.data_ptr(),
        M,
        N,
        configs[dtype]["BLOCK_SIZE_M"],
        configs[dtype]["BLOCK_SIZE_N"],
        c.element_size(),
    )
    return c, desc_a, desc_b, desc_c


def matmul_tma_persistent(a, b, c, desc_a, desc_b, desc_c):
    configs = matmul_configs()

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    matmul_kernel_tma_persistent[grid](
        desc_a,
        desc_b,
        desc_c,  #
        M,
        N,
        K,  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    return c


# Blackwell Persistent + TMA
# Restrictions:
#   - (K, N) must be a multiple of 16 on B200 for all benchmarks
#   - num_warps >= 4
#       - TMA instructions expect at least a 128-thread group
#       - 1 warp = 32 threads, so each thread block requires 128 / 32 = 4 warps


def blackwell_persistent_tma(a, b, scale_a_ptr, scale_b_ptr, acc_dtype, scaling_mode):
    configs = matmul_configs_blackwell()

    # Check constraints.
    assert (
        a.shape[1] == b.shape[1]
    ), "Incompatible dimensions"  # a.shape = (M, K), b.shape = (N, K)
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    shape_dtype = a.dtype  # low-precision dtype, e.g. fp8

    NUM_SMS = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count

    c = torch.zeros((M, N), device=a.device, dtype=acc_dtype)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device=a.device)

    if hasattr(triton, "set_allocator"):
        triton.set_allocator(alloc_fn)
    else:
        return c

    if acc_dtype == torch.float16:
        acc_dtype_tl = tl.float16
    elif acc_dtype == torch.bfloat16:
        acc_dtype_tl = tl.bfloat16
    else:
        raise NotImplementedError(
            "Output types other than torch.float16 and torch.bfloat16 unsupported for FP8 Blackwell persistent + TMA kernels"
        )

    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    blackwell_persistent_tma_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        scale_a_ptr,  #
        scale_b_ptr,  #
        BLOCK_SIZE_M=configs[shape_dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[shape_dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[shape_dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[shape_dtype]["GROUP_SIZE_M"],  #
        ACC_TYPE=acc_dtype_tl,
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[shape_dtype]["num_stages"],  #
        num_warps=configs[shape_dtype]["num_warps"],  #
        SCALING_MODE=scaling_mode,  #
        WARP_SPECIALIZE=configs[shape_dtype]["WARP_SPECIALIZE"],  #
        EPILOGUE_SUBTILE=configs[shape_dtype]["EPILOGUE_SUBTILE"],  #
    )
    return c


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def blackwell_persistent_tma_kernel(
    a,
    b,
    acc,
    M,  #
    N,  #
    K,  #
    scale_a_ptr,  #
    scale_b_ptr,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ACC_TYPE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SCALING_MODE: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):  #
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    acc_desc = tl.make_tensor_descriptor(
        acc,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // 2]
        if EPILOGUE_SUBTILE
        else [BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    if SCALING_MODE == ScalingType.RowWise:
        # For row-wise scaling, we'll use the pointers as-is
        scale_a = scale_a_ptr
        scale_b = scale_b_ptr
    else:
        # For per-tensor scaling, we'll load the scalar values
        scale_a = tl.load(scale_a_ptr)
        scale_b = tl.load(scale_b_ptr)

    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
        )  # accumulate in high precision (fp32) for high accuracy
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a_block = a_desc.load([offs_am, offs_k])
            b_block = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a_block, b_block.T, accumulator, out_dtype=tl.float32)

        if SCALING_MODE == ScalingType.RowWise:
            offs_scale_m = offs_am + tl.arange(0, BLOCK_SIZE_M)
            offs_scale_n = offs_bn + tl.arange(0, BLOCK_SIZE_N)

            scale_a_block = tl.load(scale_a + offs_scale_m, mask=offs_am < M, other=0.0)
            scale_b_block = tl.load(scale_b + offs_scale_n, mask=offs_bn < N, other=0.0)

            a_scales = scale_a_block[:, None]
            b_scales = scale_b_block[None, :]
        else:
            a_scales = scale_a
            b_scales = scale_b

        accumulator *= a_scales * b_scales

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc_reshaped = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc_permuted = tl.permute(acc_reshaped, (0, 2, 1))
            acc0, acc1 = tl.split(acc_permuted)
            c0 = acc0.to(ACC_TYPE)
            acc_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(ACC_TYPE)
            acc_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(ACC_TYPE)
            acc_desc.store([offs_cm, offs_cn], c)
