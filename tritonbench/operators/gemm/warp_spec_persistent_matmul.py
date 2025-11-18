"""
This is copied from Triton matmul tutorial 9 and reduced to just the persistent matmul kernel
on blackwell with/without warpspec.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from tritonbench.utils.env_utils import is_tile_enabled

from .triton_matmul_configs import get_tileir_configs

# TODO: Add proton support


def torch_dtype_to_triton_dtype(dtype):
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float8_e4m3fn:
        return tl.float8e4nv
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def check_tma_alignment(strides, elem_bytes):
    for stride in strides[:-1]:
        if (stride * elem_bytes) % 16 != 0:
            raise RuntimeError("strides must be 16-byte aligned")
    if strides[-1] != 1:
        raise RuntimeError("Last dimension must be contiguous")


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        # ceil division to capture the correct number of bytes
        bytes_per_elem = (args["DTYPE"].int_bitwidth + 7) // 8
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


small_block_range = [32, 64, 128]
small_stage_range = [1, 2, 3, 4]
include_small_configs = os.environ.get("INCLUDE_SMALL_CONFIGS", "0") == "1"
if include_small_configs:
    bm_range = small_block_range
    bn_range = small_block_range + [256]
    bk_range = small_block_range
    default_s_range = small_stage_range
    tma_persistent_s_range = small_stage_range
else:
    bm_range = [128]
    bn_range = [128, 256]
    bk_range = [64, 128]
    default_s_range = [3, 4]
    tma_persistent_s_range = [2, 3, 4]


def matmul_get_configs(pre_hook=None):
    if is_tile_enabled():
        base_configs = get_tileir_configs()
        # Append warp_spec specific information
        configs = []
        for config in base_configs:
            new_kwargs = config.kwargs.copy()
            new_kwargs["BLOCK_SIZE_M"] = new_kwargs.pop("BLOCK_M")
            new_kwargs["BLOCK_SIZE_N"] = new_kwargs.pop("BLOCK_N")
            new_kwargs["BLOCK_SIZE_K"] = new_kwargs.pop("BLOCK_K")
            new_kwargs["GROUP_SIZE_M"] = new_kwargs.pop("GROUP_M")
            configs.append(
                triton.Config(
                    new_kwargs,
                    pre_hook=pre_hook,
                    num_stages=config.num_stages,
                    num_ctas=config.num_ctas,
                    num_warps=config.num_warps,
                )
            )
        return configs
    else:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": BM,
                    "BLOCK_SIZE_N": BN,
                    "BLOCK_SIZE_K": BK,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=s,
                num_warps=w,
                pre_hook=pre_hook,
            )
            for BM in bm_range
            for BN in bn_range
            for BK in bk_range
            for s in default_s_range
            for w in [4, 8]
        ]


def matmul_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    if nargs["TRANSPOSE_A"]:
        a_block_shape = [BLOCK_K, BLOCK_M]
    else:
        a_block_shape = [BLOCK_M, BLOCK_K]
    nargs["a_desc"].block_shape = a_block_shape
    if nargs["TRANSPOSE_B"]:
        b_block_shape = [BLOCK_N, BLOCK_K]
    else:
        b_block_shape = [BLOCK_K, BLOCK_N]
    nargs["b_desc"].block_shape = b_block_shape
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


@triton.autotune(
    configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma(
    a_desc,
    b_desc,
    c_desc,  #
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    DTYPE: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    dtype = DTYPE

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE_K
        if TRANSPOSE_A:
            a = a_desc.load([offs_k, offs_am])
            arg1 = a.T
        else:
            a = a_desc.load([offs_am, offs_k])
            arg1 = a
        if TRANSPOSE_B:
            b = b_desc.load([offs_bn, offs_k])
            arg2 = b.T
        else:
            b = b_desc.load([offs_k, offs_bn])
            arg2 = b
        accumulator = tl.dot(arg1, arg2, accumulator)

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


def blackwell_matmul_tma(a, b, warp_specialize: bool):
    # High-Level Options for B's layout
    # 1. (K, N) contiguous in N
    # 2. (K, N) contiguous in K
    # 3. (N, K) contiguous in N
    # 4. (N, K) contiguous in K
    # In practice, since you always load in the contiguous dimension
    # there are actually only 2 options
    # 1. Load in the K stride 1 (2 and 4)
    # 2. Load in the N stride 1 (1 and 3)
    transpose_a = a.stride()[-1] != 1
    transpose_b = (a.shape[1] != b.shape[1] and b.stride()[-1] != 1) or (
        a.shape[1] == b.shape[1] and b.stride()[-1] == 1
    )
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    if a.shape[1] != b.shape[1]:
        K, N = b.shape
    else:
        N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_strides = [a.stride()[1] if transpose_a else a.stride()[0], 1]
    check_tma_alignment(a_strides, (torch.finfo(a.dtype).bits + 7) // 8)
    if transpose_a:
        a_desc = TensorDescriptor(a, [K, M], a_strides, dummy_block)
    else:
        a_desc = TensorDescriptor(a, [M, K], a_strides, dummy_block)
    b_strides = [b.stride()[1] if b.stride()[0] == 1 else b.stride()[0], 1]
    check_tma_alignment(b_strides, (torch.finfo(b.dtype).bits + 7) // 8)
    if transpose_b:
        b_desc = TensorDescriptor(b, [N, K], b_strides, dummy_block)
    else:
        b_desc = TensorDescriptor(b, [K, N], b_strides, dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    matmul_kernel_tma[grid](
        a_desc,
        b_desc,
        c_desc,  #
        M,
        N,
        K,  #
        WARP_SPECIALIZE=warp_specialize,  #
        DTYPE=torch_dtype_to_triton_dtype(dtype),  #
        TRANSPOSE_A=transpose_a,
        TRANSPOSE_B=transpose_b,
    )
    return c


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def matmul_tma_persistent_get_configs(pre_hook=None):
    if is_tile_enabled():
        base_configs = get_tileir_configs()
        # Append warp_spec specific information
        configs = []
        for config in base_configs:
            new_kwargs = config.kwargs.copy()
            new_kwargs["BLOCK_SIZE_M"] = new_kwargs.pop("BLOCK_M")
            new_kwargs["BLOCK_SIZE_N"] = new_kwargs.pop("BLOCK_N")
            new_kwargs["BLOCK_SIZE_K"] = new_kwargs.pop("BLOCK_K")
            new_kwargs["GROUP_SIZE_M"] = new_kwargs.pop("GROUP_M")
            for SUBTILE in [True, False]:
                configs.append(
                    triton.Config(
                        {
                            "EPILOGUE_SUBTILE": SUBTILE,
                            **new_kwargs,
                        },
                        pre_hook=pre_hook,
                        num_stages=config.num_stages,
                        num_ctas=config.num_ctas,
                        num_warps=config.num_warps,
                    )
                )
        return configs
    else:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": BM,
                    "BLOCK_SIZE_N": BN,
                    "BLOCK_SIZE_K": BK,
                    "GROUP_SIZE_M": 8,
                    "EPILOGUE_SUBTILE": SUBTILE,
                },
                num_stages=s,
                num_warps=w,
                pre_hook=pre_hook,
            )  #
            for BM in bm_range  #
            for BN in bn_range  #
            for BK in bk_range  #
            for s in tma_persistent_s_range  #
            for w in [4, 8]  #
            for SUBTILE in [True, False]  #
        ]


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(
    a_desc,
    b_desc,
    c_desc,  #
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    DTYPE: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    dtype = DTYPE
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Enable warp specialization to leverage async warp scheduling in the GPU.
    # FIXME: This only works on Blackwell right now. On older GPUs, this will
    # use software pipelining.
    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            if TRANSPOSE_A:
                a = a_desc.load([offs_k, offs_am])
                arg1 = a.T
            else:
                a = a_desc.load([offs_am, offs_k])
                arg1 = a
            if TRANSPOSE_B:
                b = b_desc.load([offs_bn, offs_k])
                arg2 = b.T
            else:
                b = b_desc.load([offs_k, offs_bn])
                arg2 = b
            accumulator = tl.dot(arg1, arg2, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)


def blackwell_matmul_tma_persistent(a, b, warp_specialize: bool):
    # High-Level Options for B's layout
    # 1. (K, N) contiguous in N
    # 2. (K, N) contiguous in K
    # 3. (N, K) contiguous in N
    # 4. (N, K) contiguous in K
    # In practice, since you always load in the contiguous dimension
    # there are actually only 2 options
    # 1. Load in the K stride 1 (2 and 4)
    # 2. Load in the N stride 1 (1 and 3)
    transpose_a = a.stride()[-1] != 1
    transpose_b = (a.shape[1] != b.shape[1] and b.stride()[-1] != 1) or (
        a.shape[1] == b.shape[1] and b.stride()[-1] == 1
    )
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    if a.shape[1] != b.shape[1]:
        K, N = b.shape
    else:
        N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_strides = [a.stride()[1] if transpose_a else a.stride()[0], 1]
    check_tma_alignment(a_strides, (torch.finfo(a.dtype).bits + 7) // 8)
    if transpose_a:
        a_desc = TensorDescriptor(a, [K, M], a_strides, dummy_block)
    else:
        a_desc = TensorDescriptor(a, [M, K], a_strides, dummy_block)
    b_strides = [b.stride()[1] if b.stride()[0] == 1 else b.stride()[0], 1]
    check_tma_alignment(b_strides, (torch.finfo(b.dtype).bits + 7) // 8)
    if transpose_b:
        b_desc = TensorDescriptor(b, [N, K], b_strides, dummy_block)
    else:
        b_desc = TensorDescriptor(b, [K, N], b_strides, dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            ),
        )

    matmul_kernel_tma_persistent[grid](
        a_desc,
        b_desc,
        c_desc,  #
        M,
        N,
        K,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        DTYPE=torch_dtype_to_triton_dtype(dtype),  #
        TRANSPOSE_A=transpose_a,
        TRANSPOSE_B=transpose_b,
    )
    return c


def prune_invalid_configs(configs, named_args, **kwargs):
    FLATTEN = kwargs["FLATTEN"]
    # Filter out configs where EPILOGUE_SUBTILE is true and HOPPER is true
    return [
        conf
        for conf in configs
        if not (conf.kwargs.get("EPILOGUE_SUBTILE", True) and FLATTEN is False)
    ]


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K", "WARP_SPECIALIZE", "FLATTEN"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    a_stride,
    b_stride,
    c_stride,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    FLATTEN: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    if TRANSPOSE_A:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[K, M],
            strides=[a_stride, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_M],
        )
    else:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[a_stride, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
    if TRANSPOSE_B:
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[N, K],
            strides=[b_stride, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
    else:
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[K, N],
            strides=[b_stride, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[c_stride, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            if TRANSPOSE_A:
                a = a_desc.load([offs_k, offs_am])
                arg1 = a.T
            else:
                a = a_desc.load([offs_am, offs_k])
                arg1 = a
            if TRANSPOSE_B:
                b = b_desc.load([offs_bn, offs_k])
                arg2 = b.T
            else:
                b = b_desc.load([offs_k, offs_bn])
                arg2 = b
            accumulator = tl.dot(arg1, arg2, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


def blackwell_matmul_descriptor_persistent(a, b, warp_specialize: bool):
    # High-Level Options for B's layout
    # 1. (K, N) contiguous in N
    # 2. (K, N) contiguous in K
    # 3. (N, K) contiguous in N
    # 4. (N, K) contiguous in K
    # In practice, since you always load in the contiguous dimension
    # there are actually only 2 options
    # 1. Load in the K stride 1 (2 and 4)
    # 2. Load in the N stride 1 (1 and 3)
    transpose_a = a.stride()[-1] != 1
    transpose_b = (a.shape[1] != b.shape[1] and b.stride()[-1] != 1) or (
        a.shape[1] == b.shape[1] and b.stride()[-1] == 1
    )
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    if a.shape[1] != b.shape[1]:
        K, N = b.shape
    else:
        N, K = b.shape

    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    if a.stride()[0] == 1:
        a_stride = a.stride()[1]
    else:
        a_stride = a.stride()[0]
    if b.stride()[0] == 1:
        b_stride = b.stride()[1]
    else:
        b_stride = b.stride()[0]
    if c.stride()[0] == 1:
        c_stride = c.stride()[1]
    else:
        c_stride = c.stride()[0]
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    matmul_kernel_descriptor_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a_stride,  #
        b_stride,  #
        c_stride,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        # Note: This assumes blackwell.
        FLATTEN=True,
        TRANSPOSE_A=transpose_a,
        TRANSPOSE_B=transpose_b,
    )
    return c
