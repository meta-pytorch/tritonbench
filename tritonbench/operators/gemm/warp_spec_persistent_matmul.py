"""
This is copied from Triton matmul tutorial 9 and reduced to just the
persistent matmul kernel on blackwell with/without warpspec.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from tritonbench.utils.env_utils import IS_BLACKWELL, IS_HOPPER, is_tile_enabled

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
    # Hopper has smaller WGMMA tiles and tighter register budgets than
    # Blackwell, so BLOCK_M=256 is rarely useful; trim the M range there.
    bm_range = [64, 128] if IS_HOPPER else [128, 256]
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


def _use_meta_ws():
    """Check if the Meta warp specialization pipeline is enabled."""
    try:
        return triton.knobs.nvidia.use_meta_ws
    except AttributeError:
        return False


def _pingpong_options():
    """``pingpongAutoWS`` is a triton.Config autotune kwarg in the beta
    branch (see test_tutorial09_warp_specialization.py for canonical
    usage). It schedules producer/consumer warps in a ping-pong pattern
    and is only useful on Hopper with the meta WS pipeline."""
    if IS_HOPPER and _use_meta_ws():
        return [True, False]
    return [None]


def _prune_warp_specialize_configs(configs, named_args, **kwargs):
    """When warp specialization is enabled with the Meta WS pipeline,
    only num_warps=4 is supported."""
    ws = kwargs.get("WARP_SPECIALIZE", False)
    if not ws or not _use_meta_ws():
        return configs
    return [c for c in configs if c.num_warps == 4]


def matmul_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
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
    c = nargs.get("c_desc_or_ptr")
    if isinstance(c, TensorDescriptor):
        c.block_shape = [BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]


@triton.autotune(
    configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
    prune_configs_by={"early_config_prune": _prune_warp_specialize_configs},
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
    # Determine if b needs transposed TMA descriptor.
    # TMA requires the last descriptor dimension to be contiguous (stride 1).
    # If b's first physical dim is contiguous (column-major), we need transpose_b=True
    # so the descriptor shape becomes [N, K] with K (contiguous) as the last dim.
    if a.shape[1] != b.shape[1]:
        # K != N: can infer layout from shape
        transpose_b = b.stride()[-1] != 1
    else:
        # K == N: shapes are ambiguous, use stride to determine layout
        transpose_b = b.stride()[0] == 1 and b.stride()[-1] != 1
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
            for SUBTILE in [1, 2, 4]:
                for FLATTEN in [True, False]:
                    kwargs = {
                        "EPILOGUE_SUBTILE": SUBTILE,
                        **new_kwargs,
                    }
                    kwargs["FLATTEN"] = FLATTEN
                    configs.append(
                        triton.Config(
                            kwargs,
                            pre_hook=pre_hook,
                            num_stages=config.num_stages,
                            num_ctas=config.num_ctas,
                            num_warps=config.num_warps,
                        )
                    )
        return configs
    else:
        early_tma_store_lowering = [1] if _use_meta_ws() else [None]

        def _make_config(
            BM, BN, BK, s, w, SUBTILE, FLATTEN, DP, pp, store_lowering, maxReg=None
        ):
            extras = {}
            if pp is not None:
                extras["pingpongAutoWS"] = pp
            if store_lowering is not None:
                extras["early_tma_store_lowering"] = store_lowering
            if maxReg is not None:
                extras["maxRegAutoWS"] = maxReg
            return triton.Config(
                {
                    "BLOCK_SIZE_M": BM,
                    "BLOCK_SIZE_N": BN,
                    "BLOCK_SIZE_K": BK,
                    "GROUP_SIZE_M": 8,
                    "EPILOGUE_SUBTILE": SUBTILE,
                    "FLATTEN": FLATTEN,
                    "DATA_PARTITION_FACTOR": DP,
                },
                num_stages=s,
                num_warps=w,
                pre_hook=pre_hook,
                **extras,
            )

        return [
            _make_config(BM, BN, BK, s, w, SUBTILE, FLATTEN, DP, pp, sl)
            for BM in bm_range  #
            for BN in bn_range  #
            for BK in bk_range  #
            for s in tma_persistent_s_range  #
            for w in [4, 8]  #
            for SUBTILE in [1, 2, 4]  #
            for FLATTEN in [True, False]  #
            for DP in [1, 2]  #
            for pp in _pingpong_options()  #
            for sl in early_tma_store_lowering  #
        ]


def _prune_tma_persistent_configs(configs, named_args, **kwargs):
    """Prune configs for the TMA persistent kernel.

    FLATTEN rules. FLATTEN=True is only supported on Blackwell with the
    OAI Triton WS path.
    - Hopper: require FLATTEN=False (regardless of WS / meta WS)
    - Blackwell + WARP_SPECIALIZE=False: require FLATTEN=False
    - Blackwell + WARP_SPECIALIZE=True + meta WS: require FLATTEN=False
    - Blackwell + WARP_SPECIALIZE=True + OAI Triton WS: require FLATTEN=True

    DATA_PARTITION_FACTOR rules. DP=2 is only valid on the meta WS path,
    and the partitioned M tile must match the arch's largest usable
    BLOCK_M (128 on Hopper, 256 on Blackwell).
    """
    ws = kwargs.get("WARP_SPECIALIZE", False)
    flatten_required = ws and not _use_meta_ws() and not IS_HOPPER
    kept = []
    for c in configs:
        subtile = c.kwargs.get("EPILOGUE_SUBTILE", 1)
        if IS_HOPPER and subtile != 1:
            continue
        if c.kwargs.get("BLOCK_SIZE_N", 1) % subtile != 0:
            continue
        flatten = c.kwargs.get("FLATTEN", False)
        if flatten != flatten_required:
            continue
        data_partition_factor = c.kwargs.get("DATA_PARTITION_FACTOR", 1)
        if data_partition_factor != 1:
            if not _use_meta_ws():
                continue
            block_m = c.kwargs.get("BLOCK_SIZE_M", 1)
            # Data partitioning needs the largest M tile that still fits in
            # registers: BLOCK_M=128 on Hopper, BLOCK_M=256 on Blackwell.
            required_block_m = 128 if IS_HOPPER else 256
            if block_m != required_block_m:
                continue
        kept.append(c)
    return _prune_warp_specialize_configs(kept, named_args, **kwargs)


@triton.jit
def _matmul_tma_persistent_loop_body(
    tile_id,
    tile_id_c,
    a_desc,
    b_desc,
    c_desc_or_ptr,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    num_pid_in_group,
    num_pid_m,
    k_tiles,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    DTYPE: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    TMA_STORE: tl.constexpr,
):
    dtype = DTYPE
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

    if TMA_STORE:
        # Epilogue subtiling is a technique to break our computation and stores
        # into multiple pieces. By subtiling we can reduce shared memory consumption
        # by the epilogue and instead use that memory to increase our stage count.
        if EPILOGUE_SUBTILE == 1:
            accumulator = accumulator.to(dtype)
            c_desc_or_ptr.store([offs_am_c, offs_bn_c], accumulator)
        elif EPILOGUE_SUBTILE == 2:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc_or_ptr.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc_or_ptr.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        elif EPILOGUE_SUBTILE == 4:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc_lo, acc_hi = tl.split(acc)
            acc_lo = tl.reshape(acc_lo, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4))
            acc_lo = tl.permute(acc_lo, (0, 2, 1))
            acc0, acc1 = tl.split(acc_lo)
            acc_hi = tl.reshape(acc_hi, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4))
            acc_hi = tl.permute(acc_hi, (0, 2, 1))
            acc2, acc3 = tl.split(acc_hi)
            c0 = acc0.to(dtype)
            c_desc_or_ptr.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc_or_ptr.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 4], c1)
            c2 = acc2.to(dtype)
            c_desc_or_ptr.store([offs_am_c, offs_bn_c + 2 * BLOCK_SIZE_N // 4], c2)
            c3 = acc3.to(dtype)
            c_desc_or_ptr.store([offs_am_c, offs_bn_c + 3 * BLOCK_SIZE_N // 4], c3)
        else:
            tl.static_assert(False, "Unsupported EPILOGUE_SUBTILE value")
    else:
        accumulator = accumulator.to(dtype)
        offs_cm = offs_am_c + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = offs_bn_c + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_desc_or_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
    return tile_id_c


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
    prune_configs_by={"early_config_prune": _prune_tma_persistent_configs},
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(
    a_desc,
    b_desc,
    c_desc_or_ptr,  #
    M,
    N,
    K,  #
    stride_cm,
    stride_cn,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    FLATTEN: tl.constexpr,  #
    DTYPE: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    TMA_STORE: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,  #
    USE_META_WS: tl.constexpr,  #
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    if USE_META_WS:
        for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=FLATTEN,
            warp_specialize=WARP_SPECIALIZE,
            data_partition_factor=DATA_PARTITION_FACTOR,
            smem_alloc_algo=1,
            separate_epilogue_store=True,
        ):
            tile_id_c = _matmul_tma_persistent_loop_body(
                tile_id,
                tile_id_c,
                a_desc,
                b_desc,
                c_desc_or_ptr,
                M,
                N,
                K,
                stride_cm,
                stride_cn,
                num_pid_in_group,
                num_pid_m,
                k_tiles,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                EPILOGUE_SUBTILE,
                NUM_SMS,
                DTYPE,
                TRANSPOSE_A,
                TRANSPOSE_B,
                TMA_STORE,
            )
    else:
        for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=FLATTEN,
            warp_specialize=WARP_SPECIALIZE,
        ):
            tile_id_c = _matmul_tma_persistent_loop_body(
                tile_id,
                tile_id_c,
                a_desc,
                b_desc,
                c_desc_or_ptr,
                M,
                N,
                K,
                stride_cm,
                stride_cn,
                num_pid_in_group,
                num_pid_m,
                k_tiles,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                EPILOGUE_SUBTILE,
                NUM_SMS,
                DTYPE,
                TRANSPOSE_A,
                TRANSPOSE_B,
                TMA_STORE,
            )


def blackwell_matmul_tma_persistent(
    a, b, warp_specialize: bool, tma_store: bool = True
):
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
    # Determine if b needs transposed TMA descriptor.
    # TMA requires the last descriptor dimension to be contiguous (stride 1).
    # If b's first physical dim is contiguous (column-major), we need transpose_b=True
    # so the descriptor shape becomes [N, K] with K (contiguous) as the last dim.
    if a.shape[1] != b.shape[1]:
        # K != N: can infer layout from shape
        transpose_b = b.stride()[-1] != 1
    else:
        # K == N: shapes are ambiguous, use stride to determine layout
        transpose_b = b.stride()[0] == 1 and b.stride()[-1] != 1
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
    if tma_store:
        c_desc_or_ptr = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
    else:
        c_desc_or_ptr = c

    def grid(META):
        nonlocal a_desc, b_desc, c_desc_or_ptr
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
        c_desc_or_ptr,  #
        M,
        N,
        K,  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        DTYPE=torch_dtype_to_triton_dtype(dtype),  #
        TRANSPOSE_A=transpose_a,
        TRANSPOSE_B=transpose_b,
        TMA_STORE=tma_store,
        USE_META_WS=_use_meta_ws(),
    )
    return c


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K", "WARP_SPECIALIZE", "FLATTEN"],
    prune_configs_by={"early_config_prune": _prune_tma_persistent_configs},
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
            BLOCK_SIZE_N // EPILOGUE_SUBTILE,
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

        if EPILOGUE_SUBTILE == 1:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)
        elif EPILOGUE_SUBTILE == 2:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        elif EPILOGUE_SUBTILE == 4:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc_lo, acc_hi = tl.split(acc)
            acc_lo = tl.reshape(acc_lo, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4))
            acc_lo = tl.permute(acc_lo, (0, 2, 1))
            acc0, acc1 = tl.split(acc_lo)
            acc_hi = tl.reshape(acc_hi, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4))
            acc_hi = tl.permute(acc_hi, (0, 2, 1))
            acc2, acc3 = tl.split(acc_hi)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 4], c1)
            c2 = acc2.to(dtype)
            c_desc.store([offs_cm, offs_cn + 2 * BLOCK_SIZE_N // 4], c2)
            c3 = acc3.to(dtype)
            c_desc.store([offs_cm, offs_cn + 3 * BLOCK_SIZE_N // 4], c3)
        else:
            tl.static_assert(False, "Unsupported EPILOGUE_SUBTILE value")


# ============================================================================
# Split-K Persistent Matmul with Warp Specialization
#
# Decomposes the K dimension across SPLIT_K groups of CTAs, each computing
# a partial sum into a (SPLIT_K * M, N) workspace. A separate reduce kernel
# folds the partial sums into the final output in fp32. This is beneficial
# for undersaturated GEMMs where M*N tiles do not fill all SMs but K is
# large enough to split.
# ============================================================================


def splitk_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    if nargs["TRANSPOSE_A"]:
        nargs["a_desc"].block_shape = [BLOCK_K, BLOCK_M]
    else:
        nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    if nargs["TRANSPOSE_B"]:
        nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    else:
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    nargs["ws_desc"].block_shape = [BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]


def matmul_splitk_get_configs(pre_hook=None):
    base_extra = {}
    if _use_meta_ws():
        base_extra["early_tma_store_lowering"] = 1

    splitk_bm_range = [64, 128] if IS_HOPPER else [128]

    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
                "EPILOGUE_SUBTILE": SUBTILE,
                "FLATTEN": False,
            },
            num_stages=s,
            num_warps=4,
            pre_hook=pre_hook,
            **base_extra,
        )
        for BM in splitk_bm_range
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [2, 3]
        for SUBTILE in [1, 2, 4]
        if BN % SUBTILE == 0
    ]


@triton.autotune(
    configs=matmul_splitk_get_configs(pre_hook=splitk_set_block_size_hook),
    key=["M", "N", "K", "SPLIT_K"],
    prune_configs_by={"early_config_prune": _prune_warp_specialize_configs},
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent_splitk(
    a_desc,
    b_desc,
    ws_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FLATTEN: tl.constexpr,
    DTYPE: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    USE_META_WS: tl.constexpr,
):
    dtype = DTYPE
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_mn_tiles = num_pid_m * num_pid_n
    num_tiles = num_mn_tiles * SPLIT_K
    k_per_split = tl.cdiv(k_tiles, SPLIT_K)

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    if USE_META_WS:
        loop_iter = tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=FLATTEN,
            warp_specialize=True,
            disallow_acc_multi_buffer=True,
        )
    else:
        loop_iter = tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=FLATTEN,
            warp_specialize=True,
        )

    for tile_id in loop_iter:
        split_id = tile_id // num_mn_tiles
        mn_tile_id = tile_id % num_mn_tiles
        k_start = split_id * k_per_split
        k_end = tl.minimum(k_start + k_per_split, k_tiles)

        pid_m, pid_n = _compute_pid(
            mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_start, k_end):
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
        split_id_c = tile_id_c // num_mn_tiles
        mn_tile_id_c = tile_id_c % num_mn_tiles
        pid_m, pid_n = _compute_pid(
            mn_tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N
        row_base = split_id_c * M

        if EPILOGUE_SUBTILE == 1:
            c = accumulator.to(dtype)
            ws_desc.store([row_base + offs_am_c, offs_bn_c], c)
        elif EPILOGUE_SUBTILE == 2:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            ws_desc.store([row_base + offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            ws_desc.store([row_base + offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        elif EPILOGUE_SUBTILE == 4:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc_lo, acc_hi = tl.split(acc)
            acc_lo = tl.reshape(acc_lo, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4))
            acc_lo = tl.permute(acc_lo, (0, 2, 1))
            acc0, acc1 = tl.split(acc_lo)
            acc_hi = tl.reshape(acc_hi, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 4))
            acc_hi = tl.permute(acc_hi, (0, 2, 1))
            acc2, acc3 = tl.split(acc_hi)
            c0 = acc0.to(dtype)
            ws_desc.store([row_base + offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            ws_desc.store([row_base + offs_am_c, offs_bn_c + BLOCK_SIZE_N // 4], c1)
            c2 = acc2.to(dtype)
            ws_desc.store([row_base + offs_am_c, offs_bn_c + 2 * BLOCK_SIZE_N // 4], c2)
            c3 = acc3.to(dtype)
            ws_desc.store([row_base + offs_am_c, offs_bn_c + 3 * BLOCK_SIZE_N // 4], c3)
        else:
            tl.static_assert(False, "Unsupported EPILOGUE_SUBTILE value")


@triton.jit
def _reduce_k_kernel(
    workspace_ptr,
    c_ptr,
    M,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        partial = tl.load(workspace_ptr + base + s * M * N, mask=mask, other=0.0)
        acc += partial.to(tl.float32)
    tl.store(c_ptr + base, acc.to(OUTPUT_DTYPE), mask=mask)


def blackwell_matmul_tma_persistent_splitk(a, b, split_k=None):
    """Persistent TMA matmul with warp specialization + Split-K.

    Targets undersaturated GEMMs where M*N tiles do not fill all SMs but K
    is large. Each of SPLIT_K groups computes a partial sum; a reduce kernel
    folds them into the final output in fp32.
    """
    transpose_a = a.stride()[-1] != 1
    if a.shape[1] != b.shape[1]:
        transpose_b = b.stride()[-1] != 1
    else:
        transpose_b = b.stride()[0] == 1 and b.stride()[-1] != 1
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    if a.shape[1] != b.shape[1]:
        K, N = b.shape
    else:
        N, K = b.shape
    dtype = a.dtype

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    if split_k is None:
        mn_tiles = triton.cdiv(M, 128) * triton.cdiv(N, 128)
        k_tiles = triton.cdiv(K, 64)
        split_k = 1
        for sk in [4, 2, 8]:
            if k_tiles >= sk * 4 and mn_tiles * sk <= NUM_SMS * 2:
                split_k = sk
                break
        if split_k == 1:
            split_k = 2

    k_tiles = triton.cdiv(K, 64)
    k_per_split = triton.cdiv(k_tiles, split_k)
    if k_per_split * (split_k - 1) >= k_tiles:
        split_k = 2

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    workspace = torch.empty((split_k * M, N), device=a.device, dtype=dtype)

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
    ws_desc = TensorDescriptor(
        workspace, workspace.shape, workspace.stride(), dummy_block
    )

    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"])
                * split_k,
            ),
        )

    matmul_kernel_tma_persistent_splitk[grid](
        a_desc,
        b_desc,
        ws_desc,
        M,
        N,
        K,
        NUM_SMS=NUM_SMS,
        SPLIT_K=split_k,
        DTYPE=torch_dtype_to_triton_dtype(dtype),
        TRANSPOSE_A=transpose_a,
        TRANSPOSE_B=transpose_b,
        USE_META_WS=_use_meta_ws(),
    )

    REDUCE_BM, REDUCE_BN = 32, 32
    reduce_grid = (triton.cdiv(M, REDUCE_BM), triton.cdiv(N, REDUCE_BN))
    _reduce_k_kernel[reduce_grid](
        workspace,
        c,
        M,
        N,
        SPLIT_K=split_k,
        BLOCK_SIZE_M=REDUCE_BM,
        BLOCK_SIZE_N=REDUCE_BN,
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(dtype),
        num_warps=4,
    )

    return c


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
    # Determine if b needs transposed TMA descriptor.
    # TMA requires the last descriptor dimension to be contiguous (stride 1).
    # If b's first physical dim is contiguous (column-major), we need transpose_b=True
    # so the descriptor shape becomes [N, K] with K (contiguous) as the last dim.
    if a.shape[1] != b.shape[1]:
        # K != N: can infer layout from shape
        transpose_b = b.stride()[-1] != 1
    else:
        # K == N: shapes are ambiguous, use stride to determine layout
        transpose_b = b.stride()[0] == 1 and b.stride()[-1] != 1
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
        TRANSPOSE_A=transpose_a,
        TRANSPOSE_B=transpose_b,
    )
    return c
