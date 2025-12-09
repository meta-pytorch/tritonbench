# TLX GEMM kernel optimized for Blackwell Warp Specialization
import math

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
                "NUM_SMEM_BUFFERS": s,
                "NUM_TMEM_BUFFERS": t,
                "EPILOGUE_SUBTILE": subtile,
                "PAIR_CTA": pairCTA,
                "PERSISTENT": persistent,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4, 5, 6, 7]
        for t in [2, 3]
        for subtile in [1, 2, 4, 8]
        for pairCTA in [True, False]
        for persistent in [True, False]
    ]


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    if nargs.get("PAIR_CTA", False):
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N // 2]
    else:
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def preprocess_configs(configs, named_args, **kwargs):
    for conf in configs:
        M = named_args["M"]
        N = named_args["N"]
        BLOCK_M = conf.kwargs["BLOCK_SIZE_M"]
        BLOCK_N = conf.kwargs["BLOCK_SIZE_N"]

        num_tiles_m = math.ceil(M / BLOCK_M)
        num_tiles_n = math.ceil(N / BLOCK_N)
        # checking num_tiles_m should be sufficent in this case, but adding num_tiles_n for clarity
        pair_cta_compatible = (
            num_tiles_m % 2 == 0 and (num_tiles_m * num_tiles_n) % 2 == 0
        )
        if not pair_cta_compatible:
            # fall back to non-pair CTA mode
            conf.kwargs["PAIR_CTA"] = False

    return configs


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@triton.jit
def _compute_grid_info(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M):
    """Compute common grid information used across async tasks."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    return start_pid, num_pid_m, num_pid_n, num_pid_in_group, num_tiles, k_tiles


@triton.jit
def _process_tile_epilogue_inner(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    GROUP_SIZE_M,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    EPILOGUE_SUBTILE,
    c_desc,
    tmem_buffers,
    tmem_full_bars,
    tmem_empty_bars,
    cur_tmem_buf,
    tmem_read_phase,
    PERSISTENT,
):
    """Process epilogue for a single tile."""
    pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    tlx.barrier_wait(tmem_full_bars[cur_tmem_buf], tmem_read_phase)

    # load the result from TMEM to registers
    acc_tmem = tmem_buffers[cur_tmem_buf]
    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    for slice_id in tl.static_range(EPILOGUE_SUBTILE):
        acc_tmem_subslice = tlx.local_slice(
            acc_tmem,
            [0, slice_id * slice_size],
            [BLOCK_SIZE_M, slice_size],
        )
        result = tlx.local_load(acc_tmem_subslice)
        c = result.to(tl.float16)
        c_desc.store([offs_am, offs_bn + slice_id * slice_size], c)

    # Signal MMA consumer if in persistent mode
    if PERSISTENT:
        tlx.barrier_arrive(tmem_empty_bars[cur_tmem_buf], 1)


@triton.jit
def _process_tile_mma_inner(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    GROUP_SIZE_M,
    k_tiles,
    NUM_SMEM_BUFFERS,
    buffers_A,
    buffers_B,
    tmem_buffers,
    smem_full_bars,
    smem_empty_bars,
    tmem_full_bars,
    cur_tmem_buf,
    tmem_empty_bars,
    tmem_write_phase,
    smem_accum_cnt,
    PAIR_CTA,
    cta_bars,
    pred_cta0,
    PERSISTENT,
):
    """Process MMA for a single tile. Returns updated smem_accum_cnt."""
    pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)

    if PERSISTENT:
        # wait epilogue consumer to be done with the buffer before reusing it
        # Note: we start with phase 1 since epilogue starts with phase 0
        tlx.barrier_wait(tmem_empty_bars[cur_tmem_buf], tmem_write_phase ^ 1)

    # now iterate along K to compute result for the block
    for k in range(0, k_tiles):
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

        # wait for current phase(round) of load for this buf
        tlx.barrier_wait(smem_full_bars[buf], phase)
        # CTA0 waits for CTA0 and CTA1 to finish loading A and B before issuing dot op
        if PAIR_CTA:
            cta_bar = tlx.remote_view(cta_bars[buf], 0)
            tlx.barrier_arrive(cta_bar, 1)
            tlx.barrier_wait(cta_bar, phase=phase, pred=pred_cta0)

        # buffer is now ready with loaded data, tlx.async_dot will signal `mBarrier` when done
        tlx.async_dot(
            buffers_A[buf],
            buffers_B[buf],
            tmem_buffers[cur_tmem_buf],
            use_acc=k > 0,
            mBarriers=[smem_empty_bars[buf]],
            two_ctas=PAIR_CTA,
            out_dtype=tl.float32,
        )
        smem_accum_cnt += 1

    # Wait for last MMA to complete before signaling epilogue
    last_buf, last_phase = _get_bufidx_phase(smem_accum_cnt - 1, NUM_SMEM_BUFFERS)
    tlx.barrier_wait(smem_empty_bars[last_buf], last_phase)

    # Done filling this buffer, signal epilogue consumer
    tlx.barrier_arrive(tmem_full_bars[cur_tmem_buf], 1)

    return smem_accum_cnt


@triton.jit
def _process_tile_producer_inner(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    GROUP_SIZE_M,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    k_tiles,
    NUM_SMEM_BUFFERS,
    a_desc,
    b_desc,
    buffers_A,
    buffers_B,
    smem_full_bars,
    smem_empty_bars,
    smem_accum_cnt,
    PAIR_CTA,
    cluster_cta_rank,
):
    """Process TMA loads for a single tile. Returns updated smem_accum_cnt."""
    pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    offs_am = pid_m * BLOCK_SIZE_M
    # split B into two parts so each CTA has different offset
    if PAIR_CTA:
        offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // 2)
    else:
        offs_bn = pid_n * BLOCK_SIZE_N

    for k in range(0, k_tiles):
        buf, load_phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        # wait for previous phase(round) of dot for this buf
        tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)
        # buffer is now ready to be used again
        offs_k = k * BLOCK_SIZE_K
        if PAIR_CTA:
            tlx.barrier_expect_bytes(
                smem_full_bars[buf],
                2 * (BLOCK_SIZE_M + BLOCK_SIZE_N // 2) * BLOCK_SIZE_K,
            )  # float16
        else:
            tlx.barrier_expect_bytes(
                smem_full_bars[buf],
                2 * (BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K,
            )  # float16
        tlx.async_descriptor_load(
            a_desc, buffers_A[buf], [offs_am, offs_k], smem_full_bars[buf]
        )
        tlx.async_descriptor_load(
            b_desc, buffers_B[buf], [offs_k, offs_bn], smem_full_bars[buf]
        )
        smem_accum_cnt += 1

    return smem_accum_cnt


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": preprocess_configs},
)
@triton.jit
def matmul_kernel_tma_ws_blackwell(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMEM_BUFFERS: tl.constexpr,  #
    NUM_TMEM_BUFFERS: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    PAIR_CTA: tl.constexpr,  #
    PERSISTENT: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
):
    # allocate NUM_SMEM_BUFFERS buffers
    buffers_A = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, NUM_SMEM_BUFFERS
    )
    # In pair CTA mode, each cta only needs to load half of B.
    if PAIR_CTA:
        buffers_B = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_SIZE_N // 2), tl.float16, NUM_SMEM_BUFFERS
        )
    else:
        buffers_B = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_SIZE_N), tl.float16, NUM_SMEM_BUFFERS
        )
    # use multiple TMEM buffers to overlap MMA and epilogue (only in persistent mode)
    tmem_buffers = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS,
        tlx.storage_kind.tmem,
    )

    # CTA pairs are placed along M dim
    if PAIR_CTA:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        cta_bars = tlx.alloc_barriers(
            num_barriers=NUM_SMEM_BUFFERS, arrive_count=2
        )  # CTA0 waits for CTA1's data before mma
    else:
        cluster_cta_rank = 0
        pred_cta0 = False
        cta_bars = None

    # allocate barriers
    smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    if PERSISTENT:
        tmem_empty_bars = tlx.alloc_barriers(
            num_barriers=NUM_TMEM_BUFFERS, arrive_count=1
        )
    else:
        tmem_empty_bars = None

    with tlx.async_tasks():
        with tlx.async_task("default"):  # epilogue consumer
            start_pid, num_pid_m, num_pid_n, num_pid_in_group, num_tiles, k_tiles = (
                _compute_grid_info(
                    M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
                )
            )

            if PERSISTENT:
                # Persistent mode: process multiple tiles
                tmem_accum_cnt = 0
                for tile_id in range(start_pid, num_tiles, NUM_SMS):
                    cur_tmem_buf, tmem_read_phase = _get_bufidx_phase(
                        tmem_accum_cnt, NUM_TMEM_BUFFERS
                    )
                    _process_tile_epilogue_inner(
                        tile_id,
                        num_pid_in_group,
                        num_pid_m,
                        GROUP_SIZE_M,
                        BLOCK_SIZE_M,
                        BLOCK_SIZE_N,
                        EPILOGUE_SUBTILE,
                        c_desc,
                        tmem_buffers,
                        tmem_full_bars,
                        tmem_empty_bars,
                        cur_tmem_buf,
                        tmem_read_phase,
                        PERSISTENT,
                    )
                    tmem_accum_cnt += 1
            else:
                # Non-persistent mode: process single tile
                tile_id = start_pid
                cur_tmem_buf = 0
                tmem_read_phase = 0
                _process_tile_epilogue_inner(
                    tile_id,
                    num_pid_in_group,
                    num_pid_m,
                    GROUP_SIZE_M,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    EPILOGUE_SUBTILE,
                    c_desc,
                    tmem_buffers,
                    tmem_full_bars,
                    tmem_empty_bars,
                    cur_tmem_buf,
                    tmem_read_phase,
                    PERSISTENT,
                )

        with tlx.async_task(num_warps=1, num_regs=24):  # MMA consumer
            start_pid, num_pid_m, num_pid_n, num_pid_in_group, num_tiles, k_tiles = (
                _compute_grid_info(
                    M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
                )
            )

            if PERSISTENT:
                # Persistent mode: process multiple tiles
                tmem_accum_cnt = 0
                smem_accum_cnt = 0

                for tile_id in range(start_pid, num_tiles, NUM_SMS):
                    cur_tmem_buf, tmem_write_phase = _get_bufidx_phase(
                        tmem_accum_cnt, NUM_TMEM_BUFFERS
                    )
                    smem_accum_cnt = _process_tile_mma_inner(
                        tile_id,
                        num_pid_in_group,
                        num_pid_m,
                        GROUP_SIZE_M,
                        k_tiles,
                        NUM_SMEM_BUFFERS,
                        buffers_A,
                        buffers_B,
                        tmem_buffers,
                        smem_full_bars,
                        smem_empty_bars,
                        tmem_full_bars,
                        cur_tmem_buf,
                        tmem_empty_bars,
                        tmem_write_phase,
                        smem_accum_cnt,
                        PAIR_CTA,
                        cta_bars,
                        pred_cta0,
                        PERSISTENT,
                    )
                    tmem_accum_cnt += 1
            else:
                # Non-persistent mode: process single tile
                tile_id = start_pid
                smem_accum_cnt = 0
                cur_tmem_buf = 0
                tmem_write_phase = 0  # Not used in non-persistent mode
                _process_tile_mma_inner(
                    tile_id,
                    num_pid_in_group,
                    num_pid_m,
                    GROUP_SIZE_M,
                    k_tiles,
                    NUM_SMEM_BUFFERS,
                    buffers_A,
                    buffers_B,
                    tmem_buffers,
                    smem_full_bars,
                    smem_empty_bars,
                    tmem_full_bars,
                    cur_tmem_buf,
                    tmem_empty_bars,
                    tmem_write_phase,
                    smem_accum_cnt,
                    PAIR_CTA,
                    cta_bars,
                    pred_cta0,
                    PERSISTENT,
                )

        with tlx.async_task(num_warps=1, num_regs=24):  # producer, TMA load
            start_pid, num_pid_m, num_pid_n, num_pid_in_group, num_tiles, k_tiles = (
                _compute_grid_info(
                    M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
                )
            )

            if PERSISTENT:
                # Persistent mode: process multiple tiles
                smem_accum_cnt = 0
                for tile_id in range(start_pid, num_tiles, NUM_SMS):
                    smem_accum_cnt = _process_tile_producer_inner(
                        tile_id,
                        num_pid_in_group,
                        num_pid_m,
                        GROUP_SIZE_M,
                        BLOCK_SIZE_M,
                        BLOCK_SIZE_N,
                        BLOCK_SIZE_K,
                        k_tiles,
                        NUM_SMEM_BUFFERS,
                        a_desc,
                        b_desc,
                        buffers_A,
                        buffers_B,
                        smem_full_bars,
                        smem_empty_bars,
                        smem_accum_cnt,
                        PAIR_CTA,
                        cluster_cta_rank,
                    )
            else:
                # Non-persistent mode: process single tile
                tile_id = start_pid
                smem_accum_cnt = 0
                _process_tile_producer_inner(
                    tile_id,
                    num_pid_in_group,
                    num_pid_m,
                    GROUP_SIZE_M,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    k_tiles,
                    NUM_SMEM_BUFFERS,
                    a_desc,
                    b_desc,
                    buffers_A,
                    buffers_B,
                    smem_full_bars,
                    smem_empty_bars,
                    smem_accum_cnt,
                    PAIR_CTA,
                    cluster_cta_rank,
                )


def tlx_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # Grid function that adapts based on PERSISTENT parameter from autotune
    # For persistent mode: launch fewer CTAs (min of NUM_SMS or total tiles)
    # For non-persistent mode: launch one CTA per tile
    def grid(META):
        total_tiles = triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]
        )
        if META.get("PERSISTENT", False):
            return (min(NUM_SMS, total_tiles),)
        else:
            return (total_tiles,)

    matmul_kernel_tma_ws_blackwell[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        NUM_SMS=NUM_SMS,
    )
    return c
