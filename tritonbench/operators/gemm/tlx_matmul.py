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
                "NUM_MMA_GROUPS": m,
                "EPILOGUE_SUBTILE": subtile,
                "PAIR_CTA": pairCTA,
                "PERSISTENT": persistent,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(2, 1, 1) if pairCTA else None,
        )
        for BM in [128, 256]
        for BN in [128, 256, 512]
        for BK in [64, 128]
        for s in [2, 3, 4, 5, 6, 7]
        for t in [1, 2, 3]
        for m in [1, 2]
        for subtile in [1, 2, 4, 8]
        # for pairCTA in [True, False]
        for pairCTA in [False]
        for persistent in [True, False]
    ]


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    NUM_MMA_GROUPS = nargs.get("NUM_MMA_GROUPS", 1)
    nargs["a_desc"].block_shape = [BLOCK_M // NUM_MMA_GROUPS, BLOCK_K]
    if nargs.get("PAIR_CTA", False):
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N // 2]
    else:
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    nargs["c_desc"].block_shape = [
        BLOCK_M // NUM_MMA_GROUPS,
        BLOCK_N // EPILOGUE_SUBTILE,
    ]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def preprocess_configs(configs, named_args, **kwargs):
    # Blackwell B200A resource limits
    MAX_SHARED_MEMORY = 232 * 1024  # bytes (232KB)
    MAX_TENSOR_MEMORY = 256 * 1024  # bytes (256KB TMEM per SM)

    SZ_MBAR = 8  # bytes

    pruned_configs = []
    for conf in configs:
        M = named_args["M"]
        N = named_args["N"]
        BLOCK_M = conf.kwargs["BLOCK_SIZE_M"]
        BLOCK_N = conf.kwargs["BLOCK_SIZE_N"]
        BLOCK_K = conf.kwargs["BLOCK_SIZE_K"]
        NUM_SMEM_BUFFERS = conf.kwargs["NUM_SMEM_BUFFERS"]
        NUM_TMEM_BUFFERS = conf.kwargs["NUM_TMEM_BUFFERS"]
        PAIR_CTA = conf.kwargs["PAIR_CTA"]
        PERSISTENT = conf.kwargs["PERSISTENT"]
        NUM_MMA_GROUPS = conf.kwargs["NUM_MMA_GROUPS"]

        # Filter out invalid config that causes wrong hardware MMA
        if BLOCK_M // NUM_MMA_GROUPS > 128:
            continue

        # Non-persistent mode only processes a single tile and doesn't benefit from pipelining
        # so we force single TMEM buffer and disable PAIR_CTA optimizations
        if not PERSISTENT:
            NUM_TMEM_BUFFERS = 1
            conf.kwargs["NUM_TMEM_BUFFERS"] = 1

        num_tiles_m = math.ceil(M / BLOCK_M)
        num_tiles_n = math.ceil(N / BLOCK_N)
        # checking num_tiles_m should be sufficent in this case, but adding num_tiles_n for clarity
        pair_cta_compatible = (
            num_tiles_m % 2 == 0 and (num_tiles_m * num_tiles_n) % 2 == 0
        )
        if not pair_cta_compatible:
            # fall back to non-pair CTA mode
            conf.kwargs["PAIR_CTA"] = False
            PAIR_CTA = False

        # Estimate Shared Memory Usage
        # buffers_A: BLOCK_M x BLOCK_K x float16 x NUM_SMEM_BUFFERS
        smem_a = BLOCK_M * BLOCK_K * 2 * NUM_SMEM_BUFFERS
        # buffers_B: BLOCK_K x BLOCK_N x float16 x NUM_SMEM_BUFFERS
        # In PAIR_CTA mode, each CTA only loads half of B
        smem_b_size = (BLOCK_N // 2) if PAIR_CTA else BLOCK_N
        smem_b = BLOCK_K * smem_b_size * 2 * NUM_SMEM_BUFFERS
        # Epilogue staging buffer: BLOCK_M x (BLOCK_N // EPILOGUE_SUBTILE) x float16
        # The epilog group uses local_load/local_slice which implicitly stages data
        # from TMEM to shared memory before TMA store to global memory
        EPILOGUE_SUBTILE = conf.kwargs["EPILOGUE_SUBTILE"]
        smem_epilog = BLOCK_M * (BLOCK_N // EPILOGUE_SUBTILE) * 2
        smem_barriers = NUM_SMEM_BUFFERS * SZ_MBAR
        if PAIR_CTA:
            smem_barriers += NUM_SMEM_BUFFERS * NUM_MMA_GROUPS * SZ_MBAR  # cta_bars
        # tmem_full_bars
        smem_barriers += NUM_TMEM_BUFFERS

        total_smem = smem_a + smem_b + smem_epilog + smem_barriers
        # Prune configs that exceed memory limits
        if total_smem > MAX_SHARED_MEMORY:
            continue

        # Estimate Tensor Memory (TMEM) Usage
        # tmem_buffers: BLOCK_M x BLOCK_N x float32 x NUM_TMEM_BUFFERS
        # TMEM stores the accumulation buffers for MMA operations
        # Non-persistent mode only needs 1 TMEM buffer (processes single tile)
        # Persistent mode uses NUM_TMEM_BUFFERS to overlap MMA and epilogue
        total_tmem = BLOCK_M * BLOCK_N * 4 * NUM_TMEM_BUFFERS
        if total_tmem > MAX_TENSOR_MEMORY:
            continue

        # TODO: add back configs that triggers compiler bug
        if BLOCK_N == 512:
            continue

        pruned_configs.append(conf)

    return pruned_configs


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
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
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
    offs_bn = pid_n * BLOCK_SIZE_N
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS

    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        # Wait for TMEM to be filled
        buf_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_read_phase)

        # load the result from TMEM to registers
        acc_tmem = tmem_buffers[buf_idx]
        offs_am = pid_m * BLOCK_SIZE_M + group_id * BLOCK_M_SPLIT
        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
            acc_tmem_subslice = tlx.local_slice(
                acc_tmem,
                [0, slice_id * slice_size],
                [BLOCK_M_SPLIT, slice_size],
            )
            result = tlx.local_load(acc_tmem_subslice)
            c = result.to(tlx.dtype_of(c_desc))
            c_desc.store([offs_am, offs_bn + slice_id * slice_size], c)

        # Signal MMA consumer if in persistent mode
        if PERSISTENT:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)


@triton.jit
def _process_tile_mma_inner(
    k_tiles,
    NUM_SMEM_BUFFERS,
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
    buffers_A,
    buffers_B,
    tmem_buffers,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
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

    # Wait for epilogue to be done with all TMEM buffers if persistent
    if PERSISTENT:
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            cur_barrier_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
            tlx.barrier_wait(tmem_empty_bars[cur_barrier_idx], tmem_write_phase ^ 1)

    # now iterate along K to compute result for the block
    for k in range(0, k_tiles):
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

        # wait for current phase(round) of load for this buf
        tlx.barrier_wait(B_smem_full_bars[buf], phase)

        # Process all subtiles for this K iteration
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            # Calculate buffer indices
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

            # Wait for this A subtile buffer to be loaded
            tlx.barrier_wait(A_smem_full_bars[a_buf], phase)

            # CTA0 waits for CTA0 and CTA1 to finish loading A and B before issuing dot op
            if PAIR_CTA:
                cta_bar = tlx.remote_view(cta_bars[a_buf], 0)
                tlx.barrier_arrive(cta_bar, 1)
                tlx.barrier_wait(cta_bar, phase=phase, pred=pred_cta0)

            # Perform MMA: use_acc=False for first K iteration, True otherwise
            tlx.async_dot(
                buffers_A[a_buf],
                buffers_B[buf],
                tmem_buffers[acc_buf],
                use_acc=k > 0,
                mBarriers=[A_smem_empty_bars[a_buf]],
                two_ctas=PAIR_CTA,
                out_dtype=tl.float32,
            )

        smem_accum_cnt += 1

    # Wait for last MMA to complete and signal epilogue for all subtiles
    last_buf, last_phase = _get_bufidx_phase(smem_accum_cnt - 1, NUM_SMEM_BUFFERS)
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        a_buf = group_id * NUM_SMEM_BUFFERS + last_buf
        tlx.barrier_wait(A_smem_empty_bars[a_buf], last_phase)
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        # Done filling this buffer, signal epilogue consumer
        tlx.barrier_arrive(tmem_full_bars[acc_buf], 1)

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
    NUM_MMA_GROUPS,
    k_tiles,
    NUM_SMEM_BUFFERS,
    a_desc,
    b_desc,
    buffers_A,
    buffers_B,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    smem_accum_cnt,
    PAIR_CTA,
    cluster_cta_rank,
):
    """Process TMA loads for a single tile with all subtiles."""
    pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    dsize: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    if PAIR_CTA:
        offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // 2)
        expected_bytes: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K // 2
    else:
        offs_bn = pid_n * BLOCK_SIZE_N
        expected_bytes: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K

    # Iterate along K dimension
    for k in range(0, k_tiles):
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        offs_k = k * BLOCK_SIZE_K

        # Load A for the first group
        a_buf = buf
        tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)
        offs_am = pid_m * BLOCK_SIZE_M
        tlx.barrier_expect_bytes(
            A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
        )
        tlx.async_descriptor_load(
            a_desc, buffers_A[a_buf], [offs_am, offs_k], A_smem_full_bars[a_buf]
        )

        # Load B once per K iteration (shared across all subtiles)
        # Wait for the last B subtile buffer from the previous phase to complete its dot operation
        # This ensures A buffer is ready to be reused
        last_a_buf = (NUM_MMA_GROUPS - 1) * NUM_SMEM_BUFFERS + buf
        tlx.barrier_wait(A_smem_empty_bars[last_a_buf], phase ^ 1)
        # Set expected bytes for B and load B (once per K iteration)
        tlx.barrier_expect_bytes(B_smem_full_bars[buf], expected_bytes)
        tlx.async_descriptor_load(
            b_desc, buffers_B[buf], [offs_k, offs_bn], B_smem_full_bars[buf]
        )

        # Load all remaining A subtiles for this K iteration
        for group_id in tl.static_range(1, NUM_MMA_GROUPS):
            # Calculate buffer index: subtile_id * NUM_SMEM_BUFFERS + buf
            a_buf = group_id * NUM_SMEM_BUFFERS + buf

            # Wait for previous phase of dot for this B buffer
            tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)

            # Calculate B offset for this subtile
            offs_am2 = offs_am + group_id * BLOCK_M_SPLIT

            tlx.barrier_expect_bytes(
                A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
            )
            tlx.async_descriptor_load(
                a_desc, buffers_A[a_buf], [offs_am2, offs_k], A_smem_full_bars[a_buf]
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
    NUM_MMA_GROUPS: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    PAIR_CTA: tl.constexpr,  #
    PERSISTENT: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
):
    # allocate NUM_SMEM_BUFFERS buffers
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    buffers_A = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_K),
        tlx.dtype_of(a_desc),
        NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
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
    # Non-persistent mode: 1 TMEM buffer (processes single tile)
    # Persistent mode: NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    # Each buffer holds one subtile: BLOCK_M_SPLIT x BLOCK_SIZE_N
    # Total buffers: NUM_TMEM_BUFFERS * NUM_MMA_GROUPS
    tmem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )

    # CTA pairs are placed along M dim
    if PAIR_CTA:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        cta_bars = tlx.alloc_barriers(
            num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=2
        )  # CTA0 waits for CTA1's data before mma
    else:
        cluster_cta_rank = 0
        pred_cta0 = False
        cta_bars = None

    # allocate barriers - each subtile needs its own barriers
    # NUM_SMEM_BUFFERS barriers per subtile for synchronization
    A_smem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    A_smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    B_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    # Persistent mode: NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    tmem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    if PERSISTENT:
        tmem_empty_bars = tlx.alloc_barriers(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
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
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                        c_desc=c_desc,
                        tmem_buffers=tmem_buffers,
                        tmem_full_bars=tmem_full_bars,
                        tmem_empty_bars=tmem_empty_bars,
                        cur_tmem_buf=cur_tmem_buf,
                        tmem_read_phase=tmem_read_phase,
                        PERSISTENT=PERSISTENT,
                    )
                    tmem_accum_cnt += 1
            else:
                # Non-persistent mode: process single tile
                tile_id = start_pid
                cur_tmem_buf = 0
                tmem_read_phase = 0
                _process_tile_epilogue_inner(
                    tile_id=tile_id,
                    num_pid_in_group=num_pid_in_group,
                    num_pid_m=num_pid_m,
                    GROUP_SIZE_M=GROUP_SIZE_M,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                    NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                    NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                    c_desc=c_desc,
                    tmem_buffers=tmem_buffers,
                    tmem_full_bars=tmem_full_bars,
                    tmem_empty_bars=tmem_empty_bars,
                    cur_tmem_buf=cur_tmem_buf,
                    tmem_read_phase=tmem_read_phase,
                    PERSISTENT=PERSISTENT,
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
                        k_tiles=k_tiles,
                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                        buffers_A=buffers_A,
                        buffers_B=buffers_B,
                        tmem_buffers=tmem_buffers,
                        A_smem_full_bars=A_smem_full_bars,
                        B_smem_full_bars=B_smem_full_bars,
                        A_smem_empty_bars=A_smem_empty_bars,
                        tmem_full_bars=tmem_full_bars,
                        cur_tmem_buf=cur_tmem_buf,
                        tmem_empty_bars=tmem_empty_bars,
                        tmem_write_phase=tmem_write_phase,
                        smem_accum_cnt=smem_accum_cnt,
                        PAIR_CTA=PAIR_CTA,
                        cta_bars=cta_bars,
                        pred_cta0=pred_cta0,
                        PERSISTENT=PERSISTENT,
                    )
                    tmem_accum_cnt += 1
            else:
                # Non-persistent mode: process single tile
                tile_id = start_pid
                smem_accum_cnt = 0
                cur_tmem_buf = 0
                tmem_write_phase = 0  # Not used in non-persistent mode
                _process_tile_mma_inner(
                    k_tiles=k_tiles,
                    NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                    NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                    NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                    buffers_A=buffers_A,
                    buffers_B=buffers_B,
                    tmem_buffers=tmem_buffers,
                    A_smem_full_bars=A_smem_full_bars,
                    B_smem_full_bars=B_smem_full_bars,
                    A_smem_empty_bars=A_smem_empty_bars,
                    tmem_full_bars=tmem_full_bars,
                    cur_tmem_buf=cur_tmem_buf,
                    tmem_empty_bars=tmem_empty_bars,
                    tmem_write_phase=tmem_write_phase,
                    smem_accum_cnt=smem_accum_cnt,
                    PAIR_CTA=PAIR_CTA,
                    cta_bars=cta_bars,
                    pred_cta0=pred_cta0,
                    PERSISTENT=PERSISTENT,
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
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        k_tiles=k_tiles,
                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                        a_desc=a_desc,
                        b_desc=b_desc,
                        buffers_A=buffers_A,
                        buffers_B=buffers_B,
                        A_smem_full_bars=A_smem_full_bars,
                        B_smem_full_bars=B_smem_full_bars,
                        A_smem_empty_bars=A_smem_empty_bars,
                        smem_accum_cnt=smem_accum_cnt,
                        PAIR_CTA=PAIR_CTA,
                        cluster_cta_rank=cluster_cta_rank,
                    )
            else:
                # Non-persistent mode: process single tile
                tile_id = start_pid
                smem_accum_cnt = 0
                _process_tile_producer_inner(
                    tile_id=tile_id,
                    num_pid_in_group=num_pid_in_group,
                    num_pid_m=num_pid_m,
                    GROUP_SIZE_M=GROUP_SIZE_M,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                    k_tiles=k_tiles,
                    NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                    a_desc=a_desc,
                    b_desc=b_desc,
                    buffers_A=buffers_A,
                    buffers_B=buffers_B,
                    A_smem_full_bars=A_smem_full_bars,
                    B_smem_full_bars=B_smem_full_bars,
                    A_smem_empty_bars=A_smem_empty_bars,
                    smem_accum_cnt=smem_accum_cnt,
                    PAIR_CTA=PAIR_CTA,
                    cluster_cta_rank=cluster_cta_rank,
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
