# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

import functools
import math
import os
from typing import Optional

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import triton.language.core as core  # @manual=//triton:triton
import triton.language.extra.tlx as tlx  # @manual=//triton:triton
from triton.tools.tensor_descriptor import TensorDescriptor  # @manual=//triton:triton


@core.builtin
def _add_f32x2(a, b, _semantic=None):
    return core.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            add.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=core.float32,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )


DISABLE_AUTOTUNE = os.environ.get("GEO_DISABLE_AUTOTUNE") == "1"


# Cached SM count — never changes during program lifetime.
# Calling torch.cuda.get_device_properties() on every matmul() call
# adds measurable overhead that degrades benchmark throughput on fast kernels.
@functools.lru_cache(maxsize=1)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def _select_group_size_m(M, N, block_m):
    num_m_tiles = (M + block_m - 1) // block_m
    ratio = M / max(N, 1)

    if ratio > 10:
        # M >> N: sweep M, reuse B
        return 1
    elif ratio < 0.1:
        # N >> M: sweep N, reuse A
        return min(64, num_m_tiles)
    else:
        # Balanced: moderate group size for L2 locality
        return min(8, num_m_tiles)


def get_heuristic_config(M, N, K, num_sms=148):
    """
    Select optimal GEMM config based on problem shape characteristics.

    The selection uses shape-characteristic rules (not exact shape matching):
    1. M/N ratio determines tile shape preference
    2. MN tiles vs SM count determines parallelization strategy (Split-K vs data-parallel)
    3. Arithmetic intensity determines pipeline depth

    Args:
        M, N, K: GEMM dimensions (A is MxK, B is KxN, C is MxN)
        num_sms: Number of SMs on the GPU (default 148 for B200)

    Returns:
        dict: Configuration parameters for the TLX GEMM kernel
    """
    MAX_SMEM = 232 * 1024  # 232KB shared memory limit
    MAX_TMEM = 256 * 1024  # 256KB tensor memory limit per SM

    # ==========================================================================
    # Shape-characteristic analysis
    # ==========================================================================
    mn_ratio = M / max(N, 1)
    is_tall_m = mn_ratio > 4  # M much larger than N
    is_tall_n = mn_ratio < 0.25  # N much larger than M

    # Estimate MN tiles with representative tile sizes
    # Use 256x128 for tall-M, 128x256 for tall-N, 256x256 for balanced
    if is_tall_m:
        ref_bm, ref_bn = 256, 128
    elif is_tall_n:
        ref_bm, ref_bn = 128, 256
    else:
        ref_bm, ref_bn = 256, 256

    num_tiles_m = math.ceil(M / ref_bm)
    num_tiles_n = math.ceil(N / ref_bn)
    num_mn_tiles = num_tiles_m * num_tiles_n

    is_gpu_saturated = num_mn_tiles >= num_sms
    is_undersaturated = num_mn_tiles < num_sms

    # ==========================================================================
    # Shape-characteristic config selection
    # ==========================================================================

    # Characteristic 1: Tall-M shapes benefit from 2-CTA B-tile sharing
    # When M >> N, adjacent M-tiles can share B via 2-CTA clusters
    # Use arithmetic intensity to select tile shape, and K size to select BLOCK_K
    if is_tall_m and is_gpu_saturated:
        arithmetic_intensity = K / max(min(M, N), 1)
        # For low arithmetic intensity (memory-bound), use narrower tiles with larger BLOCK_K
        if arithmetic_intensity <= 1.5:
            return {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
                "NUM_SMEM_BUFFERS": 2,
                "NUM_TMEM_BUFFERS": 2,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": 1,
                "NUM_CTAS": 2,
                "SPLIT_K": 1,
                "INTERLEAVE_EPILOGUE": 1,
                "ctas_per_cga": (2, 1, 1),
                "pre_hook": matmul_tma_set_block_size_hook,
            }
        else:
            # High arithmetic intensity: use wider tiles
            # For large K, use BLOCK_K=128 to reduce K-iterations
            # For smaller K, use BLOCK_K=64 with more SMEM buffers
            if K > N * 2:
                return {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
                    "NUM_SMEM_BUFFERS": 2,
                    "NUM_TMEM_BUFFERS": 1,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 4,
                    "NUM_CTAS": 2,
                    "SPLIT_K": 1,
                    "INTERLEAVE_EPILOGUE": 0,
                    "ctas_per_cga": (2, 1, 1),
                    "pre_hook": matmul_tma_set_block_size_hook,
                }
            else:
                return {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
                    "NUM_SMEM_BUFFERS": 4,
                    "NUM_TMEM_BUFFERS": 1,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 4,
                    "NUM_CTAS": 2,
                    "SPLIT_K": 1,
                    "INTERLEAVE_EPILOGUE": 1,
                    "ctas_per_cga": (2, 1, 1),
                    "pre_hook": matmul_tma_set_block_size_hook,
                }

    # Characteristic 2: Undersaturated GPU needs Split-K for parallelism
    if is_undersaturated:
        # Use MN product to determine tile size - larger MN benefits from wider tiles
        mn_product = M * N
        is_large_output = mn_product >= 1_000_000  # ~1M elements in output

        if is_large_output:
            block_m, block_n, block_k = 256, 128, 64
            k_tiles = math.ceil(K / block_k)
        else:
            block_m, block_n, block_k = 128, 64, 128
            k_tiles = math.ceil(K / block_k)

        split_k = 1
        # Prefer lower Split-K values that still provide enough parallelism
        for sk in [4, 2, 8]:
            if k_tiles >= sk and k_tiles // sk >= 4:
                split_k = sk
                break
        if split_k > 1:
            if is_large_output:
                # Larger output: wider tiles, more epilogue subtiling, fewer TMEM buffers
                return {
                    "BLOCK_SIZE_M": block_m,
                    "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": block_k,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, block_m),
                    "NUM_SMEM_BUFFERS": 4,
                    "NUM_TMEM_BUFFERS": 2,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 8,
                    "NUM_CTAS": 1,
                    "SPLIT_K": split_k,
                    "INTERLEAVE_EPILOGUE": 0,
                    "ctas_per_cga": None,
                    "pre_hook": matmul_tma_set_block_size_hook,
                }
            else:
                # Smaller output: narrower tiles
                return {
                    "BLOCK_SIZE_M": block_m,
                    "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": block_k,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, block_m),
                    "NUM_SMEM_BUFFERS": 4,
                    "NUM_TMEM_BUFFERS": 3,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 1,
                    "NUM_CTAS": 1,
                    "SPLIT_K": split_k,
                    "INTERLEAVE_EPILOGUE": 0,
                    "ctas_per_cga": None,
                    "pre_hook": matmul_tma_set_block_size_hook,
                }

    # Characteristic 3: GPU-saturated shapes use wide tiles for data reuse
    if is_gpu_saturated:
        return {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
            "NUM_SMEM_BUFFERS": 3,
            "NUM_TMEM_BUFFERS": 1,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": 8,
            "NUM_CTAS": 1,
            "SPLIT_K": 1,
            "INTERLEAVE_EPILOGUE": 0,
            "ctas_per_cga": None,
            "pre_hook": matmul_tma_set_block_size_hook,
        }

    # ==========================================================================
    # Fallback: General wave efficiency heuristic for remaining shapes
    # ==========================================================================

    # Candidate configs: (BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS, NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS, NUM_MMA_GROUPS, EPILOGUE_SUBTILE)
    # Based on autotuning results - best configs use BLOCK_K=128, 2-CTA clusters, and balanced buffers
    candidates = [
        # Best config for tall-M shapes (3159809, 384, 384) - prioritize before square config
        (256, 128, 128, 2, 2, 2, 2, 1),  # Best for (3159809, 384, 384)
        # Best config for large square matrices (8192x8192x8192)
        (256, 256, 64, 1, 3, 1, 2, 4),  # Best for 8192x8192x8192
        # Best config for large-K shapes (1024, 256, 16384) - needs Split-K
        (128, 64, 128, 1, 4, 3, 2, 1),  # Best for (1024, 256, 16384) with Split-K
        # 2-CTA configs with BLOCK_K=128 (best performing from autotuning)
        (256, 128, 64, 2, 5, 2, 2, 4),  # Best for (1152, 1024, 213120)
        (128, 256, 64, 2, 4, 2, 1, 2),  # Good general config
        (256, 64, 128, 2, 5, 2, 2, 4),  # Best for skinny-N shapes
        (128, 64, 128, 2, 5, 2, 2, 1),  # Best for (1152, 1024, 12800)
        # 1-CTA configs
        (256, 64, 128, 1, 5, 2, 2, 8),  # Good for skinny-N
        (128, 256, 64, 1, 3, 2, 1, 2),  # Wide tiles
        (128, 128, 64, 1, 4, 2, 1, 2),  # Square tiles
        (256, 128, 64, 1, 3, 1, 2, 2),  # Tall tiles
        (128, 64, 64, 1, 5, 2, 1, 1),  # Small tiles for small problems
        (64, 128, 64, 1, 5, 2, 1, 1),  # Small tiles, wide
        (64, 64, 64, 1, 6, 2, 1, 1),  # Smallest tiles
    ]

    def estimate_smem(
        bm, bn, bk, num_ctas, num_smem_buffers, num_mma_groups, epilogue_subtile
    ):
        """Estimate shared memory usage for a config."""
        smem_a = bm * bk * 2 * num_smem_buffers
        smem_b = bk * (bn // num_ctas) * 2 * num_smem_buffers
        smem_epilog = bm * (bn // epilogue_subtile) * 2
        smem_barriers = (
            num_smem_buffers * num_mma_groups * 8 * (2 if num_ctas == 2 else 1)
        )
        return smem_a + smem_b + smem_epilog + smem_barriers

    def estimate_tmem(bm, bn, num_tmem_buffers):
        """Estimate tensor memory usage for a config."""
        return bm * bn * 4 * num_tmem_buffers

    def compute_wave_score(bm, bn, num_ctas, split_k=1):
        """
        Compute wave efficiency score (lower is better).
        Score = fraction of SMs idle in the last wave.
        """
        ctas_m = (M + bm - 1) // bm
        ctas_n = (N + bn - 1) // bn
        # Round up ctas_m to multiple of num_ctas for cluster alignment
        ctas_m = ((ctas_m + num_ctas - 1) // num_ctas) * num_ctas
        total_ctas = ctas_m * ctas_n * split_k

        if total_ctas == 0:
            return float("inf"), 0, 0

        waves = (total_ctas + num_sms - 1) // num_sms
        fractional_waves = total_ctas / num_sms
        score = waves - fractional_waves  # 0 = perfect, 1 = worst
        return score, total_ctas, waves

    best_config = None
    best_score = float("inf")
    best_waves = float("inf")

    for (
        bm,
        bn,
        bk,
        num_ctas,
        num_smem_buffers,
        num_tmem_buffers,
        num_mma_groups,
        epilogue_subtile,
    ) in candidates:
        # Skip if SMEM exceeds limit
        smem = estimate_smem(
            bm, bn, bk, num_ctas, num_smem_buffers, num_mma_groups, epilogue_subtile
        )
        if smem > MAX_SMEM:
            continue

        # Skip if TMEM exceeds limit
        tmem = estimate_tmem(bm, bn, num_tmem_buffers)
        if tmem > MAX_TMEM:
            continue

        # Skip if MMA group size is invalid (must be <= 128 for hardware)
        if bm // num_mma_groups > 128:
            continue

        # Skip if tiles are larger than the problem
        if bm > M * 2 or bn > N * 2:
            continue

        # Compute wave efficiency
        score, total_ctas, waves = compute_wave_score(bm, bn, num_ctas)

        # Consider split-K only when MN tiles don't saturate GPU
        split_k = 1
        num_tiles_m_c = math.ceil(M / bm)
        num_tiles_n_c = math.ceil(N / bn)
        num_mn_tiles_c = num_tiles_m_c * num_tiles_n_c

        if num_mn_tiles_c < num_sms:
            k_tiles = math.ceil(K / bk)
            # Try split-K values (higher first), each split must have enough K tiles
            for sk in [8, 4, 2]:
                if k_tiles >= sk and k_tiles // sk >= 4:
                    sk_score, sk_ctas, sk_waves = compute_wave_score(
                        bm, bn, num_ctas, sk
                    )
                    if sk_score < score or (sk_score == score and sk_ctas > total_ctas):
                        score, total_ctas, waves, split_k = (
                            sk_score,
                            sk_ctas,
                            sk_waves,
                            sk,
                        )
                    break  # Use the first valid split-K

        # Selection criteria:
        # 1. Prefer lower wave inefficiency score
        # 2. With same score, prefer fewer waves (less overhead)
        # 3. With same waves, prefer larger tiles (less total overhead)
        # 4. Prefer multi-CTA configs for better B-tile sharing
        score_slack = 0.1
        adjusted_score = score

        if (
            adjusted_score < best_score - score_slack
            or (adjusted_score < best_score + score_slack and waves < best_waves)
            or (
                adjusted_score < best_score + score_slack
                and waves == best_waves
                and num_ctas > 1
            )
        ):
            best_score = adjusted_score
            best_waves = waves
            best_config = {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": _select_group_size_m(M, N, bm),
                "NUM_SMEM_BUFFERS": num_smem_buffers,
                "NUM_TMEM_BUFFERS": num_tmem_buffers,
                "NUM_MMA_GROUPS": num_mma_groups,
                "EPILOGUE_SUBTILE": epilogue_subtile,
                "NUM_CTAS": num_ctas,
                "SPLIT_K": split_k,
                "INTERLEAVE_EPILOGUE": 0,
                "ctas_per_cga": (num_ctas, 1, 1) if num_ctas > 1 else None,
                "pre_hook": matmul_tma_set_block_size_hook,
            }

    return best_config


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_SMEM_BUFFERS": 13,
                "NUM_TMEM_BUFFERS": 4,
                "NUM_MMA_GROUPS": 1,
                "EPILOGUE_SUBTILE": 4,
                "NUM_CTAS": 2,
                "SPLIT_K": 1,
                "INTERLEAVE_EPILOGUE": 0,
                "USE_WARP_BARRIER": True,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(2, 1, 1),
        ),
    ]


def get_reduced_autotune_config():
    """Reduced config set for testing — covers key axes without full 62K combinatorial explosion."""
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": g,
                "NUM_SMEM_BUFFERS": 3,
                "NUM_TMEM_BUFFERS": 2,
                "NUM_MMA_GROUPS": m,
                "EPILOGUE_SUBTILE": subtile,
                "NUM_CTAS": num_ctas,
                "SPLIT_K": split_k,
                "INTERLEAVE_EPILOGUE": interleave,
                "USE_WARP_BARRIER": False,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(num_ctas, 1, 1) if num_ctas > 1 else None,
        )
        for g in [1, 8, 64]
        for m in [1, 2]
        for subtile in [1, 4]
        for num_ctas in [1, 2]
        for split_k in [1, 4]
        for interleave in [0, 1]
    ]


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    NUM_MMA_GROUPS = nargs.get("NUM_MMA_GROUPS", 1)
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    BLOCK_N_PER_CTA = BLOCK_N // NUM_CTAS
    # For column-major inputs, TMA descriptor block shape matches the transposed view
    if nargs.get("A_ROW_MAJOR", True):
        nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    else:
        nargs["a_desc"].block_shape = [BLOCK_K, BLOCK_M_SPLIT]
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N_PER_CTA]
    else:
        nargs["b_desc"].block_shape = [BLOCK_N_PER_CTA, BLOCK_K]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    nargs["c_desc"].block_shape = [
        BLOCK_M_SPLIT,
        BLOCK_N // EPILOGUE_SUBTILE,
    ]
    nargs["bias_desc"].block_shape = [BLOCK_N]
    SPLIT_K = nargs.get("SPLIT_K", 1)
    if SPLIT_K > 1:
        M = nargs["M"]
        N = nargs["N"]
        workspace = torch.empty(
            (SPLIT_K * M, N),
            device=nargs["c_desc"].base.device,
            dtype=nargs["c_desc"].base.dtype,
        )
        nargs["workspace_desc"].base = workspace
        nargs["workspace_desc"].shape = list(workspace.shape)
    else:
        nargs["workspace_desc"].base = nargs["c_desc"].base
        nargs["workspace_desc"].shape = list(nargs["c_desc"].base.shape)
    nargs["workspace_desc"].block_shape = [
        BLOCK_M_SPLIT,
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
    NUM_SMS = _get_num_sms()
    MAX_SHARED_MEMORY = 232 * 1024  # bytes (232KB)
    MAX_TENSOR_MEMORY = 256 * 1024  # bytes (256KB TMEM per SM)

    MBARRIER_SIZE = 8  # bytes

    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]

    best_sk1_occupancy = 0.0
    for conf in configs:
        if conf.kwargs.get("SPLIT_K", 1) == 1:
            bm = conf.kwargs["BLOCK_SIZE_M"]
            bn = conf.kwargs["BLOCK_SIZE_N"]
            mn_tiles = math.ceil(M / bm) * math.ceil(N / bn)
            best_sk1_occupancy = max(best_sk1_occupancy, mn_tiles / NUM_SMS)
    skip_split_k = best_sk1_occupancy >= 0.9

    pruned_configs = []
    for conf in configs:
        BLOCK_M = conf.kwargs["BLOCK_SIZE_M"]
        BLOCK_N = conf.kwargs["BLOCK_SIZE_N"]
        BLOCK_K = conf.kwargs["BLOCK_SIZE_K"]
        NUM_SMEM_BUFFERS = conf.kwargs["NUM_SMEM_BUFFERS"]
        NUM_TMEM_BUFFERS = conf.kwargs["NUM_TMEM_BUFFERS"]
        NUM_CTAS = conf.kwargs["NUM_CTAS"]
        NUM_MMA_GROUPS = conf.kwargs["NUM_MMA_GROUPS"]
        SPLIT_K = conf.kwargs.get("SPLIT_K", 1)
        EPILOGUE_SUBTILE = conf.kwargs["EPILOGUE_SUBTILE"]
        INTERLEAVE_EPILOGUE = conf.kwargs.get("INTERLEAVE_EPILOGUE", 0)
        GROUP_SIZE_M = conf.kwargs["GROUP_SIZE_M"]

        # Filter out invalid config that causes wrong hardware MMA
        if BLOCK_M // NUM_MMA_GROUPS > 128:
            continue
        # GROUP_SIZE_M must be a multiple of NUM_CTAS so that consecutive
        # tile_ids (assigned to paired CTAs in a cluster) always map to the
        # same pid_n. Otherwise, at group boundaries a CTA pair can straddle
        # two different pid_n values, breaking 2-CTA B-tile sharing.
        if GROUP_SIZE_M % NUM_CTAS != 0:
            continue

        # EPILOGUE_SUBTILE must evenly divide BLOCK_N
        if BLOCK_N % EPILOGUE_SUBTILE != 0:
            continue

        # Interleaved epilogue requires NUM_MMA_GROUPS == 2
        if INTERLEAVE_EPILOGUE and NUM_MMA_GROUPS != 2:
            continue

        # Blackwell MMA requires BLOCK_M_SPLIT >= 64
        if BLOCK_M // NUM_MMA_GROUPS < 64:
            continue

        num_tiles_m = math.ceil(M / BLOCK_M)
        num_tiles_n = math.ceil(N / BLOCK_N)
        num_mn_tiles = num_tiles_m * num_tiles_n

        # BM=64 tiles only help when MN is too small with larger tiles.
        # Skip them for shapes that already have enough spatial tiles
        # to avoid bloating the autotuner search space.
        if BLOCK_M == 64 and math.ceil(M / 128) * math.ceil(N / 128) > 16:
            continue

        # --- Split-K gating: only allow SPLIT_K > 1 for small shapes ---
        # Split-K helps when MN tiles are too few to saturate the GPU.
        # For large shapes with plenty of MN tiles, SPLIT_K=1 is better
        # since it avoids the atomic reduction overhead.
        if SPLIT_K > 1:
            if skip_split_k:
                continue
            if num_mn_tiles >= NUM_SMS:
                continue
            k_tiles = math.ceil(K / BLOCK_K)
            if k_tiles < SPLIT_K:
                continue
            # Reject SK values where cdiv overallocation leaves the last split empty
            # (causes deadlock: producer loop is empty but MMA consumer waits on barrier)
            k_tiles_per_split = math.ceil(k_tiles / SPLIT_K)
            if k_tiles_per_split * (SPLIT_K - 1) >= k_tiles:
                continue
            # Each split must have enough K tiles to be worthwhile
            if k_tiles // SPLIT_K < 4:
                continue
            # Split-K with bias would add bias SPLIT_K times; disable for now
            if named_args.get("HAS_BIAS", False):
                continue

        # --- Shared Memory estimation ---
        smem_a = BLOCK_M * BLOCK_K * 2 * NUM_SMEM_BUFFERS
        smem_b_size = BLOCK_N // NUM_CTAS
        smem_b = BLOCK_K * smem_b_size * 2 * NUM_SMEM_BUFFERS
        smem_epilog = BLOCK_M * (BLOCK_N // EPILOGUE_SUBTILE) * 2
        smem_barriers = NUM_SMEM_BUFFERS * NUM_MMA_GROUPS * MBARRIER_SIZE
        if NUM_CTAS == 2:
            smem_barriers += NUM_SMEM_BUFFERS * NUM_MMA_GROUPS * MBARRIER_SIZE
        smem_barriers += NUM_TMEM_BUFFERS

        total_smem = smem_a + smem_b + smem_epilog + smem_barriers
        if named_args.get("HAS_BIAS", False):
            total_smem += BLOCK_N * 2 * NUM_TMEM_BUFFERS  # bf16 * double-buffer
            smem_barriers += NUM_TMEM_BUFFERS * 2 * MBARRIER_SIZE  # full + empty
            total_smem += NUM_TMEM_BUFFERS * 2 * MBARRIER_SIZE
        if total_smem > MAX_SHARED_MEMORY:
            continue

        # --- Tensor Memory (TMEM) estimation ---
        total_tmem = BLOCK_M * BLOCK_N * 4 * NUM_TMEM_BUFFERS
        if total_tmem > MAX_TENSOR_MEMORY:
            continue

        pruned_configs.append(conf)

    # Two-level SPLIT_K filter (per tile-size group):
    #   1. Minimize wave count (fewer waves = less wall-clock time).
    #   2. Within the same wave count, maximize SPLIT_K (more K-parallelism
    #      across SMs). E.g. with 148 SMs and 40 base tiles: SPLIT_K=3
    #      gives 120 tiles (120 SMs active, each does K/3 work) vs SPLIT_K=1
    #      giving 40 tiles (40 SMs active, each does K/1 work) — both 1 wave,
    #      but SPLIT_K=3 is faster because work is spread across more SMs.
    # Applied per (BM, BN, BK) group because different tile sizes have
    # vastly different compute characteristics.
    # Note: for saturated shapes, SPLIT_K>1 configs are already pruned by
    # the base_tiles >= NUM_SMS gate above, so only SPLIT_K=1 survives.
    if pruned_configs:

        def _total_tiles(c):
            return (
                math.ceil(M / c.kwargs["BLOCK_SIZE_M"])
                * math.ceil(N / c.kwargs["BLOCK_SIZE_N"])
                * c.kwargs.get("SPLIT_K", 1)
            )

        def _num_waves(c):
            return math.ceil(_total_tiles(c) / NUM_SMS)

        def _tile_key(c):
            return (
                c.kwargs["BLOCK_SIZE_M"],
                c.kwargs["BLOCK_SIZE_N"],
                c.kwargs["BLOCK_SIZE_K"],
            )

        # Group by tile size
        tile_groups = {}
        for c in pruned_configs:
            tile_groups.setdefault(_tile_key(c), []).append(c)

        result = []
        for group_configs in tile_groups.values():
            min_waves = min(_num_waves(c) for c in group_configs)
            best = [c for c in group_configs if _num_waves(c) == min_waves]
            max_sk = max(c.kwargs.get("SPLIT_K", 1) for c in best)
            best = [c for c in best if c.kwargs.get("SPLIT_K", 1) == max_sk]
            result.extend(best)

        pruned_configs = result

    # --- Golden Rule: sweep the large dimension, fix the small one ---
    # A[M,K] changes with M; B[K,N] changes with N.
    # GROUP_SIZE_M controls how many M-tiles are grouped before advancing N.
    #   GROUP_SIZE_M = 1  → sweep M first (column-major), B (small-N side) reused
    #   GROUP_SIZE_M = large → sweep N first (row-major), A (small-M side) reused
    # When M >> N: prefer small GROUP_SIZE_M (sweep M, fix B for reuse)
    # When N >> M: prefer large GROUP_SIZE_M (sweep N, fix A for reuse)
    if pruned_configs:
        IMBALANCE_THRESHOLD = 10  # ratio at which we enforce the rule
        if M > N * IMBALANCE_THRESHOLD:
            # M >> N: keep only small GROUP_SIZE_M to sweep M
            pruned_configs = [
                c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] == 1
            ]
        elif N > M * IMBALANCE_THRESHOLD:
            # N >> M: keep only large GROUP_SIZE_M to sweep N
            pruned_configs = [
                c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] >= 32
            ]
        else:
            # Balanced M ≈ N: keep moderate GROUP_SIZE_M for L2 locality
            pruned_configs = [
                c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] == 8
            ]

    # Pareto-optimal filtering on (NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS,
    # NUM_MMA_GROUPS): these are independent resource dimensions where more
    # buffers / groups generally means better pipelining, but no single
    # dimension dominates the others.  Keep a config unless another config
    # in the same (BM, BN, BK, SUBTILE, NUM_CTAS, SPLIT_K) group dominates
    # it (>= in all dimensions, > in at least one).
    if pruned_configs:

        def _group_key(c):
            return (
                c.kwargs["BLOCK_SIZE_M"],
                c.kwargs["BLOCK_SIZE_N"],
                c.kwargs["BLOCK_SIZE_K"],
                c.kwargs["EPILOGUE_SUBTILE"],
                c.kwargs["NUM_CTAS"],
                c.kwargs.get("SPLIT_K", 1),
                c.kwargs.get("INTERLEAVE_EPILOGUE", 0),
            )

        def _val(c):
            return (
                c.kwargs["NUM_SMEM_BUFFERS"],
                c.kwargs["NUM_TMEM_BUFFERS"],
                c.kwargs["NUM_MMA_GROUPS"],
            )

        def _dominates(a, b):
            """Return True if a dominates b (>= in all, > in at least one)."""
            va, vb = _val(a), _val(b)
            return all(x >= y for x, y in zip(va, vb)) and any(
                x > y for x, y in zip(va, vb)
            )

        groups = {}
        for c in pruned_configs:
            groups.setdefault(_group_key(c), []).append(c)

        pruned_configs = []
        for members in groups.values():
            for c in members:
                if not any(_dominates(other, c) for other in members if other is not c):
                    pruned_configs.append(c)

    return pruned_configs


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@triton.jit
def _compute_grid_info(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    SPLIT_K,
    NUM_CTAS: tl.constexpr,
):
    """Compute common grid information used across async tasks."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Pad num_pid_m to multiple of NUM_CTAS so CTA clusters tile evenly along M.
    num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_mn_tiles = num_pid_m * num_pid_n
    num_tiles = num_mn_tiles * SPLIT_K
    k_tiles_total = tl.cdiv(K, BLOCK_SIZE_K)
    return (
        start_pid,
        num_pid_m,
        num_pid_n,
        num_pid_in_group,
        num_mn_tiles,
        num_tiles,
        k_tiles_total,
    )


@triton.jit
def _process_tile_epilogue_inner(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    num_mn_tiles,
    GROUP_SIZE_M,
    M,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    EPILOGUE_SUBTILE,
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
    SPLIT_K,
    INTERLEAVE_EPILOGUE,
    c_desc,
    workspace_desc,
    c_smem_buffers,
    tmem_buffers,
    tmem_full_bars,
    tmem_empty_bars,
    cur_tmem_buf,
    tmem_read_phase,
    bias_smem,
    bias_full_bars,
    bias_empty_bars,
    HAS_BIAS: tl.constexpr,
    NUM_CTAS: tl.constexpr = 1,
):
    """Process epilogue for a single tile."""
    mn_tile_id = tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS

    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    if SPLIT_K > 1:
        split_id = tile_id // num_mn_tiles
        out_desc = workspace_desc
        row_base = split_id * M
    else:
        out_desc = c_desc
        row_base = 0

    if HAS_BIAS:
        tlx.barrier_wait(bias_full_bars[cur_tmem_buf], tmem_read_phase)

    if INTERLEAVE_EPILOGUE:
        # Interleaved TMA stores across two groups to improve memory throughput.
        # Pattern: wait g0, store g0s0, wait g1, store g1s0,
        #          then alternate g0/g1 for slices 1-3.
        buf_idx_0 = 0 * NUM_TMEM_BUFFERS + cur_tmem_buf
        buf_idx_1 = 1 * NUM_TMEM_BUFFERS + cur_tmem_buf
        acc_tmem_0 = tmem_buffers[buf_idx_0]
        acc_tmem_1 = tmem_buffers[buf_idx_1]
        offs_am_0 = pid_m * BLOCK_SIZE_M + 0 * BLOCK_M_SPLIT
        offs_am_1 = pid_m * BLOCK_SIZE_M + 1 * BLOCK_M_SPLIT

        # --- Wait for group 0, store group 0 slice 0 ---
        tlx.barrier_wait(tmem_full_bars[buf_idx_0], tmem_read_phase)
        acc_sub = tlx.local_slice(
            acc_tmem_0, [0, 0 * slice_size], [BLOCK_M_SPLIT, slice_size]
        )
        result = tlx.local_load(acc_sub)
        if NUM_CTAS == 2:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1, remote_cta_rank=0)
        else:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1)
        if HAS_BIAS:
            bias_vals = tlx.local_load(
                tlx.local_slice(bias_smem[cur_tmem_buf], [0], [slice_size])
            ).to(tl.float32)
            result = _add_f32x2(result, bias_vals)
        c = result.to(tlx.dtype_of(out_desc))
        c_smem = c_smem_buffers[0]
        tlx.local_store(c_smem, c)
        tlx.fence_async_shared()
        tlx.async_descriptor_store(
            out_desc,
            c_smem,
            [row_base + offs_am_0, offs_bn + 0 * slice_size],
            eviction_policy="evict_first",
        )

        # --- Wait for group 1, store group 1 slice 0 ---
        tlx.barrier_wait(tmem_full_bars[buf_idx_1], tmem_read_phase)
        acc_sub = tlx.local_slice(
            acc_tmem_1, [0, 0 * slice_size], [BLOCK_M_SPLIT, slice_size]
        )
        result = tlx.local_load(acc_sub)
        if NUM_CTAS == 2:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1, remote_cta_rank=0)
        else:
            tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1)
        if HAS_BIAS:
            bias_vals = tlx.local_load(
                tlx.local_slice(bias_smem[cur_tmem_buf], [0], [slice_size])
            ).to(tl.float32)
            result = _add_f32x2(result, bias_vals)
        c = result.to(tlx.dtype_of(out_desc))
        c_smem = c_smem_buffers[1]
        tlx.local_store(c_smem, c)
        tlx.fence_async_shared()
        tlx.async_descriptor_store(
            out_desc,
            c_smem,
            [row_base + offs_am_1, offs_bn + 0 * slice_size],
            eviction_policy="evict_first",
        )

        # --- Slices 1-3: alternate group 0, group 1 ---
        for slice_id in tl.static_range(1, EPILOGUE_SUBTILE):
            # Group 0
            acc_sub = tlx.local_slice(
                acc_tmem_0, [0, slice_id * slice_size], [BLOCK_M_SPLIT, slice_size]
            )
            result = tlx.local_load(acc_sub)
            if NUM_CTAS == 2:
                tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1)
            if HAS_BIAS:
                bias_vals = tlx.local_load(
                    tlx.local_slice(
                        bias_smem[cur_tmem_buf], [slice_id * slice_size], [slice_size]
                    )
                ).to(tl.float32)
                result = _add_f32x2(result, bias_vals)
            c = result.to(tlx.dtype_of(out_desc))
            c_smem = c_smem_buffers[0]
            tlx.async_descriptor_store_wait(1)
            tlx.local_store(c_smem, c)
            tlx.fence("async_shared")
            tlx.async_descriptor_store(
                out_desc,
                c_smem,
                [row_base + offs_am_0, offs_bn + slice_id * slice_size],
                eviction_policy="evict_first",
            )

            # Group 1
            acc_sub = tlx.local_slice(
                acc_tmem_1, [0, slice_id * slice_size], [BLOCK_M_SPLIT, slice_size]
            )
            result = tlx.local_load(acc_sub)
            if NUM_CTAS == 2:
                tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1, remote_cta_rank=0)
            else:
                tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1)
            if HAS_BIAS:
                bias_vals = tlx.local_load(
                    tlx.local_slice(
                        bias_smem[cur_tmem_buf], [slice_id * slice_size], [slice_size]
                    )
                ).to(tl.float32)
                result = _add_f32x2(result, bias_vals)
            c = result.to(tlx.dtype_of(out_desc))
            c_smem = c_smem_buffers[1]
            tlx.async_descriptor_store_wait(1)
            tlx.local_store(c_smem, c)
            tlx.fence("async_shared")
            tlx.async_descriptor_store(
                out_desc,
                c_smem,
                [row_base + offs_am_1, offs_bn + slice_id * slice_size],
                eviction_policy="evict_first",
            )
    else:
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
                if NUM_CTAS == 2:
                    tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1, remote_cta_rank=0)
                else:
                    tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
                if HAS_BIAS:
                    bias_vals = tlx.local_load(
                        tlx.local_slice(
                            bias_smem[cur_tmem_buf],
                            [slice_id * slice_size],
                            [slice_size],
                        )
                    ).to(tl.float32)
                    result = _add_f32x2(result, bias_vals)
                c = result.to(tlx.dtype_of(out_desc))
                c_smem = c_smem_buffers[(group_id * EPILOGUE_SUBTILE + slice_id) % 2]
                tlx.async_descriptor_store_wait(1)
                tlx.local_store(c_smem, c)
                tlx.fence_async_shared()
                tlx.async_descriptor_store(
                    out_desc,
                    c_smem,
                    [row_base + offs_am, offs_bn + slice_id * slice_size],
                    eviction_policy="evict_first",
                )

    # Wait for all TMA stores to complete
    tlx.async_descriptor_store_wait(0)

    if HAS_BIAS:
        tlx.barrier_arrive(bias_empty_bars[cur_tmem_buf], 1)


@triton.jit
def _process_tile_mma_inner(
    k_tiles,
    k_tile_start,
    k_tile_end,
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
    NUM_CTAS,
    cluster_cta_rank,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
):
    """Process MMA for a single tile over [k_tile_start, k_tile_end). Returns updated smem_accum_cnt."""
    local_k_tiles = k_tile_end - k_tile_start

    # Peeled first K-iteration: wait for data before acquiring TMEM
    buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

    is_leader = cluster_cta_rank == 0

    if is_leader:
        tlx.barrier_wait(B_smem_full_bars[buf], phase)

        # Process first K iteration (peeled) with use_acc=False
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

            tlx.barrier_wait(A_smem_full_bars[a_buf], phase)

            cur_barrier_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
            tlx.barrier_wait(tmem_empty_bars[cur_barrier_idx], tmem_write_phase ^ 1)

            if not A_ROW_MAJOR:
                a_operand = tlx.local_trans(buffers_A[a_buf])
            else:
                a_operand = buffers_A[a_buf]
            if not B_ROW_MAJOR:
                b_operand = tlx.local_trans(buffers_B[buf])
            else:
                b_operand = buffers_B[buf]

            tlx.async_dot(
                a_operand,
                b_operand,
                acc=tmem_buffers[acc_buf],
                use_acc=False,
                mBarriers=[A_smem_empty_bars[a_buf]],
                two_ctas=NUM_CTAS == 2,
                out_dtype=tl.float32,
            )

        smem_accum_cnt += 1

        # Remaining K iterations with use_acc=True
        for _ in range(1, local_k_tiles):
            buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

            tlx.barrier_wait(B_smem_full_bars[buf], phase)

            for group_id in tl.static_range(NUM_MMA_GROUPS):
                a_buf = group_id * NUM_SMEM_BUFFERS + buf
                acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

                tlx.barrier_wait(A_smem_full_bars[a_buf], phase)

                if not A_ROW_MAJOR:
                    a_operand = tlx.local_trans(buffers_A[a_buf])
                else:
                    a_operand = buffers_A[a_buf]
                if not B_ROW_MAJOR:
                    b_operand = tlx.local_trans(buffers_B[buf])
                else:
                    b_operand = buffers_B[buf]

                tlx.async_dot(
                    a_operand,
                    b_operand,
                    acc=tmem_buffers[acc_buf],
                    use_acc=True,
                    mBarriers=[A_smem_empty_bars[a_buf]],
                    two_ctas=NUM_CTAS == 2,
                    out_dtype=tl.float32,
                )

            smem_accum_cnt += 1

        # Wait for last MMA to complete and signal epilogue
        last_buf, last_phase = _get_bufidx_phase(smem_accum_cnt - 1, NUM_SMEM_BUFFERS)
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + last_buf
            tlx.barrier_wait(A_smem_empty_bars[a_buf], last_phase)
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
            tlx.barrier_arrive(tmem_full_bars[acc_buf], 1)
            if NUM_CTAS == 2:
                tlx.barrier_arrive(tmem_full_bars[acc_buf], 1, remote_cta_rank=1)
    else:
        smem_accum_cnt += local_k_tiles

    return smem_accum_cnt


@triton.jit
def _process_tile_producer_inner(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    num_mn_tiles,
    GROUP_SIZE_M,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    NUM_MMA_GROUPS,
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    a_desc,
    b_desc,
    buffers_A,
    buffers_B,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    smem_accum_cnt,
    NUM_CTAS,
    cluster_cta_rank,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
):
    """Process TMA loads for a single tile with all subtiles over [k_tile_start, k_tile_end)."""
    mn_tile_id = tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    dsize: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // NUM_CTAS)
    expected_bytes: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K

    local_k_tiles = k_tile_end - k_tile_start
    is_leader = cluster_cta_rank == 0

    # Iterate along K dimension for this split's range
    for k_idx in range(0, local_k_tiles):
        k = k_tile_start + k_idx
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        offs_k = k * BLOCK_SIZE_K

        # Load A for the first group
        a_buf = buf
        tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)
        offs_am = pid_m * BLOCK_SIZE_M
        if is_leader:
            tlx.barrier_expect_bytes(
                A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K * NUM_CTAS
            )
        if not A_ROW_MAJOR:
            tlx.async_descriptor_load(
                a_desc,
                buffers_A[a_buf],
                [offs_k, offs_am],
                A_smem_full_bars[a_buf],
                eviction_policy="evict_last",
                two_ctas=NUM_CTAS == 2,
            )
        else:
            tlx.async_descriptor_load(
                a_desc,
                buffers_A[a_buf],
                [offs_am, offs_k],
                A_smem_full_bars[a_buf],
                eviction_policy="evict_last",
                two_ctas=NUM_CTAS == 2,
            )

        # Load B once per K iteration (shared across all subtiles)
        last_a_buf = (NUM_MMA_GROUPS - 1) * NUM_SMEM_BUFFERS + buf
        tlx.barrier_wait(A_smem_empty_bars[last_a_buf], phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(B_smem_full_bars[buf], expected_bytes)
        if not B_ROW_MAJOR:
            tlx.async_descriptor_load(
                b_desc,
                buffers_B[buf],
                [offs_bn, offs_k],
                B_smem_full_bars[buf],
                eviction_policy="evict_last",
                two_ctas=NUM_CTAS == 2,
            )
        else:
            tlx.async_descriptor_load(
                b_desc,
                buffers_B[buf],
                [offs_k, offs_bn],
                B_smem_full_bars[buf],
                eviction_policy="evict_last",
                two_ctas=NUM_CTAS == 2,
            )

        # Load all remaining A subtiles for this K iteration
        for group_id in tl.static_range(1, NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf

            tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)

            offs_am2 = offs_am + group_id * BLOCK_M_SPLIT

            if is_leader:
                tlx.barrier_expect_bytes(
                    A_smem_full_bars[a_buf],
                    dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K * NUM_CTAS,
                )
            if not A_ROW_MAJOR:
                tlx.async_descriptor_load(
                    a_desc,
                    buffers_A[a_buf],
                    [offs_k, offs_am2],
                    A_smem_full_bars[a_buf],
                    eviction_policy="evict_last",
                    two_ctas=NUM_CTAS == 2,
                )
            else:
                tlx.async_descriptor_load(
                    a_desc,
                    buffers_A[a_buf],
                    [offs_am2, offs_k],
                    A_smem_full_bars[a_buf],
                    eviction_policy="evict_last",
                    two_ctas=NUM_CTAS == 2,
                )

        smem_accum_cnt += 1

    return smem_accum_cnt


TORCH_DTYPE_TO_TRITON = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


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
    base_offs = offs_m[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        ws_offs = base_offs + s * M * N
        partial = tl.load(workspace_ptr + ws_offs, mask=mask, other=0.0)
        acc += partial.to(tl.float32)

    tl.store(c_ptr + base_offs, acc.to(OUTPUT_DTYPE), mask=mask)


def reduce_post_hook(nargs, exception=None):
    if exception is not None:
        return
    split_k = nargs.get("SPLIT_K", 1)
    if split_k > 1:
        M = nargs["M"]
        N = nargs["N"]
        workspace = nargs["workspace_desc"].base
        c = nargs["c_desc"].base
        reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
        _reduce_k_kernel[reduce_grid](
            workspace,
            c,
            M,
            N,
            SPLIT_K=split_k,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[workspace.dtype],
            num_warps=4,
        )


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_SMEM_BUFFERS": 13,
                "NUM_TMEM_BUFFERS": 4,
                "NUM_MMA_GROUPS": 1,
                "EPILOGUE_SUBTILE": 4,
                "NUM_CTAS": 2,
                "SPLIT_K": 1,
                "INTERLEAVE_EPILOGUE": 0,
                "USE_WARP_BARRIER": True,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(2, 1, 1),
        ),
    ],
    key=["M", "N", "K"],
    post_hook=reduce_post_hook,
)
@triton.jit
def matmul_kernel_tma_ws_blackwell(
    a_desc: TensorDescriptor,
    b_desc: TensorDescriptor,
    c_desc: TensorDescriptor,
    workspace_desc: TensorDescriptor,
    bias_desc: TensorDescriptor,
    M: int,
    N: int,
    K: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    INTERLEAVE_EPILOGUE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
    USE_WARP_BARRIER: tl.constexpr = False,
) -> None:
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    if not A_ROW_MAJOR:
        buffers_A = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_M_SPLIT),
            tlx.dtype_of(a_desc),
            NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        )
    else:
        buffers_A = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_SIZE_K),
            tlx.dtype_of(a_desc),
            NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        )
    # In 2-CTA mode, each CTA only needs to load BLOCK_N // NUM_CTAS of B.
    if not B_ROW_MAJOR:
        buffers_B = tlx.local_alloc(
            (BLOCK_SIZE_N // NUM_CTAS, BLOCK_SIZE_K),
            tlx.dtype_of(b_desc),
            NUM_SMEM_BUFFERS,
        )
    else:
        buffers_B = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_SIZE_N // NUM_CTAS),
            tlx.dtype_of(b_desc),
            NUM_SMEM_BUFFERS,
        )
    tmem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )

    # Allocate SMEM buffers for epilogue TMA store (at least 2 for multi-buffering)
    NUM_EPILOGUE_SMEM_BUFFERS: tl.constexpr = (
        NUM_MMA_GROUPS if NUM_MMA_GROUPS > 2 else 2
    )
    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    c_smem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, slice_size),
        tlx.dtype_of(c_desc),
        NUM_EPILOGUE_SMEM_BUFFERS,
    )

    # CTA pairs are placed along M dim
    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
    else:
        cluster_cta_rank = 0

    A_smem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    A_smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    B_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    if USE_WARP_BARRIER:
        tmem_full_bars = tlx.alloc_warp_barrier(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, num_warps=1
        )
        tmem_empty_bars = tlx.alloc_warp_barrier(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
            num_warps=4,
            num_arrivals=EPILOGUE_SUBTILE * NUM_CTAS,
        )
    else:
        tmem_full_bars = tlx.alloc_barriers(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
        )
        tmem_empty_bars = tlx.alloc_barriers(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
            arrive_count=EPILOGUE_SUBTILE * NUM_CTAS,
        )

    if HAS_BIAS:
        bias_smem = tlx.local_alloc(
            (BLOCK_SIZE_N,), tlx.dtype_of(a_desc), NUM_TMEM_BUFFERS
        )
        bias_full_bars = tlx.alloc_barriers(
            num_barriers=NUM_TMEM_BUFFERS, arrive_count=1
        )
        bias_empty_bars = tlx.alloc_barriers(
            num_barriers=NUM_TMEM_BUFFERS, arrive_count=1
        )

    with tlx.async_tasks():
        with tlx.async_task("default"):  # epilogue consumer
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )

            tmem_accum_cnt = 0
            tile_id = start_pid

            while tile_id < num_tiles:
                # Skip tiles whose split has zero K-tiles (last split
                # can be empty when cdiv(k_tiles_total, SPLIT_K) * (SPLIT_K-1)
                # >= k_tiles_total).
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)
                if k_tile_end > k_tile_start:
                    cur_tmem_buf, tmem_read_phase = _get_bufidx_phase(
                        tmem_accum_cnt, NUM_TMEM_BUFFERS
                    )
                    _process_tile_epilogue_inner(
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        num_mn_tiles=num_mn_tiles,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        M=M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                        SPLIT_K=SPLIT_K,
                        INTERLEAVE_EPILOGUE=INTERLEAVE_EPILOGUE,
                        c_desc=c_desc,
                        workspace_desc=workspace_desc,
                        c_smem_buffers=c_smem_buffers,
                        tmem_buffers=tmem_buffers,
                        tmem_full_bars=tmem_full_bars,
                        tmem_empty_bars=tmem_empty_bars,
                        cur_tmem_buf=cur_tmem_buf,
                        tmem_read_phase=tmem_read_phase,
                        bias_smem=bias_smem if HAS_BIAS else None,
                        bias_full_bars=bias_full_bars if HAS_BIAS else None,
                        bias_empty_bars=bias_empty_bars if HAS_BIAS else None,
                        HAS_BIAS=HAS_BIAS,
                        NUM_CTAS=NUM_CTAS,
                    )
                    tmem_accum_cnt += 1
                tile_id += NUM_SMS

        with tlx.async_task(num_warps=1, num_regs=24):  # MMA consumer
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )

            tmem_accum_cnt = 0
            smem_accum_cnt = 0
            tile_id = start_pid

            while tile_id < num_tiles:
                # Compute K range for this split
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)

                # Skip tiles whose split has zero K-tiles
                if k_tile_end > k_tile_start:
                    cur_tmem_buf, tmem_write_phase = _get_bufidx_phase(
                        tmem_accum_cnt, NUM_TMEM_BUFFERS
                    )
                    smem_accum_cnt = _process_tile_mma_inner(
                        k_tiles=k_tiles_total,
                        k_tile_start=k_tile_start,
                        k_tile_end=k_tile_end,
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
                        NUM_CTAS=NUM_CTAS,
                        cluster_cta_rank=cluster_cta_rank,
                        A_ROW_MAJOR=A_ROW_MAJOR,
                        B_ROW_MAJOR=B_ROW_MAJOR,
                    )
                    tmem_accum_cnt += 1
                tile_id += NUM_SMS

        with tlx.async_task(num_warps=1, num_regs=24):  # producer, TMA load
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )

            smem_accum_cnt = 0
            bias_tile_cnt = 0
            tile_id = start_pid

            while tile_id < num_tiles:
                # Compute K range for this split
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)

                # Skip tiles whose split has zero K-tiles
                if k_tile_end > k_tile_start:
                    if HAS_BIAS:
                        mn_tile_id = tile_id % num_mn_tiles
                        pid_m_b, pid_n_b = _compute_pid(
                            mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M
                        )
                        offs_bn_b = pid_n_b * BLOCK_SIZE_N
                        cur_bias_buf, bias_phase = _get_bufidx_phase(
                            bias_tile_cnt, NUM_TMEM_BUFFERS
                        )
                        tlx.barrier_wait(bias_empty_bars[cur_bias_buf], bias_phase ^ 1)
                        dsize_b: tl.constexpr = tlx.size_of(tlx.dtype_of(bias_desc))
                        tlx.barrier_expect_bytes(
                            bias_full_bars[cur_bias_buf], dsize_b * BLOCK_SIZE_N
                        )
                        tlx.async_descriptor_load(
                            bias_desc,
                            bias_smem[cur_bias_buf],
                            [offs_bn_b],
                            bias_full_bars[cur_bias_buf],
                        )
                        bias_tile_cnt += 1

                    smem_accum_cnt = _process_tile_producer_inner(
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        num_mn_tiles=num_mn_tiles,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        k_tile_start=k_tile_start,
                        k_tile_end=k_tile_end,
                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                        a_desc=a_desc,
                        b_desc=b_desc,
                        buffers_A=buffers_A,
                        buffers_B=buffers_B,
                        A_smem_full_bars=A_smem_full_bars,
                        B_smem_full_bars=B_smem_full_bars,
                        A_smem_empty_bars=A_smem_empty_bars,
                        smem_accum_cnt=smem_accum_cnt,
                        NUM_CTAS=NUM_CTAS,
                        cluster_cta_rank=cluster_cta_rank,
                        A_ROW_MAJOR=A_ROW_MAJOR,
                        B_ROW_MAJOR=B_ROW_MAJOR,
                    )

                tile_id += NUM_SMS


def _is_tma_compatible(t):
    """Check if tensor strides are compatible with TMA descriptors.
    TMA requires 16-byte aligned strides. For bf16/fp16 (2 bytes),
    all strides except the innermost (stride=1) must be divisible by 8.
    """
    for s in t.stride():
        if s != 1 and s % 8 != 0:
            return False
    return True


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_warps=4,
            num_stages=3,
        ),
    ]
    if DISABLE_AUTOTUNE
    else [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_warps=4,
            num_stages=3,
        )
        for BM in [64, 128]
        for BN in [64, 128]
        for BK in [32, 64]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel_fallback(
    a_ptr,
    b_ptr,
    c_ptr,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Standard GEMM kernel using tl.load/tl.store for unaligned strides."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N)
        acc += bias[None, :].to(tl.float32)

    c = acc.to(OUTPUT_DTYPE)
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@torch.library.custom_op("ads_mkl::tlx_addmm", mutates_args=())
def _tlx_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    if b is not None:
        assert b.shape[0] == w.shape[1], "Bias dimension must match output columns"
    M, K = x.shape
    K, N = w.shape

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Fallback to tl.load kernel when strides are not TMA-compatible
    if not _is_tma_compatible(x) or not _is_tma_compatible(w):
        has_bias = b is not None

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        _matmul_kernel_fallback[grid](
            x,
            w,
            out,
            b if has_bias else x,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            out.stride(0),
            out.stride(1),
            HAS_BIAS=has_bias,
            OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
        )
        return out

    # Detect column-major inputs.
    # A column-major (M, K) tensor has strides (1, M); its .T is row-major (K, M).
    a_row_major = x.is_contiguous()
    b_row_major = w.is_contiguous()

    dummy_block = [1, 1]
    if not a_row_major:
        x_t = x.T  # (K, M) with strides (M, 1) — row-major
        a_desc = TensorDescriptor(x_t, x_t.shape, x_t.stride(), dummy_block)
    else:
        a_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    if not b_row_major:
        w_t = w.T  # (N, K) with strides (K, 1) — row-major
        b_desc = TensorDescriptor(w_t, w_t.shape, w_t.stride(), dummy_block)
    else:
        b_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    c_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)
    # Pass c as dummy workspace_desc. Pre_hook dynamically allocates
    # the right-sized workspace per config based on SPLIT_K.
    workspace_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    NUM_SMS = _get_num_sms()

    def grid(META):
        NUM_CTAS = META["NUM_CTAS"]
        num_pid_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
        num_pid_n = triton.cdiv(N, META["BLOCK_SIZE_N"])
        # Pad num_pid_m to multiple of NUM_CTAS so CTA clusters tile evenly along M.
        num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
        mn_tiles = num_pid_m * num_pid_n
        total_tiles = mn_tiles * META["SPLIT_K"]
        return (min(NUM_SMS, total_tiles),)

    has_bias = b is not None
    if has_bias:
        bias_desc = TensorDescriptor(b, b.shape, b.stride(), [1])
    else:
        bias_dummy = torch.empty(256, device=x.device, dtype=x.dtype)
        bias_desc = TensorDescriptor(
            bias_dummy, bias_dummy.shape, bias_dummy.stride(), [1]
        )

    matmul_kernel_tma_ws_blackwell[grid](
        a_desc,
        b_desc,
        c_desc,
        workspace_desc,
        bias_desc,
        M,
        N,
        K,
        A_ROW_MAJOR=a_row_major,
        B_ROW_MAJOR=b_row_major,
        NUM_SMS=NUM_SMS,
        HAS_BIAS=has_bias,
    )
    return out


# ─── Public API ──────────────────────────────────────────────────────


def addmm(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """GEMM + optional bias: output = x @ w + b.

    Args:
        x: [M, K] input tensor
        w: [K, N] weight tensor
        b: [N] optional bias vector
        config: explicit autotune config dict. If None, uses heuristic.
                Keys: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
                NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS, NUM_MMA_GROUPS,
                EPILOGUE_SUBTILE, NUM_CTAS, SPLIT_K, INTERLEAVE_EPILOGUE

    Returns:
        [M, N] output tensor
    """
    if config:
        return _addmm_with_config(x, w, b, config)
    return _tlx_matmul(x, w, b)


def _addmm_with_config(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    config: dict,
) -> torch.Tensor:
    """Run addmm with an explicit config, bypassing autotune/heuristic.

    Follows the same pattern as blackwell_gemm_ws.py tutorials/matmul():
    1. Pop pre_hook and ctas_per_cga from config
    2. Run pre_hook to set TensorDescriptor block sizes
    3. Call kernel via .fn[] with explicit config (bypass autotune)
    4. Handle Split-K reduction if needed
    """
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    if b is not None:
        assert b.shape[0] == w.shape[1], "Bias dimension must match output columns"
    M, K = x.shape
    K, N = w.shape

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    a_row_major = x.is_contiguous()
    b_row_major = w.is_contiguous()

    dummy_block = [1, 1]
    if not a_row_major:
        x_t = x.T
        a_desc = TensorDescriptor(x_t, x_t.shape, x_t.stride(), dummy_block)
    else:
        a_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    if not b_row_major:
        w_t = w.T
        b_desc = TensorDescriptor(w_t, w_t.shape, w_t.stride(), dummy_block)
    else:
        b_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    c_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    has_bias = b is not None
    if has_bias:
        bias_desc = TensorDescriptor(b, b.shape, b.stride(), [1])
    else:
        bias_dummy = torch.empty(256, device=x.device, dtype=x.dtype)
        bias_desc = TensorDescriptor(
            bias_dummy, bias_dummy.shape, bias_dummy.stride(), [1]
        )

    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    NUM_SMS = _get_num_sms()

    # Copy config to avoid mutating caller's dict
    cfg = dict(config)
    ctas_per_cga = cfg.pop("ctas_per_cga", None)
    pre_hook = cfg.pop("pre_hook", None)
    split_k = cfg.get("SPLIT_K", 1)

    # Allocate workspace for Split-K
    if split_k > 1:
        workspace = torch.empty((split_k * M, N), device=x.device, dtype=x.dtype)
        workspace_desc = TensorDescriptor(
            workspace, workspace.shape, workspace.stride(), dummy_block
        )
    else:
        workspace_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    # Run pre_hook to set TensorDescriptor block sizes
    hook_args = {
        "a_desc": a_desc,
        "b_desc": b_desc,
        "c_desc": c_desc,
        "workspace_desc": workspace_desc,
        "bias_desc": bias_desc,
        "M": M,
        "N": N,
        "K": K,
        "A_ROW_MAJOR": a_row_major,
        "B_ROW_MAJOR": b_row_major,
        **cfg,
    }
    if pre_hook:
        pre_hook(hook_args)
    else:
        matmul_tma_set_block_size_hook(hook_args)

    # Compute grid
    NUM_CTAS = cfg.get("NUM_CTAS", 1)
    num_pid_m = triton.cdiv(M, cfg["BLOCK_SIZE_M"])
    num_pid_n = triton.cdiv(N, cfg["BLOCK_SIZE_N"])
    num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
    total_tiles = num_pid_m * num_pid_n * split_k
    grid = (min(NUM_SMS, total_tiles),)

    # Call kernel with explicit config via .fn[] (bypass autotune)
    matmul_kernel_tma_ws_blackwell.fn[grid](
        a_desc,
        b_desc,
        c_desc,
        workspace_desc,
        bias_desc,
        M,
        N,
        K,
        A_ROW_MAJOR=a_row_major,
        B_ROW_MAJOR=b_row_major,
        NUM_SMS=NUM_SMS,
        HAS_BIAS=has_bias,
        ctas_per_cga=ctas_per_cga,
        **cfg,
    )

    # Split-K reduction
    if split_k > 1:
        reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
        _reduce_k_kernel[reduce_grid](
            workspace_desc.base,
            out,
            M,
            N,
            SPLIT_K=split_k,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[x.dtype],
            num_warps=4,
        )

    return out
