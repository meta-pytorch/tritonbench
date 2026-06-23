"""
Triton kernel for scaled_mm (FP8 GEMM) with AutoWS (Automatic Warp Specialization).

Based on _scaled_mm_v2 semantics from PyTorch. Supports TensorWise, RowWise,
MXFP8 (1x32 e8m0 microscaling), and fp32 blockwise (symmetric 1x128/1x128 and
DeepSeek 1x128/128x128) scaling recipes with a Blackwell warp-specialized
persistent TMA matmul.

Backend support:
  - Blackwell (sm_100+): AutoWS persistent loop. MXFP8 (1x32 e8m0) lowers via
    tl.dot_scaled -> ttng.tc_gen5_mma_scaled (warp-specialized). The fp32
    blockwise modes (symmetric 1x128/1x128 and DeepSeek 1x128/128x128) use
    tl.dot + an in-loop fp32 rescale (no WS). TensorWise/RowWise scale in the
    epilogue.
  - Hopper (sm_90): plain persistent loop (no warp_specialize),
    EPILOGUE_SUBTILE forced to 1. MXFP8 is rejected (tl.dot_scaled MMA is
    Blackwell-only); the fp32 paths work but are unoptimized.

The tl.range(..., flatten=...) loop-flattening pragma is intentionally left
off on both backends (FLATTEN=False); enabling it regressed perf in this
kernel.

TMA feasibility by scaling mode:
  - TensorWise: scalar scales — NOT feasible (single element load)
  - RowWise: 1D [M] / [N] vectors — feasible via 1D host TensorDescriptor
  - BlockWise MXFP8: feasible via 5D (1, M//128, K//VEC_SIZE//4, 2, 256) repack
    of the cuBLAS layout (32*4*4 == 2*256). Loaded tile is reshaped to
    (REP_M, REP_K, 32, 4, 4), transposed (0,3,2,1,4) and flattened to
    (BLOCK_M, BLOCK_K//VEC_SIZE) for tl.dot_scaled.

Reference implementations:
  - tritonbench/operators/fp8_gemm/persistent.py (blackwell_persistent_tma_kernel)
  - tritonbench/operators/gemm/warp_spec_persistent_matmul.py (Meta WS / upstream WS paths)
  - torch/_inductor/kernel/templates/triton_blackwell_ws_persistent_tma_mm.py.jinja
  - tritonbench/operators/fb/fp8_grouped_gemm/kernels.py (block-scale TMA pattern)
  - triton_mtia/third_party/triton/python/tutorials/10-block-scaled-matmul.py (block scaling)
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# ---------------------------------------------------------------------------
# ScalingType enum (mirrors torch.nn.functional.ScalingType / at::blas::ScalingType)
#
# Wrapped in tl.constexpr so that @triton.jit kernels can reference them as
# module globals (jit kernels only see globals that are tl.constexpr).
# ---------------------------------------------------------------------------
TENSORWISE = tl.constexpr(0)
ROWWISE = tl.constexpr(1)
# MXFP8 == at::blas::ScalingType.BlockWise1x32: 1x32 e8m0 microscaling, lowered
# via tl.dot_scaled -> ttng.tc_gen5_mma_scaled on Blackwell.
MXFP8 = tl.constexpr(3)
BLOCKWISE_1x128 = tl.constexpr(4)
BLOCKWISE_128x128 = tl.constexpr(5)

MXFP8_VEC_SIZE = 32


# ---------------------------------------------------------------------------
# Meta WS detection (deferred to first use for lazy-import compatibility)
# ---------------------------------------------------------------------------
_use_meta_ws_cached: bool | None = None


def _use_meta_ws() -> bool:
    global _use_meta_ws_cached
    if _use_meta_ws_cached is None:
        try:
            _use_meta_ws_cached = triton.knobs.nvidia.use_meta_ws  # type: ignore[attr-defined]
        except AttributeError:
            _use_meta_ws_cached = False
    return _use_meta_ws_cached


# ---------------------------------------------------------------------------
# Blackwell detection (deferred until CUDA is initialized at first launch)
# ---------------------------------------------------------------------------
_is_blackwell_cached: bool | None = None


def _is_blackwell() -> bool:
    global _is_blackwell_cached
    if _is_blackwell_cached is None:
        try:
            major, _ = torch.cuda.get_device_capability()
            _is_blackwell_cached = major >= 10
        except (AssertionError, RuntimeError):
            _is_blackwell_cached = False
    return _is_blackwell_cached


# ---------------------------------------------------------------------------
# Triton descriptor allocator
# ---------------------------------------------------------------------------
_triton_allocator_device: torch.device | None = None


def _ensure_triton_allocator(device: torch.device) -> None:
    """Register the process-global TMA descriptor allocator for this device."""
    global _triton_allocator_device
    if _triton_allocator_device == device:
        return

    # TMA descriptors require a global memory allocation. The stream parameter
    # is unused; torch.empty uses the current CUDA stream.
    def alloc_fn(size: int, alignment: int, stream: Optional[int]) -> torch.Tensor:
        return torch.empty(size, device=device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    _triton_allocator_device = device


# ---------------------------------------------------------------------------
# Autotuning configs
# ---------------------------------------------------------------------------
def _host_descriptor_pre_hook(nargs):
    """Resize host-side TMA TensorDescriptors to match autotune block sizes.

    Mutating block_shape in-place is the pattern used by
    tritonbench/kernels/triton_fused_attention.py.
    """
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    epilogue_subtile = nargs["EPILOGUE_SUBTILE"]

    a = nargs.get("a_desc")
    b = nargs.get("b_desc")
    c = nargs.get("c_desc")
    if isinstance(a, TensorDescriptor):
        a.block_shape = [block_m, block_k]
    if isinstance(b, TensorDescriptor):
        b.block_shape = [block_n, block_k]
    if isinstance(c, TensorDescriptor):
        c.block_shape = [block_m, block_n // epilogue_subtile]

    sa = nargs.get("scale_a_ptr")
    sb = nargs.get("scale_b_ptr")
    a_mode = nargs.get("SCALE_A_MODE", TENSORWISE)
    b_mode = nargs.get("SCALE_B_MODE", TENSORWISE)
    if a_mode == ROWWISE and b_mode == ROWWISE:
        if isinstance(sa, TensorDescriptor):
            sa.block_shape = [block_m]
        if isinstance(sb, TensorDescriptor):
            sb.block_shape = [block_n // epilogue_subtile]
    elif a_mode == MXFP8 and b_mode == MXFP8:
        vec_size = nargs["VEC_SIZE"]
        rep_k = triton.cdiv(block_k // vec_size, 4)
        if isinstance(sa, TensorDescriptor):
            sa.block_shape = [1, block_m // 128, rep_k, 2, 256]
        if isinstance(sb, TensorDescriptor):
            sb.block_shape = [1, block_n // 128, rep_k, 2, 256]
    # The fp32 blockwise modes -- symmetric 1x128/1x128 and DeepSeek
    # (1x128 / 128x128) -- use plain tensors, not TensorDescriptors, so there is
    # nothing to resize here.


def _get_autotune_configs():
    """Generate autotuning configs for scaled_mm with AutoWS."""
    configs = []
    block_configs = [
        # (BLOCK_M, BLOCK_N, BLOCK_K)
        (128, 128, 64),
        (128, 256, 64),
        (256, 128, 64),
        (128, 128, 128),
        (128, 256, 128),
        (256, 128, 128),
    ]

    # TWO_CTAS=True enables 2-CTA (cta_group::2) MMA via ctas_per_cga=(2,1,1) on
    # the plain-tl.dot path. _prune_configs offers it as a tuning option for
    # TensorWise/RowWise on Blackwell + Meta-WS, BM>=128 (see the RowWise caveat
    # there: the autotuner can mis-pick the 2-CTA RowWise config).
    #
    # ctas_per_cga is an FBTriton/Meta-WS-only triton.Config arg, so only emit the
    # 2-CTA configs when Meta-WS is available -- on OSS Triton _use_meta_ws() is
    # False and no ctas_per_cga config is ever constructed.
    two_cta_options = [False, True] if _use_meta_ws() else [False]
    for num_stages in [3, 4, 5]:
        for BLOCK_M, BLOCK_N, BLOCK_K in block_configs:
            for EPILOGUE_SUBTILE in [1, 2, 4]:
                for TWO_CTAS in two_cta_options:
                    extras = {"ctas_per_cga": (2, 1, 1)} if TWO_CTAS else {}
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_SIZE_M": BLOCK_M,
                                "BLOCK_SIZE_N": BLOCK_N,
                                "BLOCK_SIZE_K": BLOCK_K,
                                "GROUP_SIZE_M": 8,
                                "EPILOGUE_SUBTILE": EPILOGUE_SUBTILE,
                                "TWO_CTAS": TWO_CTAS,
                            },
                            num_stages=num_stages,
                            num_warps=4,
                            pre_hook=_host_descriptor_pre_hook,
                            **extras,
                        )
                    )

    return configs


def _prune_configs(configs, named_args, **kwargs):
    """Prune invalid configs based on WS mode, backend, and problem size."""
    pruned = []
    # Constexpr meta-params (SCALE_A_MODE, ...) are passed as keyword args, so
    # they arrive here in **kwargs, NOT in named_args (which holds only the
    # positional kernel args -- a_desc..K). Fall back to named_args/default.
    a_mode = kwargs.get("SCALE_A_MODE", named_args.get("SCALE_A_MODE", TENSORWISE))
    b_mode = kwargs.get("SCALE_B_MODE", named_args.get("SCALE_B_MODE", TENSORWISE))
    vec_size = kwargs.get("VEC_SIZE", named_args.get("VEC_SIZE", 0))
    is_blackwell = _is_blackwell()
    is_tensorwise = a_mode == TENSORWISE and b_mode == TENSORWISE
    is_rowwise = a_mode == ROWWISE and b_mode == ROWWISE
    is_mxfp8 = a_mode == MXFP8 and b_mode == MXFP8
    is_blockwise_1x128 = a_mode == BLOCKWISE_1x128 and b_mode == BLOCKWISE_1x128
    is_deepseek = a_mode == BLOCKWISE_1x128 and b_mode == BLOCKWISE_128x128
    for config in configs:
        num_warps = config.num_warps
        if num_warps < 4:
            continue
        M = named_args.get("M", 0)
        N = named_args.get("N", 0)
        if M and config.kwargs["BLOCK_SIZE_M"] > M * 2:
            continue
        if N and config.kwargs["BLOCK_SIZE_N"] > N * 2:
            continue
        # 2-CTA (cta_group::2): a tuning option for the epilogue-scaled modes
        # (TensorWise, RowWise) -- like CUTLASS, which uses the same 2-SM cluster
        # for per-tensor and per-token/-channel.
        # Requires Blackwell + Meta-WS + BLOCK_M >= 128.
        if config.kwargs.get("TWO_CTAS", False):
            if not (is_tensorwise or is_rowwise):
                continue
            if not (is_blackwell and _use_meta_ws()):
                continue
            if config.kwargs["BLOCK_SIZE_M"] < 128:
                continue
        # MXFP8 block scaling requires BLOCK_K >= VEC_SIZE * 4
        if is_mxfp8 and vec_size > 0:
            if config.kwargs["BLOCK_SIZE_K"] < vec_size * 4:
                continue
        # DeepSeek blockwise (1x128/128x128): one 128-wide scale group per K
        # tile and one B N-block per tile -> pin BLOCK_K and BLOCK_N to 128.
        if is_deepseek:
            if config.kwargs["BLOCK_SIZE_K"] != 128:
                continue
            if config.kwargs["BLOCK_SIZE_N"] != 128:
                continue
            # BLOCK_M=128 crashes the Canonicalizer for the DeepSeek path
            # (symmetric 1x128 compiles fine at 128). The autotuner cannot skip a
            # config that fails to *compile* (only OOM), so exclude it here.
            if config.kwargs["BLOCK_SIZE_M"] == 128:
                continue
        # Symmetric fp32 blockwise (1x128/1x128): one 128-wide K group per tile
        # so kb == ki for the in-loop rescale (BLOCK_N is free, sb is a vector).
        if is_blockwise_1x128:
            if config.kwargs["BLOCK_SIZE_K"] != 128:
                continue
        # num_stages gating (the autotuner only skips OOM configs, not ones that
        # fail to compile, so gate per mode here):
        #   DeepSeek / symmetric 1x128 -- the WS-off in-loop fp32 rescale does not
        #     hold up at num_stages>=4 (symmetric OOMs SMEM from the sb-over-M
        #     broadcast; DeepSeek crashes the Canonicalizer), so cap at 3.
        #   MXFP8 -- try every num_stages; the autotuner drops any that overflow.
        #   TensorWise / RowWise -- the [4, 5] tuning set.
        if is_deepseek or is_blockwise_1x128:
            if config.num_stages > 3:
                continue
        elif not is_mxfp8 and config.num_stages < 4:
            continue
        # Hopper: epilogue subtiling forced off (subtile=1). The reshape/permute/
        # split pattern is a Blackwell register-pressure relief that doesn't carry
        # over cleanly without WS scheduling.
        if not is_blackwell and config.kwargs["EPILOGUE_SUBTILE"] != 1:
            continue
        pruned.append(config)
    return pruned


# ---------------------------------------------------------------------------
# Helper: compute tile pid with group-M swizzling
# ---------------------------------------------------------------------------
@triton.jit
def _compute_pid(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    GROUP_SIZE_M: tl.constexpr,
):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ---------------------------------------------------------------------------
# Helper: subtile accumulator for reduced register pressure in epilogue
# ---------------------------------------------------------------------------
@triton.jit
def _subtile_accumulator(
    acc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_FACTOR: tl.constexpr,
):
    tl.static_assert(SUBTILE_FACTOR > 0, "SUBTILE_FACTOR must be positive")
    tl.static_assert(
        (SUBTILE_FACTOR & (SUBTILE_FACTOR - 1)) == 0,
        "SUBTILE_FACTOR must be a power of 2",
    )
    if SUBTILE_FACTOR == 1:
        return (acc,)
    else:
        tl.static_assert(BLOCK_N % 2 == 0)
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        left, right = tl.split(acc)
        left_sub = _subtile_accumulator(
            left, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2
        )
        right_sub = _subtile_accumulator(
            right, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2
        )
        return left_sub + right_sub


# ---------------------------------------------------------------------------
# Main kernel: scaled_mm with AutoWS
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def scaled_mm_autows_kernel(
    a_desc,
    b_desc,
    c_desc,
    scale_a_ptr,
    scale_b_ptr,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    # Per-operand scaling recipes (real ScalingType values). The kernel selects
    # its algorithm from the (SCALE_A_MODE, SCALE_B_MODE) pair:
    #   (TENSORWISE, TENSORWISE)             -> scalar epilogue scale
    #   (ROWWISE, ROWWISE)                   -> per-row/col epilogue scale
    #   (MXFP8, MXFP8)                       -> 1x32 e8m0 via tl.dot_scaled (Blackwell)
    #   (BLOCKWISE_1x128, BLOCKWISE_1x128)   -> symmetric fp32 1x128 in-loop rescale
    #   (BLOCKWISE_1x128, BLOCKWISE_128x128) -> DeepSeek in-loop fp32 rescale
    SCALE_A_MODE: tl.constexpr,
    SCALE_B_MODE: tl.constexpr,
    # Use host-side TMA for RowWise scale loads (TensorWise = scalar load,
    # MXFP8 uses host-side 5D TMA via the (1, ..., 2, 256) repack)
    USE_SCALE_TMA: tl.constexpr,
    # Block scaling vector size (e.g. 32 for MXFP8, 0 when unused)
    VEC_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    # Backend gating: Blackwell uses AutoWS + flattened tile loop;
    # Hopper falls back to a plain persistent loop (no WS, no flatten).
    FLATTEN: tl.constexpr = True,
    WARP_SPECIALIZE: tl.constexpr = True,
    SEPARATE_EPILOGUE_STORE: tl.constexpr = True,
    # 2-CTA (cta_group::2) tcgen05 MMA across a contiguous CTA pair. Set by
    # autotune configs that also carry ctas_per_cga=(2,1,1); only the plain
    # tl.dot path uses it (compiler splits B per CTA + inserts cross-CTA sync).
    # _prune_configs offers it as a tuning option for TensorWise/RowWise on
    # Blackwell + Meta-WS, BM>=128.
    TWO_CTAS: tl.constexpr = False,
):
    """
    Persistent TMA matmul kernel for FP8 scaled_mm with warp specialization.
    Computes: C = (A @ B^T) * scale_a * scale_b
    """
    # Compile-time algorithm selection from the per-operand recipe pair.
    IS_TENSORWISE: tl.constexpr = (SCALE_A_MODE == TENSORWISE) and (
        SCALE_B_MODE == TENSORWISE
    )
    IS_ROWWISE: tl.constexpr = (SCALE_A_MODE == ROWWISE) and (SCALE_B_MODE == ROWWISE)
    IS_MXFP8: tl.constexpr = (SCALE_A_MODE == MXFP8) and (SCALE_B_MODE == MXFP8)
    IS_BLOCKWISE_1x128: tl.constexpr = (SCALE_A_MODE == BLOCKWISE_1x128) and (
        SCALE_B_MODE == BLOCKWISE_1x128
    )
    IS_DEEPSEEK: tl.constexpr = (SCALE_A_MODE == BLOCKWISE_1x128) and (
        SCALE_B_MODE == BLOCKWISE_128x128
    )

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # 2-CTA pairs tile the M axis, so round the M tile count up to even.
    if TWO_CTAS:
        num_pid_m = (num_pid_m + 1) // 2 * 2
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    if IS_TENSORWISE:
        scale_a_scalar = tl.load(scale_a_ptr)
        scale_b_scalar = tl.load(scale_b_ptr)

    # ROWWISE host-side TMA: scale_a_ptr / scale_b_ptr are passed in as
    # TensorDescriptors when USE_SCALE_TMA -- no in-kernel descriptor build.

    if IS_MXFP8:
        # Host-side 5D TMA on the (1, M//128, K//VEC_SIZE//4, 2, 256) repack
        # of the cuBLAS block-scale layout (32*4*4 == 2*256). Same trick used
        # by tritonbench/operators/fb/fp8_grouped_gemm/kernels.py.
        REP_M: tl.constexpr = BLOCK_SIZE_M // 128
        REP_N: tl.constexpr = BLOCK_SIZE_N // 128
        REP_K: tl.constexpr = triton.cdiv(BLOCK_SIZE_K // VEC_SIZE, 4)

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE,
        separate_epilogue_store=SEPARATE_EPILOGUE_STORE,
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        if IS_MXFP8:
            # Block scaled K-loop: 5D TMA load of scales, fused into dot_scaled.
            scale_m_tile = pid_m * REP_M
            scale_n_tile = pid_n * REP_N
            SCALE_K_PER_TILE: tl.constexpr = BLOCK_SIZE_K // VEC_SIZE

            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K
                a_tile = a_desc.load([offs_am, offs_k])
                b_tile = b_desc.load([offs_bn, offs_k])

                scale_k_tile = ki * REP_K
                sa_packed = scale_a_ptr.load([0, scale_m_tile, scale_k_tile, 0, 0])
                sb_packed = scale_b_ptr.load([0, scale_n_tile, scale_k_tile, 0, 0])
                sa = (
                    sa_packed.reshape(REP_M, REP_K, 32, 4, 4)
                    .trans(0, 3, 2, 1, 4)
                    .reshape(BLOCK_SIZE_M, SCALE_K_PER_TILE)
                )
                sb = (
                    sb_packed.reshape(REP_N, REP_K, 32, 4, 4)
                    .trans(0, 3, 2, 1, 4)
                    .reshape(BLOCK_SIZE_N, SCALE_K_PER_TILE)
                )

                accumulator = tl.dot_scaled(
                    a_tile, sa, "e4m3", b_tile.T, sb, "e4m3", accumulator
                )
        elif IS_DEEPSEEK:
            # DeepSeek-style blockwise: 1x128 scales for A, 128x128 for B.
            # Plain fp32 scales (NOT MX); they vary along K so they cannot factor
            # out to the epilogue -- each 128-wide K group's partial dot is
            # rescaled by sa[m,kb]*sb[n//128,kb] before accumulation.
            # _prune_configs pins BLOCK_SIZE_K=128 (one group per K tile) and
            # BLOCK_SIZE_N=128 (one B N-block per tile), so kb==ki and the B
            # N-block index == pid_n. scale_a is row-major [M, K//128];
            # scale_b is row-major [N//128, K//128].
            num_k_groups = K // 128
            offs_m_scale = offs_am + tl.arange(0, BLOCK_SIZE_M)
            mask_m_scale = offs_m_scale < M
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K
                a_tile = a_desc.load([offs_am, offs_k])
                b_tile = b_desc.load([offs_bn, offs_k])
                partial = tl.dot(
                    a_tile,
                    b_tile.T,
                    out_dtype=tl.float32,
                    allow_tf32=True,
                )
                sa = tl.load(
                    scale_a_ptr + offs_m_scale * num_k_groups + ki,
                    mask=mask_m_scale,
                    other=0.0,
                )
                sb = tl.load(scale_b_ptr + pid_n * num_k_groups + ki)
                accumulator += partial * sa[:, None] * sb
        elif IS_BLOCKWISE_1x128:
            # Symmetric fp32 blockwise: 1x128 scales for BOTH A and B (arbitrary
            # fp32, one per row per 128-wide K group). Like DeepSeek but B's scale
            # is a per-N-row vector instead of a per-128xN-block scalar, so each
            # K group's partial dot is rescaled by sa[m,kb]*sb[n,kb]. NOT MX --
            # these fp32 scales cannot use tl.dot_scaled. _prune_configs pins
            # BLOCK_SIZE_K=128 (one group per K tile) so kb==ki. scale_a is
            # row-major [M, K//128]; scale_b is row-major [N, K//128].
            num_k_groups = K // 128
            offs_m_scale = offs_am + tl.arange(0, BLOCK_SIZE_M)
            offs_n_scale = offs_bn + tl.arange(0, BLOCK_SIZE_N)
            mask_m_scale = offs_m_scale < M
            mask_n_scale = offs_n_scale < N
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K
                a_tile = a_desc.load([offs_am, offs_k])
                b_tile = b_desc.load([offs_bn, offs_k])
                partial = tl.dot(
                    a_tile,
                    b_tile.T,
                    out_dtype=tl.float32,
                    allow_tf32=True,
                )
                sa = tl.load(
                    scale_a_ptr + offs_m_scale * num_k_groups + ki,
                    mask=mask_m_scale,
                    other=0.0,
                )
                sb = tl.load(
                    scale_b_ptr + offs_n_scale * num_k_groups + ki,
                    mask=mask_n_scale,
                    other=0.0,
                )
                accumulator += partial * sa[:, None] * sb[None, :]
        else:
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K
                a_tile = a_desc.load([offs_am, offs_k])
                b_tile = b_desc.load([offs_bn, offs_k])
                # Triton TR011: keep Tensor Core TF32 behavior explicit.
                # two_ctas is an FBTriton-only tl.dot kwarg -- only pass it on the
                # 2-CTA path (TWO_CTAS is set only under Meta-WS) so OSS Triton,
                # which has no such kwarg, never receives it.
                if TWO_CTAS:
                    accumulator = tl.dot(
                        a_tile,
                        b_tile.T,
                        accumulator,
                        out_dtype=tl.float32,
                        allow_tf32=True,
                        two_ctas=True,
                    )
                else:
                    accumulator = tl.dot(
                        a_tile,
                        b_tile.T,
                        accumulator,
                        out_dtype=tl.float32,
                        allow_tf32=True,
                    )

            if IS_ROWWISE:
                if USE_SCALE_TMA:
                    sa = scale_a_ptr.load([offs_am])
                else:
                    offs_scale_m = offs_am + tl.arange(0, BLOCK_SIZE_M)
                    mask_m = offs_scale_m < M
                    sa = tl.load(scale_a_ptr + offs_scale_m, mask=mask_m, other=0.0)

        # Epilogue: store with subtiling via TMA
        tile_id_c += NUM_SMS
        pid_m_c, pid_n_c = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )
        offs_cm = pid_m_c * BLOCK_SIZE_M
        offs_cn = pid_n_c * BLOCK_SIZE_N

        subtiles = _subtile_accumulator(
            accumulator, BLOCK_SIZE_M, BLOCK_SIZE_N, EPILOGUE_SUBTILE
        )
        sub_n: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
        for i in tl.static_range(EPILOGUE_SUBTILE):
            subtile = subtiles[i]
            if IS_TENSORWISE:
                subtile *= scale_a_scalar * scale_b_scalar
            elif IS_ROWWISE:
                if USE_SCALE_TMA:
                    sb = scale_b_ptr.load([offs_bn + i * sub_n])
                else:
                    offs_scale_n = offs_bn + i * sub_n + tl.arange(0, sub_n)
                    mask_n = offs_scale_n < N
                    sb = tl.load(scale_b_ptr + offs_scale_n, mask=mask_n, other=0.0)
                subtile *= sa[:, None] * sb[None, :]
            subtile = subtile.to(OUT_DTYPE)
            c_desc.store([offs_cm, offs_cn + i * sub_n], subtile)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def scaled_mm_autows(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    scale_a_mode: int = TENSORWISE,
    scale_b_mode: Optional[int] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Compute scaled FP8 matmul: C = (A @ B^T) * scale_a * scale_b

    Args:
        a: [M, K] FP8 tensor (e.g., float8_e4m3fn)
        b: [N, K] FP8 tensor (row-major; computes A @ B^T)
        scale_a: Scale for A -- scalar (TensorWise), [M] (RowWise),
                 flat blocked e8m0 [M*K//32] (MXFP8), or [M, K//128] fp32
                 (BLOCKWISE_1x128 symmetric / DeepSeek).
        scale_b: Scale for B -- scalar (TensorWise), [N] (RowWise),
                 flat blocked e8m0 [N*K//32] (MXFP8), [N, K//128] fp32
                 (BLOCKWISE_1x128 symmetric), or [N//128, K//128] fp32
                 (BLOCKWISE_128x128 DeepSeek).
        scale_a_mode: per-operand recipe for A (TENSORWISE / ROWWISE / MXFP8 /
                 BLOCKWISE_1x128).
        scale_b_mode: per-operand recipe for B; defaults to scale_a_mode
                 (symmetric). Supported pairs: (TENSORWISE, TENSORWISE),
                 (ROWWISE, ROWWISE), (MXFP8, MXFP8) 1x32 e8m0,
                 (BLOCKWISE_1x128, BLOCKWISE_1x128) symmetric fp32, and
                 (BLOCKWISE_1x128, BLOCKWISE_128x128) DeepSeek.
        out_dtype: Output dtype (default bfloat16)

    Returns:
        C: [M, N] tensor in out_dtype
    """
    if scale_b_mode is None:
        scale_b_mode = scale_a_mode

    assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D"
    M, K = a.shape
    N, K_b = b.shape
    assert K == K_b, f"K mismatch: {K} vs {K_b}"

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)

    _dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    tl_out_dtype = _dtype_map[out_dtype]

    device_props = torch.cuda.get_device_properties(a.device)
    num_sms = device_props.multi_processor_count
    is_blackwell = device_props.major >= 10

    is_rowwise = scale_a_mode == ROWWISE and scale_b_mode == ROWWISE
    is_mxfp8 = scale_a_mode == MXFP8 and scale_b_mode == MXFP8
    is_blockwise_1x128 = (
        scale_a_mode == BLOCKWISE_1x128 and scale_b_mode == BLOCKWISE_1x128
    )
    is_deepseek = scale_a_mode == BLOCKWISE_1x128 and scale_b_mode == BLOCKWISE_128x128

    assert not is_mxfp8 or is_blackwell, (
        "MXFP8 blockwise requires Blackwell (sm_100+); tl.dot_scaled MMA is "
        "Blackwell-only."
    )

    use_scale_tma = is_rowwise
    vec_size = MXFP8_VEC_SIZE if is_mxfp8 else 0

    dummy_block_2d = [1, 1]
    a_arg = TensorDescriptor(a, a.shape, a.stride(), dummy_block_2d)
    b_arg = TensorDescriptor(b, b.shape, b.stride(), dummy_block_2d)
    c_arg = TensorDescriptor(c, c.shape, c.stride(), dummy_block_2d)

    # ROWWISE: pass host-side TMA descriptors. block_shape gets resized per-
    # autotune-config by `_host_descriptor_pre_hook`; the value here is just
    # an initial placeholder.
    if use_scale_tma:
        scale_a_arg = TensorDescriptor.from_tensor(scale_a, [128])
        scale_b_arg = TensorDescriptor.from_tensor(scale_b, [128])
    elif is_mxfp8:
        scale_a_arg = TensorDescriptor.from_tensor(
            scale_a.reshape(1, M // 128, K // vec_size // 4, 2, 256),
            [1, 1, 1, 2, 256],
        )
        scale_b_arg = TensorDescriptor.from_tensor(
            scale_b.reshape(1, N // 128, K // vec_size // 4, 2, 256),
            [1, 1, 1, 2, 256],
        )
    elif is_deepseek or is_blockwise_1x128:
        # Plain fp32 2D scales, pointer-loaded (no TMA, no MX repack). Force
        # row-major so in-kernel pointer arithmetic (row * K//128 + kb) holds.
        #   DeepSeek 1x128(A)+128x128(B): scale_a [M, K//128], scale_b [N//128, K//128]
        #   symmetric 1x128/1x128:        scale_a [M, K//128], scale_b [N, K//128]
        scale_a_arg = scale_a.contiguous()
        scale_b_arg = scale_b.contiguous()
    else:
        scale_a_arg = scale_a
        scale_b_arg = scale_b

    def grid(META):
        num_pid_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
        if META.get("TWO_CTAS", False):
            num_pid_m = triton.cdiv(num_pid_m, 2) * 2
        num_tiles = num_pid_m * triton.cdiv(N, META["BLOCK_SIZE_N"])
        return (min(num_sms, num_tiles),)

    _ensure_triton_allocator(a.device)

    # DeepSeek and symmetric fp32 1x128 rescale inside the K-loop (dot -> *sa*sb
    # -> accumulate). That in-loop compute chain is not partitionable by the
    # warp-specialization pass (it trips "multiple cross-partition producers"),
    # so run them through the plain persistent loop (no WS) like the Hopper path.
    #
    # MXFP8 uses tl.dot_scaled -> ttng.tc_gen5_mma_scaled. The scaled-dot lowering
    # works, but the AutoWS partition scheduler (NVGPUWarpSpecialization /
    # tritongpu-automatic-warp-specialization) does not yet partition a
    # tc_gen5_mma_scaled loop, so WS is disabled for MXFP8 for now (it still emits
    # tc_gen5_mma_scaled via the plain persistent loop). Enabling WS on the
    # scaled MMA is the AutoWS-compiler follow-up.
    ws_ok = not is_deepseek and not is_blockwise_1x128 and not is_mxfp8
    warp_specialize = is_blackwell and ws_ok
    separate_epilogue_store = is_blackwell and ws_ok

    scaled_mm_autows_kernel[grid](
        a_arg,
        b_arg,
        c_arg,
        scale_a_arg,
        scale_b_arg,
        M,
        N,
        K,
        NUM_SMS=num_sms,
        SCALE_A_MODE=scale_a_mode,
        SCALE_B_MODE=scale_b_mode,
        USE_SCALE_TMA=use_scale_tma,
        VEC_SIZE=vec_size,
        OUT_DTYPE=tl_out_dtype,
        FLATTEN=False,
        WARP_SPECIALIZE=warp_specialize,
        SEPARATE_EPILOGUE_STORE=separate_epilogue_store,
    )

    return c


# ---------------------------------------------------------------------------
# Convenience: match _scaled_mm_v2 calling convention
# ---------------------------------------------------------------------------
def scaled_mm_v2(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: list[torch.Tensor],
    recipe_a: list[int],
    scale_b: list[torch.Tensor],
    recipe_b: list[int],
    out_dtype: torch.dtype = torch.bfloat16,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """
    Triton implementation matching _scaled_mm_v2 signature.

    Currently supports:
      - TensorWise + TensorWise
      - RowWise + RowWise
      - BlockWise 1x128 (MXFP8) + BlockWise 1x128
      - BlockWise 1x128 (DeepSeek) + BlockWise 128x128

    Args follow the _scaled_mm_v2 convention:
      scale_a/scale_b: list of scale tensors
      recipe_a/recipe_b: list of ScalingType int values
    """
    if use_fast_accum:
        raise ValueError("scaled_mm_autows does not support use_fast_accum=True")

    assert len(recipe_a) == 1 and len(recipe_b) == 1, (
        "Multi-level scaling (NVFP4/MX) not yet supported"
    )
    r_a, r_b = recipe_a[0], recipe_b[0]

    supported = {TENSORWISE, ROWWISE, MXFP8, BLOCKWISE_1x128, BLOCKWISE_128x128}
    assert r_a in supported, f"Unsupported recipe_a: {r_a}"
    assert r_b in supported, f"Unsupported recipe_b: {r_b}"

    return scaled_mm_autows(
        mat_a,
        mat_b,
        scale_a[0],
        scale_b[0],
        scale_a_mode=r_a,
        scale_b_mode=r_b,
        out_dtype=out_dtype,
    )


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        exit(0)

    device = torch.device("cuda")
    M, N, K = 1024, 2048, 512

    print(f"scaled_mm AutoWS test: M={M}, N={N}, K={K}")
    print(f"Meta WS available: {_use_meta_ws()}")

    a_fp32 = torch.randn(M, K, device=device)
    b_fp32 = torch.randn(N, K, device=device)

    try:
        a_fp8 = a_fp32.to(torch.float8_e4m3fn)
        b_fp8 = b_fp32.to(torch.float8_e4m3fn)
    except (RuntimeError, TypeError):
        print("float8_e4m3fn not available on this device")
        exit(0)

    # bf16 matmul tolerance: FP8 quantization + bf16 accumulation slack.
    RTOL = 1e-2
    ATOL = 1e-2

    def _check_close(name: str, got: torch.Tensor, expected: torch.Tensor) -> bool:
        if not torch.allclose(got.float(), expected.float(), rtol=RTOL, atol=ATOL):
            diff = (got.float() - expected.float()).abs()
            print(
                f"  [FAIL] {name}: max_abs={diff.max().item():.4g}, "
                f"mean_abs={diff.mean().item():.4g}"
            )
            return False
        print(f"  [OK]   {name}: within rtol={RTOL}, atol={ATOL}")
        return True

    all_passed = True

    # --- Test 1: TensorWise scaling ---
    print("\n--- TensorWise scaling ---")
    scale_a = torch.tensor([1.0], dtype=torch.float32, device=device)
    scale_b = torch.tensor([1.0], dtype=torch.float32, device=device)

    c_triton = scaled_mm_autows(a_fp8, b_fp8, scale_a, scale_b, TENSORWISE)
    c_ref = torch._scaled_mm(
        a_fp8,
        b_fp8.T,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )
    all_passed &= _check_close("TensorWise", c_triton, c_ref)

    # --- Test 2: RowWise scaling (uses 1D TMA for scales) ---
    print("\n--- RowWise scaling ---")
    scale_a_row = torch.ones(M, dtype=torch.float32, device=device)
    scale_b_row = torch.ones(N, dtype=torch.float32, device=device)

    c_triton_row = scaled_mm_autows(a_fp8, b_fp8, scale_a_row, scale_b_row, ROWWISE)
    c_ref_row = torch._scaled_mm(
        a_fp8,
        b_fp8.T,
        scale_a_row.unsqueeze(1),
        scale_b_row.unsqueeze(0),
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )
    all_passed &= _check_close("RowWise", c_triton_row, c_ref_row)

    # --- Test 3: MXFP8 (1x32 e8m0 microscaling) via tl.dot_scaled ---
    print("\n--- MXFP8 (1x32 e8m0) scaling ---")
    try:
        from torch.testing._internal.common_quantized import to_blocked, to_mxfp

        def _deq_mx(xq, s2d, vec=32):
            # fp32 emulation of dot_scaled: dequant = quant * 2^scale per vec-block.
            R, Kk = xq.shape
            xf = xq.float().reshape(R, Kk // vec, vec)
            sf = s2d.float().reshape(R, Kk // vec, 1)
            return (xf * sf).reshape(R, Kk)

        sa2d, a_mx = to_mxfp(a_fp32.to(torch.bfloat16), block_size=32, format="mxfp8")
        sb2d, b_mx = to_mxfp(b_fp32.to(torch.bfloat16), block_size=32, format="mxfp8")
        c_triton_mx = scaled_mm_autows(
            a_mx, b_mx, to_blocked(sa2d), to_blocked(sb2d), MXFP8, MXFP8
        )
        # Reference = exact fp32 emulation of what dot_scaled computes (so this
        # also validates the to_blocked <-> in-kernel de-swizzle round-trip).
        c_ref_mx = (_deq_mx(a_mx, sa2d) @ _deq_mx(b_mx, sb2d).T).to(torch.bfloat16)
        all_passed &= _check_close("MXFP8 (1x32 e8m0)", c_triton_mx, c_ref_mx)
    except ImportError:
        print("  to_mxfp/to_blocked not available, skipping MXFP8 test")

    # --- Test 3b: symmetric fp32 BlockWise 1x128 (both operands) ---
    print("\n--- BlockWise 1x128/1x128 (symmetric fp32) scaling ---")

    def _quant_1x128(x):
        # x [R, K] fp32 -> (x_q e4m3, scale [R, K//128] fp32), matching the
        # tritonbench harness convention (scale = amax/448, x_q = x * scale).
        R, Kk = x.shape
        xb = x.reshape(R, Kk // 128, 128)
        amax = xb.abs().amax(dim=2, keepdim=True).float()
        scale = (torch.finfo(torch.float8_e4m3fn).max / amax).reciprocal()
        xq = (xb * scale).reshape(R, Kk).to(torch.float8_e4m3fn)
        return xq, scale.reshape(R, Kk // 128).to(torch.float32)

    def _deq_1x128(xq, s):
        R, Kk = xq.shape
        return (
            xq.float().reshape(R, Kk // 128, 128) * s.float().reshape(R, Kk // 128, 1)
        ).reshape(R, Kk)

    a_q, sa_bw = _quant_1x128(a_fp32)
    b_q, sb_bw = _quant_1x128(b_fp32)
    c_triton_bw = scaled_mm_autows(
        a_q, b_q, sa_bw, sb_bw, BLOCKWISE_1x128, BLOCKWISE_1x128
    )
    c_ref_bw = (_deq_1x128(a_q, sa_bw) @ _deq_1x128(b_q, sb_bw).T).to(torch.bfloat16)
    all_passed &= _check_close("BlockWise1x128 symmetric (fp32)", c_triton_bw, c_ref_bw)

    # --- Test 4: v2 API ---
    print("\n--- _scaled_mm_v2 API ---")
    c_v2 = scaled_mm_v2(
        a_fp8,
        b_fp8,
        scale_a=[scale_a],
        recipe_a=[TENSORWISE],
        scale_b=[scale_b],
        recipe_b=[TENSORWISE],
    )
    all_passed &= _check_close("v2 (TensorWise)", c_v2, c_ref)

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nFAILED: one or more tests exceeded tolerance.")
        exit(1)
