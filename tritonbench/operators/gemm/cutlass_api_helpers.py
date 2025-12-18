import ctypes
import time
from dataclasses import dataclass
from typing import Any, List, Tuple

import cutlass_api
import torch
from tqdm import tqdm


# Valid cutlass_api configurations for SM100
CUTLASS_API_VALID_TILE_M = {64, 128, 256}
CUTLASS_API_VALID_TILE_N = {32, 64, 128, 256}
CUTLASS_API_VALID_CLUSTER_M = {1, 2, 4, 8, 16}
CUTLASS_API_VALID_CLUSTER_N = {1, 2, 4, 8, 16}


@dataclass
class HeuristicConfig:
    """Configuration recommended by nvMatmulHeuristics."""

    tile_m: int
    tile_n: int
    tile_k: int
    cluster_m: int
    cluster_n: int
    estimated_runtime: float


def _compile_kernels(
    kernels: List[cutlass_api.Kernel],
    args: cutlass_api.arguments.GemmArguments,
) -> List[Any]:
    """Compile a list of kernels and return their compiled artifacts."""
    print(f"\nPre-compiling {len(kernels)} kernels...")
    compiled_artifacts = []
    for kernel in tqdm(kernels):
        compiled_artifacts.append(kernel.compile(args))
    print("Done compiling.")
    return compiled_artifacts


def _benchmark_kernels(
    kernels: List[cutlass_api.Kernel],
    compiled_artifacts: List[Any],
    args: cutlass_api.arguments.GemmArguments,
    num_iters: int = 10,
    num_warmup: int = 3,
) -> Tuple[cutlass_api.Kernel, Any]:
    """Benchmark kernels and return the best one with its compiled artifact."""
    results = []
    print(f"\nBenchmarking {len(kernels)} kernels...")

    for idx, kernel in enumerate(kernels):
        compiled_artifact = compiled_artifacts[idx]

        for _ in range(num_warmup):
            kernel.run(
                args,
                compiled_artifact,
                stream=torch.cuda.current_stream(),
                assume_supported_args=True,
            )

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            kernel.run(
                args,
                compiled_artifact,
                stream=torch.cuda.current_stream(),
                assume_supported_args=True,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time_ms = (end - start) / num_iters * 1000
        results.append((idx, kernel.metadata.kernel_name, avg_time_ms))
        print(f"  [{idx}] {kernel.metadata.kernel_name}: {avg_time_ms:.4f} ms")

    best_idx, _, _ = min(results, key=lambda x: x[2])
    return kernels[best_idx], compiled_artifacts[best_idx]


def _make_cutlass_api_validity_callback(nvmmh_lib):
    """
    Callback for nvMatmulHeuristics that only accepts configurations
    matching cutlass_api's supported tile and cluster shapes.

    Note: We don't filter on tile_k because nvMatmulHeuristics uses K=64/128
    while cutlass_api uses K=256. We match on (M, N, cluster) only.
    """

    def validity_check(kernel_config_ptr, problem_ptr):
        kernel = kernel_config_ptr.contents
        tile_m = kernel.cta[0]
        tile_n = kernel.cta[1]
        cluster_m = kernel.cluster[0]
        cluster_n = kernel.cluster[1]

        if tile_m not in CUTLASS_API_VALID_TILE_M:
            return 0
        if tile_n not in CUTLASS_API_VALID_TILE_N:
            return 0
        if cluster_m not in CUTLASS_API_VALID_CLUSTER_M:
            return 0
        if cluster_n not in CUTLASS_API_VALID_CLUSTER_N:
            return 0
        if cluster_m * cluster_n > 16:
            return 0

        return 1

    return validity_check


def _get_layout_enum(layout_a: str, layout_b: str):
    """
    Map layout strings to NvMatmulHeuristicsMatmulLayout enum.

    CUTLASS convention: T means row-major, N means column-major.
    """
    import nvMatmulHeuristics

    trans_a = "T" if layout_a == "row" else "N"
    trans_b = "T" if layout_b == "row" else "N"
    layout_str = f"{trans_a}{trans_b}_ROW_MAJOR"
    return nvMatmulHeuristics.NvMatmulHeuristicsMatmulLayout[layout_str]


def get_best_cutlass_api_kernel(
    args: cutlass_api.arguments.GemmArguments,
    num_iters: int = 10,
    num_warmup: int = 3,
) -> Tuple[cutlass_api.Kernel, Any]:
    """Get best kernel by exhaustively benchmarking all cutlass_api kernels."""
    kernels = cutlass_api.get_kernels(args)
    compiled_artifacts = _compile_kernels(kernels, args)
    return _benchmark_kernels(kernels, compiled_artifacts, args, num_iters, num_warmup)


def get_heuristic_configs(
    m: int,
    n: int,
    k: int,
    dtype_a: torch.dtype,
    dtype_b: torch.dtype,
    layout_a: str = "row",
    layout_b: str = "col",
    count: int = 10,
) -> List[HeuristicConfig]:
    """
    Get kernel configurations recommended by nvMatmulHeuristics.

    Uses KERNEL_ADDITIONAL_VALIDITY_CHECK callback to filter to cutlass_api-compatible configs.
    """
    try:
        import nvMatmulHeuristics
    except ImportError:
        print("nvMatmulHeuristics not available")
        return []

    dtype_to_cublas = {
        torch.float64: "D",
        torch.float32: "S",
        torch.float16: "H",
        torch.bfloat16: "T",
    }

    a_char = dtype_to_cublas.get(dtype_a, "H")
    precision = f"{a_char}S{a_char}"

    lh = nvMatmulHeuristics.NvMatmulHeuristicsInterfaceEx(
        backend=nvMatmulHeuristics.NvMatmulHeuristicsTarget.CUTLASS3,
        flags=nvMatmulHeuristics.NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
        load_discovery_implicitly=True,
    )

    backend = lh.createBackend(nvMatmulHeuristics.NvMatmulHeuristicsTarget.CUTLASS3)

    validity_callback = _make_cutlass_api_validity_callback(nvMatmulHeuristics)
    try:
        lh.setBackendCallbackProperty(
            backend,
            nvMatmulHeuristics.NvMatmulHeuristicsBackendPropertyCallbackKind.KERNEL_ADDITIONAL_VALIDITY_CHECK,
            validity_callback,
        )
    except Exception as e:
        print(f"Warning: Could not set validity callback: {e}")

    cta_n_div = ctypes.c_int(32)
    lh.setBackendValueProperty(
        backend,
        nvMatmulHeuristics.NvMatmulHeuristicsBackendProperty.CTA_TILE_N_DIV_REQUIREMENT,
        ctypes.byref(cta_n_div),
        ctypes.sizeof(cta_n_div),
    )

    layout = _get_layout_enum(layout_a, layout_b)

    try:
        lh.loadInternalDiscoverySet(layout, precision=precision)
    except Exception:
        pass

    problem = lh.makeNvMatmulHeuristicsProblem(m, n, k, layout, batch_size=1)
    raw_configs = lh.getEx(problem, count, backend, precision=precision)
    lh.destroyBackend(backend)

    if not raw_configs:
        return []

    configs = []
    for cfg in raw_configs:
        kernel = cfg["kernel"]
        configs.append(
            HeuristicConfig(
                tile_m=kernel.cta_tile_m,
                tile_n=kernel.cta_tile_n,
                tile_k=kernel.cta_tile_k,
                cluster_m=kernel.cluster_m,
                cluster_n=kernel.cluster_n,
                estimated_runtime=cfg["runtime"],
            )
        )

    configs.sort(key=lambda c: c.estimated_runtime)
    return configs


def get_heuristic_filtered_kernels(
    args: cutlass_api.arguments.GemmArguments,
    m: int,
    n: int,
    k: int,
    dtype_a: torch.dtype,
    dtype_b: torch.dtype,
    layout_a: str = "row",
    layout_b: str = "col",
    count: int = 5,
) -> Tuple[List[cutlass_api.Kernel], List[HeuristicConfig]]:
    """
    Use nvMatmulHeuristic to narrow down cutlass_api kernel choices.

    Matches on (tile_m, tile_n, cluster_m, cluster_n). Returns kernels sorted
    by heuristic estimated runtime.
    """
    heuristic_configs = get_heuristic_configs(
        m, n, k, dtype_a, dtype_b, layout_a, layout_b, count
    )

    all_kernels = cutlass_api.get_kernels(args, cc=100)

    if not heuristic_configs:
        print(f"No heuristic configs found, returning first {count} kernels")
        return all_kernels[:count], []

    print(f"Heuristic recommended {len(heuristic_configs)} configs:")
    for i, cfg in enumerate(heuristic_configs[:5]):
        print(
            f"  [{i}] tile=({cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}) "
            f"cluster=({cfg.cluster_m}x{cfg.cluster_n}) "
            f"runtime={cfg.estimated_runtime * 1000:.4f}ms"
        )

    config_key_to_runtime = {}
    for cfg in heuristic_configs:
        key = (cfg.tile_m, cfg.tile_n, cfg.cluster_m, cfg.cluster_n)
        if key not in config_key_to_runtime:
            config_key_to_runtime[key] = cfg.estimated_runtime

    matched_kernels_with_runtime = []
    for kernel in all_kernels:
        meta = kernel.metadata
        if hasattr(meta, "design"):
            design = meta.design
            tile_shape = design.tile_shape
            cluster_shape = design.cluster_shape
            key = (tile_shape[0], tile_shape[1], cluster_shape[0], cluster_shape[1])
            if key in config_key_to_runtime:
                matched_kernels_with_runtime.append(
                    (kernel, config_key_to_runtime[key])
                )

    if not matched_kernels_with_runtime:
        print(f"No matching kernels found, returning first {count} kernels")
        return all_kernels[:count], heuristic_configs

    matched_kernels_with_runtime.sort(key=lambda x: x[1])
    matched_kernels = [k for k, _ in matched_kernels_with_runtime]

    print(f"Matched {len(matched_kernels)} kernels from heuristics")
    return matched_kernels, heuristic_configs


def get_best_heuristic_kernel(
    args: cutlass_api.arguments.GemmArguments,
    m: int,
    n: int,
    k: int,
    dtype_a: torch.dtype,
    dtype_b: torch.dtype,
    layout_a: str = "row",
    layout_b: str = "col",
    heuristic_count: int = 5,
    num_iters: int = 10,
    num_warmup: int = 3,
) -> Tuple[cutlass_api.Kernel, Any]:
    """Use nvMatmulHeuristic to narrow down kernels, then autotune among them."""
    kernels, _ = get_heuristic_filtered_kernels(
        args, m, n, k, dtype_a, dtype_b, layout_a, layout_b, heuristic_count
    )
    print(f"\nHeuristic narrowed to {len(kernels)} kernels")
    compiled_artifacts = _compile_kernels(kernels, args)
    return _benchmark_kernels(kernels, compiled_artifacts, args, num_iters, num_warmup)
