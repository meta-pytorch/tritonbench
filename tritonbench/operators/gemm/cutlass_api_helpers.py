import time
from typing import Any, Tuple

import cutlass_api
import torch
from tqdm import tqdm


def get_best_cutlass_api_kernel(
    args: cutlass_api.arguments.GemmArguments,
    num_iters: int = 10,
    num_warmup: int = 3,
) -> tuple[cutlass_api.Kernel, Any]:
    kernels = cutlass_api.get_kernels(args)

    # Pre-compile all kernels
    print(f"\nPre-compiling {len(kernels)} kernels...")
    compiled_artifacts = []
    for idx, kernel in enumerate(tqdm(kernels)):
        compiled_artifact = kernel.compile(args)
        compiled_artifacts.append(compiled_artifact)
    print("Done compiling.")

    # Benchmark all kernels
    num_warmup = 3
    num_iters = 10
    results = []

    print(f"\nBenchmarking {len(kernels)} kernels...")
    for idx, kernel in enumerate(kernels):
        compiled_artifact = compiled_artifacts[idx]

        # Warmup
        for _ in range(num_warmup):
            kernel.run(args, compiled_artifact, assume_supported_args=True)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            kernel.run(args, compiled_artifact, assume_supported_args=True)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time_ms = (end - start) / num_iters * 1000
        results.append((idx, kernel.metadata.kernel_name, avg_time_ms))
        print(f"  [{idx}] {kernel.metadata.kernel_name}: {avg_time_ms:.4f} ms")

    best_idx, best_name, best_time = min(results, key=lambda x: x[2])

    return kernels[best_idx], compiled_artifacts[best_idx]
