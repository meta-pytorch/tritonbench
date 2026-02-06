"""Utils for compare_benchmarks custom TritonBench benchmark"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"
os.environ["ENABLE_PERSISTENT_TMA_MATMUL"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import pandas as pd
import torch
import triton

from dsi.logger.configs.TritonMultiOperatorBenchmarkComparisonsLoggerConfig.logger import (
    TritonMultiOperatorBenchmarkComparisonsLogEntry,
)
from dsi.logger.py3.whence_logged.types import WhenceScribeLogged

DEFAULT_OPS = ["gemm", "addmm", "bmm", "scaled_mm"]
DEFAULT_METRICS = ["latency", "tflops"]
DEFAULT_WORKLOADS = ["cmf"]

SUPPORTED_GPU_TYPES = ["A100", "H100", "B200", "GB200", "MI300", "MI350"]


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    custom_bench: str = None
    ops: List[str] = field(default_factory=lambda: DEFAULT_OPS.copy())
    metrics: List[str] = field(default_factory=lambda: DEFAULT_METRICS.copy())
    workloads: List[str] = field(default_factory=lambda: DEFAULT_WORKLOADS.copy())
    benchmark_map: dict[str, tuple[str, str]] = field(default_factory=dict)
    parse_autotune_logs: bool = False
    log_scuba: bool = False
    scuba_eval_id: str = None

@dataclass
class DiodeBenchmarkConfig(BenchmarkConfig):
    """Configuration for a Diode benchmark run. Inherits from BenchmarkConfig."""

    custom_bench: str = "diode"
    diode_version: str = "recommended"
    diode_topk: int = 1


def detect_gpu() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("[Compare Benchmarks] CUDA/ROCm not available - cannot detect GPU type")

    device_name = torch.cuda.get_device_name(0)
    print(f"[Compare Benchmarks] Detected GPU: {device_name}")

    for gpu in SUPPORTED_GPU_TYPES:
        if gpu in device_name or f"{gpu}x" in device_name:
            return gpu.lower()

    raise RuntimeError(f"[Compare Benchmarks] Unknown GPU type: {device_name}")


def get_cuda_version() -> Optional[str]:
    try:
        return torch.version.cuda
    except Exception:
        return None


def get_rocm_version() -> Optional[str]:
    try:
        return torch.version.hip
    except Exception:
        return None


def get_triton_type() -> str:
    version = triton.__version__
    if "dev" in version or "git" in version:
        return "trunk"
    if "rc" in version or "beta" in version or "alpha" in version:
        return "beta"
    return "stable"


def parse_shape(shape_str: str) -> tuple[int, int, int]:
    if not shape_str or pd.isna(shape_str):
        return (0, 0, 0)
    try:
        parts = shape_str.split(", ")
        left = parts[0].split("x")
        right = parts[1].split("x")
        m = int(left[0])
        k = int(left[1])
        n = int(right[1])
        return (m, n, k)
    except (IndexError, ValueError):
        return (0, 0, 0)


def parse_pct(pct_str) -> Optional[float]:
    if pct_str is None or pd.isna(pct_str) or pct_str == "":
        return None
    try:
        s = str(pct_str).strip().rstrip("%")
        return float(s)
    except (ValueError, TypeError):
        return None


def safe_int(val, default: int = 0) -> Optional[int]:
    if val is None or pd.isna(val) or val == "":
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_bool(val) -> Optional[bool]:
    if val is None or pd.isna(val) or val == "":
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def safe_float(val) -> Optional[float]:
    if val is None or pd.isna(val) or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_str(val) -> str:
    if val is None or pd.isna(val):
        return ""
    return str(val)


def log_benchmark(
    df: pd.DataFrame,
    config: BenchmarkConfig,
) -> None:
    pytorch_version = torch.__version__
    triton_version = triton.__version__
    cuda_version = get_cuda_version()
    rocm_version = get_rocm_version()

    lhs_benchmark = df["lhs_benchmark_name"].iloc[0]
    rhs_benchmark = df["rhs_benchmark_name"].iloc[0]

    scuba_eval_id = config.scuba_eval_id
    custom_bench = config.custom_bench
    triton_type = get_triton_type()
    diode_version = getattr(config, "diode_version", None)
    diode_topk = getattr(config, "diode_topk", None)

    async def log_row(row: pd.Series) -> None:
        try:
            m_dim, n_dim, k_dim = parse_shape(row.get("Shape"))
            log_entry = TritonMultiOperatorBenchmarkComparisonsLogEntry(
                eval_id=scuba_eval_id,
                custom_bench=custom_bench,
                op=safe_str(row.get("Operation")),
                gpu=safe_str(row.get("gpu")),
                workload=safe_str(row.get("workload")),
                lhs_benchmark=lhs_benchmark,
                rhs_benchmark=rhs_benchmark,
                m_dim=m_dim,
                n_dim=n_dim,
                k_dim=k_dim,
                batch=None,
                bias_m=None,
                bias_n=None,
                dtype=safe_str(row.get("Dtype")),
                a_stride=safe_str(row.get("A Stride")),
                b_stride=safe_str(row.get("B Stride")),
                bias_stride=safe_str(row.get("C Stride")) if row.get("C Stride") else None,
                lhs_best_aten_kernel=safe_str(row.get("A: Best Aten Kernel")) or None,
                lhs_best_aten_runtime_ms=safe_float(row.get("A: Best Aten Runtime (ms)")),
                lhs_best_triton_kernel=safe_str(row.get("A: Best Triton Kernel")) or None,
                lhs_best_triton_runtime_ms=safe_float(row.get("A: Best Triton Runtime (ms)")),
                lhs_winner=safe_str(row.get("A: Winner")) or None,
                lhs_speedup_pct=parse_pct(row.get("A: Speedup (%)")),
                rhs_best_aten_kernel=safe_str(row.get("B: Best Aten Kernel")) or None,
                rhs_best_aten_runtime_ms=safe_float(row.get("B: Best Aten Runtime (ms)")),
                rhs_best_triton_kernel=safe_str(row.get("B: Best Triton Kernel")) or None,
                rhs_best_triton_runtime_ms=safe_float(row.get("B: Best Triton Runtime (ms)")),
                rhs_winner=safe_str(row.get("B: Winner")) or None,
                rhs_speedup_pct=parse_pct(row.get("B: Speedup (%)")),
                aten_improvement_pct=parse_pct(row.get("Aten Improvement (A vs B)")),
                triton_improvement_pct=parse_pct(row.get("Triton Improvement (A vs B)")),
                winner_change=safe_str(row.get("Winner Change (A→B)")) or None,
                lhs_acc_type=safe_str(row.get("A: Triton Metadata ACC_TYPE")),
                lhs_allow_tf32=safe_bool(row.get("A: Triton Metadata ALLOW_TF32")) or False,
                lhs_a_row_major=safe_bool(row.get("A: Triton Metadata A_ROW_MAJOR")),
                lhs_block_k=safe_int(row.get("A: Triton Metadata BLOCK_K")),
                lhs_block_m=safe_int(row.get("A: Triton Metadata BLOCK_M")),
                lhs_block_n=safe_int(row.get("A: Triton Metadata BLOCK_N")),
                lhs_b_row_major=safe_bool(row.get("A: Triton Metadata B_ROW_MAJOR")),
                lhs_even_k=safe_bool(row.get("A: Triton Metadata EVEN_K")) or False,
                lhs_group_m=safe_int(row.get("A: Triton Metadata GROUP_M")),
                lhs_num_sms=safe_int(row.get("A: Triton Metadata NUM_SMS"), default=None),
                lhs_tma_experimental_api=safe_bool(row.get("A: Triton Metadata TMA_EXPERIMENTAL_API")),
                lhs_tma_size=safe_int(row.get("A: Triton Metadata TMA_SIZE"), default=None),
                lhs_use_fast_accum=safe_bool(row.get("A: Triton Metadata USE_FAST_ACCUM")) or False,
                lhs_num_stages=safe_int(row.get("A: Triton Metadata num_stages")),
                lhs_num_warps=safe_int(row.get("A: Triton Metadata num_warps")),
                rhs_acc_type=safe_str(row.get("B: Triton Metadata ACC_TYPE")),
                rhs_allow_tf32=safe_bool(row.get("B: Triton Metadata ALLOW_TF32")) or False,
                rhs_a_row_major=safe_bool(row.get("B: Triton Metadata A_ROW_MAJOR")),
                rhs_block_k=safe_int(row.get("B: Triton Metadata BLOCK_K")),
                rhs_block_m=safe_int(row.get("B: Triton Metadata BLOCK_M")),
                rhs_block_n=safe_int(row.get("B: Triton Metadata BLOCK_N")),
                rhs_b_row_major=safe_bool(row.get("B: Triton Metadata B_ROW_MAJOR")),
                rhs_even_k=safe_bool(row.get("B: Triton Metadata EVEN_K")) or False,
                rhs_group_m=safe_int(row.get("B: Triton Metadata GROUP_M")),
                rhs_num_sms=safe_int(row.get("B: Triton Metadata NUM_SMS"), default=None),
                rhs_tma_experimental_api=safe_bool(row.get("B: Triton Metadata TMA_EXPERIMENTAL_API")),
                rhs_tma_size=safe_int(row.get("B: Triton Metadata TMA_SIZE"), default=None),
                rhs_use_fast_accum=safe_bool(row.get("B: Triton Metadata USE_FAST_ACCUM")) or False,
                rhs_num_stages=safe_int(row.get("B: Triton Metadata num_stages")),
                rhs_num_warps=safe_int(row.get("B: Triton Metadata num_warps")),
                triton_type=triton_type,
                pytorch_version=pytorch_version,
                triton_version=triton_version,
                cuda_version=cuda_version,
                rocm_version=rocm_version,
                diode_version=diode_version,
                diode_topk=diode_topk,
            )
            await log_entry.log(whence_scribe_logged=WhenceScribeLogged.PROD)
        except Exception as e:
            logging.warning(f"[Compare Benchmarks] Failed to log row: {e}")

    for _, row in df.iterrows():
        asyncio.run(log_row(row))
