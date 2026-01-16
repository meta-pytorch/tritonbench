"""
Utils for checking and modifying the environment.
Requires PyTorch
"""

import argparse
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager, ExitStack
from functools import lru_cache
from typing import Optional

import torch
import triton
from tritonbench.utils.path_utils import REPO_PATH

log = logging.getLogger(__name__)

MAIN_RANDOM_SEED = 1337
AVAILABLE_PRECISIONS = [
    "bypass",
    "fp32",
    "tf32",
    "fp16",
    "amp",
    "fx_int8",
    "bf16",
    "amp_fp16",
    "amp_bf16",
    "fp8",
]


def is_fbcode() -> bool:
    return not hasattr(torch.version, "git_version")


def is_cuda() -> bool:
    return torch.version.cuda is not None


def has_manifold() -> bool:
    return shutil.which("manifold") is not None


def get_nvidia_gpu_model() -> str:
    """
    Retrieves the model of the NVIDIA GPU being used.
    Will return the name of the first GPU listed.
    Returns:
        str: The model of the NVIDIA GPU or empty str if not found.
    """
    try:
        model = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]
        )
        return model.decode().strip().split("\n")[0]
    except OSError:
        logging.warning("nvidia-smi not found. Returning empty str.")
        return ""


def is_hip() -> bool:
    return torch.version.hip is not None


def is_hip_mi200():
    try:
        target = triton.runtime.driver.active.get_current_target()
        return is_hip() and target.arch == "gfx90a"
    except Exception:
        return False


def is_hip_mi300():
    try:
        target = triton.runtime.driver.active.get_current_target()
        return is_hip() and target.arch == "gfx942"
    except Exception:
        return False


def is_blackwell() -> bool:
    """Check if running on an NVIDIA Blackwell GPU (B200 or B300 series)."""
    if not is_cuda():
        return False
    gpu_model = get_nvidia_gpu_model()
    return "B200" in gpu_model or "B300" in gpu_model


IS_BLACKWELL = is_blackwell()


def is_h100() -> bool:
    """Check if running on an NVIDIA H100 GPU."""
    if not is_cuda():
        return False
    gpu_model = get_nvidia_gpu_model()
    return "H100" in gpu_model


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_cu130():
    return is_cuda() and torch.version.cuda == "13.0"


@lru_cache
def is_tile_enabled():
    # Note: This assumes you have the TileIR backend.
    # We don't have a reliable way to check this at this time.
    return os.getenv("ENABLE_TILE", "0") == "1"


def is_mtia():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "mtia"
    except Exception:
        return False


def set_env():
    # set cutlass dir
    # by default we use the cutlass version built with pytorch
    import torch._inductor.config as inductor_config

    if hasattr(inductor_config, "cutlass"):
        cutlass_namespace = True
        current_cutlass_dir = inductor_config.cutlass.cutlass_dir
    else:
        cutlass_namespace = False
        current_cutlass_dir = inductor_config.cuda.cutlass_dir

    if not os.path.exists(current_cutlass_dir):
        tb_cutlass_dir = REPO_PATH.joinpath("submodules", "cutlass")
        if tb_cutlass_dir.is_dir():
            if cutlass_namespace:
                inductor_config.cutlass.cutlass_dir = str(tb_cutlass_dir)
            else:
                inductor_config.cuda.cutlass_dir = str(tb_cutlass_dir)


def set_random_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""
    import random

    import numpy
    import torch

    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator

        seed = MAIN_RANDOM_SEED
        import torch.cuda

        if not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(seed)

        import torch.xpu

        if not torch.xpu._is_in_bad_fork():
            torch.xpu.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)
    torch.manual_seed = deterministic_torch_manual_seed


@contextmanager
def nested(*contexts):
    """
    Chain and apply a list of contexts
    """
    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield contexts


@contextmanager
def fresh_inductor_cache(parallel_compile=False):
    INDUCTOR_DIR = f"/tmp/torchinductor_{os.environ['USER']}"
    if os.path.exists(INDUCTOR_DIR):
        shutil.rmtree(INDUCTOR_DIR)
    if parallel_compile:
        old_parallel_compile_threads = os.environ.get(
            "TORCHINDUCTOR_COMPILE_THREADS", None
        )
        cpu_count: Optional[int] = os.cpu_count()
        if cpu_count is not None and cpu_count > 1:
            cpu_count = min(32, cpu_count)
            log.warning(f"Set env var TORCHINDUCTOR_COMPILE_THREADS to {cpu_count}")
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(cpu_count)
    yield
    # clean up parallel compile directory and env
    if parallel_compile and "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        if old_parallel_compile_threads:
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = old_parallel_compile_threads
        else:
            del os.environ["TORCHINDUCTOR_COMPILE_THREADS"]
    if os.path.exists(INDUCTOR_DIR):
        shutil.rmtree(INDUCTOR_DIR)


@contextmanager
def fresh_triton_cache():
    """
    Run with a fresh triton cache.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        old = os.environ.get("TRITON_CACHE_DIR", None)
        os.environ["TRITON_CACHE_DIR"] = tmpdir
        old_cache_manager = os.environ.get("TRITON_CACHE_MANAGER", None)
        os.environ.pop("TRITON_CACHE_MANAGER", None)
        yield
        if old:
            os.environ["TRITON_CACHE_DIR"] = old
        else:
            del os.environ["TRITON_CACHE_DIR"]
        if old_cache_manager:
            os.environ["TRITON_CACHE_MANAGER"] = old_cache_manager


def apply_precision(
    op,
    precision: str,
):
    if precision == "bypass" or precision == "fp32":
        return
    if precision == "fp16":
        op.enable_fp16()
    elif precision == "bf16":
        op.enable_bf16()
    elif precision == "tf32":
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        log.warning(f"[tritonbench] Precision {precision} is handled by operator.")


def override_default_precision_for_input_loader(
    args: argparse.Namespace,
    override_value: str = "bypass",
):
    # If loading shapes via input_loader, respect override value
    args.precision = override_value if args.input_loader else args.precision


def set_allow_tf32(allow_tf32: bool) -> bool:
    old_allow_tf32, torch.backends.cuda.matmul.allow_tf32 = (
        torch.backends.cuda.matmul.allow_tf32,
        allow_tf32,
    )
    return old_allow_tf32


def reset_allow_tf32(old_allow_tf32: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32


def get_logger(name, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
