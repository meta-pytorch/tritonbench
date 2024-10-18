import os
import logging
import shutil
from contextlib import contextmanager, ExitStack

from typing import Optional

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
