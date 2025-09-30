"""
Set of common attention utils that are exclusive to Blackwell. Separated to avoid issues with more
generic attention kernels.
"""

import os
from functools import lru_cache

import torch
import triton


@lru_cache
def is_tile_enabled():
    # Note: This assumes you have the TileIR backend.
    # We don't have a reliable way to check this at this time.
    return os.getenv("ENABLE_TILE", "0") == "1"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


# Note: This seems to only be set at autotuning and cannot be reliably used.
def is_cuda_tileir():
    return (
        triton.runtime.driver.active.get_current_target().backend == "triton_cuda_tile"
    )


def is_cuda_triton():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_cuda():
    return is_cuda_triton() or is_cuda_tileir()


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9
