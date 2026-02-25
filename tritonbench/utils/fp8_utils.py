"""FP8 utilities for tritonbench operators."""

import functools
import os
from typing import Tuple

import torch
import triton.language as tl


@functools.lru_cache
def supports_float8_fnuz(throw_on_hip_incompatibility: bool = True) -> bool:
    if torch.version.hip:
        device_capability = torch.cuda.get_device_capability()

        if device_capability < (9, 4):
            gpu_arch = torch.cuda.get_device_properties("cuda").gcnArchName
            msg = f"Unsupported GPU arch: {gpu_arch} for FP8"
            if throw_on_hip_incompatibility:
                raise RuntimeError(msg)
            else:
                import logging

                logging.error(msg)
                return False

        elif device_capability >= (9, 4):
            # gfx942 (MI300) reports (9, 4), gfx950 (MI350) reports (9, 5)
            return True

    return False


def get_fp8_constants() -> Tuple[torch.dtype, tl.dtype, float, float]:
    """
    Helper function to get constant values for the current platform.

    Returns:
        pt_dtype (torch.dtype): The correct torch fp8 datatype.
        tl_dtype (tl.dtype): The correct triton fp8 datatype.
        max_fp8 (float): The maximum reprsentable value for the fp8 datatype.
        eps (float): Minimum clip value to prevent divide by zero.
    """
    running_on_github: bool = os.getenv("GITHUB_ENV") is not None
    if supports_float8_fnuz(throw_on_hip_incompatibility=(not running_on_github)):
        pt_fp8_dtype = torch.float8_e4m3fnuz
        tl_fp8_dtype = tl.float8e4b8
    else:
        pt_fp8_dtype = torch.float8_e4m3fn
        tl_fp8_dtype = tl.float8e4nv

    return pt_fp8_dtype, tl_fp8_dtype, torch.finfo(pt_fp8_dtype).max, 1e-12
