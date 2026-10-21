"""
Common kernels used for profiling.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from tritonbench.utils.env_utils import is_cuda, is_hip




def has_gpu_timer() -> bool:
    """Whether ``time()``/``smid()`` below can be compiled on this platform.

    Only CUDA and HIP expose the required intrinsics. Triton's Intel backend
    declares ``tl.extra.intel.globaltimer``/``smid``, but both are verbatim
    copies of the NVIDIA PTX inline asm, so they fail to build on SPIR-V
    ("Constraints for inline assembly cannot be validated"). The portable
    replacement, SPIR-V's ``OpReadClockKHR``, needs ``SPV_KHR_shader_clock``,
    which Triton's SPIR-V translator does not currently allow either.

    Callers should pass ``profile_mem=None`` when this returns False so that
    the instrumented branches are never instantiated, rather than recording
    bogus zero timestamps.
    """
    return is_cuda() or is_hip()


@triton.jit
def _time_hip(_semantic=None):
    return tl.inline_asm_elementwise(
        """
        s_memrealtime $0
        s_waitcnt vmcnt(0)
        """,
        "=r",
        [],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _time_cuda(_semantic=None):
    return tl.extra.cuda.globaltimer()


@triton.jit
def _time_unsupported(_semantic=None):
    tl.static_assert(False, "Unsupported platform")


# Import-time selection: no body reads a global, so instrumented callers stay
# eligible for the C launcher without threading platform flags. Callers must
# still guard on has_gpu_timer() so the unsupported body is never instantiated.
if is_hip():
    time = _time_hip
elif is_cuda():
    time = _time_cuda
else:
    time = _time_unsupported


@triton.jit
def _smid_hip(_semantic=None):
    # Reference: https://github.com/triton-lang/triton/blob/4d3c4980044eb40fae31490ee07fcf94b6bb7f4a/third_party/amd/backend/include/hip/amd_detail/amd_device_functions.h#L891-L937
    HW_ID_CU_ID_OFFSET: tl.constexpr = 8
    HW_ID_CU_ID_SIZE: tl.constexpr = 4
    cu_id = tl.inline_asm_elementwise(
        f"""
        s_getreg_b32 $0, hwreg(HW_REG_HW_ID, {HW_ID_CU_ID_OFFSET}, {HW_ID_CU_ID_SIZE})
        """,
        "=s",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )

    HW_ID_SE_ID_OFFSET: tl.constexpr = 13
    HW_ID_SE_ID_SIZE: tl.constexpr = 3
    se_id = tl.inline_asm_elementwise(
        f"""
        s_getreg_b32 $0, hwreg(HW_REG_HW_ID, {HW_ID_SE_ID_OFFSET}, {HW_ID_SE_ID_SIZE})
        """,
        "=s",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )

    XCC_ID_XCC_ID_OFFSET: tl.constexpr = 0
    XCC_ID_XCC_ID_SIZE: tl.constexpr = 4
    xcc_id = tl.inline_asm_elementwise(
        f"""
        s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, {XCC_ID_XCC_ID_OFFSET}, {XCC_ID_XCC_ID_SIZE})
        """,
        "=s",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )

    # Format: 1 | 3 digits for xcc_id | 3 digits for se_id | 3 digits for cu_id
    return 1_000_000_000 + xcc_id * 1_000_000 + se_id * 1_000 + cu_id


@triton.jit
def _smid_cuda(_semantic=None):
    return tl.extra.cuda.smid()


@triton.jit
def _smid_unsupported(_semantic=None):
    tl.static_assert(False, "Unsupported platform")


# Import-time selection: see the note on time() above.
if is_hip():
    smid = _smid_hip
elif is_cuda():
    smid = _smid_cuda
else:
    smid = _smid_unsupported


def profile_mem_pre_hook(nargs):
    """
    Pre-hook for allocating memory for profiling.
    """
    profile_mem: torch.Tensor | None = nargs.get("profile_mem")
    if profile_mem is not None:
        shape: Tuple[int] = nargs["grid"](nargs)
        # Add another dimension for simd, start, and end time
        shape = (*shape, 3)
        # We use `resize_` to avoid creating a new tensor.
        # Otherwise, the caller cannot reference the tensor.
        profile_mem.resize_(shape)
