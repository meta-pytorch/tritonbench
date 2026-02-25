"""
Common utils for attention kernels that can be shared between the fused attention
example and the proton implementation.
"""

import os

import triton.language as tl

# check if we have the TMA version in Triton PR #4498 (https://github.com/triton-lang/triton/pull/4498).
HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)
WITH_COMPPIPE = os.getenv("ENABLE_COMPPIPE")
PEEL_LAST = os.getenv("PEEL_LAST_ITER")
WITH_TMA = os.getenv("WITH_TMA")
HAS_EXPLICIT_WS = os.getenv("ENABLE_EXPLICIT_WS")
SUPPORT_GLUON = os.getenv("WITH_GLUON") == "1"
WITH_MAXNREG = os.getenv("WITH_MAXNREG")
WITH_OSS_WARPSPEC = os.getenv("WITH_OSS_WARPSPEC")
