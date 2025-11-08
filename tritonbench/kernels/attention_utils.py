"""
Common utils for attention kernels that can be shared between the fused attention
example and the proton implementation.
"""

import os

import torch

import triton
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


class TmaAutoTuneHelper:
    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        if HAS_TMA_DESC:
            self.fill_1d_tma_descriptor_inner = (
                triton.runtime.driver.active.utils.fill_1d_tma_descriptor
            )
            self.fill_2d_tma_descriptor_inner = (
                triton.runtime.driver.active.utils.fill_2d_tma_descriptor
            )
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
            )
        else:
            self.cuda_descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8
            )

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]
