"""
Group GEMM
============================
This group gemm kernel launches a fixed number of CTA to compute a group
of gemms. The scheduling is static and we do it on device.
"""

import itertools
from typing import Optional

# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import triton
import triton.language as tl

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore

except ImportError:
    # suppress type checking errors
    tlx = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148


def torch_dtype_to_triton_dtype(dtype):
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float8_e4m3fn:
        return tl.float8e4nv
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": BLOCK_M,
                "BLOCK_SIZE_N": BLOCK_N,
                "BLOCK_SIZE_K": BLOCK_K,
                "NUM_SMS": num_sms(),
            }
        )
        for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product([128, 256], repeat=3)
    ],
    key=["group_size"],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    DTYPE: tl.constexpr,
    # number of virtual SM
    NUM_SMS: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(DTYPE))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(DTYPE))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(DTYPE))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(DTYPE)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SMS

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def triton_group_gemm_fn(
    d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_C, group_size, dtype
):
    grid = lambda META: (META["NUM_SMS"],)
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        torch_dtype_to_triton_dtype(dtype),
    )

    return group_C


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


tlx_configs = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "NUM_SMS": num_sms(),
            "NUM_SMEM_BUFFERS": 4,
            "NUM_TMEM_BUFFERS": 2,
            "EPILOGUE_SUBTILE": 2,
        },
        num_warps=4,
        num_stages=1,
    ),
]


@triton.autotune(
    tlx_configs,
    key=["group_a_ptrs", "group_b_ptrs", "gropup_c_ptrs", "group_size"],
)
@triton.jit
def grouped_matmul_tlx_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    DTYPE: tl.constexpr,
    # number of virtual SM
    NUM_SMS: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,  #
    NUM_TMEM_BUFFERS: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
):
    # allocate NUM_SMEM_BUFFERS buffers
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), DTYPE, NUM_SMEM_BUFFERS)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), DTYPE, NUM_SMEM_BUFFERS)
    # use multiple TMEM buffers to overlap MMA and epilogue
    tmem_buffers = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS,
        tlx.storage_kind.tmem,
    )

    # allocate barriers
    smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):  # epilogue consumer
            tile_idx = tl.program_id(0)
            last_problem_end = 0
            accum_cnt_tmem = 0
            for g in range(group_size):
                # get the gemm size of the current problem
                gm = tl.load(group_gemm_sizes + g * 3)
                gn = tl.load(group_gemm_sizes + g * 3 + 1)
                gk = tl.load(group_gemm_sizes + g * 3 + 2)
                num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
                num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
                num_tiles = num_m_tiles * num_n_tiles
                if (
                    tile_idx >= last_problem_end
                    and tile_idx < last_problem_end + num_tiles
                ):
                    ldc = tl.load(g_lds + g * 3 + 2)
                    c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(DTYPE))
                    c_desc = tl.make_tensor_descriptor(
                        c_ptr,
                        shape=[gm, gn],
                        strides=[ldc, 1],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE],
                    )

                    # iterate through the tiles in the current gemm problem
                    while (
                        tile_idx >= last_problem_end
                        and tile_idx < last_problem_end + num_tiles
                    ):
                        # figure out tile coordinates
                        tile_idx_in_gemm = tile_idx - last_problem_end
                        tile_m_idx = tile_idx_in_gemm // num_n_tiles
                        tile_n_idx = tile_idx_in_gemm % num_n_tiles

                        tmem_buf, tmem_phase = _get_bufidx_phase(
                            accum_cnt_tmem, NUM_TMEM_BUFFERS
                        )
                        tlx.barrier_wait(tmem_full_bars[tmem_buf], tmem_phase)

                        # load the result from TMEM to registers
                        acc_tmem = tmem_buffers[tmem_buf]

                        offs_cm = tile_m_idx * BLOCK_SIZE_M
                        offs_cn = tile_n_idx * BLOCK_SIZE_N

                        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            acc_slice = tlx.local_slice(
                                acc_tmem,
                                [0, slice_id * slice_size],
                                [BLOCK_SIZE_M, slice_size],
                            )
                            result = tlx.local_load(acc_slice)
                            c = result.to(DTYPE)
                            c_desc.store([offs_cm, offs_cn + slice_id * slice_size], c)

                        # done storing this buffer, signal MMA consumer to resume writing to it
                        tlx.barrier_arrive(tmem_empty_bars[tmem_buf], 1)
                        accum_cnt_tmem += 1
                        # go to the next tile by advancing NUM_SMS
                        tile_idx += NUM_SMS

                # get ready to go to the next gemm problem
                last_problem_end = last_problem_end + num_tiles

        with tlx.async_task(num_warps=1, num_regs=48):  # MMA consumer
            tile_idx = tl.program_id(0)
            last_problem_end = 0
            accum_cnt_smem = 0
            accum_cnt_tmem = 0
            for g in range(group_size):
                # get the gemm size of the current problem
                gm = tl.load(group_gemm_sizes + g * 3)
                gn = tl.load(group_gemm_sizes + g * 3 + 1)
                gk = tl.load(group_gemm_sizes + g * 3 + 2)
                num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
                num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
                num_tiles = num_m_tiles * num_n_tiles
                if (
                    tile_idx >= last_problem_end
                    and tile_idx < last_problem_end + num_tiles
                ):
                    # iterate through the tiles in the current gemm problem
                    while (
                        tile_idx >= last_problem_end
                        and tile_idx < last_problem_end + num_tiles
                    ):
                        k = gk

                        # do regular gemm here
                        tmem_buf, tmem_phase = _get_bufidx_phase(
                            accum_cnt_tmem, NUM_TMEM_BUFFERS
                        )

                        # wait epilogue consumer to be done with the buffer before reusing it
                        tlx.barrier_wait(tmem_empty_bars[tmem_buf], tmem_phase ^ 1)

                        for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                            smem_buf, smem_phase = _get_bufidx_phase(
                                accum_cnt_smem, NUM_SMEM_BUFFERS
                            )
                            # wait for current phase(round) of load for this buf
                            tlx.barrier_wait(smem_full_bars[smem_buf], smem_phase)
                            # buffer is now ready with loaded data, tlx.async_dot will signal `mBarrier` when done
                            tlx.async_dot(
                                buffers_A[smem_buf],
                                buffers_B[smem_buf],
                                tmem_buffers[tmem_buf],
                                use_acc=kk > 0,
                                mBarriers=[smem_empty_bars[smem_buf]],
                                out_dtype=tl.float32,
                            )
                            accum_cnt_smem += 1

                        # done filling this buffer, signal epilogue consumer
                        tlx.tcgen05_commit(tmem_full_bars[tmem_buf])
                        accum_cnt_tmem += 1
                        # go to the next tile by advancing NUM_SMS
                        tile_idx += NUM_SMS

                # get ready to go to the next gemm problem
                last_problem_end = last_problem_end + num_tiles

        with tlx.async_task(num_warps=1, num_regs=48):  # producer, TMA load
            tile_idx = tl.program_id(0)
            last_problem_end = 0
            accum_cnt = 0
            accum_cnt_outer = 0

            # Allocate global scratch for tensor descriptors (pipelining)
            desc_a_ptrs = tlx.allocate_tensor_descriptor(num=NUM_SMEM_BUFFERS + 1)
            desc_b_ptrs = tlx.allocate_tensor_descriptor(num=NUM_SMEM_BUFFERS + 1)

            for g in range(group_size):
                # get the gemm size of the current problem
                gm = tl.load(group_gemm_sizes + g * 3)
                gn = tl.load(group_gemm_sizes + g * 3 + 1)
                gk = tl.load(group_gemm_sizes + g * 3 + 2)
                num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
                num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
                num_tiles = num_m_tiles * num_n_tiles
                num_k_tiles = tl.cdiv(gk, BLOCK_SIZE_K)

                if (
                    tile_idx >= last_problem_end
                    and tile_idx < last_problem_end + num_tiles
                ):
                    # pick up a tile from the current gemm problem
                    lda = tl.load(g_lds + g * 3)
                    ldb = tl.load(g_lds + g * 3 + 1)

                    a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(DTYPE))
                    b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(DTYPE))

                    desc_buf, _ = _get_bufidx_phase(
                        accum_cnt_outer, NUM_SMEM_BUFFERS + 1
                    )

                    # Create tensor descriptors in global scratch (for pipelining across problems)
                    tlx.make_tensor_descriptor(
                        desc_ptr=desc_a_ptrs[desc_buf],
                        base=a_ptr,
                        shape=[gm, gk],
                        strides=[lda, 1],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                    )

                    tlx.make_tensor_descriptor(
                        desc_ptr=desc_b_ptrs[desc_buf],
                        base=b_ptr,
                        shape=[gk, gn],
                        strides=[ldb, 1],
                        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                    )

                    # iterate through the tiles in the current gemm problem
                    while (
                        tile_idx >= last_problem_end
                        and tile_idx < last_problem_end + num_tiles
                    ):
                        # figure out tile coordinates
                        tile_idx_in_gemm = tile_idx - last_problem_end
                        tile_m_idx = tile_idx_in_gemm // num_n_tiles
                        tile_n_idx = tile_idx_in_gemm % num_n_tiles

                        # Reinterpret descriptor pointers for TMA operations
                        a_desc = tlx.reinterpret_tensor_descriptor(
                            desc_ptr=desc_a_ptrs[desc_buf],
                            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                            dtype=DTYPE,
                        )
                        b_desc = tlx.reinterpret_tensor_descriptor(
                            desc_ptr=desc_b_ptrs[desc_buf],
                            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                            dtype=DTYPE,
                        )

                        # do regular gemm here
                        offs_am = tile_m_idx * BLOCK_SIZE_M
                        offs_bn = tile_n_idx * BLOCK_SIZE_N

                        for kk in range(0, num_k_tiles):
                            buf, phase = _get_bufidx_phase(accum_cnt, NUM_SMEM_BUFFERS)
                            tlx.barrier_wait(smem_empty_bars[buf], phase ^ 1)
                            tlx.barrier_expect_bytes(
                                smem_full_bars[buf],
                                2 * (BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K,
                            )  # float16
                            tlx.async_descriptor_load(
                                a_desc,
                                buffers_A[buf],
                                [offs_am, kk * BLOCK_SIZE_K],
                                smem_full_bars[buf],
                            )
                            tlx.async_descriptor_load(
                                b_desc,
                                buffers_B[buf],
                                [kk * BLOCK_SIZE_K, offs_bn],
                                smem_full_bars[buf],
                            )
                            accum_cnt += 1

                        # go to the next tile by advancing NUM_SMS
                        tile_idx += NUM_SMS

                    accum_cnt_outer += 1
                # get ready to go to the next gemm problem
                last_problem_end = last_problem_end + num_tiles


def tlx_group_gemm_fn(
    d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_C, group_size, dtype
):
    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    grid = lambda META: (META["NUM_SMS"],)
    grouped_matmul_tlx_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        torch_dtype_to_triton_dtype(dtype),
    )

    return group_C
