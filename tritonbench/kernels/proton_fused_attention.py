"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team. This has been integrated with Proton as a tutorial to enable proton integration with
TritonBench.

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import logging
import os
from typing import Optional
from uuid import uuid4

from tritonbench.utils.env_utils import is_fbcode

if is_fbcode():
    import manifold.clients.python as manifold
else:
    manifold = None

import numpy as np
import torch
import triton

# Note: This only works with 3.3.1+fb or earlier
try:
    import triton.intraprof as proton  # @manual=//triton:triton

    SUPPORTS_PROTON = True
except ModuleNotFoundError:
    SUPPORTS_PROTON = False

import triton.language as tl

from .attention_utils import HAS_TMA_DESC, TmaAutoTuneHelper, WITH_TMA

if HAS_TMA_DESC:
    logging.info(
        "TMA benchmarks will be running with experimental grid constant TMA descriptor.",
    )
else:
    logging.info(
        "TMA benchmarks will be running without grid constant TMA descriptor.",
    )

HAS_NEW_TMA = hasattr(triton, "set_allocator") and hasattr(tl, "make_tensor_descriptor")
ENABLE_TMA = (WITH_TMA == "1") and HAS_NEW_TMA


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    desc_k,
    desc_v,
    Q,
    qvk_offset,
    stride_kn,
    stride_vn,
    stride_vk,  #
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    if not ENABLE_TMA:
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):  # , loop_schedule=LOOP_SCHEDULE):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if ENABLE_TMA:
            k = desc_k.load(
                [start_n.to(tl.int32) + (qvk_offset // stride_kn).to(tl.int32), 0]
            )
        else:
            k = tl.load(K_block_ptr)
        if ENABLE_TMA:
            k = tl.trans(k)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if ENABLE_TMA:
            if fp8_v:
                v = desc_v.load(
                    [(qvk_offset // stride_vn).to(tl.int32), start_n.to(tl.int32)]
                )
            else:
                v = desc_v.load([(qvk_offset // stride_vk + start_n).to(tl.int32), 0])
        else:
            v = tl.load(V_block_ptr)
        if fp8_v:
            if ENABLE_TMA:
                v = tl.trans(v)
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        if not ENABLE_TMA:
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_compute(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    off_hz,
    pid,
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
):
    start_m = pid  # tl.program_id(0)
    # off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    K_block_ptr = None
    V_block_ptr = None
    Q_block_ptr = None
    O_block_ptr = None
    if not ENABLE_TMA:
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    if ENABLE_TMA:
        q = desc_q.load([(qvk_offset // stride_qm + start_m * BLOCK_M).to(tl.int32), 0])
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            desc_k,
            desc_v,
            Q,
            qvk_offset,
            stride_kn,
            stride_vn,
            stride_vk,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
            ENABLE_TMA,
            LOOP_SCHEDULE,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            desc_k,
            desc_v,
            Q,
            qvk_offset,
            stride_kn,
            stride_vn,
            stride_vk,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
            ENABLE_TMA,
            LOOP_SCHEDULE,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    m_mask = off_hz * N_CTX + offs_m < N_CTX
    tl.store(m_ptrs, m_i, mask=m_mask)
    if ENABLE_TMA:
        desc_o.store(
            [(qvk_offset // stride_om + start_m * BLOCK_M).to(tl.int32), 0],
            acc.to(Out.type.element_ty),
        )
    else:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))


# This was the best config as determined on Blackwell
base_config = [
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "ENABLE_TMA": ENABLE_TMA,
            "LOOP_SCHEDULE": "default",
            "ENABLE_WS": False,
        },
        num_stages=3,
        num_warps=8,
    ),
]


@triton.autotune(base_config, key=["N_CTX"])
@triton.jit
def _attn_fwd_base_opt(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    profile_mem,  # *Pointer* to profile memory.
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    pid = tl.program_id(0)
    off_hz = tl.program_id(1)

    # TMA descriptor creation. This only supports device side TMA.
    desc_q = None
    desc_k = None
    desc_v = None
    desc_o = None

    if ENABLE_TMA:
        desc_k = tl.make_tensor_descriptor(
            K,
            shape=[Z * H * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
        if V.dtype == torch.float8_e5m2:
            desc_v = tl.make_tensor_descriptor(
                V,
                shape=[Z * H * HEAD_DIM, N_CTX],
                strides=[N_CTX, 1],
                block_shape=[HEAD_DIM, BLOCK_N],
            )
        else:
            desc_v = tl.make_tensor_descriptor(
                V,
                shape=[Z * H * N_CTX, HEAD_DIM],
                strides=[HEAD_DIM, 1],
                block_shape=[BLOCK_N, HEAD_DIM],
            )

        desc_q = tl.make_tensor_descriptor(
            Q,
            shape=[Z * H * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )
        desc_o = tl.make_tensor_descriptor(
            Out,
            shape=[Z * H * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )

    # Both base and opt use the same compute function
    _attn_fwd_compute(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        off_hz,
        pid,
        Z,
        H,
        N_CTX,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        STAGE,
        ENABLE_TMA,
        LOOP_SCHEDULE,
    )


class _attention_opt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, baseVariant):
        if not SUPPORTS_PROTON:
            raise RuntimeError(
                "This version of Triton does not support running this file with proton."
            )
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1

        BATCH, H, N_CTX = q.shape[0], q.shape[1], q.shape[2]

        # no autotune with fixed BLOCK_N
        if HAS_TMA_DESC is True and torch.version.hip is None:
            desc_helper = TmaAutoTuneHelper()
            desc_helper.init_tma_descriptor("k")
            desc_helper.init_tma_descriptor("v")
            desc_helper.init_tma_descriptor("q")
            desc_helper.init_tma_descriptor("o")
        else:
            desc_helper = None

        def grid_tma(META):
            if META["ENABLE_TMA"] is False or HAS_TMA_DESC is False:
                return (
                    # grid partitioning: num_consumer_groups * BLOCK_M
                    # data partitioning: BLOCK_M
                    triton.cdiv(q.shape[2], META["BLOCK_M"]),  # num_consumer_groups
                    q.shape[0] * q.shape[1],
                    1,
                )
            nonlocal desc_helper
            desc_helper.fill_2d_tma_descriptor(
                "k",
                k.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META["BLOCK_N"],
                HEAD_DIM_Q,
                k.element_size(),
            )
            if v.dtype == torch.float8_e5m2:
                desc_helper.fill_2d_tma_descriptor(
                    "v",
                    v.data_ptr(),
                    BATCH * H * HEAD_DIM_Q,
                    N_CTX,
                    HEAD_DIM_Q,
                    META["BLOCK_N"],
                    v.element_size(),
                )
            else:
                desc_helper.fill_2d_tma_descriptor(
                    "v",
                    v.data_ptr(),
                    BATCH * H * N_CTX,
                    HEAD_DIM_Q,
                    META["BLOCK_N"],
                    HEAD_DIM_Q,
                    v.element_size(),
                )
            desc_helper.fill_2d_tma_descriptor(
                "q",
                q.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META["BLOCK_M"]
                // (2 if META["ENABLE_WS"] else 1),  # data partitioning: halve
                HEAD_DIM_Q,
                q.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "o",
                o.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META["BLOCK_M"]
                // (2 if META["ENABLE_WS"] else 1),  # data partitioning: halve
                HEAD_DIM_Q,
                o.element_size(),
            )
            return (
                triton.cdiv(q.shape[2], META["BLOCK_M"]),
                q.shape[0] * q.shape[1],
                1,
            )

        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        if HAS_NEW_TMA:
            triton.set_allocator(alloc_fn)

        if baseVariant != "base_opt":
            raise ValueError(f"Unknown base variant {baseVariant}")

        # TODO: Extend this as you modify the IR.
        named_region = {
            0: "whole_kernel_time",
            1: "prologue",
            2: "tensor_map_alloc",
            3: "loop_body",
            4: "epilogue",
            5: "loop_section1",
            6: "loop_section2",
            7: "loop_section3",
            8: "loop_section4",
            9: "loop_section5",
        }
        proton_grid = proton.const_grid(
            grid_tma,
            # config from autotune
            autotune_configs=base_config,
            # local variables that used in grid_tma function
            func_args={
                "HAS_TMA_DESC": HAS_TMA_DESC,
                "q": q,
                "k": k,
                "v": v,
                "o": o,
                "BATCH": BATCH,
                "H": H,
                "HEAD_DIM_Q": HEAD_DIM_Q,
            },
            # copy all named args except `profile_mem` in the kernel callsite
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )
        pconfig = proton.get_intra_kernel_config(num_warps=8, names=named_region)
        profile_size = proton.intra_kernel_memsize(np.prod(proton_grid), pconfig)
        profile_mem = torch.empty(profile_size, device="cuda", dtype=torch.uint32)
        kernel_info = _attn_fwd_base_opt[grid_tma](
            q,
            k,
            v,
            sm_scale,
            M,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            profile_mem,
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )
        # Note: You need to run this with TRITON_ALWAYS_COMPILE=1 and
        # TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=override_dir
        # to actually generate a trace.
        user = os.getenv("USER")
        trace_filepath = f"/home/{user}/chrome_trace.json"
        proton.dump_chrome_trace(
            np.prod(proton_grid),
            pconfig,
            profile_mem,
            trace_filepath,
            kernel_info,
        )
        if is_fbcode():
            if os.path.exists(trace_filepath):
                bucket_name = "tc_bench_ci"
                random_id = uuid4()
                dest_filepath = (
                    f"tree/tritonbench/proton_traces/fa_trace_{random_id}.json"
                )
                client = manifold.ManifoldClient(bucket_name)
                client.sync_put(dest_filepath, trace_filepath)
                logging.info(
                    f"Trace uploaded to manifold. Trace URL: https://www.internalfb.com/intern/perfdoctor/trace_view?filepath={dest_filepath}&bucket={bucket_name}"
                )
            else:
                logging.info(
                    "No proton trace generated. Please verify you have provided a modified IR with proton ops."
                )

        return o


attention_opt = _attention_opt.apply
