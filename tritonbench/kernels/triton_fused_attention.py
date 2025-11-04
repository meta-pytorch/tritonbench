"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import sys

from typing import Optional

import torch

import triton
import triton.language as tl

from .attention_utils import (
    HAS_EXPLICIT_WS,  # guard new tuning configs such as num_consumer_groups
    HAS_TMA_DESC,
    PEEL_LAST,
    TmaAutoTuneHelper,
    WITH_COMPPIPE,
    WITH_TMA,
)


if HAS_TMA_DESC:
    print(
        "TMA benchmarks will be running with experimental grid constant TMA descriptor.",
        file=sys.stderr,
    )
else:
    print(
        "TMA benchmarks will be running without grid constant TMA descriptor.",
        file=sys.stderr,
    )


@triton.jit
def _attn_fwd_iteration(
    q,
    k,
    offs_m,
    start_n,
    offs_n,
    qk_scale,
    l_i,
    m_i,
    acc,
    v,
    fp8_v: tl.constexpr,
    STAGE: tl.constexpr,
):
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
    if fp8_v:
        p = p.to(tl.float8e5)
    else:
        p = p.to(tl.bfloat16)
    acc = tl.dot(p, v, acc)
    # update m_i and l_i
    m_i = m_ij
    return l_i, m_i, acc


@triton.jit
def _attn_fwd_inner_autows(
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
    WARP_SPECIALIZE: tl.constexpr,
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
    for start_n in tl.range(
        lo, hi, BLOCK_N, warp_specialize=WARP_SPECIALIZE
    ):  # , loop_schedule=LOOP_SCHEDULE):
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
        if ENABLE_TMA:
            if fp8_v:
                v = desc_v.load(
                    [(qvk_offset // stride_vn).to(tl.int32), start_n.to(tl.int32)]
                )
                v = tl.trans(v)
            else:
                v = desc_v.load([(qvk_offset // stride_vk + start_n).to(tl.int32), 0])
        else:
            v = tl.load(V_block_ptr)
        l_i, m_i, acc = _attn_fwd_iteration(
            q, k, offs_m, start_n, offs_n, qk_scale, l_i, m_i, acc, v, fp8_v, STAGE
        )
        if not ENABLE_TMA:
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


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
    for start_n in tl.range(lo, hi, BLOCK_N):
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
        if ENABLE_TMA:
            if fp8_v:
                v = desc_v.load(
                    [(qvk_offset // stride_vn).to(tl.int32), start_n.to(tl.int32)]
                )
                v = tl.trans(v)
            else:
                v = desc_v.load([(qvk_offset // stride_vk + start_n).to(tl.int32), 0])
        else:
            v = tl.load(V_block_ptr)
        l_i, m_i, acc = _attn_fwd_iteration(
            q, k, offs_m, start_n, offs_n, qk_scale, l_i, m_i, acc, v, fp8_v, STAGE
        )
        if not ENABLE_TMA:
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_inner_ws(
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
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
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
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
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
def _attn_fwd_inner_ws_with_dp(
    acc0,
    acc1,
    l_i0,
    l_i1,
    m_i0,
    m_i1,
    q0,  #
    q1,
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
    offs_m0: tl.constexpr,
    offs_m1: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    FIRST_MMA: tl.constexpr,
    LAST_MMA: tl.constexpr,
    FIRST_SOFTMAX: tl.constexpr,
    LAST_SOFTMAX: tl.constexpr,
    LOAD_K: tl.constexpr,
    LOAD_V: tl.constexpr,
    FIRST_CORRECTION: tl.constexpr,
    LAST_CORRECTION: tl.constexpr,
    ALPHA_REMAT: tl.constexpr,
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
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")

        if ENABLE_TMA:  # feeds into gemm
            k = tl.trans(k)
        qk0 = tl.dot(q0, k)
        qk1 = tl.dot(q1, k)
        if STAGE == 2:
            mask = offs_m0[:, None] >= (start_n + offs_n[None, :])
            qk0 = qk0 * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij0 = tl.maximum(m_i0, tl.max(qk0, 1))
            qk0 -= m_ij0[:, None]
        else:
            m_ij0 = tl.maximum(m_i0, tl.max(qk0, 1) * qk_scale)
            qk0 = qk0 * qk_scale - m_ij0[:, None]
        p0 = tl.math.exp2(qk0)
        l_ij0 = tl.sum(p0, 1)
        # -- update m_i and l_i
        alpha0 = tl.math.exp2(m_i0 - m_ij0)
        l_i0 = l_i0 * alpha0 + l_ij0
        if ALPHA_REMAT:
            alpha0_re = tl.math.exp2(m_i0 - m_ij0)
            # -- update output accumulator --
            acc0 = acc0 * alpha0_re[:, None]
        else:
            acc0 = acc0 * alpha0[:, None]
        # update acc
        if fp8_v:
            p0 = p0.to(tl.float8e5)
        else:
            p0 = p0.to(tl.bfloat16)
        # update m_i and l_i
        m_i0 = m_ij0
        if STAGE == 2:
            mask = offs_m1[:, None] >= (start_n + offs_n[None, :])
            qk1 = qk1 * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij1 = tl.maximum(m_i1, tl.max(qk1, 1))
            qk1 -= m_ij1[:, None]
        else:
            m_ij1 = tl.maximum(m_i1, tl.max(qk1, 1) * qk_scale)
            qk1 = qk1 * qk_scale - m_ij1[:, None]
        p1 = tl.math.exp2(qk1)
        l_ij1 = tl.sum(p1, 1)
        # -- update m_i and l_i
        alpha1 = tl.math.exp2(m_i1 - m_ij1)
        l_i1 = l_i1 * alpha1 + l_ij1
        if ALPHA_REMAT:
            alpha1_re = tl.math.exp2(m_i1 - m_ij1)
            # -- update output accumulator --
            acc1 = acc1 * alpha1_re[:, None]
        else:
            acc1 = acc1 * alpha1[:, None]
        # update acc
        if fp8_v:
            p1 = p1.to(tl.float8e5)
        else:
            p1 = p1.to(tl.bfloat16)
        # update m_i and l_i
        m_i1 = m_ij1
        if ENABLE_TMA:
            if fp8_v:
                v = desc_v.load(
                    [(qvk_offset // stride_vn).to(tl.int32), start_n.to(tl.int32)]
                )
            else:
                v = desc_v.load([(qvk_offset // stride_vk + start_n).to(tl.int32), 0])
        else:
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        if fp8_v:
            if ENABLE_TMA:
                v = tl.trans(v)
        acc0 = tl.dot(p0, v, acc0)
        acc1 = tl.dot(p1, v, acc1)
        if not ENABLE_TMA:
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc0, acc1, l_i0, l_i1, m_i0, m_i1


# We don't run auto-tuning every time to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
HAS_NEW_TMA = hasattr(triton, "set_allocator") and hasattr(tl, "make_tensor_descriptor")

# This part is for computation pipelining
schedList = ["default", "FA_firstDot", "FA_secondDot"] if WITH_COMPPIPE else ["default"]
# TODO: incorrect result with PEEL_LAST + FA_firstDot + WarpSpec + TMA
schedList = ["FA_secondDot"] if PEEL_LAST else schedList

tmaList = [True] if WITH_TMA and HAS_NEW_TMA else [False]


def get_fwd_config_space(
    persistent: bool, enable_ws: bool, support_explicit_ws: bool, enable_tma: bool
):
    configs = []
    bmList = [128] if enable_ws else [64, 128]
    bnList = [64, 128]  # To handle hDim of 64, we need BLOCK_N to be <= 64
    wList = [4] if enable_ws else [4, 8]
    stageList = [2] if enable_ws else [3, 4, 7]
    for BM in bmList:
        for BN in bnList:
            for sched in schedList:  # set in global scope
                for w in wList:
                    for stage in stageList:
                        base_config_dict = {
                            "BLOCK_M": BM,
                            "BLOCK_N": BN,
                            "ENABLE_TMA": enable_tma,
                            "LOOP_SCHEDULE": sched,
                        }
                        config_dicts = []
                        if persistent:
                            config_dicts.append(
                                {
                                    **base_config_dict,
                                    "GRID_MULTIPLE": 1,  # This can be set to multiple values
                                }
                            )
                        else:
                            config_dicts.append(base_config_dict)
                        for config_dict in config_dicts:
                            if support_explicit_ws:
                                if enable_ws:
                                    configs.append(
                                        triton.Config(
                                            config_dict,
                                            num_warps=w,
                                            num_stages=stage,
                                            num_buffers_warp_spec=2,
                                            num_consumer_groups=2,
                                            reg_dec_producer=24,
                                            reg_inc_consumer=240,
                                        )
                                    )
                                else:
                                    configs.append(
                                        triton.Config(
                                            config_dict,
                                            num_warps=w,
                                            num_stages=stage,
                                            num_buffers_warp_spec=0,
                                            num_consumer_groups=0,
                                        )
                                    )
                            else:
                                configs.append(
                                    triton.Config(
                                        config_dict,
                                        num_warps=w,
                                        num_stages=stage,
                                    )
                                )
    return configs


# BLOCK_M: 128, BLOCK_N: 128, ENABLE_TMA: False, LOOP_SCHEDULE: default, num_warps: 8, num_ctas: 1, num_stages: 3
if torch.version.hip is None:
    configsOrig = get_fwd_config_space(False, False, HAS_EXPLICIT_WS, False)
else:
    configsOrig = [
        (
            triton.Config(
                {
                    "BLOCK_M": BM,
                    "BLOCK_N": BN,
                    "ENABLE_TMA": False,
                    "LOOP_SCHEDULE": "default",
                    "waves_per_eu": wpe,
                    "kpack": 2,
                },
                num_stages=s,
                num_warps=w,
            )
        )
        for BM in [16, 32, 64, 128]
        for BN in [16, 32, 64, 128]
        for s in ([1, 2])
        for w in [1, 2, 4, 8]
        for wpe in [0, 1, 2, 3, 4]
    ]
# no WS, no TMA, with CompPipe
configsOpt = get_fwd_config_space(False, False, HAS_EXPLICIT_WS, False)
# no WS, with TMA and CompPipe
configsTma = get_fwd_config_space(False, False, HAS_EXPLICIT_WS, True)
# no TMA, with WS and CompPipe
configsWS = get_fwd_config_space(False, True, HAS_EXPLICIT_WS, False)
# TMA, WS, and CompPipe
configsTmaWS = get_fwd_config_space(
    False, True, HAS_EXPLICIT_WS, WITH_TMA and HAS_NEW_TMA
)
configsTmaWSPersistent = get_fwd_config_space(
    True, True, HAS_EXPLICIT_WS, WITH_TMA and HAS_NEW_TMA
)


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


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
    WARP_SPECIALIZE: tl.constexpr,
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
        if WARP_SPECIALIZE:
            acc, l_i, m_i = _attn_fwd_inner_autows(
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
                WARP_SPECIALIZE,
            )
        else:
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
        if WARP_SPECIALIZE:
            acc, l_i, m_i = _attn_fwd_inner_autows(
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
                WARP_SPECIALIZE,
            )
        else:
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


@triton.jit
def _attn_fwd_compute_ws(
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
        q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_ws(
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
        acc, l_i, m_i = _attn_fwd_inner_ws(
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
    tl.store(m_ptrs, m_i)
    if ENABLE_TMA:
        desc_o.store(
            [(qvk_offset // stride_om + start_m * BLOCK_M).to(tl.int32), 0],
            acc.to(Out.type.element_ty),
        )
    else:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# only supports TMA, and explicit async_task
@triton.jit
def _attn_fwd_compute_ws_with_dp(
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
    BLOCK_M_HALF: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    FIRST_MMA: tl.constexpr,
    LAST_MMA: tl.constexpr,
    FIRST_SOFTMAX: tl.constexpr,
    LAST_SOFTMAX: tl.constexpr,
    LOAD_K: tl.constexpr,
    LOAD_V: tl.constexpr,
    FIRST_CORRECTION: tl.constexpr,
    LAST_CORRECTION: tl.constexpr,
    ALPHA_REMAT: tl.constexpr,
    FIRST_LOADQ: tl.constexpr,
    LAST_LOADQ: tl.constexpr,
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
    # initialize offsets
    offs_m0 = start_m * BLOCK_M + tl.arange(0, BLOCK_M_HALF)
    offs_m1 = start_m * BLOCK_M + BLOCK_M_HALF + tl.arange(0, BLOCK_M_HALF)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i0 = tl.zeros([BLOCK_M_HALF], dtype=tl.float32) - float("inf")
    l_i0 = tl.zeros([BLOCK_M_HALF], dtype=tl.float32) + 1.0
    acc0 = tl.zeros([BLOCK_M_HALF, HEAD_DIM], dtype=tl.float32)
    m_i1 = tl.zeros([BLOCK_M_HALF], dtype=tl.float32) - float("inf")
    l_i1 = tl.zeros([BLOCK_M_HALF], dtype=tl.float32) + 1.0
    acc1 = tl.zeros([BLOCK_M_HALF, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    # q0 will be BLOCK_M, each kernel invocation will handle 2 * BLOCK_M
    if ENABLE_TMA:
        q0 = desc_q.load(
            [(qvk_offset // stride_qm + start_m * BLOCK_M).to(tl.int32), 0]
        )
    else:
        q0 = tl.load(Q_block_ptr)
    if ENABLE_TMA:
        q1 = desc_q.load(
            [
                (qvk_offset // stride_qm + start_m * BLOCK_M + BLOCK_M_HALF).to(
                    tl.int32
                ),
                0,
            ]
        )
    else:
        q1 = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc0, acc1, l_i0, l_i1, m_i0, m_i1 = _attn_fwd_inner_ws_with_dp(
            acc0,
            acc1,
            l_i0,
            l_i1,
            m_i0,
            m_i1,
            q0,
            q1,
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
            offs_m0,
            offs_m1,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
            ENABLE_TMA,
            LOOP_SCHEDULE,
            FIRST_MMA,
            LAST_MMA,
            FIRST_SOFTMAX,
            LAST_SOFTMAX,
            LOAD_K,
            LOAD_V,
            FIRST_CORRECTION,
            LAST_CORRECTION,
            ALPHA_REMAT,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_ws_with_dp(
            acc0,
            acc1,
            l_i0,
            l_i1,
            m_i0,
            m_i1,
            q0,
            q1,
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
            offs_m0,
            offs_m1,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
            ENABLE_TMA,
            LOOP_SCHEDULE,
            FIRST_MMA,
            LAST_MMA,
            FIRST_SOFTMAX,
            LAST_SOFTMAX,
            LOAD_K,
            LOAD_V,
            FIRST_CORRECTION,
            LAST_CORRECTION,
            ALPHA_REMAT,
        )
    # epilogue
    m_i0 += tl.math.log2(l_i0)
    acc0 = acc0 / l_i0[:, None]
    m_ptrs0 = M + off_hz * N_CTX + offs_m0
    tl.store(m_ptrs0, m_i0)
    if ENABLE_TMA:
        desc_o.store(
            [(qvk_offset // stride_om + start_m * BLOCK_M).to(tl.int32), 0],
            acc0.to(Out.type.element_ty),
        )
    else:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    m_i1 += tl.math.log2(l_i1)
    acc1 = acc1 / l_i1[:, None]
    m_ptrs1 = M + off_hz * N_CTX + offs_m1
    tl.store(m_ptrs1, m_i1)
    if ENABLE_TMA:
        desc_o.store(
            [
                (qvk_offset // stride_om + start_m * BLOCK_M + BLOCK_M_HALF).to(
                    tl.int32
                ),
                0,
            ],
            acc1.to(Out.type.element_ty),
        )
    else:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.autotune(list(filter(keep, configsWS)), key=["N_CTX"])
@triton.jit
def _attn_fwd_ws(
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
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
    HAS_EXPLICIT_WS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    pid = tl.program_id(0)
    off_hz = tl.program_id(1)
    if HAS_EXPLICIT_WS:
        _attn_fwd_compute_ws(
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
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            STAGE,
            ENABLE_TMA,
            LOOP_SCHEDULE,
        )
    else:
        _attn_fwd_compute(
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
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            STAGE,
            ENABLE_TMA,
            LOOP_SCHEDULE,
            False,  # warp_specialize on hopper is not ready yet
        )


@triton.autotune(list(filter(keep, configsOrig + configsOpt)), key=["N_CTX"])
@triton.jit
def _attn_fwd_base_opt(
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
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
):
    tl.assume(stride_qz >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_qm >= 0)
    tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_kn >= 0)
    tl.assume(stride_kk >= 0)
    tl.assume(stride_vz >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_vk >= 0)
    tl.assume(stride_vn >= 0)
    tl.assume(stride_oz >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_om >= 0)
    tl.assume(stride_on >= 0)
    tl.assume(Z >= 0)
    tl.assume(H >= 0)

    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

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
        start_m,
        Z,
        H,
        N_CTX,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        STAGE,
        ENABLE_TMA,
        LOOP_SCHEDULE,
        False,  # WARP_SPECIALIZE
    )


def prune_invalid_configs(configs, named_args, **kwargs):
    ENABLE_WS = kwargs["ENABLE_WS"]
    # Choose configsTmaWS when ENABLE_WS is True
    if ENABLE_WS:
        return [conf for conf in configs if conf in configsTmaWS]
    return [conf for conf in configs if conf in configsTma]


# when ENABLE_WS is true, we can't use configsTma
# use either configsTma or configsTmaWS, not configsTma + configsTmaWS
@triton.autotune(
    list(filter(keep, configsTma + configsTmaWS)),
    key=["N_CTX"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd_tma_unified(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
    HAS_EXPLICIT_WS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    pid = tl.program_id(0)
    off_hz = tl.program_id(1)

    # TMA descriptor creation - shared for both WS and non-WS paths
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

    # Call appropriate compute function based on ENABLE_WS
    if ENABLE_WS:
        if HAS_EXPLICIT_WS:
            _attn_fwd_compute_ws(
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
        else:
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
                False,  # warp_specialize on hopper is not ready yet
            )
    else:
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
            False,  # WARP_SPECIALIZE
        )


@triton.autotune(list(filter(keep, configsTmaWSPersistent)), key=["N_CTX"])
@triton.jit
def _attn_fwd_tma_ws_persistent(  # Q, V, desc_k, desc_v, sm_scale, M, Out,  #
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
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
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
    GRID_MULTIPLE: tl.constexpr,
    HAS_EXPLICIT_WS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    # original grid
    #   triton.cdiv(q.shape[2], META["BLOCK_M"]),
    #   q.shape[0] * q.shape[1],
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    # Initialize descriptors as None
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

    for _ in range(0, tiles_per_sm):
        # This has much better cache locality than
        #     pid = tile_idx // (Z * H)
        #     off_hz = tile_idx % (Z * H)  # tl.program_id(1)
        pid = tile_idx % n_tile_num
        off_hz = tile_idx // n_tile_num
        if HAS_EXPLICIT_WS:
            _attn_fwd_compute_ws(
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
                BLOCK_M,
                BLOCK_N,
                HEAD_DIM,
                STAGE,
                ENABLE_TMA,
                LOOP_SCHEDULE,
            )
        else:
            _attn_fwd_compute(
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
                BLOCK_M,
                BLOCK_N,
                HEAD_DIM,
                STAGE,
                ENABLE_TMA,
                LOOP_SCHEDULE,
                False,  # warp_specialize on hopper is not ready yet
            )
        tile_idx += num_progs


configsCutlassBlackwell = [
    (
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_M_HALF": BMhalf,
                "BLOCK_N": BN,
                "ENABLE_TMA": enable_tma,
                "LOOP_SCHEDULE": sched,
                "GRID_MULTIPLE": mult,
                "FIRST_MMA": 1,
                "LAST_MMA": 1,
                "FIRST_SOFTMAX": 3,
                "LAST_SOFTMAX": 4,
                "LOAD_K": 2,
                "LOAD_V": 2,
                "FIRST_CORRECTION": 0,
                "LAST_CORRECTION": 0,
                "ALPHA_REMAT": True,
                "FIRST_LOADQ": 2,
                "LAST_LOADQ": 2,
            },
            num_stages=2,
            num_warps=w,
        )
    )
    for BM in [128]
    for BMhalf in [64]
    for BN in [128]
    for mult in [1]
    for sched in schedList
    for enable_tma in [True]
    for enable_ws in [True]
    for w in [4]
]


configsTKBlackwell = [  # ThunderKitten
    (
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_M_HALF": BMhalf,
                "BLOCK_N": BN,
                "ENABLE_TMA": enable_tma,
                "LOOP_SCHEDULE": sched,
                "GRID_MULTIPLE": mult,
                "FIRST_MMA": 2,
                "LAST_MMA": 3,
                "FIRST_SOFTMAX": 0,
                "LAST_SOFTMAX": 1,
                "LOAD_K": 4,
                "LOAD_V": 5,
                "FIRST_CORRECTION": 2,
                "LAST_CORRECTION": 3,
                "ALPHA_REMAT": False,
                "FIRST_LOADQ": 0,
                "LAST_LOADQ": 1,
            },
            num_stages=2,
            num_warps=w,
        )
    )
    for BM in [128]
    for BMhalf in [64]
    for BN in [128]
    for mult in [1]
    for sched in schedList
    for enable_tma in [True]
    for enable_ws in [True]
    for w in [8]
]


@triton.autotune(list(filter(keep, configsCutlassBlackwell)), key=["N_CTX"])
@triton.jit
def _attn_fwd_tma_ws_persistent_with_dp(  # Q, V, desc_k, desc_v, sm_scale, M, Out,  #
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
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
    Z,
    H,
    N_CTX,  #: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_M_HALF: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    ENABLE_TMA: tl.constexpr,
    LOOP_SCHEDULE: tl.constexpr,
    ENABLE_WS: tl.constexpr,
    GRID_MULTIPLE: tl.constexpr,
    FIRST_MMA: tl.constexpr,
    LAST_MMA: tl.constexpr,
    FIRST_SOFTMAX: tl.constexpr,
    LAST_SOFTMAX: tl.constexpr,
    LOAD_K: tl.constexpr,
    LOAD_V: tl.constexpr,
    FIRST_CORRECTION: tl.constexpr,
    LAST_CORRECTION: tl.constexpr,
    ALPHA_REMAT: tl.constexpr,
    FIRST_LOADQ: tl.constexpr,
    LAST_LOADQ: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    # original grid
    #   triton.cdiv(q.shape[2], META["BLOCK_M"]),
    #   q.shape[0] * q.shape[1],
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

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
        block_shape=[BLOCK_M_HALF, HEAD_DIM],
    )
    desc_o = tl.make_tensor_descriptor(
        Out,
        shape=[Z * H * N_CTX, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M_HALF, HEAD_DIM],
    )

    for _ in tl.range(0, tiles_per_sm, num_stages=1):
        # This has much better cache locality than
        #     pid = tile_idx // (Z * H)
        #     off_hz = tile_idx % (Z * H)  # tl.program_id(1)
        pid = tile_idx % n_tile_num
        off_hz = tile_idx // n_tile_num
        _attn_fwd_compute_ws_with_dp(
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
            BLOCK_M,
            BLOCK_M_HALF,
            BLOCK_N,
            HEAD_DIM,
            STAGE,
            ENABLE_TMA,
            LOOP_SCHEDULE,
            FIRST_MMA,
            LAST_MMA,
            FIRST_SOFTMAX,
            LAST_SOFTMAX,
            LOAD_K,
            LOAD_V,
            FIRST_CORRECTION,
            LAST_CORRECTION,
            ALPHA_REMAT,
            FIRST_LOADQ,
            LAST_LOADQ,
        )
        tile_idx += num_progs


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,
    Delta,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #  #  #  #
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    k,
    v,
    sm_scale,  #
    DO,  #
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.bfloat16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.bfloat16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,  #
    do,
    m,
    D,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,  #
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    # shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    tl.assume(stride_z >= 0)
    tl.assume(stride_h >= 0)
    tl.assume(stride_tok >= 0)
    tl.assume(stride_d >= 0)
    tl.assume(H > 0)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        MASK_BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=True,  #
    )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=False,  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        MASK_BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        end_n - num_steps * MASK_BLOCK_N2,
        num_steps,  #
        MASK=True,  #
    )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        end_n - num_steps * BLOCK_N2,
        num_steps,  #
        MASK=False,  #
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention_opt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, baseVariant):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        TMA_SIZE = 128
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

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        def grid_tma_persistent(META):
            if META["ENABLE_TMA"] is False or HAS_TMA_DESC is False:
                return (
                    min(
                        NUM_SMS * META["GRID_MULTIPLE"],
                        triton.cdiv(q.shape[2], META["BLOCK_M"])
                        * q.shape[0]
                        * q.shape[1],
                    ),
                    1,
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
                min(
                    NUM_SMS * META["GRID_MULTIPLE"],
                    triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1],
                ),
                1,
                1,
            )

        desc_q = None
        desc_k = None
        desc_v = None
        desc_o = None
        if desc_helper is not None:
            desc_q = desc_helper.get_tma_descriptor_kernel_param("q")
            desc_k = desc_helper.get_tma_descriptor_kernel_param("k")
            desc_v = desc_helper.get_tma_descriptor_kernel_param("v")
            desc_o = desc_helper.get_tma_descriptor_kernel_param("o")

        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        if HAS_NEW_TMA:
            triton.set_allocator(alloc_fn)

        if baseVariant == "base_opt":
            _attn_fwd_base_opt[grid_tma](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                desc_q,
                desc_k,
                desc_v,
                desc_o,
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
                N_CTX=q.shape[2],
                HEAD_DIM=HEAD_DIM_K,
                STAGE=stage,
                ENABLE_WS=False,
                **extra_kern_args,
            )
        elif baseVariant == "ws":
            _attn_fwd_ws[grid_tma](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[0],
                q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                STAGE=stage,  #
                ENABLE_WS=True,
                HAS_EXPLICIT_WS=HAS_EXPLICIT_WS,
                **extra_kern_args,
            )
        elif baseVariant == "tma_ws" or baseVariant == "tma":
            _attn_fwd_tma_unified[grid_tma](
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
                N_CTX=q.shape[2],
                HEAD_DIM=HEAD_DIM_K,
                STAGE=stage,
                ENABLE_WS=True if baseVariant == "tma_ws" else False,
                HAS_EXPLICIT_WS=HAS_EXPLICIT_WS,
                **extra_kern_args,
            )
        elif baseVariant == "tma_ws_persistent":
            _attn_fwd_tma_ws_persistent[grid_tma_persistent](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[0],
                q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                STAGE=stage,  #
                ENABLE_WS=True,
                HAS_EXPLICIT_WS=HAS_EXPLICIT_WS,
                **extra_kern_args,
            )
        elif baseVariant == "tma_ws_persistent_blackwell":
            _attn_fwd_tma_ws_persistent_with_dp[grid_tma_persistent](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[0],
                q.shape[1],  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                STAGE=stage,  #
                ENABLE_WS=True,
                **extra_kern_args,
            )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid_tma
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        if not ctx.causal:
            raise NotImplementedError("only causal backward is implemented on Triton")
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o,
            do,  #
            delta,  #
            BATCH,
            N_HEAD,
            N_CTX,  #
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            N_HEAD,
            N_CTX,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
        )

        return dq, dk, dv, None, None, None


attention_opt = _attention_opt.apply
