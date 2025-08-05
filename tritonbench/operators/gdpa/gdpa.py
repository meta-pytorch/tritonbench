"""
Generalized Dot Product Attention
===============

This is a variant of the Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
"""

# pyre-ignore-all-errors

import math
import os
import types
from functools import lru_cache
from typing import Tuple

import torch
import torch.nn.functional as F

import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton

from torch._library.triton import capture_triton

from .gdpa_utils import (
    custom_triton_op,
    get_autotune_kernel,
    get_num_sms,
    is_tma_supported,
)
from .math import (
    activation_string_to_int,
    fast_gelu,
    fast_gelu_grad,
    gelu,
    gelu_grad,
    # raw,
    # raw_grad,
    tanh_approx_fp32,
)

try:
    BF16_ATOMIC_ADD_SUPPORTED = torch.cuda.get_device_capability() >= (9, 0)
except RuntimeError:  # Not running on GPU device
    BF16_ATOMIC_ADD_SUPPORTED = False


try:
    from triton.fb.compat import Config as triton_config  # @manual=//triton:triton
except ModuleNotFoundError:
    # Fall back to the default config if we don't have
    # trtion.fb.compat supported
    from triton import Config as triton_config  # @manual=//triton:triton


def is_hip():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except RuntimeError:
        return False


@torch.fx.wrap
def create_dummy_tensor(x):
    return torch.ones(1, device=x.device, dtype=torch.int32)


@triton.jit
def _gdpa_fwd_inner_ws(
    acc,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    desc_k,
    desc_v,
    kv_offset,
    begin_k,
    stride_kn,
    stride_kh,
    start_m,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    offs_d: tl.constexpr,  #
    qlen,
    klen,
    fp8_v: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    enable_tma: tl.constexpr,
    v_dtype,
    activation_enum_int: tl.constexpr,
    N_CTX,
    qk_scale,  #
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, klen
    if WINDOW_SIZE is not None:
        lo = max(lo, ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N) * BLOCK_N)
        hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)

    if not enable_tma:
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        with tl.async_task([0]):
            if enable_tma:
                k = tl._experimental_descriptor_load(
                    desc_k,
                    [
                        (begin_k + start_n).to(tl.int32),
                        kv_offset.to(tl.int32),
                    ],
                    [BLOCK_N, BLOCK_D],
                    q.dtype,
                )
                k = tl.trans(k)
            else:
                k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")

        q = q.to(k.dtype)
        with tl.async_task([1, 2]):
            qk = tl.dot(q, k)
            if WINDOW_SIZE is not None:
                window_mask = (
                    tl.abs(offs_m[:, None] - (start_n + offs_n[None, :])) <= WINDOW_SIZE
                )
                qk = tl.where(window_mask, qk, 0.0)
            # activation = gelu
            if activation_enum_int == 0:
                p = raw(qk)
            elif activation_enum_int == 1:
                # activation = gelu TypeError("cannot convert JITFunction(ads_mkl.ops.triton.math:gelu) of type <class 'triton.runtime.jit.JITFunction'> to tensor")
                p = gelu(qk)
            elif activation_enum_int == 2:
                p = gelu_approx(qk)
            elif activation_enum_int == 3:
                p = fast_gelu(qk)
            elif activation_enum_int == 4:
                p = leaky_relu(qk)
            elif activation_enum_int == 5:
                p = relu(qk)
            elif activation_enum_int == 6:
                qk = qk.to(v_dtype)
                p = fast_gelu_bf16(qk)
            elif activation_enum_int == 7:
                p = silu(qk)
            elif activation_enum_int == 8:
                p = fast_silu(qk)
            else:
                p = qk

            p *= qk_scale
            p = p.to(v_dtype)

        with tl.async_task([0]):
            if enable_tma:
                v = tl._experimental_descriptor_load(
                    desc_v,
                    [
                        (begin_k + start_n).to(tl.int32),
                        kv_offset.to(tl.int32),
                    ],
                    [BLOCK_N, BLOCK_D],
                    q.dtype,
                )
            else:
                v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        with tl.async_task([1, 2]):
            p = p.to(v_dtype)
            acc = tl.dot(p, v, acc)

        if not enable_tma:
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton_config(
        {"BLOCK_M": BM, "BLOCK_N": BN, "NUM_CONSUMER_GROUPS": 1},
        num_stages=s,
        num_warps=w,
    )
    for BM in [32, 64, 128]  # [32, 64, 128, 256]
    for BN in [32, 64, 128]  # 32, 64, 128]
    for s in ([1, 2] if is_hip() else [1, 3])  # 3, 4, 7])
    for w in [4, 8]  # 4, 8]
]


configsWS = [
    (
        triton_config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "NUM_CONSUMER_GROUPS": 2},
            num_stages=s,
            num_warps=w,
            # num_buffers_warp_spec=2,
            # num_consumer_groups=2,
        )
    )
    for BM in [128]
    for BN in [64, 128]  # 32 will cause ws hang
    for s in ([1, 2] if is_hip() else [0])
    for w in [4]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


DISABLE_AUTOTUNE = os.environ.get("ADS_MKL_DISABLE_AUTOTUNE") == "1"

# Manually autotuned configs for faster autotuning
AUTOTUNE_CONFIG_SET = os.environ.get("ADS_MKL_AUTOTUNE_CONFIG_SET", "default")

if DISABLE_AUTOTUNE:
    configs = [
        triton_config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "NUM_CONSUMER_GROUPS": 1},
            num_stages=2,
            num_warps=4,
        )
    ]

fwd_autotune_configs = {
    "default": tuple(configs),
}

fwd_autotune_configs_ws = {
    "default": tuple(configsWS),
}


@lru_cache
def get_autotune_fwd_kernel(kernel, enable_ws):
    config_map = fwd_autotune_configs_ws if enable_ws else fwd_autotune_configs
    return get_autotune_kernel(
        kernel,
        config_map.get(AUTOTUNE_CONFIG_SET, config_map.get("default")),
        key=["N_CTX", "HEAD_DIM", "H", "G", "FUSED_QKV", "FUSED_KV"],
    )


@triton.jit
def _gdpa_fwd_compute(
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    workspace_ptr,
    Out,  #
    Out_offsets,
    ad_to_request_offset_ptr,
    stride_qm,
    stride_qh,
    stride_qk,  #
    stride_kn,
    stride_kh,
    stride_kk,  #
    stride_vn,
    stride_vh,
    stride_vk,  #
    stride_om,
    stride_oh,
    stride_ok,  #
    H,  # number of q heads.
    G,  # number of q head in each group. number of k v head will be H//G
    off_z,
    off_q_z,
    off_h,
    off_h_kv,
    pid,
    N_CTX,  #
    qk_scale,  #
    is_predict: tl.constexpr,  #
    Q_SHAPE_0,
    FUSED_QKV: tl.constexpr,  #
    FUSED_KV: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    enable_tma: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
    desc_k,
    desc_v,
):
    start_m = pid
    q_offset = off_h.to(tl.int64) * stride_qh
    kv_offset = off_h_kv.to(tl.int64) * stride_kh

    begin_q = tl.load(Q_offsets + off_q_z)
    end_q = tl.load(Q_offsets + off_q_z + 1)

    qlen = end_q - begin_q
    qlen = tl.minimum(qlen, N_CTX)

    if FUSED_QKV:
        begin_k = begin_q
        end_k = end_q
        klen = qlen
        K = Q + HEAD_DIM
        V = Q + HEAD_DIM * 2
    else:
        if is_predict:
            off_zkv = tl.load(ad_to_request_offset_ptr + off_z)
            begin_k = tl.load(K_offsets + off_zkv)
            end_k = tl.load(K_offsets + off_zkv + 1)
        else:
            begin_k = tl.load(K_offsets + off_z)
            end_k = tl.load(K_offsets + off_z + 1)
        klen = end_k - begin_k

    if FUSED_KV:
        V = K + HEAD_DIM

    if start_m * BLOCK_M < qlen:
        begin_o = tl.load(Out_offsets + off_z)

        # block pointers
        Q_block_ptr = None
        K_block_ptr = None
        V_block_ptr = None
        desc_q = None
        # can not reuse desc_k. jit error
        desc_k_tmp = None
        desc_v_tmp = None
        if not enable_tma:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + q_offset + begin_q * stride_qm,
                shape=(qlen, HEAD_DIM),
                strides=(stride_qm, stride_qk),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            v_order: tl.constexpr = (
                (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
            )
            V_block_ptr = tl.make_block_ptr(
                base=V + kv_offset + begin_k * stride_vn,
                shape=(klen, HEAD_DIM),
                strides=(stride_vn, stride_vk),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=v_order,
            )
            K_block_ptr = tl.make_block_ptr(
                base=K + kv_offset + begin_k * stride_kn,
                shape=(HEAD_DIM, klen),
                strides=(stride_kk, stride_kn),
                offsets=(0, 0),
                block_shape=(BLOCK_D, BLOCK_N),
                order=(0, 1),
            )
        else:
            with tl.async_task([0]):
                TMA_SIZE: tl.constexpr = 128
                workspace_base = workspace_ptr
                desc_q = workspace_base
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_q,
                    global_address=Q,
                    load_size=[BLOCK_M // NUM_CONSUMER_GROUPS, BLOCK_D],
                    global_size=[end_q.to(tl.int32), HEAD_DIM * H],
                    element_ty=Q.dtype.element_ty,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_q)
                if not IS_DENSE_KV:
                    desc_k_tmp = workspace_base + TMA_SIZE
                    desc_v_tmp = workspace_base + 2 * TMA_SIZE
                    tl.extra.cuda.experimental_device_tensormap_create2d(
                        desc_ptr=desc_k_tmp,
                        global_address=K,
                        load_size=[BLOCK_N, BLOCK_D],
                        global_size=[end_k.to(tl.int32), HEAD_DIM * H // G],
                        element_ty=K.dtype.element_ty,
                    )
                    tl.extra.cuda.experimental_device_tensormap_create2d(
                        desc_ptr=desc_v_tmp,
                        global_address=V,
                        load_size=[BLOCK_N, BLOCK_D],
                        global_size=[end_k.to(tl.int32), HEAD_DIM * H // G],
                        element_ty=V.dtype.element_ty,
                    )

                    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_k_tmp)
                    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_v_tmp)

        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        o_ptrs = (
            Out
            + off_h.to(tl.int64) * stride_oh
            + begin_o * stride_om
            + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
        )

        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        with tl.async_task([0]):
            # load q: it will stay in SRAM throughout
            if enable_tma:
                q = tl._experimental_descriptor_load(
                    desc_q,
                    [
                        (begin_q + start_m * BLOCK_M).to(tl.int32),
                        (q_offset).to(tl.int32),
                    ],
                    [BLOCK_M, BLOCK_D],
                    Q.dtype.element_ty,
                )
            else:
                q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # stage 1: off-band
        # For causal = True, STAGE = 3 and _gdpa_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _gdpa_fwd_inner gets 3 as its STAGE
        # if STAGE & 1:
        acc = _gdpa_fwd_inner_ws(
            acc,
            q,
            K_block_ptr,
            V_block_ptr,  #
            desc_k if IS_DENSE_KV else desc_k_tmp,
            desc_v if IS_DENSE_KV else desc_v_tmp,
            kv_offset,
            begin_k,
            stride_kn,
            stride_kh,
            start_m,  #
            BLOCK_M,
            BLOCK_D,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            offs_d,
            qlen,
            klen,
            V.dtype.element_ty == tl.float8e5,  #
            WINDOW_SIZE,
            enable_tma,
            V.dtype.element_ty,
            activation_enum_int,
            N_CTX,
            qk_scale,  #
        )

        # epilogue
        with tl.async_task([1, 2]):
            o_mask = (offs_m[:, None] < qlen) & (offs_d[None, :] < HEAD_DIM)
            if WINDOW_SIZE is not None:
                acc = tl.where(offs_m[:, None] < (klen + WINDOW_SIZE), acc, 0.0)

            tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=o_mask)


@triton.jit
def _gdpa_fwd(
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    workspace_ptr,
    Out,  #
    Out_offsets,
    ad_to_request_offset_ptr,
    seq_index,
    stride_qm,
    stride_qh,
    stride_qk,  #
    stride_kn,
    stride_kh,
    stride_kk,  #
    stride_vn,
    stride_vh,
    stride_vk,  #
    stride_om,
    stride_oh,
    stride_ok,  #
    Z,
    H,  # number of q heads.
    G,  # number of q head in each group. number of k v head will be H//G
    N_CTX,
    N_CTX_KV,  #
    qk_scale,  #
    is_predict: tl.constexpr,  #
    Q_SHAPE_0,
    FUSED_QKV: tl.constexpr,  #
    FUSED_KV: tl.constexpr,  #
    SORT_BY_SEQ_LENGTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    USE_START_END_OFFSETS: tl.constexpr,
    enable_tma: tl.constexpr,
    enable_ws: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
):
    off_hz = tl.program_id(1)
    if USE_START_END_OFFSETS:
        off_z = (off_hz // H) * 2
    else:
        off_z = off_hz // H
    if SORT_BY_SEQ_LENGTH:
        off_z = tl.load(seq_index + off_z)
    if BROADCAST_Q:
        off_q_z = 0
    else:
        off_q_z = off_z
    off_h = off_hz % H
    off_h_kv = off_h // G
    pid = tl.program_id(0)
    if enable_tma:
        workspace_ptr = (
            workspace_ptr
            + (tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)) * 3 * 128
        )
    else:
        workspace_ptr = None
    _gdpa_fwd_compute(
        Q,
        Q_offsets,
        K,
        K_offsets,
        V,
        workspace_ptr,
        Out,  #
        Out_offsets,
        ad_to_request_offset_ptr,
        stride_qm,
        stride_qh,
        stride_qk,  #
        stride_kn,
        stride_kh,
        stride_kk,  #
        stride_vn,
        stride_vh,
        stride_vk,  #
        stride_om,
        stride_oh,
        stride_ok,  #
        H,
        G,
        off_z,
        off_q_z,
        off_h,
        off_h_kv,
        pid,
        N_CTX,  #
        qk_scale,  #
        is_predict,
        Q_SHAPE_0,
        FUSED_QKV,
        FUSED_KV,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        STAGE,
        enable_tma,
        NUM_CONSUMER_GROUPS,
        WINDOW_SIZE,
        BROADCAST_Q,
        False,  # IS_DENSE_KV
        activation_enum_int,
        None,
        None,
    )


@triton.jit
def _gdpa_fwd_persistent(
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    workspace_ptr,
    Out,  #
    Out_offsets,
    ad_to_request_offset_ptr,
    seq_index,
    stride_qm,
    stride_qh,
    stride_qk,  #
    stride_kn,
    stride_kh,
    stride_kk,  #
    stride_vn,
    stride_vh,
    stride_vk,  #
    stride_om,
    stride_oh,
    stride_ok,  #
    Z,
    H,  # number of q heads.
    G,  # number of q head in each group. number of k v head will be H//G
    N_CTX,
    N_CTX_KV,  #
    qk_scale,  #
    is_predict: tl.constexpr,  #
    Q_SHAPE_0,
    FUSED_QKV: tl.constexpr,  #
    FUSED_KV: tl.constexpr,  #
    SORT_BY_SEQ_LENGTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    USE_START_END_OFFSETS: tl.constexpr,
    enable_tma: tl.constexpr,
    enable_ws: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
):
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    desc_k = None
    desc_v = None
    if enable_tma:
        workspace_ptr_base = workspace_ptr + prog_id * 3 * 128
        if IS_DENSE_KV:
            with tl.async_task([0]):
                TMA_SIZE: tl.constexpr = 128
                workspace_base = workspace_ptr_base
                desc_k = workspace_base + TMA_SIZE
                desc_v = workspace_base + 2 * TMA_SIZE
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_k,
                    global_address=K,
                    load_size=[BLOCK_N, BLOCK_D],
                    global_size=[N_CTX_KV * Z, HEAD_DIM * H // G],
                    element_ty=K.dtype.element_ty,
                )
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_v,
                    global_address=V,
                    load_size=[BLOCK_N, BLOCK_D],
                    global_size=[N_CTX_KV * Z, HEAD_DIM * H // G],
                    element_ty=V.dtype.element_ty,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_k)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_v)
    else:
        workspace_ptr_base = None

    for _ in range(0, tiles_per_sm):
        pid = tile_idx % n_tile_num
        off_hz = tile_idx // n_tile_num
        if USE_START_END_OFFSETS:
            off_z = (off_hz // H) * 2
        else:
            off_z = off_hz // H
        if SORT_BY_SEQ_LENGTH:
            off_z = tl.load(seq_index + off_z)
        if BROADCAST_Q:
            off_q_z = 0
        else:
            off_q_z = off_z
        off_h = off_hz % H
        off_h_kv = off_h // G

        _gdpa_fwd_compute(
            Q,
            Q_offsets,
            K,
            K_offsets,
            V,
            workspace_ptr_base,
            Out,  #
            Out_offsets,
            ad_to_request_offset_ptr,
            stride_qm,
            stride_qh,
            stride_qk,  #
            stride_kn,
            stride_kh,
            stride_kk,  #
            stride_vn,
            stride_vh,
            stride_vk,  #
            stride_om,
            stride_oh,
            stride_ok,  #
            H,
            G,
            off_z,
            off_q_z,
            off_h,
            off_h_kv,
            pid,
            N_CTX,  #
            qk_scale,  #
            is_predict,
            Q_SHAPE_0,
            FUSED_QKV,
            FUSED_KV,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            STAGE,
            enable_tma,
            NUM_CONSUMER_GROUPS,
            WINDOW_SIZE,
            BROADCAST_Q,
            IS_DENSE_KV,
            activation_enum_int,
            desc_k,
            desc_v,
        )
        tile_idx += num_progs


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _gdpa_bwd_dkdv(
    dk,
    dv,  #
    Q,
    desc_q,
    desc_do,
    desc_dq,
    k,
    v,
    DO,  #
    # shared by Q/K/V/DO.
    stride_qm,
    stride_d,  #
    stride_dom,
    stride_qh,
    H,
    N_CTX,
    qk_scale,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D: tl.constexpr,
    # Filled in by the wrapper.
    start_n,
    start_m,
    qlen,
    klen,
    num_steps,  #
    begin_q,
    offs_k,
    off_h2,
    MASK: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    DQ,
    USE_DQ_ATOMIC_ADD: tl.constexpr,
    enable_tma: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    activation_enum_int: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    qT_ptrs = None
    do_ptrs = None
    ts_encoding_bucket_values_ptr = None
    ts_event_request_even_bucket_values_ptr = None
    ts_event_request_log_bucket_values_ptr = None
    ts_adjacent_event_log_bucket_values_ptr = None
    pos_emb_bucket_values_ptr = None

    if not enable_tma:
        qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_d
        do_ptrs = DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_d

    if USE_DQ_ATOMIC_ADD:
        dq_ptrs = DQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for _blk_idx in tl.range(0, num_steps, 1, loop_unroll_factor=1):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        qmask = (offs_k[:, None] < HEAD_DIM) & (offs_m[None, :] < qlen)
        with tl.async_task([0]):
            if enable_tma:
                # qmask = (offs_k[:, None] < HEAD_DIM) & (offs_m[None, :] < qlen)
                q = tl._experimental_descriptor_load(
                    desc_q,
                    [
                        (begin_q + curr_m).to(tl.int32),
                        (off_h2 * stride_qh).to(tl.int32),
                    ],
                    [BLOCK_M1, BLOCK_D],
                    Q.dtype.element_ty,
                )
                qT = tl.trans(q)
                # we may need this line for correctness
                # qT = tl.where(qmask, qT, 0.0)
            else:
                qT = tl.load(qT_ptrs, mask=qmask)

        qT = qT.to(k.dtype)
        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
            qkT = tl.dot(k, qT)

        with tl.async_task([0]):
            # move dot ahead hoping to overlap with other computations
            omask = (offs_m[:, None] < qlen) & (offs_k[None, :] < HEAD_DIM)
            if enable_tma:
                do = tl._experimental_descriptor_load(
                    desc_do,
                    [
                        (begin_q + curr_m).to(tl.int32),
                        (off_h2 * stride_qh).to(tl.int32),
                    ],
                    [BLOCK_M1, BLOCK_D],
                    DO.dtype.element_ty,
                )
                # do = tl.where(omask, do, 0.0)
            else:
                do = tl.load(do_ptrs, mask=omask)  # Compute dP and dS.

        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
            dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
            pT = qkT
            # Autoregressive masking.
            if MASK:
                mask = offs_m[None, :] >= offs_n[:, None]
                pT = tl.where(mask, pT, 0.0)
            # Sliding window masking.
            if WINDOW_SIZE is not None:
                window_mask = tl.abs(offs_m[None, :] - offs_n[:, None]) <= WINDOW_SIZE
                pT = tl.where(window_mask, pT, 0.0)
            # Compute dV.
            if activation_enum_int == 0:
                ppT = raw(pT)
            elif activation_enum_int == 1:
                ppT = gelu(pT)
            elif activation_enum_int == 2:
                tanh_out = tanh(0.7978845608 * pT * (1 + 0.044715 * pT * pT))
                ppT = 0.5 * pT * (1 + tanh_out)
            elif activation_enum_int == 3:
                tanh_out = tanh_approx_fp32(
                    0.7978845608 * pT * (1 + 0.044715 * pT * pT)
                )
                ppT = 0.5 * pT * (1 + tanh_out)
            elif activation_enum_int == 4:
                ppT = leaky_relu(pT)
            elif activation_enum_int == 5:
                ppT = relu(pT)
            elif activation_enum_int == 6:
                pT = pT.to(Q.dtype.element_ty)
                tanh_out = tanh_approx_bf16(
                    0.7978845608 * pT * (1 + 0.044715 * pT * pT)
                )
                ppT = 0.5 * pT * (1 + tanh_out)
            elif activation_enum_int == 7:
                ppT = silu(pT)
            elif activation_enum_int == 8:
                ppT = fast_silu(pT)
            else:
                ppT = pT
            ppT *= qk_scale
            ppT = ppT.to(Q.dtype.element_ty)
            dv += tl.dot(ppT, do)

            ##
            if activation_enum_int == 0:
                pT = raw_grad(pT)
            elif activation_enum_int == 1:
                # activation = gelu TypeError("cannot convert JITFunction(ads_mkl.ops.triton.math:gelu) of type <class 'triton.runtime.jit.JITFunction'> to tensor")
                pT = gelu_grad(pT)
            elif (activation_enum_int == 2) or (activation_enum_int == 3):
                pT = (
                    0.5
                    * pT
                    * (1 - tanh_out * tanh_out)
                    * (0.7978845608 + 0.1070322243 * pT * pT)
                ) + 0.5 * (1 + tanh_out)
            elif activation_enum_int == 4:
                pT = leaky_relu_grad(pT)
            elif activation_enum_int == 5:
                pT = relu_grad(pT)
            elif activation_enum_int == 6:  # A or B or C is not supported
                pT = 0.5 * pT * (1 - tanh_out * tanh_out) * (
                    0.7978845608 + 0.1070322243 * pT * pT
                ) + 0.5 * (1 + tanh_out)
            elif activation_enum_int == 7:
                pT = silu_grad(pT)
            elif activation_enum_int == 8:
                pT = fast_silu_grad(pT)
            else:
                pT = 1
            pT *= qk_scale

            # Autoregressive masking.
            if MASK:
                mask = offs_m[None, :] >= offs_n[:, None]
                pT = tl.where(mask, pT, 0.0)
            # Sliding window masking.
            if WINDOW_SIZE is not None:
                window_mask = tl.abs(offs_m[None, :] - offs_n[:, None]) <= WINDOW_SIZE
                pT = tl.where(window_mask, pT, 0.0)
            dsT = pT * dpT
            dsT = dsT.to(Q.dtype.element_ty)
            dk += tl.dot(dsT, tl.trans(qT))
            if USE_DQ_ATOMIC_ADD:
                dq = tl.dot(tl.trans(dsT), k)
                if enable_tma:
                    tl._experimental_descriptor_store(
                        desc_dq,
                        dq.to(Q.dtype.element_ty),
                        [
                            (begin_q + curr_m).to(tl.int32),
                            (off_h2 * stride_qh).to(tl.int32),
                        ],
                        store_reduce="add",
                    )
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_k[None, :] < HEAD_DIM) & (offs_m[:, None] < qlen),
                        sem="relaxed",
                    )
        # Increment pointers.
        curr_m += step_m
        with tl.async_task([0]):
            if not enable_tma:
                qT_ptrs += step_m * stride_qm
                do_ptrs += step_m * stride_dom

        if USE_DQ_ATOMIC_ADD:
            dq_ptrs += step_m * stride_qm

    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _gdpa_bwd_dq(
    dq,
    q,
    K,
    V,  #
    desc_k2,
    desc_v2,
    do,
    # shared by Q/K/V/DO.
    stride_km,
    stride_d,  #
    stride_kh,
    H,
    N_CTX,  #
    qk_scale,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,  #
    # Filled in by the wrapper.
    start_m,
    start_n,
    qlen,
    klen,
    num_steps,  #
    begin_k,
    offs_k,
    off_h_kv,
    # ln_scale,
    MASK: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    enable_tma: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    activation_enum_int: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    kT_ptrs = None
    vT_ptrs = None
    if not enable_tma:
        kT_ptrs = K + offs_n[None, :] * stride_km + offs_k[:, None] * stride_d
        vT_ptrs = V + offs_n[None, :] * stride_km + offs_k[:, None] * stride_d

    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for _blk_idx in tl.range(0, num_steps, 1, loop_unroll_factor=1):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        with tl.async_task([0]):
            kmask = (offs_k[:, None] < HEAD_DIM) & (offs_n[None, :] < klen)
            if enable_tma:
                k = tl._experimental_descriptor_load(
                    desc_k2,
                    [
                        (begin_k + curr_n).to(tl.int32),
                        (off_h_kv * stride_kh).to(tl.int32),
                    ],
                    [BLOCK_N2, BLOCK_D],
                    K.dtype.element_ty,
                )
                kT = tl.trans(k)
                # kT = tl.where(kmask, kT, 0.0)
                v = tl._experimental_descriptor_load(
                    desc_v2,
                    [
                        (begin_k + curr_n).to(tl.int32),
                        (off_h_kv * stride_kh).to(tl.int32),
                    ],
                    [BLOCK_N2, BLOCK_D],
                    V.dtype.element_ty,
                )
                vT = tl.trans(v)
                # vT = tl.where(kmask, vT, 0.0)
            else:
                kT = tl.load(kT_ptrs, mask=kmask)
                vT = tl.load(vT_ptrs, mask=kmask)
        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
            qk = tl.dot(q, kT)
            if activation_enum_int == 0:
                p = raw_grad(qk)
            elif activation_enum_int == 1:
                # activation = gelu TypeError("cannot convert JITFunction(ads_mkl.ops.triton.math:gelu) of type <class 'triton.runtime.jit.JITFunction'> to tensor")
                p = gelu_grad(qk)
            elif activation_enum_int == 2:
                p = gelu_approx_grad(qk)
            elif activation_enum_int == 3:
                p = fast_gelu_grad(qk)
            elif activation_enum_int == 4:
                p = leaky_relu_grad(qk)
            elif activation_enum_int == 5:
                p = relu_grad(qk)
            elif activation_enum_int == 6:
                qk = qk.to(tl.bfloat16)
                p = fast_gelu_bf16_grad(qk)
            elif activation_enum_int == 7:
                p = silu_grad(qk)
            elif activation_enum_int == 8:
                p = fast_silu_grad(qk)
            else:
                p = 1
            p *= qk_scale

            # Autoregressive masking.
            if MASK:
                mask = offs_m[:, None] >= offs_n[None, :]
                p = tl.where(mask, p, 0.0)
            # Sliding window masking.
            if WINDOW_SIZE is not None:
                window_mask = tl.abs(offs_m[:, None] - offs_n[None, :]) <= WINDOW_SIZE
                p = tl.where(window_mask, p, 0.0)

            # Compute dP and dS.
            dp = tl.dot(do, vT).to(tl.float32)
            ds = p * dp  # - Di[:, None])
            ds = ds.to(K.dtype.element_ty)
            # Compute dQ.
            # NOTE: We need to de-qk_scale dq in the end, because kT was pre-scaled.
            dq += tl.dot(ds, tl.trans(kT))  # * ln_scale
        # Increment pointers.
        curr_n += step_n
        with tl.async_task([0]):
            if not enable_tma:
                kT_ptrs += step_n * stride_km
                vT_ptrs += step_n * stride_km
    return dq


bwd_configs = [
    triton_config(
        {
            "BLOCK_M1": BM1,
            "BLOCK_N1": BN1,
            "BLOCK_M2": BN1,
            "BLOCK_N2": BM1,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=s,
        num_warps=w,
    )
    for BM1 in [32, 64]
    for BN1 in [32, 64, 128]
    for s in ([1, 2] if is_hip() else [1, 3])
    for w in [4, 8]
]

bwd_configs_ws = [
    (
        triton_config(
            {
                "BLOCK_M1": BM1,
                "BLOCK_N1": BN1,
                "BLOCK_M2": BN1,
                "BLOCK_N2": BM1,
                "NUM_CONSUMER_GROUPS": 2,
            },
            num_stages=s,
            num_warps=w,
            # num_buffers_warp_spec=buf,
            # num_consumer_groups=2,
        )
    )
    for buf in [2]
    for BM1 in [64]
    for BN1 in [128]
    for s in ([1, 2] if is_hip() else [0])
    for w in [4]
]


if DISABLE_AUTOTUNE:
    bwd_configs = [
        triton_config(
            {
                "BLOCK_M1": 16,
                "BLOCK_N1": 16,
                "BLOCK_M2": 16,
                "BLOCK_N2": 16,
                "NUM_CONSUMER_GROUPS": 1,
            },
            num_stages=1,
            num_warps=1,
        )
    ]


def bwd_keep(conf):
    # static assert in bwd code
    if conf.kwargs["BLOCK_N1"] % conf.kwargs["BLOCK_M1"] != 0:
        return False
    return True


def next_power_of_2(x):
    return 2 ** (math.ceil(math.log(x, 2)))


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if x is not None and x.stride(-1) != 1:
        return x.contiguous()
    return x


bwd_autotune_configs = {
    "default": tuple(bwd_configs),
}

bwd_autotune_configs_ws = {
    "default": tuple(bwd_configs_ws),
}


@lru_cache
def get_autotune_bwd_kernel(kernel, restore_value, enable_ws):
    config_map = bwd_autotune_configs_ws if enable_ws else bwd_autotune_configs
    return get_autotune_kernel(
        kernel,
        list(
            filter(
                bwd_keep, config_map.get(AUTOTUNE_CONFIG_SET, config_map.get("default"))
            )
        ),
        key=["N_CTX", "HEAD_DIM", "H", "G", "FUSED_QKV", "FUSED_KV"],
        restore_value=list(restore_value) if restore_value is not None else None,
    )


@triton.jit
def _gdpa_bwd_compute(
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    workspace_ptr,
    DO,  #
    Out_offsets,
    DQ,
    DK,
    DV,  #
    # shared by Q/K/V/DO.
    stride_qm,
    stride_km,
    stride_qh,
    stride_kh,
    stride_d,
    stride_dom,
    stride_doh,  #
    off_z,
    off_q_z,
    off_h,
    off_h_kv,
    pid,
    H,
    G,
    N_CTX,  #
    qk_scale,  #
    FUSED_QKV: tl.constexpr,  #
    FUSED_KV: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    USE_START_END_OFFSETS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    USE_DQ_ATOMIC_ADD: tl.constexpr,
    enable_tma: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
    desc_k,
    desc_v,
    desc_k2,
    desc_v2,
):
    begin_q = tl.load(Q_offsets + off_q_z)
    end_q = tl.load(Q_offsets + off_q_z + 1)

    qlen = end_q - begin_q

    if FUSED_QKV:
        begin_k = begin_q
        end_k = end_q
        klen = qlen

        K = Q + HEAD_DIM
        V = Q + HEAD_DIM * 2
        DK = DQ + HEAD_DIM
        DV = DQ + HEAD_DIM * 2
    else:
        begin_k = tl.load(K_offsets + off_z)
        end_k = tl.load(K_offsets + off_z + 1)

        klen = end_k - begin_k

    if FUSED_KV:
        V = K + HEAD_DIM
        DV = DK + HEAD_DIM

    start_n = pid * BLOCK_N1

    start_m = pid * BLOCK_M2

    offs_k = tl.arange(0, BLOCK_D)

    desc_q = None
    desc_do = None
    desc_q2 = None
    desc_do2 = None
    desc_dq = None
    desc_dk = None
    desc_dv = None

    if enable_tma:
        with tl.async_task([0]):
            TMA_SIZE: tl.constexpr = 128
            workspace_base = workspace_ptr
            desc_q = workspace_base
            desc_q2 = workspace_base + 1 * TMA_SIZE
            desc_do = workspace_base + 2 * TMA_SIZE
            desc_do2 = workspace_base + 3 * TMA_SIZE
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=desc_q,
                global_address=Q,
                load_size=[BLOCK_M1, BLOCK_D],
                global_size=[end_q.to(tl.int32), HEAD_DIM * H],
                element_ty=Q.dtype.element_ty,
            )
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=desc_do,
                global_address=DO,
                load_size=[BLOCK_M1, BLOCK_D],
                global_size=[end_q.to(tl.int32), HEAD_DIM * H],
                element_ty=Q.dtype.element_ty,
            )
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_q)
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_k)
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_v)
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_do)

            if not USE_DQ_ATOMIC_ADD:
                desc_q2 = workspace_base + 7 * TMA_SIZE
                desc_k2 = workspace_base + 8 * TMA_SIZE
                desc_v2 = workspace_base + 9 * TMA_SIZE
                desc_do2 = workspace_base + 10 * TMA_SIZE

                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_q2,
                    global_address=Q,
                    load_size=[BLOCK_M2 // NUM_CONSUMER_GROUPS, BLOCK_D],
                    global_size=[end_q.to(tl.int32), HEAD_DIM * H],
                    element_ty=Q.dtype.element_ty,
                )
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_k2,
                    global_address=K,
                    load_size=[BLOCK_N2, BLOCK_D],
                    global_size=[end_k.to(tl.int32), HEAD_DIM * H],
                    element_ty=K.dtype.element_ty,
                )
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_v2,
                    global_address=V,
                    load_size=[BLOCK_N2, BLOCK_D],
                    global_size=[end_k.to(tl.int32), HEAD_DIM * H],
                    element_ty=V.dtype.element_ty,
                )
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_do2,
                    global_address=DO,
                    load_size=[BLOCK_M2 // NUM_CONSUMER_GROUPS, BLOCK_D],
                    global_size=[end_q.to(tl.int32), HEAD_DIM * H],
                    element_ty=Q.dtype.element_ty,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_q2)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_k2)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_v2)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_do2)

        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
            desc_dq = workspace_base + 4 * TMA_SIZE
            desc_dk = workspace_base + 5 * TMA_SIZE
            desc_dv = workspace_base + 6 * TMA_SIZE
            if USE_DQ_ATOMIC_ADD:
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_dq,
                    global_address=DQ,
                    load_size=[BLOCK_M1, BLOCK_D],
                    global_size=[end_q.to(tl.int32), HEAD_DIM * H],
                    element_ty=Q.dtype.element_ty,
                )
            else:
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_dq,
                    global_address=DQ,
                    load_size=[BLOCK_M1 // NUM_CONSUMER_GROUPS, BLOCK_D],
                    global_size=[end_q.to(tl.int32), HEAD_DIM * H],
                    element_ty=Q.dtype.element_ty,
                )
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=desc_q2,
                global_address=Q,
                load_size=[BLOCK_M2 // NUM_CONSUMER_GROUPS, BLOCK_D],
                global_size=[end_q.to(tl.int32), HEAD_DIM * H],
                element_ty=Q.dtype.element_ty,
            )
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=desc_do2,
                global_address=DO,
                load_size=[BLOCK_M2 // NUM_CONSUMER_GROUPS, BLOCK_D],
                global_size=[end_q.to(tl.int32), HEAD_DIM * H],
                element_ty=Q.dtype.element_ty,
            )
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_q)
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_do)
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_q2)
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_do2)
            if not IS_DENSE_KV:
                desc_k = workspace_base + 4 * TMA_SIZE
                desc_v = workspace_base + 5 * TMA_SIZE
                desc_k2 = workspace_base + 6 * TMA_SIZE
                desc_v2 = workspace_base + 7 * TMA_SIZE

                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_k,
                    global_address=K,
                    load_size=[BLOCK_N1 // NUM_CONSUMER_GROUPS, BLOCK_D],
                    global_size=[end_k.to(tl.int32), HEAD_DIM * H],
                    element_ty=K.dtype.element_ty,
                )
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_v,
                    global_address=V,
                    load_size=[BLOCK_N1 // NUM_CONSUMER_GROUPS, BLOCK_D],
                    global_size=[end_k.to(tl.int32), HEAD_DIM * H],
                    element_ty=V.dtype.element_ty,
                )

                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_k2,
                    global_address=K,
                    load_size=[BLOCK_N2, BLOCK_D],
                    global_size=[end_k.to(tl.int32), HEAD_DIM * H],
                    element_ty=K.dtype.element_ty,
                )
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_v2,
                    global_address=V,
                    load_size=[BLOCK_N2, BLOCK_D],
                    global_size=[end_k.to(tl.int32), HEAD_DIM * H],
                    element_ty=V.dtype.element_ty,
                )

                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_k)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_v)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_k2)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_v2)

    # if start_n > klen and start_m > qlen:
    #    return
    # Some of the ops are used for both producer and consumer, some are used by consumer
    # Try to correctly specialize the IfOp by marking all ops.
    with tl.async_task([0, 1, NUM_CONSUMER_GROUPS]):
        # invert of start_n > klen and start_m > qlen
        if start_n <= klen or start_m <= qlen:
            begin_o = tl.load(Out_offsets + off_z)

            off_h2 = off_h.to(tl.int64)
            qadj = off_h2 * stride_qh + begin_q * stride_qm
            kadj = off_h_kv * stride_kh + begin_k * stride_km
            doadj = off_h2 * stride_doh + begin_o * stride_dom

            # offset pointers for batch/head
            Q += qadj
            K += kadj
            V += kadj
            DO += doadj
            DQ += qadj
            DK += kadj
            DV += kadj

    # load qk_scales
    if start_n < klen:
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        kmask = (offs_k[None, :] < HEAD_DIM) & (offs_n[:, None] < klen)

        dv = tl.zeros([BLOCK_N1, BLOCK_D], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, BLOCK_D], dtype=tl.float32)

        # load K and V: they stay in SRAM throughout the inner loop.

        with tl.async_task([0]):
            if enable_tma:
                k = tl._experimental_descriptor_load(
                    desc_k,
                    [
                        (begin_k + start_n).to(tl.int32),
                        (off_h_kv * stride_kh).to(tl.int32),
                    ],
                    [BLOCK_N1, BLOCK_D],
                    K.dtype.element_ty,
                )
                v = tl._experimental_descriptor_load(
                    desc_v,
                    [
                        (begin_k + start_n).to(tl.int32),
                        (off_h_kv * stride_kh).to(tl.int32),
                    ],
                    [BLOCK_N1, BLOCK_D],
                    V.dtype.element_ty,
                )

            else:
                k = tl.load(
                    K + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d,
                    mask=kmask,
                )
                v = tl.load(
                    V + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d,
                    mask=kmask,
                )

        start_m_inner = 0
        num_steps = tl.cdiv((qlen - start_m_inner), BLOCK_M1)
        if WINDOW_SIZE is not None:
            start_m_inner = (
                max(start_m_inner, start_n - WINDOW_SIZE) // BLOCK_M1
            ) * BLOCK_M1
            end_m_inner = (
                tl.cdiv(min(qlen, start_n + BLOCK_N1 + WINDOW_SIZE), BLOCK_M1)
                * BLOCK_M1
            )
            num_steps = (end_m_inner - start_m_inner) // BLOCK_M1

        # Compute dK and dV for non-masked blocks.
        # Q is reloaded for each block of M.
        dk, dv = _gdpa_bwd_dkdv(  #
            dk,
            dv,  #
            Q,
            desc_q,
            desc_do,
            desc_dq,
            k,
            v,
            DO,  #
            stride_qm,
            stride_d,  #
            stride_dom,
            stride_qh,
            H,
            N_CTX,  #
            qk_scale,  #
            BLOCK_M1,
            BLOCK_N1,
            HEAD_DIM,  #
            BLOCK_D,
            start_n,
            start_m_inner,
            qlen,
            klen,
            num_steps,  #
            begin_q,
            offs_k,
            off_h.to(tl.int64),
            MASK=False,  #
            WINDOW_SIZE=WINDOW_SIZE,
            DQ=DQ,
            USE_DQ_ATOMIC_ADD=USE_DQ_ATOMIC_ADD,
            enable_tma=enable_tma,
            NUM_CONSUMER_GROUPS=NUM_CONSUMER_GROUPS,
            activation_enum_int=activation_enum_int,
        )

        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
            # rematerialize
            dv_ptrs = DV + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d
            if USE_START_END_OFFSETS or G > 1:
                tl.atomic_add(dv_ptrs, dv, mask=kmask, sem="relaxed")
            else:
                if enable_tma:
                    tl._experimental_descriptor_store(
                        desc_dv,
                        dv.to(K.dtype.element_ty),
                        [
                            (begin_k + start_n).to(tl.int32),
                            (off_h_kv * stride_kh).to(tl.int32),
                        ],
                    )
                else:
                    tl.store(dv_ptrs, dv, mask=kmask)

            # Write back dK.
            dk_ptrs = DK + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d
            if USE_START_END_OFFSETS or G > 1:
                tl.atomic_add(dk_ptrs, dk, mask=kmask, sem="relaxed")
            else:
                if enable_tma:
                    tl._experimental_descriptor_store(
                        desc_dk,
                        dk.to(K.dtype.element_ty),
                        [
                            (begin_k + start_n).to(tl.int32),
                            (off_h_kv * stride_kh).to(tl.int32),
                        ],
                    )
                else:
                    tl.store(dk_ptrs, dk, mask=kmask)

    # THIS BLOCK DOES DQ:
    # use nested if to avoid jit error
    if not USE_DQ_ATOMIC_ADD:
        if start_m < qlen:
            end_n = klen
            offs_m = start_m + tl.arange(0, BLOCK_M2)
            off_h2 = off_h.to(tl.int64)
            qmask = (offs_k[None, :] < HEAD_DIM) & (offs_m[:, None] < qlen)

            with tl.async_task([0]):
                if enable_tma:
                    q = tl._experimental_descriptor_load(
                        desc_q2,
                        [
                            (begin_q + start_m).to(tl.int32),
                            (off_h2 * stride_qh).to(tl.int32),
                        ],
                        [BLOCK_M2, BLOCK_D],
                        Q.dtype.element_ty,
                    )
                    do = tl._experimental_descriptor_load(
                        desc_do2,
                        [
                            (begin_q + start_m).to(tl.int32),
                            (off_h2 * stride_doh).to(tl.int32),
                        ],
                        [BLOCK_M2, BLOCK_D],
                        DO.dtype.element_ty,
                    )
                else:
                    q = tl.load(
                        Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_d,
                        mask=qmask,
                    )
                    do = tl.load(
                        DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_d,
                        mask=qmask,
                    )
            dq = tl.zeros([BLOCK_M2, BLOCK_D], dtype=tl.float32)

            start_n_inner = 0
            num_steps = tl.cdiv(end_n, BLOCK_N2)
            if WINDOW_SIZE is not None:
                start_n_inner = (
                    max(start_n_inner, start_m - WINDOW_SIZE) // BLOCK_N2
                ) * BLOCK_N2
                end_n_inner = (
                    tl.cdiv(min(klen, start_m + BLOCK_M2 + WINDOW_SIZE), BLOCK_N2)
                    * BLOCK_N2
                )
                num_steps = (end_n_inner - start_n_inner) // BLOCK_N2

            dq = _gdpa_bwd_dq(
                dq,
                q,
                K,
                V,  #
                desc_k2,
                desc_v2,
                do,
                stride_km,
                stride_d,  #
                stride_kh,
                H,
                N_CTX,  #
                qk_scale,  #
                BLOCK_M2,
                BLOCK_N2,
                HEAD_DIM,  #
                BLOCK_D,
                start_m,
                start_n_inner,
                qlen,
                klen,
                num_steps,  #
                begin_k,
                offs_k,
                off_h_kv,
                MASK=False,  #
                WINDOW_SIZE=WINDOW_SIZE,
                enable_tma=enable_tma,
                NUM_CONSUMER_GROUPS=NUM_CONSUMER_GROUPS,
                activation_enum_int=activation_enum_int,
            )

            with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                # Write back dQ.
                dq_ptrs = DQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_d
                # dq *= LN2
                if BROADCAST_Q or USE_START_END_OFFSETS:
                    tl.atomic_add(dq_ptrs, dq, mask=qmask, sem="relaxed")
                else:
                    if enable_tma:
                        tl._experimental_descriptor_store(
                            desc_dq,
                            dq.to(Q.dtype.element_ty),
                            [
                                (begin_q + start_m).to(tl.int32),
                                (off_h2 * stride_qh).to(tl.int32),
                            ],
                            store_reduce="add",
                        )
                    else:
                        tl.store(dq_ptrs, dq, mask=qmask)


@triton.jit
def _gdpa_bwd(
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    workspace_ptr,
    seq_index,  #
    DO,  #
    Out_offsets,
    DQ,
    DK,
    DV,  #
    # shared by Q/K/V/DO.
    stride_qm,
    stride_km,
    stride_qh,
    stride_kh,
    stride_d,
    stride_dom,
    stride_doh,
    Z,
    H,
    G,
    N_CTX,  #
    N_CTX_KV,  #
    qk_scale,  #
    FUSED_QKV: tl.constexpr,  #
    FUSED_KV: tl.constexpr,  #
    SORT_BY_SEQ_LENGTH: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    USE_START_END_OFFSETS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    USE_DQ_ATOMIC_ADD: tl.constexpr,
    enable_tma: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
):
    if USE_START_END_OFFSETS:
        off_z = tl.program_id(2) * 2
    else:
        off_z = tl.program_id(2)
    if SORT_BY_SEQ_LENGTH:
        off_z = tl.load(seq_index + off_z)
    if BROADCAST_Q:
        off_q_z = 0
    else:
        off_q_z = off_z

    off_seq_h = tl.program_id(0)
    off_h = off_seq_h % H
    off_h_kv = off_h // G
    pid = off_seq_h // H

    if enable_tma:
        if USE_DQ_ATOMIC_ADD:
            num_desc = 7
        else:
            num_desc = 11
        workspace_ptr = (
            workspace_ptr
            + (tl.program_id(0) + tl.program_id(2) * tl.num_programs(0))
            * num_desc
            * 128
        )
    else:
        workspace_ptr = None

    _gdpa_bwd_compute(
        Q,
        Q_offsets,
        K,
        K_offsets,
        V,
        workspace_ptr,
        DO,  #
        Out_offsets,
        DQ,
        DK,
        DV,  #
        # shared by Q/K/V/DO.
        stride_qm,
        stride_km,
        stride_qh,
        stride_kh,
        stride_d,
        stride_dom,
        stride_doh,
        off_z,
        off_q_z,
        off_h,
        off_h_kv,
        pid,
        H,
        G,
        N_CTX,  #
        qk_scale,  #
        FUSED_QKV,  #
        FUSED_KV,  #
        BLOCK_D,  #
        HEAD_DIM,
        BLOCK_M1,  #
        BLOCK_N1,  #
        BLOCK_M2,  #
        BLOCK_N2,  #
        USE_START_END_OFFSETS,
        WINDOW_SIZE,
        BROADCAST_Q,
        USE_DQ_ATOMIC_ADD,
        enable_tma,
        NUM_CONSUMER_GROUPS,
        False,  # For non persistent kernel, does not matter where we create the descriptor
        activation_enum_int,
        None,
        None,
        None,
        None,
    )


@triton.jit
def _gdpa_bwd_persistent(
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    workspace,
    seq_index,  #
    DO,  #
    Out_offsets,
    DQ,
    DK,
    DV,  #
    stride_qm,
    stride_km,
    stride_qh,
    stride_kh,
    stride_d,
    stride_dom,
    stride_doh,
    Z,
    H,
    G,
    N_CTX,  #
    N_CTX_KV,  #
    qk_scale,  #
    FUSED_QKV: tl.constexpr,  #
    FUSED_KV: tl.constexpr,  #
    SORT_BY_SEQ_LENGTH: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    USE_START_END_OFFSETS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    USE_DQ_ATOMIC_ADD: tl.constexpr,
    enable_tma: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
):
    n_tile_num = max(tl.cdiv(N_CTX_KV, BLOCK_N1), tl.cdiv(N_CTX, BLOCK_M2))
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    workspace_ptr_base = None
    desc_k = None
    desc_v = None
    desc_k2 = None
    desc_v2 = None
    if enable_tma:
        workspace_ptr_base = workspace + prog_id * 8 * 128
        if IS_DENSE_KV:
            with tl.async_task([0]):
                TMA_SIZE: tl.constexpr = 128
                workspace_base = workspace_ptr_base
                desc_k = workspace_base + 4 * TMA_SIZE
                desc_v = workspace_base + 5 * TMA_SIZE
                desc_k2 = workspace_base + 6 * TMA_SIZE
                desc_v2 = workspace_base + 7 * TMA_SIZE
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_k,
                    global_address=K,
                    load_size=[BLOCK_N1 // NUM_CONSUMER_GROUPS, BLOCK_D],
                    global_size=[N_CTX_KV * Z, HEAD_DIM * H],
                    element_ty=K.dtype.element_ty,
                )
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_v,
                    global_address=V,
                    load_size=[BLOCK_N1 // NUM_CONSUMER_GROUPS, BLOCK_D],
                    global_size=[N_CTX_KV * Z, HEAD_DIM * H],
                    element_ty=V.dtype.element_ty,
                )
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_k2,
                    global_address=K,
                    load_size=[BLOCK_N2, BLOCK_D],
                    global_size=[N_CTX_KV * Z, HEAD_DIM * H],
                    element_ty=K.dtype.element_ty,
                )
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=desc_v2,
                    global_address=V,
                    load_size=[BLOCK_N2, BLOCK_D],
                    global_size=[N_CTX_KV * Z, HEAD_DIM * H],
                    element_ty=V.dtype.element_ty,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_k)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_v)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_k2)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_v2)

    for _ in tl.range(0, tiles_per_sm, 1, num_stages=1):
        pid = tile_idx % n_tile_num
        off_hz = tile_idx // n_tile_num
        if USE_START_END_OFFSETS:
            off_z = (off_hz // H) * 2
        else:
            off_z = off_hz // H
        if SORT_BY_SEQ_LENGTH:
            off_z = tl.load(seq_index + off_z)
        off_h = off_hz % H
        off_h_kv = off_h // G
        if BROADCAST_Q:
            off_q_z = 0
        else:
            off_q_z = off_z
        _gdpa_bwd_compute(
            Q,
            Q_offsets,
            K,
            K_offsets,
            V,
            workspace_ptr_base,
            DO,  #
            Out_offsets,
            DQ,
            DK,
            DV,  #
            # shared by Q/K/V/DO.
            stride_qm,
            stride_km,
            stride_qh,
            stride_kh,
            stride_d,
            stride_dom,
            stride_doh,  #
            off_z,
            off_q_z,
            off_h,
            off_h_kv,
            pid,
            H,
            G,
            N_CTX,  #
            qk_scale,  #
            FUSED_QKV,  #
            FUSED_KV,  #
            BLOCK_D,  #
            HEAD_DIM,
            BLOCK_M1,  #
            BLOCK_N1,  #
            BLOCK_M2,  #
            BLOCK_N2,  #
            USE_START_END_OFFSETS,
            WINDOW_SIZE,
            BROADCAST_Q,
            USE_DQ_ATOMIC_ADD,
            enable_tma,
            NUM_CONSUMER_GROUPS,
            IS_DENSE_KV,
            activation_enum_int,
            desc_k,
            desc_v,
            desc_k2,
            desc_v2,
        )
        tile_idx += num_progs


@custom_triton_op("ads_mkl::generalized_dot_product_attention", mutates_args=())
def generalized_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,  #
    max_seq_len_kv: int,  #
    ad_to_request_offset: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    attn_offset: torch.Tensor | None = None,  #
    is_causal: bool = False,
    qk_scale: float | None = None,
    seq_index: torch.Tensor | None = None,
    allow_tf32: bool = True,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    activation: str = "raw",
    enable_persistent: bool = False,
    enable_tma: bool = False,
    enable_ws: bool = False,
    use_dq_atomic_add: bool = False,
    total_num_objects: int | None = None,
    bwd_opt_tech: str = "base",
) -> torch.Tensor:
    assert attn_mask is None, "attn_mask/bias unsupported"
    assert not is_causal, "Causal unsupported"
    fused_qkv = key is None and value is None
    fused_kv = key is not None and value is None
    if use_start_end_offsets:
        assert (
            not fused_qkv and not fused_kv and not broadcast_q
        ), "fused_qkv/fused_kv/broadcast_q not supported with start/end offsets"
        assert total_num_objects is not None, "total_num_objects must be provided"

    if qk_scale is None:
        qk_scale = 1.0

    # shape constraints
    if fused_qkv:
        HEAD_DIM_Q = query.shape[-1] // 3
        HEAD_DIM_K = HEAD_DIM_V = HEAD_DIM_Q
    elif fused_kv:
        HEAD_DIM_Q = query.shape[-1]
        HEAD_DIM_K = key.shape[-1] // 2
        HEAD_DIM_V = HEAD_DIM_K
    else:
        HEAD_DIM_Q = query.shape[-1]
        HEAD_DIM_K = key.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

    sort_by_seq_length = seq_index is not None

    if output_offset is None:
        output_offset = query_offset

    # check whether kv is dense tensor
    bs = key_offset.size(0) - 1
    L, _, _ = key.shape
    is_dense_kv = bs * max_seq_len_kv == L

    BLOCK_D = max(next_power_of_2(HEAD_DIM_Q), 16)
    if broadcast_q:
        BATCH = key_offset.size(0) - 1
    else:
        BATCH = (
            query_offset.size(0) // 2
            if use_start_end_offsets
            else query_offset.size(0) - 1
        )

    if use_start_end_offsets:
        o = torch.empty(
            (
                total_num_objects,
                query.shape[1],
                HEAD_DIM_Q,
            ),
            device=query.device,
            dtype=query.dtype,
        )
    else:
        o = torch.empty(
            (
                BATCH * query.shape[0] if broadcast_q else query.shape[0],
                query.shape[1],
                HEAD_DIM_Q,
            ),
            device=query.device,
            dtype=query.dtype,
        )

    stage = 1  # When supporting causal, change to 3

    extra_kern_args = {}
    # Tuning for AMD target
    if is_hip():
        waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

    nheads = query.shape[1]
    if fused_qkv:
        # fused_qkv only supports q/k/v have the same shape
        G = 1
    else:
        G = query.shape[1] // key.shape[1]
        assert query.shape[1] % key.shape[1] == 0

    batch_size = BATCH * nheads
    grid = lambda args: (
        triton.cdiv(max_seq_len_q, args["BLOCK_M"]),
        batch_size,
        1,
    )

    def grid_tma(META):
        return (
            triton.cdiv(max_seq_len_q, META["BLOCK_M"]),
            BATCH * nheads,
            1,
        )

    NUM_SMS = (
        get_num_sms() or 1000000
    ) * 8  # if num sms is None, use a large number so that it is a no-op

    def grid_tma_persistent(META):
        return (
            min(NUM_SMS, triton.cdiv(max_seq_len_q, META["BLOCK_M"]) * BATCH * nheads),
            1,
            1,
        )

    q = expect_contiguous(query)
    k = expect_contiguous(key)
    v = expect_contiguous(value)

    if fused_qkv:
        kstrides = q.stride(0), q.stride(1), q.stride(2)
        vstrides = q.stride(0), q.stride(1), q.stride(2)
    elif fused_kv:
        kstrides = k.stride(0), k.stride(1), k.stride(2)
        vstrides = k.stride(0), k.stride(1), k.stride(2)
    else:
        kstrides = k.stride()
        vstrides = v.stride()

    tma_size = 128
    workspace = None
    # TODO: support non-power of 2 D (has some overhead), shared seq, & fused KV.
    enable_tma &= (
        is_tma_supported()
        and BLOCK_D == HEAD_DIM_Q
        and not (fused_qkv or fused_kv)
        and not use_start_end_offsets
    )
    # enable_ws &= enable_tma

    if enable_tma:
        num_tiles = (
            NUM_SMS
            if enable_persistent
            else triton.cdiv(max_seq_len_q, 32) * batch_size
        )
        workspace = torch.empty(
            num_tiles * 3 * tma_size,
            dtype=torch.uint8,
            device="cuda",
        )
    if enable_persistent:
        kernel_fn = _gdpa_fwd_persistent
        grid = grid_tma_persistent
    else:
        kernel_fn = _gdpa_fwd
        grid = grid_tma

    kernel_fn = get_autotune_fwd_kernel(kernel_fn, enable_ws)
    is_predict = True
    if ad_to_request_offset is None:
        # ad_to_request_offset is only used in inference,
        # use a dummy tensor in training to avoid empty address in kernel
        is_predict = False
        ad_to_request_offset = create_dummy_tensor(query)

    activation_enum_int = activation_string_to_int(activation)
    print("activation_enum_int", activation, activation_enum_int)
    kernel_info = capture_triton(kernel_fn)[grid](
        q,
        query_offset,
        k,
        key_offset,
        v,
        workspace,
        o,  #
        output_offset,
        ad_to_request_offset,
        seq_index,
        q.stride(0),
        q.stride(1),
        q.stride(2),  #
        kstrides[0],
        kstrides[1],
        kstrides[2],  #
        vstrides[0],
        vstrides[1],
        vstrides[2],  #
        o.stride(0),
        o.stride(1),
        o.stride(2),  #
        BATCH,
        nheads,  #
        G,
        N_CTX=max_seq_len_q,
        N_CTX_KV=max_seq_len_kv,  #
        qk_scale=qk_scale,
        is_predict=is_predict,
        Q_SHAPE_0=query.shape[0],
        FUSED_QKV=fused_qkv,
        FUSED_KV=fused_kv,
        SORT_BY_SEQ_LENGTH=sort_by_seq_length,
        HEAD_DIM=HEAD_DIM_K,  #
        BLOCK_D=BLOCK_D,
        STAGE=stage,  #
        USE_START_END_OFFSETS=use_start_end_offsets,
        WINDOW_SIZE=window_size,
        BROADCAST_Q=broadcast_q,
        enable_tma=enable_tma,
        enable_ws=enable_ws,
        IS_DENSE_KV=is_dense_kv,
        activation_enum_int=activation_enum_int,
        **extra_kern_args,
    )
    return o


def _generalized_dot_product_attention_setup_context(ctx, inputs, output):
    (
        query,
        key,
        value,
        query_offset,
        key_offset,
        max_seq_len_q,  #
        max_seq_len_kv,  #
        _,
        attn_mask,
        attn_offset,  #
        is_causal,
        qk_scale,
        seq_index,
        _,
        output_offset,
        use_start_end_offsets,
        window_size,
        broadcast_q,
        activation,
        enable_persistent,
        enable_tma,
        enable_ws,
        use_dq_atomic_add,
        _,
        bwd_opt_tech,
    ) = inputs
    o = output
    fused_qkv = key is None and value is None
    fused_kv = key is not None and value is None
    if fused_qkv:
        HEAD_DIM_K = query.shape[-1] // 3
    elif fused_kv:
        HEAD_DIM_K = key.shape[-1] // 2
    else:
        HEAD_DIM_K = key.shape[-1]
    BLOCK_D = max(next_power_of_2(HEAD_DIM_K), 32)

    if broadcast_q:
        BATCH = key_offset.size(0) - 1
    else:
        BATCH = (
            query_offset.size(0) // 2
            if use_start_end_offsets
            else query_offset.size(0) - 1
        )
    nheads = query.shape[1]
    sort_by_seq_length = seq_index is not None
    if fused_qkv:
        # fused qkv only support q/k/v have the same shape
        nheads_k = nheads
    else:
        nheads_k = key.shape[1]
    assert (nheads % nheads_k) == 0

    if output_offset is None:
        output_offset = query_offset

    ctx.save_for_backward(
        query,
        key,
        value,
        o,
        query_offset,
        key_offset,
        output_offset,
        seq_index,
    )
    ctx.HEAD_DIM = HEAD_DIM_K
    ctx.BLOCK_D = BLOCK_D
    ctx.causal = is_causal
    ctx.N_CTX = max_seq_len_q
    ctx.N_CTX_KV = max_seq_len_kv
    ctx.BATCH = BATCH
    ctx.N_HEAD = nheads
    ctx.G = nheads // nheads_k
    ctx.fused_qkv = fused_qkv
    ctx.fused_kv = fused_kv
    ctx.use_start_end_offsets = use_start_end_offsets
    ctx.window_size = window_size
    ctx.broadcast_q = broadcast_q
    # ctx only support int type
    ctx.activation_enum_int = activation_string_to_int(activation)
    ctx.enable_persistent = enable_persistent and "persistent" in bwd_opt_tech
    ctx.enable_tma = enable_tma and "tma" in bwd_opt_tech
    ctx.enable_ws = enable_ws and "ws" in bwd_opt_tech
    ctx.use_dq_atomic_add = use_dq_atomic_add and "dq_atomic_add" in bwd_opt_tech
    ctx.qk_scale = qk_scale
    ctx.sort_by_seq_length = sort_by_seq_length


@custom_triton_op(
    "ads_mkl::generalized_dot_product_attention_backward", mutates_args=()
)
def generalized_dot_product_attention_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    BATCH: int,
    N_HEAD: int,
    Q_GROUP: int,
    N_CTX: int,
    N_CTX_KV: int,
    BLOCK_D: int,
    HEAD_DIM: int,
    fused_qkv: bool = False,
    fused_kv: bool = False,
    qk_scale: float | None = None,
    sort_by_seq_length: bool = False,
    seq_index: torch.Tensor | None = None,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    use_dq_atomic_add: bool = False,
    activation_enum_int: int = 0,
    enable_persistent: bool = False,
    enable_tma: bool = False,
    enable_ws: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if qk_scale is None:
        qk_scale = 1.0
    dk_dv_init = torch.zeros if use_start_end_offsets or Q_GROUP > 1 else torch.empty
    dq_init = (
        torch.zeros
        if use_dq_atomic_add or use_start_end_offsets or broadcast_q
        else torch.empty
    )
    if broadcast_q:
        dq = dq_init(q.shape, device=q.device, dtype=torch.float32)
    else:
        dq = dq_init(q.shape, device=q.device, dtype=q.dtype)

    enable_tma &= (
        is_tma_supported() and not (fused_qkv or fused_kv) and not use_start_end_offsets
    )
    # enable_ws &= enable_tma

    if use_start_end_offsets and fused_qkv:
        raise ValueError("fused_qkv not supported with use_start_end_offsets")

    dk_dv_dtype, orig_dtype = None, None
    if not fused_qkv:  # k is None with fused_qkv
        dk_dv_dtype = k.dtype
        orig_dtype = k.dtype

    # A100 doesn't support bf16 atomic adds. See: https://fb.workplace.com/groups/385893200869952/posts/541649245294346/?comment_id=541651068627497&reply_comment_id=541661001959837
    if (
        not BF16_ATOMIC_ADD_SUPPORTED
        and dk_dv_dtype == torch.bfloat16
        and use_start_end_offsets
    ):
        dk_dv_dtype = torch.float32

    if fused_qkv:
        dk, dv = None, None
    elif fused_kv:
        dk = dk_dv_init(k.shape, device=k.device, dtype=dk_dv_dtype)
        dv = None
    else:
        dk = dk_dv_init(k.shape, device=k.device, dtype=dk_dv_dtype)
        dv = dk_dv_init(v.shape, device=v.device, dtype=dk_dv_dtype)

    do = expect_contiguous(do)

    if output_offset is None:
        output_offset = q_offsets

    NUM_SMS = (
        get_num_sms() or 1000000
    ) * 8  # if num sms is None, use a large number so that it is a no-op

    grid_persistent = lambda args: (
        min(
            NUM_SMS,
            max(
                triton.cdiv(N_CTX_KV, args["BLOCK_N1"]),
                triton.cdiv(N_CTX, args["BLOCK_M2"]),
            )
            * BATCH
            * N_HEAD,
        ),
        1,
        1,
    )

    grid = lambda args: (
        N_HEAD
        * max(
            triton.cdiv(N_CTX_KV, args["BLOCK_N1"]),
            triton.cdiv(N_CTX, args["BLOCK_M2"]),
        ),
        1,
        BATCH,
    )

    restore_value = ()
    if use_start_end_offsets:
        if fused_kv:
            restore_value = ("DK",)
        else:
            restore_value = (
                "DK",
                "Dv",
                "DQ",
            )

    if len(restore_value) == 0:
        restore_value = None

    tma_size = 128
    workspace = None
    if enable_tma:
        num_tiles = (
            NUM_SMS
            if enable_persistent
            else max(
                triton.cdiv(N_CTX_KV, 32),
                triton.cdiv(N_CTX, 32),
            )
            * BATCH
            * N_HEAD
        )
        num_desc = 7 if use_dq_atomic_add else 11
        workspace = torch.empty(
            num_tiles * num_desc * tma_size,
            dtype=torch.uint8,
            device="cuda",
        )
    kernel_grid = None

    if enable_persistent:
        kernel_fn = _gdpa_bwd_persistent
        kernel_grid = grid_persistent
    else:
        kernel_fn = _gdpa_bwd
        kernel_grid = grid
    # check whether kv is dense tensor
    bs = k_offsets.size(0) - 1
    L, _, _ = k.shape
    is_dense_kv = bs * N_CTX_KV == L
    kernel_fn = get_autotune_bwd_kernel(kernel_fn, restore_value, enable_ws)

    kernel_info = capture_triton(kernel_fn)[kernel_grid](
        q,
        q_offsets,
        k,
        k_offsets,
        v,
        workspace,
        seq_index,  #
        do,
        output_offset,
        dq,
        dk,
        dv,  #
        q.stride(0),
        q.stride(0) if fused_qkv else k.stride(0),
        q.stride(1),
        q.stride(1) if fused_qkv else k.stride(1),
        q.stride(2),
        do.stride(0),
        do.stride(1),
        BATCH,
        N_HEAD,
        Q_GROUP,  #
        N_CTX,
        N_CTX_KV,  #
        qk_scale,  #
        FUSED_QKV=fused_qkv,
        FUSED_KV=fused_kv,
        SORT_BY_SEQ_LENGTH=sort_by_seq_length,  #
        BLOCK_D=BLOCK_D,
        HEAD_DIM=HEAD_DIM,  #
        USE_START_END_OFFSETS=use_start_end_offsets,
        WINDOW_SIZE=window_size,
        BROADCAST_Q=broadcast_q,
        USE_DQ_ATOMIC_ADD=use_dq_atomic_add,
        enable_tma=enable_tma,
        IS_DENSE_KV=is_dense_kv,
        activation_enum_int=activation_enum_int,
    )

    if (
        not BF16_ATOMIC_ADD_SUPPORTED
        and dk_dv_dtype == torch.float32
        and orig_dtype == torch.bfloat16
    ):
        dk = dk.to(orig_dtype)
        dv = dv.to(orig_dtype)

    return (
        dq,
        dk,
        dv,
    )


def _generalized_dot_product_attention_backward(ctx, do):
    (
        q,
        k,
        v,
        o,
        q_offsets,
        k_offsets,
        output_offset,
        seq_index,
    ) = ctx.saved_tensors
    (
        dq,
        dk,
        dv,
    ) = generalized_dot_product_attention_backward(
        do,
        q,
        k,
        v,
        o,
        q_offsets,
        k_offsets,
        ctx.BATCH,
        ctx.N_HEAD,
        ctx.G,
        ctx.N_CTX,
        ctx.N_CTX_KV,
        ctx.BLOCK_D,
        ctx.HEAD_DIM,
        ctx.fused_qkv,
        ctx.fused_kv,
        ctx.qk_scale,
        ctx.sort_by_seq_length,
        seq_index,
        output_offset,
        ctx.use_start_end_offsets,
        ctx.window_size,
        ctx.broadcast_q,
        use_dq_atomic_add=ctx.use_dq_atomic_add,
        activation_enum_int=ctx.activation_enum_int,
        enable_persistent=ctx.enable_persistent,
        enable_tma=ctx.enable_tma,
        enable_ws=ctx.enable_ws,
    )
    return (
        dq,
        dk,
        dv,
        *((None,) * 22),
    )


if not isinstance(
    generalized_dot_product_attention, types.FunctionType
):  # In case of duplicate registration, `@custom_triton_op` returns the base function
    generalized_dot_product_attention.register_autograd(
        _generalized_dot_product_attention_backward,
        setup_context=_generalized_dot_product_attention_setup_context,
    )


@torch.fx.wrap
def gdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    ad_to_request_offset: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    attn_offset: torch.Tensor | None = None,
    is_causal: bool = False,
    qk_scale: float | None = None,
    seq_index: torch.Tensor | None = None,
    allow_tf32: bool = True,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    activation: str = "raw",
    enable_persistent: bool = False,
    enable_tma: bool = False,
    enable_ws: bool = False,
    use_dq_atomic_add: bool = False,
    total_num_objects: int | None = None,
    bwd_opt_tech: str = "base",
) -> torch.Tensor:
    """
    Perform Flash attention for jagged tensors. Typically offsets are of the
    format [0, start1, start2, start3, ...] where start(i+1)=end(i). To support paradigms like
    shared sequence, we support `use_start_end_offsets` which would expect offsets array like [start0, end0, start1, end1, ...],
    thus supporting overlapping offsets. Currently query offsets need to be non-overlapping as output
    shape is same as query shape.
    """
    return generalized_dot_product_attention(
        query=query,
        key=key,
        value=value,
        query_offset=query_offset,
        key_offset=key_offset,
        ad_to_request_offset=ad_to_request_offset,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_kv=max_seq_len_kv,
        attn_mask=attn_mask,
        attn_offset=attn_offset,
        is_causal=is_causal,
        qk_scale=qk_scale,
        seq_index=seq_index,
        allow_tf32=allow_tf32,
        output_offset=output_offset,
        use_start_end_offsets=use_start_end_offsets,
        window_size=window_size,
        broadcast_q=broadcast_q,
        activation=activation,
        enable_persistent=enable_persistent,
        enable_tma=enable_tma,
        enable_ws=enable_ws,
        use_dq_atomic_add=use_dq_atomic_add,
        total_num_objects=total_num_objects,
        bwd_opt_tech=bwd_opt_tech,
    )
