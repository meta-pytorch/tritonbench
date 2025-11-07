"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from tritonbench.utils.env_utils import is_tile_enabled

from .attention_utils import WITH_MAXNREG

from .blackwell_attention_utils import (
    is_blackwell,
    is_cuda,
    is_hip,
    is_hopper,
    supports_host_descriptor,
)


# Check if Triton version supports minRegAutoWS and maxRegAutoWS
# These parameters are only available in triton/tree/ws-3.5
def _supports_reg_auto_ws():
    """Check if the current Triton version supports minRegAutoWS/maxRegAutoWS"""
    try:
        # Try to create a Config with minRegAutoWS to test support
        test_config = triton.Config({}, minRegAutoWS=24, maxRegAutoWS=152)
        return True
    except (TypeError, AttributeError):
        # Parameter not supported in this Triton version
        return False


HAS_REG_AUTO_WS = _supports_reg_auto_ws()


@triton.jit
def _attn_fwd_subtile(
    q,
    k,
    offs_m,
    start_n,
    offs_n,
    qk_scale,
    l_i0,
    l_i1,  # used when FADD2_REDUCE is true
    m_i,
    acc,
    v,
    dtype: tl.constexpr,
    STAGE: tl.constexpr,
    SUBTILING: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    qk = tl.dot(q, k)
    if STAGE == 2:
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
    else:
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        if VECT_MUL & 2:
            qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
        else:
            qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)
    # -- compute correction factor
    alpha = tl.math.exp2(m_i - m_ij)
    if not FADD2_REDUCE:
        l_ij = tl.sum(p, 1)

    # -- update output accumulator --
    BM: tl.constexpr = acc.shape[0]
    BN: tl.constexpr = acc.shape[1]

    if SUBTILING:
        acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
        if VECT_MUL & 1:
            acc0 = _mul_f32x2(acc0, alpha[:, None])
            acc1 = _mul_f32x2(acc1, alpha[:, None])
        else:
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
        acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
    else:
        acc = acc * alpha[:, None]

    PM: tl.constexpr = p.shape[0]
    PN: tl.constexpr = p.shape[1]
    if FADD2_REDUCE:
        p0, p1 = p.reshape([PM, 2, PN // 2]).permute(0, 2, 1).split()
        l_ij0, l_ij1 = tl.reduce((p0, p1), axis=1, combine_fn=_reduce_fadd2)
        l_i0 = l_i0 * alpha + l_ij0
        l_i1 = l_i1 * alpha + l_ij1

    # prepare p and v for the dot
    p = p.to(dtype)
    # note that this non transposed v for FP8 is only supported on Blackwell
    acc = tl.dot(p, v, acc)
    # update m_i and l_i
    # place this at the end of the loop to reduce register pressure
    if not FADD2_REDUCE:
        l_i0 = l_i0 * alpha + l_ij
    m_i = m_ij

    return l_i0, l_i1, m_i, acc


@triton.jit
def _attn_fwd_inner_oss_dp(
    acc0,
    l_i0,
    l_i0_1,
    m_i0,
    q0,
    desc_k,
    desc_v,  #
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m0: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    SUBTILING: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
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
    offsetkv_y = offset_y + lo

    # loop over k, v and update accumulator
    for start_n in tl.range(
        lo,
        hi,
        BLOCK_N,
        warp_specialize=warp_specialize,
        disallow_acc_multi_buffer=True,
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = desc_k.load([offsetkv_y, 0]).T
        v = desc_v.load([offsetkv_y, 0])

        l_i0, l_i0_1, m_i0, acc0 = _attn_fwd_subtile(
            q0,
            k,
            offs_m0,
            start_n,
            offs_n,
            qk_scale,
            l_i0,
            l_i0_1,
            m_i0,
            acc0,
            v,
            dtype,
            STAGE,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
        )

        offsetkv_y += BLOCK_N

    return acc0, l_i0, l_i0_1, m_i0


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [3]
else:
    NUM_STAGES_OPTIONS = [3]

if is_tile_enabled():
    # Helper to build config with optional minRegAutoWS/maxRegAutoWS
    def make_tile_config(BM, BN, occ, subtile, vectmul, add2reduce):
        config_kwargs = {
            "BLOCK_M": BM,
            "BLOCK_N": BN,
            "occupancy": occ,
            "SUBTILING": subtile,
            "VECT_MUL": vectmul,
            "FADD2_REDUCE": add2reduce,
        }
        extra_kwargs = {"pre_hook": _host_descriptor_pre_hook}

        # Only add minRegAutoWS/maxRegAutoWS if supported (triton/tree/ws-3.5)
        if HAS_REG_AUTO_WS:
            extra_kwargs["minRegAutoWS"] = 24
            extra_kwargs["maxRegAutoWS"] = 152

        return triton.Config(config_kwargs, **extra_kwargs)

    configs = [
        make_tile_config(BM, BN, occ, subtile, vectmul, add2reduce)
        for BM in [64, 128, 256]
        for BN in [64, 128]
        for occ in [1, 2]
        for subtile in [True]
        for vectmul in [0]
        for add2reduce in [False]
    ]
else:
    # Helper to build config with optional minRegAutoWS/maxRegAutoWS
    def make_standard_config(BM, BN, s, w, subtile, vectmul, add2reduce, maxreg):
        config_kwargs = {
            "BLOCK_M": BM,
            "BLOCK_N": BN,
            "SUBTILING": subtile,
            "VECT_MUL": vectmul,
            "FADD2_REDUCE": add2reduce,
            "DP_FACTOR": 2,
        }
        extra_kwargs = {
            "num_stages": s,
            "num_warps": w,
            "pre_hook": _host_descriptor_pre_hook,
        }

        # Only add minRegAutoWS/maxRegAutoWS if supported (triton/tree/ws-3.5)
        if HAS_REG_AUTO_WS:
            extra_kwargs["minRegAutoWS"] = 24
            extra_kwargs["maxRegAutoWS"] = maxreg

        return triton.Config(config_kwargs, **extra_kwargs)

    configs = [
        make_standard_config(BM, BN, s, w, subtile, vectmul, add2reduce, maxreg)
        for BM in [256]
        for BN in [64, 128]
        for s in NUM_STAGES_OPTIONS
        for w in [4]
        for subtile in [True]
        for vectmul in [1]
        for add2reduce in [False]
        for maxreg in [152, 192]
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        is_cuda()
        and torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M * BLOCK_N < 128 * 128
        and conf.num_warps == 8
    )


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]

    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _mul_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _fma_f32x2(a, b, c):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc, rd;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mov.b64 rc, { $6, $7 };
            fma.rn.f32x2 rd, ra, rb, rc;
            mov.b64 { $0, $1 }, rd;
        }
        """,
        "=r,=r,r,r,r,r,r,r",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _reduce_fadd2(p0a, p1a, p0b, p1b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 rc, ra, rb;
            mov.b64 ra, { $2, $4 };
            mov.b64 rb, { $3, $5 };
            add.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [p0a, p0b, p1a, p1b],
        dtype=[tl.float32, tl.float32],
        is_pure=True,
        pack=1,
    )


@triton.jit
def _attn_fwd_tma_dp(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    pid,
    off_hz,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    start_m = pid  # tl.program_id(0)
    # off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m0 = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i0 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i0_0 = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc0 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    q0 = desc_q.load([qo_offset_y, 0])

    if FADD2_REDUCE:
        l_i0_1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
    else:
        l_i0_1 = 0

    if STAGE & 1:
        acc0, l_i0_0, l_i0_1, m_i0 = _attn_fwd_inner_oss_dp(
            acc0,
            l_i0_0,
            l_i0_1,
            m_i0,
            q0,
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m0,
            offs_n,
            N_CTX,  #
            warp_specialize,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
        )
    if STAGE & 2:
        acc0, l_i0_0, l_i0_1, m_i0 = _attn_fwd_inner_oss_dp(
            acc0,
            l_i0_0,
            l_i0_1,
            m_i0,
            q0,
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m0,
            offs_n,
            N_CTX,  #
            warp_specialize,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
        )

    if FADD2_REDUCE:
        l_i0 = l_i0_0 + l_i0_1
    else:
        l_i0 = l_i0_0

    m_i0 += tl.math.log2(l_i0)
    acc0 = acc0 / l_i0[:, None]
    m_ptrs0 = M + off_hz * N_CTX + offs_m0
    tl.store(m_ptrs0, m_i0)
    desc_o.store([qo_offset_y, 0], acc0.to(dtype))


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    pid = tl.program_id(0)
    off_hz = tl.program_id(1)
    _attn_fwd_tma_dp(
        sm_scale,
        M,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        pid,
        off_hz,
        N_CTX,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        FP8_OUTPUT,
        STAGE,
        warp_specialize,
        dtype,
        SUBTILING,
        VECT_MUL,
        FADD2_REDUCE,
    )


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd_persist(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    OUTER_LOOP: tl.constexpr,
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
    DP_FACTOR: tl.constexpr,
):
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    desc_q = tl.make_tensor_descriptor(
        desc_q,
        shape=[Z * H * N_CTX, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_k = tl.make_tensor_descriptor(
        desc_k,
        shape=[Z * H * N_CTX, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_v = tl.make_tensor_descriptor(
        desc_v,
        shape=[Z * H * N_CTX, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = tl.make_tensor_descriptor(
        desc_o,
        shape=[Z * H * N_CTX, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    # inner loop warpspec vs. outer loop warpspec
    for _ in tl.range(
        0,
        tiles_per_sm,
        warp_specialize=warp_specialize and OUTER_LOOP,
        data_partition_factor=DP_FACTOR,
    ):
        pid = tile_idx % n_tile_num
        off_hz = tile_idx // n_tile_num
        _attn_fwd_tma_dp(
            sm_scale,
            M,
            Z,
            H,
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            pid,
            off_hz,
            N_CTX,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            FP8_OUTPUT,
            STAGE,
            warp_specialize and not OUTER_LOOP,
            dtype,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
        )
        tile_idx += num_progs


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,  #
    Delta,  #
    Z,
    H,
    N_CTX,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
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
    desc_q,
    k,
    v,
    sm_scale,  #
    desc_do,  #
    desc_dq,
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    off_bh,
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
    dtype: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        q = desc_q.load([(off_bh + curr_m).to(tl.int32), 0])
        qT = tl.trans(q)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = desc_do.load([(off_bh + curr_m).to(tl.int32), 0])
        # Compute dV.
        ppT = pT
        ppT = ppT.to(dtype)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(dtype)
        dk += tl.dot(dsT, tl.trans(qT))
        # Compute dq = tl.dot(tl.trans(dsT), k)
        dq = tl.dot(tl.trans(dsT), k) * LN2
        desc_dq.atomic_add([(off_bh + curr_m).to(tl.int32), 0], dq)
        # Increment pointers.
        curr_m += step_m

    return dk, dv


def _bwd_host_descriptor_pre_hook(nargs):
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    HEAD_DIM = nargs["HEAD_DIM"]
    nargs["desc_q"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_do"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_dq"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_dv"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_dk"].block_shape = [BLOCK_N1, HEAD_DIM]


configs_bwd = [
    triton.Config(
        {
            "BLOCK_M1": 32,
            "BLOCK_N1": 128,
            "BLOCK_M2": 128,
            "BLOCK_N2": 32,
        },
        num_warps=4,
        num_stages=1,
        pre_hook=_bwd_host_descriptor_pre_hook,
    )
]


@triton.autotune(configs=configs_bwd, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd(
    desc_q,
    desc_k,
    desc_v,
    sm_scale,  #
    desc_do,  #
    desc_dq,
    desc_dk,
    desc_dv,  #
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
    dtype: tl.constexpr,
):
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    off_bh = (
        (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    ) // stride_tok
    pid = tl.program_id(0)

    # offset pointers for batch/head
    M += off_chz
    D += off_chz

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    start_n = pid * BLOCK_N1
    start_m = 0

    # load K and V: they stay in SRAM throughout the inner loop.
    k = desc_k.load([(off_bh + start_n).to(tl.int32), 0])
    v = desc_v.load([(off_bh + start_n).to(tl.int32), 0])
    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        desc_q,
        k,
        v,
        sm_scale,  #
        desc_do,  #
        desc_dq,
        M,
        D,  #
        stride_tok,
        stride_d,  #
        off_bh,
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=False,  #
        dtype=dtype,
    )

    desc_dv.store(
        [(off_bh + start_n).to(tl.int32), 0],
        dv.to(dtype),
    )

    # Write back dK.
    dk *= sm_scale
    desc_dk.store(
        [(off_bh + start_n).to(tl.int32), 0],
        dk.to(dtype),
    )


def torch_dtype_to_triton(dtype):
    if dtype == torch.float8_e5m2:
        return tl.float8e5
    return getattr(tl, str(dtype).split(".")[1])


class _attention_opt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, baseVariant):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        warp_specialize = True
        desc_q = q
        desc_v = v
        desc_k = k
        desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        def grid(META):
            return (
                triton.cdiv(q.shape[2], META["BLOCK_M"]),
                q.shape[0] * q.shape[1],
                1,
            )

        def grid_persist(META):
            return (
                min(
                    NUM_SMS,
                    triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1],
                ),
                1,
                1,
            )

        def grid_debug(META):
            return (
                1,
                1,
                1,
            )

        ctx.grid = grid
        persistent = baseVariant == "persistent" or baseVariant == "ws_persistent"
        if WITH_MAXNREG and is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and (
                q.dtype == torch.float16 or q.dtype == torch.bfloat16
            ):
                extra_kern_args["maxnreg"] = 128
            else:
                extra_kern_args["maxnreg"] = 128
        if persistent:
            _attn_fwd_persist[grid_persist](
                sm_scale,
                M,  #
                q.shape[0],
                q.shape[1],  #
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=stage,  #
                warp_specialize=warp_specialize,
                OUTER_LOOP=True,
                dtype=torch_dtype_to_triton(q.dtype),
                **extra_kern_args,
            )
        else:
            _attn_fwd[grid](
                sm_scale,
                M,  #
                q.shape[0],
                q.shape[1],  #
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=stage,  #
                warp_specialize=warp_specialize,
                dtype=torch_dtype_to_triton(q.dtype),
                **extra_kern_args,
            )

        ctx.save_for_backward(q, k, v, o, M)

        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
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

        dummy_block = [1, 1]
        HEAD_DIM = ctx.HEAD_DIM
        desc_k = TensorDescriptor(
            arg_k,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_q = TensorDescriptor(
            q,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_do = TensorDescriptor(
            do,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_dq = TensorDescriptor(
            dq,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_dk = TensorDescriptor(
            dk,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )
        desc_dv = TensorDescriptor(
            dv,
            shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=dummy_block,
        )

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(meta):
            return (
                triton.cdiv(N_CTX, meta["BLOCK_N1"]),  # tiles along N (K/V)
                1,  # (or cdiv over M if you need)
                BATCH * N_HEAD,
            )  # batch*heads

        _attn_bwd[grid](
            desc_q,
            desc_k,
            desc_v,
            ctx.sm_scale,
            desc_do,
            desc_dq,
            desc_dk,
            desc_dv,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            N_HEAD,
            N_CTX,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            dtype=torch_dtype_to_triton(q.dtype),
        )

        return dq, dk, dv, None, None, None, None


attention_opt = _attention_opt.apply
