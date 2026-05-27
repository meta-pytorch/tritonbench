# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe


from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore[attr-defined]

    HAS_TLX = True
except ImportError:
    tlx = None
    HAS_TLX = False

from generative_recommenders.common import (
    autotune_max_seq_len,
    switch_to_contiguous_if_needed,
)
from hammer.v2.ops.triton.template.triton_attention_utils import (
    _get_bufidx_phase,
    fast_silu,
)
from hammer.v2.ops.triton.template.triton_hstu_attention import (
    backward_off_common_preprocess,
    backward_valid_mask,
    forward_uih_common_preprocess,
    forward_valid_mask,
    target_common_preprocess,
)

try:
    # @manual=//triton:triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    TMA_AVAILABLE = True
except ImportError:
    TMA_AVAILABLE = False
    pass


def _host_descriptor_pre_hook(nargs) -> None:
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    DimQ = nargs["DimQ"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    DimQ = nargs["DimQ"]
    DimV = nargs["DimV"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, DimQ]
    nargs["desc_v"].block_shape = [BLOCK_N, DimV]
    nargs["desc_k"].block_shape = [BLOCK_N, DimQ]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, DimV]


def get_fwd_persistent_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "NUM_BUFFERS_Q": 1,
                "NUM_BUFFERS_KV": kv,
                "NUM_BUFFERS_QK": 1,
                "NUM_MMA_GROUPS": 2,
                "NUM_MMA_SLICES": 2,
                "GROUP_SIZE_N": gs,
                "NUM_REGS_CORR": corr,
                "NUM_REGS_ACT": act,
                "NUM_REGS_MMA": mma,
                "NUM_REGS_LOAD": load,
                "NUM_REGS_EPI": epi,
                "MOVE_QK_WAIT": qk_wait,
                "SLICE_MASK": slice_mask,
            },
            num_stages=0,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        )
        for kv in [3, 6]
        for gs in [1, 2, 4]
        for corr in [168, 184]
        for act in [184, 192]  # , 208]
        for mma in [24]  # , 32, 64]
        for load in [32]  # , 64]
        for epi in [48, 64]
        for qk_wait in [False]  # True, False]
        for slice_mask in [True]  # , False]
    ]
    return configs


def prune_configs_by_hdim(configs, named_args, **kwargs):
    DimQ = kwargs["DimQ"]
    STAGE = kwargs["STAGE"]
    target_kv_buffers = 6 if DimQ == 64 else 3
    target_group_size_n = 4 if STAGE == 3 else 1
    return [
        conf
        for conf in configs
        if conf.kwargs.get("NUM_BUFFERS_KV", 0) == target_kv_buffers
        # and conf.kwargs.get("GROUP_SIZE_N", 0) == target_group_size_n
    ]


@triton.jit
def tanh_approx_fp32(x):
    output = tl.inline_asm_elementwise(
        asm="""
        tanh.approx.f32 $0, $1;
        """,
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return output


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
def _get_unfused_loop_bounds(
    fused_lo, fused_hi, start_m, max_seq_len, BLOCK_N, STAGE: tl.constexpr
):
    if STAGE == 1:
        # First part of STAGE == 3 in _get_fused_loop_bounds
        lo = fused_lo
        hi = (start_m - fused_lo) // BLOCK_N * BLOCK_N + fused_lo
    elif STAGE == 2:
        # Second part of STAGE == 3 in _get_fused_loop_bounds
        lo = (start_m - fused_lo) // BLOCK_N * BLOCK_N + fused_lo
        hi = fused_hi
    else:
        tl.static_assert(STAGE == 3)
        # Maps to STAGE=1 in _get_fused_loop_bounds
        lo, hi = fused_lo, fused_hi
    return lo, hi


@triton.jit
def _attn_seq_info_preprocess(  # noqa C901
    off_z,
    num_targets,
    start_m,
    seq_len_q,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    max_attn_len: tl.constexpr,
    full_attn_size: tl.constexpr,
    contextual_seq_len: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
):
    n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
    uih_end = forward_uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
        low = 0
        high = seq_len_q
    else:
        low = 0
        high = start_m + BLOCK_M
        high = seq_len_q if seq_len_q < high else high
        if HAS_MAX_ATTN_LEN:
            if HAS_FULL_ATTN_SIZE > 0 and start_m + BLOCK_M >= uih_end - full_attn_size:
                low = 0
            else:
                if start_m > uih_end:
                    low = uih_end - max_attn_len
                else:
                    low = start_m - max_attn_len
            if HAS_CONTEXTUAL_SEQ_LEN:
                low = low if low > contextual_seq_len else 0
            else:
                low = low if low > 0 else 0
        if HAS_NUM_TARGETS:
            uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
            if uih_end < start_m:
                high = seq_len_q - n_targets
    return low, high, n_targets


@triton.jit
def _compute_offsets(
    tile_idx,
    H,
    num_pid_n,
    num_pid_in_group,
    seq_offsets,
    num_targets,
    max_attn_len,
    full_attn_size,
    contextual_seq_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
):
    group_id = tile_idx // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    start_m = (tile_idx % num_pid_in_group) // group_size_n
    start_m = start_m * BLOCK_M
    off_hz = first_pid_n + (tile_idx % group_size_n)
    off_z = off_hz // H
    off_h = off_hz % H

    # load sequence offsets
    seq_start = tl.load(seq_offsets + off_z)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)

    if start_m < seq_len:
        lo, hi, n_targets = _attn_seq_info_preprocess(
            off_z,
            num_targets,
            start_m,
            seq_len,
            BLOCK_M,
            BLOCK_N,
            max_attn_len,
            full_attn_size,
            contextual_seq_len,
            HAS_NUM_TARGETS,
            HAS_MAX_ATTN_LEN,
            HAS_FULL_ATTN_SIZE,
            HAS_CONTEXTUAL_SEQ_LEN,
        )
    else:
        lo, hi, n_targets = 0, 0, 0

    return start_m, off_z, off_h, lo, hi, seq_start, seq_len, n_targets


@triton.jit
def _split_n(x, SPLIT_FACTOR: tl.constexpr):
    if SPLIT_FACTOR == 1:
        return (x,)
    else:
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        return _split_n(x0, SPLIT_FACTOR // 2) + _split_n(x1, SPLIT_FACTOR // 2)


@triton.jit
def _join_n(xs):
    if len(xs) == 1:
        return xs[0]
    else:
        x0 = _join_n(xs[: len(xs) // 2])
        x1 = _join_n(xs[len(xs) // 2 :])
        x = tl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] * 2])
        return x


@triton.jit
def _mask_scalar(qk, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s
    col_lim_right_cur = max(col_lim_right_s, 0)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return tl.where(mask_i_bit, qk, -float("inf"))


@triton.jit
def _apply_causal_mask(qk, col_limit_right, BLOCK_N: tl.constexpr):
    # Apply causal mask via a bitmask calculated for each block of 16 elements.
    # This allows the efficient R2P (register to predicate) instruction to be used at the SASS level.
    # Credit to Tri Dao,
    # https://github.com/Dao-AILab/flash-attention/commit/bac1001e4f6caa09d70537495d6746a685a2fa78
    #
    # NOTE: We use map_elementiwse here in order to generate an interleaved sequence of instructions
    # that processes one element of qk at a time. This improves ptxas's resulting SASS.
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    s = offs_n & ~0xF
    i = offs_n & 0xF
    return tl.map_elementwise(_mask_scalar, qk, col_limit_right, s, i)


@triton.jit
def _softmax_inner_loop(
    qk_fulls,
    qk_tiles,
    p_fulls,
    p_tiles,
    alpha_empties,
    alpha_fulls,
    alpha_tiles,
    cid,
    accum_cnt_qk,
    qk_scale,
    offs_m,
    m_i,
    l_i,
    start_m,
    max_seq_len,
    fused_lo,
    fused_hi,
    out_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DimQ: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    STAGE: tl.constexpr,
):
    lo, hi = _get_unfused_loop_bounds(
        fused_lo, fused_hi, start_m, max_seq_len, BLOCK_N, STAGE
    )

    for start_n in tl.range(lo, hi, BLOCK_N):
        _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
        tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
        qk = tlx.local_load(tlx.local_view(qk_tiles, cid))

        if STAGE == 2:
            col_limit_right = (offs_m - start_n + 1)[:, None]
            qk = _apply_causal_mask(qk, col_limit_right, BLOCK_N)

        # compute m_i, p in registers
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        tlx.barrier_wait(tlx.local_view(alpha_empties, cid), qk_phase ^ 1)
        # Use alpha[0] for cid=0, and alpha[BLOCK_N] for cid=1
        tlx.local_store(tlx.local_view(alpha_tiles, cid * BLOCK_N), alpha[:, None])
        tlx.barrier_arrive(tlx.local_view(alpha_fulls, cid))

        qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
        qks = _split_n(qk, NUM_MMA_SLICES)
        ps = ()
        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
            # prepare p for the v dot
            # Use p[NUM_MMA_SLICES + slice_id] for cid=0, and
            # p[NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id] for cid=1
            p_bufIdx = cid * NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id
            p_i = tl.math.exp2(qks[slice_id])
            tlx.local_store(tlx.local_view(p_tiles, p_bufIdx), p_i.to(out_dtype))
            tlx.barrier_arrive(tlx.local_view(p_fulls, slice_id + cid * NUM_MMA_SLICES))
            ps = ps + (p_i,)

        p = _join_n(ps)
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        accum_cnt_qk += 1

    return m_i, l_i, accum_cnt_qk


@triton.jit
def _silu_inner_loop(
    qk_fulls,
    qk_tiles,
    p_fulls,
    p_tiles,
    cid,
    accum_cnt_qk,
    alpha,
    offs_m,
    start_m,
    seq_len,
    lo,
    hi,
    out_dtype,
    contextual_seq_len,
    max_attn_len,
    n_targets,
    BLOCK_M_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DimQ: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    MOVE_QK_WAIT: tl.constexpr,
    SLICE_MASK: tl.constexpr,
):
    # Pre-scale alpha by 0.5 to match the SiLU identity:
    # x*sigmoid(x) = (x/2)*(1+tanh(x/2)) = fma(x/2, tanh(x/2), x/2)
    # This moves the per-element *0.5 to a one-time scalar multiply.
    alpha = alpha * 0.5

    # It is numerically correct to process every block in the part 2 below with masking
    if HAS_MAX_ATTN_LEN or start_m + (1 + cid) * BLOCK_M_SPLIT >= (seq_len - n_targets):
        low, mid, high = lo, lo, hi
    else:
        low = lo
        mid = (start_m + cid * BLOCK_M_SPLIT - lo) // BLOCK_N * BLOCK_N + lo
        high = hi

    # first part where there is no masking
    for start_n in tl.range(low, mid, BLOCK_N):
        _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
        tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
        qk = tlx.local_load(tlx.local_view(qk_tiles, cid))

        qk = qk * alpha

        qks = _split_n(qk, NUM_MMA_SLICES)
        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
            # prepare p for the v dot
            # Use p[NUM_MMA_SLICES + slice_id] for cid=0, and
            # p[NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id] for cid=1
            p_bufIdx = cid * NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id
            p_i = fast_silu(qks[slice_id], MULT_BY_X=True)
            tlx.local_store(tlx.local_view(p_tiles, p_bufIdx), p_i.to(out_dtype))
            tlx.barrier_arrive(tlx.local_view(p_fulls, slice_id + cid * NUM_MMA_SLICES))
        accum_cnt_qk += 1

    # second part where there are causal and other maskings (target, seq_len etc.)
    for start_n in tl.range(mid, high, BLOCK_N):
        _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

        if SLICE_MASK:
            # When doing per-slice masking, load qk unconditionally
            tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
            qk = tlx.local_load(tlx.local_view(qk_tiles, cid))
        else:
            # When doing full-block masking, use MOVE_QK_WAIT optimization
            if MOVE_QK_WAIT:
                # Calculate mask first (overlapping with barrier wait), then load qk
                offs_n = start_n + tl.arange(0, BLOCK_N)
                valid_mask = forward_valid_mask(
                    offs_m,
                    offs_n,
                    seq_len,
                    contextual_seq_len,
                    max_attn_len,
                    n_targets,
                    HAS_CONTEXTUAL_SEQ_LEN,
                    HAS_NUM_TARGETS,
                    HAS_MAX_ATTN_LEN,
                )
                masked_alpha = tl.where(valid_mask, alpha, 0.0)

                tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
                qk = tlx.local_load(tlx.local_view(qk_tiles, cid))
            else:
                # Load qk first, then calculate mask
                tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
                qk = tlx.local_load(tlx.local_view(qk_tiles, cid))

                offs_n = start_n + tl.arange(0, BLOCK_N)
                valid_mask = forward_valid_mask(
                    offs_m,
                    offs_n,
                    seq_len,
                    contextual_seq_len,
                    max_attn_len,
                    n_targets,
                    HAS_CONTEXTUAL_SEQ_LEN,
                    HAS_NUM_TARGETS,
                    HAS_MAX_ATTN_LEN,
                )
                masked_alpha = tl.where(valid_mask, alpha, 0.0)

            qk = qk * masked_alpha

        qks = _split_n(qk, NUM_MMA_SLICES)
        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
            # prepare p for the v dot
            # Use p[NUM_MMA_SLICES + slice_id] for cid=0, and
            # p[NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id] for cid=1
            p_bufIdx = cid * NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id
            if SLICE_MASK:
                offs_n = start_n + tl.arange(
                    slice_id * BLOCK_N // NUM_MMA_SLICES,
                    (slice_id + 1) * BLOCK_N // NUM_MMA_SLICES,
                )
                valid_mask = forward_valid_mask(
                    offs_m,
                    offs_n,
                    seq_len,
                    contextual_seq_len,
                    max_attn_len,
                    n_targets,
                    HAS_CONTEXTUAL_SEQ_LEN,
                    HAS_NUM_TARGETS,
                    HAS_MAX_ATTN_LEN,
                )
                masked_alpha = tl.where(valid_mask, alpha, 0.0)
                qks_t = qks[slice_id] * masked_alpha
                p_i = fast_silu(qks_t, MULT_BY_X=True)
            else:
                p_i = fast_silu(qks[slice_id], MULT_BY_X=True)
            tlx.local_store(tlx.local_view(p_tiles, p_bufIdx), p_i.to(out_dtype))
            tlx.barrier_arrive(tlx.local_view(p_fulls, slice_id + cid * NUM_MMA_SLICES))
        accum_cnt_qk += 1

    return accum_cnt_qk


@triton.jit
def _softmax_correction_inner_loop(
    desc_o,
    lo,
    hi,
    accum_cnt,
    tile_cnt,
    start_m,
    seq_start,
    seq_len,
    off_h,
    alpha_fulls,
    alpha_empties,
    alpha_tiles,
    acc_fulls,
    acc_empties,
    acc_tiles,
    o_fulls,
    o_empties,
    o_tiles,
    l_fulls,
    l_tiles,
    m_tiles,
    qk_empties,
    M,
    stride_mm,
    DimQ: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    BLOCK_M_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    for _ in tl.range(lo, hi, BLOCK_N):
        _, phase = _get_bufidx_phase(accum_cnt, 1)
        for cid in tl.static_range(0, NUM_MMA_GROUPS):
            # -- update output accumulator --
            tlx.barrier_wait(alpha_fulls[cid], phase)
            # Use alpha[0] for cid=0, and alpha[BLOCK_N] for cid=1
            alpha_1 = tlx.local_load(alpha_tiles[cid * BLOCK_N])
            tlx.barrier_arrive(alpha_empties[cid])
            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                subslice = tlx.subslice(
                    acc_tiles[cid],
                    DimQ * slice_id // NUM_MMA_SLICES,
                    DimQ // NUM_MMA_SLICES,
                )
                acc = tlx.local_load(subslice)
                # acc = acc * alpha_1
                acc = _mul_f32x2(acc, alpha_1)
                tlx.local_store(subslice, acc)
            tlx.barrier_arrive(acc_fulls[cid])
        accum_cnt += 1

    _, phase = _get_bufidx_phase(tile_cnt, 1)
    for cid in tl.static_range(0, NUM_MMA_GROUPS):
        # epilogue
        tlx.barrier_wait(l_fulls[cid], phase)
        # Use l[1]/l[1+BLOCK_N] and m[2][2 + BLOCK_N]
        # to disambigulate from alpha[0]/alpha[BLOCK_N]
        l = tlx.local_load(l_tiles[cid * BLOCK_N + 1])
        m = tlx.local_load(m_tiles[cid * BLOCK_N + 2])
        # Signal qk_empties after both l and m loads complete,
        # since both tiles share the same synchronization group.
        tlx.barrier_arrive(qk_empties[cid])
        m += tl.math.log2(l)
        offs_m = start_m + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
        m_ptrs = M + (offs_m + seq_start) * stride_mm + off_h
        m = tl.reshape(m, [BLOCK_M_SPLIT])
        tl.store(m_ptrs, m, mask=offs_m < seq_len)

        tlx.barrier_wait(acc_empties[cid], phase)
        tlx.barrier_wait(o_empties[cid], phase ^ 1)
        scale = 1 / l
        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
            subslice = tlx.subslice(
                acc_tiles[cid],
                DimQ * slice_id // NUM_MMA_SLICES,
                DimQ // NUM_MMA_SLICES,
            )
            acc = tlx.local_load(subslice)
            acc = _mul_f32x2(acc, scale)
            acc = acc.to(tlx.dtype_of(desc_o))
            subslice_o = tlx.local_slice(
                o_tiles[cid],
                [0, DimQ * slice_id // NUM_MMA_SLICES],
                [BLOCK_M_SPLIT, DimQ // NUM_MMA_SLICES],
            )
            tlx.local_store(subslice_o, acc)
        tlx.barrier_arrive(o_fulls[cid])
    return accum_cnt


@triton.autotune(
    configs=get_fwd_persistent_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "DimQ", "STAGE"],
    prune_configs_by={"early_config_prune": prune_configs_by_hdim},
)
@triton.jit
def _attn_fwd_ws(
    Out,
    sm_scale,
    Z,
    H,
    M,  #
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    seq_offsets,
    max_seq_len,  #
    num_targets,
    max_attn_len,
    full_attn_size,
    contextual_seq_len,
    stride_mm,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_oh,
    attn_scale,
    SOFTMAX: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    AUTOTUNE_MAX_SEQ_LEN: tl.constexpr,
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    NUM_BUFFERS_Q: tl.constexpr,  #
    NUM_BUFFERS_KV: tl.constexpr,  #
    NUM_BUFFERS_QK: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    NUM_MMA_SLICES: tl.constexpr,  #
    GROUP_SIZE_N: tl.constexpr,  #
    NUM_REGS_CORR: tl.constexpr,
    NUM_REGS_ACT: tl.constexpr,
    NUM_REGS_MMA: tl.constexpr,
    NUM_REGS_LOAD: tl.constexpr,
    NUM_REGS_EPI: tl.constexpr,
    MOVE_QK_WAIT: tl.constexpr,
    SLICE_MASK: tl.constexpr,
):
    tl.static_assert(NUM_MMA_GROUPS == 2)
    tl.static_assert(DimV == DimQ)

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    # original grid
    #   triton.cdiv(max_seq_len, META["BLOCK_M"]),
    #   Z * H
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_m = tl.cdiv(max_seq_len, BLOCK_M)
    num_pid_n = Z * H
    num_pid_in_group = num_pid_m * GROUP_SIZE_N
    total_tiles = num_pid_m * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, DimQ), tlx.dtype_of(desc_q), NUM_MMA_GROUPS * NUM_BUFFERS_Q
    )
    kv_tiles = tlx.local_alloc((BLOCK_N, DimV), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    o_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, DimV), tlx.dtype_of(desc_o), NUM_MMA_GROUPS
    )

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    o_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    o_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    # allocate TMEM buffers and barriers
    qk_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem
    )
    # Shared buffer for QK, P and Alpha, l, and m.
    # A single QK buffer is split evenly:
    #   - First half  : stores P
    #   - Second half  : stores Alpha, l, and m
    #     QK : |                              BLK_M/2 * BLOCK_N * fp32                  |
    #     P:                                                |  BLK_M/2 * BLOCK_N * fp16 |
    #  Alpha : |BLK_M/2*1*fp32|
    #     l :                 |BLK_M/2*1*fp32|
    #     m :                                |BLK_M/2*1*fp32|
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N // NUM_MMA_SLICES),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS * NUM_MMA_SLICES * 2,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    alpha_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    l_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    m_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )

    acc_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, DimV), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem
    )

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_MMA_SLICES)
    acc_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    alpha_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    alpha_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    l_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    with tlx.async_tasks():
        # correction group
        with tlx.async_task("default"):
            accum_cnt = 0
            phase = 0
            tile_cnt = 0
            for _ in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_z, off_h, lo, hi, seq_start, seq_len, n_targets = (
                    _compute_offsets(
                        tile_idx,
                        H,
                        num_pid_n,
                        num_pid_in_group,
                        seq_offsets,
                        num_targets,
                        max_attn_len,
                        full_attn_size,
                        contextual_seq_len,
                        BLOCK_M,
                        BLOCK_N,
                        GROUP_SIZE_N,
                        HAS_NUM_TARGETS,
                        HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE,
                        HAS_CONTEXTUAL_SEQ_LEN,
                    )
                )
                if start_m < seq_len:
                    if SOFTMAX:
                        accum_cnt = _softmax_correction_inner_loop(
                            desc_o,
                            lo,
                            hi,
                            accum_cnt,
                            tile_cnt,
                            start_m,
                            seq_start,
                            seq_len,
                            off_h,
                            alpha_fulls,
                            alpha_empties,
                            alpha_tiles,
                            acc_fulls,
                            acc_empties,
                            acc_tiles,
                            o_fulls,
                            o_empties,
                            o_tiles,
                            l_fulls,
                            l_tiles,
                            m_tiles,
                            qk_empties,
                            M,
                            stride_mm,
                            DimQ,
                            NUM_MMA_SLICES,
                            NUM_MMA_GROUPS,
                            BLOCK_M_SPLIT,
                            BLOCK_N,
                        )
                    else:
                        # silu correction, apply attn_scale
                        tl.static_assert(ATTN_SCALE_TYPE == "scalar")
                        scale = tl.load(attn_scale).to(tl.float32)
                        _, phase = _get_bufidx_phase(tile_cnt, 1)
                        for cid in tl.static_range(0, NUM_MMA_GROUPS):
                            tlx.barrier_wait(acc_empties[cid], phase)
                            tlx.barrier_wait(o_empties[cid], phase ^ 1)
                            # epilogue: load from TMEM, scale, store to SMEM

                            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                                subslice = tlx.subslice(
                                    acc_tiles[cid],
                                    DimQ * slice_id // NUM_MMA_SLICES,
                                    DimQ // NUM_MMA_SLICES,
                                )
                                acc = tlx.local_load(subslice)
                                acc = _mul_f32x2(acc, scale)
                                acc = acc.to(tlx.dtype_of(desc_o))
                                subslice_o = tlx.local_slice(
                                    o_tiles[cid],
                                    [0, DimQ * slice_id // NUM_MMA_SLICES],
                                    [BLOCK_M_SPLIT, DimQ // NUM_MMA_SLICES],
                                )
                                tlx.local_store(subslice_o, acc)
                            tlx.barrier_arrive(o_fulls[cid])
                            # Signal qk_empties so MMA group can proceed with next tile
                            tlx.barrier_arrive(qk_empties[cid])

                    tile_cnt += 1

                tile_idx += num_progs

        # softmax/silu groups
        with tlx.async_task(
            num_warps=4, registers=NUM_REGS_ACT, replicate=NUM_MMA_GROUPS
        ):
            accum_cnt_qk = 0
            for _ in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_z, off_h, lo, hi, seq_start, seq_len, n_targets = (
                    _compute_offsets(
                        tile_idx,
                        H,
                        num_pid_n,
                        num_pid_in_group,
                        seq_offsets,
                        num_targets,
                        max_attn_len,
                        full_attn_size,
                        contextual_seq_len,
                        BLOCK_M,
                        BLOCK_N,
                        GROUP_SIZE_N,
                        HAS_NUM_TARGETS,
                        HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE,
                        HAS_CONTEXTUAL_SEQ_LEN,
                    )
                )
                if start_m < seq_len:
                    if SOFTMAX:
                        # initialize pointer to m and l
                        m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                        l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
                        acc = tl.zeros([BLOCK_M_SPLIT, DimQ], dtype=tl.float32)
                        qk_scale = sm_scale
                        qk_scale *= 1.44269504  # 1/log(2)
                        out_dtype = tlx.dtype_of(desc_v)

                        cid = tlx.async_task_replica_id()
                        offs_m = start_m + (
                            (cid * BLOCK_M_SPLIT) + tl.arange(0, BLOCK_M_SPLIT)
                        )
                        if STAGE & 1:
                            m_i, l_i, accum_cnt_qk = _softmax_inner_loop(
                                qk_fulls,
                                qk_tiles,
                                p_fulls,
                                p_tiles,
                                alpha_empties,
                                alpha_fulls,
                                alpha_tiles,
                                cid,
                                accum_cnt_qk,
                                qk_scale,
                                offs_m,
                                m_i,
                                l_i,
                                start_m,
                                max_seq_len,
                                lo,
                                hi,
                                out_dtype,
                                BLOCK_M,
                                BLOCK_N,
                                DimQ,
                                NUM_MMA_SLICES,
                                NUM_MMA_GROUPS,
                                STAGE=4 - STAGE,
                            )

                        if STAGE & 2:
                            m_i, l_i, accum_cnt_qk = _softmax_inner_loop(
                                qk_fulls,
                                qk_tiles,
                                p_fulls,
                                p_tiles,
                                alpha_empties,
                                alpha_fulls,
                                alpha_tiles,
                                cid,
                                accum_cnt_qk,
                                qk_scale,
                                offs_m,
                                m_i,
                                l_i,
                                start_m,
                                max_seq_len,
                                lo,
                                hi,
                                out_dtype,
                                BLOCK_M,
                                BLOCK_N,
                                DimQ,
                                NUM_MMA_SLICES,
                                NUM_MMA_GROUPS,
                                STAGE=2,
                            )

                        # prepare l_i for the epilog
                        # Use l[1]/l[1+BLOCK_N] and m[2][2 + BLOCK_N]
                        # to disambigulate from alpha[0]/alpha[BLOCK_N]
                        tlx.local_store(l_tiles[cid * BLOCK_N + 1], l_i[:, None])
                        tlx.local_store(m_tiles[cid * BLOCK_N + 2], m_i[:, None])
                        tlx.barrier_arrive(l_fulls[cid])
                    else:
                        # silu
                        out_dtype = tlx.dtype_of(desc_v)
                        cid = tlx.async_task_replica_id()
                        offs_m = start_m + (
                            (cid * BLOCK_M_SPLIT) + tl.arange(0, BLOCK_M_SPLIT)
                        )
                        accum_cnt_qk = _silu_inner_loop(
                            qk_fulls,
                            qk_tiles,
                            p_fulls,
                            p_tiles,
                            cid,
                            accum_cnt_qk,
                            sm_scale,
                            offs_m,
                            start_m,
                            seq_len,
                            lo,
                            hi,
                            out_dtype,
                            contextual_seq_len,
                            max_attn_len,
                            n_targets,
                            BLOCK_M_SPLIT,
                            BLOCK_N,
                            DimQ,
                            NUM_MMA_SLICES,
                            NUM_MMA_GROUPS,
                            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                            HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                            MOVE_QK_WAIT=MOVE_QK_WAIT,
                            SLICE_MASK=SLICE_MASK,
                        )

                tile_idx += num_progs

        # mma group
        with tlx.async_task(num_warps=1, registers=NUM_REGS_MMA):
            accum_cnt_kv = 0
            accum_cnt_qk = 0

            tile_cnt = 0
            for _ in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_z, off_h, lo, hi, seq_start, seq_len, n_targets = (
                    _compute_offsets(
                        tile_idx,
                        H,
                        num_pid_n,
                        num_pid_in_group,
                        seq_offsets,
                        num_targets,
                        max_attn_len,
                        full_attn_size,
                        contextual_seq_len,
                        BLOCK_M,
                        BLOCK_N,
                        GROUP_SIZE_N,
                        HAS_NUM_TARGETS,
                        HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE,
                        HAS_CONTEXTUAL_SEQ_LEN,
                    )
                )
                if start_m < seq_len:
                    q_bufIdx, q_phase = _get_bufidx_phase(tile_cnt, NUM_BUFFERS_Q)
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    v_bufIdx, v_phase = _get_bufidx_phase(
                        accum_cnt_kv + 1, NUM_BUFFERS_KV
                    )

                    # wait for the K buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)

                    # wait for the Q buffer to be populated by the producer
                    tlx.barrier_wait(q_fulls[q_bufIdx], q_phase)

                    # -- compute q0 @ k ----
                    k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                    tlx.barrier_wait(qk_empties[0], q_phase ^ 1)
                    tlx.async_dot(
                        q_tiles[0],
                        k_tile,
                        qk_tiles[0],
                        use_acc=False,
                        mBarriers=[qk_fulls[0]],
                    )

                    # -- compute q1 @ k ----
                    tlx.barrier_wait(q_fulls[q_bufIdx + NUM_BUFFERS_Q], q_phase)
                    tlx.barrier_wait(qk_empties[1], q_phase ^ 1)
                    tlx.async_dot(
                        q_tiles[1],
                        k_tile,
                        qk_tiles[1],
                        use_acc=False,
                        mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                    )

                    _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

                    # -- compute p0 @ v ----
                    # wait for the V buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                    if SOFTMAX:
                        tlx.barrier_wait(acc_fulls[0], qk_phase)
                    # Use p[NUM_MMA_SLICES + slice_id] for cid=0, and
                    # p[NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id] for cid=1
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        tlx.barrier_wait(
                            p_fulls[slice_id + 0 * NUM_MMA_SLICES], qk_phase
                        )
                        kv_slice = tlx.local_slice(
                            kv_tiles[v_bufIdx],
                            [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                            [BLOCK_N // NUM_MMA_SLICES, DimQ],
                        )
                        p_bufIdx = NUM_MMA_SLICES + slice_id
                        tlx.async_dot(
                            p_tiles[p_bufIdx],
                            kv_slice,
                            acc_tiles[0],
                            use_acc=slice_id > 0,
                        )

                    acc1_init = False

                    for i in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                        v_bufIdx_prev = v_bufIdx
                        qk_phase_prev = qk_phase

                        accum_cnt_qk += 1
                        accum_cnt_kv += 2
                        k_bufIdx, k_phase = _get_bufidx_phase(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        v_bufIdx, v_phase = _get_bufidx_phase(
                            accum_cnt_kv + 1, NUM_BUFFERS_KV
                        )

                        # -- compute q0 @ k ----
                        # wait for the K buffer to be populated by the producer
                        tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                        k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                        _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

                        tlx.async_dot(
                            q_tiles[0],
                            k_tile,
                            qk_tiles[0],
                            use_acc=False,
                            mBarriers=[qk_fulls[0]],
                        )

                        # -- compute p1 @ v from the previous iteration----
                        if SOFTMAX:
                            tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            tlx.barrier_wait(
                                p_fulls[slice_id + 1 * NUM_MMA_SLICES], qk_phase_prev
                            )
                            kv_slice = tlx.local_slice(
                                kv_tiles[v_bufIdx_prev],
                                [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                                [BLOCK_N // NUM_MMA_SLICES, DimQ],
                            )
                            p_bufIdx = (
                                1 * NUM_MMA_GROUPS * NUM_MMA_SLICES
                                + NUM_MMA_SLICES
                                + slice_id
                            )
                            use_acc = acc1_init if slice_id == 0 else True
                            mBarriers = (
                                [kv_empties[v_bufIdx_prev]]
                                if slice_id == NUM_MMA_SLICES - 1
                                else []
                            )
                            tlx.async_dot(
                                p_tiles[p_bufIdx],
                                kv_slice,
                                acc_tiles[1],
                                use_acc=use_acc,
                                mBarriers=mBarriers,
                            )

                        acc1_init = True

                        # -- compute q1 @ k ----
                        tlx.async_dot(
                            q_tiles[1],
                            k_tile,
                            qk_tiles[1],
                            use_acc=False,
                            mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                        )

                        # -- compute p0 @ v ----
                        # wait for the V buffer to be populated by the producer
                        tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)

                        if SOFTMAX:
                            tlx.barrier_wait(acc_fulls[0], qk_phase)
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            tlx.barrier_wait(
                                p_fulls[slice_id + 0 * NUM_MMA_SLICES], qk_phase
                            )
                            # Use p[1] for cid=0, and p[3] for cid=1
                            kv_slice = tlx.local_slice(
                                kv_tiles[v_bufIdx],
                                [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                                [BLOCK_N // NUM_MMA_SLICES, DimQ],
                            )
                            p_bufIdx = NUM_MMA_SLICES + slice_id
                            tlx.async_dot(
                                p_tiles[p_bufIdx],
                                kv_slice,
                                acc_tiles[0],
                                use_acc=True,
                            )

                    tlx.tcgen05_commit(q_empties[q_bufIdx])
                    tlx.tcgen05_commit(q_empties[q_bufIdx + NUM_BUFFERS_Q])
                    tlx.tcgen05_commit(acc_empties[0])

                    # -- compute p1 @ v ----
                    if SOFTMAX:
                        tlx.barrier_wait(acc_fulls[1], qk_phase)
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        tlx.barrier_wait(p_fulls[slice_id + NUM_MMA_SLICES], qk_phase)
                        # Use p[1] for cid=0, and p[3] for cid=1
                        kv_slice = tlx.local_slice(
                            kv_tiles[v_bufIdx],
                            [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                            [BLOCK_N // NUM_MMA_SLICES, DimQ],
                        )
                        p_bufIdx = (
                            1 * NUM_MMA_GROUPS * NUM_MMA_SLICES
                            + NUM_MMA_SLICES
                            + slice_id
                        )
                        use_acc = acc1_init if slice_id == 0 else True
                        mBarriers = (
                            [acc_empties[1], kv_empties[v_bufIdx]]
                            if slice_id == NUM_MMA_SLICES - 1
                            else []
                        )
                        tlx.async_dot(
                            p_tiles[p_bufIdx],
                            kv_slice,
                            acc_tiles[1],
                            use_acc=use_acc,
                            mBarriers=mBarriers,
                        )

                    accum_cnt_qk += 1
                    accum_cnt_kv += 2
                    tile_cnt += 1
                tile_idx += num_progs

        # load
        with tlx.async_task(num_warps=1, registers=NUM_REGS_LOAD):
            accum_cnt_kv = 0
            tile_cnt = 0
            for _ in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_z, off_h, lo, hi, seq_start, seq_len, n_targets = (
                    _compute_offsets(
                        tile_idx,
                        H,
                        num_pid_n,
                        num_pid_in_group,
                        seq_offsets,
                        num_targets,
                        max_attn_len,
                        full_attn_size,
                        contextual_seq_len,
                        BLOCK_M,
                        BLOCK_N,
                        GROUP_SIZE_N,
                        HAS_NUM_TARGETS,
                        HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE,
                        HAS_CONTEXTUAL_SEQ_LEN,
                    )
                )

                if start_m < seq_len:
                    # load q0
                    q_bufIdx, q_phase = _get_bufidx_phase(tile_cnt, NUM_BUFFERS_Q)
                    tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        q_fulls[q_bufIdx], 2 * BLOCK_M_SPLIT * DimQ
                    )  # float16
                    qo_offset_y_split = seq_start + start_m

                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_bufIdx],
                        [qo_offset_y_split.to(tl.int32), off_h * stride_qh],
                        q_fulls[q_bufIdx],
                    )

                    # loop over loading k, v
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # wait for the K buffer to be released by the consumer
                    k_empty = tlx.local_view(kv_empties, k_bufIdx)
                    tlx.barrier_wait(k_empty, k_phase ^ 1)

                    # load K
                    kv_offset_y = seq_start + lo
                    k_full = tlx.local_view(kv_fulls, k_bufIdx)
                    k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                    tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * DimQ)  # float16
                    tlx.async_descriptor_load(
                        desc_k,
                        k_tile,
                        [kv_offset_y.to(tl.int32), off_h * stride_kh],
                        k_full,
                    )

                    # load q1
                    q_bufIdx += NUM_BUFFERS_Q
                    tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        q_fulls[q_bufIdx], 2 * BLOCK_M_SPLIT * DimQ
                    )  # float16
                    qo_offset_y_split += BLOCK_M_SPLIT
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_bufIdx],
                        [qo_offset_y_split.to(tl.int32), off_h * stride_qh],
                        q_fulls[q_bufIdx],
                    )

                    v_bufIdx, v_phase = _get_bufidx_phase(
                        accum_cnt_kv + 1, NUM_BUFFERS_KV
                    )
                    # wait for the V buffer to be released by the consumer
                    v_empty = tlx.local_view(kv_empties, v_bufIdx)
                    tlx.barrier_wait(v_empty, v_phase ^ 1)
                    # load V
                    v_full = tlx.local_view(kv_fulls, v_bufIdx)
                    v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                    tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * DimQ)  # float16
                    tlx.async_descriptor_load(
                        desc_v,
                        v_tile,
                        [kv_offset_y.to(tl.int32), off_h * stride_vh],
                        v_full,
                    )

                    kv_offset_y += BLOCK_N
                    accum_cnt_kv += 2

                    for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                        k_bufIdx, k_phase = _get_bufidx_phase(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        # wait for the K buffer to be released by the consumer
                        k_empty = tlx.local_view(kv_empties, k_bufIdx)
                        tlx.barrier_wait(k_empty, k_phase ^ 1)
                        # load K
                        k_full = tlx.local_view(kv_fulls, k_bufIdx)
                        k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                        tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * DimQ)  # float16
                        tlx.async_descriptor_load(
                            desc_k,
                            k_tile,
                            [kv_offset_y.to(tl.int32), off_h * stride_kh],
                            k_full,
                        )

                        v_bufIdx, v_phase = _get_bufidx_phase(
                            accum_cnt_kv + 1, NUM_BUFFERS_KV
                        )
                        # wait for the V buffer to be released by the consumer
                        v_empty = tlx.local_view(kv_empties, v_bufIdx)
                        tlx.barrier_wait(v_empty, v_phase ^ 1)
                        # load V
                        v_full = tlx.local_view(kv_fulls, v_bufIdx)
                        v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                        tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * DimQ)  # float16
                        tlx.async_descriptor_load(
                            desc_v,
                            v_tile,
                            [kv_offset_y.to(tl.int32), off_h * stride_vh],
                            v_full,
                        )

                        kv_offset_y += BLOCK_N
                        accum_cnt_kv += 2
                    tile_cnt += 1

                tile_idx += num_progs

        # epilog group
        with tlx.async_task(num_warps=1, registers=NUM_REGS_EPI):
            # initialize offsets
            tile_cnt = 0
            for _ in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_z, off_h, lo, hi, seq_start, seq_len, n_targets = (
                    _compute_offsets(
                        tile_idx,
                        H,
                        num_pid_n,
                        num_pid_in_group,
                        seq_offsets,
                        num_targets,
                        max_attn_len,
                        full_attn_size,
                        contextual_seq_len,
                        BLOCK_M,
                        BLOCK_N,
                        GROUP_SIZE_N,
                        HAS_NUM_TARGETS,
                        HAS_MAX_ATTN_LEN,
                        HAS_FULL_ATTN_SIZE,
                        HAS_CONTEXTUAL_SEQ_LEN,
                    )
                )
                if start_m < seq_len:
                    _, phase = _get_bufidx_phase(tile_cnt, 1)
                    qo_offset_y = seq_start + start_m

                    out_offset = off_h.to(tl.int64) * stride_oh
                    end_o = seq_start + seq_len
                    # keeping output as device TMA even for host TMA enabled for jagged
                    o_desc = tl.make_tensor_descriptor(
                        Out,
                        shape=[end_o.to(tl.int32), DimV * H],
                        strides=[DimV * H, 1],
                        block_shape=[BLOCK_M_SPLIT, DimV],
                    )

                    for cid in tl.static_range(0, NUM_MMA_GROUPS):
                        tlx.barrier_wait(o_fulls[cid], phase)
                        tlx.fence_async_shared()
                        qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                        tlx.async_descriptor_store(
                            o_desc,
                            o_tiles[cid],
                            [
                                qo_offset_y_split.to(tl.int32),
                                (out_offset).to(tl.int32),
                            ],
                        )
                        tlx.async_descriptor_store_wait(0)
                        tlx.barrier_arrive(o_empties[cid])
                    tile_cnt += 1

                tile_idx += num_progs


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,  #
    Delta,  #
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


@triton.jit
def _hstu_attn_bwd_preprocess(
    O,
    DO,  #
    Delta,  #
    seq_offsets,
    stride_oh,
    stride_doh,
    stride_deltah,
    H,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
):
    off_z = tl.program_id(1) // H
    off_h = tl.program_id(1) % H

    seq_start = tl.load(seq_offsets + off_z)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)

    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, HEAD_DIM)
    mask_m = off_m < seq_len

    o_offset = (
        (seq_start + off_m[:, None]) * H * HEAD_DIM + off_h * stride_oh + off_d[None, :]
    )
    do_offset = (
        (seq_start + off_m[:, None]) * H * HEAD_DIM
        + off_h * stride_doh
        + off_d[None, :]
    )

    # load
    o = tl.load(O + o_offset, mask=mask_m[:, None], other=0.0)
    do = tl.load(DO + do_offset, mask=mask_m[:, None], other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    # write-back: Delta has shape [total_seq_len, H]
    delta_offset = (seq_start + off_m) * H + off_h
    tl.store(Delta + delta_offset, delta, mask=mask_m)


@triton.jit
def _hstu_bwd_calculate_offsets(
    tile_idx,
    n_tile_num,
    num_pid_m,
    seq_offsets,
    num_targets,
    max_attn_len,
    full_attn_size,
    contextual_seq_len,
    H,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    STAGE: tl.constexpr,
):
    """Calculate offsets for HSTU backward pass with jagged sequences.

    Block Range Computation for Semi-local + Global Attention
    ==========================================================

    For K/V block at start_n:
    =========================

                        max_attn_len
                    |<-------------->|
                    |                |
        |---------[K/V]--------------|--------|------------------------|
        0        start_n           high1     low2                   seq_len
                   ↑                 ↑         ↑                       ↑
                  low1               |         |                       |
                   |                 |         |                      high2
                   |<-- PART 1 ----->|  (gap)  |<------ PART 2 ------->|
                   |    MASKED       |  skip   |       NO MASK         |
                   |                 |         |                       |
                   |  semi-local     |         | global UIH + targets  |
                   |  + causal       |         | (all see everything)  |


        Boundaries:
        -----------
        uih_len = seq_len - n_targets                     # End of UIH
        low1    = start_n                                 # Causal start
        high1   = start_n + block_n + max_attn_len        # End of semi-local window (+ BLOCK_M1)
        low2    = uih_len - full_attn_size                # Start of global attention in UIH
        high2   = seq_len                                 # END OF FULL SEQUENCE (includes targets!)


    ==========================================================================
        PART 1: MASKED (semi-local + causal)
    ==========================================================================

        Range: [low1, high1) = [start_n, min(start_n + max_attn_len + BLOCK_M1, low2))

        - Q positions within semi-local window
        - Need element-wise mask check


    ==========================================================================
        PART 2: NO MASK (global UIH + all targets)
    ==========================================================================

        Range: [low2, seq_len) = [uih_len - full_attn_size, seq_len)

        Includes TWO regions that both have full attention:

          a) Global UIH zone: [uih_len - full_attn_size, uih_len)
             - Last full_attn_size positions of UIH
             - Can see ALL of UIH

          b) All targets: [uih_len, seq_len)
             - All n_targets positions
             - Targets attend to ENTIRE sequence

        → Both can see K/V at start_n without any mask!

    Returns:
        off_z: batch index
        off_h: head index
        seq_start: start offset of this sequence in jagged tensor
        seq_len: length of this sequence
        n_targets: number of target positions
        start_n: K/V block start position
        low1: start of PART 1 (masked region)
        high1: end of PART 1 (masked region), also start of gap
        low2: start of PART 2 (unmasked region)
        high2: end of PART 2 (unmasked region) = seq_len
        num_steps_masked: number of BLOCK_M1 steps in PART 1
        num_steps_unmasked: number of BLOCK_M1 steps in PART 2
    """
    bhid = tile_idx // n_tile_num
    pid = tile_idx % n_tile_num
    pid, bhid = tl.swizzle2d(pid, bhid, n_tile_num, num_pid_m, GROUP_SIZE_M)

    off_z = bhid // H
    off_h = bhid % H

    # Load sequence offsets
    seq_start = tl.load(seq_offsets + off_z)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)

    start_n = pid * BLOCK_N1

    n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
    uih_len = forward_uih_common_preprocess(n_targets, seq_len, HAS_NUM_TARGETS)

    # PART 1: Masked region (semi-local + causal)
    low1 = start_n

    causal_end = start_n + BLOCK_N1

    if HAS_MAX_ATTN_LEN:
        # Semi-local end: Q at position m can see K at position n if n >= m - max_attn_len
        # For K/V block [start_n, start_n + BLOCK_N1), the last K is at start_n + BLOCK_N1 - 1
        # Q can see this last K up to position (start_n + BLOCK_N1 - 1) + max_attn_len
        semi_local_end = start_n + BLOCK_N1 + max_attn_len
        # high1 = max(causal_end, semi_local_end)
        high1 = semi_local_end if semi_local_end > causal_end else causal_end
        # Round up to block boundary
        high1 = low1 + tl.cdiv(high1 - low1, BLOCK_M1) * BLOCK_M1
    else:
        # No semi-local constraint: only causal masking needed in the diagonal block
        # PART 1 covers [start_n, start_n + BLOCK_N1) where causal mask is needed
        # Round up to block boundary
        high1 = low1 + tl.cdiv(causal_end - low1, BLOCK_M1) * BLOCK_M1

    high1 = high1 if high1 < seq_len else seq_len

    # PART 2: Unmasked region
    # This region has no masking because:
    #   a) For positions beyond the diagonal block (>= start_n + BLOCK_N1), all K/V in the
    #      block are causally visible (no causal mask needed)
    #   b) Global UIH zone: last full_attn_size positions of UIH can see ALL of UIH
    #   c) All targets: targets attend to ENTIRE sequence
    #
    # low2 = high1 (continue from where PART 1 ended)
    # high2 = seq_len (extends to end of sequence)

    if HAS_MAX_ATTN_LEN:
        # With max_attn_len, PART 1 already covers the semi-local window
        # PART 2 only needs to cover global attention zone and targets
        if HAS_FULL_ATTN_SIZE or HAS_NUM_TARGETS:
            if HAS_FULL_ATTN_SIZE:
                # Start of global attention zone in UIH
                low2 = uih_len - full_attn_size
            else:
                # No global attention, but targets still have no mask
                # Targets start at uih_len
                low2 = uih_len

            # Ensure low2 doesn't overlap with PART 1 (must be >= high1)
            low2 = low2 if low2 >= high1 else high1
        else:
            # No global attention and no targets, skip PART 2
            low2 = seq_len
    else:
        # No max_attn_len: PART 2 must cover ALL Q positions from high1 to seq_len
        # since there's no semi-local constraint limiting the range
        low2 = high1

    # high2 = end of sequence (includes all targets)
    high2 = seq_len

    return (
        off_z,
        off_h,
        seq_start,
        seq_len,
        n_targets,
        start_n,
        low1,
        high1,
        low2,
        high2,
    )


@triton.jit
def _hstu_bwd_load_q_do(
    desc_q,
    desc_do,
    q_tiles,
    do_tiles,
    q_empties,
    q_fulls,
    do_empties,
    do_fulls,
    seq_start,
    off_h,
    stride_qh,
    stride_doh,
    low,
    high,
    blk_idx,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Load Q and dO blocks for backward pass inner loop.

    This function loads Q and dO blocks for a range of M positions [low, high).

    Args:
        desc_q: TMA descriptor for Q tensor
        desc_do: TMA descriptor for dO tensor
        q_tiles: Local tiles for Q
        do_tiles: Local tiles for dO
        q_empties: Barrier for Q empty slots
        q_fulls: Barrier for Q full slots
        do_empties: Barrier for dO empty slots
        do_fulls: Barrier for dO full slots
        seq_start: Start offset of this sequence in jagged tensor
        off_h: Head index
        stride_qh: Stride for Q head dimension
        stride_doh: Stride for dO head dimension
        low: Start of M range to load
        high: End of M range to load (exclusive)
        blk_idx: Current block index for buffer management
        NUM_BUFFERS_Q: Number of Q buffers
        NUM_BUFFERS_DO: Number of dO buffers
        BLOCK_M1: Block size for M dimension
        HEAD_DIM: Head dimension size

    Returns:
        blk_idx: Updated block index
    """
    for curr_m in tl.range(low, high, BLOCK_M1):
        q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
        do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
        # Load Q
        q_empty = tlx.local_view(q_empties, q_buf_id)
        q_full = tlx.local_view(q_fulls, q_buf_id)
        q_tile = tlx.local_view(q_tiles, q_buf_id)
        tlx.barrier_wait(q_empty, q_phase ^ 1)
        tlx.barrier_expect_bytes(q_full, 2 * BLOCK_M1 * HEAD_DIM)
        tlx.async_descriptor_load(
            desc_q,
            q_tile,
            [
                (seq_start + curr_m).to(tl.int32),
                (off_h * stride_qh).to(tl.int32),
            ],
            q_full,
        )

        # Load dO
        do_empty = tlx.local_view(do_empties, do_buf_id)
        do_full = tlx.local_view(do_fulls, do_buf_id)
        do_tile = tlx.local_view(do_tiles, do_buf_id)
        tlx.barrier_wait(do_empty, do_phase ^ 1)
        tlx.barrier_expect_bytes(do_full, 2 * BLOCK_M1 * HEAD_DIM)
        tlx.async_descriptor_load(
            desc_do,
            do_tile,
            [
                (seq_start + curr_m).to(tl.int32),
                (off_h * stride_doh).to(tl.int32),
            ],
            do_full,
        )
        blk_idx += 1

    return blk_idx


@triton.jit
def _hstu_bwd_compute_inner_loop_softmax(
    start_n,
    qk_fulls,
    qk_tiles,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_empties,
    dp_fulls,
    dp_tiles,
    ds_tiles,
    ds_fulls,
    M,
    D,
    seq_start,
    seq_len,
    off_h,
    stride_mh,
    stride_deltah,
    curr_m,
    blk_idx,
    step_m,
    do_out_dtype,
    q_out_dtype,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    STAGE: tl.constexpr,
    REUSE_DP_FOR_DQ: tl.constexpr,
):
    """Backward compute inner loop for softmax activation with jagged sequences."""
    start_block_n = start_n
    offs_n = start_block_n + tl.arange(0, BLOCK_N1)

    # Calculate loop bounds
    if STAGE == 1:
        lo, hi = start_n, start_n + BLOCK_N1
    elif STAGE == 2:
        lo, hi = start_n + BLOCK_N1, seq_len
    else:
        lo, hi = 0, seq_len

    num_steps = (hi - lo) // BLOCK_M1
    for _ in range(num_steps):
        tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
        ds_buf_id, _ = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)

        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        mask_m = offs_m < seq_len

        # Load M (logsumexp) for softmax
        m_offset = (seq_start + offs_m) * stride_mh + off_h
        m = tl.load(M + m_offset, mask=mask_m, other=0.0)

        # wait for qkT = tl.dot(k, qT)
        tlx.barrier_wait(tlx.local_view(qk_fulls, tmem_buf_id), tmem_phase)
        qkT = tlx.local_load(tlx.local_view(qk_tiles, tmem_buf_id))
        tlx.barrier_arrive(tlx.local_view(qk_empties, tmem_buf_id))

        pT = tl.math.exp2(qkT - m[None, :])
        if STAGE == 1:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)

        ppT = pT
        ppT = ppT.to(do_out_dtype)
        tlx.local_store(tlx.local_view(p_tiles, tmem_buf_id), ppT)
        tlx.barrier_arrive(tlx.local_view(p_fulls, tmem_buf_id))

        # Load D (= delta)
        delta_offset = (seq_start + offs_m) * stride_deltah + off_h
        Di = tl.load(D + delta_offset, mask=mask_m, other=0.0)

        # Wait for dpT = tl.dot(v, tl.trans(do))
        tlx.barrier_wait(tlx.local_view(dp_fulls, tmem_buf_id), tmem_phase)
        dpT = tlx.local_load(tlx.local_view(dp_tiles, tmem_buf_id))
        if not REUSE_DP_FOR_DQ:
            tlx.barrier_arrive(tlx.local_view(dp_empties, tmem_buf_id))

        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(q_out_dtype)
        tlx.local_store(tlx.local_view(ds_tiles, ds_buf_id), dsT)
        tlx.fence_async_shared()
        tlx.barrier_arrive(tlx.local_view(ds_fulls, ds_buf_id))
        curr_m += step_m
        blk_idx += 1
    return curr_m, blk_idx


@triton.jit
def _hstu_bwd_compute_inner_loop_silu(
    start_n,
    qk_fulls,
    qk_tiles,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_empties,
    dp_fulls,
    dp_tiles,
    ds_tiles,
    ds_fulls,
    seq_start,
    seq_len,
    off_h,
    alpha,
    scale,
    low,
    high,
    blk_idx,
    do_out_dtype,
    q_out_dtype,
    contextual_seq_len,
    max_attn_len,
    n_targets,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    REUSE_DP_FOR_DQ: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    APPLY_MASK: tl.constexpr,
    ALPHA_BWD_PRESCALE: tl.constexpr,
):
    """Backward compute inner loop for SiLU activation with jagged sequences.

    Args:
        start_n: K/V block start position
        low: Start of M range to process
        high: End of M range to process (exclusive)
        blk_idx: Current block index for buffer management
        APPLY_MASK: Whether to apply causal/semi-local masking in this range

    Returns:
        blk_idx: Updated block index
    """
    start_block_n = start_n
    offs_n = start_block_n + tl.arange(0, BLOCK_N1)

    # Precompute max_ids and pos_offs_n once before the loop
    if APPLY_MASK:
        max_ids, pos_offs_n = backward_off_common_preprocess(
            seq_len,
            contextual_seq_len,
            n_targets,
            offs_n,
            HAS_CONTEXTUAL_SEQ_LEN,
            HAS_NUM_TARGETS,
        )

    # Boundary mask for N dimension (for last K/V block)
    mask_n = offs_n < seq_len

    if ALPHA_BWD_PRESCALE:
        # Pre-scale alpha by 0.5, matching CUTLASS (line 382: alpha = args.alpha * 0.5f).
        # With y = alpha*0.5*QK, we store 1+tanh(y) = 2*sigmoid(2y) = 2*sigmoid(alpha*QK).
        # Then P = y*(1+tanh(y)) = silu(alpha*QK), and the 2x and 0.5x cancel.
        alpha = alpha * 0.5
        half_scale = scale * 0.5

    for curr_m in tl.range(low, high, BLOCK_M1):
        tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
        ds_buf_id, _ = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)

        offs_m = curr_m + tl.arange(0, BLOCK_M1)

        # wait for qkT = tl.dot(k, qT)
        tlx.barrier_wait(tlx.local_view(qk_fulls, tmem_buf_id), tmem_phase)
        qkT = tlx.local_load(tlx.local_view(qk_tiles, tmem_buf_id))
        tlx.barrier_arrive(tlx.local_view(qk_empties, tmem_buf_id))

        # Boundary mask for M dimension (for last Q/dO block)
        mask_m = offs_m < seq_len

        if APPLY_MASK:
            # Compute valid mask for HSTU attention (masked region)
            valid_mask_trans = backward_valid_mask(
                offs_m,
                # pyre-fixme[61]: `pos_offs_n` is defined when APPLY_MASK is True
                pos_offs_n,
                offs_n,
                # pyre-fixme[61]: `max_ids` is defined when APPLY_MASK is True
                max_ids,
                contextual_seq_len,
                max_attn_len,
                HAS_CONTEXTUAL_SEQ_LEN,
                HAS_NUM_TARGETS,
                HAS_MAX_ATTN_LEN,
            )
            # Also apply boundary mask for last K/V and Q/dO blocks
            boundary_mask_trans = mask_n[:, None] & mask_m[None, :]
            valid_mask_trans = valid_mask_trans & boundary_mask_trans

            if ALPHA_BWD_PRESCALE:
                # CUTLASS-style: scale ALL, compute 1+tanh (=2*sigmoid), then zero for masked.
                qkT_scaled = qkT * alpha
                one_plus_tanh = _fma_f32x2(tanh_approx_fp32(qkT_scaled), 1.0, 1.0)
                one_plus_tanh = tl.where(valid_mask_trans, one_plus_tanh, 0.0)
            else:
                masked_alpha = tl.where(valid_mask_trans, alpha, 0.0)
                qkT_scaled = qkT * masked_alpha
        else:
            # Unmasked region: only need boundary mask for last Q/dO and K/V blocks
            # Interior tiles (all elements in bounds): skip masking entirely
            is_interior = (
                curr_m + BLOCK_M1 <= seq_len and start_block_n + BLOCK_N1 <= seq_len
            )
            mask_m = offs_m < seq_len
            qkT_scaled = qkT * alpha
            if ALPHA_BWD_PRESCALE:
                one_plus_tanh = _fma_f32x2(tanh_approx_fp32(qkT_scaled), 1.0, 1.0)
                if not is_interior:
                    boundary_mask_trans = mask_n[:, None] & mask_m[None, :]
                    one_plus_tanh = tl.where(boundary_mask_trans, one_plus_tanh, 0.0)
            else:
                if not is_interior:
                    boundary_mask_trans = mask_n[:, None] & mask_m[None, :]
                    masked_alpha = tl.where(boundary_mask_trans, alpha, 0.0)
                    qkT_scaled = qkT * masked_alpha

        if ALPHA_BWD_PRESCALE:
            # P = y*(1+tanh(y))*scale = silu(alpha*QK)*scale
            # pyre-fixme[61]: `one_plus_tanh` is undefined, or not always defined.
            silu_trans = qkT_scaled * one_plus_tanh * scale
        else:
            sig_trans = fast_silu(qkT_scaled, MULT_BY_X=False)
            silu_trans = qkT_scaled * sig_trans * scale

        # pT = silu activation value
        pT = silu_trans
        ppT = pT.to(do_out_dtype)
        tlx.local_store(tlx.local_view(p_tiles, tmem_buf_id), ppT)
        tlx.barrier_arrive(tlx.local_view(p_fulls, tmem_buf_id))

        # Wait for dpT = tl.dot(v, tl.trans(do))
        tlx.barrier_wait(tlx.local_view(dp_fulls, tmem_buf_id), tmem_phase)
        dpT = tlx.local_load(tlx.local_view(dp_tiles, tmem_buf_id))
        if not REUSE_DP_FOR_DQ:
            tlx.barrier_arrive(tlx.local_view(dp_empties, tmem_buf_id))

        # Derivative computation
        if ALPHA_BWD_PRESCALE:
            # Derivative of silu(x) = x/2*(1+tanh(x/2)) with y = x/2 = qkT_scaled:
            # d/dx[silu(x)] = (1+tanh(y))/2 * (1 + y*(1-tanh(y)))
            # dsT = dpT * (1+tanh(y)) * (1 + y*(1-tanh(y))) * scale * 0.5
            # The 0.5 is folded into half_scale.
            # Masked elements have one_plus_tanh=0, so dsT=0 automatically.
            # Derive 1-tanh from 1+tanh: (1-tanh) = 2-(1+tanh), avoiding keeping tanh_y live.
            # pyre-fixme[61]: `one_plus_tanh` is undefined, or not always defined.
            one_minus_tanh = _fma_f32x2(one_plus_tanh, -1.0, 2.0)
            dsT = (
                dpT
                # pyre-fixme[61]: `one_plus_tanh` is undefined, or not always defined.
                * one_plus_tanh
                * _fma_f32x2(qkT_scaled, one_minus_tanh, 1.0)
                # pyre-fixme[61]: `half_scale` is undefined, or not always defined.
                * half_scale
            )
        else:
            # Old path: dsT = dpT * sig * (1 + x*(1-sig)) * scale
            # pyre-fixme[61]: `sig_trans` is undefined, or not always defined.
            one_minus_sig = _fma_f32x2(sig_trans, -1.0, 1.0)
            # pyre-fixme[61]: `sig_trans` is undefined, or not always defined.
            dsT = dpT * sig_trans * _fma_f32x2(qkT_scaled, one_minus_sig, 1.0) * scale
            if APPLY_MASK:
                # pyre-fixme[61]: `valid_mask_trans` is undefined, or not always
                #  defined.
                dsT = tl.where(valid_mask_trans, dsT, 0.0)
            else:
                # pyre-fixme[61]: `is_interior` is undefined, or not always defined.
                if not is_interior:
                    # pyre-fixme[61]: `boundary_mask_trans` is undefined, or not
                    #  always defined.
                    dsT = tl.where(boundary_mask_trans, dsT, 0.0)
        dsT = dsT.to(q_out_dtype)
        tlx.local_store(tlx.local_view(ds_tiles, ds_buf_id), dsT)
        tlx.fence_async_shared()
        tlx.barrier_arrive(tlx.local_view(ds_fulls, ds_buf_id))
        blk_idx += 1

    return blk_idx


@triton.jit
def _hstu_bwd_reduction_accumulate_dq(
    desc_dq,
    dq_fulls,
    dq_tiles,
    dq_empties,
    seq_start,
    off_h,
    stride_dqh,
    curr_m,
    blk_idx,
    sm_scale,
    NUM_BUFFERS_TMEM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):
    """Reduction group helper for SiLU: accumulate dQ with atomic add.

    Scales by sm_scale (alpha) for SiLU backward.
    """
    tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

    # wait for dq = tl.dot(tl.trans(dsT), k)
    tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
    slice_size: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
    for slice_id in tl.static_range(EPILOGUE_SUBTILE):
        dq_slice = tlx.local_slice(
            dq_tiles[tmem_buf_id],
            [0, slice_id * slice_size],
            [BLOCK_M1, slice_size],
        )
        dq = tlx.local_load(dq_slice)
        dq = dq * sm_scale
        desc_dq.atomic_add(
            [
                (seq_start + curr_m).to(tl.int32),
                (off_h * stride_dqh + slice_id * slice_size).to(tl.int32),
            ],
            dq,
        )

    # release dq
    tlx.barrier_arrive(dq_empties[tmem_buf_id])
    blk_idx += 1

    return blk_idx


@triton.jit
def bwd_calculate_offsets(
    tile_idx,
    n_tile_num,
    num_pid_m,
    stride_z,
    stride_h,
    stride_tok,
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    STAGE: tl.constexpr,
):
    bhid = tile_idx // n_tile_num
    pid = tile_idx % n_tile_num
    pid, bhid = tl.swizzle2d(pid, bhid, n_tile_num, num_pid_m, GROUP_SIZE_M)
    off_chz = (bhid * N_CTX).to(tl.int64)
    off_bh = (
        (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    ) // stride_tok
    start_n = pid
    start_m = _get_start_m_bwd(start_n, BLOCK_N1, STAGE)
    num_steps = (N_CTX - start_m) // BLOCK_M1
    return off_chz, off_bh, start_m, start_n, num_steps


@triton.jit
def _get_start_m_bwd(start_n, BLOCK_N1, STAGE: tl.constexpr):
    if STAGE == 1:
        return 0
    else:
        tl.static_assert(STAGE == 3)
        return start_n * BLOCK_N1


@triton.jit
def _get_unfused_bwd_loop_bounds(start_n, N_CTX, BLOCK_N1, STAGE: tl.constexpr):
    if STAGE == 1:
        # First part of STAGE == 3
        lo, hi = start_n * BLOCK_N1, (start_n + 1) * BLOCK_N1
    elif STAGE == 2:
        # Second part of STAGE == 3 in this function
        lo, hi = (start_n + 1) * BLOCK_N1, N_CTX
    else:
        tl.static_assert(STAGE == 3)
        lo, hi = 0, N_CTX
    return lo, hi


@triton.jit
def _bwd_compute_inner_loop(
    start_n,
    qk_fulls,
    qk_tiles,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_empties,
    dp_fulls,
    dp_tiles,
    ds_tiles,
    ds_fulls,
    M,
    D,
    curr_m,
    blk_idx,
    step_m,
    do_out_dtype,
    q_out_dtype,
    N_CTX,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    STAGE: tl.constexpr,
    REUSE_DP_FOR_DQ: tl.constexpr,
):
    start_block_n = start_n * BLOCK_N1
    offs_n = start_block_n + tl.arange(0, BLOCK_N1)
    lo, hi = _get_unfused_bwd_loop_bounds(start_n, N_CTX, BLOCK_N1, STAGE)
    num_steps = (hi - lo) // BLOCK_M1
    for _ in range(num_steps):
        tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)
        ds_buf_id, _ = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DS)

        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)

        # wait for qkT = tl.dot(k, qT)
        tlx.barrier_wait(tlx.local_view(qk_fulls, tmem_buf_id), tmem_phase)
        qkT = tlx.local_load(tlx.local_view(qk_tiles, tmem_buf_id))
        tlx.barrier_arrive(tlx.local_view(qk_empties, tmem_buf_id))

        pT = tl.math.exp2(qkT - m[None, :])
        if STAGE == 1:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)

        # ppT *= qk_scale
        ppT = pT
        ppT = ppT.to(do_out_dtype)
        tlx.local_store(tlx.local_view(p_tiles, tmem_buf_id), ppT)
        tlx.barrier_arrive(tlx.local_view(p_fulls, tmem_buf_id))

        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)

        # Wait for dpT = tl.dot(v, tl.trans(do))
        tlx.barrier_wait(tlx.local_view(dp_fulls, tmem_buf_id), tmem_phase)
        dpT = tlx.local_load(tlx.local_view(dp_tiles, tmem_buf_id))
        # We can only signal the arrive if DP is not shared with DQ.
        # Otherwise we need to wait for DQ to be done.
        if not REUSE_DP_FOR_DQ:
            tlx.barrier_arrive(tlx.local_view(dp_empties, tmem_buf_id))
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(q_out_dtype)
        tlx.local_store(tlx.local_view(ds_tiles, ds_buf_id), dsT)
        tlx.fence_async_shared()
        tlx.barrier_arrive(tlx.local_view(ds_fulls, ds_buf_id))
        curr_m += step_m
        blk_idx += 1
    return curr_m, blk_idx


def _hstu_bwd_host_descriptor_pre_hook(nargs):
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    HEAD_DIM = nargs["HEAD_DIM"]
    EPILOGUE_SUBTILE = nargs["EPILOGUE_SUBTILE"]
    nargs["DQ"].zero_()

    nargs["desc_q"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_do"].block_shape = [BLOCK_M1, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N1, HEAD_DIM]
    DQ_REDUCE_FACTOR = nargs["DQ_REDUCE_FACTOR"]
    nargs["desc_dq"].block_shape = [
        BLOCK_M1,
        HEAD_DIM // (EPILOGUE_SUBTILE * DQ_REDUCE_FACTOR),
    ]
    nargs["desc_dv"].block_shape = [BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE]
    nargs["desc_dk"].block_shape = [BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE]


def get_hstu_bwd_configs() -> List[triton.Config]:
    return [
        triton.Config(
            {
                "BLOCK_M1": 128,
                "BLOCK_N1": 128,
                "BLOCK_M2": 128,
                "BLOCK_N2": 128,
                "NUM_BUFFERS_KV": 1,
                "NUM_BUFFERS_Q": 2,
                "NUM_BUFFERS_DO": 1,
                "NUM_BUFFERS_DS": 1,
                "NUM_BUFFERS_TMEM": 1,
                "EPILOGUE_SUBTILE": 2,
                "DQ_REDUCE_FACTOR": 2,
                "DQ_REDUCE_STAGES": 2,
                "EARLY_RELEASE_SUBTILES": er,
                "ALPHA_BWD_PRESCALE": True,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=_hstu_bwd_host_descriptor_pre_hook,
        )
        for er in [1, 2]
    ] + [
        triton.Config(
            {
                "BLOCK_M1": 64,
                "BLOCK_N1": 128,
                "BLOCK_M2": 128,
                "BLOCK_N2": 128,
                "NUM_BUFFERS_KV": 1,
                "NUM_BUFFERS_Q": 2,
                "NUM_BUFFERS_DO": 1,
                "NUM_BUFFERS_DS": 1,
                "NUM_BUFFERS_TMEM": 1,
                "EPILOGUE_SUBTILE": 2,
                "DQ_REDUCE_FACTOR": 2,
                "DQ_REDUCE_STAGES": 2,
                "EARLY_RELEASE_SUBTILES": er,
                "ALPHA_BWD_PRESCALE": True,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=_hstu_bwd_host_descriptor_pre_hook,
        )
        for er in [1, 2]
    ]


@triton.autotune(
    configs=get_hstu_bwd_configs(), key=["AUTOTUNE_MAX_SEQ_LEN", "HEAD_DIM"]
)
@triton.jit
def _hstu_attn_bwd_ws(
    DQ,
    DK,
    DV,
    desc_q,
    desc_k,
    desc_v,
    sm_scale,  # alpha
    desc_do,  #
    desc_dq,
    desc_dk,
    desc_dv,  #
    M,
    D,
    seq_offsets,
    attn_scale,
    num_targets,
    max_attn_len,
    full_attn_size,
    contextual_seq_len,
    # shared by Q/K/V/DO.
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    stride_mh,
    stride_deltah,
    H,
    Z,
    max_seq_len,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    DQ_REDUCE_FACTOR: tl.constexpr,
    DQ_REDUCE_STAGES: tl.constexpr,
    EARLY_RELEASE_SUBTILES: tl.constexpr,
    ALPHA_BWD_PRESCALE: tl.constexpr,
    STAGE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SOFTMAX: tl.constexpr,
    AUTOTUNE_MAX_SEQ_LEN: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
):
    """HSTU backward pass kernel with jagged sequences and SiLU/Softmax activation."""
    # Kernel hangs if NUM_BUFFERS_Q != 2.
    tl.static_assert(NUM_BUFFERS_Q == 2)
    # Runtime error if NUM_BUFFERS_DO != 1
    tl.static_assert(NUM_BUFFERS_DO == 1)
    # TODO: Add support for contextual seq len
    tl.static_assert(not HAS_CONTEXTUAL_SEQ_LEN)
    tl.static_assert(
        EARLY_RELEASE_SUBTILES >= 1 and EARLY_RELEASE_SUBTILES <= 2,
        "EARLY_RELEASE_SUBTILES must be 1 or 2",
    )

    REUSE_DP_FOR_DQ: tl.constexpr = (BLOCK_M1 == 128) and (HEAD_DIM == 128)

    n_tile_num = tl.cdiv(max_seq_len, BLOCK_N1)
    num_pid_m = Z * H
    tile_idx = tl.program_id(0)

    # allocate smem buffers
    k_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV
    )
    v_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS_KV
    )
    q_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    do_tiles = tlx.local_alloc(
        (BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_do), NUM_BUFFERS_DO
    )

    # Use SMEM for dsT
    ds_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1), tlx.dtype_of(desc_q), NUM_BUFFERS_DS
    )

    # SMEM staging buffer for double-buffered dQ TMA reduce-add (from D97693127)
    DQ_REDUCE_NCOL: tl.constexpr = HEAD_DIM // (EPILOGUE_SUBTILE * DQ_REDUCE_FACTOR)
    DQ_REDUCE_ITERS: tl.constexpr = HEAD_DIM // DQ_REDUCE_NCOL
    dq_store_buf = tlx.local_alloc(
        (BLOCK_M1, DQ_REDUCE_NCOL), tlx.dtype_of(desc_dq), DQ_REDUCE_STAGES
    )

    # SMEM staging buffers for dKV async TMA store (from D99341217)
    # sdv reuses v_tiles (free after dv_fulls; MMA's last v_tiles read precedes dv_fulls)
    # sdk reuses k_tiles (MMA's dq dot still reads k_tiles after dk_fulls,
    #   so compute waits on k_mma_done before writing sdk)
    DKV_STORE_NCOL: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
    DKV_STORE_ITERS: tl.constexpr = HEAD_DIM // DKV_STORE_NCOL
    sdv_store_buf = tlx.local_alloc(
        (BLOCK_N1, DKV_STORE_NCOL), tlx.dtype_of(desc_dv), NUM_BUFFERS_KV, reuse=v_tiles
    )
    sdk_store_buf = tlx.local_alloc(
        (BLOCK_N1, DKV_STORE_NCOL), tlx.dtype_of(desc_dk), NUM_BUFFERS_KV, reuse=k_tiles
    )

    # allocate barriers for smem buffers
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_mma_done = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    ds_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    # allocate tmem buffers
    qk_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem
    )
    p_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_do),
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    dp_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
    )

    dv_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem
    )
    dk_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem
    )

    # allocate barriers for tmem buffers
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    if REUSE_DP_FOR_DQ:
        dq_tiles = tlx.local_alloc(
            (BLOCK_M1, HEAD_DIM),
            tl.float32,
            NUM_BUFFERS_TMEM,
            tlx.storage_kind.tmem,
            reuse=dp_tiles,
        )
        dp_empties = dq_empties
    else:
        dq_tiles = tlx.local_alloc(
            (BLOCK_M1, HEAD_DIM),
            tl.float32,
            NUM_BUFFERS_TMEM,
            tlx.storage_kind.tmem,
        )
        dp_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    clc_context = tlx.clc_create_context(num_consumers=4)

    with tlx.async_tasks():
        # reduction group: accumulate dq
        with tlx.async_task("default"):
            LN2 = 0.6931471824645996  # = ln(2)
            blk_idx = 0
            clc_phase_producer = 1
            clc_phase_consumer = 0
            while tile_idx != -1:
                tlx.clc_producer(clc_context, clc_phase_producer)
                clc_phase_producer ^= 1
                (
                    off_z,
                    off_h,
                    seq_start,
                    seq_len,
                    n_targets,
                    start_n,
                    low1,
                    high1,
                    low2,
                    high2,
                ) = _hstu_bwd_calculate_offsets(
                    tile_idx,
                    n_tile_num,
                    num_pid_m,
                    seq_offsets,
                    num_targets,
                    max_attn_len,
                    full_attn_size,
                    contextual_seq_len,
                    H,
                    BLOCK_M1,
                    BLOCK_N1,
                    GROUP_SIZE_M,
                    HAS_NUM_TARGETS,
                    HAS_MAX_ATTN_LEN,
                    HAS_FULL_ATTN_SIZE,
                    HAS_CONTEXTUAL_SEQ_LEN,
                    STAGE,
                )
                if start_n < seq_len:
                    # if SOFTMAX:
                    #     # Original softmax reduction: single fused loop
                    #     num_steps_masked = (high1 - low1) // BLOCK_M1
                    #     num_steps_unmasked = (high2 - low2) // BLOCK_M1
                    #     num_steps = num_steps_masked + num_steps_unmasked
                    #     curr_m = low1
                    #     step_m = BLOCK_M1
                    #     for _ in range(num_steps):
                    #         tmem_buf_id, tmem_phase = _get_bufidx_phase(
                    #             blk_idx, NUM_BUFFERS_TMEM
                    #         )

                    #         # wait for dq = tl.dot(tl.trans(dsT), k)
                    #         tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                    #         slice_size: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
                    #         for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                    #             dq_slice = tlx.local_slice(
                    #                 dq_tiles[tmem_buf_id],
                    #                 [0, slice_id * slice_size],
                    #                 [BLOCK_M1, slice_size],
                    #             )
                    #             dq = tlx.local_load(dq_slice)
                    #             dq = dq * LN2
                    #             desc_dq.atomic_add(
                    #                 [
                    #                     (seq_start + curr_m).to(tl.int32),
                    #                     (off_h * stride_dqh + slice_id * slice_size).to(
                    #                         tl.int32
                    #                     ),
                    #                 ],
                    #                 dq,
                    #             )

                    #         # release dq
                    #         tlx.barrier_arrive(dq_empties[tmem_buf_id])
                    #         # Increment pointers.
                    #         curr_m += step_m
                    #         blk_idx += 1
                    # else:
                    # SiLU reduction: two loops for PART 1 and PART 2
                    # INLINED to avoid TLX region isolation issues with helper functions
                    # dQ early TMEM release optimization (from D98825322):
                    # Pre-load last EARLY_RELEASE_SUBTILES from TMEM into registers,
                    # release TMEM early, then process from registers.
                    # PART 1: Masked region [low1, high1)
                    for curr_m in tl.range(low1, high1, BLOCK_M1):
                        tmem_buf_id, tmem_phase = _get_bufidx_phase(
                            blk_idx, NUM_BUFFERS_TMEM
                        )
                        tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                        for slice_id in tl.static_range(
                            DQ_REDUCE_ITERS - EARLY_RELEASE_SUBTILES
                        ):
                            dq_smem_idx = slice_id % DQ_REDUCE_STAGES
                            dq_slice = tlx.local_slice(
                                dq_tiles[tmem_buf_id],
                                [0, slice_id * DQ_REDUCE_NCOL],
                                [BLOCK_M1, DQ_REDUCE_NCOL],
                            )
                            dq = tlx.local_load(dq_slice)
                            dq = dq * sm_scale
                            tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                            tlx.local_store(
                                dq_store_buf[dq_smem_idx],
                                dq.to(tlx.dtype_of(desc_dq)),
                            )
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_dq,
                                dq_store_buf[dq_smem_idx],
                                [
                                    (seq_start + curr_m).to(tl.int32),
                                    (slice_id * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                        tl.int32
                                    ),
                                ],
                                store_reduce="add",
                            )
                        if EARLY_RELEASE_SUBTILES == 1:
                            er0: tl.constexpr = DQ_REDUCE_ITERS - 1
                            dq_er0 = tlx.local_load(
                                tlx.local_slice(
                                    dq_tiles[tmem_buf_id],
                                    [0, er0 * DQ_REDUCE_NCOL],
                                    [BLOCK_M1, DQ_REDUCE_NCOL],
                                )
                            )
                            tlx.barrier_arrive(dq_empties[tmem_buf_id])
                            dq_er0 = dq_er0 * sm_scale
                            tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                            tlx.local_store(
                                dq_store_buf[er0 % DQ_REDUCE_STAGES],
                                dq_er0.to(tlx.dtype_of(desc_dq)),
                            )
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_dq,
                                dq_store_buf[er0 % DQ_REDUCE_STAGES],
                                [
                                    (seq_start + curr_m).to(tl.int32),
                                    (er0 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                        tl.int32
                                    ),
                                ],
                                store_reduce="add",
                            )
                        elif EARLY_RELEASE_SUBTILES == 2:
                            er0: tl.constexpr = DQ_REDUCE_ITERS - 2
                            er1: tl.constexpr = DQ_REDUCE_ITERS - 1
                            dq_er0 = tlx.local_load(
                                tlx.local_slice(
                                    dq_tiles[tmem_buf_id],
                                    [0, er0 * DQ_REDUCE_NCOL],
                                    [BLOCK_M1, DQ_REDUCE_NCOL],
                                )
                            )
                            dq_er1 = tlx.local_load(
                                tlx.local_slice(
                                    dq_tiles[tmem_buf_id],
                                    [0, er1 * DQ_REDUCE_NCOL],
                                    [BLOCK_M1, DQ_REDUCE_NCOL],
                                )
                            )
                            tlx.barrier_arrive(dq_empties[tmem_buf_id])
                            dq_er0 = dq_er0 * sm_scale
                            tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                            tlx.local_store(
                                dq_store_buf[er0 % DQ_REDUCE_STAGES],
                                dq_er0.to(tlx.dtype_of(desc_dq)),
                            )
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_dq,
                                dq_store_buf[er0 % DQ_REDUCE_STAGES],
                                [
                                    (seq_start + curr_m).to(tl.int32),
                                    (er0 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                        tl.int32
                                    ),
                                ],
                                store_reduce="add",
                            )
                            dq_er1 = dq_er1 * sm_scale
                            tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                            tlx.local_store(
                                dq_store_buf[er1 % DQ_REDUCE_STAGES],
                                dq_er1.to(tlx.dtype_of(desc_dq)),
                            )
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_dq,
                                dq_store_buf[er1 % DQ_REDUCE_STAGES],
                                [
                                    (seq_start + curr_m).to(tl.int32),
                                    (er1 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                        tl.int32
                                    ),
                                ],
                                store_reduce="add",
                            )
                        blk_idx += 1

                    # PART 2: Unmasked region [low2, high2)
                    for curr_m in tl.range(low2, high2, BLOCK_M1):
                        tmem_buf_id, tmem_phase = _get_bufidx_phase(
                            blk_idx, NUM_BUFFERS_TMEM
                        )
                        tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                        for slice_id in tl.static_range(
                            DQ_REDUCE_ITERS - EARLY_RELEASE_SUBTILES
                        ):
                            dq_smem_idx = slice_id % DQ_REDUCE_STAGES
                            dq_slice = tlx.local_slice(
                                dq_tiles[tmem_buf_id],
                                [0, slice_id * DQ_REDUCE_NCOL],
                                [BLOCK_M1, DQ_REDUCE_NCOL],
                            )
                            dq = tlx.local_load(dq_slice)
                            dq = dq * sm_scale
                            tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                            tlx.local_store(
                                dq_store_buf[dq_smem_idx],
                                dq.to(tlx.dtype_of(desc_dq)),
                            )
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_dq,
                                dq_store_buf[dq_smem_idx],
                                [
                                    (seq_start + curr_m).to(tl.int32),
                                    (slice_id * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                        tl.int32
                                    ),
                                ],
                                store_reduce="add",
                            )
                        if EARLY_RELEASE_SUBTILES == 1:
                            er0: tl.constexpr = DQ_REDUCE_ITERS - 1
                            dq_er0 = tlx.local_load(
                                tlx.local_slice(
                                    dq_tiles[tmem_buf_id],
                                    [0, er0 * DQ_REDUCE_NCOL],
                                    [BLOCK_M1, DQ_REDUCE_NCOL],
                                )
                            )
                            tlx.barrier_arrive(dq_empties[tmem_buf_id])
                            dq_er0 = dq_er0 * sm_scale
                            tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                            tlx.local_store(
                                dq_store_buf[er0 % DQ_REDUCE_STAGES],
                                dq_er0.to(tlx.dtype_of(desc_dq)),
                            )
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_dq,
                                dq_store_buf[er0 % DQ_REDUCE_STAGES],
                                [
                                    (seq_start + curr_m).to(tl.int32),
                                    (er0 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                        tl.int32
                                    ),
                                ],
                                store_reduce="add",
                            )
                        elif EARLY_RELEASE_SUBTILES == 2:
                            er0: tl.constexpr = DQ_REDUCE_ITERS - 2
                            er1: tl.constexpr = DQ_REDUCE_ITERS - 1
                            dq_er0 = tlx.local_load(
                                tlx.local_slice(
                                    dq_tiles[tmem_buf_id],
                                    [0, er0 * DQ_REDUCE_NCOL],
                                    [BLOCK_M1, DQ_REDUCE_NCOL],
                                )
                            )
                            dq_er1 = tlx.local_load(
                                tlx.local_slice(
                                    dq_tiles[tmem_buf_id],
                                    [0, er1 * DQ_REDUCE_NCOL],
                                    [BLOCK_M1, DQ_REDUCE_NCOL],
                                )
                            )
                            tlx.barrier_arrive(dq_empties[tmem_buf_id])
                            dq_er0 = dq_er0 * sm_scale
                            tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                            tlx.local_store(
                                dq_store_buf[er0 % DQ_REDUCE_STAGES],
                                dq_er0.to(tlx.dtype_of(desc_dq)),
                            )
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_dq,
                                dq_store_buf[er0 % DQ_REDUCE_STAGES],
                                [
                                    (seq_start + curr_m).to(tl.int32),
                                    (er0 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                        tl.int32
                                    ),
                                ],
                                store_reduce="add",
                            )
                            dq_er1 = dq_er1 * sm_scale
                            tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                            tlx.local_store(
                                dq_store_buf[er1 % DQ_REDUCE_STAGES],
                                dq_er1.to(tlx.dtype_of(desc_dq)),
                            )
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_dq,
                                dq_store_buf[er1 % DQ_REDUCE_STAGES],
                                [
                                    (seq_start + curr_m).to(tl.int32),
                                    (er1 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                        tl.int32
                                    ),
                                ],
                                store_reduce="add",
                            )
                        blk_idx += 1
                tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # compute group: compute softmax/silu backward
        with tlx.async_task(num_warps=8, registers=192, replicate=1):
            blk_idx = 0
            tile_cnt = 0
            clc_phase_consumer = 0
            while tile_idx != -1:
                (
                    off_z,
                    off_h,
                    seq_start,
                    seq_len,
                    n_targets,
                    start_n,
                    low1,
                    high1,
                    low2,
                    high2,
                ) = _hstu_bwd_calculate_offsets(
                    tile_idx,
                    n_tile_num,
                    num_pid_m,
                    seq_offsets,
                    num_targets,
                    max_attn_len,
                    full_attn_size,
                    contextual_seq_len,
                    H,
                    BLOCK_M1,
                    BLOCK_N1,
                    GROUP_SIZE_M,
                    HAS_NUM_TARGETS,
                    HAS_MAX_ATTN_LEN,
                    HAS_FULL_ATTN_SIZE,
                    HAS_CONTEXTUAL_SEQ_LEN,
                    STAGE,
                )
                if start_n < seq_len:
                    start_block_n = start_n

                    curr_m = low1
                    do_out_dtype = tlx.dtype_of(desc_do)
                    q_out_dtype = tlx.dtype_of(desc_q)

                    # SiLU activation backward
                    scale = tl.load(attn_scale).to(tl.float32)
                    # PART 1: Masked region [low1, high1)
                    blk_idx = _hstu_bwd_compute_inner_loop_silu(
                        start_n,
                        qk_fulls,
                        qk_tiles,
                        qk_empties,
                        p_tiles,
                        p_fulls,
                        dp_empties,
                        dp_fulls,
                        dp_tiles,
                        ds_tiles,
                        ds_fulls,
                        seq_start,
                        seq_len,
                        off_h,
                        sm_scale,  # alpha
                        scale,
                        low1,
                        high1,
                        blk_idx,
                        do_out_dtype,
                        q_out_dtype,
                        contextual_seq_len,
                        max_attn_len,
                        n_targets,
                        NUM_BUFFERS_TMEM,
                        NUM_BUFFERS_DS,
                        BLOCK_M1,
                        BLOCK_N1,
                        REUSE_DP_FOR_DQ=REUSE_DP_FOR_DQ,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                        APPLY_MASK=True,
                        ALPHA_BWD_PRESCALE=ALPHA_BWD_PRESCALE,
                    )
                    # PART 2: Unmasked region [low2, high2)
                    blk_idx = _hstu_bwd_compute_inner_loop_silu(
                        start_n,
                        qk_fulls,
                        qk_tiles,
                        qk_empties,
                        p_tiles,
                        p_fulls,
                        dp_empties,
                        dp_fulls,
                        dp_tiles,
                        ds_tiles,
                        ds_fulls,
                        seq_start,
                        seq_len,
                        off_h,
                        sm_scale,  # alpha
                        scale,
                        low2,
                        high2,
                        blk_idx,
                        do_out_dtype,
                        q_out_dtype,
                        contextual_seq_len,
                        max_attn_len,
                        n_targets,
                        NUM_BUFFERS_TMEM,
                        NUM_BUFFERS_DS,
                        BLOCK_M1,
                        BLOCK_N1,
                        REUSE_DP_FOR_DQ=REUSE_DP_FOR_DQ,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                        APPLY_MASK=False,
                        ALPHA_BWD_PRESCALE=ALPHA_BWD_PRESCALE,
                    )

                    # epilogue: dKV via SMEM staging + async TMA store (D99341217)
                    kv_buf_id, kv_phase = _get_bufidx_phase(tile_cnt, NUM_BUFFERS_KV)
                    tile_cnt += 1

                    # Create on-device TMA descriptors for jagged sequence writes
                    end_kv = seq_start + seq_len
                    dv_desc = tl.make_tensor_descriptor(
                        DV,
                        shape=[end_kv.to(tl.int32), stride_dvh * H],
                        strides=[stride_dvn, 1],
                        block_shape=[BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE],
                    )
                    dk_desc = tl.make_tensor_descriptor(
                        DK,
                        shape=[end_kv.to(tl.int32), stride_dkh * H],
                        strides=[stride_dkn, 1],
                        block_shape=[BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE],
                    )

                    # dV: TMEM → regs → SMEM staging → async TMA store
                    tlx.barrier_wait(dv_fulls[kv_buf_id], kv_phase)
                    for slice_id in tl.static_range(DKV_STORE_ITERS):
                        dv_slice = tlx.local_slice(
                            dv_tiles[kv_buf_id],
                            [0, slice_id * DKV_STORE_NCOL],
                            [BLOCK_N1, DKV_STORE_NCOL],
                        )
                        dv = tlx.local_load(dv_slice)
                        tlx.async_descriptor_store_wait(0)
                        tlx.local_store(
                            sdv_store_buf[kv_buf_id],
                            dv.to(tlx.dtype_of(dv_desc)),
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            dv_desc,
                            sdv_store_buf[kv_buf_id],
                            [
                                (seq_start + start_block_n).to(tl.int32),
                                (off_h * stride_dvh + slice_id * DKV_STORE_NCOL).to(
                                    tl.int32
                                ),
                            ],
                        )
                    tlx.barrier_arrive(dv_empties[kv_buf_id])

                    # dK: wait for MMA's dQ dot (last k_tiles read) before staging
                    tlx.barrier_wait(dk_fulls[kv_buf_id], kv_phase)
                    tlx.barrier_wait(k_mma_done[kv_buf_id], kv_phase)
                    for slice_id in tl.static_range(DKV_STORE_ITERS):
                        dk_slice = tlx.local_slice(
                            dk_tiles[kv_buf_id],
                            [0, slice_id * DKV_STORE_NCOL],
                            [BLOCK_N1, DKV_STORE_NCOL],
                        )
                        dk = tlx.local_load(dk_slice)
                        dk *= sm_scale
                        tlx.async_descriptor_store_wait(0)
                        tlx.local_store(
                            sdk_store_buf[kv_buf_id],
                            dk.to(tlx.dtype_of(dk_desc)),
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            dk_desc,
                            sdk_store_buf[kv_buf_id],
                            [
                                (seq_start + start_block_n).to(tl.int32),
                                (off_h * stride_dkh + slice_id * DKV_STORE_NCOL).to(
                                    tl.int32
                                ),
                            ],
                        )
                    tlx.async_descriptor_store_wait(0)
                    tlx.barrier_arrive(k_empties[kv_buf_id])
                    tlx.barrier_arrive(dk_empties[kv_buf_id])
                tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1
        # mma group
        with tlx.async_task(num_warps=1, registers=80):
            blk_idx = 0
            tile_cnt = 0
            clc_phase_consumer = 0
            while tile_idx != -1:
                (
                    off_z,
                    off_h,
                    seq_start,
                    seq_len,
                    n_targets,
                    start_n,
                    low1,
                    high1,
                    low2,
                    high2,
                ) = _hstu_bwd_calculate_offsets(
                    tile_idx,
                    n_tile_num,
                    num_pid_m,
                    seq_offsets,
                    num_targets,
                    max_attn_len,
                    full_attn_size,
                    contextual_seq_len,
                    H,
                    BLOCK_M1,
                    BLOCK_N1,
                    GROUP_SIZE_M,
                    HAS_NUM_TARGETS,
                    HAS_MAX_ATTN_LEN,
                    HAS_FULL_ATTN_SIZE,
                    HAS_CONTEXTUAL_SEQ_LEN,
                    STAGE,
                )
                if start_n < seq_len:
                    # Use ceiling division to match tl.range behavior
                    # tl.range(low, high, step) gives ceil((high - low) / step) iterations
                    num_steps_masked = tl.cdiv(high1 - low1, BLOCK_M1)
                    num_steps_unmasked = tl.cdiv(high2 - low2, BLOCK_M1)
                    num_steps = num_steps_masked + num_steps_unmasked

                    kv_buf_id, kv_phase = _get_bufidx_phase(tile_cnt, NUM_BUFFERS_KV)
                    tile_cnt += 1

                    tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)
                    tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)

                    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

                    # Prolog
                    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(
                        blk_idx, NUM_BUFFERS_TMEM
                    )

                    # Compute qkT = tl.dot(k, qT)
                    tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
                    tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
                    qT = tlx.local_trans(q_tiles[q_buf_id])
                    tlx.async_dot(
                        k_tiles[kv_buf_id],
                        qT,
                        qk_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[qk_fulls[tmem_buf_id]],
                    )

                    # Compute dpT = tl.dot(v, tl.trans(do))
                    tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                    tlx.barrier_wait(dp_empties[tmem_buf_id], tmem_phase ^ 1)
                    doT = tlx.local_trans(do_tiles[do_buf_id])
                    tlx.async_dot(
                        v_tiles[kv_buf_id],
                        doT,
                        dp_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[dp_fulls[tmem_buf_id]],
                    )

                    # Compute dv += tl.dot(ppT, do)
                    tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
                    tlx.barrier_wait(dv_empties[kv_buf_id], kv_phase ^ 1)
                    tlx.async_dot(
                        p_tiles[tmem_buf_id],
                        do_tiles[do_buf_id],
                        dv_tiles[kv_buf_id],
                        use_acc=False,
                        mBarriers=[do_empties[do_buf_id]],
                    )
                    blk_idx += 1

                    # Main loop
                    tlx.barrier_wait(dk_empties[kv_buf_id], kv_phase ^ 1)
                    for j in range(1, num_steps):
                        q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                        tmem_buf_id, tmem_phase = _get_bufidx_phase(
                            blk_idx, NUM_BUFFERS_TMEM
                        )
                        # Compute qkT = tl.dot(k, qT)
                        tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
                        tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
                        qT = tlx.local_trans(q_tiles[q_buf_id])
                        tlx.async_dot(
                            k_tiles[kv_buf_id],
                            qT,
                            qk_tiles[tmem_buf_id],
                            use_acc=False,
                            mBarriers=[qk_fulls[tmem_buf_id]],
                        )

                        prev_blk_idx = blk_idx - 1
                        q_buf_id_prev, _ = _get_bufidx_phase(
                            prev_blk_idx, NUM_BUFFERS_Q
                        )
                        tmem_buf_id_prev, tmem_phase_prev = _get_bufidx_phase(
                            prev_blk_idx, NUM_BUFFERS_TMEM
                        )
                        ds_buf_id_prev, ds_phase_prev = _get_bufidx_phase(
                            prev_blk_idx, NUM_BUFFERS_DS
                        )

                        # Compute dq = tl.dot(tl.trans(dsT), k)
                        tlx.barrier_wait(ds_fulls[ds_buf_id_prev], ds_phase_prev)
                        tlx.barrier_wait(
                            dq_empties[tmem_buf_id_prev], tmem_phase_prev ^ 1
                        )
                        dsT_view = tlx.local_trans(ds_tiles[ds_buf_id_prev])
                        tlx.async_dot(
                            dsT_view,
                            k_tiles[kv_buf_id],
                            dq_tiles[tmem_buf_id_prev],
                            use_acc=False,
                            mBarriers=[
                                dq_fulls[tmem_buf_id_prev],
                            ],
                        )

                        # Compute dk += tl.dot(dsT, tl.trans(qT))
                        tlx.async_dot(
                            ds_tiles[ds_buf_id_prev],
                            q_tiles[q_buf_id_prev],
                            dk_tiles[kv_buf_id],
                            use_acc=(j - 1) > 0,
                            mBarriers=[
                                q_empties[q_buf_id_prev],
                            ],
                        )

                        do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                        # Compute dpT = tl.dot(v, tl.trans(do))
                        tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                        tlx.barrier_wait(dp_empties[tmem_buf_id], tmem_phase ^ 1)
                        doT = tlx.local_trans(do_tiles[do_buf_id])
                        tlx.async_dot(
                            v_tiles[kv_buf_id],
                            doT,
                            dp_tiles[tmem_buf_id],
                            use_acc=False,
                            mBarriers=[dp_fulls[tmem_buf_id]],
                        )

                        # Compute dv += tl.dot(ppT, do)
                        tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
                        tlx.async_dot(
                            p_tiles[tmem_buf_id],
                            do_tiles[do_buf_id],
                            dv_tiles[kv_buf_id],
                            use_acc=True,
                            mBarriers=[do_empties[do_buf_id]],
                        )
                        blk_idx += 1

                    tlx.tcgen05_commit(dv_fulls[kv_buf_id])

                    # Epilog
                    prev_blk_idx = blk_idx - 1
                    q_buf_id, _ = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(
                        prev_blk_idx, NUM_BUFFERS_TMEM
                    )
                    ds_buf_id, ds_phase = _get_bufidx_phase(
                        prev_blk_idx, NUM_BUFFERS_DS
                    )
                    # Compute dk += tl.dot(dsT, tl.trans(qT))
                    tlx.barrier_wait(ds_fulls[ds_buf_id], ds_phase)
                    tlx.async_dot(
                        ds_tiles[ds_buf_id],
                        q_tiles[q_buf_id],
                        dk_tiles[kv_buf_id],
                        use_acc=num_steps > 1,
                        mBarriers=[q_empties[q_buf_id], dk_fulls[tmem_buf_id]],
                    )

                    # Compute dq = tl.dot(tl.trans(dsT), k)
                    tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
                    dsT_view = tlx.local_trans(ds_tiles[ds_buf_id])
                    tlx.async_dot(
                        dsT_view,
                        k_tiles[kv_buf_id],
                        dq_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[
                            dq_fulls[tmem_buf_id],
                        ],
                    )
                    tlx.tcgen05_commit(k_mma_done[kv_buf_id])
                tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1

        # load group
        with tlx.async_task(num_warps=1, registers=80):
            blk_idx = 0
            tile_cnt = 0
            clc_phase_consumer = 0
            while tile_idx != -1:
                (
                    off_z,
                    off_h,
                    seq_start,
                    seq_len,
                    n_targets,
                    start_n,
                    low1,
                    high1,
                    low2,
                    high2,
                ) = _hstu_bwd_calculate_offsets(
                    tile_idx,
                    n_tile_num,
                    num_pid_m,
                    seq_offsets,
                    num_targets,
                    max_attn_len,
                    full_attn_size,
                    contextual_seq_len,
                    H,
                    BLOCK_M1,
                    BLOCK_N1,
                    GROUP_SIZE_M,
                    HAS_NUM_TARGETS,
                    HAS_MAX_ATTN_LEN,
                    HAS_FULL_ATTN_SIZE,
                    HAS_CONTEXTUAL_SEQ_LEN,
                    STAGE,
                )
                if start_n < seq_len:
                    start_block_n = start_n
                    # Load K
                    kv_buf_id, kv_phase = _get_bufidx_phase(tile_cnt, NUM_BUFFERS_KV)
                    tile_cnt += 1
                    tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        k_fulls[kv_buf_id], 2 * BLOCK_N1 * HEAD_DIM
                    )  # float16
                    tlx.async_descriptor_load(
                        desc_k,
                        k_tiles[kv_buf_id],
                        [
                            (seq_start + start_block_n).to(tl.int32),
                            (off_h * stride_kh).to(tl.int32),
                        ],
                        k_fulls[kv_buf_id],
                    )

                    # Load V
                    tlx.barrier_expect_bytes(
                        v_fulls[kv_buf_id], 2 * BLOCK_N1 * HEAD_DIM
                    )  # float16
                    tlx.async_descriptor_load(
                        desc_v,
                        v_tiles[kv_buf_id],
                        [
                            (seq_start + start_block_n).to(tl.int32),
                            (off_h * stride_vh).to(tl.int32),
                        ],
                        v_fulls[kv_buf_id],
                    )

                    # PART 1: Load Q and dO for masked region [low1, high1)
                    for curr_m in tl.range(low1, high1, BLOCK_M1):
                        q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                        do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                        # Load Q
                        tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                        tlx.barrier_expect_bytes(
                            q_fulls[q_buf_id], 2 * BLOCK_M1 * HEAD_DIM
                        )
                        tlx.async_descriptor_load(
                            desc_q,
                            q_tiles[q_buf_id],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (off_h * stride_qh).to(tl.int32),
                            ],
                            q_fulls[q_buf_id],
                        )
                        # Load dO
                        tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                        tlx.barrier_expect_bytes(
                            do_fulls[do_buf_id], 2 * BLOCK_M1 * HEAD_DIM
                        )
                        tlx.async_descriptor_load(
                            desc_do,
                            do_tiles[do_buf_id],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (off_h * stride_doh).to(tl.int32),
                            ],
                            do_fulls[do_buf_id],
                        )
                        blk_idx += 1

                    # PART 2: Load Q and dO for unmasked region [low2, high2)
                    for curr_m in tl.range(low2, high2, BLOCK_M1):
                        q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                        do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                        # Load Q
                        tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                        tlx.barrier_expect_bytes(
                            q_fulls[q_buf_id], 2 * BLOCK_M1 * HEAD_DIM
                        )
                        tlx.async_descriptor_load(
                            desc_q,
                            q_tiles[q_buf_id],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (off_h * stride_qh).to(tl.int32),
                            ],
                            q_fulls[q_buf_id],
                        )
                        # Load dO
                        tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                        tlx.barrier_expect_bytes(
                            do_fulls[do_buf_id], 2 * BLOCK_M1 * HEAD_DIM
                        )
                        tlx.async_descriptor_load(
                            desc_do,
                            do_tiles[do_buf_id],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (off_h * stride_doh).to(tl.int32),
                            ],
                            do_fulls[do_buf_id],
                        )
                        blk_idx += 1
                tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                clc_phase_consumer ^= 1


@triton.autotune(
    configs=get_hstu_bwd_configs(), key=["AUTOTUNE_MAX_SEQ_LEN", "HEAD_DIM"]
)
@triton.jit
def _hstu_attn_bwd_ws_non_persistent(
    DQ,
    DK,
    DV,
    desc_q,
    desc_k,
    desc_v,
    sm_scale,  # alpha
    desc_do,  #
    desc_dq,
    desc_dk,
    desc_dv,  #
    M,
    D,
    seq_offsets,
    attn_scale,
    num_targets,
    max_attn_len,
    full_attn_size,
    contextual_seq_len,
    # shared by Q/K/V/DO.
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    stride_mh,
    stride_deltah,
    H,
    Z,
    max_seq_len,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    DQ_REDUCE_FACTOR: tl.constexpr,
    DQ_REDUCE_STAGES: tl.constexpr,
    EARLY_RELEASE_SUBTILES: tl.constexpr,
    ALPHA_BWD_PRESCALE: tl.constexpr,
    STAGE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SOFTMAX: tl.constexpr,
    AUTOTUNE_MAX_SEQ_LEN: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_FULL_ATTN_SIZE: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
):
    """HSTU backward pass kernel with jagged sequences and SiLU/Softmax activation."""
    # Kernel hangs if NUM_BUFFERS_Q != 2.
    tl.static_assert(NUM_BUFFERS_Q == 2)
    # Runtime error if NUM_BUFFERS_DO != 1
    tl.static_assert(NUM_BUFFERS_DO == 1)
    # TODO: Add support for contextual seq len
    tl.static_assert(not HAS_CONTEXTUAL_SEQ_LEN)
    tl.static_assert(
        EARLY_RELEASE_SUBTILES >= 1 and EARLY_RELEASE_SUBTILES <= 2,
        "EARLY_RELEASE_SUBTILES must be 1 or 2",
    )

    REUSE_DP_FOR_DQ: tl.constexpr = (BLOCK_M1 == 128) and (HEAD_DIM == 128)

    n_tile_num = tl.cdiv(max_seq_len, BLOCK_N1)
    num_pid_m = Z * H
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    # allocate smem buffers
    k_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV
    )
    v_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS_KV
    )
    q_tiles = tlx.local_alloc((BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    do_tiles = tlx.local_alloc(
        (BLOCK_M1, HEAD_DIM), tlx.dtype_of(desc_do), NUM_BUFFERS_DO
    )

    # Use SMEM for dsT
    ds_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1), tlx.dtype_of(desc_q), NUM_BUFFERS_DS
    )

    # SMEM staging buffer for double-buffered dQ TMA reduce-add (from D97693127)
    DQ_REDUCE_NCOL: tl.constexpr = HEAD_DIM // (EPILOGUE_SUBTILE * DQ_REDUCE_FACTOR)
    DQ_REDUCE_ITERS: tl.constexpr = HEAD_DIM // DQ_REDUCE_NCOL
    dq_store_buf = tlx.local_alloc(
        (BLOCK_M1, DQ_REDUCE_NCOL), tlx.dtype_of(desc_dq), DQ_REDUCE_STAGES
    )

    # allocate barriers for smem buffers
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    ds_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    # allocate tmem buffers
    qk_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem
    )
    p_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_do),
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    dp_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
    )

    dv_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem
    )
    dk_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem
    )

    # allocate barriers for tmem buffers
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    if REUSE_DP_FOR_DQ:
        dq_tiles = tlx.local_alloc(
            (BLOCK_M1, HEAD_DIM),
            tl.float32,
            NUM_BUFFERS_TMEM,
            tlx.storage_kind.tmem,
            reuse=dp_tiles,
        )
        dp_empties = dq_empties
    else:
        dq_tiles = tlx.local_alloc(
            (BLOCK_M1, HEAD_DIM),
            tl.float32,
            NUM_BUFFERS_TMEM,
            tlx.storage_kind.tmem,
        )
        dp_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    with tlx.async_tasks():
        # reduction group: accumulate dq
        with tlx.async_task("default"):
            LN2 = 0.6931471824645996  # = ln(2)
            blk_idx = 0
            (
                off_z,
                off_h,
                seq_start,
                seq_len,
                n_targets,
                start_n,
                low1,
                high1,
                low2,
                high2,
            ) = _hstu_bwd_calculate_offsets(
                tile_idx,
                n_tile_num,
                num_pid_m,
                seq_offsets,
                num_targets,
                max_attn_len,
                full_attn_size,
                contextual_seq_len,
                H,
                BLOCK_M1,
                BLOCK_N1,
                GROUP_SIZE_M,
                HAS_NUM_TARGETS,
                HAS_MAX_ATTN_LEN,
                HAS_FULL_ATTN_SIZE,
                HAS_CONTEXTUAL_SEQ_LEN,
                STAGE,
            )
            if start_n < seq_len:
                # if SOFTMAX:
                #     # Original softmax reduction: single fused loop
                #     num_steps_masked = (high1 - low1) // BLOCK_M1
                #     num_steps_unmasked = (high2 - low2) // BLOCK_M1
                #     num_steps = num_steps_masked + num_steps_unmasked
                #     curr_m = low1
                #     step_m = BLOCK_M1
                #     for _ in range(num_steps):
                #         tmem_buf_id, tmem_phase = _get_bufidx_phase(
                #             blk_idx, NUM_BUFFERS_TMEM
                #         )

                #         # wait for dq = tl.dot(tl.trans(dsT), k)
                #         tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                #         slice_size: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
                #         for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                #             dq_slice = tlx.local_slice(
                #                 dq_tiles[tmem_buf_id],
                #                 [0, slice_id * slice_size],
                #                 [BLOCK_M1, slice_size],
                #             )
                #             dq = tlx.local_load(dq_slice)
                #             dq = dq * LN2
                #             desc_dq.atomic_add(
                #                 [
                #                     (seq_start + curr_m).to(tl.int32),
                #                     (off_h * stride_dqh + slice_id * slice_size).to(
                #                         tl.int32
                #                     ),
                #                 ],
                #                 dq,
                #             )

                #         # release dq
                #         tlx.barrier_arrive(dq_empties[tmem_buf_id])
                #         # Increment pointers.
                #         curr_m += step_m
                #         blk_idx += 1
                # else:
                # SiLU reduction: two loops for PART 1 and PART 2
                # INLINED to avoid TLX region isolation issues with helper functions
                # dQ early TMEM release optimization (from D98825322)
                # PART 1: Masked region [low1, high1)
                for curr_m in tl.range(low1, high1, BLOCK_M1):
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(
                        blk_idx, NUM_BUFFERS_TMEM
                    )
                    tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                    for slice_id in tl.static_range(
                        DQ_REDUCE_ITERS - EARLY_RELEASE_SUBTILES
                    ):
                        dq_smem_idx = slice_id % DQ_REDUCE_STAGES
                        dq_slice = tlx.local_slice(
                            dq_tiles[tmem_buf_id],
                            [0, slice_id * DQ_REDUCE_NCOL],
                            [BLOCK_M1, DQ_REDUCE_NCOL],
                        )
                        dq = tlx.local_load(dq_slice)
                        dq = dq * sm_scale
                        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                        tlx.local_store(
                            dq_store_buf[dq_smem_idx],
                            dq.to(tlx.dtype_of(desc_dq)),
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            desc_dq,
                            dq_store_buf[dq_smem_idx],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (slice_id * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                    tl.int32
                                ),
                            ],
                            store_reduce="add",
                        )
                    if EARLY_RELEASE_SUBTILES == 1:
                        er0: tl.constexpr = DQ_REDUCE_ITERS - 1
                        dq_er0 = tlx.local_load(
                            tlx.local_slice(
                                dq_tiles[tmem_buf_id],
                                [0, er0 * DQ_REDUCE_NCOL],
                                [BLOCK_M1, DQ_REDUCE_NCOL],
                            )
                        )
                        tlx.barrier_arrive(dq_empties[tmem_buf_id])
                        dq_er0 = dq_er0 * sm_scale
                        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                        tlx.local_store(
                            dq_store_buf[er0 % DQ_REDUCE_STAGES],
                            dq_er0.to(tlx.dtype_of(desc_dq)),
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            desc_dq,
                            dq_store_buf[er0 % DQ_REDUCE_STAGES],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (er0 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                    tl.int32
                                ),
                            ],
                            store_reduce="add",
                        )
                    elif EARLY_RELEASE_SUBTILES == 2:
                        er0: tl.constexpr = DQ_REDUCE_ITERS - 2
                        er1: tl.constexpr = DQ_REDUCE_ITERS - 1
                        dq_er0 = tlx.local_load(
                            tlx.local_slice(
                                dq_tiles[tmem_buf_id],
                                [0, er0 * DQ_REDUCE_NCOL],
                                [BLOCK_M1, DQ_REDUCE_NCOL],
                            )
                        )
                        dq_er1 = tlx.local_load(
                            tlx.local_slice(
                                dq_tiles[tmem_buf_id],
                                [0, er1 * DQ_REDUCE_NCOL],
                                [BLOCK_M1, DQ_REDUCE_NCOL],
                            )
                        )
                        tlx.barrier_arrive(dq_empties[tmem_buf_id])
                        dq_er0 = dq_er0 * sm_scale
                        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                        tlx.local_store(
                            dq_store_buf[er0 % DQ_REDUCE_STAGES],
                            dq_er0.to(tlx.dtype_of(desc_dq)),
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            desc_dq,
                            dq_store_buf[er0 % DQ_REDUCE_STAGES],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (er0 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                    tl.int32
                                ),
                            ],
                            store_reduce="add",
                        )
                        dq_er1 = dq_er1 * sm_scale
                        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                        tlx.local_store(
                            dq_store_buf[er1 % DQ_REDUCE_STAGES],
                            dq_er1.to(tlx.dtype_of(desc_dq)),
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            desc_dq,
                            dq_store_buf[er1 % DQ_REDUCE_STAGES],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (er1 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                    tl.int32
                                ),
                            ],
                            store_reduce="add",
                        )
                    blk_idx += 1

                # PART 2: Unmasked region [low2, high2)
                for curr_m in tl.range(low2, high2, BLOCK_M1):
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(
                        blk_idx, NUM_BUFFERS_TMEM
                    )
                    tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                    for slice_id in tl.static_range(
                        DQ_REDUCE_ITERS - EARLY_RELEASE_SUBTILES
                    ):
                        dq_smem_idx = slice_id % DQ_REDUCE_STAGES
                        dq_slice = tlx.local_slice(
                            dq_tiles[tmem_buf_id],
                            [0, slice_id * DQ_REDUCE_NCOL],
                            [BLOCK_M1, DQ_REDUCE_NCOL],
                        )
                        dq = tlx.local_load(dq_slice)
                        dq = dq * sm_scale
                        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                        tlx.local_store(
                            dq_store_buf[dq_smem_idx],
                            dq.to(tlx.dtype_of(desc_dq)),
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            desc_dq,
                            dq_store_buf[dq_smem_idx],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (slice_id * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                    tl.int32
                                ),
                            ],
                            store_reduce="add",
                        )
                    if EARLY_RELEASE_SUBTILES == 1:
                        er0: tl.constexpr = DQ_REDUCE_ITERS - 1
                        dq_er0 = tlx.local_load(
                            tlx.local_slice(
                                dq_tiles[tmem_buf_id],
                                [0, er0 * DQ_REDUCE_NCOL],
                                [BLOCK_M1, DQ_REDUCE_NCOL],
                            )
                        )
                        tlx.barrier_arrive(dq_empties[tmem_buf_id])
                        dq_er0 = dq_er0 * sm_scale
                        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                        tlx.local_store(
                            dq_store_buf[er0 % DQ_REDUCE_STAGES],
                            dq_er0.to(tlx.dtype_of(desc_dq)),
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            desc_dq,
                            dq_store_buf[er0 % DQ_REDUCE_STAGES],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (er0 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                    tl.int32
                                ),
                            ],
                            store_reduce="add",
                        )
                    elif EARLY_RELEASE_SUBTILES == 2:
                        er0: tl.constexpr = DQ_REDUCE_ITERS - 2
                        er1: tl.constexpr = DQ_REDUCE_ITERS - 1
                        dq_er0 = tlx.local_load(
                            tlx.local_slice(
                                dq_tiles[tmem_buf_id],
                                [0, er0 * DQ_REDUCE_NCOL],
                                [BLOCK_M1, DQ_REDUCE_NCOL],
                            )
                        )
                        dq_er1 = tlx.local_load(
                            tlx.local_slice(
                                dq_tiles[tmem_buf_id],
                                [0, er1 * DQ_REDUCE_NCOL],
                                [BLOCK_M1, DQ_REDUCE_NCOL],
                            )
                        )
                        tlx.barrier_arrive(dq_empties[tmem_buf_id])
                        dq_er0 = dq_er0 * sm_scale
                        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                        tlx.local_store(
                            dq_store_buf[er0 % DQ_REDUCE_STAGES],
                            dq_er0.to(tlx.dtype_of(desc_dq)),
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            desc_dq,
                            dq_store_buf[er0 % DQ_REDUCE_STAGES],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (er0 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                    tl.int32
                                ),
                            ],
                            store_reduce="add",
                        )
                        dq_er1 = dq_er1 * sm_scale
                        tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                        tlx.local_store(
                            dq_store_buf[er1 % DQ_REDUCE_STAGES],
                            dq_er1.to(tlx.dtype_of(desc_dq)),
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            desc_dq,
                            dq_store_buf[er1 % DQ_REDUCE_STAGES],
                            [
                                (seq_start + curr_m).to(tl.int32),
                                (er1 * DQ_REDUCE_NCOL + off_h * stride_dqh).to(
                                    tl.int32
                                ),
                            ],
                            store_reduce="add",
                        )
                    blk_idx += 1

        # compute group: compute softmax/silu backward
        with tlx.async_task(num_warps=8, registers=192, replicate=1):
            blk_idx = 0
            tile_cnt = 0
            (
                off_z,
                off_h,
                seq_start,
                seq_len,
                n_targets,
                start_n,
                low1,
                high1,
                low2,
                high2,
            ) = _hstu_bwd_calculate_offsets(
                tile_idx,
                n_tile_num,
                num_pid_m,
                seq_offsets,
                num_targets,
                max_attn_len,
                full_attn_size,
                contextual_seq_len,
                H,
                BLOCK_M1,
                BLOCK_N1,
                GROUP_SIZE_M,
                HAS_NUM_TARGETS,
                HAS_MAX_ATTN_LEN,
                HAS_FULL_ATTN_SIZE,
                HAS_CONTEXTUAL_SEQ_LEN,
                STAGE,
            )
            if start_n < seq_len:
                start_block_n = start_n

                curr_m = low1
                do_out_dtype = tlx.dtype_of(desc_do)
                q_out_dtype = tlx.dtype_of(desc_q)

                # SiLU activation backward
                scale = tl.load(attn_scale).to(tl.float32)
                # PART 1: Masked region [low1, high1)
                blk_idx = _hstu_bwd_compute_inner_loop_silu(
                    start_n,
                    qk_fulls,
                    qk_tiles,
                    qk_empties,
                    p_tiles,
                    p_fulls,
                    dp_empties,
                    dp_fulls,
                    dp_tiles,
                    ds_tiles,
                    ds_fulls,
                    seq_start,
                    seq_len,
                    off_h,
                    sm_scale,  # alpha
                    scale,
                    low1,
                    high1,
                    blk_idx,
                    do_out_dtype,
                    q_out_dtype,
                    contextual_seq_len,
                    max_attn_len,
                    n_targets,
                    NUM_BUFFERS_TMEM,
                    NUM_BUFFERS_DS,
                    BLOCK_M1,
                    BLOCK_N1,
                    REUSE_DP_FOR_DQ=REUSE_DP_FOR_DQ,
                    HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                    HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                    HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                    APPLY_MASK=True,
                    ALPHA_BWD_PRESCALE=ALPHA_BWD_PRESCALE,
                )
                # PART 2: Unmasked region [low2, high2)
                blk_idx = _hstu_bwd_compute_inner_loop_silu(
                    start_n,
                    qk_fulls,
                    qk_tiles,
                    qk_empties,
                    p_tiles,
                    p_fulls,
                    dp_empties,
                    dp_fulls,
                    dp_tiles,
                    ds_tiles,
                    ds_fulls,
                    seq_start,
                    seq_len,
                    off_h,
                    sm_scale,  # alpha
                    scale,
                    low2,
                    high2,
                    blk_idx,
                    do_out_dtype,
                    q_out_dtype,
                    contextual_seq_len,
                    max_attn_len,
                    n_targets,
                    NUM_BUFFERS_TMEM,
                    NUM_BUFFERS_DS,
                    BLOCK_M1,
                    BLOCK_N1,
                    REUSE_DP_FOR_DQ=REUSE_DP_FOR_DQ,
                    HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                    HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                    HAS_NUM_TARGETS=HAS_NUM_TARGETS,
                    APPLY_MASK=False,
                    ALPHA_BWD_PRESCALE=ALPHA_BWD_PRESCALE,
                )

                # epilogue
                kv_buf_id, kv_phase = _get_bufidx_phase(tile_cnt, NUM_BUFFERS_KV)
                tile_cnt += 1

                # Create on-device TMA descriptors for jagged sequence writes
                end_kv = seq_start + seq_len
                dv_desc = tl.make_tensor_descriptor(
                    DV,
                    shape=[end_kv.to(tl.int32), stride_dvh * H],
                    strides=[stride_dvn, 1],
                    block_shape=[BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE],
                )
                dk_desc = tl.make_tensor_descriptor(
                    DK,
                    shape=[end_kv.to(tl.int32), stride_dkh * H],
                    strides=[stride_dkn, 1],
                    block_shape=[BLOCK_N1, HEAD_DIM // EPILOGUE_SUBTILE],
                )

                tlx.barrier_wait(dv_fulls[kv_buf_id], kv_phase)
                tlx.fence_async_shared()
                slice_size: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
                for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                    dv_slice = tlx.local_slice(
                        dv_tiles[kv_buf_id],
                        [0, slice_id * slice_size],
                        [BLOCK_N1, slice_size],
                    )
                    dv = tlx.local_load(dv_slice)
                    dv_desc.store(
                        [
                            (seq_start + start_block_n).to(tl.int32),
                            (off_h * stride_dvh + slice_id * slice_size).to(tl.int32),
                        ],
                        dv.to(tlx.dtype_of(desc_dv)),
                    )
                tlx.barrier_arrive(dv_empties[kv_buf_id])

                tlx.barrier_wait(dk_fulls[kv_buf_id], kv_phase)
                for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                    dk_slice = tlx.local_slice(
                        dk_tiles[kv_buf_id],
                        [0, slice_id * slice_size],
                        [BLOCK_N1, slice_size],
                    )
                    dk = tlx.local_load(dk_slice)
                    dk *= sm_scale
                    dk_desc.store(
                        [
                            (seq_start + start_block_n).to(tl.int32),
                            (off_h * stride_dkh + slice_id * slice_size).to(tl.int32),
                        ],
                        dk.to(tlx.dtype_of(desc_dk)),
                    )
                tlx.barrier_arrive(dk_empties[kv_buf_id])
        # mma group
        with tlx.async_task(num_warps=1, registers=80):
            blk_idx = 0
            tile_cnt = 0
            (
                off_z,
                off_h,
                seq_start,
                seq_len,
                n_targets,
                start_n,
                low1,
                high1,
                low2,
                high2,
            ) = _hstu_bwd_calculate_offsets(
                tile_idx,
                n_tile_num,
                num_pid_m,
                seq_offsets,
                num_targets,
                max_attn_len,
                full_attn_size,
                contextual_seq_len,
                H,
                BLOCK_M1,
                BLOCK_N1,
                GROUP_SIZE_M,
                HAS_NUM_TARGETS,
                HAS_MAX_ATTN_LEN,
                HAS_FULL_ATTN_SIZE,
                HAS_CONTEXTUAL_SEQ_LEN,
                STAGE,
            )

            if start_n < seq_len:
                # Use ceiling division to match tl.range behavior
                # tl.range(low, high, step) gives ceil((high - low) / step) iterations
                num_steps_masked = tl.cdiv(high1 - low1, BLOCK_M1)
                num_steps_unmasked = tl.cdiv(high2 - low2, BLOCK_M1)
                num_steps = num_steps_masked + num_steps_unmasked

                kv_buf_id, kv_phase = _get_bufidx_phase(tile_cnt, NUM_BUFFERS_KV)
                tile_cnt += 1

                tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)
                tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)

                tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

                # Prolog
                q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

                # Compute qkT = tl.dot(k, qT)
                tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
                tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
                qT = tlx.local_trans(q_tiles[q_buf_id])
                tlx.async_dot(
                    k_tiles[kv_buf_id],
                    qT,
                    qk_tiles[tmem_buf_id],
                    use_acc=False,
                    mBarriers=[qk_fulls[tmem_buf_id]],
                )

                # Compute dpT = tl.dot(v, tl.trans(do))
                tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                tlx.barrier_wait(dp_empties[tmem_buf_id], tmem_phase ^ 1)
                doT = tlx.local_trans(do_tiles[do_buf_id])
                tlx.async_dot(
                    v_tiles[kv_buf_id],
                    doT,
                    dp_tiles[tmem_buf_id],
                    use_acc=False,
                    mBarriers=[dp_fulls[tmem_buf_id]],
                )

                # Compute dv += tl.dot(ppT, do)
                tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
                tlx.barrier_wait(dv_empties[kv_buf_id], kv_phase ^ 1)
                tlx.async_dot(
                    p_tiles[tmem_buf_id],
                    do_tiles[do_buf_id],
                    dv_tiles[kv_buf_id],
                    use_acc=False,
                    mBarriers=[do_empties[do_buf_id]],
                )
                blk_idx += 1

                # Main loop
                tlx.barrier_wait(dk_empties[kv_buf_id], kv_phase ^ 1)
                for j in range(1, num_steps):
                    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(
                        blk_idx, NUM_BUFFERS_TMEM
                    )
                    # Compute qkT = tl.dot(k, qT)
                    tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
                    tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
                    qT = tlx.local_trans(q_tiles[q_buf_id])
                    tlx.async_dot(
                        k_tiles[kv_buf_id],
                        qT,
                        qk_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[qk_fulls[tmem_buf_id]],
                    )

                    prev_blk_idx = blk_idx - 1
                    q_buf_id_prev, _ = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                    tmem_buf_id_prev, tmem_phase_prev = _get_bufidx_phase(
                        prev_blk_idx, NUM_BUFFERS_TMEM
                    )
                    ds_buf_id_prev, ds_phase_prev = _get_bufidx_phase(
                        prev_blk_idx, NUM_BUFFERS_DS
                    )

                    # Compute dq = tl.dot(tl.trans(dsT), k)
                    tlx.barrier_wait(ds_fulls[ds_buf_id_prev], ds_phase_prev)
                    tlx.barrier_wait(dq_empties[tmem_buf_id_prev], tmem_phase_prev ^ 1)
                    dsT_view = tlx.local_trans(ds_tiles[ds_buf_id_prev])
                    tlx.async_dot(
                        dsT_view,
                        k_tiles[kv_buf_id],
                        dq_tiles[tmem_buf_id_prev],
                        use_acc=False,
                        mBarriers=[
                            dq_fulls[tmem_buf_id_prev],
                        ],
                    )

                    # Compute dk += tl.dot(dsT, tl.trans(qT))
                    tlx.async_dot(
                        ds_tiles[ds_buf_id_prev],
                        q_tiles[q_buf_id_prev],
                        dk_tiles[kv_buf_id],
                        use_acc=(j - 1) > 0,
                        mBarriers=[
                            q_empties[q_buf_id_prev],
                        ],
                    )

                    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                    # Compute dpT = tl.dot(v, tl.trans(do))
                    tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
                    tlx.barrier_wait(dp_empties[tmem_buf_id], tmem_phase ^ 1)
                    doT = tlx.local_trans(do_tiles[do_buf_id])
                    tlx.async_dot(
                        v_tiles[kv_buf_id],
                        doT,
                        dp_tiles[tmem_buf_id],
                        use_acc=False,
                        mBarriers=[dp_fulls[tmem_buf_id]],
                    )

                    # Compute dv += tl.dot(ppT, do)
                    tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
                    tlx.async_dot(
                        p_tiles[tmem_buf_id],
                        do_tiles[do_buf_id],
                        dv_tiles[kv_buf_id],
                        use_acc=True,
                        mBarriers=[do_empties[do_buf_id]],
                    )
                    blk_idx += 1

                tlx.tcgen05_commit(dv_fulls[kv_buf_id])

                # Epilog
                prev_blk_idx = blk_idx - 1
                q_buf_id, _ = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
                tmem_buf_id, tmem_phase = _get_bufidx_phase(
                    prev_blk_idx, NUM_BUFFERS_TMEM
                )
                ds_buf_id, ds_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)
                # Compute dk += tl.dot(dsT, tl.trans(qT))
                tlx.barrier_wait(ds_fulls[ds_buf_id], ds_phase)
                tlx.async_dot(
                    ds_tiles[ds_buf_id],
                    q_tiles[q_buf_id],
                    dk_tiles[kv_buf_id],
                    use_acc=num_steps > 1,
                    mBarriers=[q_empties[q_buf_id], dk_fulls[tmem_buf_id]],
                )

                # Compute dq = tl.dot(tl.trans(dsT), k)
                tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
                dsT_view = tlx.local_trans(ds_tiles[ds_buf_id])
                tlx.async_dot(
                    dsT_view,
                    k_tiles[kv_buf_id],
                    dq_tiles[tmem_buf_id],
                    use_acc=False,
                    mBarriers=[
                        dq_fulls[tmem_buf_id],
                    ],
                )
                tlx.tcgen05_commit(k_empties[kv_buf_id])

        # load group
        with tlx.async_task(num_warps=1, registers=80):
            blk_idx = 0
            tile_cnt = 0
            (
                off_z,
                off_h,
                seq_start,
                seq_len,
                n_targets,
                start_n,
                low1,
                high1,
                low2,
                high2,
            ) = _hstu_bwd_calculate_offsets(
                tile_idx,
                n_tile_num,
                num_pid_m,
                seq_offsets,
                num_targets,
                max_attn_len,
                full_attn_size,
                contextual_seq_len,
                H,
                BLOCK_M1,
                BLOCK_N1,
                GROUP_SIZE_M,
                HAS_NUM_TARGETS,
                HAS_MAX_ATTN_LEN,
                HAS_FULL_ATTN_SIZE,
                HAS_CONTEXTUAL_SEQ_LEN,
                STAGE,
            )
            if start_n < seq_len:
                start_block_n = start_n
                # Load K
                kv_buf_id, kv_phase = _get_bufidx_phase(tile_cnt, NUM_BUFFERS_KV)
                tile_cnt += 1
                tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
                tlx.barrier_expect_bytes(
                    k_fulls[kv_buf_id], 2 * BLOCK_N1 * HEAD_DIM
                )  # float16
                tlx.async_descriptor_load(
                    desc_k,
                    k_tiles[kv_buf_id],
                    [
                        (seq_start + start_block_n).to(tl.int32),
                        (off_h * stride_kh).to(tl.int32),
                    ],
                    k_fulls[kv_buf_id],
                )

                # Load V
                tlx.barrier_expect_bytes(
                    v_fulls[kv_buf_id], 2 * BLOCK_N1 * HEAD_DIM
                )  # float16
                tlx.async_descriptor_load(
                    desc_v,
                    v_tiles[kv_buf_id],
                    [
                        (seq_start + start_block_n).to(tl.int32),
                        (off_h * stride_vh).to(tl.int32),
                    ],
                    v_fulls[kv_buf_id],
                )

                # PART 1: Load Q and dO for masked region [low1, high1)
                for curr_m in tl.range(low1, high1, BLOCK_M1):
                    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                    # Load Q
                    tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                    tlx.barrier_expect_bytes(q_fulls[q_buf_id], 2 * BLOCK_M1 * HEAD_DIM)
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_buf_id],
                        [
                            (seq_start + curr_m).to(tl.int32),
                            (off_h * stride_qh).to(tl.int32),
                        ],
                        q_fulls[q_buf_id],
                    )
                    # Load dO
                    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        do_fulls[do_buf_id], 2 * BLOCK_M1 * HEAD_DIM
                    )
                    tlx.async_descriptor_load(
                        desc_do,
                        do_tiles[do_buf_id],
                        [
                            (seq_start + curr_m).to(tl.int32),
                            (off_h * stride_doh).to(tl.int32),
                        ],
                        do_fulls[do_buf_id],
                    )
                    blk_idx += 1

                # PART 2: Load Q and dO for unmasked region [low2, high2)
                for curr_m in tl.range(low2, high2, BLOCK_M1):
                    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
                    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
                    # Load Q
                    tlx.barrier_wait(q_empties[q_buf_id], q_phase ^ 1)
                    tlx.barrier_expect_bytes(q_fulls[q_buf_id], 2 * BLOCK_M1 * HEAD_DIM)
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_buf_id],
                        [
                            (seq_start + curr_m).to(tl.int32),
                            (off_h * stride_qh).to(tl.int32),
                        ],
                        q_fulls[q_buf_id],
                    )
                    # Load dO
                    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        do_fulls[do_buf_id], 2 * BLOCK_M1 * HEAD_DIM
                    )
                    tlx.async_descriptor_load(
                        desc_do,
                        do_tiles[do_buf_id],
                        [
                            (seq_start + curr_m).to(tl.int32),
                            (off_h * stride_doh).to(tl.int32),
                        ],
                        do_fulls[do_buf_id],
                    )
                    blk_idx += 1


def backward_custom_vars(
    dout: torch.Tensor,
    num_softmax_heads: int,
    idx: int,
    saved_tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Load M from saved tensors and compute Delta for backward pass."""
    M = saved_tensors[idx]
    idx += 1
    out = saved_tensors[idx]

    # Compute Delta = sum(out * dout, dim=-1)
    Delta = torch.empty_like(M)
    total_seq_len, H, HEAD_DIM = out.shape
    BLOCK_M = 128
    grid = (triton.cdiv(total_seq_len, BLOCK_M), H)
    # For jagged, we compute delta per head
    Delta = (out * dout).sum(dim=-1).to(torch.float32)

    stride_mm = M.stride(0)
    return M, Delta, stride_mm


def tlx_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    M: torch.Tensor,
    Delta: torch.Tensor,
    stride_mm: int,
    num_softmax_heads: int,
    num_targets: Optional[torch.Tensor],
    causal: bool,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
    contextual_seq_len: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for HSTU attention with jagged sequences."""
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)
    dout = switch_to_contiguous_if_needed(dout)

    Z = seq_offsets.numel() - 1
    total_seq_len_q, H, HEAD_DIM = q.shape
    _, _, DimV = v.shape

    if total_seq_len_q == 0:
        return dq, dk, dv

    HAS_NUM_TARGETS = num_targets is not None
    if not HAS_NUM_TARGETS:
        num_targets = torch.empty(0, device=q.device, dtype=torch.int32)
    HAS_MAX_ATTN_LEN = max_attn_len != 0
    HAS_CONTEXTUAL_SEQ_LEN = contextual_seq_len != 0

    if dq.dtype != torch.float32:
        # accumulate dq in fp32
        dq = torch.empty_like(q, dtype=torch.float32)

    # TMA descriptors
    dummy_block = [1, 1]
    desc_q = TensorDescriptor(
        q,
        shape=[total_seq_len_q, H * HEAD_DIM],
        strides=[q.stride(0), 1],
        block_shape=dummy_block,
    )
    desc_k = TensorDescriptor(
        k,
        shape=[total_seq_len_q, H * HEAD_DIM],
        strides=[k.stride(0), 1],
        block_shape=dummy_block,
    )
    desc_v = TensorDescriptor(
        v,
        shape=[total_seq_len_q, H * DimV],
        strides=[v.stride(0), 1],
        block_shape=dummy_block,
    )
    desc_do = TensorDescriptor(
        dout,
        shape=[total_seq_len_q, H * DimV],
        strides=[dout.stride(0), 1],
        block_shape=dummy_block,
    )
    desc_dq = TensorDescriptor(
        dq,
        shape=[total_seq_len_q, H * HEAD_DIM],
        strides=[dq.stride(0), 1],
        block_shape=dummy_block,
    )
    desc_dk = TensorDescriptor(
        dk,
        shape=[total_seq_len_q, H * HEAD_DIM],
        strides=[dk.stride(0), 1],
        block_shape=dummy_block,
    )
    desc_dv = TensorDescriptor(
        dv,
        shape=[total_seq_len_q, H * DimV],
        strides=[dv.stride(0), 1],
        block_shape=dummy_block,
    )

    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Apply scaling to k for backward (only for softmax, not SiLU)
    # For softmax: K is pre-scaled by alpha * RCP_LN2 to enable flash2 trick
    # For SiLU: K is not pre-scaled, alpha scaling is done at the end
    SOFTMAX = num_softmax_heads != 0
    if SOFTMAX:
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        k_scaled = k * (alpha * RCP_LN2)
        desc_k = TensorDescriptor(
            k_scaled,
            shape=[total_seq_len_q, H * HEAD_DIM],
            strides=[k.stride(0), 1],
            block_shape=dummy_block,
        )
    else:
        # SiLU case: no pre-scaling of K
        desc_k = TensorDescriptor(
            k,
            shape=[total_seq_len_q, H * HEAD_DIM],
            strides=[k.stride(0), 1],
            block_shape=dummy_block,
        )

    BLOCK_M1 = 128
    stage = 3 if causal else 1

    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count

    use_persistent = True
    if use_persistent:
        grid = lambda meta: (  # noqa E731
            H * Z * triton.cdiv(max_seq_len, meta["BLOCK_N1"]),
        )
        bwd_kernel = _hstu_attn_bwd_ws
    else:
        grid = lambda meta: (  # noqa E731
            H * Z * triton.cdiv(max_seq_len, meta["BLOCK_N1"]),
        )
        bwd_kernel = _hstu_attn_bwd_ws_non_persistent

    bwd_kernel[grid](
        dq,
        dk,
        dv,
        desc_q=desc_q,
        desc_k=desc_k,
        desc_v=desc_v,
        sm_scale=alpha,
        desc_do=desc_do,
        desc_dq=desc_dq,
        desc_dk=desc_dk,
        desc_dv=desc_dv,
        M=M,
        D=Delta,
        seq_offsets=seq_offsets,
        attn_scale=attn_scale,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        full_attn_size=full_attn_size,
        contextual_seq_len=contextual_seq_len,
        stride_qh=q.stride(1),
        stride_kh=k.stride(1),
        stride_vh=v.stride(1),
        stride_doh=dout.stride(1),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        stride_mh=stride_mm,
        stride_deltah=Delta.stride(0) if Delta.ndim > 1 else 1,
        H=H,
        Z=Z,
        max_seq_len=max_seq_len,
        BLK_SLICE_FACTOR=2,
        HEAD_DIM=HEAD_DIM,
        STAGE=stage,
        GROUP_SIZE_M=1,
        SOFTMAX=num_softmax_heads != 0,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
        HAS_FULL_ATTN_SIZE=full_attn_size != 0,
        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
    )

    return dq, dk, dv


def forward_custom_vars(
    q: torch.Tensor, num_softmax_heads: int, saved_tensors: List[torch.Tensor]
) -> Tuple[torch.Tensor, int]:
    assert num_softmax_heads <= q.shape[1]
    M = torch.empty(
        (q.shape[0], num_softmax_heads), device=q.device, dtype=torch.float32
    )
    saved_tensors.append(M)
    stride_mm = M.stride(0)
    return M, stride_mm


def tlx_hstu_attention_fwd(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    sort_by_length_indices: Optional[torch.Tensor],
    M: torch.Tensor,
    stride_mm: int,
    num_softmax_heads: int,
    num_targets: Optional[torch.Tensor],
    causal: bool,
    max_attn_len: int,
    full_attn_size: int,
    contextual_seq_len: int,
) -> torch.Tensor:
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)
    Z = seq_offsets.numel() - 1
    # Previously this is AUTOTUNE_Z=prev_power_of_2(Z)
    # We rollback to Z to avoid the .item() call in prev_power_of_2
    # TODO: remove this once we have a better way to handle the .item() call
    AUTOTUNE_Z = Z
    total_seq_len_q, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
    if total_seq_len_q == 0:
        return out
    if attn_scale.ndim == 0:
        attn_scale_type = "scalar"
    else:
        attn_scale_type = "dynamic"
    assert num_softmax_heads == 0 or num_softmax_heads == H, (
        "num_softmax_heads must be 0 or H in tlx version due to a compilation issue"
    )
    HAS_NUM_TARGETS = num_targets is not None
    if not HAS_NUM_TARGETS:
        num_targets = torch.empty(0)
    HAS_MAX_ATTN_LEN = max_attn_len != 0
    HAS_FULL_ATTN_SIZE = full_attn_size != 0
    HAS_CONTEXTUAL_SEQ_LEN = contextual_seq_len != 0
    _, H, DimQ = q.shape
    _, _, DimV = v.shape

    total_seq_len_q, H, DimQ = q.shape
    total_seq_len_kv, _, _ = k.shape
    # TMA
    dummy_block = [1, 1]
    desc_q = TensorDescriptor(
        q,
        shape=[total_seq_len_q, H * DimQ],
        strides=[q.stride(0), 1],
        block_shape=dummy_block,
    )
    desc_o = TensorDescriptor(
        out,
        shape=[total_seq_len_q, H * DimV],
        strides=[out.stride(0), 1],
        block_shape=dummy_block,
    )
    desc_v = TensorDescriptor(
        v,
        shape=[total_seq_len_kv, H * DimV],
        strides=[v.stride(0), 1],
        block_shape=dummy_block,
    )
    desc_k = TensorDescriptor(
        k,
        shape=[total_seq_len_kv, H * DimQ],
        strides=[k.stride(0), 1],
        block_shape=dummy_block,
    )

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    grid = lambda meta: (  # noqa E731
        min(num_sms, H * Z * triton.cdiv(max_seq_len, meta["BLOCK_M"])),
    )
    _attn_fwd_ws[grid](
        Out=out,
        sm_scale=alpha,
        Z=Z,
        H=H,
        M=M,
        desc_q=desc_q,
        desc_k=desc_k,
        desc_v=desc_v,
        desc_o=desc_o,
        seq_offsets=seq_offsets,
        max_seq_len=max_seq_len,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        full_attn_size=full_attn_size,
        contextual_seq_len=contextual_seq_len,
        stride_mm=stride_mm,
        stride_qh=q.stride(1),
        stride_kh=k.stride(1),
        stride_vh=v.stride(1),
        stride_oh=out.stride(1),
        attn_scale=attn_scale,
        SOFTMAX=num_softmax_heads != 0,
        HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
        HAS_FULL_ATTN_SIZE=HAS_FULL_ATTN_SIZE,
        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
        ATTN_SCALE_TYPE=attn_scale_type,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        DimQ=DimQ,
        DimV=DimV,
        STAGE=3 if causal else 1,
    )

    return out


class _AttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        attn_scale: torch.Tensor,
        sort_by_length: bool,
        num_softmax_heads: int,
        num_targets: Optional[torch.Tensor],
        causal: bool,
        max_attn_len: int,
        full_attn_size: int,
        contextual_seq_len: int,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets, attn_scale]
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.num_softmax_heads = num_softmax_heads
        if num_targets is not None:
            saved_tensors.append(num_targets)
        ctx.has_num_targets = num_targets is not None
        ctx.causal = causal
        M, stride_mm = forward_custom_vars(q, num_softmax_heads, saved_tensors)
        ctx.alpha = alpha
        ctx.max_seq_len = max_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.max_attn_len = max_attn_len
        ctx.full_attn_size = full_attn_size
        ctx.contextual_seq_len = contextual_seq_len
        out = tlx_hstu_attention_fwd(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            attn_scale=attn_scale,
            sort_by_length_indices=sort_by_length_indices,
            M=M,
            stride_mm=stride_mm,
            num_softmax_heads=num_softmax_heads,
            num_targets=num_targets,
            causal=True,
            max_attn_len=max_attn_len,
            full_attn_size=full_attn_size,
            contextual_seq_len=contextual_seq_len,
        )
        saved_tensors.append(out)
        ctx.save_for_backward(*saved_tensors)
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        saved_tensors = ctx.saved_tensors
        q, k, v, seq_offsets, attn_scale = saved_tensors[:5]
        idx = 5
        if ctx.sort_by_length:
            sort_by_length_indices = saved_tensors[idx]
            idx += 1
        else:
            sort_by_length_indices = None
        num_softmax_heads = ctx.num_softmax_heads
        if ctx.has_num_targets:
            num_targets = ctx.saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        causal = ctx.causal
        Z, H, DimQ = q.shape
        _, _, DimV = v.shape
        d_qkv = torch.zeros((Z, H, DimQ * 2 + DimV), device=q.device, dtype=q.dtype)
        dq, dk, dv = d_qkv.split([DimQ, DimQ, DimV], dim=-1)
        M, Delta, stride_mm = backward_custom_vars(
            dout, num_softmax_heads, idx, saved_tensors
        )
        dq, dk, dv = tlx_hstu_attention_bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            seq_offsets=seq_offsets,
            attn_scale=attn_scale,
            max_seq_len=ctx.max_seq_len,
            alpha=ctx.alpha,
            M=M,
            Delta=Delta,
            stride_mm=stride_mm,
            num_softmax_heads=num_softmax_heads,
            num_targets=num_targets,
            causal=causal,
            max_attn_len=ctx.max_attn_len,
            full_attn_size=ctx.full_attn_size,
            contextual_seq_len=ctx.contextual_seq_len,
        )
        return (
            None,
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.jit.unused
@torch.fx.wrap
def tlx_bw_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
    contextual_seq_len: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    return _AttentionFunction.apply(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        attn_scale,
        sort_by_length,
        num_softmax_heads,
        num_targets,
        causal,
        max_attn_len,
        full_attn_size,
        contextual_seq_len,
    )


def tlx_bw_hstu_mha_wrapper(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    return tlx_bw_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        attn_scale=attn_scale,
        sort_by_length=sort_by_length,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
