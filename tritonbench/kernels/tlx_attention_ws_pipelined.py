import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    HEAD_DIM = nargs["HEAD_DIM"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]


configs = [
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
        },
        num_stages=0,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    ),
]

configs_persistent = [
    triton.Config(
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 3,
            "NUM_BUFFERS_QK": 1,
            "NUM_MMA_GROUPS": 2,
            "NUM_MMA_SLICES": 2,
        },
        num_stages=0,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    ),
]


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@triton.jit
def _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE: tl.constexpr):
    if STAGE == 1:
        # First part of STAGE == 3 in _get_fused_loop_bounds
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # Second part of STAGE == 3 in _get_fused_loop_bounds
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        tl.static_assert(STAGE == 3)
        # Maps to STAGE=1 in _get_fused_loop_bounds
        lo, hi = 0, N_CTX
    return lo, hi


@triton.jit
def _get_fused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE: tl.constexpr):
    if STAGE == 1:
        return 0, N_CTX
    else:
        tl.static_assert(STAGE == 3)
        return 0, (start_m + 1) * BLOCK_M


@triton.jit
def _compute_offsets(H, N_CTX, BLOCK_M, STAGE: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    lo, hi = _get_fused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)
    kv_offset_y = offset_y + lo
    return start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y


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
def _mask_scalar(qk, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s
    col_lim_right_cur = max(col_lim_right_s, 0)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return tl.where(mask_i_bit, qk, -float("inf"))


@triton.jit
def _apply_causal_mask(qk, col_limit_right, HEAD_DIM: tl.constexpr):
    # Apply causal mask via a bitmask calculated for each block of 16 elements.
    # This allows the efficient R2P (register to predicate) instruction to be used at the SASS level.
    # Credit to Tri Dao,
    # https://github.com/Dao-AILab/flash-attention/commit/bac1001e4f6caa09d70537495d6746a685a2fa78
    #
    # NOTE: We use map_elementiwse here in order to generate an interleaved sequence of instructions
    # that processes one element of qk at a time. This improves ptxas's resulting SASS.
    offs_n = tl.arange(0, HEAD_DIM)[None, :]
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
    N_CTX,
    out_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_QK: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    STAGE: tl.constexpr,
):
    lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)

    for start_n in tl.range(lo, hi, BLOCK_N):
        qk_bufIdx, qk_phase = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
        qk_bufIdx += cid * NUM_BUFFERS_QK

        tlx.barrier_wait(tlx.local_view(qk_fulls, qk_bufIdx), qk_phase)
        qk = tlx.local_load(tlx.local_view(qk_tiles, qk_bufIdx))

        if STAGE == 2:
            col_limit_right = (offs_m - start_n + 1)[:, None]
            qk = _apply_causal_mask(qk, col_limit_right, HEAD_DIM)

        # compute m_i, p in registers
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        tlx.barrier_wait(tlx.local_view(alpha_empties, qk_bufIdx), qk_phase ^ 1)
        # Use alpha[0] for cid=0, and alpha[HEAD_DIM * NUM_BUFFERS_QK] for cid=1
        tlx.local_store(
            tlx.local_view(alpha_tiles, cid * HEAD_DIM * NUM_BUFFERS_QK), alpha[:, None]
        )
        tlx.barrier_arrive(tlx.local_view(alpha_fulls, qk_bufIdx))

        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        p = p.to(out_dtype)

        # prepare p for the v dot
        # Use p[1] for cid=0, and p[3] for cid=1
        p_bufIdx = 1 + cid * NUM_MMA_GROUPS * NUM_BUFFERS_QK
        tlx.local_store(tlx.local_view(p_tiles, p_bufIdx), p)
        tlx.barrier_arrive(tlx.local_view(p_fulls, qk_bufIdx))

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        accum_cnt_qk += 1

    return m_i, l_i, accum_cnt_qk


@triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "STAGE"])
@triton.jit
def _attn_fwd_ws(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    NUM_BUFFERS_Q: tl.constexpr,  #
    NUM_BUFFERS_KV: tl.constexpr,  #
    NUM_BUFFERS_QK: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(NUM_MMA_GROUPS == 2)
    tl.static_assert(NUM_BUFFERS_QK == 1)

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS
    )
    kv_tiles = tlx.local_alloc(
        (BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV
    )

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    # allocate TMEM buffers and barriers
    qk_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, HEAD_DIM),
        tl.float32,
        NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
    )
    # Shared buffer for QK, P and Alpha, l, and m.
    # Alpha/l/m lives in the lower half of qk_buf, and P lives in the upper half.
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, HEAD_DIM),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS * NUM_BUFFERS_QK * 2,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    alpha_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        HEAD_DIM * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    l_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        HEAD_DIM * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    m_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        HEAD_DIM * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )

    acc_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, HEAD_DIM),
        tl.float32,
        NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
    )

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    acc_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)

    alpha_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    alpha_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    l_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    with tlx.async_tasks():
        # correction group
        with tlx.async_task("default"):
            # initialize offsets
            start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                H,
                N_CTX,
                BLOCK_M,
                STAGE,
            )
            accum_cnt = 0
            buf_idx = 0
            phase = 0

            for _ in tl.range(lo, hi, BLOCK_N):
                buf_idx, phase = _get_bufidx_phase(accum_cnt, NUM_BUFFERS_QK)
                for cid in tl.range(
                    0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS
                ):
                    buf_idx_2 = buf_idx + cid * NUM_BUFFERS_QK

                    # -- update output accumulator --
                    tlx.barrier_wait(alpha_fulls[buf_idx_2], phase)
                    # Use alpha[0] for cid=0, and alpha[HEAD_DIM * NUM_BUFFERS_QK] for cid=1
                    alpha_1 = tlx.local_load(
                        alpha_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK]
                    )
                    tlx.barrier_arrive(alpha_empties[buf_idx_2])

                    acc = tlx.local_load(acc_tiles[buf_idx_2])
                    acc = acc * alpha_1
                    tlx.local_store(acc_tiles[buf_idx_2], acc)
                    tlx.barrier_arrive(acc_fulls[buf_idx_2])
                accum_cnt += 1

            for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                # epilogue
                tlx.barrier_wait(l_fulls[cid], 0)
                # Use l[1]/l[1+HEAD_DIM * NUM_BUFFERS_QK] and m[2][2 + HEAD_DIM * NUM_BUFFERS_QK]
                # to disambigulate from alpha[0]/alpha[HEAD_DIM * NUM_BUFFERS_QK]
                l = tlx.local_load(l_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK + 1])
                m = tlx.local_load(m_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK + 2])
                m += tl.math.log2(l)
                offs_m = (
                    start_m * BLOCK_M
                    + cid * BLOCK_M_SPLIT
                    + tl.arange(0, BLOCK_M_SPLIT)
                )
                m_ptrs = M + off_hz * N_CTX + offs_m
                tl.store(m_ptrs, tl.reshape(m, [BLOCK_M_SPLIT]))

                tlx.barrier_wait(acc_empties[cid], 0)
                acc = tlx.local_load(acc_tiles[cid])
                acc = acc / l
                qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                desc_o.store([qo_offset_y_split, 0], acc.to(tlx.dtype_of(desc_o)))

        # softmax groups
        with tlx.async_task(num_warps=4, registers=152, replicate=NUM_MMA_GROUPS):
            # initialize offsets
            start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                H,
                N_CTX,
                BLOCK_M,
                STAGE,
            )
            # initialize pointer to m and l
            m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
            qk_scale = sm_scale
            qk_scale *= 1.44269504  # 1/log(2)

            accum_cnt_qk = 0
            out_dtype = tlx.dtype_of(desc_v)

            cid = tlx.async_task_replica_id()
            offs_m = start_m * BLOCK_M + (
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
                    N_CTX,
                    out_dtype,
                    BLOCK_M,
                    BLOCK_N,
                    HEAD_DIM,
                    NUM_BUFFERS_QK,
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
                    N_CTX,
                    out_dtype,
                    BLOCK_M,
                    BLOCK_N,
                    HEAD_DIM,
                    NUM_BUFFERS_QK,
                    NUM_MMA_GROUPS,
                    STAGE=2,
                )

            # prepare l_i for the epilog
            # Use l[1]/l[1+HEAD_DIM * NUM_BUFFERS_QK] and m[2][2 + HEAD_DIM * NUM_BUFFERS_QK]
            # to disambigulate from alpha[0]/alpha[HEAD_DIM * NUM_BUFFERS_QK]
            tlx.local_store(l_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK + 1], l_i[:, None])
            tlx.local_store(m_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK + 2], m_i[:, None])
            tlx.barrier_arrive(l_fulls[cid])

        # mma group
        with tlx.async_task(num_warps=1, registers=24):
            _, _, lo, hi, _, _ = _compute_offsets(H, N_CTX, BLOCK_M, STAGE)

            # loop over k, v and update accumulator
            accum_cnt_kv = 0
            accum_cnt_qk = 0
            k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
            v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

            # -- compute q @ k ----
            # wait for the K buffer to be populated by the producer
            tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
            tlx.barrier_wait(q_fulls[0], 0)

            k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
            _, qk_phase = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
            tlx.async_dot(
                q_tiles[0],
                k_tile,
                qk_tiles[0],
                use_acc=False,
                mBarriers=[qk_fulls[0]],
            )

            tlx.barrier_wait(q_fulls[1], 0)
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
            tlx.barrier_wait(p_fulls[0], qk_phase)
            tlx.barrier_wait(acc_fulls[0], qk_phase)
            # As p shares the second half of the qk buffer, use p[2]/p[3] instead of p[0]/p[1]
            tlx.async_dot(
                p_tiles[1],
                kv_tiles[v_bufIdx],
                acc_tiles[0],
                use_acc=False,
            )

            acc1_init = False

            for i in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                v_bufIdx_prev = v_bufIdx
                qk_phase_prev = qk_phase

                accum_cnt_qk += 1
                accum_cnt_kv += 2
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

                # -- compute q0 @ k ----
                _, qk_phase = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
                tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                tlx.async_dot(
                    q_tiles[0],
                    k_tile,
                    qk_tiles[0],
                    use_acc=False,
                    mBarriers=[qk_fulls[0]],
                )

                # -- compute p1 @ v from the previous iteration----
                tlx.barrier_wait(p_fulls[1], qk_phase_prev)
                tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
                tlx.async_dot(
                    p_tiles[3],
                    kv_tiles[v_bufIdx_prev],
                    acc_tiles[1],
                    use_acc=acc1_init,
                    mBarriers=[kv_empties[v_bufIdx_prev]],
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
                tlx.barrier_wait(p_fulls[0], qk_phase)
                tlx.barrier_wait(acc_fulls[0], qk_phase)
                tlx.async_dot(
                    p_tiles[1],
                    kv_tiles[v_bufIdx],
                    acc_tiles[0],
                    use_acc=True,
                )

            tlx.tcgen05_commit(acc_empties[0])

            # -- compute p1 @ v ----
            tlx.barrier_wait(p_fulls[1], qk_phase)
            tlx.barrier_wait(acc_fulls[1], qk_phase)
            tlx.async_dot(
                p_tiles[3],
                kv_tiles[v_bufIdx],
                acc_tiles[1],
                use_acc=acc1_init,
                mBarriers=[acc_empties[1], kv_empties[v_bufIdx]],
            )

        # load
        with tlx.async_task(num_warps=1, registers=24):
            # initialize offsets
            start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                H,
                N_CTX,
                BLOCK_M,
                STAGE,
            )

            # load q0
            tlx.barrier_expect_bytes(
                q_fulls[0], 2 * BLOCK_M_SPLIT * HEAD_DIM
            )  # float16
            qo_offset_y_split = qo_offset_y
            tlx.async_descriptor_load(
                desc_q, q_tiles[0], [qo_offset_y_split, 0], q_fulls[0]
            )

            # loop over loading k, v
            accum_cnt_kv = 0
            k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
            # wait for the K buffer to be released by the consumer
            k_empty = tlx.local_view(kv_empties, k_bufIdx)
            tlx.barrier_wait(k_empty, k_phase ^ 1)

            # load K
            k_full = tlx.local_view(kv_fulls, k_bufIdx)
            k_tile = tlx.local_view(kv_tiles, k_bufIdx)
            tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
            tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

            # load q1
            tlx.barrier_expect_bytes(
                q_fulls[1], 2 * BLOCK_M_SPLIT * HEAD_DIM
            )  # float16
            qo_offset_y_split = qo_offset_y + BLOCK_M_SPLIT
            tlx.async_descriptor_load(
                desc_q, q_tiles[1], [qo_offset_y_split, 0], q_fulls[1]
            )

            v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
            # wait for the V buffer to be released by the consumer
            v_empty = tlx.local_view(kv_empties, v_bufIdx)
            tlx.barrier_wait(v_empty, v_phase ^ 1)
            # load V
            v_full = tlx.local_view(kv_fulls, v_bufIdx)
            v_tile = tlx.local_view(kv_tiles, v_bufIdx)
            tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
            tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

            kv_offset_y += BLOCK_N
            accum_cnt_kv += 2

            for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                # wait for the K buffer to be released by the consumer
                k_empty = tlx.local_view(kv_empties, k_bufIdx)
                tlx.barrier_wait(k_empty, k_phase ^ 1)
                # load K
                k_full = tlx.local_view(kv_fulls, k_bufIdx)
                k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(kv_empties, v_bufIdx)
                tlx.barrier_wait(v_empty, v_phase ^ 1)
                # load V
                v_full = tlx.local_view(kv_fulls, v_bufIdx)
                v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                kv_offset_y += BLOCK_N
                accum_cnt_kv += 2


@triton.jit
def _compute_offsets_persistent(tile_idx, n_tile_num, H, N_CTX, BLOCK_M, STAGE):
    start_m = tile_idx % n_tile_num
    off_hz = tile_idx // n_tile_num
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    lo, hi = _get_fused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)
    kv_offset_y = offset_y + lo
    return start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y


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
def _pipelined_softmax_inner_loop(
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
    N_CTX,
    out_dtype,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    STAGE: tl.constexpr,
):
    lo, hi = _get_unfused_loop_bounds(start_m, N_CTX, BLOCK_M, STAGE)

    for start_n in tl.range(lo, hi, BLOCK_N):
        _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
        tlx.barrier_wait(tlx.local_view(qk_fulls, cid), qk_phase)
        qk = tlx.local_load(tlx.local_view(qk_tiles, cid))

        if STAGE == 2:
            col_limit_right = (offs_m - start_n + 1)[:, None]
            qk = _apply_causal_mask(qk, col_limit_right, HEAD_DIM)

        # compute m_i, p in registers
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        tlx.barrier_wait(tlx.local_view(alpha_empties, cid), qk_phase ^ 1)
        # Use alpha[0] for cid=0, and alpha[HEAD_DIM] for cid=1
        tlx.local_store(tlx.local_view(alpha_tiles, cid * HEAD_DIM), alpha[:, None])
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


@triton.autotune(
    configs=configs_persistent, key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "STAGE"]
)
@triton.jit
def _attn_fwd_ws_persistent(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    NUM_BUFFERS_Q: tl.constexpr,  #
    NUM_BUFFERS_KV: tl.constexpr,  #
    NUM_BUFFERS_QK: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    NUM_MMA_SLICES: tl.constexpr,  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(NUM_MMA_GROUPS == 2)
    tl.static_assert(NUM_BUFFERS_QK == 1)
    tl.static_assert(NUM_BUFFERS_Q == 1)

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2

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

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS * NUM_BUFFERS_Q
    )
    kv_tiles = tlx.local_alloc(
        (BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV
    )
    o_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_o), NUM_MMA_GROUPS
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
        (BLOCK_M_SPLIT, HEAD_DIM), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem
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
            for i in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = (
                    _compute_offsets_persistent(
                        tile_idx, n_tile_num, H, N_CTX, BLOCK_M, STAGE
                    )
                )
                for _ in tl.range(lo, hi, BLOCK_N):
                    _, phase = _get_bufidx_phase(accum_cnt, 1)
                    for cid in tl.static_range(0, NUM_MMA_GROUPS):
                        # -- update output accumulator --
                        tlx.barrier_wait(alpha_fulls[cid], phase)
                        # Use alpha[0] for cid=0, and alpha[HEAD_DIM] for cid=1
                        alpha_1 = tlx.local_load(alpha_tiles[cid * HEAD_DIM])
                        tlx.barrier_arrive(alpha_empties[cid])
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            subslice = tlx.subslice(
                                acc_tiles[cid],
                                HEAD_DIM * slice_id // NUM_MMA_SLICES,
                                HEAD_DIM // NUM_MMA_SLICES,
                            )
                            acc = tlx.local_load(subslice)
                            # acc = acc * alpha_1
                            acc = _mul_f32x2(acc, alpha_1)
                            tlx.local_store(subslice, acc)
                        tlx.barrier_arrive(acc_fulls[cid])
                    accum_cnt += 1

                _, phase = _get_bufidx_phase(i, 1)
                for cid in tl.static_range(0, NUM_MMA_GROUPS):
                    # epilogue
                    tlx.barrier_wait(l_fulls[cid], phase)
                    # Use l[1]/l[1+HEAD_DIM] and m[2][2 + HEAD_DIM]
                    # to disambigulate from alpha[0]/alpha[HEAD_DIM]
                    l = tlx.local_load(l_tiles[cid * HEAD_DIM + 1])
                    tlx.barrier_arrive(qk_empties[cid])
                    m = tlx.local_load(m_tiles[cid * HEAD_DIM + 2])
                    m += tl.math.log2(l)
                    offs_m = (
                        start_m * BLOCK_M
                        + cid * BLOCK_M_SPLIT
                        + tl.arange(0, BLOCK_M_SPLIT)
                    )
                    m_ptrs = M + off_hz * N_CTX + offs_m
                    tl.store(m_ptrs, tl.reshape(m, [BLOCK_M_SPLIT]))

                    tlx.barrier_wait(acc_empties[cid], phase)
                    tlx.barrier_wait(o_empties[cid], phase ^ 1)
                    scale = 1 / l
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        subslice = tlx.subslice(
                            acc_tiles[cid],
                            HEAD_DIM * slice_id // NUM_MMA_SLICES,
                            HEAD_DIM // NUM_MMA_SLICES,
                        )
                        acc = tlx.local_load(subslice)
                        acc = _mul_f32x2(acc, scale)
                        acc = acc.to(tlx.dtype_of(desc_o))
                        subslice_o = tlx.local_slice(
                            o_tiles[cid],
                            [0, HEAD_DIM * slice_id // NUM_MMA_SLICES],
                            [BLOCK_M_SPLIT, HEAD_DIM // NUM_MMA_SLICES],
                        )
                        tlx.local_store(subslice_o, acc)
                    tlx.barrier_arrive(o_fulls[cid])

                tile_idx += num_progs

        # softmax groups
        with tlx.async_task(num_warps=4, registers=168, replicate=NUM_MMA_GROUPS):
            accum_cnt_qk = 0
            for i in range(0, tiles_per_sm):
                # initialize offsets
                start_m, off_hz, _, _, qo_offset_y, kv_offset_y = (
                    _compute_offsets_persistent(
                        tile_idx, n_tile_num, H, N_CTX, BLOCK_M, STAGE
                    )
                )
                # initialize pointer to m and l
                m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
                qk_scale = sm_scale
                qk_scale *= 1.44269504  # 1/log(2)
                out_dtype = tlx.dtype_of(desc_v)

                cid = tlx.async_task_replica_id()
                offs_m = (start_m * BLOCK_M) + (
                    (cid * BLOCK_M_SPLIT) + tl.arange(0, BLOCK_M_SPLIT)
                )

                if STAGE & 1:
                    m_i, l_i, accum_cnt_qk = _pipelined_softmax_inner_loop(
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
                        N_CTX,
                        out_dtype,
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM,
                        NUM_MMA_SLICES,
                        NUM_MMA_GROUPS,
                        STAGE=4 - STAGE,
                    )

                if STAGE & 2:
                    m_i, l_i, accum_cnt_qk = _pipelined_softmax_inner_loop(
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
                        N_CTX,
                        out_dtype,
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM,
                        NUM_MMA_SLICES,
                        NUM_MMA_GROUPS,
                        STAGE=2,
                    )

                # prepare l_i for the epilog
                # Use l[1]/l[1+HEAD_DIM] and m[2][2 + HEAD_DIM]
                # to disambigulate from alpha[0]/alpha[HEAD_DIM]
                tlx.local_store(l_tiles[cid * HEAD_DIM + 1], l_i[:, None])
                tlx.local_store(m_tiles[cid * HEAD_DIM + 2], m_i[:, None])
                tlx.barrier_arrive(l_fulls[cid])
                tile_idx += num_progs

            # mma group
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            accum_cnt_qk = 0

            for j in range(0, tiles_per_sm):
                # initialize offsets
                _, _, lo, hi, _, _ = _compute_offsets_persistent(
                    tile_idx, n_tile_num, H, N_CTX, BLOCK_M, STAGE
                )

                q_bufIdx, q_phase = _get_bufidx_phase(j, NUM_BUFFERS_Q)
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

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
                tlx.barrier_wait(acc_fulls[0], qk_phase)
                # Use p[NUM_MMA_SLICES + slice_id] for cid=0, and
                # p[NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id] for cid=1
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    tlx.barrier_wait(p_fulls[slice_id + 0 * NUM_MMA_SLICES], qk_phase)
                    kv_slice = tlx.local_slice(
                        kv_tiles[v_bufIdx],
                        [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                        [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
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
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
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
                    tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        tlx.barrier_wait(
                            p_fulls[slice_id + 1 * NUM_MMA_SLICES], qk_phase_prev
                        )
                        kv_slice = tlx.local_slice(
                            kv_tiles[v_bufIdx_prev],
                            [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                            [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
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

                    tlx.barrier_wait(acc_fulls[0], qk_phase)
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        tlx.barrier_wait(
                            p_fulls[slice_id + 0 * NUM_MMA_SLICES], qk_phase
                        )
                        # Use p[1] for cid=0, and p[3] for cid=1
                        kv_slice = tlx.local_slice(
                            kv_tiles[v_bufIdx],
                            [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                            [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
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
                tlx.barrier_wait(acc_fulls[1], qk_phase)
                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                    tlx.barrier_wait(p_fulls[slice_id + NUM_MMA_SLICES], qk_phase)
                    # Use p[1] for cid=0, and p[3] for cid=1
                    kv_slice = tlx.local_slice(
                        kv_tiles[v_bufIdx],
                        [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                        [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                    )
                    p_bufIdx = (
                        1 * NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id
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
                tile_idx += num_progs

        # load
        with tlx.async_task(num_warps=1, registers=24):
            accum_cnt_kv = 0
            for i in range(0, tiles_per_sm):
                # initialize offsets
                _, _, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets_persistent(
                    tile_idx, n_tile_num, H, N_CTX, BLOCK_M, STAGE
                )

                # load q0
                q_bufIdx, q_phase = _get_bufidx_phase(i, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(
                    q_fulls[q_bufIdx], 2 * BLOCK_M_SPLIT * HEAD_DIM
                )  # float16
                qo_offset_y_split = qo_offset_y
                tlx.async_descriptor_load(
                    desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx]
                )

                # loop over loading k, v
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                # wait for the K buffer to be released by the consumer
                k_empty = tlx.local_view(kv_empties, k_bufIdx)
                tlx.barrier_wait(k_empty, k_phase ^ 1)

                # load K
                k_full = tlx.local_view(kv_fulls, k_bufIdx)
                k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                # load q1
                q_bufIdx += NUM_BUFFERS_Q
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(
                    q_fulls[q_bufIdx], 2 * BLOCK_M_SPLIT * HEAD_DIM
                )  # float16
                qo_offset_y_split = qo_offset_y + BLOCK_M_SPLIT
                tlx.async_descriptor_load(
                    desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx]
                )

                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(kv_empties, v_bufIdx)
                tlx.barrier_wait(v_empty, v_phase ^ 1)
                # load V
                v_full = tlx.local_view(kv_fulls, v_bufIdx)
                v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                kv_offset_y += BLOCK_N
                accum_cnt_kv += 2

                for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # wait for the K buffer to be released by the consumer
                    k_empty = tlx.local_view(kv_empties, k_bufIdx)
                    tlx.barrier_wait(k_empty, k_phase ^ 1)
                    # load K
                    k_full = tlx.local_view(kv_fulls, k_bufIdx)
                    k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                    tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                    tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                    v_bufIdx, v_phase = _get_bufidx_phase(
                        accum_cnt_kv + 1, NUM_BUFFERS_KV
                    )
                    # wait for the V buffer to be released by the consumer
                    v_empty = tlx.local_view(kv_empties, v_bufIdx)
                    tlx.barrier_wait(v_empty, v_phase ^ 1)
                    # load V
                    v_full = tlx.local_view(kv_fulls, v_bufIdx)
                    v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                    tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                    tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                    kv_offset_y += BLOCK_N
                    accum_cnt_kv += 2

                tile_idx += num_progs

        # epilog group
        with tlx.async_task(num_warps=1, registers=24):
            # initialize offsets
            for i in range(0, tiles_per_sm):
                # initialize offsets
                _, _, _, _, qo_offset_y, kv_offset_y = _compute_offsets_persistent(
                    tile_idx, n_tile_num, H, N_CTX, BLOCK_M, STAGE
                )
                _, phase = _get_bufidx_phase(i, 1)
                for cid in tl.static_range(0, NUM_MMA_GROUPS):
                    tlx.barrier_wait(o_fulls[cid], phase)
                    tlx.fence_async_shared()
                    qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                    tlx.async_descriptor_store(
                        desc_o, o_tiles[cid], [qo_offset_y_split, 0]
                    )
                    tlx.async_descriptor_store_wait(0)
                    tlx.barrier_arrive(o_empties[cid])

                tile_idx += num_progs


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, use_persistent=False):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        stage = 3 if causal else 1

        o = torch.empty_like(q)
        extra_kern_args = {}

        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            q,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        if q.dtype == torch.float8_e5m2:
            desc_v = TensorDescriptor(
                v,
                shape=[HEAD_DIM_K, y_dim],
                strides=[q.shape[2], 1],
                block_shape=dummy_block,
            )
        else:
            desc_v = TensorDescriptor(
                v,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
        desc_k = TensorDescriptor(
            k,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )
        desc_o = TensorDescriptor(
            o,
            shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (
                triton.cdiv(q.shape[2], META["BLOCK_M"]),
                q.shape[0] * q.shape[1],
                1,
            )

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        def grid_persistent(META):
            return (
                min(
                    NUM_SMS,
                    triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1],
                ),
                1,
                1,
            )

        # persistent kernel
        if use_persistent:
            ctx.grid = grid_persistent
            _attn_fwd_ws_persistent[grid_persistent](
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
                **extra_kern_args,
            )
        else:
            ctx.grid = grid
            _attn_fwd_ws[grid](
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
                **extra_kern_args,
            )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o


attention = _attention.apply
