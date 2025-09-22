# TLX GDPA kernel optimized for Blackwell Warp Specialization

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

from .gdpa_utils import get_num_sms
from .math import activation_string_to_int, fast_gelu_grad, gelu, gelu_grad


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_D = nargs["BLOCK_D"]
    if not isinstance(nargs["Q"], TensorDescriptor):
        # early return for on-device TMA
        return
    NUM_MMA_GROUPS = 2
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["Q"].block_shape = [BLOCK_M_SPLIT, BLOCK_D]
    nargs["V"].block_shape = [BLOCK_N, BLOCK_D]
    nargs["K"].block_shape = [BLOCK_N, BLOCK_D]
    nargs["Out"].block_shape = [BLOCK_M_SPLIT, BLOCK_D]


# Pick one configuration if ENABLE_PROTON.
def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "NUM_BUFFERS_Q": bq,
                "NUM_BUFFERS_KV": bkv,
                "NUM_BUFFERS_QK": bqk,
                "NUM_BUFFERS_O": bo,
                "SUBTILING": SUBTILE,
                "PINGPONG": pp,
                "ACT_REGS": ar,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=_host_descriptor_pre_hook,
        )
        for BM in [256]  # 128 or 256
        for BN in [128]
        for bq in [1]
        for bkv in [3]
        for bqk in [1]  # in tmem
        for bo in [1]  # in tmem
        for SUBTILE in [True]  # doesn't support False
        for pp in [True, False]
        for ar in [192, 232]
    ]


## Iterative tuning with intra-kernel profiler
## 1. identify critical resource
## 2. assuming it is gemm, make sure there is no bubble in gemm partition

## Potential issues
## -- bubbles in gemm partition due to _compute_qlen
## ---- if that is the case via intra-kernel profiler, try pre-compute _compute_qlen
## -- load imbalance
## ---- use dynamic scheduler
## ---- grab the next tile one iteration ahead (i.e SWP of the outer loop)
## -- if descriptor setup is an issue, try SWP the setup for inner loop (i.e desc_k,v)


## Overall warpspec configuration
## default + 3 partitions:
##   default is activation0 with 4 warps, partition0 is activatation1 with 4 warps
##   partition1 is gemm, partition 2 is load
@triton.jit
def _compute_qlen(
    tile_idx,
    n_tile_num,
    Q_offsets,
    K_offsets,
    seq_index,
    SORT_BY_SEQ_LENGTH: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
):
    off_hz = tile_idx // n_tile_num
    off_z = off_hz // H
    if SORT_BY_SEQ_LENGTH:
        off_z = tl.load(seq_index + off_z)
    off_q_z = off_z
    begin_q = tl.load(Q_offsets + off_q_z)
    end_q = tl.load(Q_offsets + off_q_z + 1)

    qlen = end_q - begin_q
    qlen = tl.minimum(qlen, N_CTX)

    begin_k = tl.load(K_offsets + off_z)
    end_k = tl.load(K_offsets + off_z + 1)
    klen = end_k - begin_k

    return begin_q, end_q, begin_k, qlen, klen


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


@triton.jit
def _load_tma(
    bufIdx, phase, empty_bars, full_bars, buffers, desc, offset_1, offset_0, num_bytes
):
    # producer acquire
    empty_view = tlx.local_view(empty_bars, bufIdx)
    tlx.barrier_wait(empty_view, phase ^ 1)
    # barrier for producer commit
    full_view = tlx.local_view(full_bars, bufIdx)
    tlx.barrier_expect_bytes(full_view, num_bytes)
    smem_view = tlx.local_view(buffers, bufIdx)
    tlx.async_descriptor_load(
        desc,
        smem_view,
        [
            (offset_1).to(tl.int32),
            (offset_0).to(tl.int32),
        ],
        full_view,
    )

    return smem_view


# Block sizes: 128 x 128
# Barriers:
#   producer_acquire uses the same barrier as consumer_release
#   producer_commit uses the same barriers as consumer_wait
# Channels:
#   If consumer of the channel, will have two barriers consumer_x and consumer_release_x
#   If producer of the channel, will have two barriers producer_x and producer_commit_x
#   q0, q1, k, v: consumers of the channels
#   qk0, qk1: producers
#   p0, p1: sharing tmem spaces, and barriers with qk0, qk1 (consumers)
#   o0, o1


@triton.jit
def _add_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            add.f32x2 rc, ra, rb;
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


# typical configuration is 3/fast_gelu
@triton.jit
def fast_gelu(x):
    # following D80750725
    # WAS: x * 0.5 * (1 + tanh_approx_fp32(0.7978845608 * x * (1.0 + 0.044715 * x * x))) * scaling
    # NOW: x * tanh((c1 * x * x + c0)*x) + x
    c1 = 0.0356774081
    c0 = 0.7978845608
    square = _mul_f32x2(x, x)
    inner = _fma_f32x2(c1, square, c0)
    inner = _mul_f32x2(inner, x)
    out = _fma_f32x2(x, tanh_approx_fp32(inner), x)
    return out


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["N_CTX", "HEAD_DIM", "H", "G", "FUSED_QKV", "FUSED_KV"],
)
@triton.jit
def gdpa_kernel_tma_ws_blackwell(
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
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
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
    USE_ON_DEVICE_TMA: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_QK: tl.constexpr,
    NUM_BUFFERS_O: tl.constexpr,
    SUBTILING: tl.constexpr,
    PINGPONG: tl.constexpr,
    ACT_REGS: tl.constexpr,
    NAME_BARRIER: tl.constexpr,
    MERGE_EPI: tl.constexpr,
    ENABLE_PROTON: tl.constexpr,
    PROTON_TILE: tl.constexpr,
):
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    if not USE_ON_DEVICE_TMA:
        q_desc = Q
        k_desc = K
        v_desc = V
        o_desc = Out

    # start with on-device TMA where descriptors for k, v are set up outside of the persistent
    # loop and descriptor for q is set up inside the persistent loop.
    if USE_ON_DEVICE_TMA:
        k_desc = tl.make_tensor_descriptor(
            K,
            shape=[N_CTX_KV * Z, HEAD_DIM * H // G],
            strides=[HEAD_DIM * H // G, 1],
            block_shape=[BLOCK_N, BLOCK_D],
        )
        v_desc = tl.make_tensor_descriptor(
            V,
            shape=[N_CTX_KV * Z, HEAD_DIM * H // G],
            strides=[HEAD_DIM * H // G, 1],
            block_shape=[BLOCK_N, BLOCK_D],
        )

    if USE_ON_DEVICE_TMA:
        dtype = V.dtype.element_ty
    else:
        dtype = tlx.dtype_of(v_desc)

    # allocate buffers for q0, q1
    q0_buf = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), dtype, 1)
    q1_buf = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), dtype, 1)

    # allocate buffers for k, v
    kv_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), dtype, NUM_BUFFERS_KV)  # k
    o0_smem = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), dtype, 1)
    o1_smem = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), dtype, 1)

    # allocate tmem for outputs of 4 dots (after partitioning)
    # qk0 = q0 dot k, qk1 = q1 dot k, acc0 = p0 dot v, acc1 = p1 dot v
    qk0_buf = tlx.local_alloc(
        (BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem
    )
    qk1_buf = tlx.local_alloc(
        (BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem
    )
    p0_buf = tlx.local_alloc(
        (BLOCK_M // 2, HEAD_DIM), dtype, 1, tlx.storage_kind.tmem, reuse=qk0_buf
    )
    p1_buf = tlx.local_alloc(
        (BLOCK_M // 2, HEAD_DIM), dtype, 1, tlx.storage_kind.tmem, reuse=qk1_buf
    )
    o0_buf = tlx.local_alloc(
        (BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem
    )
    o1_buf = tlx.local_alloc(
        (BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem
    )

    # allocate barriers
    consumer_q0 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    consumer_q1 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    consumer_release_q0 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    consumer_release_q1 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    consumer_kv = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=1)
    consumer_release_kv = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_KV, arrive_count=1
    )
    tlx.barrier_arrive(consumer_release_kv[0], 1)
    tlx.barrier_arrive(consumer_release_kv[1], 1)
    tlx.barrier_arrive(consumer_release_kv[2], 1)

    producer_qk0 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_QK, arrive_count=1)
    producer_commit_qk0 = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_QK, arrive_count=1
    )
    producer_qk1 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_QK, arrive_count=1)
    producer_commit_qk1 = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_QK, arrive_count=1
    )

    producer_o0 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    producer_commit_o0 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    producer_o1 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    producer_commit_o1 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    producer_pp = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    producer_commit_pp = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)

    with tlx.async_tasks():
        # activation calculation
        with tlx.async_task("default", registers=ACT_REGS):
            accum_cnt = 0
            accum_cnt_outer = 0
            for idx in range(0, tiles_per_sm):
                if ENABLE_PROTON and idx == PROTON_TILE:
                    pl.enter_scope("ele0_tile")
                begin_q, end_q, begin_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                )
                pid = tile_idx % n_tile_num
                start_m = pid
                off_hz = tile_idx // n_tile_num
                off_h = off_hz % H
                out_offset = off_h.to(tl.int64) * stride_oh
                if start_m * BLOCK_M < qlen:
                    lo, hi = 0, klen
                    # tl.device_print("default", hi)
                    for start_n in range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        # tl.device_print("default start_n", start_n)
                        bufIdx = accum_cnt % NUM_BUFFERS_QK
                        phase = (accum_cnt // NUM_BUFFERS_QK) & 1
                        qk_view = tlx.local_view(qk0_buf, bufIdx)
                        consumer_qk_view = tlx.local_view(producer_commit_qk0, bufIdx)
                        # tl.device_print("default producer_commit_qk0", accum_cnt)
                        # tl.device_print("default producer_commit_qk0_phase", phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("consumer_qk_view")
                        tlx.barrier_wait(consumer_qk_view, phase)

                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("consumer_qk_view")
                        # qk_view: BLOCK_M // 2, HEAD_DIM
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("elementwise_0")
                        qk_view_1st = tlx.subslice(qk_view, 0, HEAD_DIM // 2)
                        qk0 = tlx.local_load(qk_view_1st)
                        qk_view_2nd = tlx.subslice(
                            qk_view, HEAD_DIM // 2, HEAD_DIM // 2
                        )
                        qk1 = tlx.local_load(qk_view_2nd)
                        c1 = 0.0356774081
                        c0 = 0.7978845608
                        square = _mul_f32x2(qk0, qk0)
                        inner = _fma_f32x2(c1, square, c0)
                        inner0 = _mul_f32x2(inner, qk0)
                        square = _mul_f32x2(qk1, qk1)
                        inner = _fma_f32x2(c1, square, c0)
                        inner1 = _mul_f32x2(inner, qk1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("elementwise_0")
                        if PINGPONG:
                            if NAME_BARRIER:
                                tlx.named_barrier_wait(9, 128)
                            else:
                                # assume NUM_BUFFERS_QK is 1
                                tlx.barrier_wait(producer_pp[0], phase ^ 1)  # acquire
                        # p0 = fast_gelu(qk0)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("tanh")
                        p0 = _fma_f32x2(qk0, tanh_approx_fp32(inner0), qk0)
                        p0 = p0.to(dtype)
                        p0_view = tlx.local_view(p0_buf, bufIdx)
                        p0_view_1st = tlx.subslice(p0_view, 0, HEAD_DIM // 2)
                        tlx.local_store(p0_view_1st, p0)

                        # p1 = fast_gelu(qk1)
                        p1 = _fma_f32x2(qk1, tanh_approx_fp32(inner1), qk1)
                        p1 = p1.to(dtype)
                        p0_view_2nd = tlx.subslice(
                            p0_view, HEAD_DIM // 2, HEAD_DIM // 2
                        )
                        tlx.local_store(p0_view_2nd, p1)

                        # p and qk reuse tmem space, single producer commit for p via consumer_release_qk
                        consumer_release_qk_view = tlx.local_view(producer_qk0, bufIdx)
                        tlx.barrier_arrive(consumer_release_qk_view, 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("tanh")
                        if PINGPONG:
                            if NAME_BARRIER:
                                tlx.named_barrier_arrive(10, 128)
                            else:
                                tlx.barrier_arrive(producer_commit_pp[0], 1)
                        # wait for o0, o1 per iteration
                        bufIdx = accum_cnt % NUM_BUFFERS_O
                        phase = (accum_cnt // NUM_BUFFERS_O) & 1
                        # consumer wait of o0: producer_commit
                        consumer_o0_view = tlx.local_view(producer_commit_o0, bufIdx)
                        # tl.device_print("default producer_commit_o0", accum_cnt)
                        # tl.device_print("default producer_commit_o0_phase", phase)
                        # there is no need to wait for o0 at each iteration
                        # tlx.barrier_wait(consumer_o0_view, phase)
                        accum_cnt += 1

                    # epilogue here, load from tmem
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("elementwise_0_epi")
                    # FIXME: wait till o0 is done for the inner loop
                    bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
                        accum_cnt_outer, NUM_BUFFERS_O
                    )
                    o0_view = tlx.local_view(o0_buf, bufIdx_o_outer)
                    o0 = tlx.local_load(o0_view)
                    # release o0 here
                    consumer_release_o0_view = tlx.local_view(
                        producer_o0, bufIdx_o_outer
                    )
                    # tl.device_print("default producer_o0", accum_cnt_outer)
                    tlx.barrier_arrive(consumer_release_o0_view, 1)
                    if USE_ON_DEVICE_TMA and MERGE_EPI:
                        o_desc = tl.make_tensor_descriptor(
                            Out,
                            shape=[end_q.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_M // 2, BLOCK_D],
                        )
                    if USE_ON_DEVICE_TMA:
                        o0 = o0.to(Out.type.element_ty)
                    else:
                        o0 = o0.to(tlx.dtype_of(o_desc))
                    if MERGE_EPI:
                        o_desc.store(
                            [
                                (begin_q + start_m * BLOCK_M).to(tl.int32),
                                (out_offset).to(tl.int32),
                            ],
                            o0,
                        )
                    else:
                        tlx.local_store(o0_smem[0], o0)
                    accum_cnt_outer += 1
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("elementwise_0_epi")
                tile_idx += num_progs
                if ENABLE_PROTON and idx == PROTON_TILE:
                    pl.exit_scope("ele0_tile")

        with tlx.async_task(num_warps=4, registers=ACT_REGS):
            accum_cnt = 0
            accum_cnt_outer = 0
            if PINGPONG:
                if NAME_BARRIER:
                    tlx.named_barrier_arrive(9, 128)
            for idx in range(0, tiles_per_sm):
                if ENABLE_PROTON and idx == PROTON_TILE:
                    pl.enter_scope("ele1_tile")
                pid = tile_idx % n_tile_num
                start_m = pid
                off_hz = tile_idx // n_tile_num
                off_h = off_hz % H
                out_offset = off_h.to(tl.int64) * stride_oh
                begin_q, end_q, begin_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                )
                if start_m * BLOCK_M < qlen:
                    lo, hi = 0, klen
                    for start_n in range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        ## communication channel for qk1, p1
                        bufIdx = accum_cnt % NUM_BUFFERS_QK
                        phase = (accum_cnt // NUM_BUFFERS_QK) & 1
                        qk_view = tlx.local_view(qk1_buf, bufIdx)
                        consumer_qk_view = tlx.local_view(producer_commit_qk1, bufIdx)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("consumer_qk_view")
                        tlx.barrier_wait(consumer_qk_view, phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("consumer_qk_view")

                        # qk_view: BLOCK_M // 2, HEAD_DIM
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("elementwise_1")
                        qk_view_1st = tlx.subslice(qk_view, 0, HEAD_DIM // 2)
                        qk0 = tlx.local_load(qk_view_1st)
                        qk_view_2nd = tlx.subslice(
                            qk_view, HEAD_DIM // 2, HEAD_DIM // 2
                        )
                        qk1 = tlx.local_load(qk_view_2nd)
                        c1 = 0.0356774081
                        c0 = 0.7978845608
                        square = _mul_f32x2(qk0, qk0)
                        inner = _fma_f32x2(c1, square, c0)
                        inner0 = _mul_f32x2(inner, qk0)
                        square = _mul_f32x2(qk1, qk1)
                        inner = _fma_f32x2(c1, square, c0)
                        inner1 = _mul_f32x2(inner, qk1)

                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("elementwise_1")
                        if PINGPONG:
                            if NAME_BARRIER:
                                tlx.named_barrier_wait(10, 128)
                            else:
                                tlx.barrier_wait(
                                    producer_commit_pp[0], phase
                                )  # consumer_wait
                        # p0 = fast_gelu(qk0)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("tanh")
                        p0 = _fma_f32x2(qk0, tanh_approx_fp32(inner0), qk0)
                        p0 = p0.to(dtype)
                        p1_view = tlx.local_view(p1_buf, bufIdx)
                        p1_view_1st = tlx.subslice(p1_view, 0, HEAD_DIM // 2)
                        tlx.local_store(p1_view_1st, p0)

                        # p1 = fast_gelu(qk1)
                        p1 = _fma_f32x2(qk1, tanh_approx_fp32(inner1), qk1)
                        p1 = p1.to(dtype)
                        p1_view_2nd = tlx.subslice(
                            p1_view, HEAD_DIM // 2, HEAD_DIM // 2
                        )
                        tlx.local_store(p1_view_2nd, p1)

                        # p and qk reuse tmem space, single producer commit for p via consumer_release_qk
                        consumer_release_qk_view = tlx.local_view(producer_qk1, bufIdx)
                        tlx.barrier_arrive(consumer_release_qk_view, 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("tanh")
                        if PINGPONG:
                            if NAME_BARRIER:
                                tlx.named_barrier_arrive(9, 128)
                            else:
                                tlx.barrier_arrive(producer_pp[0], 1)

                        # wait for o0, o1 per iteration
                        bufIdx = accum_cnt % NUM_BUFFERS_O
                        phase = (accum_cnt // NUM_BUFFERS_O) & 1
                        # consumer wait of o1
                        consumer_o1_view = tlx.local_view(producer_commit_o1, bufIdx)
                        # there is no need to wait for o1 at each iteration
                        # tlx.barrier_wait(consumer_o1_view, phase)
                        accum_cnt += 1
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("elementwise_1_epi")
                    # epilogue here, load from tmem
                    # FIXME: wait till o1 is done for the inner loop
                    bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
                        accum_cnt_outer, NUM_BUFFERS_O
                    )
                    if USE_ON_DEVICE_TMA and MERGE_EPI:
                        o_desc = tl.make_tensor_descriptor(
                            Out,
                            shape=[end_q.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_M // 2, BLOCK_D],
                        )
                    o1_view = tlx.local_view(o1_buf, bufIdx_o_outer)
                    o1 = tlx.local_load(o1_view)
                    # release o1 here
                    consumer_release_o1_view = tlx.local_view(
                        producer_o1, bufIdx_o_outer
                    )
                    tlx.barrier_arrive(consumer_release_o1_view, 1)
                    if USE_ON_DEVICE_TMA:
                        o1 = o1.to(Out.type.element_ty)
                    else:
                        o1 = o1.to(tlx.dtype_of(o_desc))
                    if MERGE_EPI:
                        o_desc.store(
                            [
                                (begin_q + start_m * BLOCK_M + BLOCK_M // 2).to(
                                    tl.int32
                                ),
                                (out_offset).to(tl.int32),
                            ],
                            o1,
                        )
                    else:
                        tlx.local_store(o1_smem[0], o1)
                    accum_cnt_outer += 1
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("elementwise_1_epi")
                tile_idx += num_progs
                if ENABLE_PROTON and idx == PROTON_TILE:
                    pl.exit_scope("ele1_tile")

        with tlx.async_task(num_warps=1, registers=24):  # gemm
            accum_cnt_q = 0
            accum_cnt_kv = 0
            accum_cnt_o = 0
            accum_cnt_qk = 0
            accum_cnt_outer = 0
            for idx in range(0, tiles_per_sm):
                if ENABLE_PROTON and idx == PROTON_TILE:
                    pl.enter_scope("dot_tile")
                pid = tile_idx % n_tile_num
                start_m = pid
                begin_q, end_q, begin_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                )
                if start_m * BLOCK_M < qlen:
                    # prologue
                    bufIdx_q, phase_q = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                    bufIdx_k, phase_k = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    bufIdx_qk, phase_qk = _get_bufidx_phase(
                        accum_cnt_qk, NUM_BUFFERS_QK
                    )
                    accum_cnt_qk1 = accum_cnt_qk

                    consumer_q0_view = tlx.local_view(consumer_q0, bufIdx_q)
                    # consumer_k_view = tlx.local_view(consumer_kv, bufIdx_k)
                    # producer_qk0_view = tlx.local_view(producer_qk0, bufIdx_qk)
                    # tl.device_print("gemm consumer_q0_prologue", accum_cnt_q)
                    # tl.device_print("gemm consumer_q0_phase", phase_q)
                    tlx.barrier_wait(consumer_q0_view, phase_q)  # consumer wait for q0
                    # tl.device_print("gemm consumer_k", accum_cnt_kv)
                    # tl.device_print("gemm consumer_k_buf", bufIdx_k)
                    # tl.device_print("gemm consumer_k_phase", phase_k)
                    tlx.barrier_wait(
                        consumer_kv[bufIdx_k], phase_k
                    )  # consumer wait for k
                    # Do we need the initial acquire here?
                    # dot partition has producer commit for qk0, activation partition consumer wait for qk0
                    # activation partition producer commit for p0, dot partition has consumer wait for p0
                    # tlx.barrier_wait(producer_qk0_view, phase_qk)  # producer acquire for qk0
                    # producer commit for qk0
                    q0_view = tlx.local_view(q0_buf, bufIdx_q)
                    k_view = tlx.local_view(kv_buf, bufIdx_k)
                    qk0_view = tlx.local_view(qk0_buf, bufIdx_qk)
                    producer_commit_qk0_view = tlx.local_view(
                        producer_commit_qk0, bufIdx_qk
                    )
                    tlx.async_dot(
                        q0_view,
                        k_view,
                        qk0_view,
                        use_acc=False,
                        mBarriers=[producer_commit_qk0_view],
                    )
                    # accum_cnt_qk += 1

                    consumer_q1_view = tlx.local_view(consumer_q1, bufIdx_q)
                    # producer_qk1_view = tlx.local_view(producer_qk1, bufIdx_qk)
                    # tl.device_print("gemm consumer_q1", accum_cnt_q)
                    # tl.device_print("gemm consumer_q1_phase", phase_q)
                    tlx.barrier_wait(consumer_q1_view, phase_q)  # consumer wait for q1
                    # tlx.barrier_wait(producer_qk1_view, phase_qk)  # producer acquire for qk1
                    # consumer release for k, producer commit for qk1
                    q1_view = tlx.local_view(q1_buf, bufIdx_q)
                    qk1_view = tlx.local_view(qk1_buf, bufIdx_qk)
                    consumer_release_k_view = tlx.local_view(
                        consumer_release_kv, bufIdx_k
                    )
                    producer_commit_qk1_view = tlx.local_view(
                        producer_commit_qk1, bufIdx_qk
                    )
                    tlx.async_dot(
                        q1_view,
                        k_view,
                        qk1_view,
                        use_acc=False,
                        mBarriers=[consumer_release_k_view, producer_commit_qk1_view],
                    )
                    # tl.device_print("gemm consumer_release_k", accum_cnt_kv)
                    # tl.device_print("gemm consumer_release_k_buf", bufIdx_k)
                    # accum_cnt_qk1 += 1

                    bufIdx_v, phase_v = _get_bufidx_phase(
                        accum_cnt_kv + 1, NUM_BUFFERS_KV
                    )
                    # consumer_v_view = tlx.local_view(consumer_kv, bufIdx_v)
                    # tl.device_print("gemm consumer_v", accum_cnt_kv + 1)
                    # tl.device_print("gemm consumer_v_buf", bufIdx_v)
                    # tl.device_print("gemm consumer_v_phase", phase_v)
                    tlx.barrier_wait(
                        consumer_kv[bufIdx_v], phase_v
                    )  # consumer wait for v
                    # need to acquire o0 to make sure epilogue is done, this is needed for each outer loop
                    bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
                        accum_cnt_outer, NUM_BUFFERS_O
                    )
                    producer_o0_view = tlx.local_view(producer_o0, bufIdx_o_outer)
                    producer_o1_view = tlx.local_view(producer_o1, bufIdx_o_outer)
                    # tl.device_print("gemm producer_o0", accum_cnt_outer)
                    # tl.device_print("gemm producer_o0_phase", phase_o_outer)
                    # DEBUG_PERF
                    tlx.barrier_wait(
                        producer_o0_view, phase_o_outer ^ 1
                    )  # producer acquire for o0
                    # For reuse of qk0 and p0, we can simplify the barriers
                    #   activation partition: consumer wait for qk0, ... update p, producer commit of p0
                    #   dot partition: producer commit of qk0, ..., consumer wait for p0 (use the same barrier as producer_qk0)
                    bufIdx_p, phase_p = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
                    consumer_p0_view = tlx.local_view(producer_qk0, bufIdx_p)
                    # tl.device_print("gemm producer_qk0", accum_cnt_qk)
                    # tl.device_print("gemm producer_qk0_phase", phase_p)
                    # DEBUG_PERF_P
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("dot_wait_p0")
                    tlx.barrier_wait(
                        consumer_p0_view, phase_p
                    )  # consumer wait for p0 due to reuse of p0 and qk0
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("dot_wait_p0")
                    # reinterpret qk0 as p0
                    p0_view = tlx.local_view(p0_buf, bufIdx_p)

                    bufIdx_o, phase_o = _get_bufidx_phase(accum_cnt_o, NUM_BUFFERS_O)
                    producer_commit_o0_view = tlx.local_view(
                        producer_commit_o0, bufIdx_o
                    )
                    o0_view = tlx.local_view(o0_buf, bufIdx_o)
                    v_view = tlx.local_view(kv_buf, bufIdx_v)
                    tlx.async_dot(  # p0 . v -> o0
                        p0_view,
                        v_view,
                        o0_view,
                        use_acc=False,
                        mBarriers=[producer_commit_o0_view],
                    )
                    accum_cnt_o1 = accum_cnt_o

                    lo, hi = 0, klen
                    first = True
                    mma_iters = (hi - lo) // BLOCK_N
                    accum_cnt_kv += 2
                    accum_cnt_qk += 1
                    accum_cnt_o += 1
                    # tl.device_print("gemm for ", hi)
                    # tl.device_print("gemm mma_iters ", mma_iters)
                    for it in range(BLOCK_N, hi, BLOCK_N):
                        # for it in range(mma_iters - 1):
                        # tl.device_print("gemm iter ", it)
                        bufIdx_k, phase_k = _get_bufidx_phase(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        bufIdx_qk, phase_qk = _get_bufidx_phase(
                            accum_cnt_qk, NUM_BUFFERS_QK
                        )

                        # q0 dot k
                        # consumer_k_view = tlx.local_view(consumer_kv, bufIdx_k)
                        # tl.device_print("gemm consumer_k", accum_cnt_kv)
                        # tl.device_print("gemm consumer_k_buf", bufIdx_k)
                        # tl.device_print("gemm consumer_k_phase", phase_k)
                        tlx.barrier_wait(
                            consumer_kv[bufIdx_k], phase_k
                        )  # consumer wait for k
                        k_view = tlx.local_view(kv_buf, bufIdx_k)
                        qk0_view = tlx.local_view(qk0_buf, bufIdx_qk)
                        producer_commit_qk0_view = tlx.local_view(
                            producer_commit_qk0, bufIdx_qk
                        )
                        tlx.async_dot(
                            q0_view,
                            k_view,
                            qk0_view,
                            use_acc=False,
                            mBarriers=[producer_commit_qk0_view],
                        )

                        # p1 dot v for previous iteration
                        bufIdx_qk1, phase_qk1 = _get_bufidx_phase(
                            accum_cnt_qk1, NUM_BUFFERS_QK
                        )
                        consumer_p1_view = tlx.local_view(producer_qk1, bufIdx_qk1)
                        # tl.device_print("gemm producer_o1", accum_cnt_outer)
                        # tl.device_print("gemm producer_o1_phase", phase_o_outer)
                        # DEBUG_PERF
                        tlx.barrier_wait(
                            producer_o1_view, phase_o_outer ^ 1, first
                        )  # producer acquire for o1, only needed for first iteration
                        # tl.device_print("gemm producer_qk1", accum_cnt_qk1)
                        # tl.device_print("gemm producer_qk1_phase", phase_qk1)
                        # DEBUG_PERF_P
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("dot_wait_p1")
                        tlx.barrier_wait(
                            consumer_p1_view, phase_qk1
                        )  # consumer wait for p1 use producer_qk1 due to reuse
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("dot_wait_p1")
                        # done using v from previous iteration
                        bufIdx_o1, phase_o1 = _get_bufidx_phase(
                            accum_cnt_o1,
                            NUM_BUFFERS_O,  # previous iteration
                        )
                        o1_view = tlx.local_view(o1_buf, bufIdx_o1)
                        producer_commit_o1_view = tlx.local_view(
                            producer_commit_o1, bufIdx_o1
                        )
                        # release v for previous iteartion, accum_cnt_kv already advanced
                        bufIdx_v, phase_v = _get_bufidx_phase(
                            accum_cnt_kv - 1, NUM_BUFFERS_KV
                        )
                        consumer_release_v_view = tlx.local_view(
                            consumer_release_kv, bufIdx_v
                        )
                        # reinterpret as p1
                        p1_view = tlx.local_view(p1_buf, bufIdx_qk1)
                        tlx.async_dot(  # p1 . v from previous iteration
                            p1_view,
                            v_view,
                            o1_view,
                            use_acc=not first,
                            mBarriers=[
                                producer_commit_o1_view,
                                consumer_release_v_view,
                            ],
                        )
                        # tl.device_print("gemm consumer_release_v", accum_cnt_kv - 1)
                        # tl.device_print("gemm consumer_release_v_buf", bufIdx_v)

                        # q1 dot k, done using k for this iteration
                        bufIdx_qk1_next, phase_qk1_next = _get_bufidx_phase(
                            accum_cnt_qk1 + 1, NUM_BUFFERS_QK
                        )
                        qk1_view = tlx.local_view(qk1_buf, bufIdx_qk1_next)
                        consumer_release_k_view = tlx.local_view(
                            consumer_release_kv, bufIdx_k
                        )
                        producer_commit_qk1_view = tlx.local_view(
                            producer_commit_qk1, bufIdx_qk1_next
                        )
                        tlx.async_dot(
                            q1_view,
                            k_view,
                            qk1_view,
                            use_acc=False,
                            mBarriers=[
                                consumer_release_k_view,
                                producer_commit_qk1_view,
                            ],
                        )
                        # tl.device_print("gemm consumer_release_k", accum_cnt_kv)
                        # tl.device_print("gemm consumer_release_k_buf", bufIdx_k)

                        # p0 dot v
                        bufIdx_v, phase_v = _get_bufidx_phase(
                            accum_cnt_kv + 1, NUM_BUFFERS_KV
                        )
                        # consumer_v_view = tlx.local_view(consumer_kv, bufIdx_v)
                        # tl.device_print("gemm consumer_v", accum_cnt_kv + 1)
                        # tl.device_print("gemm consumer_v_buf", bufIdx_v)
                        # tl.device_print("gemm consumer_v_phase", phase_v)
                        tlx.barrier_wait(
                            consumer_kv[bufIdx_v], phase_v
                        )  # consumer wait for v
                        # no need to acquire o0 as this is the only partition updating it
                        # tlx.barrier_wait(producer_o0)  # producer acquire for o0
                        consumer_p0_view = tlx.local_view(producer_qk0, bufIdx_qk)
                        # tl.device_print("gemm producer_qk0", accum_cnt_qk)
                        # tl.device_print("gemm producer_qk0_phase", phase_qk)
                        # DEBUG_PERF_P
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("dot_wait_p0")
                        tlx.barrier_wait(
                            consumer_p0_view, phase_qk
                        )  # consumer wait for p0 use producer_qk0 due to reuse
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("dot_wait_p0")

                        v_view = tlx.local_view(kv_buf, bufIdx_v)
                        bufIdx_o, phase_o = _get_bufidx_phase(
                            accum_cnt_o, NUM_BUFFERS_O
                        )
                        producer_commit_o0_view = tlx.local_view(
                            producer_commit_o0, bufIdx_o
                        )
                        o0_view = tlx.local_view(o0_buf, bufIdx_o)
                        tlx.async_dot(
                            p0_buf[bufIdx_qk],
                            v_view,
                            o0_view,
                            use_acc=True,
                            mBarriers=[producer_commit_o0_view],
                        )

                        first = False
                        accum_cnt_kv += 2
                        accum_cnt_qk += 1
                        accum_cnt_qk1 += 1
                        accum_cnt_o += 1
                        accum_cnt_o1 += 1

                    # epilogue
                    # commit to release q0, q1
                    release_q0_view = tlx.local_view(consumer_release_q0, bufIdx_q)
                    tlx.tcgen05_commit(release_q0_view)
                    release_q1_view = tlx.local_view(consumer_release_q1, bufIdx_q)
                    tlx.tcgen05_commit(release_q1_view)
                    # tl.device_print("gemm producer_o1_epilogue", accum_cnt_outer)
                    # tl.device_print("gemm producer_o1_phase", phase_o_outer)
                    # DEBUG_PERF
                    tlx.barrier_wait(
                        producer_o1_view, phase_o_outer ^ 1, first
                    )  # producer acquire for o1 at the first iteration
                    bufIdx_qk1, phase_qk1 = _get_bufidx_phase(
                        accum_cnt_qk1, NUM_BUFFERS_QK
                    )
                    consumer_p1_view = tlx.local_view(producer_qk1, bufIdx_qk1)
                    # tl.device_print("gemm producer_qk1_epilogue", accum_cnt_qk1)
                    # tl.device_print("gemm producer_qk1_phase", phase_qk1)
                    # DEBUG_PERF_P
                    tlx.barrier_wait(
                        consumer_p1_view, phase_qk1
                    )  # consumer wait for p1 due to reuse of p1 and qk1

                    accum_cnt_qk1 += 1
                    # release p0, p1 via producer_commit_qk0, qk1 barriers
                    # accum_cnt_qk should be equal to accum_cnt_qk1 here
                    # bufIdx_qk, phase_qk = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
                    # consumer_release_p0_view = tlx.local_view(producer_commit_qk0, bufIdx_qk)
                    # consumer_release_p1_view = tlx.local_view(producer_commit_qk1, bufIdx_qk)
                    bufIdx_o, phase_o = _get_bufidx_phase(accum_cnt_o1, NUM_BUFFERS_O)
                    producer_commit_o1_view = tlx.local_view(
                        producer_commit_o1, bufIdx_o
                    )
                    # we already advanced the counter
                    bufIdx_v, phase_v = _get_bufidx_phase(
                        accum_cnt_kv - 1, NUM_BUFFERS_KV
                    )
                    consumer_release_v_view = tlx.local_view(
                        consumer_release_kv, bufIdx_v
                    )
                    o1_view = tlx.local_view(o1_buf, bufIdx_o)
                    tlx.async_dot(  # p1 . v in last iteration
                        p1_buf[bufIdx_qk1],
                        v_view,
                        o1_view,
                        use_acc=not first,
                        mBarriers=[
                            producer_commit_o1_view,
                            consumer_release_v_view,  # , consumer_release_p0_view, consumer_release_p1_view
                        ],
                    )
                    # tl.device_print("gemm consumer_release_v", accum_cnt_kv - 1)
                    # tl.device_print("gemm consumer_release_v_buf", bufIdx_v)
                    accum_cnt_q += 1
                    accum_cnt_outer += 1
                    # signal producer commit of epi0 and epi1, we don't want to block the gemm partition
                    # to wait for the completion
                tile_idx += num_progs
                if ENABLE_PROTON and idx == PROTON_TILE:
                    pl.exit_scope("dot_tile")

        with tlx.async_task(num_warps=1, registers=24):  # load
            accum_count_q = 0
            accum_cnt_kv = 0
            for _ in range(0, tiles_per_sm):
                pid = tile_idx % n_tile_num
                off_hz = tile_idx // n_tile_num
                off_z = off_hz // H
                if SORT_BY_SEQ_LENGTH:
                    off_z = tl.load(seq_index + off_z)
                off_h = off_hz % H
                off_h_kv = off_h // G

                start_m = pid
                q_offset = off_h.to(tl.int64) * stride_qh
                kv_offset = off_h_kv.to(tl.int64) * stride_kh

                begin_q, end_q, begin_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                )

                if start_m * BLOCK_M < qlen:
                    # begin_o = tl.load(Out_offsets + off_z) # confirm if tma store should use begin_q

                    if USE_ON_DEVICE_TMA:
                        q_desc = tl.make_tensor_descriptor(
                            Q,
                            shape=[end_q.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_M // 2, BLOCK_D],
                        )

                    # calculate bufIdx and phase from accum_count_q
                    q_bufIdx = accum_count_q % NUM_BUFFERS_Q
                    q_phase = (accum_count_q // NUM_BUFFERS_Q) & 1
                    # producer acquire: consumer_release_q0
                    # _load_tma(
                    #    q_bufIdx,
                    #    q_phase,
                    #    consumer_release_q0,
                    #    consumer_q0,
                    #    q0_buf,
                    #    q_desc,
                    #    begin_q + start_m * BLOCK_M,
                    #    q_offset,
                    #    BLOCK_M * BLOCK_D * 2,
                    # )
                    # producer acquire
                    q0_empty_view = tlx.local_view(consumer_release_q0, q_bufIdx)
                    tlx.barrier_wait(q0_empty_view, q_phase ^ 1)
                    # barrier for producer commit
                    q0_full_view = tlx.local_view(
                        consumer_q0, q_bufIdx
                    )  # full_bars, bufIdx)
                    tlx.barrier_expect_bytes(
                        q0_full_view, BLOCK_M // 2 * BLOCK_D * 2
                    )  # num_bytes)
                    q0_smem_view = tlx.local_view(q0_buf, q_bufIdx)
                    tlx.async_descriptor_load(
                        q_desc,
                        q0_smem_view,
                        [
                            (begin_q + start_m * BLOCK_M).to(tl.int32),
                            (q_offset).to(tl.int32),
                        ],
                        q0_full_view,
                    )

                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # producer acquire
                    k_empty_view = tlx.local_view(consumer_release_kv, k_bufIdx)
                    tlx.barrier_wait(k_empty_view, k_phase)  # ^ 1)
                    # barrier for producer commit
                    k_full_view = tlx.local_view(consumer_kv, k_bufIdx)
                    tlx.barrier_expect_bytes(
                        k_full_view, BLOCK_N * BLOCK_D * 2
                    )  # num_bytes)
                    k_view = tlx.local_view(kv_buf, k_bufIdx)
                    start_n = 0
                    tlx.async_descriptor_load(
                        k_desc,
                        k_view,
                        [
                            (begin_k + start_n).to(tl.int32),
                            (kv_offset).to(tl.int32),
                        ],
                        k_full_view,
                    )

                    # producer acquire
                    q1_empty_view = tlx.local_view(consumer_release_q1, q_bufIdx)
                    tlx.barrier_wait(q1_empty_view, q_phase ^ 1)
                    # barrier for producer commit
                    q1_full_view = tlx.local_view(consumer_q1, q_bufIdx)
                    tlx.barrier_expect_bytes(
                        q1_full_view, BLOCK_M // 2 * BLOCK_D * 2
                    )  # num_bytes)
                    q1_smem_view = tlx.local_view(q1_buf, q_bufIdx)
                    tlx.async_descriptor_load(
                        q_desc,
                        q1_smem_view,
                        [
                            (begin_q + start_m * BLOCK_M + BLOCK_M // 2).to(tl.int32),
                            (q_offset).to(tl.int32),
                        ],
                        q1_full_view,
                    )

                    v_bufIdx, v_phase = _get_bufidx_phase(
                        accum_cnt_kv + 1, NUM_BUFFERS_KV
                    )
                    v_empty_view = tlx.local_view(consumer_release_kv, v_bufIdx)
                    tlx.barrier_wait(v_empty_view, v_phase)  # ^ 1)
                    # barrier for producer commit
                    v_full_view = tlx.local_view(consumer_kv, v_bufIdx)
                    tlx.barrier_expect_bytes(v_full_view, BLOCK_N * BLOCK_D * 2)
                    v_smem_view = tlx.local_view(kv_buf, v_bufIdx)
                    tlx.async_descriptor_load(
                        v_desc,
                        v_smem_view,
                        [
                            (begin_k + start_n).to(tl.int32),
                            (kv_offset).to(tl.int32),
                        ],
                        v_full_view,
                    )
                    accum_cnt_kv += 2

                    lo, hi = 0, klen
                    for start_n in range(BLOCK_N, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        k_bufIdx, k_phase = _get_bufidx_phase(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        # producer acquire
                        k_empty_view = tlx.local_view(consumer_release_kv, k_bufIdx)
                        # tl.device_print("load consumer_release_k", accum_cnt_kv)
                        # tl.device_print("load consumer_release_k_buf", k_bufIdx)
                        # tl.device_print("load consumer_release_k_phase", k_phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("load_wait_k_empty")
                        tlx.barrier_wait(k_empty_view, k_phase)  # ^ 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_wait_k_empty")
                        # barrier for producer commit
                        k_full_view = tlx.local_view(consumer_kv, k_bufIdx)
                        tlx.barrier_expect_bytes(
                            k_full_view, BLOCK_N * BLOCK_D * 2
                        )  # num_bytes)
                        k_view = tlx.local_view(kv_buf, k_bufIdx)
                        tlx.async_descriptor_load(
                            k_desc,
                            k_view,
                            [
                                (begin_k + start_n).to(tl.int32),
                                (kv_offset).to(tl.int32),
                            ],
                            k_full_view,
                        )
                        # tl.device_print("load accum_cnt_kv", accum_cnt_kv)
                        # tl.device_print("load consumer_k_buf", k_bufIdx)
                        # k_view = tlx.local_trans(k_view)

                        # producer acquire
                        v_bufIdx, v_phase = _get_bufidx_phase(
                            accum_cnt_kv + 1, NUM_BUFFERS_KV
                        )
                        v_empty_view = tlx.local_view(consumer_release_kv, v_bufIdx)
                        # tl.device_print("load accum_cnt_kv", accum_cnt_kv + 1)
                        # tl.device_print("load consumer_release_v_buf", v_bufIdx)
                        # tl.device_print("load consumer_release_v_phase", v_phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("load_wait_v_empty")
                        tlx.barrier_wait(v_empty_view, v_phase)  # ^ 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_wait_v_empty")
                        # barrier for producer commit
                        v_full_view = tlx.local_view(consumer_kv, v_bufIdx)
                        tlx.barrier_expect_bytes(v_full_view, BLOCK_N * BLOCK_D * 2)
                        v_smem_view = tlx.local_view(kv_buf, v_bufIdx)
                        tlx.async_descriptor_load(
                            v_desc,
                            v_smem_view,
                            [
                                (begin_k + start_n).to(tl.int32),
                                (kv_offset).to(tl.int32),
                            ],
                            v_full_view,
                        )
                        # tl.device_print("load consumer_v_buf", v_bufIdx)
                        accum_cnt_kv += 2
                    # outside of inner for
                    accum_count_q += 1
                tile_idx += num_progs
        with tlx.async_task(num_warps=1, registers=24):  # epilogue
            # Can we guard this with not MERGE_EPI?
            accum_cnt_outer = 0
            for idx in range(0, tiles_per_sm):
                pid = tile_idx % n_tile_num
                start_m = pid
                off_hz = tile_idx // n_tile_num
                off_h = off_hz % H
                out_offset = off_h.to(tl.int64) * stride_oh

                begin_q, end_q, begin_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                )

                if start_m * BLOCK_M < qlen:
                    out_offset = off_h.to(tl.int64) * stride_oh
                    if USE_ON_DEVICE_TMA:
                        o_desc = tl.make_tensor_descriptor(
                            Out,
                            shape=[end_q.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_M // 2, BLOCK_D],
                        )
                    # wait for o0
                    tlx.async_descriptor_store(
                        o_desc,
                        o0_smem[0],
                        [
                            (begin_q + start_m * BLOCK_M).to(tl.int32),
                            (out_offset).to(tl.int32),
                        ],
                    )
                    tlx.async_descriptor_store(
                        o_desc,
                        o1_smem[0],
                        [
                            (begin_q + start_m * BLOCK_M + BLOCK_M // 2).to(tl.int32),
                            (out_offset).to(tl.int32),
                        ],
                    )
                    accum_cnt_outer += 1
                tile_idx += num_progs


def next_power_of_2(x):
    return 2 ** (math.ceil(math.log(x, 2)))


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if x is not None and x.stride(-1) != 1:
        return x.contiguous()
    return x


# assume is_predict: tl.constexpr,  #  false
#    FUSED_QKV: tl.constexpr,  # false
#    FUSED_KV: tl.constexpr,  # false
#    SORT_BY_SEQ_LENGTH: tl.constexpr,  false
#    STAGE: tl.constexpr,  #
#    USE_START_END_OFFSETS: tl.constexpr,  false
#    WINDOW_SIZE: tl.constexpr,
#    BROADCAST_Q: tl.constexpr, false
#    IS_DENSE_KV: tl.constexpr,  (true)
def gdpa_forward_tlx(
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
    if qk_scale is None:
        qk_scale = 1.0

    HEAD_DIM_Q = query.shape[-1]
    HEAD_DIM_K = key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = value.shape[-1]
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
    # extra_kern_args["maxnreg"] = 168
    nheads = query.shape[1]
    G = query.shape[1] // key.shape[1]
    assert query.shape[1] % key.shape[1] == 0
    batch_size = BATCH * nheads
    NUM_SMS = (
        get_num_sms() or 1000000
    )  # * 8  # if num sms is None, use a large number so that it is a no-op
    # print("NUM_SMS", NUM_SMS)
    # print(triton.cdiv(max_seq_len_q, 256) * BATCH * nheads)

    q = expect_contiguous(query)
    k = expect_contiguous(key)
    v = expect_contiguous(value)
    kstrides = k.stride()
    vstrides = v.stride()

    dummy_block = [1, 1]
    N_CTX_KV = max_seq_len_kv
    HEAD_DIM = HEAD_DIM_K
    Z = BATCH
    H = nheads
    y_dim = N_CTX_KV * Z
    x_dim = HEAD_DIM * H // G
    USE_ON_DEVICE_TMA = True
    if not USE_ON_DEVICE_TMA:
        desc_q = TensorDescriptor(
            q,
            shape=[y_dim, HEAD_DIM * H],
            strides=[HEAD_DIM * H, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v, shape=[y_dim, x_dim], strides=[x_dim, 1], block_shape=dummy_block
        )
        desc_k = TensorDescriptor(
            k, shape=[y_dim, x_dim], strides=[x_dim, 1], block_shape=dummy_block
        )
        desc_o = TensorDescriptor(
            o,
            shape=[y_dim, HEAD_DIM * H],
            strides=[HEAD_DIM * H, 1],
            block_shape=dummy_block,
        )

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    def grid_tma_persistent(META):
        return (
            min(NUM_SMS, triton.cdiv(max_seq_len_q, META["BLOCK_M"]) * BATCH * nheads),
            1,
            1,
        )

    activation_enum_int = activation_string_to_int(activation)
    # print(q.shape, k.shape, v.shape)
    # print("activation_enum_int", activation, activation_enum_int)
    # print(query_offset)
    # print(key_offset)

    enable_proton = True if os.getenv("ENABLE_PROTON") == "1" else False

    gdpa_kernel_tma_ws_blackwell[grid_tma_persistent](
        q if USE_ON_DEVICE_TMA else desc_q,
        query_offset,
        k if USE_ON_DEVICE_TMA else desc_k,
        key_offset,
        v if USE_ON_DEVICE_TMA else desc_v,
        o if USE_ON_DEVICE_TMA else desc_o,
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
        is_predict=False,
        Q_SHAPE_0=query.shape[0],
        FUSED_QKV=False,  # fused_qkv,
        FUSED_KV=False,  # fused_kv,
        SORT_BY_SEQ_LENGTH=sort_by_seq_length,
        HEAD_DIM=HEAD_DIM_K,  #
        BLOCK_D=BLOCK_D,
        STAGE=stage,  #
        USE_START_END_OFFSETS=use_start_end_offsets,
        WINDOW_SIZE=window_size,
        BROADCAST_Q=broadcast_q,
        IS_DENSE_KV=is_dense_kv,
        activation_enum_int=activation_enum_int,
        USE_ON_DEVICE_TMA=USE_ON_DEVICE_TMA,
        NAME_BARRIER=False,
        MERGE_EPI=False,
        ENABLE_PROTON=enable_proton,
        PROTON_TILE=10,
        **extra_kern_args,
    )
    return o


@triton.jit
def _gdpa_bwd_tlx_compute_num_steps(
    start_n,
    qlen,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    start_m_inner = 0
    num_steps = tl.cdiv((qlen - start_m_inner), BLOCK_M1)
    if WINDOW_SIZE is not None:
        start_m_inner = (
            max(start_m_inner, start_n - WINDOW_SIZE) // BLOCK_M1
        ) * BLOCK_M1
        end_m_inner = (
            tl.cdiv(min(qlen, start_n + BLOCK_N1 + WINDOW_SIZE), BLOCK_M1) * BLOCK_M1
        )
        num_steps = (end_m_inner - start_m_inner) // BLOCK_M1
    return num_steps


@triton.jit
def _gdpa_bwd_tlx_compute_activation(
    desc_q,
    DK,
    DV,
    k_tiles,
    v_tiles,
    qk_tiles,
    dpT_tiles,
    ppT_tiles,
    dsT_tiles,
    dv_tiles,
    dk_tiles,
    qk_fulls,
    dpT_fulls,
    ppT_fulls,
    dsT_fulls,
    dv_fulls,
    dk_fulls,
    accum_cnt_outer,
    accum_cnt_inner,
    stride_km,
    stride_d,
    start_n,
    start_m,
    qlen,
    klen,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    MASK: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    activation_enum_int: tl.constexpr,
):
    dkv_buf_id, dkv_phase = _get_bufidx_phase(accum_cnt_outer, NUM_BUFFERS_TMEM)
    dk_tile = tlx.local_view(dk_tiles, dkv_buf_id)
    dv_tile = tlx.local_view(dv_tiles, dkv_buf_id)
    dk_full = tlx.local_view(dk_fulls, dkv_buf_id)
    dv_full = tlx.local_view(dv_fulls, dkv_buf_id)

    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    num_steps = _gdpa_bwd_tlx_compute_num_steps(
        qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
    )

    for i in tl.range(0, num_steps, 1, loop_unroll_factor=1):
        tmem_buf_id, tmem_phase = _get_bufidx_phase(
            accum_cnt_inner + i, NUM_BUFFERS_TMEM
        )
        qk_tile = tlx.local_view(qk_tiles, tmem_buf_id)
        ppT_tile = tlx.local_view(ppT_tiles, tmem_buf_id)
        dpT_tile = tlx.local_view(dpT_tiles, tmem_buf_id)
        dsT_tile = tlx.local_view(dsT_tiles, tmem_buf_id)
        qk_full = tlx.local_view(qk_fulls, tmem_buf_id)
        ppT_full = tlx.local_view(ppT_fulls, tmem_buf_id)
        dpT_full = tlx.local_view(dpT_fulls, tmem_buf_id)
        dsT_full = tlx.local_view(dsT_fulls, tmem_buf_id)

        # wait for qkT = tl.dot(k, qT)
        tlx.barrier_wait(qk_full, tmem_phase)
        pT = tlx.local_load(qk_tile)

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
            ppT = pT
        elif activation_enum_int == 1:
            ppT = gelu(pT)
        elif activation_enum_int == 2:
            tanh_out = tanh_approx_fp32(0.7978845608 * pT * (1 + 0.044715 * pT * pT))
            ppT = 0.5 * pT * (1 + tanh_out)
        else:
            # rest of the enums are not supported yet
            ppT = pT
        # ppT *= qk_scale
        ppT = ppT.to(tlx.dtype_of(desc_q))
        tlx.local_store(ppT_tile, ppT)
        tlx.barrier_arrive(ppT_full)

        ##
        if activation_enum_int == 0:
            pT = 1.0
        elif activation_enum_int == 1:
            # activation = gelu TypeError("cannot convert JITFunction(ads_mkl.ops.triton.math:gelu) of type <class 'triton.runtime.jit.JITFunction'> to tensor")
            pT = gelu_grad(pT)
        elif activation_enum_int == 2:
            pT = (
                0.5
                * pT
                * (1 - tanh_out * tanh_out)
                * (0.7978845608 + 0.1070322243 * pT * pT)
            ) + 0.5 * (1 + tanh_out)
        else:
            pT = 1
        # pT *= qk_scale

        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        # Sliding window masking.
        if WINDOW_SIZE is not None:
            window_mask = tl.abs(offs_m[None, :] - offs_n[:, None]) <= WINDOW_SIZE
            pT = tl.where(window_mask, pT, 0.0)

        # Wait for dpT = tl.dot(v, tl.trans(do))
        tlx.barrier_wait(dpT_full, tmem_phase)
        dpT = tlx.local_load(dpT_tile)
        dsT = pT * dpT
        # TODO: release dpT

        dsT = dsT.to(tlx.dtype_of(desc_q))
        tlx.local_store(dsT_tile, dsT)
        tlx.barrier_arrive(dsT_full)

    offs_k = tl.arange(0, BLOCK_D)
    kmask = (offs_k[None, :] < HEAD_DIM) & (offs_n[:, None] < klen)

    # Write back dV.
    tlx.barrier_wait(dv_full, dkv_phase)
    dv_r = tlx.local_load(dv_tile)
    dv_ptrs = DV + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv_r, mask=kmask)

    # Write back dK.
    tlx.barrier_wait(dk_full, dkv_phase)
    dk_r = tlx.local_load(dk_tile)
    dk_ptrs = DK + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk_r, mask=kmask)

    return num_steps


@triton.jit
def bwd_caculate_offsets(
    seq_index,
    H,
    G,
    SORT_BY_SEQ_LENGTH: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
):
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

    return off_z, off_h, off_h_kv, off_q_z, pid


@triton.jit
def bwd_caculate_tile_num(
    Z,
    H,
    N_CTX: tl.constexpr,
    N_CTX_KV: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
):
    n_tile_num = max(tl.cdiv(N_CTX_KV, BLOCK_N1), tl.cdiv(N_CTX, BLOCK_M2))
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H
    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1
    return tiles_per_sm, n_tile_num


def _bwd_host_descriptor_pre_hook(nargs):
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    BLOCK_D = nargs["BLOCK_D"]
    nargs["desc_q"].block_shape = [BLOCK_M1, BLOCK_D]
    nargs["desc_v"].block_shape = [BLOCK_N1, BLOCK_D]
    nargs["desc_k"].block_shape = [BLOCK_N1, BLOCK_D]
    nargs["desc_do"].block_shape = [BLOCK_M1, BLOCK_D]
    nargs["desc_dq"].block_shape = [BLOCK_M1, BLOCK_D]


def get_tlx_bwd_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_M1": BM,
                "BLOCK_N1": BN,
                "BLOCK_M2": BN,
                "BLOCK_N2": BM,
                "NUM_BUFFERS_KV": 1,
                "NUM_BUFFERS_Q": 1,
                "NUM_BUFFERS_DO": 1,
                "NUM_BUFFERS_DS": 1,
                "NUM_BUFFERS_TMEM": 1,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=_bwd_host_descriptor_pre_hook,
        )
        for BM in [128]  # 128 or 256
        for BN in [128]
    ]


@triton.jit
def gdpa_backward_tlx(
    desc_q,
    Q_offsets,
    desc_k,
    K_offsets,
    desc_v,
    seq_index,  #
    desc_do,  #
    Out_offsets,
    desc_dq,
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
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    enable_tma: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
):
    # allocate smem buffers
    k_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_D), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    v_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_D), tlx.dtype_of(desc_v), NUM_BUFFERS_KV)
    q_tiles = tlx.local_alloc((BLOCK_M1, BLOCK_D), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    do_tiles = tlx.local_alloc(
        (BLOCK_M1, BLOCK_D), tlx.dtype_of(desc_do), NUM_BUFFERS_DO
    )
    dsT_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_D), tlx.dtype_of(desc_q), NUM_BUFFERS_DS
    )

    # allocate barriers for smem buffers
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    dsT_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    dsT_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)

    # allocate tmem buffers
    qk_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_D), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem
    )
    ppT_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_D),
        tlx.dtype_of(desc_do),
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    dpT_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_D),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    dv_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_D), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem
    )
    dk_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_D), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem
    )
    dq_tiles = tlx.local_alloc(
        (BLOCK_M1, BLOCK_D), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem
    )

    # allocate barriers for tmem buffers
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    ppT_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dpT_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    ppT_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    with tlx.async_tasks():
        # activation
        with tlx.async_task("default"):
            off_z, off_h, off_h_kv, off_q_z, pid = bwd_caculate_offsets(
                seq_index,
                H,
                G,
                SORT_BY_SEQ_LENGTH,
                BROADCAST_Q,
            )

            begin_q = tl.load(Q_offsets + off_q_z)
            end_q = tl.load(Q_offsets + off_q_z + 1)
            start_n = pid * BLOCK_N1
            start_m = pid * BLOCK_M2
            off_h2 = off_h.to(tl.int64)
            qlen = end_q - begin_q
            if FUSED_QKV:
                begin_k = begin_q
                end_k = end_q
                klen = qlen
            else:
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k

            if start_n < klen:
                _gdpa_bwd_tlx_compute_activation(
                    desc_q=desc_q,
                    DK=DK,
                    DV=DV,
                    k_tiles=k_tiles,
                    v_tiles=v_tiles,
                    qk_tiles=qk_tiles,
                    dpT_tiles=dpT_tiles,
                    ppT_tiles=ppT_tiles,
                    dsT_tiles=dsT_tiles,
                    dv_tiles=dv_tiles,
                    dk_tiles=dk_tiles,
                    qk_fulls=qk_fulls,
                    dpT_fulls=dpT_fulls,
                    ppT_fulls=ppT_fulls,
                    dsT_fulls=dsT_fulls,
                    dv_fulls=dv_fulls,
                    dk_fulls=dk_fulls,
                    accum_cnt_outer=0,
                    accum_cnt_inner=0,
                    stride_km=stride_km,
                    stride_d=stride_d,
                    start_n=start_n,
                    start_m=start_m,
                    qlen=qlen,
                    klen=klen,
                    HEAD_DIM=HEAD_DIM,
                    BLOCK_D=BLOCK_D,
                    BLOCK_M1=BLOCK_M1,
                    BLOCK_N1=BLOCK_N1,
                    MASK=False,
                    WINDOW_SIZE=WINDOW_SIZE,
                    NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                    activation_enum_int=activation_enum_int,
                )

        # reduction
        with tlx.async_task(num_warps=4):
            off_z, off_h, off_h_kv, off_q_z, pid = bwd_caculate_offsets(
                seq_index,
                H,
                G,
                SORT_BY_SEQ_LENGTH,
                BROADCAST_Q,
            )

            begin_q = tl.load(Q_offsets + off_q_z)
            end_q = tl.load(Q_offsets + off_q_z + 1)
            start_n = pid * BLOCK_N1
            start_m = pid * BLOCK_M2
            off_h2 = off_h.to(tl.int64)
            qlen = end_q - begin_q
            if FUSED_QKV:
                begin_k = begin_q
                end_k = end_q
                klen = qlen
            else:
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k

            if start_n < klen:
                curr_m = start_m
                step_m = BLOCK_M1
                num_steps = _gdpa_bwd_tlx_compute_num_steps(
                    qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                )

                for i in tl.range(0, num_steps, 1, loop_unroll_factor=1):
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(i, NUM_BUFFERS_TMEM)
                    dq_tile = tlx.local_view(dq_tiles, tmem_buf_id)
                    dq_full = tlx.local_view(dq_fulls, tmem_buf_id)
                    dq_empty = tlx.local_view(dq_empties, tmem_buf_id)

                    # wait for dq = tl.dot(tl.trans(dsT), k)
                    tlx.barrier_wait(dq_full, tmem_phase)
                    dq_r = tlx.local_load(dq_tile)
                    dq_r = dq_r.to(tlx.dtype_of(desc_dq))
                    desc_dq.atomic_add(
                        [
                            (begin_q + curr_m).to(tl.int32),
                            (off_h2 * stride_qh).to(tl.int32),
                        ],
                        dq_r,
                    )

                    # release dq
                    tlx.barrier_arrive(dq_empty)

                    # Increment pointers.
                    curr_m += step_m

        # mma
        with tlx.async_task(num_warps=1):
            off_z, off_h, off_h_kv, off_q_z, pid = bwd_caculate_offsets(
                seq_index,
                H,
                G,
                SORT_BY_SEQ_LENGTH,
                BROADCAST_Q,
            )

            begin_q = tl.load(Q_offsets + off_q_z)
            end_q = tl.load(Q_offsets + off_q_z + 1)
            start_n = pid * BLOCK_N1
            start_m = pid * BLOCK_M2
            off_h2 = off_h.to(tl.int64)
            qlen = end_q - begin_q
            if FUSED_QKV:
                begin_k = begin_q
                end_k = end_q
                klen = qlen
            else:
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k

            if start_n < klen:
                # Wait for K, V ready
                kv_buf_id, kv_phase = _get_bufidx_phase(0, NUM_BUFFERS_KV)
                k_tile = tlx.local_view(k_tiles, kv_buf_id)
                v_tile = tlx.local_view(v_tiles, kv_buf_id)
                k_full = tlx.local_view(k_fulls, kv_buf_id)
                v_full = tlx.local_view(v_fulls, kv_buf_id)
                tlx.barrier_wait(k_full, kv_phase)
                tlx.barrier_wait(v_full, kv_phase)

                # Wait for dK, dV to be released
                dkv_buf_id, _ = _get_bufidx_phase(0, NUM_BUFFERS_TMEM)
                dk_tile = tlx.local_view(dk_tiles, dkv_buf_id)
                dv_tile = tlx.local_view(dv_tiles, dkv_buf_id)
                dk_full = tlx.local_view(dk_fulls, dkv_buf_id)
                dv_full = tlx.local_view(dv_fulls, dkv_buf_id)

                # Compute dK and dV for non-masked blocks.
                # Q is reloaded for each block of M.

                # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
                tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

                num_steps = _gdpa_bwd_tlx_compute_num_steps(
                    qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                )

                for i in tl.range(0, num_steps, 1, num_stages=0):
                    q_buf_id, q_phase = _get_bufidx_phase(i, NUM_BUFFERS_Q)
                    q_tile = tlx.local_view(q_tiles, q_buf_id)
                    q_full = tlx.local_view(q_fulls, q_buf_id)
                    q_empty = tlx.local_view(q_empties, q_buf_id)

                    do_buf_id, do_phase = _get_bufidx_phase(i, NUM_BUFFERS_DO)
                    do_tile = tlx.local_view(do_tiles, do_buf_id)
                    do_full = tlx.local_view(do_fulls, do_buf_id)
                    do_empty = tlx.local_view(do_empties, do_buf_id)

                    tmem_buf_id, tmem_phase = _get_bufidx_phase(i, NUM_BUFFERS_TMEM)
                    qk_tile = tlx.local_view(qk_tiles, tmem_buf_id)
                    ppT_tile = tlx.local_view(ppT_tiles, tmem_buf_id)
                    dpT_tile = tlx.local_view(dpT_tiles, tmem_buf_id)
                    dsT_tile = tlx.local_view(dsT_tiles, tmem_buf_id)
                    dq_tile = tlx.local_view(dq_tiles, tmem_buf_id)
                    ppT_empty = tlx.local_view(ppT_empties, tmem_buf_id)
                    dsT_empty = tlx.local_view(dsT_empties, tmem_buf_id)
                    dq_empty = tlx.local_view(dq_empties, tmem_buf_id)
                    qk_full = tlx.local_view(qk_fulls, tmem_buf_id)
                    ppT_full = tlx.local_view(ppT_fulls, tmem_buf_id)
                    dpT_full = tlx.local_view(dpT_fulls, tmem_buf_id)
                    dsT_full = tlx.local_view(dsT_fulls, tmem_buf_id)
                    dq_full = tlx.local_view(dq_fulls, tmem_buf_id)

                    # Compute qkT = tl.dot(k, qT)
                    tlx.barrier_wait(q_full, q_phase)
                    tlx.barrier_wait(ppT_empty, tmem_phase ^ 1)
                    qT = tlx.local_trans(q_tile)
                    tlx.async_dot(
                        k_tile,
                        qT,
                        qk_tile,
                        use_acc=False,
                        mBarriers=[qk_full],
                    )

                    # Compute dpT = tl.dot(v, tl.trans(do))
                    tlx.barrier_wait(do_full, do_phase)
                    tlx.barrier_wait(dsT_empty, tmem_phase ^ 1)
                    doT = tlx.local_trans(do_tile)
                    tlx.async_dot(
                        v_tile,
                        doT,
                        dpT_tile,
                        use_acc=False,
                        mBarriers=[dpT_full],
                    )

                    # Compute dv += tl.dot(ppT, do)
                    tlx.barrier_wait(ppT_full, tmem_phase)
                    tlx.async_dot(
                        ppT_tile,
                        do_tile,
                        dv_tile,
                        use_acc=i > 0,
                        mBarriers=[do_empty, ppT_empty],
                    )

                    # Compute dk += tl.dot(dsT, tl.trans(qT))
                    tlx.barrier_wait(dsT_full, tmem_phase)
                    tlx.async_dot(
                        dsT_tile,
                        q_tile,
                        dk_tile,
                        use_acc=i > 0,
                        mBarriers=[q_empty],
                    )

                    # Compute dq = tl.dot(tl.trans(dsT), k)
                    tlx.barrier_wait(dq_empty, tmem_phase ^ 1)
                    dsT_view = tlx.local_trans(dsT_tile)
                    tlx.async_dot(
                        dsT_view,
                        k_tile,
                        dq_tile,
                        use_acc=0,
                        mBarriers=[dq_full, dsT_empty],
                    )

                # commit dk, dv
                tlx.tcgen05_commit(dk_full)
                tlx.tcgen05_commit(dv_full)

        # load
        with tlx.async_task(num_warps=1):
            off_z, off_h, off_h_kv, off_q_z, pid = bwd_caculate_offsets(
                seq_index,
                H,
                G,
                SORT_BY_SEQ_LENGTH,
                BROADCAST_Q,
            )

            begin_q = tl.load(Q_offsets + off_q_z)
            end_q = tl.load(Q_offsets + off_q_z + 1)
            start_n = pid * BLOCK_N1
            start_m = pid * BLOCK_M2
            off_h2 = off_h.to(tl.int64)
            qlen = end_q - begin_q
            if FUSED_QKV:
                begin_k = begin_q
                end_k = end_q
                klen = qlen
            else:
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k

            # Some of the ops are used for both producer and consumer, some are used by consumer
            # Try to correctly specialize the IfOp by marking all ops.
            # invert of start_n > klen and start_m > qlen
            if start_n < klen:
                # load K and V: they stay in SRAM throughout the inner loop.
                kv_buf_id, kv_phase = _get_bufidx_phase(0, NUM_BUFFERS_KV)

                # tlx.barrier_wait(k_empties[kv_buf_id], kv_phase)
                k_tile = tlx.local_view(k_tiles, kv_buf_id)
                v_tile = tlx.local_view(v_tiles, kv_buf_id)
                k_full = tlx.local_view(k_fulls, kv_buf_id)
                v_full = tlx.local_view(v_fulls, kv_buf_id)

                tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N1 * BLOCK_D)  # float16
                tlx.async_descriptor_load(
                    desc_k,
                    k_tile,
                    [
                        (begin_k + start_n).to(tl.int32),
                        (off_h_kv * stride_kh).to(tl.int32),
                    ],
                    k_full,
                )

                # tlx.barrier_wait(v_empties[kv_buf_id], kv_phase)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N1 * BLOCK_D)  # float16
                tlx.async_descriptor_load(
                    desc_v,
                    v_tile,
                    [
                        (begin_k + start_n).to(tl.int32),
                        (off_h_kv * stride_kh).to(tl.int32),
                    ],
                    v_full,
                )

                curr_m = start_m
                step_m = BLOCK_M1
                num_steps = _gdpa_bwd_tlx_compute_num_steps(
                    qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                )

                for i in tl.range(0, num_steps, 1, loop_unroll_factor=1):
                    q_buf_id, q_phase = _get_bufidx_phase(i, NUM_BUFFERS_Q)
                    q_tile = tlx.local_view(q_tiles, q_buf_id)
                    q_full = tlx.local_view(q_fulls, q_buf_id)
                    q_empty = tlx.local_view(q_empties, q_buf_id)

                    do_buf_id, do_phase = _get_bufidx_phase(i, NUM_BUFFERS_DO)
                    do_tile = tlx.local_view(do_tiles, do_buf_id)
                    do_full = tlx.local_view(do_fulls, do_buf_id)
                    do_empty = tlx.local_view(do_empties, do_buf_id)

                    tlx.barrier_wait(q_empty, q_phase ^ 1)
                    tlx.barrier_expect_bytes(q_full, 2 * BLOCK_M1 * BLOCK_D)  # float16
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tile,
                        [
                            (begin_q + curr_m).to(tl.int32),
                            (off_h2 * stride_qh).to(tl.int32),
                        ],
                        q_full,
                    )

                    # we may need this line for correctness
                    # offs_m = curr_m + tl.arange(0, BLOCK_M1)
                    # qmask = (offs_k[:, None] < HEAD_DIM) & (offs_m[None, :] < qlen)
                    # qT = tl.where(qmask, qT, 0.0)

                    # move dot ahead hoping to overlap with other computations
                    tlx.barrier_wait(do_empty, do_phase ^ 1)
                    tlx.barrier_expect_bytes(do_full, 2 * BLOCK_M1 * BLOCK_D)  # float16
                    tlx.async_descriptor_load(
                        desc_do,
                        do_tile,
                        [
                            (begin_q + curr_m).to(tl.int32),
                            (off_h2 * stride_qh).to(tl.int32),
                        ],
                        do_full,
                    )

                    # omask = (offs_m[:, None] < qlen) & (offs_k[None, :] < HEAD_DIM)
                    # do = tl.where(omask, do, 0.0)

                    # Increment pointers.
                    curr_m += step_m
