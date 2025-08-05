# TLX GDPA kernel optimized for Blackwell Warp Specialization

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from .gdpa_utils import get_num_sms
from .math import activation_string_to_int


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "NUM_BUFFERS_Q": bq,
                "NUM_BUFFERS_K": bk,
                "NUM_BUFFERS_QK": bq,
                "NUM_BUFFERS_O": bo,
            },
            num_warps=4,
            num_stages=1,
        )
        for BM in [128]
        for BN in [128]
        for bq in [1]
        for bk in [2]
        for bqk in [1]  # in tmem
        for bo in [1]  # in tmem
    ]


## Overall warpspec configuration
## 4 partitions:
##   default is activation0 with 4 warps, partition0 is activatation1 with 4 warps
##   partition1 is gemm, partition 2 is load, partition 3 is epilogue/store
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
def _reinterpret(qk_buf, bufIdx_qk):
    qk_view = tlx.local_view(qk_buf, bufIdx_qk)
    p_view = tlx.local_reinterpret(qk_view, tl.float16)
    return p_view


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
def _do_dots(
    klen,
    q0_buf,
    q1_buf,
    k_buf,
    v_buf,
    qk0_buf,
    qk1_buf,
    o0_buf,
    o1_buf,
    consumer_q0,
    consumer_q1,
    consumer_k,
    consumer_v,
    producer_qk0,
    producer_commit_qk0,
    producer_qk1,
    producer_commit_qk1,
    consumer_release_k,
    consumer_release_v,
    consumer_release_q0,
    consumer_release_q1,
    producer_o0,
    producer_commit_o0,
    producer_o1,
    producer_commit_o1,
    accum_cnt_q,
    accum_cnt_k,
    accum_cnt_qk,
    accum_cnt_outer,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_K: tl.constexpr,
    NUM_BUFFERS_QK: tl.constexpr,
    NUM_BUFFERS_O: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # prologue
    bufIdx_q, phase_q = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
    bufIdx_k, phase_k = _get_bufidx_phase(accum_cnt_k, NUM_BUFFERS_K)
    bufIdx_qk, phase_qk = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
    accum_cnt_qk1 = accum_cnt_qk

    consumer_q0_view = tlx.local_view(consumer_q0, bufIdx_q)
    consumer_k_view = tlx.local_view(consumer_k, bufIdx_k)
    # producer_qk0_view = tlx.local_view(producer_qk0, bufIdx_qk)
    tlx.barrier_wait(consumer_q0_view, phase_q)  # consumer wait for q0
    tlx.barrier_wait(consumer_k_view, phase_k)  # consumer wait for k
    # Do we need the initial acquire here?
    # tlx.barrier_wait(producer_qk0_view, phase_qk)  # producer acquire for qk0
    # producer commit for qk0
    q0_view = tlx.local_view(q0_buf, bufIdx_q)
    k_view = tlx.local_view(k_buf, bufIdx_k)
    qk0_view = tlx.local_view(qk0_buf, bufIdx_qk)
    producer_commit_qk0_view = tlx.local_view(producer_commit_qk0, bufIdx_qk)
    tlx.async_dot(
        q0_view, k_view, qk0_view, use_acc=False, mBarriers=[producer_commit_qk0_view]
    )
    # accum_cnt_qk += 1

    consumer_q1_view = tlx.local_view(consumer_q1, bufIdx_q)
    # producer_qk1_view = tlx.local_view(producer_qk1, bufIdx_qk)
    tlx.barrier_wait(consumer_q1_view, phase_q)  # consumer wait for q1
    # Do we need the initial acquire here?
    # tlx.barrier_wait(producer_qk1_view, phase_qk)  # producer acquire for qk1
    # consumer release for k, producer commit for qk1
    q1_view = tlx.local_view(q1_buf, bufIdx_q)
    qk1_view = tlx.local_view(qk1_buf, bufIdx_qk)
    consumer_release_k_view = tlx.local_view(consumer_release_k, bufIdx_k)
    producer_commit_qk1_view = tlx.local_view(producer_commit_qk1, bufIdx_qk)
    tlx.async_dot(
        q1_view,
        k_view,
        qk1_view,
        use_acc=False,
        mBarriers=[consumer_release_k_view, producer_commit_qk1_view],
    )
    # accum_cnt_qk1 += 1

    consumer_v_view = tlx.local_view(consumer_v, bufIdx_k)
    tlx.barrier_wait(consumer_v_view, phase_k)  # consumer wait for v
    # need to acquire o0 to make sure epilogue is done, this is needed for each outer loop
    bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(accum_cnt_outer, NUM_BUFFERS_O)
    producer_o0_view = tlx.local_view(producer_o0, bufIdx_o_outer)
    producer_o1_view = tlx.local_view(producer_o1, bufIdx_o_outer)
    tlx.barrier_wait(producer_o0_view, phase_o_outer)  # producer acquire for o0
    # For reuse of qk0 and p0, we can simplify the barriers
    #   activation partition: consumer wait for qk0, ... update p, producer commit of p0
    #   dot partition: producer commit of qk0, ..., consumer wait for p0 (use the same barrier as producer_qk0)
    bufIdx_p, phase_p = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
    consumer_p0_view = tlx.local_view(producer_qk0, bufIdx_p)
    tlx.barrier_wait(
        consumer_p0_view, phase_p
    )  # consumer wait for p0 due to reuse of p0 and qk0
    # reinterpret qk0 as p0
    p0_view = _reinterpret(qk0_buf, bufIdx_p)
    bufIdx_o, phase_o = _get_bufidx_phase(accum_cnt_k, NUM_BUFFERS_O)
    producer_commit_o0_view = tlx.local_view(producer_commit_o0, bufIdx_o)
    o0_view = tlx.local_view(o0_buf, bufIdx_o)
    v_view = tlx.local_view(v_buf, bufIdx_k)
    tlx.async_dot(
        p0_view, v_view, o0_view, use_acc=False, mBarriers=[producer_commit_o0_view]
    )
    accum_cnt_o1 = accum_cnt_k

    lo, hi = 0, klen
    first = True
    mma_iters = (hi - lo) // BLOCK_N
    accum_cnt_k += 1
    accum_cnt_qk += 1
    for _ in range(mma_iters - 1):
        bufIdx_k, phase_k = _get_bufidx_phase(accum_cnt_k, NUM_BUFFERS_K)
        bufIdx_qk, phase_qk = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)

        # q0 dot k
        consumer_k_view = tlx.local_view(consumer_k, bufIdx_k)
        tlx.barrier_wait(consumer_k_view, phase_k)  # consumer wait for k
        k_view = tlx.local_view(k_buf, bufIdx_k)
        qk0_view = tlx.local_view(qk0_buf, bufIdx_qk)
        producer_commit_qk0_view = tlx.local_view(producer_commit_qk0, bufIdx_qk)
        tlx.async_dot(
            q0_view,
            k_view,
            qk0_view,
            use_acc=False,
            mBarriers=[producer_commit_qk0_view],
        )

        # p1 dot v for previous iteration
        bufIdx_qk1, phase_qk1 = _get_bufidx_phase(accum_cnt_qk1, NUM_BUFFERS_QK)
        consumer_p1_view = tlx.local_view(producer_qk1, bufIdx_qk1)
        tlx.barrier_wait(
            producer_o1_view,
            phase_o_outer,  # , first FIXME
        )  # producer acquire for o1, only needed for first iteration
        tlx.barrier_wait(
            consumer_p1_view, phase_qk1
        )  # consumer wait for p1 use producer_qk1 due to reuse
        # done using v from previous iteration
        bufIdx_o1, phase_o1 = _get_bufidx_phase(accum_cnt_o1, NUM_BUFFERS_O)
        o1_view = tlx.local_view(o1_buf, bufIdx_o1)
        producer_commit_o1_view = tlx.local_view(producer_commit_o1, bufIdx_o1)
        bufIdx_v, phase_v = _get_bufidx_phase(
            accum_cnt_o1, NUM_BUFFERS_K
        )  # NUM_BUFFERS_K is NUM_BUFFERS_V
        consumer_release_v_view = tlx.local_view(consumer_release_v, bufIdx_v)
        # reinterpret as p1
        p1_view = _reinterpret(qk1_buf, bufIdx_qk1)
        tlx.async_dot(
            p1_view,
            v_view,
            o1_view,
            use_acc=not first,
            mBarriers=[producer_commit_o1_view, consumer_release_v_view],
        )

        # q1 dot k, done using k for this iteration
        bufIdx_qk1_next, phase_qk1_next = _get_bufidx_phase(
            accum_cnt_qk1 + 1, NUM_BUFFERS_QK
        )
        qk1_view = tlx.local_view(qk1_buf, bufIdx_qk1_next)
        consumer_release_k_view = tlx.local_view(consumer_release_k, bufIdx_k)
        producer_commit_qk1_view = tlx.local_view(producer_commit_qk1, bufIdx_qk1_next)
        tlx.async_dot(
            q1_view,
            k_view,
            qk1_view,
            use_acc=False,
            mBarriers=[consumer_release_k_view, producer_commit_qk1_view],
        )

        # p0 dot v
        tlx.barrier_wait(consumer_v, phase_k)  # consumer wait for v
        # no need to acquire o0 as this is the only partition updating it
        # tlx.barrier_wait(producer_o0)  # producer acquire for o0
        consumer_p0_view = tlx.local_view(producer_qk0, bufIdx_qk)
        tlx.barrier_wait(
            consumer_p0_view, phase_qk
        )  # consumer wait for p0 use producer_qk0 due to reuse
        # reinterpret as p0
        p0_view = _reinterpret(qk0_buf, bufIdx_qk)
        v_view = tlx.local_view(v_buf, bufIdx_k)
        bufIdx_o, phase_o = _get_bufidx_phase(accum_cnt_k, NUM_BUFFERS_O)
        producer_commit_o0_view = tlx.local_view(producer_commit_o0, bufIdx_o)
        o0_view = tlx.local_view(o0_buf, bufIdx_o)
        tlx.async_dot(
            p0_view, v_view, o0_view, use_acc=True, mBarriers=[producer_commit_o0_view]
        )

        first = False
        accum_cnt_k += 1
        accum_cnt_qk += 1
        accum_cnt_qk1 += 1
        accum_cnt_o1 += 1

    # epilogue
    # commit to release q0, q1? FIXME
    release_q0_view = tlx.local_view(consumer_release_q0, bufIdx_q)
    # tlx.gen5_commit(release_q0_view)
    release_q1_view = tlx.local_view(consumer_release_q1, bufIdx_q)
    # tlx.gen5_commit(release_q1_view)
    tlx.barrier_wait(
        producer_o1_view,
        phase_o_outer,  # , first
    )  # producer acquire for o1 at the first iteration
    bufIdx_qk1, phase_qk1 = _get_bufidx_phase(accum_cnt_qk1, NUM_BUFFERS_QK)
    consumer_p1_view = tlx.local_view(producer_qk1, bufIdx_qk1)
    tlx.barrier_wait(
        consumer_p1_view, phase_qk1
    )  # consumer wait for p1 due to reuse of p1 and qk1
    p1_view = _reinterpret(qk1_buf, bufIdx_qk1)

    accum_cnt_qk1 += 1
    # release p0, p1 via producer_commit_qk0, qk1 barriers
    # accum_cnt_qk should be equal to accum_cnt_qk1 here
    # bufIdx_qk, phase_qk = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
    # consumer_release_p0_view = tlx.local_view(producer_commit_qk0, bufIdx_qk)
    # consumer_release_p1_view = tlx.local_view(producer_commit_qk1, bufIdx_qk)
    bufIdx_o, phase_o = _get_bufidx_phase(accum_cnt_o1, NUM_BUFFERS_O)
    producer_commit_o1_view = tlx.local_view(producer_commit_o1, bufIdx_o)
    bufIdx_v, phase_v = _get_bufidx_phase(accum_cnt_o1, NUM_BUFFERS_K)
    consumer_release_v_view = tlx.local_view(consumer_release_v, bufIdx_v)
    o1_view = tlx.local_view(o1_buf, bufIdx_o)
    tlx.async_dot(
        p1_view,
        v_view,
        o1_view,
        use_acc=not first,
        mBarriers=[
            producer_commit_o1_view,
            consumer_release_v_view,  # , consumer_release_p0_view, consumer_release_p1_view
        ],
    )
    return accum_cnt_k, accum_cnt_qk


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
    return x * 0.5 * (1 + tanh_approx_fp32(0.7978845608 * x * (1.0 + 0.044715 * x * x)))


@triton.jit
def _do_activation(
    qk_buffers,
    qk_scale,
    consumer_qk,
    consumer_release_qk,
    accum_cnt,
    v_dtype,
    activation_enum_int: tl.constexpr,
    NUM_BUFFERS_QK: tl.constexpr,
):
    # qk in tmem, output p in tmem
    bufIdx = accum_cnt % NUM_BUFFERS_QK
    phase = (accum_cnt // NUM_BUFFERS_QK) & 1
    qk_view = tlx.local_view(qk_buffers, bufIdx)
    consumer_qk_view = tlx.local_view(consumer_qk, bufIdx)
    tlx.barrier_wait(consumer_qk_view, phase)
    qk = tlx.local_load(qk_view, tlx.storage_kind.tmem)
    # ConsumerWait for qk, ProducerAcquire for p
    if activation_enum_int == 3:
        p = fast_gelu(qk)
    else:
        p = qk

    p *= qk_scale
    p = p.to(v_dtype)
    # p and qk reuse tmem space, single producer commit for p via consumer_release_qk
    consumer_release_qk_view = tlx.local_view(consumer_release_qk, bufIdx)
    tlx.barrier_arrive(consumer_release_qk_view, 1)


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
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_K: tl.constexpr,
    NUM_BUFFERS_QK: tl.constexpr,
    NUM_BUFFERS_O: tl.constexpr,
):
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    # start with on-device TMA where descriptors for k, v are set up outside of the persistent
    # loop and descriptor for q is set up inside the persistent loop.
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

    # allocate buffers for q0, q1
    q0_buf = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), tl.float16, 1)
    q1_buf = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), tl.float16, 1)

    # allocate buffers for k, v
    k_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), tl.float16, NUM_BUFFERS_K)  # k
    v_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), tl.float16, NUM_BUFFERS_K)  # v

    # allocate tmem for outputs of 4 dots (after partitioning)
    # qk0 = q0 dot k, qk1 = q1 dot k, acc0 = p0 dot v, acc1 = p1 dot v
    qk0_buf = tlx.local_alloc(
        (BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem
    )
    qk1_buf = tlx.local_alloc(
        (BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem
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
    consumer_k = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_K, arrive_count=1)
    consumer_v = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_K, arrive_count=1)
    consumer_release_k = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_K, arrive_count=1)
    consumer_release_v = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_K, arrive_count=1)

    # producer_qk0 == consumer_release_qk0
    producer_qk0 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_QK, arrive_count=1)
    # producer_commit_qk0 == consumer_qk0
    producer_commit_qk0 = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_QK, arrive_count=1
    )
    producer_qk1 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_QK, arrive_count=1)
    producer_commit_qk1 = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_QK, arrive_count=1
    )

    producer_o0 = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_O, arrive_count=1
    )  # only acquire for the first iteration
    producer_commit_o0 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    producer_o1 = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_O, arrive_count=1
    )  # only acquire for the first iteration
    producer_commit_o1 = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)

    with tlx.async_tasks():
        # activation calculation
        with tlx.async_task("default"):
            accum_cnt = 0
            accum_cnt_outer = 0
            for _ in range(0, tiles_per_sm):
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
                    for start_n in range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        ## communication channel for qk0, p0
                        _do_activation(
                            qk0_buf,
                            qk_scale,
                            producer_commit_qk0,
                            producer_qk0,
                            accum_cnt,
                            V.dtype.element_ty,
                            activation_enum_int,
                            NUM_BUFFERS_QK,
                        )
                        # wait for o0, o1 per iteration
                        bufIdx = accum_cnt % NUM_BUFFERS_O
                        phase = (accum_cnt // NUM_BUFFERS_O) & 1
                        # consumer wait of o0: producer_commit
                        consumer_o0_view = tlx.local_view(producer_commit_o0, bufIdx)
                        tlx.barrier_wait(consumer_o0_view, phase)
                        accum_cnt += 1

                    # epilogue here, load from tmem
                    bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
                        accum_cnt_outer, NUM_BUFFERS_O
                    )
                    o0_view = tlx.local_view(
                        o0_buf, bufIdx_o_outer
                    )  # FIXME: index for the last iteration
                    o0 = tlx.local_load(o0_view, tlx.storage_kind.tmem)
                    # release o0 here
                    consumer_release_o0_view = tlx.local_view(
                        producer_o0, bufIdx_o_outer
                    )
                    tlx.barrier_arrive(consumer_release_o0_view, 1)
                    o0_desc = tl.make_tensor_descriptor(
                        Out,
                        shape=[end_q.to(tl.int32), HEAD_DIM * H],
                        strides=[HEAD_DIM * H, 1],
                        block_shape=[BLOCK_M // 2, BLOCK_D],
                    )
                    o0_desc.store(
                        [
                            (begin_q + start_m * BLOCK_M).to(tl.int32),
                            (out_offset).to(tl.int32),
                        ],
                        o0.to(Out.type.element_ty),
                    )
                    accum_cnt_outer += 1

        with tlx.async_task(num_warps=4):
            accum_cnt = 0
            accum_cnt_outer = 0
            for _ in range(0, tiles_per_sm):
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
                        _do_activation(
                            qk1_buf,
                            qk_scale,
                            producer_commit_qk1,
                            producer_qk1,
                            accum_cnt,
                            V.dtype.element_ty,
                            activation_enum_int,
                            NUM_BUFFERS_QK,
                        )
                        # wait for o0, o1 per iteration
                        bufIdx = accum_cnt % NUM_BUFFERS_O
                        phase = (accum_cnt // NUM_BUFFERS_O) & 1
                        # consumer wait of o1
                        consumer_o1_view = tlx.local_view(producer_commit_o1, bufIdx)
                        tlx.barrier_wait(consumer_o1_view, phase)
                        accum_cnt += 1
                    # epilogue here, load from tmem
                    bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
                        accum_cnt_outer, NUM_BUFFERS_O
                    )
                    o0_desc = tl.make_tensor_descriptor(
                        Out,
                        shape=[end_q.to(tl.int32), HEAD_DIM * H],
                        strides=[HEAD_DIM * H, 1],
                        block_shape=[BLOCK_M // 2, BLOCK_D],
                    )
                    o1_view = tlx.local_view(
                        o1_buf, bufIdx_o_outer
                    )  # FIXME: should be 0
                    o1 = tlx.local_load(o1_view, tlx.storage_kind.tmem)
                    # release o1 here
                    consumer_release_o1_view = tlx.local_view(
                        producer_o1, bufIdx_o_outer
                    )
                    tlx.barrier_arrive(consumer_release_o1_view, 1)
                    o0_desc.store(
                        [
                            (begin_q + start_m * BLOCK_M + BLOCK_M // 2).to(tl.int32),
                            (out_offset).to(tl.int32),
                        ],
                        o1.to(Out.type.element_ty),
                    )
                    accum_cnt_outer += 1

        with tlx.async_task(num_warps=1):  # gemm
            accum_cnt_q = 0
            accum_cnt_k = 0
            accum_cnt_qk = 0
            accum_cnt_outer = 0
            for _ in range(0, tiles_per_sm):
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
                    accum_cnt_k, accum_cnt_qk = _do_dots(
                        klen,
                        q0_buf,
                        q1_buf,
                        k_buf,
                        v_buf,
                        qk0_buf,
                        qk1_buf,
                        o0_buf,
                        o1_buf,
                        consumer_q0,
                        consumer_q1,
                        consumer_k,
                        consumer_v,
                        producer_qk0,
                        producer_commit_qk0,
                        producer_qk1,
                        producer_commit_qk1,
                        consumer_release_k,
                        consumer_release_v,
                        consumer_release_q0,
                        consumer_release_q1,
                        producer_o0,
                        producer_commit_o0,
                        producer_o1,
                        producer_commit_o1,
                        accum_cnt_q,
                        accum_cnt_k,
                        accum_cnt_qk,
                        accum_cnt_outer,
                        NUM_BUFFERS_Q,
                        NUM_BUFFERS_K,
                        NUM_BUFFERS_QK,
                        NUM_BUFFERS_O,
                        BLOCK_N,
                    )
                    accum_cnt_q += 1
                    accum_cnt_outer += 1
                    # signal producer commit of epi0 and epi1, we don't want to block the gemm partition
                    # to wait for the completion

        with tlx.async_task(num_warps=1):  # load
            accum_count_q = 0
            accum_count_k = 0
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
                    # bufIdx, phase, empty_bars, full_bars, buffers, desc, offset_1, offset_0, num_bytes
                    bufIdx = q_bufIdx
                    phase = q_phase
                    empty_bars = consumer_release_q0
                    full_bars = consumer_q0
                    buffers = q0_buf
                    desc = q_desc
                    offset_1 = begin_q + start_m * BLOCK_M
                    offset_0 = q_offset
                    # num_bytes = BLOCK_M * BLOCK_D * 2
                    # producer acquire
                    empty_view = tlx.local_view(empty_bars, bufIdx)
                    tlx.barrier_wait(empty_view, phase ^ 1)
                    # barrier for producer commit
                    full_view = tlx.local_view(
                        consumer_q0, bufIdx
                    )  # full_bars, bufIdx)
                    tlx.barrier_expect_bytes(
                        full_view, BLOCK_M * BLOCK_D * 2
                    )  # num_bytes)
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

                    # _load_tma(
                    #    q_bufIdx,
                    #    q_phase,
                    #    consumer_release_q1,
                    #    consumer_q1,
                    #    q1_buf,
                    #    q_desc,
                    #    begin_q + start_m * BLOCK_M + BLOCK_M // 2,
                    #    q_offset,
                    #    BLOCK_M * BLOCK_D * 2,
                    # )
                    # bufIdx, phase, empty_bars, full_bars, buffers, desc, offset_1, offset_0, num_bytes
                    bufIdx = q_bufIdx
                    phase = q_phase
                    empty_bars = consumer_release_q1
                    full_bars = consumer_q1
                    buffers = q1_buf
                    desc = q_desc
                    offset_1 = begin_q + start_m * BLOCK_M + BLOCK_M // 2
                    offset_0 = q_offset
                    # num_bytes = BLOCK_M * BLOCK_D * 2
                    # producer acquire
                    empty_view = tlx.local_view(empty_bars, bufIdx)
                    tlx.barrier_wait(empty_view, phase ^ 1)
                    # barrier for producer commit
                    full_view = tlx.local_view(full_bars, bufIdx)
                    tlx.barrier_expect_bytes(
                        full_view, BLOCK_M * BLOCK_D * 2
                    )  # num_bytes)
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

                    lo, hi = 0, klen
                    for start_n in range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        k_bufIdx = accum_count_k % NUM_BUFFERS_K
                        k_phase = (accum_count_k // NUM_BUFFERS_K) & 1
                        # k_view = _load_tma(
                        #    k_bufIdx,
                        #    k_phase,
                        #    consumer_release_k,
                        #    consumer_k,
                        #    k_buf,
                        #    k_desc,
                        #    begin_k + start_n,
                        #    kv_offset,
                        #    BLOCK_N * BLOCK_D * 2,
                        # )
                        # bufIdx, phase, empty_bars, full_bars, buffers, desc, offset_1, offset_0, num_bytes
                        bufIdx = k_bufIdx
                        phase = k_phase
                        empty_bars = consumer_release_k
                        full_bars = consumer_k
                        buffers = k_buf
                        desc = k_desc
                        offset_1 = begin_k + start_n
                        offset_0 = kv_offset
                        num_bytes: tl.constexpr = BLOCK_N * BLOCK_D * 2
                        # producer acquire
                        empty_view = tlx.local_view(empty_bars, bufIdx)
                        tlx.barrier_wait(empty_view, phase ^ 1)
                        # barrier for producer commit
                        full_view = tlx.local_view(full_bars, bufIdx)
                        tlx.barrier_expect_bytes(
                            full_view, BLOCK_N * BLOCK_D * 2
                        )  # num_bytes)
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
                        k_view = smem_view
                        k_view = tlx.local_trans(k_view)
                        # _load_tma(
                        #    k_bufIdx,
                        #    k_phase,
                        #    consumer_release_v,
                        #    consumer_v,
                        #    v_buf,
                        #    v_desc,
                        #    begin_k + start_n,
                        #    kv_offset,
                        #    BLOCK_N * BLOCK_D * 2,
                        # )
                        # bufIdx, phase, empty_bars, full_bars, buffers, desc, offset_1, offset_0, num_bytes
                        bufIdx = k_bufIdx
                        phase = k_phase
                        empty_bars = consumer_release_v
                        full_bars = consumer_v
                        buffers = v_buf
                        desc = v_desc
                        offset_1 = begin_k + start_n
                        offset_0 = kv_offset
                        num_bytes2: tl.constexpr = BLOCK_N * BLOCK_D * 2
                        # producer acquire
                        empty_view = tlx.local_view(empty_bars, bufIdx)
                        tlx.barrier_wait(empty_view, phase ^ 1)
                        # barrier for producer commit
                        full_view = tlx.local_view(full_bars, bufIdx)
                        tlx.barrier_expect_bytes(full_view, num_bytes2)
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

                    accum_count_q += 1


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
    nheads = query.shape[1]
    G = query.shape[1] // key.shape[1]
    assert query.shape[1] % key.shape[1] == 0
    batch_size = BATCH * nheads
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
    kstrides = k.stride()
    vstrides = v.stride()

    activation_enum_int = activation_string_to_int(activation)

    gdpa_kernel_tma_ws_blackwell[grid_tma_persistent](
        q,
        query_offset,
        k,
        key_offset,
        v,
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
        **extra_kern_args,
    )
    return o
