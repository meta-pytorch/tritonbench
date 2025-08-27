# TLX GDPA kernel optimized for Blackwell Warp Specialization

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

from .gdpa_utils import get_num_sms
from .math import activation_string_to_int


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


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "NUM_BUFFERS_Q": bq,
                "NUM_BUFFERS_KV": bk,
                "NUM_BUFFERS_QK": bqk,
                "NUM_BUFFERS_O": bo,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=_host_descriptor_pre_hook,
        )
        for BM in [256]  # 128 or 256
        for BN in [128]
        for bq in [1]
        for bk in [3]
        for bqk in [1]  # in tmem
        for bo in [1]  # in tmem
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

    with tlx.async_tasks():
        # activation calculation
        with tlx.async_task("default", registers=192):
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
                        tlx.barrier_wait(consumer_qk_view, phase)

                        # qk_view: BLOCK_M // 2, HEAD_DIM
                        qk_view_1st = tlx.subslice(qk_view, 0, HEAD_DIM // 2)
                        qk0 = tlx.local_load(qk_view_1st)
                        p0 = fast_gelu(qk0)
                        p0 = p0.to(dtype)
                        p0_view = tlx.local_reinterpret(qk_view_1st, dtype)
                        tlx.local_store(p0_view, p0)

                        qk_view_2nd = tlx.subslice(
                            qk_view, HEAD_DIM // 2, HEAD_DIM // 2
                        )
                        qk0 = tlx.local_load(qk_view_2nd)
                        p0 = fast_gelu(qk0)
                        p0 = p0.to(dtype)
                        p0_view = tlx.local_reinterpret(qk_view_2nd, dtype)
                        tlx.local_store(p0_view, p0)

                        # p and qk reuse tmem space, single producer commit for p via consumer_release_qk
                        consumer_release_qk_view = tlx.local_view(producer_qk0, bufIdx)
                        tlx.barrier_arrive(consumer_release_qk_view, 1)

                        # wait for o0, o1 per iteration
                        bufIdx = accum_cnt % NUM_BUFFERS_O
                        phase = (accum_cnt // NUM_BUFFERS_O) & 1
                        # consumer wait of o0: producer_commit
                        consumer_o0_view = tlx.local_view(producer_commit_o0, bufIdx)
                        # tl.device_print("default producer_commit_o0", accum_cnt)
                        # tl.device_print("default producer_commit_o0_phase", phase)
                        tlx.barrier_wait(consumer_o0_view, phase)
                        accum_cnt += 1

                    # epilogue here, load from tmem
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
                    if USE_ON_DEVICE_TMA:
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
                    o_desc.store(
                        [
                            (begin_q + start_m * BLOCK_M).to(tl.int32),
                            (out_offset).to(tl.int32),
                        ],
                        o0,
                    )
                    accum_cnt_outer += 1
                tile_idx += num_progs

        with tlx.async_task(num_warps=4, registers=192):
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
                        bufIdx = accum_cnt % NUM_BUFFERS_QK
                        phase = (accum_cnt // NUM_BUFFERS_QK) & 1
                        qk_view = tlx.local_view(qk1_buf, bufIdx)
                        consumer_qk_view = tlx.local_view(producer_commit_qk1, bufIdx)
                        tlx.barrier_wait(consumer_qk_view, phase)

                        # qk_view: BLOCK_M // 2, HEAD_DIM
                        qk_view_1st = tlx.subslice(qk_view, 0, HEAD_DIM // 2)
                        qk0 = tlx.local_load(qk_view_1st)
                        p0 = fast_gelu(qk0)
                        p0 = p0.to(dtype)
                        p0_view = tlx.local_reinterpret(qk_view_1st, dtype)
                        tlx.local_store(p0_view, p0)

                        qk_view_2nd = tlx.subslice(
                            qk_view, HEAD_DIM // 2, HEAD_DIM // 2
                        )
                        qk0 = tlx.local_load(qk_view_2nd)
                        p0 = fast_gelu(qk0)
                        p0 = p0.to(dtype)
                        p0_view = tlx.local_reinterpret(qk_view_2nd, dtype)
                        tlx.local_store(p0_view, p0)

                        # p and qk reuse tmem space, single producer commit for p via consumer_release_qk
                        consumer_release_qk_view = tlx.local_view(producer_qk1, bufIdx)
                        tlx.barrier_arrive(consumer_release_qk_view, 1)

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
                    if USE_ON_DEVICE_TMA:
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
                    o_desc.store(
                        [
                            (begin_q + start_m * BLOCK_M + BLOCK_M // 2).to(tl.int32),
                            (out_offset).to(tl.int32),
                        ],
                        o1,
                    )
                    accum_cnt_outer += 1
                tile_idx += num_progs

        with tlx.async_task(num_warps=1, registers=24):  # gemm
            accum_cnt_q = 0
            accum_cnt_kv = 0
            accum_cnt_o = 0
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
                    tlx.barrier_wait(
                        consumer_p0_view, phase_p
                    )  # consumer wait for p0 due to reuse of p0 and qk0
                    # reinterpret qk0 as p0
                    qk_view = tlx.local_view(qk0_buf, bufIdx_p)
                    p0_view = tlx.local_reinterpret(qk_view, dtype)

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
                        tlx.barrier_wait(
                            consumer_p1_view, phase_qk1
                        )  # consumer wait for p1 use producer_qk1 due to reuse
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
                        qk_view = tlx.local_view(qk1_buf, bufIdx_qk1)
                        p1_view = tlx.local_reinterpret(qk_view, dtype)
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
                        tlx.barrier_wait(
                            consumer_p0_view, phase_qk
                        )  # consumer wait for p0 use producer_qk0 due to reuse
                        # reinterpret as p0
                        qk_view = tlx.local_view(qk0_buf, bufIdx_qk)
                        p0_view = tlx.local_reinterpret(qk_view, dtype)

                        v_view = tlx.local_view(kv_buf, bufIdx_v)
                        bufIdx_o, phase_o = _get_bufidx_phase(
                            accum_cnt_o, NUM_BUFFERS_O
                        )
                        producer_commit_o0_view = tlx.local_view(
                            producer_commit_o0, bufIdx_o
                        )
                        o0_view = tlx.local_view(o0_buf, bufIdx_o)
                        tlx.async_dot(
                            p0_view,
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
                    qk_view = tlx.local_view(qk1_buf, bufIdx_qk1)
                    p1_view = tlx.local_reinterpret(qk_view, dtype)

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
                        p1_view,
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
                        tlx.barrier_wait(k_empty_view, k_phase)  # ^ 1)
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
                        # tl.device_print("load consumer_v_buf", v_bufIdx)
                        accum_cnt_kv += 2
                    # outside of inner for
                    accum_count_q += 1
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
    nheads = query.shape[1]
    G = query.shape[1] // key.shape[1]
    assert query.shape[1] % key.shape[1] == 0
    batch_size = BATCH * nheads
    NUM_SMS = (
        get_num_sms() or 1000000
    ) * 8  # if num sms is None, use a large number so that it is a no-op
    print("NUM_SMS", NUM_SMS)
    print(triton.cdiv(max_seq_len_q, 256) * BATCH * nheads)

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
    print(q.shape, k.shape, v.shape)
    # print("activation_enum_int", activation, activation_enum_int)
    # print(query_offset)
    # print(key_offset)

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
        **extra_kern_args,
    )
    return o
