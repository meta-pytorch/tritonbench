"""
Based on https://github.com/pytorch/pytorch/issues/120184.
Generated from Inductor for forward layernorm with and without welford
"""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice
from tritonbench.utils.env_utils import get_device_module

reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_DMODEL": 64,
            },
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["num_queries"],
)
@triton.jit
def triton_tem_fused_no_exp2(
    arg_Q,
    arg_K,
    arg_V,
    out_ptr0,
    num_queries: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    Q = arg_Q
    K = arg_K
    V = arg_V

    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head

    # Define Q Strides
    stride_qz = 4194304
    stride_qh = 262144
    stride_qm = 64
    stride_qk = 1
    # Define K Strides
    stride_kz = 4194304
    stride_kh = 262144
    stride_kn = 64
    stride_kk = 1
    # Define V Strides
    stride_vz = 4194304
    stride_vh = 262144
    stride_vk = 64
    stride_vn = 1

    Z = 16
    H = 16
    N_CTX = 4096

    # TODO I think we should do some performance work
    # to find the optimal calls for perf/accuracy to tl.dot
    qk_scale = 1.0
    MATMUL_PRECISION = tl.float16

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    qkv_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    Q_ptrs = Q + qkv_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    K_ptrs = K + qkv_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    V_ptrs = V + qkv_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # TODO fix me
    # qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_ptrs)
    q = (q * qk_scale).to(MATMUL_PRECISION)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k, v --
        k = tl.load(K_ptrs)
        v = tl.load(V_ptrs)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k.to(MATMUL_PRECISION))
        # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~

        tmp0 = tl.full([1], 1024, tl.int64)
        tmp1 = (offs_m[:, None]) <= tmp0
        tmp2 = (start_n + offs_n[None, :]) <= tmp0
        tmp3 = tmp1 & tmp2
        tmp4 = (offs_m[:, None]) >= (start_n + offs_n[None, :])
        tmp5 = tmp3 | tmp4
        tmp6 = float("-inf")
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.where(tmp5, (qk), tmp7)
        qk = tmp8

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # -- compute scaling constant ---
        row_max = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, row_max)
        masked_out_rows = m_i_new == float("-inf")

        # TODO FIX ME and use 2^x instead of exp
        # alpha = tl.math.exp2(m_i - m_i_new)
        # p = tl.math.exp2(qk - m_i_new[:, None])
        alpha = tl.math.exp(m_i - m_i_new)
        alpha = tl.where(masked_out_rows, 0, alpha)
        p = tl.math.exp(qk - m_i_new[:, None])
        p = tl.where(masked_out_rows[:, None], 0, p)

        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(MATMUL_PRECISION), v.to(MATMUL_PRECISION))

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vk

    # write back l and m
    acc = acc / l_i[:, None]
    # TODO For backward support we need to add the Logsumexp
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    idx_z = tl.program_id(1) // H
    idx_h = tl.program_id(1) % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]
    # TODO generalize and add proper mask support
    mask = (idx_m != -1) & (idx_d != -1)
    xindex = idx_d + (64 * idx_m) + (262144 * idx_h) + (4194304 * idx_z)
    tl.store(out_ptr0 + (xindex), acc, None)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_DMODEL": 64,
            },
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["num_queries"],
)
@triton.jit
def triton_tem_fused_with_exp2(
    arg_Q,
    arg_K,
    arg_V,
    out_ptr0,
    num_queries: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # updated version
    SCORE_MOD_IS_LINEAR: tl.constexpr = False
    ROWS_GUARANTEED_SAFE: tl.constexpr = False
    Q = arg_Q
    K = arg_K
    V = arg_V

    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # (Modifiable) Config options:
    # BLOCK_M
    # BLOCK_N
    # SCORE_MOD_IS_LINEAR: Is the score modifier linear? If so, we can lift the
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check

    # Define Q Strides
    stride_qz = 4194304
    stride_qh = 262144
    stride_qm = 64
    stride_qk = 1
    # Define K Strides
    stride_kz = 4194304
    stride_kh = 262144
    stride_kn = 64
    stride_kk = 1
    # Define V Strides
    stride_vz = 4194304
    stride_vh = 262144
    stride_vk = 64
    stride_vn = 1

    Z = 16
    H = 16
    N_CTX = 4096

    qk_scale = 1.0
    MATMUL_PRECISION = Q.dtype.element_ty

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    qkv_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    Q_ptrs = Q + qkv_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    K_ptrs = K + qkv_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    V_ptrs = V + qkv_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_ptrs)
    if SCORE_MOD_IS_LINEAR:
        qk_scale *= 1.44269504
    q = (q * qk_scale).to(MATMUL_PRECISION)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k, v --
        k = tl.load(K_ptrs)
        v = tl.load(V_ptrs)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.to(MATMUL_PRECISION), acc=qk)
        # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
        tmp0 = tl.full([1], 1024, tl.int64)
        tmp1 = (offs_m[:, None]) <= tmp0
        tmp2 = (start_n + offs_n[None, :]) <= tmp0
        tmp3 = tmp1 & tmp2
        tmp4 = (offs_m[:, None]) >= (start_n + offs_n[None, :])
        tmp5 = tmp3 | tmp4
        tmp6 = float("-inf")
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.where(tmp5, (qk), tmp7)
        qk = tmp8

        # TODO: In the case that score_mod is linear, this can be LICMed
        if not SCORE_MOD_IS_LINEAR:
            qk *= 1.44269504
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # -- compute scaling constant ---
        row_max = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, row_max)
        masked_out_rows = m_i_new == float("-inf")

        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        if not ROWS_GUARANTEED_SAFE:
            alpha = tl.where(masked_out_rows, 0, alpha)
            p = tl.where(masked_out_rows[:, None], 0, p)

        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc = tl.dot(p.to(MATMUL_PRECISION), v.to(MATMUL_PRECISION), acc)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vk

    # write back l and m
    acc = acc / l_i[:, None]
    # TODO For backward support we need to add the Logsumexp
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    idx_z = tl.program_id(1) // H
    idx_h = tl.program_id(1) % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]
    # TODO generalize and add proper mask support
    mask = (idx_m != -1) & (idx_d != -1)
    xindex = idx_d + (64 * idx_m) + (262144 * idx_h) + (4194304 * idx_z)
    tl.store(out_ptr0 + (xindex), acc, None)


def triton_attention_no_exp2(arg0_1, arg1_1, arg2_1):
    # 4194304: 1024*4096 = 16*4096*64, 262144 = 16 * 4096
    assert_size_stride(arg0_1, (16, 16, 4096, 64), (4194304, 262144, 64, 1))
    assert_size_stride(arg1_1, (16, 16, 4096, 64), (4194304, 262144, 64, 1))
    assert_size_stride(arg2_1, (16, 16, 4096, 64), (4194304, 262144, 64, 1))
    with get_device_module(arg0_1.device.type).device(arg0_1.device.index):
        buf0 = torch.empty_strided(
            (16, 16, 4096, 64),
            (4194304, 262144, 64, 1),
            dtype=torch.float16,
            device=arg0_1.device,
        )

        # batch_size, num_heads, num_queries: 16, 16, 4096
        num_queries = 4096
        batch_size = 16
        num_heads = 16
        grid = lambda META: (
            triton.cdiv(num_queries, META["BLOCK_M"]),
            batch_size * num_heads,
            1,
        )
        triton_tem_fused_no_exp2[grid](arg0_1, arg1_1, arg2_1, buf0, num_queries)
    return (buf0,)


def triton_attention_with_exp2(arg0_1, arg1_1, arg2_1):
    assert_size_stride(arg0_1, (16, 16, 4096, 64), (4194304, 262144, 64, 1))
    assert_size_stride(arg1_1, (16, 16, 4096, 64), (4194304, 262144, 64, 1))
    assert_size_stride(arg2_1, (16, 16, 4096, 64), (4194304, 262144, 64, 1))
    with get_device_module(arg0_1.device.type).device(arg0_1.device.index):
        buf0 = torch.empty_strided(
            (16, 16, 4096, 64),
            (4194304, 262144, 64, 1),
            dtype=torch.float16,
            device=arg0_1.device,
        )

        # batch_size, num_heads, num_queries: 16, 16, 4096
        num_queries = 4096
        batch_size = 16
        num_heads = 16
        grid = lambda META: (
            triton.cdiv(num_queries, META["BLOCK_M"]),
            batch_size * num_heads,
            1,
        )
        triton_tem_fused_with_exp2[grid](arg0_1, arg1_1, arg2_1, buf0, num_queries)
    return (buf0,)
