import functools

import torch
from tritonbench.utils.tilelang_utils import preload_cuda_driver

preload_cuda_driver()

import tilelang
import tilelang.language as T

TILELANG_DTYPE_MAP = {
    torch.bfloat16: T.bfloat16,
    torch.float16: T.float16,
}


def _to_tilelang_dtype(dtype: torch.dtype):
    try:
        return TILELANG_DTYPE_MAP[dtype]
    except KeyError as exc:
        raise NotImplementedError("TileLang Blackwell MHA only supports fp16/bf16") from exc


@tilelang.jit(
    out_idx=[3, 4],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _flashattn_fwd(batch, heads, seq_len, dim, is_causal, in_dtype, block_M, block_N):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape = [batch, seq_len, heads, dim]
    accum_dtype = T.float32

    @T.prim_func
    def flash_fwd(
        Q: T.Tensor(shape, in_dtype),
        K: T.Tensor(shape, in_dtype),
        V: T.Tensor(shape, in_dtype),
        Output: T.Tensor(shape, in_dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_len, block_M), heads, batch, threads=128
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], in_dtype)
            K_shared = T.alloc_shared([block_N, dim], in_dtype)
            V_shared = T.alloc_shared([block_N, dim], in_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], in_dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(acc_s.dtype),
                            0,
                        )

                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return flash_fwd


@tilelang.jit(
    out_idx=[2],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _flashattn_bwd_preprocess(batch, heads, seq_len, dim, in_dtype):
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
        O: T.Tensor(shape, in_dtype),
        dO: T.Tensor(shape, in_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], in_dtype)
            do = T.alloc_fragment([blk, blk], in_dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(
                    O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o
                )
                T.copy(
                    dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do
                )
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return flash_bwd_prep


def _make_dq_layout(dq):
    return T.Layout(
        dq.shape,
        lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2],
    )


@tilelang.jit(
    out_idx=[1],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _flashattn_bwd_postprocess(batch, heads, seq_len, dim, in_dtype):
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 64

    @T.prim_func
    def flash_bwd_post(
        dQ: T.Tensor(shape, accum_dtype),
        dQ_out: T.Tensor(shape, in_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (
            bx,
            by,
            bz,
        ):
            T.annotate_layout({dQ: _make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk : (bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk : (bx + 1) * blk, by, :],
            )

    return flash_bwd_post


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _flashattn_bwd(
    batch, heads, seq_len, dim, is_causal, in_dtype, block_M, block_N
):
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504
    shape = [batch, seq_len, heads, dim]
    accum_dtype = T.float32

    @T.prim_func
    def flash_bwd(
        Q: T.Tensor(shape, in_dtype),
        K: T.Tensor(shape, in_dtype),
        V: T.Tensor(shape, in_dtype),
        dO: T.Tensor(shape, in_dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(shape, accum_dtype),
        dK: T.Tensor(shape, in_dtype),
        dV: T.Tensor(shape, in_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=128) as (
            bx,
            by,
            bz,
        ):
            K_shared = T.alloc_shared([block_M, dim], in_dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], in_dtype)
            q = T.alloc_shared([block_N, dim], in_dtype)
            V_shared = T.alloc_shared([block_M, dim], in_dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], in_dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], in_dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim], in_dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim], in_dtype)
            dk_shared = T.alloc_shared([block_M, dim], in_dtype)

            T.annotate_layout({dQ: _make_dq_layout(dQ)})
            T.copy(K[bz, by * block_M : (by + 1) * block_M, bx, :], K_shared)
            T.copy(V[bz, by * block_M : (by + 1) * block_M, bx, :], V_shared)
            T.clear(dv)
            T.clear(dk)

            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=2):
                T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(
                    K_shared,
                    q,
                    qkT,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            by * block_M + i <= k * block_N + j, qkT[i, j], 0
                        )

                T.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                T.clear(dsT)
                T.gemm(
                    V_shared,
                    do,
                    dsT,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim):
                    T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, by * block_M : (by + 1) * block_M, bx, :])
            T.copy(dk_shared, dK[bz, by * block_M : (by + 1) * block_M, bx, :])

    return flash_bwd


@functools.lru_cache(maxsize=None)
def _get_fwd_kernel(batch, heads, seq_len, dim, causal, dtype):
    block_M = 64
    block_N = 64 if dim <= 128 else 32
    return _flashattn_fwd(
        batch,
        heads,
        seq_len,
        dim,
        causal,
        _to_tilelang_dtype(dtype),
        block_M,
        block_N,
    )


@functools.lru_cache(maxsize=None)
def _get_bwd_kernels(batch, heads, seq_len, dim, causal, dtype):
    in_dtype = _to_tilelang_dtype(dtype)
    block_M = 64
    block_N = 64 if dim <= 64 else 32
    return (
        _flashattn_bwd_preprocess(batch, heads, seq_len, dim, in_dtype),
        _flashattn_bwd_postprocess(batch, heads, seq_len, dim, in_dtype),
        _flashattn_bwd(
            batch, heads, seq_len, dim, causal, in_dtype, block_M, block_N
        ),
    )


class _TilelangBlackwellAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise NotImplementedError("TileLang Blackwell MHA requires matching dtypes")
        if q.shape != k.shape or q.shape != v.shape:
            raise NotImplementedError("TileLang Blackwell MHA only supports MHA shapes")

        batch, seq_len, heads, dim = q.shape
        kernel = _get_fwd_kernel(batch, heads, seq_len, dim, causal, q.dtype)
        o, lse = kernel(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        batch, seq_len, heads, dim = q.shape

        def maybe_contiguous(x):
            return x if x.stride(-1) == 1 else x.contiguous()

        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        prep_kernel, post_kernel, bwd_kernel = _get_bwd_kernels(
            batch, heads, seq_len, dim, ctx.causal, q.dtype
        )
        delta = prep_kernel(o, do)

        shape = (batch, seq_len, heads, dim)
        dq = torch.zeros(shape, dtype=torch.float32, device=q.device)
        dk = torch.empty(shape, dtype=q.dtype, device=q.device)
        dv = torch.empty(shape, dtype=q.dtype, device=q.device)
        bwd_kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = post_kernel(dq)
        return dq, dk, dv, None


def tilelang_blackwell_attention(q, k, v, causal):
    return _TilelangBlackwellAttention.apply(q, k, v, causal)
