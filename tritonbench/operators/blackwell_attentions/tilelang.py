import functools

import torch
from tritonbench.utils.tilelang_utils import preload_cuda_driver

preload_cuda_driver()

import tilelang
import tilelang.language as T

PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
}


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=128,
    block_N=128,
    variant="ss",
):
    use_ts = variant == "ts"
    threads = 256 if use_ts else 128
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            D_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            mbar_s = T.alloc_barrier(1)
            mbar_d = T.alloc_barrier(1)

            if use_ts:
                P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            else:
                P_shared = T.alloc_shared([block_M, block_N], dtype)

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)
            D_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(O_reg, 0)
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

                T.tcgen05_gemm(
                    Q_shared,
                    K_shared,
                    S_tmem,
                    transpose_B=True,
                    mbar=mbar_s,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_s, k % 2)
                T.copy(S_tmem, S_reg)

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            S_reg[i, j],
                            -T.infinity(accum_dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(accum_dtype),
                            S_reg[i, j],
                        )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(S_reg, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_M, block_N):
                    S_reg[i, j] = T.exp2(S_reg[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(S_reg, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] *= scores_scale[i]

                T.copy(S_reg, P_cast)
                if use_ts:
                    T.copy(P_cast, P_tmem)
                    P_operand = P_tmem
                else:
                    T.copy(P_cast, P_shared)
                    P_operand = P_shared

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.tcgen05_gemm(
                    P_operand,
                    V_shared,
                    D_tmem,
                    mbar=mbar_d,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_d, k % 2)

                T.copy(D_tmem, D_reg)
                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] += D_reg[i, j]

            for i, j in T.Parallel(block_M, dim):
                O_reg[i, j] /= logsum[i]
            T.copy(O_reg, O_shared)
            T.copy(
                O_shared,
                Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
            )

    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn_wasp(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=128,
    block_N=128,
    threads=256,
    num_stages=2,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared_0 = T.alloc_shared([block_N, dim], dtype)
            K_shared_1 = T.alloc_shared([block_N, dim], dtype)
            V_shared_0 = T.alloc_shared([block_N, dim], dtype)
            V_shared_1 = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            O_tmem = T.alloc_tmem([block_M, dim], accum_dtype)

            mbar_dma1_empty = T.alloc_barrier([32] * num_stages)
            mbar_dma1_full = T.alloc_barrier([32] * num_stages)
            mbar_bmm1_empty = T.alloc_barrier([128] * num_stages)
            mbar_bmm1_full = T.alloc_barrier([1] * num_stages)
            mbar_dma2_empty = T.alloc_barrier([32] * num_stages)
            mbar_dma2_full = T.alloc_barrier([32] * num_stages)
            mbar_bmm2_full = T.alloc_barrier([1] * num_stages)
            mbar_softmax_empty = T.alloc_barrier([32] * num_stages)
            mbar_softmax_full = T.alloc_barrier([128] * num_stages)
            mbar_correction_full = T.alloc_barrier([32] * num_stages)

            tid = T.get_thread_binding()

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_rescale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            if tid < 128:
                T.fill(O_reg, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.copy(O_reg, O_tmem)

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.serial(loop_range):
                parity = (k // num_stages) & 1
                parity_inv = parity ^ 1
                stage_id = k % num_stages
                is_clear_accum = k == 0

                if 128 <= tid < 160:
                    T.mbarrier_wait_parity(mbar_dma1_empty[stage_id], parity_inv)
                    if k == 0:
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

                    if stage_id == 0:
                        T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared_0)
                    else:
                        T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared_1)
                    T.mbarrier_arrive(mbar_dma1_full[stage_id])

                    T.mbarrier_wait_parity(mbar_dma2_empty[stage_id], parity_inv)
                    if stage_id == 0:
                        T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared_0)
                    else:
                        T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared_1)
                    T.mbarrier_arrive(mbar_dma2_full[stage_id])

                elif 160 <= tid < 192:
                    T.mbarrier_wait_parity(mbar_dma1_full[stage_id], parity)
                    T.mbarrier_wait_parity(mbar_bmm1_empty[stage_id], parity_inv)

                    if stage_id == 0:
                        T.tcgen05_gemm(
                            Q_shared,
                            K_shared_0,
                            S_tmem,
                            transpose_B=True,
                            mbar=mbar_bmm1_full[stage_id],
                            clear_accum=True,
                        )
                    else:
                        T.tcgen05_gemm(
                            Q_shared,
                            K_shared_1,
                            S_tmem,
                            transpose_B=True,
                            mbar=mbar_bmm1_full[stage_id],
                            clear_accum=True,
                        )
                    T.mbarrier_arrive(mbar_dma1_empty[stage_id])

                    T.mbarrier_wait_parity(mbar_softmax_full[stage_id], parity)
                    T.mbarrier_wait_parity(mbar_dma2_full[stage_id], parity)
                    if stage_id == 0:
                        T.tcgen05_gemm(
                            P_tmem,
                            V_shared_0,
                            O_tmem,
                            mbar=mbar_bmm2_full[stage_id],
                            clear_accum=is_clear_accum,
                        )
                    else:
                        T.tcgen05_gemm(
                            P_tmem,
                            V_shared_1,
                            O_tmem,
                            mbar=mbar_bmm2_full[stage_id],
                            clear_accum=is_clear_accum,
                        )

                    T.mbarrier_arrive(mbar_softmax_empty[stage_id])
                    T.mbarrier_arrive(mbar_dma2_empty[stage_id])

                    if k == loop_range - 1:
                        T.mbarrier_arrive(mbar_correction_full[0])

                elif tid < 128:
                    T.mbarrier_wait_parity(mbar_softmax_empty[stage_id], parity_inv)
                    T.mbarrier_wait_parity(mbar_bmm1_full[stage_id], parity)
                    if k > 0:
                        prev_stage = (k - 1) % num_stages
                        prev_parity = ((k - 1) // num_stages) & 1
                        T.mbarrier_wait_parity(mbar_bmm2_full[prev_stage], prev_parity)

                    T.copy(O_tmem, O_reg)
                    T.copy(S_tmem, S_reg)

                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            S_reg[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                S_reg[i, j],
                                -T.infinity(accum_dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            S_reg[i, j] = T.if_then_else(
                                k * block_N + j >= seq_len,
                                -T.infinity(accum_dtype),
                                S_reg[i, j],
                            )

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(S_reg, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_M):
                        scores_rescale[i] = T.exp2(
                            scores_max_prev[i] * scale - scores_max[i] * scale
                        )
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.exp2(S_reg[i, j] * scale - scores_max[i] * scale)

                    T.reduce_sum(S_reg, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_rescale[i] + scores_sum[i]

                    for i, j in T.Parallel(block_M, dim):
                        O_reg[i, j] *= scores_rescale[i]

                    T.copy(S_reg, P_cast)
                    T.copy(P_cast, P_tmem)
                    T.copy(O_reg, O_tmem)

                    T.mbarrier_arrive(mbar_softmax_full[stage_id])
                    T.mbarrier_arrive(mbar_bmm1_empty[stage_id])

                    if k == loop_range - 1:
                        T.mbarrier_wait_parity(mbar_correction_full[0], 0)
                        T.mbarrier_wait_parity(mbar_bmm2_full[stage_id], parity)
                        T.copy(O_tmem, O_reg)
                        for i, j in T.Parallel(block_M, dim):
                            O_reg[i, j] /= logsum[i]
                        T.copy(O_reg, O_shared)
                        T.copy(
                            O_shared,
                            Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
                        )

    return main


@tilelang.jit(out_idx=[3, 4], pass_configs=PASS_CFG)
def flashattn_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
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
                T.tcgen05_gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s, acc_s_cast)
                T.tcgen05_gemm(
                    acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return main


@tilelang.jit(out_idx=[2], pass_configs=PASS_CFG)
def flashattn_bwd_preprocess(batch, heads, seq_len, dim):
    dtype = T.bfloat16
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 32

    @T.prim_func
    def main(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(
                    O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o
                )
                T.copy(
                    dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk],
                    do,
                )
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return main


def make_dq_layout(dQ):
    return T.Layout(
        dQ.shape,
        lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2],
    )


@tilelang.jit(out_idx=[1], pass_configs=PASS_CFG)
def flashattn_bwd_postprocess(batch, heads, seq_len, dim):
    dtype = T.bfloat16
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 64

    @T.prim_func
    def main(
        dQ: T.Tensor(shape, accum_dtype),
        dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (
            bx,
            by,
            bz,
        ):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk : (bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk : (bx + 1) * blk, by, :],
            )

    return main


@tilelang.jit(pass_configs=PASS_CFG)
def flashattn_bwd(
    batch, heads, seq_len, dim, is_causal, block_M, block_N, threads=128, num_stages=2
):
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504
    shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(shape, accum_dtype),
        dK: T.Tensor(shape, dtype),
        dV: T.Tensor(shape, dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_M, dim], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim], dtype)
            dk_shared = T.alloc_shared([block_M, dim], dtype)

            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(K[bz, by * block_M : (by + 1) * block_M, bx, :], K_shared)
            T.copy(V[bz, by * block_M : (by + 1) * block_M, bx, :], V_shared)
            T.clear(dv)
            T.clear(dk)
            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.tcgen05_gemm(
                    K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
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
                T.tcgen05_gemm(
                    V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                )
                T.copy(qkT, qkT_cast)
                T.tcgen05_gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.tcgen05_gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.tcgen05_gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim):
                    T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, by * block_M : (by + 1) * block_M, bx, :])
            T.copy(dk_shared, dK[bz, by * block_M : (by + 1) * block_M, bx, :])

    return main


def flashattn_bwd_warp(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    return flashattn_bwd(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        block_M,
        block_N,
        threads=256,
        num_stages=2,
    )


@functools.lru_cache(maxsize=None)
def _get_wasp_fwd_kernel(batch, heads, seq_len, dim, causal):
    return flashattn_wasp(
        batch,
        heads,
        seq_len,
        dim,
        causal,
        block_M=128,
        block_N=128,
        threads=256,
        num_stages=2,
    )


@functools.lru_cache(maxsize=None)
def _get_aux_fwd_kernel(batch, heads, seq_len, dim, causal):
    block_M = 64
    block_N = 64 if dim <= 64 else 32
    return flashattn_fwd(batch, heads, seq_len, dim, causal, block_M, block_N)


@functools.lru_cache(maxsize=None)
def _get_warp_bwd_kernels(batch, heads, seq_len, dim, causal):
    block_M = 64
    block_N = 64 if dim <= 64 else 32
    return (
        flashattn_bwd_preprocess(batch, heads, seq_len, dim),
        flashattn_bwd_postprocess(batch, heads, seq_len, dim),
        flashattn_bwd_warp(batch, heads, seq_len, dim, causal, block_M, block_N),
    )


class _TilelangBlackwellAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise NotImplementedError("TileLang Blackwell MHA only supports bf16")
        if q.shape != k.shape or q.shape != v.shape:
            raise NotImplementedError("TileLang Blackwell MHA only supports MHA shapes")

        batch, seq_len, heads, dim = q.shape
        kernel = _get_wasp_fwd_kernel(batch, heads, seq_len, dim, causal)
        o = kernel(q, k, v)
        ctx.save_for_backward(q, k, v, o)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o = ctx.saved_tensors
        batch, seq_len, heads, dim = q.shape

        def maybe_contiguous(x):
            return x if x.stride(-1) == 1 else x.contiguous()

        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]

        aux_fwd = _get_aux_fwd_kernel(batch, heads, seq_len, dim, ctx.causal)
        _, lse = aux_fwd(q, k, v)
        prep_kernel, post_kernel, bwd_kernel = _get_warp_bwd_kernels(
            batch, heads, seq_len, dim, ctx.causal
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
