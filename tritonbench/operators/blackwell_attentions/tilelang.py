import functools

import tilelang
import tilelang.language as T
import torch

PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
}


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
                        T.copy(
                            Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared
                        )

                    if stage_id == 0:
                        T.copy(
                            K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared_0
                        )
                    else:
                        T.copy(
                            K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared_1
                        )
                    T.mbarrier_arrive(mbar_dma1_full[stage_id])

                    T.mbarrier_wait_parity(mbar_dma2_empty[stage_id], parity_inv)
                    if stage_id == 0:
                        T.copy(
                            V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared_0
                        )
                    else:
                        T.copy(
                            V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared_1
                        )
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
                        S_reg[i, j] = T.exp2(
                            S_reg[i, j] * scale - scores_max[i] * scale
                        )

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


def tilelang_blackwell_attention(q, k, v, causal):
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise NotImplementedError("TileLang Blackwell MHA only supports bf16")
    if q.shape != k.shape or q.shape != v.shape:
        raise NotImplementedError("TileLang Blackwell MHA only supports MHA shapes")
    batch, seq_len, heads, dim = q.shape
    kernel = _get_wasp_fwd_kernel(batch, heads, seq_len, dim, causal)
    return kernel(q, k, v)
