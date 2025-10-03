# Original source: https://github.com/tile-ai/tilelang/blob/main/examples/gemm_sm100/gemm_tcgen5mma.py
import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    trans_A,
                    trans_B,
                    mbar=mbar,
                    wg_wait=-1,
                    clear_accum=k == 0,
                )
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


TILELANG_DTYPE_MAP = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float",
}


def tilelang_matmul_func(a, b):
    M, K = a.size()
    K, N = b.size()
    b_T = b.T.contiguous()
    block_M, block_N, block_K = 128, 256, 128
    trans_A, trans_B = False, True
    in_dtype = TILELANG_DTYPE_MAP[a.dtype]
    out_dtype = TILELANG_DTYPE_MAP[a.dtype]
    accum_dtype = "float"
    num_stages = 2
    threads = 256
    func = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        threads,
    )
    jit_kernel = tilelang.compile(
        func,
        out_idx=[2],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    return lambda: jit_kernel(a, b_T)
