# Source: https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py
import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

TILELANG_DTYPE_MAP = {
    torch.float8_e4m3fn: T.float8_e4m3fn,
}


@tilelang.jit
def deepgemm_fp8_2xacc(
    A,
    B,
    scales_a,
    scales_b,
    block_M,
    block_N,
    block_K,
    group_size,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
):
    M, N, K, M_pad, K_groups, N_groups = T.const("M, N, K, M_pad, K_groups, N_groups")

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[N, K], in_dtype]
    scales_a: T.Tensor[[M_pad, K_groups], T.float32]
    scales_b: T.Tensor[[N_groups, K_groups], T.float32]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
        bx,
        by,
    ):
        A_shared = T.alloc_shared((block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((block_N, block_K), in_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)
        Scale_C_shared = T.alloc_shared((block_M), T.float32)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.use_swizzle(panel_size=10)

        T.clear(C_local)
        T.clear(C_local_accum)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            Scale_B = scales_b[bx * block_N // group_size, k]
            for i in T.Parallel(block_M):
                Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
            T.clear(C_local)
        T.copy(C_local_accum, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])

    return C


def tilelang_deepgemm_fp8_func(xq, wq, x_scale, w_scale):
    if xq.dtype not in TILELANG_DTYPE_MAP:
        raise NotImplementedError("TileLang DeepGEMM only supports fp8 e4m3 inputs")
    if xq.dtype != wq.dtype:
        raise NotImplementedError("TileLang DeepGEMM requires matching dtypes")
    if xq.dim() != 2 or wq.dim() != 2:
        raise NotImplementedError("TileLang DeepGEMM only supports 2D inputs")

    block_M, block_N, block_K = 128, 128, 128
    group_size = 128
    num_stages = 4

    M = xq.shape[0]
    xq = xq.contiguous()
    wq = wq.contiguous()
    m_pad = (M + block_M - 1) // block_M * block_M
    x_scale = (
        x_scale.to(torch.float32)
        .repeat_interleave(group_size, dim=0)[:m_pad]
        .contiguous()
    )
    w_scale = w_scale.to(torch.float32).contiguous()

    return lambda: deepgemm_fp8_2xacc(
        xq,
        wq,
        x_scale,
        w_scale,
        block_M,
        block_N,
        block_K,
        group_size,
        TILELANG_DTYPE_MAP[xq.dtype],
        T.bfloat16,
        T.float32,
        num_stages,
    )
