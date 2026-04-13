# Original source:
# https://github.com/tile-ai/tilelang/blob/main/examples/norm/test_rms_norm.py
import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()


def rms_norm_splitk(M, N, blk_m, blk_k):
    dtype = "float"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, blk_k), dtype)
            A_local = T.alloc_fragment((blk_m, blk_k), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            num_k_step = T.ceildiv(N, blk_k)
            T.clear(A_local)
            for k in range(num_k_step):
                T.copy(A[bx * blk_m, k * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_local[i, j] += A_shared[i, j] * A_shared[i, j]
            T.reduce_sum(A_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N) + 1e-12

            for k in range(num_k_step):
                # reverse, better cache hit rate
                T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_shared[i, j] *= A_powsum[i]
                T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return main


def rms_norm(M, N, blk_m, dtype, variance_epsilon=1e-12):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), dtype)
            A_local = T.alloc_fragment((blk_m, N), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_shared)
            T.copy(A_shared, A_local)
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
            T.reduce_sum(A_pow_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N) + variance_epsilon
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i]
            T.copy(A_local, B[bx * blk_m : (bx + 1) * blk_m, :])

    return main


TILELANG_DTYPE_MAP = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float",
}


class TileLangRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        TileLangRMSNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        M, N = hidden_states.size()
        dtype = TILELANG_DTYPE_MAP[hidden_states.dtype]
        blk_m = 1
        blk_k = 512

        kernel = rms_norm(M, N, blk_m, dtype, self.variance_epsilon)
        jit_kernel = tilelang.compile(
            kernel,
            out_idx=[-1],
            target="cuda",
            pass_configs={
                tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            },
        )
        return lambda: jit_kernel(hidden_states)
