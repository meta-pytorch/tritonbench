"""
TLX Multi-CTA Layer Normalization for benchmarking.

Copied from triton/third_party/tlx/tutorials/blackwell-multi-cta-layernorm_test.py
This uses Blackwell CTA clusters + DSM to split the N-dimension reduction
across multiple CTAs.
"""

import torch
import triton
import triton.language as tl

try:
    import triton.language.extra.tlx as tlx
    from torch._inductor.runtime.triton_compat import libdevice

    HAS_TLX = True
except (ImportError, ModuleNotFoundError):
    HAS_TLX = False


if HAS_TLX:

    @triton.jit
    def compute_multi_cta_sum(
        x,
        cta_cluster_rank,
        barrier,
        phase,
        BLOCK_SIZE_M: tl.constexpr,
        num_reduction_ctas: tl.constexpr,
    ):
        dtype_x = tlx.dtype_of(x)
        local_buff = tlx.local_alloc((BLOCK_SIZE_M, 1), dtype_x, num_reduction_ctas)
        local_partial_sum = tl.sum(x, axis=1, keep_dims=True)
        tlx.local_store(local_buff[cta_cluster_rank], local_partial_sum)
        for i in tl.static_range(num_reduction_ctas):
            if cta_cluster_rank != i:
                tlx.async_remote_shmem_store(
                    dst=local_buff[cta_cluster_rank],
                    src=local_partial_sum,
                    remote_cta_rank=i,
                    barrier=barrier,
                )
        tlx.barrier_wait(barrier, phase=phase)
        final_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=dtype_x)
        for i in tl.static_range(num_reduction_ctas):
            remote_local_buff_view = tlx.local_view(local_buff, i)
            final_sum += tlx.local_load(remote_local_buff_view)
        return final_sum

    kernel_configs_multi_cta = [
        triton.Config(
            {
                "BLOCK_SIZE_M": m,
                "BLOCK_SIZE_N": 8192,
                "num_reduction_ctas": ctas,
                "SHOULD_MASK_ROW": False,
                "SHOULD_MASK_COL": False,
            },
            num_warps=nw,
            ctas_per_cga=(1, ctas, 1),
        )
        for m in [1, 2]
        for nw in [4, 8, 16, 32]
        for ctas in [2, 4, 8]
    ]

    def prune_and_update_configs(configs, named_args, **kwargs):
        N = kwargs["N"]
        M = kwargs["M"]
        pruned = []
        for conf in configs:
            num_ctas = conf.kwargs.get("num_reduction_ctas")
            block_size_m = conf.kwargs.get("BLOCK_SIZE_M")
            blocksize_n = triton.next_power_of_2(N // num_ctas)
            if triton.cdiv(N, blocksize_n) != num_ctas:
                continue
            element_size = 2
            num_threads = conf.num_warps * 32
            bytes_per_thread = (
                block_size_m * blocksize_n * element_size
            ) // num_threads
            if bytes_per_thread < 4:
                continue
            conf.kwargs["BLOCK_SIZE_N"] = blocksize_n
            conf.kwargs["SHOULD_MASK_ROW"] = M % block_size_m != 0
            conf.kwargs["SHOULD_MASK_COL"] = N % blocksize_n != 0
            pruned.append(conf)
        return pruned

    @triton.autotune(
        configs=kernel_configs_multi_cta,
        prune_configs_by={"early_config_prune": prune_and_update_configs},
        key=["M", "N"],
    )
    @triton.jit
    def kernel_layernorm_multi_cta(
        X,
        Y,
        W,
        B,
        Mean_out,
        Rstd_out,
        row_stride,
        M,
        N,
        eps,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        num_reduction_ctas: tl.constexpr,
        SHOULD_MASK_ROW: tl.constexpr,
        SHOULD_MASK_COL: tl.constexpr,
    ):
        cta_cluster_rank = tlx.cluster_cta_rank()
        COMPUTE_DTYPE = tl.float32
        x_buffer = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), X.dtype.element_ty, 1)
        x_buf = tlx.local_view(x_buffer, 0)
        barriers = tlx.alloc_barriers(num_barriers=2)
        cross_cta_reduction_expected_bytes: tl.constexpr = (
            BLOCK_SIZE_M * tlx.size_of(COMPUTE_DTYPE) * (num_reduction_ctas - 1)
        )
        tlx.barrier_expect_bytes(barriers[0], size=cross_cta_reduction_expected_bytes)
        tlx.barrier_expect_bytes(barriers[1], size=cross_cta_reduction_expected_bytes)
        tlx.cluster_barrier()
        row_offsets = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        col_offsets = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        read_write_offsets = (row_offsets[:, None] * row_stride) + col_offsets[None, :]
        x_ptrs = X + read_write_offsets
        y_ptrs = Y + read_write_offsets
        w_ptrs = W + col_offsets
        b_ptrs = B + col_offsets
        mask_row = None
        if SHOULD_MASK_ROW:
            mask_row = row_offsets < M
        else:
            if SHOULD_MASK_COL:
                mask_row = tl.full([BLOCK_SIZE_M], True, dtype=tl.int1)
        mask_col = None
        if SHOULD_MASK_COL:
            mask_col = col_offsets < N
        else:
            if SHOULD_MASK_ROW:
                mask_col = tl.full([BLOCK_SIZE_N], True, dtype=tl.int1)
        read_write_mask = None
        SHOULD_MASK: tl.constexpr = SHOULD_MASK_ROW or SHOULD_MASK_COL
        if SHOULD_MASK:
            read_write_mask = mask_row[:, None] & mask_col[None, :]
        other = 0.0 if SHOULD_MASK else None
        token_x = tlx.async_load(x_ptrs, x_buf, mask=read_write_mask, other=other)
        tlx.async_load_commit_group([token_x])
        tlx.async_load_wait_group(0)
        x = tlx.local_load(x_buf).to(COMPUTE_DTYPE)
        multi_cta_sum = compute_multi_cta_sum(
            x,
            cta_cluster_rank,
            barriers[0],
            phase=0,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            num_reduction_ctas=num_reduction_ctas,
        )
        mean = multi_cta_sum / N
        if SHOULD_MASK:
            x_minus_mean = tl.where(read_write_mask, x - mean, 0.0)
        else:
            x_minus_mean = x - mean
        x_minus_mean_sq = x_minus_mean * x_minus_mean
        multi_cta_sum_sq = compute_multi_cta_sum(
            x_minus_mean_sq,
            cta_cluster_rank,
            barriers[1],
            phase=0,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            num_reduction_ctas=num_reduction_ctas,
        )
        var = multi_cta_sum_sq / N
        rstd = libdevice.rsqrt(var + eps)
        mean_1d = tl.reshape(mean, (BLOCK_SIZE_M,))
        tl.store(Mean_out + row_offsets, mean_1d, mask=mask_row)
        rstd_1d = tl.reshape(rstd, (BLOCK_SIZE_M,))
        w = tl.load(w_ptrs, mask=mask_col).to(COMPUTE_DTYPE)
        b = tl.load(b_ptrs, mask=mask_col).to(COMPUTE_DTYPE)
        tl.store(Rstd_out + row_offsets, rstd_1d, mask=mask_row)
        x = tlx.local_load(x_buffer[0]).to(COMPUTE_DTYPE)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        y = tl.cast(y, y_ptrs.dtype.element_ty)
        tl.store(y_ptrs, y, mask=read_write_mask)

    def multi_cta_layernorm(x, weight, bias, eps=1e-5):
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        mean = torch.empty([m], dtype=torch.float32, device=x.device)
        rstd = torch.empty([m], dtype=torch.float32, device=x.device)

        def grid_2d(meta):
            return (
                triton.cdiv(m, meta["BLOCK_SIZE_M"]),
                triton.cdiv(n, meta["BLOCK_SIZE_N"]),
            )

        kernel_layernorm_multi_cta[grid_2d](
            X=x,
            Y=out,
            W=weight,
            B=bias,
            Mean_out=mean,
            Rstd_out=rstd,
            row_stride=x.stride(0),
            M=m,
            N=n,
            eps=eps,
        )
        out = out.view(original_shape)
        return out, mean, rstd


class LayerNormTLXMultiCTA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        if not HAS_TLX:
            raise RuntimeError("TLX not available")
        y, mean, rstd = multi_cta_layernorm(x, weight, bias, eps)
        return y.clone()

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("TLX multi-CTA layernorm backward not implemented")


layer_norm_tlx_multi_cta = LayerNormTLXMultiCTA.apply
