"""
Layer Normalization with multi_cta=True annotation.

Uses tl.range(..., multi_cta=True) for loop partitioning across CTAs.
The C++ MultiCTAReduction pass handles the DSM exchange automatically.

When ctas_per_cga > (1,1,1), the compiler partitions loop iterations across
CTAs in the cluster. The cross-CTA reduction (to combine partial sums) is
handled by the compiler pass via DSM exchange + barrier synchronization.

For correctness testing with num_ctas=1, the pass is a no-op.
"""

import inspect

import torch
import triton
import triton.language as tl

try:
    # ctas_per_cga and multi_cta are meta-triton features not available in OSS triton
    triton.Config({}, ctas_per_cga=(1, 1, 1))
    HAS_MULTI_CTA = "multi_cta" in inspect.signature(tl.range).parameters
except Exception:
    HAS_MULTI_CTA = False


if HAS_MULTI_CTA:
    # Include all possible BLOCK_SIZE values in configs directly.
    # prune_configs filters out invalid combinations — no deepcopy needed.
    configs = [
        triton.Config(
            {"BLOCK_SIZE": bs, "NUM_CTAS": nc},
            num_warps=nw,
            ctas_per_cga=(1, nc, 1),
        )
        for nc in [1, 2, 4, 8]
        for nw in [4, 8, 16]
        for bs in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    ]

    def prune_configs(configs, named_args, **kwargs):
        """Prune invalid configs based on N and NUM_CTAS.

        Key constraints:
        1. chunk = N // NUM_CTAS must be divisible by BLOCK_SIZE
        2. N must be exactly divisible by NUM_CTAS (no remainder elements)
        3. For multi-CTA (nc > 1), N must be exactly equal to nc * chunk
           to avoid the last CTA reading beyond the allocation
        """
        N = named_args["N"]

        pruned = []
        for cfg in configs:
            nc = cfg.kwargs["NUM_CTAS"]
            bs = cfg.kwargs["BLOCK_SIZE"]
            if N < nc:
                continue
            # N must be exactly divisible by NUM_CTAS
            if N % nc != 0:
                continue
            chunk = N // nc
            # BLOCK_SIZE must fit in the chunk and divide it evenly
            if bs > chunk or chunk % bs != 0:
                continue
            pruned.append(cfg)
        return pruned

    @triton.autotune(
        configs=configs,
        key=["N"],
        prune_configs_by={
            "perf_model": None,
            "early_config_prune": prune_configs,
        },
    )
    @triton.jit
    def _layer_norm_fwd_multi_cta(
        X,
        Y,
        W,
        B,
        Mean,
        Rstd,
        stride,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
        NUM_CTAS: tl.constexpr,
    ):
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride

        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in tl.range(0, N, BLOCK_SIZE, multi_cta=True):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=0) / N

        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in tl.range(0, N, BLOCK_SIZE, multi_cta=True):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
            x = tl.where(cols < N, x - mean, 0.0)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)

        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)

        for off in tl.range(0, N, BLOCK_SIZE, multi_cta=True):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask)
            b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            tl.store(Y + cols, y, mask=mask)

    class LayerNormMultiCTA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, normalized_shape, weight, bias, eps):
            y = torch.empty_like(x)
            x_arg = x.reshape(-1, x.shape[-1])
            M, N = x_arg.shape
            # Guard against i32 overflow: row * stride uses i32 arithmetic,
            # so (M-1)*N must fit in signed i32. This affects ALL Triton
            # kernels using i32 stride, not just multi-CTA.
            if (M - 1) * N > 2**31 - 1:
                raise RuntimeError(
                    f"multi_cta layer norm: (M-1)*N too large for i32 pointer "
                    f"arithmetic ({M - 1}*{N}={(M - 1) * N} > 2^31-1). "
                    f"Use smaller M or N."
                )
            mean = torch.empty((M,), dtype=torch.float32, device=x.device)
            rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
            grid = lambda meta: (M, meta["NUM_CTAS"])
            _layer_norm_fwd_multi_cta[grid](
                x_arg,
                y,
                weight,
                bias,
                mean,
                rstd,
                x_arg.stride(0),
                N,
                eps,
            )
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            ctx.eps = eps
            return y

        @staticmethod
        def backward(ctx, dy):
            from . import tutorial

            x, w, b, m, v = ctx.saved_tensors
            N = w.shape[0]
            GROUP_SIZE_M = 64
            if N <= 8192:
                GROUP_SIZE_M = 96
            if N <= 4096:
                GROUP_SIZE_M = 128
            if N <= 1024:
                GROUP_SIZE_M = 256
            locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
            _dw = torch.empty(
                (GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device
            )
            _db = torch.empty(
                (GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device
            )
            dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
            db = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
            dx = torch.empty_like(dy)
            x_arg = x.reshape(-1, x.shape[-1])
            M, N = x_arg.shape
            MAX_FUSED_SIZE = 65536 // x.element_size()
            BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            bwd_num_warps = min(max(BLOCK_SIZE_N // 256, 1), 8)
            tutorial._layer_norm_bwd_dx_fused[(M,)](
                dx,
                dy,
                _dw,
                _db,
                x,
                w,
                b,
                m,
                v,
                locks,
                x_arg.stride(0),
                N,
                ctx.eps,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                GROUP_SIZE_M=GROUP_SIZE_M,
                num_warps=bwd_num_warps,
            )
            grid = lambda meta: [triton.cdiv(N, meta["BLOCK_SIZE_N"])]
            tutorial._layer_norm_bwd_dwdb[grid](
                _dw,
                _db,
                dw,
                db,
                min(GROUP_SIZE_M, M),
                N,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=128,
                num_ctas=1,
            )
            return dx, None, dw, db, None

    layer_norm_multi_cta = LayerNormMultiCTA.apply
