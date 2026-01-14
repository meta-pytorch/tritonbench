import math

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_fused_no_bias(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"M_INCREMENT": M_INCREMENT},
            num_warps=w,
        )
        for M_INCREMENT in [1, 2, 4, 8, 16]
        for w in [2, 4, 8]
    ],
    key=["N"],
)
@triton.jit
def _layer_norm_bwd_dx_fused(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of weights gradient
    X,  # pointer to the input
    W,  # pointer to the weights
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    M,  # number of rows in X
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    M_INCREMENT: tl.constexpr,
    N_POW_2: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    pid = tl.program_id(0)
    start_row = pid * BLOCK_SIZE_M
    grad_w = tl.full([BLOCK_SIZE_N], 0, tl.float32)
    cols = tl.arange(0, BLOCK_SIZE_N)
    if N_POW_2:
        col_mask = None
    else:
        col_mask = cols < N

    w = tl.load(W + cols, mask=col_mask).to(tl.float32)[None, :]

    for cur_row in tl.range(0, BLOCK_SIZE_M, M_INCREMENT):
        rows = start_row + cur_row + tl.arange(0, M_INCREMENT)
        row_indices = rows * stride
        row_mask = rows < M

        mean = tl.load(Mean + rows, mask=row_mask).to(tl.float32)[:, None]
        rstd = tl.load(Rstd + rows, mask=row_mask).to(tl.float32)[:, None]

        if N_POW_2:
            index_mask = row_mask[:, None]
        else:
            index_mask = row_mask[:, None] & col_mask[None, :]
        indices = row_indices[:, None] + cols[None, :]

        # Load data to SRAM
        x = tl.load(X + indices, mask=index_mask, other=0)
        x_dtype = x.dtype
        x_f32 = x.to(tl.float32)
        dy = tl.load(DY + indices, mask=index_mask, other=0).to(tl.float32)

        # Compute dx
        xhat = (x_f32 - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(xhat * wdy, axis=1) / N
        c2 = tl.sum(wdy, axis=1) / N

        dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd

        # Write dx
        tl.store(DX + indices, dx.to(x_dtype), mask=index_mask)

        dw = dy * xhat
        partial_dw = tl.sum(dw, axis=0)
        grad_w += partial_dw

    tl.store(DW + pid * N + cols, grad_w, mask=col_mask)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused_no_bias[(M,)](  #
            x_arg,
            y,
            weight,
            mean,
            rstd,  #
            x_arg.stride(0),
            N,
            eps,  #
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, m, v = ctx.saved_tensors
        x_arg = x.reshape(-1, x.shape[-1])
        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]
        # allocate output
        dw = torch.empty((N,), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)

        M, N = x_arg.shape
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        BLOCK_SIZE_M = min(2048, triton.next_power_of_2(M // (8 * NUM_SMS)))
        PARTIAL_SIZE = math.ceil(M / BLOCK_SIZE_M)

        # Columnwise stride for reducing partial sums at end, contiguous loads
        _dw = torch.empty((PARTIAL_SIZE, N), dtype=torch.float32, device=w.device)

        MAX_FUSED_SIZE = 65536 // x.element_size()
        assert ctx.BLOCK_SIZE <= MAX_FUSED_SIZE, (
            "This layer norm doesn't support feature dim >= 64KB."
        )
        _layer_norm_bwd_dx_fused[(PARTIAL_SIZE,)](  #
            dx,
            dy,
            _dw,
            x_arg,
            w,
            m,
            v,
            x_arg.stride(0),
            N,
            M,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            N_POW_2=(N % ctx.BLOCK_SIZE == 0),
        )

        dw = torch.sum(_dw, dim=0)

        return dx, None, dw, None, None


layer_norm = LayerNorm.apply
