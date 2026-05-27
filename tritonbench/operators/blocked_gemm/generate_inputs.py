"""Input builders for the blocked_gemm operator."""

from typing import List, Optional, Tuple

import torch


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _allocate(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> torch.Tensor:
    t = torch.randn(*shape, dtype=dtype, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def build_inputs(
    M: int,
    N: int,
    K: int,
    num_q_blocks: int,
    num_k_blocks: int,
    num_w_blocks: int,
    has_bias: bool,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = False,
) -> Tuple[
    List[List[torch.Tensor]],
    List[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
]:
    """Evenly tile (M, N, K) into (num_q, num_w, num_k) blocks.

    Shapes:
        A_list[i][k]: (M / num_q_blocks, K / num_k_blocks)
        W_list[j][k]: (N / num_w_blocks, K / num_k_blocks)
        bias_list[j]: (N / num_w_blocks,)  (when has_bias=True)
    """
    assert M % num_q_blocks == 0, (
        f"M={M} must be divisible by num_q_blocks={num_q_blocks}"
    )
    assert N % num_w_blocks == 0, (
        f"N={N} must be divisible by num_w_blocks={num_w_blocks}"
    )
    assert K % num_k_blocks == 0, (
        f"K={K} must be divisible by num_k_blocks={num_k_blocks}"
    )

    m_per_block = M // num_q_blocks
    n_per_block = N // num_w_blocks
    k_per_block = K // num_k_blocks

    A_list: List[List[torch.Tensor]] = [
        [
            _allocate((m_per_block, k_per_block), dtype, device, requires_grad)
            for _ in range(num_k_blocks)
        ]
        for _ in range(num_q_blocks)
    ]
    W_list: List[List[torch.Tensor]] = [
        [
            _allocate((n_per_block, k_per_block), dtype, device, requires_grad)
            for _ in range(num_k_blocks)
        ]
        for _ in range(num_w_blocks)
    ]
    bias_list: Optional[List[torch.Tensor]] = None
    if has_bias:
        bias_list = [
            _allocate((n_per_block,), dtype, device, requires_grad)
            for _ in range(num_w_blocks)
        ]
    return A_list, W_list, bias_list


def compute_flops(
    A_list: List[List[torch.Tensor]],
    W_list: List[List[torch.Tensor]],
    bias_list: Optional[List[torch.Tensor]],
    mode: str = "fwd",
) -> float:
    """FLOPs estimate for blocked GEMM.

    Forward computes C[i][j] = sum_k A[i][k] @ W[j][k]^T which costs
    2 * M_i * N_j * K_k FLOPs per (i, j, k) cell; optional bias adds
    2 * M_i * N_j per (i, j). bwd ~ 2x fwd, fwd_bwd ~ 3x fwd.
    """
    assert mode in ("fwd", "bwd", "fwd_bwd")
    num_a = len(A_list)
    num_b = len(W_list)
    num_k = len(A_list[0]) if A_list else 0
    fwd = 0.0
    for i in range(num_a):
        m_i = A_list[i][0].shape[0]
        for j in range(num_b):
            n_j = W_list[j][0].shape[0]
            for k_blk in range(num_k):
                k_k = A_list[i][k_blk].shape[1]
                fwd += 2.0 * m_i * n_j * k_k
            if bias_list is not None:
                fwd += 2.0 * m_i * n_j

    if mode == "fwd":
        return fwd
    if mode == "bwd":
        return 2.0 * fwd
    return 3.0 * fwd
