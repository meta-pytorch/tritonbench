"""Input builders for the blocked_attention operator."""

from typing import List, Tuple

import torch

from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.python_utils import try_import


HAS_HAMMER_V3 = False
with try_import("HAS_HAMMER_V3"):
    if not is_fbcode():
        raise ImportError("hammer.v3 is fbcode-only")
    from generative_recommenders.common import apply_sampling
    from hammer.v3.ops.pytorch.pt_attention import MaskType


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def generate_sparse_lengths(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device,
    sampling_alpha: float = 1.0,
) -> torch.Tensor:
    """Generate per-batch sequence lengths with a target average sparsity."""
    if sparsity == 0.0:
        lengths = torch.zeros(size, device=device, dtype=torch.int)
    elif sparsity == 1.0:
        lengths = torch.full((size,), max_seq_len, device=device, dtype=torch.int)
    elif sparsity >= 0.5:
        min_seq_len = int((2 * sparsity - 1.0) * max_seq_len)
        lengths = torch.randint(
            low=min_seq_len,
            high=max_seq_len,
            size=(size,),
            device=device,
            dtype=torch.int,
        )
    else:
        lengths = torch.randint(
            low=0,
            high=int(2 * sparsity * max_seq_len),
            size=(size,),
            device=device,
            dtype=torch.int,
        )

    if sampling_alpha != 1.0 and HAS_HAMMER_V3:
        lengths = apply_sampling(lengths, sampling_alpha, max_seq_len=max_seq_len)
    return lengths


def make_jagged(
    lengths: torch.Tensor,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate a [sum(lengths), heads, dim] jagged tensor and its offsets."""
    batch_size = lengths.shape[0]
    offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lengths, dim=0)

    total_len = int(offsets[-1].item())
    tensor = torch.empty((total_len, heads, dim), dtype=dtype, device=device)
    tensor.uniform_(-0.1, 0.1)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor, offsets


def split_jagged(
    tensor: torch.Tensor,
    offsets: torch.Tensor,
    split_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a jagged tensor in half per-batch at split_points[i].

    Returns (left_tensor, left_offsets, right_tensor, right_offsets).
    """
    batch_size = offsets.shape[0] - 1
    device = tensor.device

    left_lengths = split_points
    right_lengths = (offsets[1:] - offsets[:-1]) - split_points

    left_offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
    left_offsets[1:] = torch.cumsum(left_lengths, dim=0)
    right_offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
    right_offsets[1:] = torch.cumsum(right_lengths, dim=0)

    left_total = int(left_offsets[-1].item())
    right_total = int(right_offsets[-1].item())
    heads, dim = tensor.shape[1], tensor.shape[2]
    left = torch.empty((left_total, heads, dim), dtype=tensor.dtype, device=device)
    right = torch.empty((right_total, heads, dim), dtype=tensor.dtype, device=device)

    for i in range(batch_size):
        seq_start = int(offsets[i].item())
        left_start = int(left_offsets[i].item())
        right_start = int(right_offsets[i].item())
        left_len = int(left_lengths[i].item())
        right_len = int(right_lengths[i].item())

        if left_len > 0:
            left[left_start : left_start + left_len] = tensor[
                seq_start : seq_start + left_len
            ]
        if right_len > 0:
            right[right_start : right_start + right_len] = tensor[
                seq_start + left_len : seq_start + left_len + right_len
            ]

    return left, left_offsets, right, right_offsets


def _make_qkv(
    lengths: torch.Tensor,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Allocate Q, K, V jagged tensors sharing the same per-batch lengths."""
    q, offsets = make_jagged(lengths, heads, attn_dim, dtype, device)
    k, _ = make_jagged(lengths, heads, attn_dim, dtype, device)
    v, _ = make_jagged(lengths, heads, hidden_dim, dtype, device)
    return q, k, v, offsets


def _attn_scales(
    offsets_list: List[torch.Tensor], device: torch.device
) -> List[torch.Tensor]:
    return [
        torch.ones((int(off[-1].item()),), dtype=torch.float32, device=device)
        for off in offsets_list
    ]


def _mask_from_str(s: str):
    if not HAS_HAMMER_V3:
        return None
    return {
        "causal": MaskType.CAUSAL,
        "all": MaskType.ALL,
        "local": MaskType.LOCAL,
        "diagonal": MaskType.DIAGONAL,
    }.get(s.lower(), MaskType.CAUSAL)


def _set_requires_grad(tensors: List[torch.Tensor]) -> None:
    for t in tensors:
        t.requires_grad_(True)


def _build_one_block(
    lengths, heads, attn_dim, hidden_dim, dtype, device, mask_type, requires_grad
):
    q, offsets = make_jagged(lengths, heads, attn_dim, dtype, device, requires_grad)
    k, _ = make_jagged(lengths, heads, attn_dim, dtype, device, requires_grad)
    v, _ = make_jagged(lengths, heads, hidden_dim, dtype, device, requires_grad)
    return (
        [q],
        [k],
        [v],
        [offsets],
        [offsets],
        _attn_scales([offsets], device),
        [[_mask_from_str(mask_type)]],
    )


def _build_context_target(
    lengths, heads, attn_dim, hidden_dim, dtype, device, target_size, requires_grad
):
    batch_size = int(lengths.shape[0])
    num_targets = torch.randint(
        1, target_size + 1, (batch_size,), device=device, dtype=lengths.dtype
    )
    num_targets = torch.where(num_targets > lengths, lengths, num_targets)
    context_lengths = lengths - num_targets

    q_full, k_full, v_full, q_offsets_full = _make_qkv(
        lengths, heads, attn_dim, hidden_dim, dtype, device
    )

    q_ctx, q_ctx_off, q_tgt, q_tgt_off = split_jagged(
        q_full, q_offsets_full, context_lengths
    )
    k_ctx, k_ctx_off, k_tgt, k_tgt_off = split_jagged(
        k_full, q_offsets_full, context_lengths
    )
    # V offsets mirror Q's, no need to keep them.
    v_ctx, _, v_tgt, _ = split_jagged(v_full, q_offsets_full, context_lengths)

    if requires_grad:
        _set_requires_grad([q_ctx, q_tgt, k_ctx, k_tgt, v_ctx, v_tgt])

    mask_matrix = [
        [MaskType.CAUSAL, MaskType.NULL],
        [MaskType.ALL, MaskType.DIAGONAL],
    ]
    return (
        [q_ctx, q_tgt],
        [k_ctx, k_tgt],
        [v_ctx, v_tgt],
        [q_ctx_off, q_tgt_off],
        [k_ctx_off, k_tgt_off],
        _attn_scales([q_ctx_off, q_tgt_off], device),
        mask_matrix,
    )


def _build_semi_local(
    lengths,
    heads,
    attn_dim,
    hidden_dim,
    dtype,
    device,
    full_attn_size,
    requires_grad,
):
    batch_size = int(lengths.shape[0])
    num_full = torch.full(
        (batch_size,), full_attn_size, device=device, dtype=lengths.dtype
    )
    num_full = torch.where(num_full > lengths, lengths, num_full)
    sliding_lengths = lengths - num_full

    q_full, k_full, v_full, q_offsets_full = _make_qkv(
        lengths, heads, attn_dim, hidden_dim, dtype, device
    )

    q_loc, q_loc_off, q_fb, q_fb_off = split_jagged(
        q_full, q_offsets_full, sliding_lengths
    )
    k_loc, k_loc_off, k_fb, k_fb_off = split_jagged(
        k_full, q_offsets_full, sliding_lengths
    )
    v_loc, _, v_fb, _ = split_jagged(v_full, q_offsets_full, sliding_lengths)

    if requires_grad:
        _set_requires_grad([q_loc, q_fb, k_loc, k_fb, v_loc, v_fb])

    mask_matrix = [
        [MaskType.LOCAL, MaskType.NULL],
        [MaskType.ALL, MaskType.CAUSAL],
    ]
    return (
        [q_loc, q_fb],
        [k_loc, k_fb],
        [v_loc, v_fb],
        [q_loc_off, q_fb_off],
        [k_loc_off, k_fb_off],
        _attn_scales([q_loc_off, q_fb_off], device),
        mask_matrix,
    )


def _build_three_block(
    lengths,
    heads,
    attn_dim,
    hidden_dim,
    dtype,
    device,
    target_size,
    full_attn_size,
    requires_grad,
):
    batch_size = int(lengths.shape[0])
    num_targets = torch.randint(
        1, target_size + 1, (batch_size,), device=device, dtype=lengths.dtype
    )
    num_targets = torch.where(num_targets > lengths, lengths, num_targets)

    num_full = torch.full(
        (batch_size,), full_attn_size, device=device, dtype=lengths.dtype
    )
    remaining = lengths - num_targets
    num_full = torch.where(num_full > remaining, remaining, num_full)
    sliding_lengths = lengths - num_targets - num_full

    q_full, k_full, v_full, q_offsets_full = _make_qkv(
        lengths, heads, attn_dim, hidden_dim, dtype, device
    )

    q_loc, q_loc_off, q_rest, q_rest_off = split_jagged(
        q_full, q_offsets_full, sliding_lengths
    )
    k_loc, k_loc_off, k_rest, k_rest_off = split_jagged(
        k_full, q_offsets_full, sliding_lengths
    )
    v_loc, _, v_rest, _ = split_jagged(v_full, q_offsets_full, sliding_lengths)

    q_fb, q_fb_off, q_tgt, q_tgt_off = split_jagged(q_rest, q_rest_off, num_full)
    k_fb, k_fb_off, k_tgt, k_tgt_off = split_jagged(k_rest, k_rest_off, num_full)
    v_fb, _, v_tgt, _ = split_jagged(v_rest, k_rest_off, num_full)

    if requires_grad:
        _set_requires_grad(
            [q_loc, q_fb, q_tgt, k_loc, k_fb, k_tgt, v_loc, v_fb, v_tgt]
        )

    mask_matrix = [
        [MaskType.LOCAL, MaskType.NULL, MaskType.NULL],
        [MaskType.ALL, MaskType.CAUSAL, MaskType.NULL],
        [MaskType.ALL, MaskType.ALL, MaskType.DIAGONAL],
    ]
    return (
        [q_loc, q_fb, q_tgt],
        [k_loc, k_fb, k_tgt],
        [v_loc, v_fb, v_tgt],
        [q_loc_off, q_fb_off, q_tgt_off],
        [k_loc_off, k_fb_off, k_tgt_off],
        _attn_scales([q_loc_off, q_fb_off, q_tgt_off], device),
        mask_matrix,
    )


def build_inputs(
    batch_size: int,
    heads: int,
    max_seq_len: int,
    attn_dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    sparsity: float,
    target_size: int,
    full_attn_size: int,
    sampling_alpha: float,
    mask_type: str,
    requires_grad: bool,
):
    """Dispatch to one of the 4 HSTU-style block layouts based on flags.

    Returns
        (q_list, k_list, v_list, q_offsets_list, kv_offsets_list,
         attn_scale_list, mask_matrix)

    Scenario selection:
        full_attn_size == 0 and target_size == 0  -> 1-block
        full_attn_size == 0 and target_size  > 0  -> 2-block context+target
        full_attn_size  > 0 and target_size == 0  -> 2-block semi-local
        full_attn_size  > 0 and target_size  > 0  -> 3-block HSTU full
    """
    if not HAS_HAMMER_V3:
        raise RuntimeError(
            "blocked_attention requires hammer.v3 (fbcode-only build)"
        )

    lengths = generate_sparse_lengths(
        size=batch_size,
        max_seq_len=max_seq_len,
        sparsity=sparsity,
        device=device,
        sampling_alpha=sampling_alpha,
    )

    if full_attn_size > 0 and target_size > 0:
        return _build_three_block(
            lengths,
            heads,
            attn_dim,
            hidden_dim,
            dtype,
            device,
            target_size,
            full_attn_size,
            requires_grad,
        )
    if full_attn_size > 0:
        return _build_semi_local(
            lengths,
            heads,
            attn_dim,
            hidden_dim,
            dtype,
            device,
            full_attn_size,
            requires_grad,
        )
    if target_size > 0:
        return _build_context_target(
            lengths,
            heads,
            attn_dim,
            hidden_dim,
            dtype,
            device,
            target_size,
            requires_grad,
        )
    return _build_one_block(
        lengths,
        heads,
        attn_dim,
        hidden_dim,
        dtype,
        device,
        mask_type,
        requires_grad,
    )


def block_flops(
    batch_size: int,
    attn_dim: int,
    hidden_dim: int,
    nheads: int,
    q_offsets_list: List[torch.Tensor],
    kv_offsets_list: List[torch.Tensor],
    mask_matrix,
    max_attn_len: int,
    mode: str = "fwd",
) -> float:
    """FLOPS estimator walking per-mask-cell.

    Per-cell elements:
        CAUSAL   : q_len * k_len / 2
        ALL      : q_len * k_len
        LOCAL    : q_len * min(max_attn_len, k_len)  (or k_len/2 if max_attn_len=0)
        DIAGONAL : q_len
        NULL     : skipped

    Each cell contributes 2 * nheads * (attn_dim + hidden_dim) * elements
    (the two GEMMs QK^T and (QK^T)V). bwd is 3x QK^T + 2x out; fwd_bwd 4x + 3x.
    """
    assert mode in ("fwd", "bwd", "fwd_bwd")
    qk_flops = 0.0
    out_flops = 0.0
    num_blocks = len(q_offsets_list)
    for b in range(batch_size):
        for r in range(num_blocks):
            q_len = int(
                (q_offsets_list[r][b + 1] - q_offsets_list[r][b]).item()
            )
            if q_len == 0:
                continue
            for c in range(num_blocks):
                mask = mask_matrix[r][c]
                if mask == MaskType.NULL:
                    continue
                k_len = int(
                    (kv_offsets_list[c][b + 1] - kv_offsets_list[c][b]).item()
                )
                if k_len == 0:
                    continue
                if mask == MaskType.CAUSAL:
                    elements = q_len * k_len / 2.0
                elif mask == MaskType.ALL:
                    elements = q_len * k_len
                elif mask == MaskType.LOCAL:
                    if max_attn_len > 0:
                        elements = q_len * min(max_attn_len, k_len)
                    else:
                        elements = q_len * k_len / 2.0
                elif mask == MaskType.DIAGONAL:
                    elements = q_len
                else:
                    elements = q_len * k_len
                qk_flops += 2 * nheads * attn_dim * elements
                out_flops += 2 * nheads * hidden_dim * elements

    if mode == "fwd":
        return qk_flops + out_flops
    if mode == "bwd":
        return 3 * qk_flops + 2 * out_flops
    return 4 * qk_flops + 3 * out_flops
