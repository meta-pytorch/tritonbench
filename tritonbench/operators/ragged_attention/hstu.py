import torch

from tritonbench.utils.env_utils import is_fbcode

from tritonbench.utils.path_utils import add_path, SUBMODULE_PATH

if is_fbcode():
    # Internal Imports
    from generative_recommenders.common import (
        apply_sampling,
        generate_sparse_seq_len,
        set_use_runtime_max_seq_len,
    )
    from generative_recommenders.ops.triton.triton_hstu_attention import triton_hstu_mha
    from hammer.ops.triton.triton_ragged_hstu_attention import (
        triton_ragged_attention_fwd,
    )

    HAS_HAMMER = True
else:
    # OSS Import
    with add_path(str(SUBMODULE_PATH.joinpath("generative-recommenders"))):
        from generative_recommenders.common import (
            apply_sampling,
            generate_sparse_seq_len,
            set_use_runtime_max_seq_len,
        )
        from generative_recommenders.ops.triton.triton_hstu_attention import (
            triton_hstu_mha,
        )

        HAS_HAMMER = False
        triton_ragged_attention_fwd = None

from typing import Tuple

triton_hstu_mha = triton_hstu_mha
triton_ragged_attention_fwd = triton_ragged_attention_fwd
HAS_HAMMER = HAS_HAMMER

# Always autotune based on the actual max_seq_len
set_use_runtime_max_seq_len(True)


def get_test_inputs(
    batch_size,
    heads,
    seq_len,
    attn_dim,
    hidden_dim,
    seq_sparsity,
    has_delta_q,
    delta_size,
    target_size,
    max_attn_len,
    dtype,
    requires_grad,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    lengths = generate_sparse_seq_len(
        size=batch_size,
        max_seq_len=seq_len,
        sparsity=seq_sparsity,
        device=torch.device("cuda"),
    )
    lengths = apply_sampling(lengths, alpha=2.0, max_seq_len=seq_len)
    if has_delta_q:
        lengths = lengths + delta_size
        num_targets = torch.ones_like(lengths) * delta_size
        seq_len = seq_len + delta_size
    else:
        delta_size = 0
        num_targets = None
        if target_size != 0:
            num_targets = torch.randint(
                target_size,
                target_size + 1,
                (batch_size,),
                device=lengths.device,
                dtype=lengths.dtype,
            )
            num_targets = torch.where(num_targets > lengths, lengths, num_targets).to(
                torch.int32
            )
    max_attn_len = max_attn_len if max_attn_len < seq_len else seq_len
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    L = int(seq_offsets[-1].item())
    x = torch.empty(
        (L, heads, attn_dim * 2 + hidden_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)
    delta_q = torch.empty(
        (batch_size * delta_size, heads, attn_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.1, 0.1)
    delta_x_offsets = torch.arange(0, delta_size, device=torch.device("cuda"))
    delta_x_offsets = (seq_offsets[1:] - delta_size).view(
        batch_size, 1
    ) + delta_x_offsets.view(1, delta_size)
    delta_x_offsets = delta_x_offsets.view(-1)
    q = q.requires_grad_(requires_grad)
    k = k.requires_grad_(requires_grad)
    v = v.requires_grad_(requires_grad)
    return q, k, v, seq_offsets, num_targets, seq_len
