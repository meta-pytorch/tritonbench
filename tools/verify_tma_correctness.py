"""Standalone correctness check: triton_hstu_mha enable_tma=True vs False.

Confirms that turning TMA on doesn't silently produce different (or zeroed)
output relative to the TMA-off path. Same inputs, same kernel source, only
the enable_tma kwarg differs.

Usage:
    cd ~/OpenSource/tritonbench
    TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 CUDA_VISIBLE_DEVICES=3 \\
        /home/mren/.conda/envs/metamain/bin/python tools/verify_tma_correctness.py
"""

import sys

import torch

# Wire up the OSS submodules the same way the operator does.
sys.path.insert(0, "submodules/generative-recommenders")
sys.path.insert(0, "submodules/hammer")

from generative_recommenders.common import apply_sampling, generate_sparse_seq_len
from generative_recommenders.ops.triton.triton_hstu_attention import triton_hstu_mha


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Matches the OSS sliding-window setup at seq_len=4096.
    batch_size = 1024
    heads = 2
    attn_dim = 128
    hidden_dim = 128
    max_seq_len = 4096
    max_attn_len = 32
    alpha = 1.0 / attn_dim
    sparsity = 1.0

    # Generate jagged sequence offsets.
    lengths = generate_sparse_seq_len(
        size=batch_size, max_seq_len=max_seq_len, sparsity=sparsity, device=device
    )
    seq_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    total_len = int(seq_offsets[-1].item())

    q = torch.randn(total_len, heads, attn_dim, dtype=dtype, device=device)
    k = torch.randn(total_len, heads, attn_dim, dtype=dtype, device=device)
    v = torch.randn(total_len, heads, hidden_dim, dtype=dtype, device=device)

    common_kwargs = dict(
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=None,
        max_attn_len=max_attn_len,
        contextual_seq_len=0,
        sort_by_length=True,
    )

    print(
        f"Inputs: batch={batch_size} heads={heads} max_seq_len={max_seq_len} "
        f"total_len={total_len} max_attn_len={max_attn_len} dtype={dtype}"
    )
    print()

    print("Running triton_hstu_mha(enable_tma=False) ...")
    out_no_tma = triton_hstu_mha(N=max_seq_len, **common_kwargs, enable_tma=False)
    torch.cuda.synchronize()
    print(
        f"  out_no_tma: shape={tuple(out_no_tma.shape)} "
        f"dtype={out_no_tma.dtype} abs_max={out_no_tma.abs().max().item():.4f}"
    )

    print("Running triton_hstu_mha(enable_tma=True) ...")
    out_tma = triton_hstu_mha(N=max_seq_len, **common_kwargs, enable_tma=True)
    torch.cuda.synchronize()
    print(
        f"  out_tma:    shape={tuple(out_tma.shape)} "
        f"dtype={out_tma.dtype} abs_max={out_tma.abs().max().item():.4f}"
    )
    print()

    # bf16 numerics; tolerance should be loose.
    abs_diff = (out_tma.float() - out_no_tma.float()).abs()
    print(f"Max abs diff: {abs_diff.max().item():.6f}")
    print(f"Mean abs diff: {abs_diff.mean().item():.6f}")
    print(f"  abs_max(no_tma) = {out_no_tma.abs().max().item():.4f}")
    print(f"  abs_max(tma)    = {out_tma.abs().max().item():.4f}")

    # tritonbench's default tolerance for bf16 attention accuracy is rtol=1e-2,
    # atol=1e-2; use the same here.
    ok = torch.allclose(
        out_tma.float(), out_no_tma.float(), rtol=1e-2, atol=1e-2
    )
    print()
    if ok:
        print("PASS: enable_tma=True and enable_tma=False outputs agree within bf16 tolerance.")
    else:
        # Diagnose: where do they disagree?
        bad = ~torch.isclose(out_tma.float(), out_no_tma.float(), rtol=1e-2, atol=1e-2)
        print(
            f"FAIL: {bad.sum().item()}/{bad.numel()} elements outside tolerance "
            f"({100*bad.sum().item()/bad.numel():.2f}%)"
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
