"""Probe the source of the 4x OSS-vs-fbcode bwd speedup for hstu Triton.

Hypothesis: fbcode bench dispatches via `hstu_mha(kernel=HammerKernel.TRITON, ...)`
which wraps `triton_hstu_mha` in a Python dispatcher with extra autograd-graph
nodes (assertions, `switch_to_contiguous_if_needed`, enum dispatch). OSS calls
`triton_hstu_mha` directly. This script times both paths in the same OSS
process to isolate harness/methodology effects.

If `hstu_mha(...)` (dispatcher) and `triton_hstu_mha(...)` (direct) give the
same bwd time, then the difference vs fbcode is in the harness, not the
dispatcher. If `hstu_mha` is ~4x slower, then the dispatcher overhead IS the
explanation.

Usage:
    cd ~/OpenSource/tritonbench
    TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 CUDA_VISIBLE_DEVICES=3 \\
        /home/mren/.conda/envs/metamain/bin/python tools/probe_bwd_dispatcher_overhead.py
"""

import sys
import time

import torch
import triton

sys.path.insert(0, "submodules/generative-recommenders")
sys.path.insert(0, "submodules/hammer")

from generative_recommenders.common import HammerKernel, generate_sparse_seq_len
from generative_recommenders.ops.hstu_attention import hstu_mha  # dispatcher
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_hstu_mha,  # the underlying kernel direct
)


def make_inputs(seed: int = 0):
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    batch_size, heads, attn_dim, hidden_dim, max_seq_len = 1024, 2, 128, 128, 4096

    torch.manual_seed(seed)
    lengths = generate_sparse_seq_len(
        size=batch_size, max_seq_len=max_seq_len, sparsity=1.0, device=device
    )
    seq_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    total_len = int(seq_offsets[-1].item())

    q = torch.randn(total_len, heads, attn_dim, dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn(total_len, heads, attn_dim, dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn(total_len, heads, hidden_dim, dtype=dtype, device=device).requires_grad_(True)

    print(f"q.is_contiguous()={q.is_contiguous()}, stride(-1)={q.stride(-1)}")
    return seq_offsets, total_len, q, k, v


def bench_bwd(label: str, make_fwd) -> float:
    """Time bwd-only of make_fwd() using triton.testing.do_bench (same harness
    as fbcode bench)."""
    out = make_fwd()
    do = torch.randn_like(out)

    def bwd_fn():
        out.backward(do, retain_graph=True)

    # Warmup + measure with the SAME knobs fbcode bench uses internally
    # (triton.testing.do_bench defaults: 25 warmup, 100 rep).
    ms = triton.testing.do_bench(bwd_fn, warmup=25, rep=100)
    print(f"  [{label}] bwd = {ms:.3f} ms")
    return ms


def main():
    max_seq_len = 4096
    max_attn_len = 32
    alpha = 1.0 / 128

    seq_offsets, total_len, q, k, v = make_inputs(seed=0)
    print(f"Inputs: max_seq_len={max_seq_len} total_len={total_len} max_attn_len={max_attn_len}")
    print()

    print("Bench: triton_hstu_mha (direct, what OSS operator uses):")
    ms_direct = bench_bwd(
        "direct",
        lambda: triton_hstu_mha(
            N=max_seq_len, alpha=alpha, q=q, k=k, v=v, seq_offsets=seq_offsets,
            num_targets=None, max_attn_len=max_attn_len, contextual_seq_len=0,
            sort_by_length=True, enable_tma=True,
        ),
    )
    print()

    print("Bench: hstu_mha(kernel=HammerKernel.TRITON, ...) (dispatcher, what fbcode bench uses):")
    ms_dispatcher = bench_bwd(
        "dispatcher",
        lambda: hstu_mha(
            max_seq_len=max_seq_len, alpha=alpha, q=q, k=k, v=v, seq_offsets=seq_offsets,
            causal=True, num_targets=None, attn_scale=None,
            max_attn_len=max_attn_len, contextual_seq_len=0, sort_by_length=True,
            kernel=HammerKernel.TRITON, enable_tma=True,
        ),
    )
    print()

    print("=" * 60)
    print(f"direct      : {ms_direct:.3f} ms")
    print(f"dispatcher  : {ms_dispatcher:.3f} ms")
    ratio = ms_dispatcher / ms_direct if ms_direct > 0 else 0.0
    print(f"ratio       : {ratio:.2f}x")
    print()
    if ratio < 1.3:
        print("VERDICT: dispatcher overhead is NOT the explanation — the bwd-time")
        print("         is dominated by the kernel, not the Python wrapper.")
        print("         The 4x OSS-vs-fbcode gap must come from harness")
        print("         differences (e.g. triton.testing.do_bench vs")
        print("         tritonbench.do_bench_wrapper measurement).")
    else:
        print("VERDICT: dispatcher adds significant bwd overhead. The fbcode bench's")
        print("         use of hstu_mha (vs the OSS operator's direct call) explains")
        print(f"         a {ratio:.2f}x slowdown.")


if __name__ == "__main__":
    main()
