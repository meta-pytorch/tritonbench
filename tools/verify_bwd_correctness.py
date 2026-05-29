"""Standalone bwd correctness check for the ragged_attention OSS backends.

Verifies that the `hstu` (Triton baseline) and `hstu_cuda` (CUTLASS Blackwell)
backends actually compute correct gradients with respect to q/k/v.

Motivation: in the ragged_attention bench at seqlen=4096 sliding-window 32, the
OSS bwd timings come in suspiciously low (e.g. hstu 5.83 ms vs fbcode 25.61 ms).
A common cause is autograd silently falling back to a no-op:

    UserWarning: bw_hstu::bw_hstu_mha: an autograd kernel was not registered to
    the Autograd key(s) but we are trying to backprop through it. This may lead
    to silently incorrect behavior.

This script:

  1. Calls `triton_hstu_mha` (the Triton baseline) with requires_grad q/k/v,
     runs .backward(dy), and confirms q.grad/k.grad/v.grad are non-zero and
     finite.
  2. Calls `cuda_hstu_mha` (the CUTLASS dispatcher) the same way and does the
     same checks.
  3. Cross-validates: computes q.grad/k.grad/v.grad with both backends on
     identical inputs and verifies the gradients agree within bf16 tolerance.

If a backend's grads are all zero, that's a silent autograd-fallback bug — the
backward kernel never ran. Such a backend's bwd timing in the benchmark is
meaningless.

Usage:
    cd ~/OpenSource/tritonbench
    TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 CUDA_VISIBLE_DEVICES=3 \\
        /home/mren/.conda/envs/metamain/bin/python tools/verify_bwd_correctness.py
"""

import sys

import torch

# Wire up the OSS submodules the same way the operator does.
sys.path.insert(0, "submodules/generative-recommenders")
sys.path.insert(0, "submodules/hammer")

# Make sure the Blackwell CUTLASS .so is loaded so torch.ops.bw_hstu.bw_hstu_mha
# is registered before cuda_hstu_attention is imported.
sys.path.insert(
    0,
    "submodules/generative-recommenders/generative_recommenders/fb/ultra/ops/blackwell/hstu_attention",
)
import torch as _torch  # noqa: F401  - keep torch loaded before _C.so

try:
    import bw_hstu._C  # noqa: F401 - side effects: register torch.ops.bw_hstu
    HAS_CUDA_BACKEND = True
except Exception as e:
    print(f"[warn] bw_hstu._C import failed: {e}")
    print("[warn] hstu_cuda backend will be skipped.")
    HAS_CUDA_BACKEND = False

from generative_recommenders.common import generate_sparse_seq_len
from generative_recommenders.ops.triton.triton_hstu_attention import triton_hstu_mha

if HAS_CUDA_BACKEND:
    from generative_recommenders.ops.cpp.cuda_hstu_attention import cuda_hstu_mha


def make_inputs(
    *,
    batch_size: int = 1024,
    heads: int = 2,
    attn_dim: int = 128,
    hidden_dim: int = 128,
    max_seq_len: int = 4096,
    sparsity: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    seed: int = 0,
):
    """Build identical jagged q/k/v with requires_grad=True."""
    if device is None:
        device = torch.device("cuda:0")
    torch.manual_seed(seed)

    lengths = generate_sparse_seq_len(
        size=batch_size, max_seq_len=max_seq_len, sparsity=sparsity, device=device
    )
    seq_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    total_len = int(seq_offsets[-1].item())

    q = torch.randn(total_len, heads, attn_dim, dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn(total_len, heads, attn_dim, dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn(total_len, heads, hidden_dim, dtype=dtype, device=device).requires_grad_(True)
    return seq_offsets, total_len, q, k, v


def check_grad(name: str, grad: torch.Tensor | None) -> bool:
    """Return True iff grad exists, has the right finite distribution to look real."""
    if grad is None:
        print(f"    {name}.grad is None  (autograd did not run)")
        return False
    if torch.all(grad == 0):
        print(
            f"    {name}.grad is all-zero shape={tuple(grad.shape)}  "
            f"(autograd fallback or no-op kernel)"
        )
        return False
    if not torch.isfinite(grad).all():
        print(
            f"    {name}.grad has non-finite values: "
            f"nan={int(torch.isnan(grad).sum())} inf={int(torch.isinf(grad).sum())}"
        )
        return False
    print(
        f"    {name}.grad shape={tuple(grad.shape)} "
        f"abs_max={grad.abs().max().item():.6f} "
        f"mean_abs={grad.abs().mean().item():.6e} "
        f"nonzero_frac={(grad != 0).float().mean().item():.4f}"
    )
    return True


def run_bwd_and_collect(name: str, fn, q, k, v, dy):
    """Run fn() -> out, backprop with dy, return cloned q/k/v grads."""
    for t in (q, k, v):
        if t.grad is not None:
            t.grad = None
    out = fn()
    out.backward(dy)
    return out.detach().clone(), q.grad.detach().clone() if q.grad is not None else None, \
           k.grad.detach().clone() if k.grad is not None else None, \
           v.grad.detach().clone() if v.grad is not None else None


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; aborting.")
        return 1

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    max_seq_len = 4096
    max_attn_len = 32
    alpha = 1.0 / 128  # = 1/attn_dim
    sort_by_length = True

    seq_offsets, total_len, q, k, v = make_inputs(
        batch_size=1024, heads=2, attn_dim=128, hidden_dim=128,
        max_seq_len=max_seq_len, dtype=dtype, device=device, seed=0,
    )
    print(
        f"Inputs: batch=1024 heads=2 max_seq_len={max_seq_len} "
        f"total_len={total_len} max_attn_len={max_attn_len} dtype={dtype} "
        f"sort_by_length={sort_by_length}"
    )
    print()

    # Build dy with shape matching the fwd output: (total_len, heads, hidden_dim).
    torch.manual_seed(123)
    dy = torch.randn(total_len, 2, 128, dtype=dtype, device=device)

    # 1) hstu (Triton baseline) — should always have a real bwd via Triton autograd.
    print("[hstu] triton_hstu_mha bwd:")
    out_hstu, gq_hstu, gk_hstu, gv_hstu = run_bwd_and_collect(
        "hstu",
        lambda: triton_hstu_mha(
            N=max_seq_len, alpha=alpha,
            q=q, k=k, v=v,
            seq_offsets=seq_offsets,
            num_targets=None,
            max_attn_len=max_attn_len,
            contextual_seq_len=0,
            sort_by_length=sort_by_length,
            enable_tma=True,
        ),
        q, k, v, dy,
    )
    hstu_ok = all(check_grad(n, g) for n, g in [("q", gq_hstu), ("k", gk_hstu), ("v", gv_hstu)])
    print(f"  [hstu] grads ok: {hstu_ok}")
    print()

    # 2) hstu_cuda (CUTLASS Blackwell via cuda_hstu_mha).
    if HAS_CUDA_BACKEND:
        print("[hstu_cuda] cuda_hstu_mha bwd:")
        # cuda_hstu_mha takes slightly different kwargs; mirror the OSS operator.
        out_cuda, gq_cuda, gk_cuda, gv_cuda = run_bwd_and_collect(
            "hstu_cuda",
            lambda: cuda_hstu_mha(
                max_seq_len=max_seq_len, alpha=alpha,
                q=q, k=k, v=v,
                seq_offsets=seq_offsets,
                causal=False,
                num_targets=None,
                max_attn_len=max_attn_len,
                sort_by_length=True,
            ),
            q, k, v, dy,
        )
        cuda_ok = all(
            check_grad(n, g) for n, g in [("q", gq_cuda), ("k", gk_cuda), ("v", gv_cuda)]
        )
        print(f"  [hstu_cuda] grads ok: {cuda_ok}")
        print()
    else:
        cuda_ok = False
        out_cuda = gq_cuda = gk_cuda = gv_cuda = None

    # 3) Cross-validate hstu vs hstu_cuda grads if both produced real grads.
    if hstu_ok and cuda_ok:
        print("[cross-check] hstu vs hstu_cuda gradients (bf16 tolerance):")
        # First check fwd outputs match.
        out_diff = (out_hstu.float() - out_cuda.float()).abs()
        print(
            f"  fwd output: max_abs_diff={out_diff.max().item():.6f} "
            f"mean_abs_diff={out_diff.mean().item():.6e}"
        )
        rtol, atol = 1e-2, 1e-2
        all_grad_close = True
        for name, ga, gb in [
            ("q.grad", gq_hstu, gq_cuda),
            ("k.grad", gk_hstu, gk_cuda),
            ("v.grad", gv_hstu, gv_cuda),
        ]:
            close = torch.allclose(ga.float(), gb.float(), rtol=rtol, atol=atol)
            diff = (ga.float() - gb.float()).abs()
            print(
                f"  {name}: allclose={close}  max_abs_diff={diff.max().item():.6f}  "
                f"mean_abs_diff={diff.mean().item():.6e}"
            )
            if not close:
                all_grad_close = False
        print()
        print(
            "  cross-check verdict:",
            "PASS — both backends produce matching gradients" if all_grad_close
            else "FAIL — gradients disagree (one of the bwd kernels is incorrect)",
        )

    # 4) Summary.
    print()
    print("=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"  hstu       bwd produces real grads: {hstu_ok}")
    if HAS_CUDA_BACKEND:
        print(f"  hstu_cuda  bwd produces real grads: {cuda_ok}")
    else:
        print("  hstu_cuda  bwd: SKIPPED (bw_hstu._C import failed)")

    if not hstu_ok or (HAS_CUDA_BACKEND and not cuda_ok):
        print()
        print(
            "WARNING: at least one backend's bwd is not producing real gradients. "
            "Its bench-reported bwd latency is meaningless."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
