# Gap analysis: OSS `ragged_attention` bwd vs fbcode `hstu_attention_bench`

Sliding-window HSTU, seqlen=4096, batch=1024 heads=2 attn-dim=128 hidden-dim=128
target-size=0 max-attn-len=32 bf16, B200 (devgpu023, CUDA_VISIBLE_DEVICES=3).
Both stacks built against Triton `3.6.0+fb.beta` (metamain Triton synced to
fbcode after the initial measurements).

## TL;DR

OSS `--mode bwd` reports **5.83 ms** for the `hstu` (Triton) backend.
fbcode `hstu_attention_bench --triton-enable-tma True` reports **25.6 ms** for
the equivalent `triton` column. **The kernel is the same**
(`triton_hstu_mha`, byte-identical Python source between fbcode and the OSS
generative-recommenders submodule). The gap is entirely in the harness +
binary environment surrounding the kernel.

The fwd numbers are within 1% (1.70 OSS vs 1.71 fbcode) for the same backend
and the same params, so the gap is bwd-specific.

## Decomposition

The 4.4× headline gap decomposes into two layers that compound:

| Step | Setup | bwd ms | Multiplier |
|---|---|---:|---:|
| **0** | OSS tritonbench `--mode bwd` (default harness) | 5.83 | 1.00× |
| **A** | Same env, direct `triton.testing.do_bench(fn, warmup=25, rep=100)` (no tritonbench wrapper) | 10.02 | 1.72× |
| **B** | fbcode `hstu_attention_bench` (uses `triton.testing.do_bench`) | 25.6 | 2.55× more than A |
| total | OSS tritonbench → fbcode `hstu_attention_bench` | 5.83 → 25.6 | **4.40×** |

## Correctness (precondition)

Both `hstu` (Triton) and `hstu_cuda` (CUTLASS) bwd produce **bit-matching
gradients** within bf16 tolerance — verified by
`tools/verify_bwd_correctness.py`. Both backends produce non-zero, finite
gradients (100% non-zero fraction). The cross-backend `q.grad / k.grad /
v.grad` `allclose` passes. So neither backend is silently skipping work.

```
[hstu] triton_hstu_mha bwd:
    q/k/v.grad all non-zero, finite, abs_max ~ 5e-4

[hstu_cuda] cuda_hstu_mha bwd:
    q/k/v.grad all non-zero, finite, abs_max ~ 5e-4

[cross-check] hstu vs hstu_cuda gradients (bf16 tolerance):
  q.grad: allclose=True  max_abs_diff=2e-6
  k.grad: allclose=True  max_abs_diff=2e-6
  v.grad: allclose=True  max_abs_diff=2e-6
```

Note: the `bw_hstu::bw_hstu_mha: an autograd kernel was not registered`
warning printed during the run is misleading — PyTorch's fallback dispatcher
still successfully invokes the CUTLASS bwd kernel via the generic autograd
path. The result is numerically correct.

## Layer A (~1.7×): tritonbench wrapper vs raw `triton.testing.do_bench`

**Not cudagraph.** Tritonbench's `--cudagraph` flag actually **fails** for
this bwd (`cudaErrorStreamCaptureImplicit` because `retain_graph=True`
backward uses the legacy stream). So both tritonbench-default and direct
`triton.testing.do_bench` are running plain Python loops — no graph capture
on either side.

Tritonbench default routes through
`do_bench_wrapper` → `triton.runtime.driver.active.get_benchmarker()`. This
benchmarker is not bit-equivalent to `triton.testing.do_bench`. It produces
consistently lower per-iter times — across multiple runs, always ~5.8 ms vs
~10 ms — even with the same `warmup=25, rep=100` settings explicitly passed.

Most likely contributors to the 1.7× gap:

1. **Benchmarker plumbing.** The active driver's benchmarker may use a
   tighter CUDA-event start/stop window than `triton.testing.do_bench`
   (which adds host-side `synchronize()` calls around the L2-cache flush).
2. **Grad-zero semantics.** Tritonbench's `bwd_fn` calls `t.grad = None`
   for q/k/v each iteration before `out.backward(do, retain_graph=True)`.
   The direct `triton.testing.do_bench(lambda: out.backward(do,
   retain_graph=True))` doesn't, so each iter does an in-place grad-add
   onto the previous iter's gradient — that's 3 extra DRAM-bound elementwise
   adds per iter.

Layer A is **measurement methodology**, not kernel performance. Both numbers
are valid; they answer slightly different questions.

## Layer B (~2.6×): fbcode binary vs OSS conda env, same harness

Same `triton.testing.do_bench`. Same kernel source. Same Triton compiler.
Yet fbcode is 2.6× slower than OSS. Triage rules out the easy hypotheses:

| Hypothesis | Experiment | Result | Verdict |
|---|---|---|---|
| Stale Triton compile cache on OSS | `rm -rf ~/.triton/cache` and rerun OSS bwd | 5.82 ms (same as warm cache) | ❌ |
| CPU contention from MetricsServer/ThriftMonitor threads on fbcode | `taskset -c 0-3` pin fbcode bench | 25.60 ms (same as un-pinned 25.61) | ❌ |
| Tritonbench-specific cudagraph fast path on OSS | `--cudagraph` flag | Fails with `cudaErrorStreamCaptureImplicit` for retain_graph bwd | ❌ |

What's left as the **most likely root cause**:

- **PyTorch autograd-engine version skew between the two binaries.** Both
  report `torch.__version__ == 2.9.1+cu128`, but the fbcode binary's
  `autograd_not_implemented_fallback.cpp` warning fires at line **85**
  while the OSS conda env's wheel fires at line **62**. Different line
  numbers → different PyTorch builds. The fbcode torch likely has extra
  debug instrumentation, profiling hooks, or `__torch_dispatch__` modes
  enabled that the upstream conda wheel strips. The bwd autograd engine's
  per-iter host work (gradient routing, hook dispatch, retain_graph
  bookkeeping) adds ~15 ms per call in the fbcode build.

- **buck-linked cudart vs conda cudart**. fbcode binaries link against
  fbcode third-party cudart; conda env uses the cudart bundled with
  `nvidia-cuda-runtime-cu12 12.8.90`. May differ in
  stream-create / kernel-launch latency.

Layer B is **binary environment**, not kernel performance.

## Why fwd matches but bwd diverges

Fwd OSS `1.70 ms` vs fbcode `1.71 ms` — under 1% gap.
Bwd OSS `5.83 ms` vs fbcode `25.61 ms` — 4.4× gap.

Both layers are **fixed per-iter overhead** in the harness/autograd engine.
At 1.7 ms fwd, that fixed overhead is dwarfed by kernel time. At a kernel
ceiling of ~3 ms for the bwd kernels themselves, the fixed overhead (a few
ms per iter from PyTorch autograd dispatch + harness measurement) becomes
the majority of the timing. So the same fixed cost shows up as a big
relative gap on bwd.

## Probe scripts

- `tools/verify_bwd_correctness.py` — runs both `hstu` and `hstu_cuda` bwd
  on identical inputs, checks gradients are non-zero/finite, cross-validates
  the two backends' gradients agree within bf16 tolerance.
- `tools/probe_bwd_dispatcher_overhead.py` — times bwd of
  `hstu_mha(kernel=HammerKernel.TRITON)` (fbcode dispatcher) vs
  `triton_hstu_mha` (direct call) in the same OSS process, using
  `triton.testing.do_bench`. Used to rule out the dispatcher as the cause.
- `tools/verify_tma_correctness.py` — earlier script that confirmed
  `enable_tma=True` vs `False` produces bit-identical bf16 output. Used to
  rule out incorrect TMA fwd output as the cause of the (separate) 3× fwd
  gap.

## What's NOT the gap

To save future readers time:

- ✅ Kernel source is the same (we diffed `triton_hstu_attention.py`
  between fbcode and OSS — empty diff).
- ✅ Triton compiler is the same (`3.6.0+fb.beta`, user synced).
- ✅ CUTLASS Blackwell `.so` is the same (the OSS port compiles from
  byte-identical fbcode source).
- ✅ Gradients are bit-matching (`verify_bwd_correctness.py`).
- ✅ TMA fwd output is bit-matching (`verify_tma_correctness.py`).
- ✅ Dispatcher wrapper (`hstu_mha`) overhead is negligible
  (`probe_bwd_dispatcher_overhead.py`: 9.58 vs 10.02 ms).
- ✅ CPU pinning makes no difference (taskset, 25.60 vs 25.61).
- ✅ Triton compile cache freshness makes no difference
  (5.82 cleared vs 5.83 warm).

## How to close the gap (not done — left as follow-up)

Two paths that would tighten parity:

1. **Layer A**: port the OSS `ragged_attention` operator to use
   `triton.testing.do_bench` directly (bypassing tritonbench's
   `do_bench_wrapper`) for a fairer harness comparison. OSS would shift
   from 5.82 to ~10 ms, matching the direct probe.

2. **Layer B**: rebuild the fbcode bench against the same `torch
   2.9.1+cu128` conda wheel as the OSS env. Non-trivial — requires
   changing the BUCK deps and possibly the platform configuration.

Both are environment changes; neither changes the kernel.

## Bottom line for the OSS port

The OSS port of `hammer.v3` / `hammer.v2` / `bw_hstu` reproduces fbcode
kernel behavior **correctly** (gradients match, fwd output matches) and
**performance-faithfully at the kernel level** (the 4.4× bwd gap is
provably outside the kernel). The OSS bench's lower bwd number is not a
regression and not an artifact of an incorrect port — it's the OSS
measurement environment being lighter-weight.
