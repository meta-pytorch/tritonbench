# `blocked_attention`, `blocked_gemm`, and `ragged_attention` operators

This guide covers how to set up and run the `blocked_attention`,
`blocked_gemm`, and `ragged_attention` (sliding-window HSTU) operators in
tritonbench. All three dispatch into vendored hammer / generative_recommenders
kernels under `submodules/`, so a buck/fbcode build is not required.

The benchmarks have been validated on **NVIDIA B200 (Blackwell)** with the Meta
beta Triton build (TLX-enabled).

---

## 1. Prerequisites

### Hardware
- NVIDIA Hopper (H100) or Blackwell (B200) — the TLX warp-specialized backend
  uses Blackwell `tcgen05.mma`, and `cutedsl_blackwell` requires Blackwell.
  The `triton` and `pytorch` backends work on any CUDA arch supported by
  Triton.

### Software
- Python 3.11+
- PyTorch 2.9+ with CUDA 12.8 build
- Triton 3.6.0+fb.beta (or any Triton build that ships TLX under
  `triton.language.extra.tlx`)

A working conda env at Meta typically already has torch + the beta Triton
checked out and pip-installed.

### Submodules
After cloning tritonbench, init the submodules:
```bash
git submodule update --init --recursive
```
This pulls down both `submodules/generative-recommenders` (a real git
submodule) and exposes the vendored hammer source under `submodules/hammer/`
which is already part of the tritonbench repo.

### Python dependencies

Required for the tritonbench harness itself:
```bash
pip install psutil tabulate pynvml pyyaml
```

Required only for the `cutedsl_blackwell` backend of `blocked_attention`:
```bash
pip install nvidia-cutlass-dsl
```
This also pulls in `cuda-python` / `cuda-bindings`.

Required only for the `hstu_cuda` (CUTLASS Blackwell) backend of
`ragged_attention` — see section 5 for the build steps; no extra pip deps,
just nvcc + a B200 host.

> At Meta you usually need a proxy for pypi access, e.g.
> `https_proxy=http://fwdproxy:8080 pip install …`.

---

## 2. Mandatory runtime env var

```bash
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
```

The hammer.v3 attention kernels (`triton_attention.py`, `tlx_block_attention.py`)
use a handful of module-level Python integers (e.g. `MASK_NULL`) without
wrapping them in `tl.constexpr()`. Recent Triton builds reject this by default.
The above env var is the documented escape hatch and must be set before
running any blocked_attention command. (`blocked_gemm` itself does not need
this, but setting it unconditionally is harmless.)

---

## 3. Running `blocked_attention`

### Available backends

| Backend                       | Where it lives                                  | Notes                                                                 |
| ----------------------------- | ----------------------------------------------- | --------------------------------------------------------------------- |
| `triton`                      | `hammer.v3.ops.triton.triton_attention`         | General-purpose Triton kernel, works on H100/B200.                    |
| `tlx_blackwell_ws_pipelined`  | `hammer.v3.ops.triton.tlx_block_attention` (fwd) / `tlx_block_attention_bwd` (bwd via `tlx_mha_with_grad`) | Warp-specialized Blackwell TLX kernel. Supports fwd and bwd.          |
| `cutedsl_blackwell`           | `hammer.v3.ops.cutedsl.cutedsl_attention`       | CuTeDSL Blackwell kernel. Requires `nvidia-cutlass-dsl`.              |

### Sample fwd command (B200, 5 seq-len sweep, all 3 backends)

```bash
cd ~/OpenSource/tritonbench
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 CUDA_VISIBLE_DEVICES=0 python run.py \
    --op blocked_attention --precision bf16 --mode fwd \
    --batch-size 200 --heads 4 --attn-dim 128 --hidden-dim 128 \
    --target-size 20 --min-seq-len-log2 8 --max-seq-len-log2 12 \
    --seq-sparsity 0.9 --sampling-alpha 2.0 \
    --only triton,tlx_blackwell_ws_pipelined,cutedsl_blackwell \
    --metrics latency,tflops
```

Representative output (NVIDIA B200, 1 GPU):

```
seq_len | triton (TFLOPS) | tlx_blackwell_ws_pipelined | cutedsl_blackwell
   256  |       29.7      |            49.0            |        31.0
   512  |       76.4      |           174.1            |       124.7
  1024  |      137.9      |           290.4            |       245.8
  2048  |      175.0      |           453.1            |       392.8
  4096  |      171.4      |           604.8            |       549.5
```

### Bwd

```bash
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 CUDA_VISIBLE_DEVICES=0 python run.py \
    --op blocked_attention --precision bf16 --mode bwd \
    --batch-size 200 --heads 4 --attn-dim 128 --hidden-dim 128 \
    --target-size 20 --min-seq-len-log2 8 --max-seq-len-log2 12 \
    --seq-sparsity 0.9 --sampling-alpha 2.0 \
    --only triton,tlx_blackwell_ws_pipelined,cutedsl_blackwell \
    --metrics latency,tflops
```

All three backends support bwd. The TLX backend dispatches through the
`tlx_mha_with_grad` autograd wrapper (TLX forward + the standard triton
backward kernel).

### Useful operator flags

```
--batch-size N                Batch size
--heads N                     Number of heads
--attn-dim N                  Q/K head dim
--hidden-dim N                V head dim
--min-seq-len-log2 N          Inclusive log2 min seq_len for the sweep
--max-seq-len-log2 N          Inclusive log2 max seq_len for the sweep
--seq-sparsity F              Average sparsity (0..1) of generated lengths
--sampling-alpha F            Per-batch length-distribution alpha
--target-size N               Target-block size (HSTU target block)
--full-attn-size N            Local-full block size (HSTU local-full block)
--max-attn-len N              LOCAL window size (0 disables LOCAL handling)
--mask-type {causal,all,local,diagonal}
--dtype {fp32,fp16,bf16}
```

---

## 4. Running `blocked_gemm`

### Available backends

| Backend                            | Where it lives                                       | Notes                                                                                                            |
| ---------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `pytorch`                          | `hammer.v3.ops.pytorch.pt_blocked_gemm`              | Reference baseline.                                                                                              |
| `triton_blocked_gemm_persistent`   | `hammer.v3.ops.triton.triton_blocked_gemm`           | Persistent TMA fwd (`version=""`).                                                                               |
| `triton_blocked_gemm_ws`           | `hammer.v3.ops.triton.triton_blocked_gemm`           | Warp-specialized fwd. Has both Hopper (register-accumulator) and Blackwell (TMEM-accumulator) paths.             |
| `triton_blocked_gemm_bwd`          | `hammer.v3.ops.triton.triton_blocked_gemm`           | Autograd-wrapped fwd; `.backward()` dispatches to `_BlockedGemmFunction.backward → triton_blocked_gemm_backward`. |

### Sample fwd command

```bash
cd ~/OpenSource/tritonbench
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 CUDA_VISIBLE_DEVICES=0 python run.py \
    --op blocked_gemm --precision bf16 --mode fwd \
    --num-q-blocks 2 --num-k-blocks 2 --num-w-blocks 2 \
    --metrics latency,tflops
```

Representative output (NVIDIA B200, default M=N=K=8192, 2×2×2 blocking):

```
backend                          TFLOPS
pytorch                           593.9
triton_blocked_gemm_persistent    590.9
triton_blocked_gemm_ws            506.7   (Blackwell TMEM path)
triton_blocked_gemm_bwd           521.5
```

### Sample bwd command

```bash
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 CUDA_VISIBLE_DEVICES=0 python run.py \
    --op blocked_gemm --precision bf16 --mode bwd \
    --num-q-blocks 2 --num-k-blocks 2 --num-w-blocks 2 \
    --metrics latency,tflops
```

### Useful operator flags

```
--M / --N / --K               Total M / N / K dimensions (default 8192 each)
--num-q-blocks N              Number of row (A) blocks; must divide M
--num-w-blocks N              Number of W blocks; must divide N
--num-k-blocks N              Number of inner-K blocks; must divide K
--has-bias                    Add a per-W-block bias vector
--dtype {fp32,fp16,bf16}
```

---

## 5. Running `ragged_attention` (sliding-window HSTU)

The `ragged_attention` operator measures jagged HSTU attention — the
single-sequence-per-batch flavor used by GR / HSTU recommendation models.
On Blackwell it can compare a warp-specialized Triton TLX kernel against a
hand-tuned CUTLASS kernel.

### Available backends

| Backend             | Where it lives                                                       | Notes                                                                                       |
| ------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `hstu` (baseline)   | `generative_recommenders.ops.triton.triton_hstu_attention.triton_hstu_mha` | Triton baseline; H100/B200. Always available.                                              |
| `tlx`               | `hammer.v2.ops.triton.template.tlx_bw_hstu_attention.tlx_bw_hstu_mha_wrapper` | Warp-specialized Blackwell TLX kernel (fwd + bwd).                                          |
| `hstu_cuda`         | `generative_recommenders.ops.cpp.cuda_hstu_attention.cuda_hstu_mha` (dispatches to `torch.ops.bw_hstu.bw_hstu_mha` on Blackwell) | CUTLASS Blackwell kernel — see build instructions below. On B200 hand-tuned and typically 1.5× faster than TLX. |

The `hammer_hstu`, `hstu_cuda_blackwell` backends are fbcode-only and silently
skipped in OSS.

### Building the CUTLASS backend (one-time, B200 only)

The CUTLASS HSTU kernel is a C++/CUDA extension. The source is vendored under
`submodules/generative-recommenders/generative_recommenders/fb/ultra/ops/blackwell/hstu_attention/`
(33 source files + 12 per-shape instantiations + a `setup.py`). Build it via:

```bash
cd ~/OpenSource/tritonbench
https_proxy=http://fwdproxy:8080 python install.py --hstu-blackwell
```

First build is **slow** (~30–60 min cold; cutlass-4 templates are heavy).
Produces `bw_hstu/_C.cpython-...so` (~525 MB) under the extension directory.
Subsequent builds are seconds (ninja-cached). Without this step the
`hstu_cuda` backend is silently disabled.

### Sample fwd+bwd command (the sliding-window scenario)

```bash
cd ~/OpenSource/tritonbench
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 CUDA_VISIBLE_DEVICES=0 python run.py \
    --op ragged_attention --mode fwd_bwd \
    --batch-size 1024 --heads 2 --target-size 0 --max-attn-len 32 \
    --only hstu,tlx,hstu_cuda --metrics latency
```

Representative output (NVIDIA B200, fwd+bwd combined, latency in ms):

```
seq_len   hstu (triton)   tlx        hstu_cuda (cutlass)
   256        0.533       1.210               0.685
   512        0.999       2.391               1.557
  1024        1.907       4.763               2.977
```

`hstu_cuda` is the fastest at this shape; TLX is slower than the Triton
baseline (expected — TLX's sweet spot is multi-block configurations).

### Useful operator flags

```
--batch-size N                Batch size
--heads N                     Number of heads
--attn-dim N                  Q/K head dim
--hidden-dim N                V head dim
--min-seq-len-log2 N          Inclusive log2 min seq_len for the sweep (default 8)
--max-seq-len-log2 N          Inclusive log2 max seq_len for the sweep (default 10)
--target-size N               HSTU target-block size (0 = pure jagged HSTU)
--max-attn-len N              Sliding-window size (0 disables LOCAL handling)
--seq-sparsity F              Length-distribution sparsity
--sampling-alpha F            Per-batch length-distribution alpha
--contextual-seq-len N        Contextual prefix length (0 disables)
--min-full-attn-seq-len N     Min seq_len for full-attention path
--causal                      Enable causal masking
```

---

## 6. How the OSS imports resolve

In fbcode the operators do `from hammer.v3.ops...` directly and rely on buck
to add `//hammer/v3/ops/...` to the import path.

In OSS the operator files detect `is_fbcode() == False` and inject the two
relevant submodule directories into `sys.path` at module-load time:

```
~/OpenSource/tritonbench/submodules/hammer/                 → exposes `hammer.*`
~/OpenSource/tritonbench/submodules/generative-recommenders → exposes `generative_recommenders.*`
```

(See the top of `tritonbench/operators/blocked_attention/operator.py`,
`tritonbench/operators/blocked_gemm/operator.py`, and
`tritonbench/operators/ragged_attention/{operator,hstu}.py`.) The bare
`from hammer.v3.ops.triton.triton_attention import triton_mha` style imports
then resolve identically to the fbcode build.

The vendored `submodules/hammer/` tree contains:
- `hammer/v3/ops/{triton,pytorch,cutedsl}/...` — the closure of hammer.v3
  files used by `blocked_attention` and `blocked_gemm`
  (including the fwd and bwd companions for the TLX and CuTeDSL kernels:
  `tlx_block_attention_bwd.py`, `cutedsl_attention_bwd.py`).
- `hammer/v2/ops/triton/template/{triton_attention_utils,triton_hstu_attention,tlx_bw_hstu_attention}.py`
  — the hammer.v2 helpers used by `ragged_attention`'s TLX backend and by
  `blocked_attention`'s TLX warp-specialized kernel.

The vendored CUTLASS Blackwell HSTU extension lives **inside** the
generative-recommenders submodule under
`generative_recommenders/fb/ultra/ops/blackwell/hstu_attention/` (built via
`install_hstu_blackwell()` in `install.py`).

---

## 7. Common metrics

Tritonbench accepts a comma-separated `--metrics` list. The three ops support:

- `latency`     — median kernel runtime in ms.
- `tflops`      — derived from the per-input flop count
                  (`block_flops` / `compute_flops`). Not implemented for
                  `ragged_attention`.
- `accuracy`    — pass/fail (0/1) vs. the baseline backend
                  (the `@register_benchmark(baseline=True)` one — `triton`
                  for `blocked_attention`, `pytorch` for `blocked_gemm`,
                  `hstu` for `ragged_attention`).
- `hw_roofline` — fraction of hardware peak achieved.

---

## 8. Troubleshooting

### `triton.compiler.errors.CompilationError: ... NameError("Cannot access global variable MASK_NULL ...")`
You forgot `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`. See section 2.

### `ModuleNotFoundError: No module named 'hammer'` (or `generative_recommenders.ops.utils`)
Either the tritonbench submodules were not initialized
(`git submodule update --init --recursive`), or you are on an old
tritonbench commit whose `submodules/generative-recommenders` pin predates
`generative_recommenders/ops/utils.py`. Bump tritonbench past commit
`576de7c` ("Bump submodules/generative-recommenders to include ops/utils.py")
or pull `main`.

### `ModuleNotFoundError: No module named 'cuda'` from `cutedsl_attention`
Install `nvidia-cutlass-dsl` (section 1). The other two attention backends
will still work without it.

### `triton.compiler.errors.CompilationError: ... input must be a TMEM tensor`
You are running on Blackwell with an older `triton_blocked_gemm.py` that
lacks the `IS_BLACKWELL` path. The vendored version under
`submodules/hammer/hammer/v3/ops/triton/triton_blocked_gemm.py` already has
this fix.

### `ragged_attention --only hstu_cuda` silently skipped or `AttributeError: '_OpNamespace' 'bw_hstu' object has no attribute 'bw_hstu_mha'`
The CUTLASS Blackwell extension hasn't been built yet. Run
`python install.py --hstu-blackwell` on a B200 host (one-time, slow).

### `ImportError: libc10.so: cannot open shared object file` when importing `bw_hstu._C`
The extension `.so` must be loaded *after* `import torch` so torch's lib
directory is on the dlopen path. The operator already does this in the
right order; the error means you're doing a manual import without
`import torch` first.

### `TypeError: randn_like(): argument 'input' must be Tensor, not list` on `blocked_gemm --mode bwd`
You're on an old tritonbench commit. Pull past commit `2df645c`
("blocked_gemm: support List[List[Tensor]] outputs in bwd").
