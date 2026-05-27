# `blocked_attention` and `blocked_gemm` operators

This guide covers how to set up and run the `blocked_attention` and
`blocked_gemm` operators in tritonbench. Both dispatch into `hammer.v3` kernels
that are vendored under `submodules/hammer/`, so a buck/fbcode build is not
required.

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
| `tlx_blackwell_ws_pipelined`  | `hammer.v3.ops.triton.tlx_block_attention`      | Warp-specialized Blackwell TLX kernel. `fwd_only`.                    |
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
    --only triton,cutedsl_blackwell \
    --metrics latency,tflops
```

Note: `tlx_blackwell_ws_pipelined` is fwd-only; it will be silently skipped
when `--mode bwd` is passed.

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

## 5. How the OSS imports resolve

In fbcode the operators do `from hammer.v3.ops...` directly and rely on buck
to add `//hammer/v3/ops/...` to the import path.

In OSS the operator files detect `is_fbcode() == False` and inject the two
relevant submodule directories into `sys.path` at module-load time:

```
~/OpenSource/tritonbench/submodules/hammer/                 → exposes `hammer.*`
~/OpenSource/tritonbench/submodules/generative-recommenders → exposes `generative_recommenders.*`
```

(See the top of `tritonbench/operators/blocked_attention/operator.py` and
`tritonbench/operators/blocked_gemm/operator.py`.) The bare
`from hammer.v3.ops.triton.triton_attention import triton_mha` style imports
then resolve identically to the fbcode build.

The vendored `submodules/hammer/` tree contains exactly the closure of
hammer.v3 files needed by these two operators, plus one helper from hammer.v2
(`triton_attention_utils._get_bufidx_phase`).

---

## 6. Common metrics

Tritonbench accepts a comma-separated `--metrics` list. The two ops support:

- `latency`     — median kernel runtime in ms.
- `tflops`      — derived from the per-input flop count
                  (`block_flops` / `compute_flops`).
- `accuracy`    — pass/fail (0/1) vs. the baseline backend
                  (the `@register_benchmark(baseline=True)` one — `triton`
                  for blocked_attention, `pytorch` for blocked_gemm).
- `hw_roofline` — fraction of hardware peak achieved.

---

## 7. Troubleshooting

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
