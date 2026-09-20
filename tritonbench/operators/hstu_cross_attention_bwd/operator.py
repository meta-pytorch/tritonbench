# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HSTU cross-attention backward (reduce_dq) benchmark.

Benchmarks the HSTU cross-attention backward pass across its reduce_dq variants:

  * redq        - non-WS reduce_dq (trusted reference, baseline)
  * autows      - automatic warp specialization (meta-WS) on the inner Q loop
  * tlx         - hand-written TLX warp-specialized reduce_dq (attn_bwd_ws)
  * tlx_2kv     - TLX 2-KV-block data-partitioned reduce_dq (shared-KV only)
  * autows_2kv  - manual 2-KV-block data-partition + autoWS (shared-KV only)

Cross attention packs Q and K/V independently across ``batch`` sequences. The
default input sweep uses uniform lengths; ``--prod-shapes`` switches to the
production-like jagged GQA/shared-KV workload. Run the backward with, e.g.::

    python run.py --op hstu_cross_attention_bwd --mode bwd
    python run.py --op hstu_cross_attention_bwd --mode bwd --prod-shapes

The 2-KV variants require shared-KV (V aliases K); they are enabled by default
and skipped under ``--separate-kv``.
"""

import argparse
import os
from typing import Any, Callable, Generator, List, Optional

import torch
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
    register_x_val,
)

from tritonbench.operators.ragged_attention.input_utils import (
    generate_sparse_seq_len,
)

from .kernels import HAS_HSTU_CROSS_ATTN, IMPORT_ERROR, xa


def _positive_int_list(value: str) -> List[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_op_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Number of sequences (default: 4, or 1024 with --prod-shapes)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Q sequence length per sequence (Lq)"
    )
    parser.add_argument(
        "--seq-len-kv",
        type=int,
        default=None,
        help="KV sequence length (Lkv); default sweeps a few values",
    )
    parser.add_argument("--n-heads", type=int, default=2, help="Number of heads")
    parser.add_argument(
        "--n-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads (default: n-heads, or 1 with --prod-shapes)",
    )
    parser.add_argument(
        "--d-head",
        type=int,
        default=128,
        help="Head dimension (kernel is tuned for 128)",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=None,
        help="bwd num_stages (default: 2, or 1 with --prod-shapes)",
    )
    parser.add_argument("--block-m", type=int, default=64, help="bwd BLOCK_M")
    parser.add_argument(
        "--block-n",
        type=int,
        default=None,
        help="bwd BLOCK_N (default: 64, or TLX-aligned 128 with --prod-shapes)",
    )
    parser.add_argument(
        "--separate-kv",
        action="store_true",
        help="Use separate K and V (disables the shared-KV tlx_2kv/autows_2kv variants)",
    )
    parser.add_argument(
        "--num-softmax-heads",
        type=int,
        default=-1,
        help="Heads using softmax vs SiLU: the first S of n_heads use softmax, the "
        "rest use SiLU (HSTU). Mirrors fbcode hstu_cross_attn_bench --heads-softmax. "
        "-1 (default) => all heads softmax (the GQA-on-B200 recipe); 0 => all SiLU. "
        "The TLX variants require 0 or n_heads.",
    )
    parser.add_argument(
        "--max-kv",
        type=int,
        default=4096,
        help="Maximum KV length for --prod-shapes",
    )
    parser.add_argument(
        "--max-targets",
        type=_positive_int_list,
        default=[32, 128, 160, 256],
        help="Comma-separated maximum Q/target lengths for --prod-shapes",
    )
    parser.add_argument(
        "--seq-sparsity",
        type=float,
        default=0.95,
        help="Mean KV-length fraction for --prod-shapes",
    )
    parser.add_argument(
        "--input-seed",
        type=int,
        default=1001,
        help="Random seed for --prod-shapes",
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["latency", "tflops", "accuracy"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.batch = (
            args.batch if args.batch is not None else (1024 if self.prod_shapes else 4)
        )
        self.seq_len = args.seq_len
        self.seq_len_kv = args.seq_len_kv
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = 1 if self.prod_shapes else self.n_heads
        self.d_head = args.d_head
        self.num_stages = (
            args.num_stages
            if args.num_stages is not None
            else (1 if self.prod_shapes else 2)
        )
        self.block_m = args.block_m
        self.block_n = (
            args.block_n
            if args.block_n is not None
            else (128 if self.prod_shapes else 64)
        )
        self.shared = not args.separate_kv
        self.num_softmax_heads = args.num_softmax_heads
        self.max_kv = args.max_kv
        self.max_targets = args.max_targets
        self.seq_sparsity = args.seq_sparsity
        self.input_seed = args.input_seed
        if self.batch <= 0 or self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("batch, n-heads, and n-kv-heads must be positive")
        if self.max_kv <= 1:
            raise ValueError("max-kv must be greater than 1")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n-heads must be divisible by n-kv-heads")
        if not 0.0 <= self.seq_sparsity <= 1.0:
            raise ValueError("seq-sparsity must be between 0 and 1")
        if self.prod_shapes and not self.shared:
            raise ValueError("--prod-shapes requires shared K/V")
        if not HAS_HSTU_CROSS_ATTN:
            raise RuntimeError(
                f"HSTU cross-attention kernel is unavailable: {IMPORT_ERROR!r}"
            )

    # ---- config pinning ---------------------------------------------------
    def _pin_configs(self) -> None:
        """Pin bwd num_stages / block sizes on the kernel autotune configs.

        Mirrors the standalone bench_bwd.force: keep one config per distinct
        INNER_PICK for the 2-KV kernel so list-schedule autotuning still sweeps
        the inner-loop schedule when TRITON_USE_LIST_SCHEDULE=1.
        """
        ns, bm, bn = self.num_stages, self.block_m, self.block_n
        fwd = getattr(xa, "_attn_fwd_triton", None)
        if fwd is not None and hasattr(fwd, "configs"):
            c = fwd.configs[0]
            c.num_stages = max(getattr(c, "num_stages", 1), 1)
            fwd.configs = [c]
        c = xa._hstu_attn_bwd_redq.configs[0]
        c.num_stages = ns
        c.kwargs["BLOCK_M"] = bm
        c.kwargs["BLOCK_N"] = bn
        xa._hstu_attn_bwd_redq.configs = [c]
        if hasattr(xa, "_hstu_attn_bwd_redq_2kv"):
            kept, seen = [], set()
            for c2 in xa._hstu_attn_bwd_redq_2kv.configs:
                c2.num_stages = ns
                c2.kwargs["BLOCK_M"] = bm
                c2.kwargs["BLOCK_N"] = bn
                pk = c2.kwargs.get("INNER_PICK", 0)
                if pk in seen:
                    continue
                seen.add(pk)
                kept.append(c2)
            xa._hstu_attn_bwd_redq_2kv.configs = kept
        xa.set_fwd_variant(xa.FwdVariant.TRITON)

    def _bench(self, variant, ws: str, q, k, v, so_kv, so_q, asc, limits) -> Callable:
        """Return a forward callable for the given bwd variant.

        The forward records the selected bwd variant into the autograd graph, so
        the later ``get_bwd_fn`` backward dispatches to that kernel.
        """
        self._pin_configs()
        xa.set_bwd_variant(variant)
        os.environ["TRITON_USE_META_WS"] = ws
        os.environ.pop("HSTU_BWD_VARIANT", None)  # let set_bwd_variant win
        max_q_len, max_kv_len = limits
        H, D = q.shape[1], q.shape[2]
        # First `num_softmax_heads` of the H heads take the softmax path, the rest
        # take SiLU; -1 resolves to all-softmax (H). attn_scale is applied only on
        # the SiLU heads (softmax heads normalize by their own denominator).
        num_softmax_heads = H if self.num_softmax_heads < 0 else self.num_softmax_heads

        def fn():
            return xa.triton_bw_hstu_mha_wrapper(
                max_seq_len=max_kv_len,
                alpha=1.0 / D,
                q=q,
                k=k,
                v=v,
                seq_offsets=so_kv,
                attn_scale=asc,
                max_q_len=max_q_len,
                seq_offsets_q=so_q,
                num_softmax_heads=num_softmax_heads,
                shared_kv=self.shared,
                enable_tma=True,
            )

        # Deduplicate by identity: under shared-KV, v IS k (grad accumulates dk+dv).
        seen, grad_inputs = set(), []
        for t in (q, k, v):
            if t.requires_grad and id(t) not in seen:
                seen.add(id(t))
                grad_inputs.append(t)
        fn._grad_inputs = grad_inputs
        return fn

    # ---- benchmark variants ----------------------------------------------
    @register_benchmark(enabled=HAS_HSTU_CROSS_ATTN, baseline=True)
    def redq(self, q, k, v, so_kv, so_q, asc, limits) -> Callable:
        return self._bench(
            xa.BwdVariant.TRITON_REDQ, "0", q, k, v, so_kv, so_q, asc, limits
        )

    @register_benchmark(enabled=HAS_HSTU_CROSS_ATTN)
    def autows(self, q, k, v, so_kv, so_q, asc, limits) -> Callable:
        return self._bench(
            xa.BwdVariant.TRITON_AUTOWS, "1", q, k, v, so_kv, so_q, asc, limits
        )

    @register_benchmark(enabled=HAS_HSTU_CROSS_ATTN)
    def tlx(self, q, k, v, so_kv, so_q, asc, limits) -> Callable:
        return self._bench(xa.BwdVariant.TLX, "0", q, k, v, so_kv, so_q, asc, limits)

    @register_benchmark(enabled=HAS_HSTU_CROSS_ATTN)
    def tlx_2kv(self, q, k, v, so_kv, so_q, asc, limits) -> Callable:
        if not self.shared:
            raise NotImplementedError(
                "tlx_2kv requires shared-KV (run without --separate-kv)"
            )
        return self._bench(
            xa.BwdVariant.TLX_2KV, "0", q, k, v, so_kv, so_q, asc, limits
        )

    @register_benchmark(enabled=HAS_HSTU_CROSS_ATTN)
    def autows_2kv(self, q, k, v, so_kv, so_q, asc, limits) -> Callable:
        if not self.shared:
            raise NotImplementedError(
                "autows_2kv requires shared-KV (run without --separate-kv)"
            )
        return self._bench(
            xa.BwdVariant.TRITON_AUTOWS_2KV,
            "1",
            q,
            k,
            v,
            so_kv,
            so_q,
            asc,
            limits,
        )

    # ---- backward driver --------------------------------------------------
    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        o = fwd_fn()
        grad_inputs = fwd_fn._grad_inputs
        torch.manual_seed(0)
        do = (0.1 * torch.randn_like(o)).detach()

        def fn():
            for t in grad_inputs:
                t.grad = None
            o.backward(do, retain_graph=True)
            return grad_inputs

        return fn

    # ---- inputs -----------------------------------------------------------
    def _make_inputs(self, Lkv):
        Z, H, D, Lq = self.batch, self.n_heads, self.d_head, self.seq_len
        tq, tk = Z * Lq, Z * Lkv

        def g(n):
            return torch.randn(n, H, D, device=self.device, dtype=self.dtype)

        q = g(tq).requires_grad_(True)
        k = g(tk).requires_grad_(True)
        # shared-KV: V aliases K (one leaf), so k.grad accumulates dk + dv.
        v = k if self.shared else g(tk).requires_grad_(True)
        so_kv = torch.arange(0, tk + 1, Lkv, device=self.device, dtype=torch.int64)
        so_q = torch.arange(0, tq + 1, Lq, device=self.device, dtype=torch.int64)
        asc = torch.tensor(1.0 / Lkv, device=self.device, dtype=torch.float32)
        return (q, k, v, so_kv, so_q, asc, (Lq, Lkv))

    def _make_production_inputs(self, max_targets):
        Z, Hq, Hkv, D = self.batch, self.n_heads, self.n_kv_heads, self.d_head
        max_kv = self.max_kv
        device = self.device
        generator = torch.Generator(device=device).manual_seed(self.input_seed)

        lengths_kv = generate_sparse_seq_len(
            size=Z,
            max_seq_len=max_kv,
            sparsity=self.seq_sparsity,
            device=device,
            generator=generator,
        ).to(torch.int64)

        min_targets = 1 if max_targets < 400 else max_targets // 2
        lengths_q = torch.randint(
            min_targets,
            max_targets + 1,
            (Z,),
            device=device,
            dtype=torch.int64,
            generator=generator,
        )
        so_kv = torch.zeros(Z + 1, device=device, dtype=torch.int64)
        so_q = torch.zeros(Z + 1, device=device, dtype=torch.int64)
        torch.cumsum(lengths_kv, 0, out=so_kv[1:])
        torch.cumsum(lengths_q, 0, out=so_q[1:])
        total_q, total_kv = int(so_q[-1]), int(so_kv[-1])

        def tensor(tokens, heads):
            return torch.empty(
                tokens, heads, D, device=device, dtype=self.dtype
            ).uniform_(-0.1, 0.1, generator=generator)

        q = tensor(total_q, Hq).requires_grad_(True)
        k = tensor(total_kv, Hkv).requires_grad_(True)
        v = k
        asc = torch.tensor(
            1.0 / (max_kv + max_targets), device=device, dtype=torch.float32
        )
        return (q, k, v, so_kv, so_q, asc, (max_targets, max_kv))

    def get_input_iter(self) -> Generator:
        if self.prod_shapes:
            for max_targets in self.max_targets:
                yield self._make_production_inputs(max_targets)
            return
        if self.seq_len_kv is not None:
            kv_lens = [self.seq_len_kv]
        elif self.shared:
            # 384 exercises an odd number of KV blocks (partial tail pair) for the
            # 2-KV variants; the rest are round powers of two.
            kv_lens = [256, 384, 512, 1024]
        else:
            kv_lens = [128, 256, 512]
        for Lkv in kv_lens:
            yield self._make_inputs(Lkv)

    def get_available_num_inputs(self) -> int:
        if self.prod_shapes:
            return len(self.max_targets)
        if self.seq_len_kv is not None:
            return 1
        return 4 if self.shared else 3

    # ---- metrics ----------------------------------------------------------
    @register_x_val(label="(Z, Hq, Hkv, maxLq, maxLkv, D, qTokens, kvTokens)")
    def get_x_val(self, example_inputs) -> tuple:
        q, k, v, so_kv, so_q, asc, limits = example_inputs
        max_q_len, max_kv_len = limits
        Z = so_kv.numel() - 1
        return (
            Z,
            q.shape[1],
            k.shape[1],
            max_q_len,
            max_kv_len,
            q.shape[2],
            q.shape[0],
            k.shape[0],
        )

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        q, k, v, so_kv, so_q, asc, limits = example_inputs
        H, D = q.shape[1], q.shape[2]
        lengths_q = so_q[1:] - so_q[:-1]
        lengths_kv = so_kv[1:] - so_kv[:-1]
        # non-causal cross attention: two matmuls (QK^T, PV) per head per sequence.
        qk_pairs = int(torch.sum(lengths_q * lengths_kv).item())
        flops_per_matmul = 2.0 * qk_pairs * D * H
        flops = 2 * flops_per_matmul
        if self.mode == BenchmarkMode.BWD:
            flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        elif self.mode == BenchmarkMode.FWD_BWD:
            flops *= 3.5  # 1.0(fwd) + 2.0(bwd) + 0.5(recompute)
        return flops
