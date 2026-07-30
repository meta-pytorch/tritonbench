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

Ragged (variable-length) cross attention: Q has length Lq per sequence, K/V have
length Lkv, packed across ``batch`` sequences. Run the backward with, e.g.::

    python run.py --op hstu_cross_attention_bwd --mode bwd

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

from .kernels import HAS_HSTU_CROSS_ATTN, IMPORT_ERROR, xa


def parse_op_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch", type=int, default=4, help="Number of ragged sequences (Z)"
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
        "--d-head",
        type=int,
        default=128,
        help="Head dimension (kernel is tuned for 128)",
    )
    parser.add_argument("--num-stages", type=int, default=2, help="bwd num_stages")
    parser.add_argument("--block-m", type=int, default=64, help="bwd BLOCK_M")
    parser.add_argument(
        "--block-n",
        type=int,
        default=64,
        help="bwd BLOCK_N (64 fits every variant; the single-block redq/autows "
        "OOR on SMEM at 128 while the data-partitioned 2-KV variants fit)",
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
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["latency", "tflops", "accuracy"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.batch = args.batch
        self.seq_len = args.seq_len
        self.seq_len_kv = args.seq_len_kv
        self.n_heads = args.n_heads
        self.d_head = args.d_head
        self.num_stages = args.num_stages
        self.block_m = args.block_m
        self.block_n = args.block_n
        self.shared = not args.separate_kv
        self.num_softmax_heads = args.num_softmax_heads
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

    def _bench(self, variant, ws: str, q, k, v, so_kv, so_q, asc) -> Callable:
        """Return a forward callable for the given bwd variant.

        The forward records the selected bwd variant into the autograd graph, so
        the later ``get_bwd_fn`` backward dispatches to that kernel.
        """
        self._pin_configs()
        xa.set_bwd_variant(variant)
        os.environ["TRITON_USE_META_WS"] = ws
        os.environ.pop("HSTU_BWD_VARIANT", None)  # let set_bwd_variant win
        Lq = int(so_q[1] - so_q[0])
        Lkv = int(so_kv[1] - so_kv[0])
        H, D = q.shape[1], q.shape[2]
        # First `num_softmax_heads` of the H heads take the softmax path, the rest
        # take SiLU; -1 resolves to all-softmax (H). attn_scale is applied only on
        # the SiLU heads (softmax heads normalize by their own denominator).
        num_softmax_heads = H if self.num_softmax_heads < 0 else self.num_softmax_heads

        def fn():
            return xa.triton_bw_hstu_mha_wrapper(
                max_seq_len=Lkv,
                alpha=1.0 / D,
                q=q,
                k=k,
                v=v,
                seq_offsets=so_kv,
                attn_scale=asc,
                max_q_len=Lq,
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
    def redq(self, q, k, v, so_kv, so_q, asc) -> Callable:
        return self._bench(xa.BwdVariant.TRITON_REDQ, "0", q, k, v, so_kv, so_q, asc)

    @register_benchmark(enabled=HAS_HSTU_CROSS_ATTN)
    def autows(self, q, k, v, so_kv, so_q, asc) -> Callable:
        return self._bench(xa.BwdVariant.TRITON_AUTOWS, "1", q, k, v, so_kv, so_q, asc)

    @register_benchmark(enabled=HAS_HSTU_CROSS_ATTN)
    def tlx(self, q, k, v, so_kv, so_q, asc) -> Callable:
        return self._bench(xa.BwdVariant.TLX, "0", q, k, v, so_kv, so_q, asc)

    @register_benchmark(enabled=HAS_HSTU_CROSS_ATTN)
    def tlx_2kv(self, q, k, v, so_kv, so_q, asc) -> Callable:
        if not self.shared:
            raise NotImplementedError(
                "tlx_2kv requires shared-KV (run without --separate-kv)"
            )
        return self._bench(xa.BwdVariant.TLX_2KV, "0", q, k, v, so_kv, so_q, asc)

    @register_benchmark(enabled=HAS_HSTU_CROSS_ATTN)
    def autows_2kv(self, q, k, v, so_kv, so_q, asc) -> Callable:
        if not self.shared:
            raise NotImplementedError(
                "autows_2kv requires shared-KV (run without --separate-kv)"
            )
        return self._bench(
            xa.BwdVariant.TRITON_AUTOWS_2KV, "1", q, k, v, so_kv, so_q, asc
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
        g = lambda n: torch.randn(n, H, D, device=self.device, dtype=self.dtype)
        q = g(tq).requires_grad_(True)
        k = g(tk).requires_grad_(True)
        # shared-KV: V aliases K (one leaf), so k.grad accumulates dk + dv.
        v = k if self.shared else g(tk).requires_grad_(True)
        so_kv = torch.arange(0, tk + 1, Lkv, device=self.device, dtype=torch.int64)
        so_q = torch.arange(0, tq + 1, Lq, device=self.device, dtype=torch.int64)
        asc = torch.tensor(1.0 / Lkv, device=self.device, dtype=torch.float32)
        return (q, k, v, so_kv, so_q, asc)

    def get_input_iter(self) -> Generator:
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

    # ---- metrics ----------------------------------------------------------
    @register_x_val(label="(Z, H, Lq, Lkv, D)")
    def get_x_val(self, example_inputs) -> tuple:
        q, k, v, so_kv, so_q, asc = example_inputs
        Z = so_kv.numel() - 1
        Lq = int(so_q[1] - so_q[0])
        Lkv = int(so_kv[1] - so_kv[0])
        return (Z, q.shape[1], Lq, Lkv, q.shape[2])

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        q, k, v, so_kv, so_q, asc = example_inputs
        Z = so_kv.numel() - 1
        H, D = q.shape[1], q.shape[2]
        Lq = int(so_q[1] - so_q[0])
        Lkv = int(so_kv[1] - so_kv[0])
        # non-causal cross attention: two matmuls (QK^T, PV) per head per sequence.
        flops_per_matmul = 2.0 * Lq * Lkv * D
        flops = 2 * flops_per_matmul * Z * H
        if self.mode == BenchmarkMode.BWD:
            flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        elif self.mode == BenchmarkMode.FWD_BWD:
            flops *= 3.5  # 1.0(fwd) + 2.0(bwd) + 0.5(recompute)
        return flops
