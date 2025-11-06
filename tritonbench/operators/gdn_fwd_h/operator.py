import argparse
import functools
import itertools
import math
import os
import sys
from contextlib import nullcontext
from itertools import chain

from typing import Any, Callable, Generator, List, Optional

import torch

from tritonbench.utils.input import input_filter
from tritonbench.utils.python_utils import try_import

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
    register_x_val,
)

with try_import("HAS_FLA"):
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h as chunk_gated_delta_rule_fwd_h_kernel,
    )

    def fla_chunk_gated_delta_rule_fwd_h(k, w, u, g, chunk_size):
        return chunk_gated_delta_rule_fwd_h_kernel(k, w, u, g, chunk_size=chunk_size)[0]


with try_import("HAS_TILELANG"):
    import tilelang

    TL_ROOT = os.getenv("TL_ROOT", os.path.dirname(os.path.abspath(tilelang.__file__)))
    sys.path.append(f"{TL_ROOT}/examples/gdn")
    from example_chunk_delta_h import (
        tilelang_chunk_gated_delta_rule_fwd_h as tilelang_chunk_gated_delta_rule_fwd_h_kernel,
    )

    def tilelang_chunk_gated_delta_rule_fwd_h(k, w, u, g, chunk_size):
        batch, seqlen, nheads, dhead = k.shape
        expand_v = u.shape[-1] // dhead
        dtype = str(k.dtype).removeprefix("torch.")
        accum_dtype = "float32"
        initial_state = torch.empty(
            batch, nheads, dhead, expand_v * dhead, dtype=k.dtype, device=k.device
        )
        # default block settings hang
        return tilelang_chunk_gated_delta_rule_fwd_h_kernel(
            batch,
            seqlen,
            nheads,
            dhead,
            expand_v * dhead,
            dtype,
            dtype,
            accum_dtype,
            accum_dtype,
            dtype,
            use_g=True,
            use_initial_state=False,
            store_final_state=False,
            save_new_value=False,
            chunk_size=chunk_size,
            block_DK=64,
            block_DV=64,
            threads=256,
        )(k, w, u, g, initial_state)[0]


def torch_gdn_fwd_h(k, w, u, g, chunk_size):
    """
    Argument:
        k: (batch, seqlen, nheads, dhead)
        w: (batch, seqlen, nheads, dhead)
        u: (batch, seqlen, nheads, expand_v*dhead)
        g: (batch, seqlen, nheads)
        chunk_size: int
    Return:
        h: (batch, nchunks, nheads, dhead, expand_v*dhead)
    """

    batch, seqlen, nheads, dhead = k.shape
    expand_v = u.shape[-1] // dhead
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    acc_dtype = torch.float32
    dtype = k.dtype

    h = torch.empty(
        batch, nchunks, nheads, dhead, expand_v * dhead, dtype=k.dtype, device=k.device
    )
    b_h = torch.zeros(
        batch, nheads, dhead, expand_v * dhead, dtype=acc_dtype, device=k.device
    )

    k_c = k.reshape(batch, nchunks, chunk_size, nheads, dhead)
    w_c = w.reshape(batch, nchunks, chunk_size, nheads, dhead)
    u_c = u.reshape(batch, nchunks, chunk_size, nheads, expand_v * dhead)
    g_c = g.reshape(batch, nchunks, chunk_size, nheads)
    for i_t in range(nchunks):
        h[:, i_t, :, :, :] = b_h.to(dtype)
        b_w = w_c[:, i_t, :, :, :].to(acc_dtype)
        c_h = b_h.to(dtype).to(acc_dtype)
        b_v = torch.einsum("bchk,bhkv->bchv", b_w, c_h)
        p_v = u_c[:, i_t, :, :, :].to(acc_dtype)
        b_v = p_v - b_v
        last_idx = min((i_t + 1) * chunk_size, seqlen) - 1
        m_t = (i_t * chunk_size + torch.arange(0, chunk_size, device=k.device)) < seqlen
        b_g_last = g[:, last_idx, :].to(acc_dtype)
        b_g = g_c[:, i_t, :, :].to(acc_dtype)  # batch, chunk, nheads
        b_v *= torch.where(
            m_t.unsqueeze(0).unsqueeze(-1), torch.exp(b_g_last.unsqueeze(1) - b_g), 0
        ).unsqueeze(-1)
        b_g_last = torch.exp(b_g_last)
        b_h *= b_g_last.unsqueeze(-1).unsqueeze(-1)
        b_v = b_v.to(dtype).to(acc_dtype)
        p_k = k_c[:, i_t, :, :, :].to(acc_dtype)
        b_h += torch.einsum("bchk,bchv->bhkv", p_k, b_v)
    return h


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1, 16], help="Batch size"
    )
    parser.add_argument(
        "--n-heads", type=int, nargs="+", default=[6], help="Number of heads"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
        help="Sequence length",
    )
    parser.add_argument(
        "--chunk-size", type=int, nargs="+", default=[64], help="Chunk size"
    )
    parser.add_argument(
        "--d-head", type=int, nargs="+", default=[256], help="Head dimension"
    )
    parser.add_argument(
        "--expand-v", type=int, nargs="+", default=[2], help="V expansion"
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "speedup", "accuracy", "tflops", "gbps"]
    DEFAULT_METRICS = ["gbps"]
    FWD_ONLY = True

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.BATCH = args.batch
        self.NHEADS = args.n_heads
        self.SEQLEN = args.seq_len
        self.CHUNK_SIZE = args.chunk_size
        self.DHEAD = args.d_head
        self.EXPANDV = args.expand_v

    @register_benchmark(baseline=True)
    def eager(self, k, w, u, g, chunk_size):
        return lambda: torch_gdn_fwd_h(k, w, u, g, chunk_size)

    @register_benchmark()
    def compile(self, k, w, u, g, chunk_size):
        return lambda: torch.compile(
            torch_gdn_fwd_h, options={"emulate_precision_casts": True}
        )(k, w, u, g, chunk_size)

    @register_benchmark(enabled=HAS_FLA)
    def fla(self, k, w, u, g, chunk_size):
        return lambda: fla_chunk_gated_delta_rule_fwd_h(k, w, u, g, chunk_size)

    @register_benchmark(enabled=HAS_TILELANG)
    def tilelang(self, k, w, u, g, chunk_size):
        return lambda: tilelang_chunk_gated_delta_rule_fwd_h(k, w, u, g, chunk_size)

    def accuracy(self, fn, baseline_fn):
        """Override accuracy to use relaxed tolerance for float16."""
        output = fn()
        baseline_output = baseline_fn()

        # Check for NaN values
        if torch.isnan(output).any():
            return False

        try:
            # Using atol=2e-2 and rtol=1e-2 to provide some margin
            if output.dtype in [torch.bfloat16, torch.float16]:
                torch.testing.assert_close(
                    output,
                    baseline_output,
                    rtol=0.1,
                    atol=0.3,
                )
            else:
                torch.testing.assert_close(output, baseline_output)
            return True
        except Exception as e:
            return False

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        with FlopCounterMode() as flop_counter:
            out = torch_gdn_fwd_h(*example_inputs)
        return flop_counter.get_total_flops()

    @register_metric()
    def gbps(self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics) -> float:
        def nbytes(t):
            if isinstance(t, torch.Tensor):
                return t.numel() * t.element_size()
            else:
                return 0

        out = torch_gdn_fwd_h(*example_inputs)

        gb = (sum(nbytes(t) for t in example_inputs) + nbytes(out) // 8) / 1e9
        return gb / metrics.latency * 1e3

    def get_shape_iter(self) -> Generator:
        return itertools.product(
            self.BATCH,
            self.NHEADS,
            self.SEQLEN,
            self.CHUNK_SIZE,
            self.DHEAD,
            self.EXPANDV,
        )

    def get_input_iter(self) -> Generator:
        for (
            batch,
            nheads,
            seqlen,
            chunk_size,
            dhead,
            expand_v,
        ) in self.get_shape_iter():
            torch.manual_seed(
                hash((batch, nheads, seqlen, chunk_size, dhead, expand_v, 2))
            )
            nchunks = (seqlen + chunk_size - 1) // chunk_size
            k = torch.randn(
                batch, seqlen, nheads, dhead, dtype=torch.bfloat16, device="cuda"
            )
            w = torch.randn(
                batch,
                seqlen // chunk_size,
                chunk_size,
                nheads,
                dhead,
                dtype=torch.float32,
                device="cuda",
            )
            wu, ws, wv = torch.linalg.svd(w.permute(0, 1, 3, 2, 4), full_matrices=False)
            w = torch.einsum("bnhik,bnhkj->bnhij", wu, wv)
            w = (
                w.permute(0, 1, 3, 2, 4)
                .reshape(batch, seqlen, nheads, dhead)
                .to(torch.bfloat16)
            )
            u = torch.randn(
                batch,
                seqlen,
                nheads,
                expand_v * dhead,
                dtype=torch.bfloat16,
                device="cuda",
            )
            g = torch.cumsum(
                0.5
                * math.log(1 / dhead)
                * torch.rand(batch, seqlen, nheads, dtype=torch.float32, device="cuda"),
                dim=1,
            )
            yield k, w, u, g, chunk_size

    @register_x_val(label="(Batch, Heads, SeqLen, ChunkSize, Dhead, ExpandV)")
    def get_x_val(self, example_inputs) -> float:
        """
        Argument:
            k: (batch, seqlen, nheads, dhead)
            w: (batch, seqlen, nheads, dhead)
            u: (batch, seqlen, nheads, expand_v*dhead)
            g: (batch, seqlen, nheads)
        Return:
            h: (batch, nchunks, nheads, dhead, expand_v*dhead)
        """
        k, w, u, g, chunk_size = example_inputs
        batch, seqlen, nheads, dhead = k.shape
        expand_v = u.shape[-1] // dhead
        return (batch, nheads, seqlen, chunk_size, dhead, expand_v)
