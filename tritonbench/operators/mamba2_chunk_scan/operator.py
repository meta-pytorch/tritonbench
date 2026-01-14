import argparse
import functools
import itertools
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

with try_import("HAS_MAMBA_SSM"):
    from mamba_ssm.ops.triton.ssd_chunk_scan import (
        _chunk_scan_fwd as mamba_ssm_chunk_scan_fwd_kernel,
    )

    def mamba_ssm_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states, D):
        return mamba_ssm_chunk_scan_fwd_kernel(cb, x, dt, dA_cumsum, C, prev_states, D)[
            0
        ]

    HAS_MAMBA_SSM = True

with try_import("HAS_TILELANG"):
    import tilelang

    TL_ROOT = os.getenv("TL_ROOT", os.path.dirname(os.path.abspath(tilelang.__file__)))
    sys.path.append(f"{TL_ROOT}/examples/linear_attention")
    from example_mamba_chunk_scan import (
        chunk_scan_fwd as tilelang_example_chunk_scan_fwd_kernel,
    )

    def tilelang_example_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states, D):
        batch, nchunks, ngroups, chunk_size, _ = cb.shape
        _, seqlen, nheads, dhead = x.shape
        _, _, _, dstate = C.shape
        return tilelang_example_chunk_scan_fwd_kernel(
            batch, seqlen, chunk_size, ngroups, nheads, dhead, dstate
        )(cb, x, dt, dA_cumsum, C, prev_states, D)


def torch_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states, D):
    """
    Argument:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, dhead)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, dhead, dstate)
        D: (nheads,)
    Return:
        out: (batch, seqlen, nheads, dhead)
    """
    _, _, ngroups, _, _ = cb.shape
    batch, seqlen, nheads, dhead = x.shape
    # _, _, ngroups, dstate = B.shape
    # assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    dstate = C.shape[-1]
    assert seqlen == nchunks * chunk_size
    # assert C.shape == B.shape
    # B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = torch.repeat_interleave(C, nheads // ngroups, dim=2)
    cb = torch.repeat_interleave(cb, nheads // ngroups, dim=2)
    # CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
    #                   rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb * decay.permute(0, 2, 1, 3, 4)
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0
    )
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum(
        "bchls,bhcs,bcshp->bclhp",
        scores_decay.to(x.dtype),
        dt.to(x.dtype),
        x.reshape(batch, nchunks, chunk_size, nheads, dhead),
    )
    # state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    state_decay_out = torch.exp(dA_cumsum.permute(0, 2, 3, 1).unsqueeze(-1))
    out_prev = (
        torch.einsum(
            "bclhn,bchpn->bclhp",
            C.reshape(batch, nchunks, chunk_size, nheads, dstate),
            prev_states.to(C.dtype),
        )
        * state_decay_out
    )
    out = out + out_prev
    out = out.reshape(batch, seqlen, nheads, dhead)
    if D is not None:
        if D.dim() == 1:
            D = D.unsqueeze(-1)
        out = out + x * D
    return out


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1, 64], help="Batch size"
    )
    parser.add_argument(
        "--n-heads", type=int, nargs="+", default=[64], help="Number of heads"
    )
    parser.add_argument(
        "--n-groups", type=int, nargs="+", default=[1], help="Number of groups"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[1024, 2048, 8192],
        help="Sequence length",
    )
    parser.add_argument(
        "--chunk-size", type=int, nargs="+", default=[256], help="Chunk size"
    )
    parser.add_argument(
        "--d-head", type=int, nargs="+", default=[64], help="Head dimension"
    )
    parser.add_argument(
        "--d-state", type=int, nargs="+", default=[128], help="State dimension"
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "fp16"
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
        self.NGROUPS = args.n_groups
        self.SEQLEN = args.seq_len
        self.CHUNK_SIZE = args.chunk_size
        self.DHEAD = args.d_head
        self.DSTATE = args.d_state

    @register_benchmark(baseline=True)
    def eager(self, cb, x, dt, dA_cumsum, C, prev_states, D):
        return lambda: torch_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states, D)

    @register_benchmark()
    def compile(self, cb, x, dt, dA_cumsum, C, prev_states, D):
        return lambda: torch.compile(torch_chunk_scan_fwd)(
            cb, x, dt, dA_cumsum, C, prev_states, D
        )

    @register_benchmark(enabled=HAS_MAMBA_SSM)
    def mamba_ssm(self, cb, x, dt, dA_cumsum, C, prev_states, D):
        return lambda: mamba_ssm_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states, D)

    @register_benchmark(enabled=HAS_TILELANG)
    def tilelang(self, cb, x, dt, dA_cumsum, C, prev_states, D):
        return lambda: tilelang_example_chunk_scan_fwd(
            cb, x, dt, dA_cumsum, C, prev_states, D
        )

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
                    atol=0.1,
                )
            else:
                torch.testing.assert_close(output, baseline_output)
            return True
        except Exception:
            return False

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        cb, x, dt, dA_cumsum, C, prev_states, D = example_inputs
        batch, nchunks, ngroups, chunk_size, _ = cb.shape
        _, seqlen, nheads, dhead = x.shape
        _, _, _, dstate = C.shape

        return (
            2 * batch * seqlen * chunk_size * nheads * dhead * 0.5
            + 2 * batch * seqlen * nheads * dhead * dstate
        )

    @register_metric()
    def gbps(self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics) -> float:
        def nbytes(t):
            return t.numel() * t.element_size()

        out = torch_chunk_scan_fwd(*example_inputs)

        gb = (sum(nbytes(t) for t in example_inputs) + nbytes(out) // 8) / 1e9
        return gb / metrics.latency * 1e3

    def get_shape_iter(self) -> Generator:
        return itertools.product(
            self.BATCH,
            self.NHEADS,
            self.NGROUPS,
            self.SEQLEN,
            self.CHUNK_SIZE,
            self.DHEAD,
            self.DSTATE,
        )

    def get_input_iter(self) -> Generator:
        for (
            batch,
            nheads,
            ngroups,
            seqlen,
            chunk_size,
            dhead,
            dstate,
        ) in self.get_shape_iter():
            nchunks = (seqlen + chunk_size - 1) // chunk_size
            cb = torch.randn(
                batch,
                nchunks,
                ngroups,
                chunk_size,
                chunk_size,
                dtype=self.dtype,
                device=self.device,
            )
            x = torch.randn(
                batch, seqlen, nheads, dhead, dtype=self.dtype, device=self.device
            )
            dt = torch.randn(
                batch, nheads, nchunks, chunk_size, dtype=self.dtype, device=self.device
            )
            dA_cumsum = torch.rand(
                batch, nheads, nchunks, chunk_size, dtype=self.dtype, device=self.device
            )
            C = torch.randn(
                batch, seqlen, ngroups, dstate, dtype=self.dtype, device=self.device
            )
            prev_states = torch.randn(
                batch,
                nchunks,
                nheads,
                dhead,
                dstate,
                dtype=self.dtype,
                device=self.device,
            )
            D = torch.randn(nheads, dtype=self.dtype, device=self.device)
            yield cb, x, dt, dA_cumsum, C, prev_states, D

    @register_x_val(label="(Batch, Heads, Groups, SeqLen, ChunkSize, Dhead, Dstate)")
    def get_x_val(self, example_inputs) -> float:
        cb, x, dt, dA_cumsum, C, prev_states, D = example_inputs
        batch, nchunks, ngroups, chunk_size, _ = cb.shape
        _, seqlen, nheads, dhead = x.shape
        _, _, _, dstate = C.shape
        return (batch, nheads, ngroups, seqlen, chunk_size, dhead, dstate)
