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
    from mamba_ssm.ops.triton.ssd_chunk_state import (
        chunk_state as mamba_ssm_chunk_state_fwd,
    )

    mamba_ssm_chunk_state_fwd = functools.partial(
        mamba_ssm_chunk_state_fwd, states_in_fp32=False
    )
    from mamba_ssm.ops.triton.ssd_chunk_state import (
        chunk_state_ref as mamba_ssm_chunk_state_ref,
    )

    HAS_MAMBA_SSM = True

with try_import("HAS_TILELANG"):
    import tilelang

    TL_ROOT = os.getenv("TL_ROOT", os.path.dirname(os.path.abspath(tilelang.__file__)))
    sys.path.append(f"{TL_ROOT}/examples/linear_attention")
    from example_mamba_chunk_state import (
        chunk_state_fwd as tilelang_example_chunk_state_fwd_kernel,
    )

    def tilelang_example_chunk_state_fwd(B, x, dt, dA_cumsum):
        batch, seqlen, ngroups, dstate = B.shape
        batch, seqlen, nheads, dhead = x.shape
        batch, nheads, nchunks, chunk_size = dt.shape
        batch, nheads, nchunks, chunk_size = dA_cumsum.shape
        return tilelang_example_chunk_state_fwd_kernel(
            batch, seqlen, chunk_size, ngroups, nheads, dhead, dstate
        )(B, x, dt, dA_cumsum)


def torch_chunk_state_fwd(B, x, dt, dA_cumsum):
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, dhead)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, dhead, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, dhead = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, dhead)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = torch.repeat_interleave(B, nheads // ngroups, dim=2)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = x.reshape(batch, nchunks, chunk_size, nheads, dhead)
    B = B.reshape(batch, nchunks, chunk_size, nheads, dstate)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    return torch.einsum(
        "bclhn,bhcl,bhcl,bclhp->bchpn",
        B.to(x.dtype),
        decay_states.to(x.dtype),
        dt.to(x.dtype),
        x,
    )


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
    def eager(self, B, x, dt, dA_cumsum):
        return lambda: torch_chunk_state_fwd(B, x, dt, dA_cumsum)

    @register_benchmark()
    def compile(self, B, x, dt, dA_cumsum):
        return lambda: torch.compile(torch_chunk_state_fwd)(B, x, dt, dA_cumsum)

    @register_benchmark(enabled=HAS_MAMBA_SSM)
    def mamba_ssm(self, B, x, dt, dA_cumsum):
        return lambda: mamba_ssm_chunk_state_fwd(B, x, dt, dA_cumsum)

    @register_benchmark(enabled=HAS_MAMBA_SSM)
    def mamba_ssm_ref(self, B, x, dt, dA_cumsum):
        return lambda: mamba_ssm_chunk_state_ref(B, x, dt, dA_cumsum)

    @register_benchmark(enabled=HAS_TILELANG)
    def tilelang(self, B, x, dt, dA_cumsum):
        return lambda: tilelang_example_chunk_state_fwd(B, x, dt, dA_cumsum)

    def _accuracy(self, fn, baseline_fn):
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
                    output, baseline_output, rtol=1e-2, atol=2e-2
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
        B, x, dt, dA_cumsum = example_inputs
        batch, seqlen, ngroups, dstate = B.shape
        batch, seqlen, nheads, dhead = x.shape
        batch, nheads, nchunks, chunk_size = dt.shape
        batch, nheads, nchunks, chunk_size = dA_cumsum.shape

        return 2 * batch * seqlen * nheads * dhead * dstate

    @register_metric()
    def gbps(self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics) -> float:
        def nbytes(t):
            return t.numel() * t.element_size()

        out = torch_chunk_state_fwd(*example_inputs)

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
            B = torch.rand(
                batch, seqlen, ngroups, dstate, dtype=self.dtype, device=self.device
            )
            x = torch.rand(
                batch, seqlen, nheads, dhead, dtype=self.dtype, device=self.device
            )
            dt = torch.rand(
                batch, nheads, nchunks, chunk_size, dtype=self.dtype, device=self.device
            )
            dA_cumsum = torch.rand(
                batch, nheads, nchunks, chunk_size, dtype=self.dtype, device=self.device
            )
            yield B, x, dt, dA_cumsum

    @register_x_val(label="(Batch, Heads, Groups, SeqLen, ChunkSize, Dhead, Dstate)")
    def get_x_val(self, example_inputs) -> float:
        B, x, dt, dA_cumsum = example_inputs
        batch, seqlen, ngroups, dstate = B.shape
        batch, seqlen, nheads, dhead = x.shape
        batch, nheads, nchunks, chunk_size = dt.shape
        batch, nheads, nchunks, chunk_size = dA_cumsum.shape
        return (batch, nheads, ngroups, seqlen, chunk_size, dhead, dstate)
