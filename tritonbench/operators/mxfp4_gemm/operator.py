"""TritonBench operator for MXFP4 GEMM on AMD gfx950 (MI350X).

All providers compute ``C[M, N] = A[M, K] @ B[N, K].T`` where A and B hold
E2M1 (FP4) data packed two values per byte, and every group of 32 values along
K carries one E8M0 scale byte. Quantization is not part of the timed region:
the operator yields already-quantized tensors, as a real inference stack would.

The baseline is ``torch._scaled_mm``, which is hipBLASLt's MXFP4 path on ROCm.
The TLX provider is the gfx950 inter-wave kernel from the Triton tutorials. It
dispatches internally between a 256x256 8-wave tile and a 128x128 + split-K
path for occupancy-starved (skinny-N) shapes, so one provider covers the whole
shape range.

Usage from fbsource/fbcode:
  HIP_VISIBLE_DEVICES=0 buck2 run @mode/opt-amd-gpu \
      -m ovr_config//triton:beta \
      -m ovr_config//third-party/rocm/constraints:7.0 \
      -m fbcode//triton/cc:force_noop \
      -c fbcode.enable_gpu_sections=true \
      -c fbcode.rocm_arch=mi350 \
      fbcode//pytorch/tritonbench:run -- \
      --op mxfp4_gemm \
      --metrics latency,tflops,speedup,accuracy
"""

import argparse
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
from tritonbench.utils.env_utils import is_hip_mi350
from tritonbench.utils.python_utils import try_import
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)
from tritonbench.utils.triton_utils import has_tlx

HAS_TLX_MXFP4 = False
with try_import("HAS_TLX_MXFP4"):
    from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a4w4.matmul_kernel import (
        matmul as tlx_mxfp4_matmul,
    )

SCALE_GROUP_SIZE = 32

# (M, N, K). K sweeps at a fixed MN tile count isolate the main-loop, and the
# small-M rows exercise the skinny dispatch inside the TLX kernel.
# The gfx950 kernel takes M, N multiples of 256 and K a multiple of 512 that is
# at least 1536 (the main loop prefetches six K-tiles). Shapes outside that are
# not benchmarked here; the provider skips anything the kernel rejects.
BENCHMARK_SHAPES = [
    (4096, 4096, 2048),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (4096, 4096, 16384),
    (2048, 8192, 4096),
    (256, 8192, 4096),
]

SMALL_SHAPES = [
    (1024, 1024, 2048),
    (2048, 2048, 2048),
]

SHAPE_SETS = {
    "benchmark": BENCHMARK_SHAPES,
    "small": SMALL_SHAPES,
}

# E2M1 code point -> value. Index is the raw 4-bit code (sign in bit 3).
_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]  # fmt: skip


def parse_op_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MXFP4 GEMM benchmark")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--shapes",
        type=str,
        default="benchmark",
        choices=list(SHAPE_SETS),
        help="Shape set to use",
    )
    return parser.parse_args(args)


def _mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Unpack (rows, K // 2) uint8 into (rows, K) float32."""
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=x.device)
    return values[x.long()]


def _e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, x.to(torch.int16).to(torch.float32) - 127.0)


def _dequantize(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return _mxfp4_to_f32(data) * _e8m0_to_f32(scales).repeat_interleave(
        SCALE_GROUP_SIZE, dim=1
    )


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "tflops", "speedup", "accuracy"]
    DEFAULT_PRECISION = "bf16"
    FWD_ONLY = True

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ) -> None:
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args or [])
        self.m, self.n, self.k = args.m, args.n, args.k
        self.shape_set = args.shapes

    def get_input_iter(self) -> Generator:
        """Yield BF16 sources, not quantized tensors.

        The providers do not agree on a scale layout -- hipBLASLt and the TLX
        kernel take the plain [rows, K/32] block scales, others want a shuffled
        variant -- so each provider quantizes for itself, outside the timed
        region. Handing every provider one layout would silently benchmark
        somebody's slow path.
        """
        if self.m is not None and self.n is not None and self.k is not None:
            shapes = [(self.m, self.n, self.k)]
        else:
            shapes = SHAPE_SETS[self.shape_set]

        for M, N, K in shapes:
            torch.manual_seed(42)
            # /8 keeps the values inside E2M1's tiny dynamic range so the
            # per-32 scales do not all saturate.
            a = torch.randn((M, K), device=self.device, dtype=torch.bfloat16) / 8
            b = torch.randn((N, K), device=self.device, dtype=torch.bfloat16) / 8
            yield a, b, M, N, K

    @staticmethod
    def _quantize(x: torch.Tensor):
        """MXFP4-quantize [rows, K] BF16 -> packed E2M1 + plain E8M0 block scales.

        Written out here rather than pulled from a vendor library so the
        hipBLASLt and TLX providers do not inherit a dependency they do not
        otherwise need. Standard MX rounding: one power-of-two scale per 32
        values, chosen so the group maximum lands on E2M1's largest magnitude
        (6.0), then round-half-to-even onto the eight E2M1 magnitudes.
        """
        rows, K = x.shape
        g = x.float().reshape(rows, K // SCALE_GROUP_SIZE, SCALE_GROUP_SIZE)
        amax = g.abs().amax(dim=-1, keepdim=True)
        # 2 = floor(log2(6.0)); the exponent that maps amax into E2M1's range.
        exp = torch.floor(torch.log2(amax.clamp(min=1e-30))) - 2
        exp = exp.clamp(-127, 127)
        scale = torch.pow(2.0, exp)
        v = (g / scale).clamp(-6.0, 6.0)

        mags = torch.tensor(
            _E2M1_VALUES[:8], dtype=torch.float32, device=x.device
        )  # 0 .. 6
        av = v.abs()
        idx = (av.unsqueeze(-1) - mags).abs().argmin(dim=-1)
        # argmin breaks ties toward the lower index, i.e. toward zero; MX
        # rounds half to even. The E2M1 code's low bit is its mantissa bit, so
        # "even" is an even index: on an exact midpoint, step an odd index up.
        upper = (idx + 1).clamp(max=len(_E2M1_VALUES[:8]) - 1)
        midpoint = (idx < upper) & (av * 2 == mags[idx] + mags[upper])
        idx = torch.where(midpoint & (idx % 2 == 1), upper, idx)
        codes = (idx + torch.where(v < 0, 8, 0)).to(torch.uint8)
        codes = codes.reshape(rows, K)
        packed = (codes[:, 1::2] << 4) | codes[:, 0::2]

        scales = (exp.squeeze(-1) + 127).to(torch.uint8)
        # Pad the scale rows to the 256-row tile the kernels index against.
        padded = ((rows + 255) // 256) * 256
        if padded != rows:
            scales = torch.nn.functional.pad(scales, (0, 0, 0, padded - rows))
        return packed.contiguous(), scales.contiguous()

    @register_benchmark()
    def torch_bf16(
        self, a: torch.Tensor, b: torch.Tensor, M: int, N: int, K: int
    ) -> Callable:
        """Un-quantized BF16 torch.mm.

        The exact-arithmetic anchor, not a performance target: quoting a speedup
        against a full-width BF16 GEMM would flatter any FP4 kernel. Its
        accuracy column reads 0 by construction -- it is compared against an
        FP4 baseline, and un-quantized BF16 is supposed to differ from it.
        """
        return lambda: torch.mm(a, b.T)

    @register_benchmark(baseline=True)
    def torch_scaled_mm(
        self, a: torch.Tensor, b: torch.Tensor, M: int, N: int, K: int
    ) -> Callable:
        """torch._scaled_mm, i.e. hipBLASLt's MXFP4 path.

        Plain block scales are the only layout this path accepts -- _scaled_mm_v2's SWIZZLE_32_4_4 is
        rejected on ROCm ("scale_a must not be swizzled"), so there is no
        faster hipBLASLt variant to reach for here.
        """
        a_q, a_s = self._quantize(a)
        b_q, b_s = self._quantize(b)
        try:
            a_fp4 = a_q.view(torch.float4_e2m1fn_x2)
            b_fp4 = b_q.view(torch.float4_e2m1fn_x2).T
            sa = a_s[:M].contiguous().view(torch.float8_e8m0fnu)
            sb = b_s[:N].contiguous().view(torch.float8_e8m0fnu)
            torch._scaled_mm(
                a_fp4, b_fp4, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16
            )
        # Only "this build or shape cannot do MXFP4" opts out: AttributeError
        # for a torch without the FP4 dtypes, RuntimeError for a shape or
        # layout hipBLASLt rejects. Anything else is a bug in this operator and
        # must surface -- this is the baseline, and a silent None empties the
        # speedup and accuracy columns for every other provider.
        except (AttributeError, RuntimeError):
            return None
        return lambda: torch._scaled_mm(
            a_fp4, b_fp4, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16
        )

    @register_benchmark(
        enabled=HAS_TLX_MXFP4 and has_tlx() and is_hip_mi350(),
        tags=["tlx", "amd", "gfx950"],
    )
    def tlx_mxfp4_gfx950(
        self, a: torch.Tensor, b: torch.Tensor, M: int, N: int, K: int
    ) -> Callable:
        """TLX MXFP4 GEMM for gfx950 (MI350X).

        Dispatches internally between a 256x256 8-wave tile and a 128x128 +
        split-K path for occupancy-starved shapes.

        The kernel asserts its own tile/K divisibility, and which assertions
        apply depends on the tile it dispatches to, so probe once rather than
        restating the dispatch here -- a copy would rot the first time the
        kernel's constraints move. Scales must be contiguous along M/N.
        """
        a_q, a_s = self._quantize(a)
        b_q, b_s = self._quantize(b)
        sa = a_s[:M].T.contiguous().T
        sb = b_s[:N].T.contiguous().T
        try:
            tlx_mxfp4_matmul(a_q, b_q, sa, sb)
        # The kernel states its tile/K divisibility as plain asserts, so an
        # unsupported shape arrives as AssertionError. Everything else is a
        # real failure and should not be hidden.
        except AssertionError:
            return None
        return lambda: tlx_mxfp4_matmul(a_q, b_q, sa, sb)

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, args: Tuple[Any, ...]) -> Tuple[int, int, int]:
        M, N, K = args[-3], args[-2], args[-1]
        return (M, N, K)

    @register_metric()
    def tflops(
        self,
        fn: Callable,
        example_inputs: Tuple[Any, ...],
        metrics: BenchmarkOperatorMetrics,
    ) -> float:
        M, N, K = example_inputs[-3], example_inputs[-2], example_inputs[-1]
        if metrics.latency == 0:
            return float("nan")
        return 2.0 * M * N * K / metrics.latency / 1e9

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        # FP4 has 3 mantissa bits at most, so the products are exactly
        # representable and the only divergence is FP32-vs-MFMA accumulation
        # order over K.
        output = fn().to(torch.float32)
        baseline = baseline_fn().to(torch.float32)
        return torch.allclose(output, baseline, atol=1e-2, rtol=1e-2)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        return None
