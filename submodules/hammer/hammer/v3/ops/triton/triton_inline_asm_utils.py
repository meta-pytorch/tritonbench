# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
Inline-asm helpers for SM100+ (B200): packed f32x2 mul/fma and tanh.approx.

Shared between the standard and small-Q TLX block attention kernels. Callers
must guarantee SM100+ at dispatch time (see hammer/v3/ops/attention.py).
"""

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl


@triton.jit
def _tanh_approx_fp32(x):
    return tl.inline_asm_elementwise(
        asm="""
        tanh.approx.f32 $0, $1;
        """,
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_f32x2(a, b):
    # Packed multiply: result = a * b, two f32 lanes per instruction.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _fma_f32x2(a, b, c):
    # Packed FMA: result = a * b + c, two f32 lanes per instruction.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc, rd;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mov.b64 rc, { $6, $7 };
            fma.rn.f32x2 rd, ra, rb, rc;
            mov.b64 { $0, $1 }, rd;
        }
        """,
        "=r,=r,r,r,r,r,r,r",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _fast_silu_pre_halved(x_half):
    # silu(2*x_half) = x_half * (tanh(x_half) + 1) = fma(x_half, tanh(x_half), x_half)
    # Caller must have already multiplied input by 0.5 (typically by folding
    # the half into a per-row scale upstream).
    t = _tanh_approx_fp32(x_half)
    return _fma_f32x2(x_half, t, x_half)
