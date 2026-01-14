# (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

"""
This file defines common math functions, sometimes relying on optimized PTX for performance. Note that the functions relying on PTX
will only be supported on NVIDIA GPUs
"""

from enum import Enum

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
from torch._inductor.runtime.triton_helpers import libdevice

try:
    # @manual=//triton:triton
    from triton.language.extra.libdevice import fast_dividef, fast_expf
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.cuda.libdevice import fast_dividef, fast_expf
    except ImportError:
        # pyre-ignore: Undefined import [21]
        # @manual=//triton:triton
        from triton.language.math import fast_dividef, fast_expf


HAS_FAST_TANH_INSTRUCTION = (
    torch.version.cuda is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9  # >= H100
)


# Don't change the order of the enum values, as they are used to index
# Only add new activation functions at the end of the enum
class Activation(str, Enum):
    Raw = "raw"
    GeLU = "gelu"
    FastGeLU = "fast_gelu"


# pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got `Type[Activation]`.
activation_to_int = {act: i for i, act in enumerate(Activation)}
int_to_activation = {i: act for act, i in activation_to_int.items()}


def activation_string_to_int(s: str):
    # If we dont support the activation, we default to raw
    # Need a better way to do this
    enum_val = (
        Activation(s) if s in Activation._value2member_map_ else Activation("raw")
    )
    return activation_to_int.get(enum_val)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))


@triton.jit
def gelu_grad(x):
    cdf = 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
    pdf = tl.exp(-0.5 * x * x) * 0.3989422804014327
    return cdf + x * pdf


if not HAS_FAST_TANH_INSTRUCTION:

    @triton.jit
    def sigmoid_approx_fp32(x):
        exp_neg_x = fast_expf(-x)
        return fast_dividef(1.0, 1.0 + exp_neg_x)

    @triton.jit
    def tanh_approx_fp32(x):
        return 2 * sigmoid_approx_fp32(2 * x) - 1.0

else:

    @triton.jit
    def tanh_approx_fp32(x):
        output = tl.inline_asm_elementwise(
            asm="""
            tanh.approx.f32 $0, $1;
            """,
            constraints="=r,r",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return output

    @triton.jit
    def sigmoid_approx_fp32(x):
        output = 0.5 * tanh_approx_fp32(0.5 * x) + 0.5
        return output


if not HAS_FAST_TANH_INSTRUCTION:

    @triton.jit
    def fast_gelu(x):
        k = 2.0 * 0.7978845608
        x_sq = x * x
        sigmoid_out = sigmoid_approx_fp32(x * (k + k * 0.044715 * x_sq))
        return x * sigmoid_out, (x_sq, sigmoid_out)

    @triton.jit
    def fast_gelu_grad(x, _intermediates=None):
        k = 2.0 * 0.7978845608
        if _intermediates is None:
            x_sq = x * x
            sigmoid_out = sigmoid_approx_fp32(x * (k + k * 0.044715 * x_sq))
        else:
            x_sq, sigmoid_out = _intermediates

        return (
            x
            * (
                (sigmoid_out - sigmoid_out * sigmoid_out)
                * (k + 2 * 0.1070322243 * x_sq)
            )
            + sigmoid_out
        )

else:

    @triton.jit
    def fast_gelu(x):
        k = 0.7978845608
        x_sq = x * x
        tanh_out = tanh_approx_fp32(x * (k + k * 0.044715 * x_sq))
        return x * 0.5 * (1 + tanh_out), (x_sq, tanh_out)

    @triton.jit
    def fast_gelu_grad(x, _intermediates=None):
        k = 0.7978845608
        if _intermediates is None:
            x_sq = x * x
            tanh_out = tanh_approx_fp32(x * (k + k * 0.044715 * x_sq))
        else:
            x_sq, tanh_out = _intermediates

        return 0.5 * x * (
            (1 - tanh_out * tanh_out) * (k + 0.1070322243 * x_sq)
        ) + 0.5 * (1 + tanh_out)
