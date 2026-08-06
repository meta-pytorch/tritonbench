"""
Pure-PyTorch fallbacks for the FBGEMM MX4 quantization helpers.

The mx4 operators (fp32_to_mx4, mx4_to_fp32) benchmark the kernels used inside
fbgemm's quantize_comm. When fbgemm_gpu is not installed, importing
``fbgemm_gpu.quantize_utils`` fails and the operator modules cannot even be
loaded (their type annotations and input generators reference these symbols).

This module re-implements the public API of
``fbgemm_gpu/fbgemm_gpu/quantize_utils.py`` (``RoundingMode``, ``fp32_to_mx4``,
``mx4_to_fp32``) on top of the reference implementations in
``fbgemm_gpu/fbgemm_gpu/triton/quantize_ref.py`` and
``fbgemm_gpu/fbgemm_gpu/triton/common.py``, using nothing but torch. The
accelerated triton/cuda paths are intentionally not reproduced: this is only a
fallback so operators keep working (input generation, x_vals, accuracy) when
fbgemm is unavailable. The benchmarks themselves stay gated on HAS_FBGEMM.
"""

from enum import IntEnum

import torch

# We keep LUTs persistent to minimize the number of device copies required.
E2M1_LUT = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
E3M0_LUT = torch.tensor(
    [0, 0.25, 0.5, 1, 2, 4, 8, 16, -0, -0.25, -0.5, -1, -2, -4, -8, -16],
    dtype=torch.float32,
)


class RoundingMode(IntEnum):
    """Rounding options for quantization."""

    nearest = 0
    floor = 1
    even = 2
    stochastic = 3
    ceil = 4


def get_mx4_exp_bias(ebits: int) -> int:
    """Helper function to get the proper exponent bias for specified mx4 format.

    Args:
        ebits: The number of exponent bits in quantized format.

    Returns:
        The exponent bias for the specified mx4 format.
    """
    if ebits == 2:
        return 1
    elif ebits == 3:
        return 3
    else:
        raise NotImplementedError(f"MX4 with ebits={ebits} not supported.")


def get_mx4_lookup_table(ebits: int, mbits: int, device: torch.device) -> torch.Tensor:
    """Helper function to get the proper lookup table for specified mx4 format.

    Args:
        ebits: The number of exponent bits in quantized format.
        mbits: The number of mantissa bits in quantized format.
        device: The device that the LUT should be copied to.

    Returns:
        The lookup table for the specified mx4 format.
    """
    global E2M1_LUT, E3M0_LUT
    if ebits == 2 and mbits == 1:
        # Update state of LUT to minimize copies.
        if E2M1_LUT.device != device:
            E2M1_LUT = E2M1_LUT.to(device)
        return E2M1_LUT
    elif ebits == 3 and mbits == 0:
        # Update state of LUT to minimize copies.
        if E3M0_LUT.device != device:
            E3M0_LUT = E3M0_LUT.to(device)
        return E3M0_LUT
    else:
        raise NotImplementedError(
            f"MX4 with ebits={ebits} and mbits={mbits} not supported."
        )


def _compute_exp(
    group_max: torch.Tensor,
    rounding_mode: "RoundingMode | int",
    mbits: int,
) -> torch.Tensor:
    """Compute shared exponent of group using specified rounding mode.

    Args:
        group_max (Tensor): Group of values to compute exponent of.
        rounding_mode (int or RoundingMode): Which rounding mode to use.
        mbits (int): Number of mantissa bits in target mx4 format.

    Returns:
        Tensor: Shared exponent of group.
    """
    # Helpful constants.
    MBITS_FP32 = 23
    RAND_MASK = (1 << (MBITS_FP32 - mbits)) - 1
    # Nearest rounding mode.
    if rounding_mode == 0:
        return torch.floor(torch.log2(group_max) + 0.5)
    # Floor rounding mode.
    if rounding_mode == 1:
        return torch.floor(torch.log2(group_max))
    # Even pre-rounding mode.
    elif rounding_mode == 2:
        # First round to nearest even integer.
        M_ROUND = (1 << (MBITS_FP32 - mbits - 1)) - 1
        group_max = group_max.view(dtype=torch.int32) + M_ROUND
        # Then perform floor rounding of log.
        return torch.floor(torch.log2(group_max.view(dtype=torch.float32)))
    # Stochastic rounding mode.
    elif rounding_mode == 3:
        # Create random noise.
        rand_bits = torch.randint_like(group_max, high=2**31 - 1, dtype=torch.int32)
        # Add noise to group max and round down.
        group_max = group_max.view(dtype=torch.int32) + (RAND_MASK & rand_bits)
        # Now compute log and truncate.
        return torch.floor(torch.log2(group_max.view(dtype=torch.float32)))
    else:
        return torch.ceil(torch.log2(group_max))


def py_quantize_mx4(
    a: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: "RoundingMode | int" = RoundingMode.ceil,
    stochastic_casting: bool = False,
) -> torch.Tensor:
    """
    Quantize a tensor to mx4 format.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of exponent bits in target mx4 format.
        mbits (int): Number of mantissa bits in target mx4 format.
        rounding_mode (int or RoundingMode): Which type of rounding to use when
        calculating shared exponent.
        stochastic_casting (bool): Whether to use stochastic rounding when downcasting.

    Returns:
        torch.Tensor: [M / 2 + M / group_size] mx4 scaled tensor packed into in8
        with group exponents attached to each row.

        eg.
        Input with shape [1, 8192] will be quantized to [1, 4096 + 256] as
        each value contain two elements packed into an int8 and
        there are 32 groups in each row.
    """
    # Define helpful constants.
    FP32_MIN_NORMAL = 2 ** (-126)
    FP32_SIGN_OFFSET = 31
    SIGN_MASK = 0x1
    FP32_EXP_MASK = 0x7F800000
    FP32_EXP_OFFSET = 23
    FP32_MANTISSA_MASK = 0x007FFFFF
    # Set number of exponent bits and mantissa (plus implicit) bits.
    EBITS = ebits
    MBITS = mbits + 1
    # FP32 and and FP4 have very different exponent biases, adjust to fp4.
    FP32_EXP_BIAS = 127
    FP4_EXP_BIAS = get_mx4_exp_bias(EBITS)
    MAX_FP32_MANTISSA_BITS = 24
    RAND_MASK = (1 << (FP32_EXP_OFFSET - mbits)) - 1
    MANTISSA_OVERFLOW_THRESHOLD = (1 << MBITS) - 1
    EXPONENT_OVERFLOW_THRESHOLD = (1 << EBITS) - 1
    IMPLICIT_1_MASK = (1 << (MBITS - 1)) - 1

    # Make sure input has a supported shape.
    # If given an empty shape, return an empty tensor.
    if a.numel() == 0:
        return torch.empty(a.shape, device=a.device, dtype=torch.uint8)
    # Make sure input has a supported shape, if not pad each row.
    if a.shape[-1] % group_size != 0:
        pad = group_size - (a.shape[-1] % group_size)
        a = torch.nn.functional.pad(a, (0, pad))

    # Keep track of original shape.
    orig_shape = a.shape
    # Prepare for grouping by subdiving the last axis.
    a = a.view(a.numel() // group_size, group_size)
    # Now we can easily compute the shared exponents for each group.
    shared_exp, _ = torch.max(torch.abs(a), dim=1, keepdim=True)
    # Replace zero values with the minimum expressible normal value.
    shared_exp = torch.where(shared_exp == 0, FP32_MIN_NORMAL, shared_exp)
    # Convert max into an integer exponent.
    shared_exp = _compute_exp(shared_exp, rounding_mode, mbits)
    # Offset exponent by largest exponent in target datatype.
    shared_exp = shared_exp - EBITS
    # Restrict to range expressible as int8.
    shared_exp = torch.clamp(shared_exp, min=-127, max=125)
    # Convert exponent to scale and apply to input.
    # Need to do this calculation on cpu for accuracy.
    _shared_exp = shared_exp.cpu()
    scale = (2**_shared_exp).to(device=a.device)
    a = a / scale
    # View as integer for bitwise ops.
    a = a.view(torch.int32)

    # When doing ceiling rounding, we apply stochastic downcasting.
    if stochastic_casting:
        rand_bits = torch.randint_like(a, high=2**31 - 1, dtype=torch.int32)
        a = a + (rand_bits & RAND_MASK)

    # Quantization step: convert fp32 values to fp4.
    # Start by extracting float components.
    sign_bit = torch.bitwise_right_shift(a, FP32_SIGN_OFFSET).to(torch.int8)
    # Torch does arithmetic shifts so we need to isolate sign bit.
    sign_bit = torch.bitwise_and(sign_bit, SIGN_MASK)

    # Next extract exponent.
    biased_exp = torch.bitwise_and(a, FP32_EXP_MASK)
    # Shift exponent over to least significant bits.
    biased_exp = torch.bitwise_right_shift(biased_exp, FP32_EXP_OFFSET).to(torch.int8)

    # Finally extract the mantissa.
    trailing_mantissa = torch.bitwise_and(a, FP32_MANTISSA_MASK)
    new_biased_exp = biased_exp - FP32_EXP_BIAS + FP4_EXP_BIAS

    # Compute difference between ideal exponent and what can be represented.
    exp_diff = torch.where(new_biased_exp <= 0, 1 - new_biased_exp, 0)
    # Clip this difference to the maximum number of fp32 mantissa bits (23 + implicit).
    exp_diff = torch.clamp(exp_diff, max=MAX_FP32_MANTISSA_BITS)

    # Now perform mantissa rounding down to fp4.
    is_subnorm = biased_exp == 0
    # Add implied 1 to normal values.
    mantissa = torch.where(is_subnorm, trailing_mantissa, trailing_mantissa + (1 << 23))
    # Compute base number of bits corresponding to the mantissa. We use a smaller value
    # for subnorms since implicit one is included in exp_diff above.
    fp32_sig_bits = torch.where(is_subnorm, 23, 24).to(torch.int32)
    # Shift down to target bitwidth - 1 and efficiently represent.
    mantissa = torch.bitwise_right_shift(
        mantissa, fp32_sig_bits + exp_diff - MBITS - 1
    ).to(torch.int8)
    # Perform rounding by adding 1 then shifting down.
    mantissa = mantissa + 1
    mantissa = torch.bitwise_right_shift(mantissa, 1)

    # Check for overflow and adjust exponent accordingly.
    overflow = mantissa > MANTISSA_OVERFLOW_THRESHOLD
    # Allow subnorms to overflow into normals, otherwise shift off overflow.
    mantissa = torch.where(
        torch.bitwise_and(overflow, torch.bitwise_not(is_subnorm)),
        torch.bitwise_right_shift(mantissa, 1),
        mantissa,
    )
    # Special case where a value is subnorm and has a large mantissa, overflow it.
    new_biased_exp = torch.where(
        torch.bitwise_and(new_biased_exp <= 0, mantissa == 2), 1, new_biased_exp
    )
    # Remove implicit 1.
    mantissa = torch.bitwise_and(mantissa, IMPLICIT_1_MASK)
    # Add overflow to exponent.
    new_biased_exp = torch.where(overflow, new_biased_exp + 1, new_biased_exp)
    # If exp overflows, set mantissa so we're at max representable value.
    mantissa = torch.where(new_biased_exp > EXPONENT_OVERFLOW_THRESHOLD, 1, mantissa)

    # Construct fp4 value from components.
    new_biased_exp = torch.clamp(new_biased_exp, min=0, max=EXPONENT_OVERFLOW_THRESHOLD)
    mx4_value = torch.bitwise_or(
        torch.bitwise_left_shift(new_biased_exp, MBITS - 1), mantissa
    )
    mx4_value = torch.bitwise_or(
        torch.bitwise_left_shift(sign_bit, EBITS + MBITS - 1), mx4_value
    )

    # Pack int4 values into single int8 outputs.
    low_mx4 = mx4_value[:, ::2]
    high_mx4 = mx4_value[:, 1::2]
    high_mx4 = torch.bitwise_left_shift(high_mx4, 4)
    packed_mx4 = torch.bitwise_or(low_mx4, high_mx4)

    # Ravel packed values together with shared exponent.
    packed_mx4 = torch.concat(
        [
            packed_mx4.view(-1, group_size // 2),
            (shared_exp + FP32_EXP_BIAS).to(torch.int8).view(-1, 1),
        ],
        dim=1,
    )

    # Inputs are now fully quantized and ready to return.
    # Try to return in the original shape if possible.
    if orig_shape[-1] % group_size == 0:
        output_shape = list(orig_shape[:-1]) + [-1]
        return packed_mx4.view(output_shape).view(torch.uint8)
    # If we cant, return as a flat array.
    else:
        return packed_mx4.view(-1).view(torch.uint8)


def py_dequantize_mx4(
    a: torch.Tensor, group_size: int = 32, ebits: int = 2, mbits: int = 1
) -> torch.Tensor:
    """
    Dequantize a tensor from mx4 format to fp32.

    Args:
        a (Tensor): [M / 2 + M / group_size] MX4 tensor packed into int8 values
        with group exponents attached to end of each row.
        group_size (int): Size of chunks that use the same shared exponent.
        ebits (int): Number of exponent bits in target mx4 format.
        mbits (int): Number of mantissa bits in target mx4 format.

    Returns:
        torch.Tensor: [M] dequantized fp32 tensor.
    """
    # If given an empty shape, return an empty tensor.
    if a.numel() == 0:
        return torch.empty(a.shape, device=a.device, dtype=torch.float32)
    # Keep track of starting shape.
    orig_shape = a.shape
    device = a.device
    # Unravel packed inputs from shared exponents.
    a = a.view(-1, (group_size // 2) + 1).view(torch.int8)
    num_groups = a.numel() // ((group_size // 2) + 1)
    packed_input = a[:, :-1]
    shared_exp = a[:, -1:]
    # Remove fp32 exponent bias
    FP32_EXP_BIAS = 127
    shared_exp = shared_exp - FP32_EXP_BIAS

    # Pull out high and low mx4 values.
    FP4_BIT_MASK = 0xF
    low_mx4 = torch.bitwise_and(packed_input, FP4_BIT_MASK)
    high_mx4 = torch.bitwise_right_shift(packed_input, 4)
    # Remove sign bit from high values since shift was arithmetic.
    high_mx4 = torch.bitwise_and(high_mx4, FP4_BIT_MASK)
    # Recombine into a single tensor.
    a = torch.stack([low_mx4, high_mx4], dim=0).view(2, -1).t().contiguous()

    # Use a lookup table to convert
    mx4_to_fp_values = get_mx4_lookup_table(ebits, mbits, device)
    # Convert values into float32 equivalent via lookup.
    out = torch.index_select(mx4_to_fp_values, 0, a.to(torch.int32).view(-1))

    # Exponent needs to be computed on cpu for perfect precision.
    _shared_exp = shared_exp.cpu().to(torch.float)
    scale = (2**_shared_exp).to(device)

    # Finally, apply shared exponent to restore full value.
    out = out.view(-1, num_groups, group_size) * scale.view(1, num_groups, 1)
    # Restore original shape and return.
    out_shape = list(orig_shape[:-1]) + [-1]
    return out.view(out_shape)


def fp32_to_mx4(
    tensor: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: "RoundingMode | int | None" = RoundingMode.even,
    stochastic_casting: bool = False,
    use_triton: bool = True,
) -> torch.Tensor:
    """Quantize an FP32 tensor to MX4.

    Mirrors ``fbgemm_gpu.quantize_utils.fp32_to_mx4``, but always takes the
    reference (pure torch) path since the accelerated fbgemm kernels are not
    available here. ``use_triton`` is accepted for API compatibility only.

    Args:
        tensor (torch.Tensor): FP32 tensor to quantize with M total elements.
        group_size (int): Compute scale in chunks of group_size.
        ebits (int): Number of exponent bits in target mx4 format.
        mbits (int): Number of mantissa bits in target mx4 format.
        rounding_mode (RoundingMode or int): Which type of rounding to use when
        computing exponent.
        stochastic_casting (bool): Whether to use stochastic casting when downcasting.
        use_triton (bool): Ignored, kept to match the fbgemm signature.

    Return:
        output: MX4 tensor packed into int8 values with total elements
        (M / 2 + M / groupsize)
    """
    if rounding_mode is None:
        rounding_mode = RoundingMode.even

    return py_quantize_mx4(
        tensor,
        group_size,
        ebits=ebits,
        mbits=mbits,
        rounding_mode=rounding_mode,
        stochastic_casting=stochastic_casting,
    )


def mx4_to_fp32(
    tensor: torch.Tensor,
    group_size: int = 32,
    use_triton: bool = True,
    ebits: int = 2,
    mbits: int = 1,
) -> torch.Tensor:
    """Dequantize an MX4 tensor to FP32.

    Mirrors ``fbgemm_gpu.quantize_utils.mx4_to_fp32``, but always takes the
    reference (pure torch) path. ``use_triton`` is accepted for API
    compatibility only.

    Args:
        tensor (torch.Tensor): MX4 packed tensor with total elements
        (M / 2 + M / groupsize)
        group_size (int): Compute scale in chunks of group_size.
        use_triton (bool): Ignored, kept to match the fbgemm signature.
        ebits (int): Number of exponent bits in target mx4 format.
        mbits (int): Number of mantissa bits in target mx4 format.

    Return:
        output: FP32 tensor with total elements (M).
    """
    return py_dequantize_mx4(tensor, group_size, ebits=ebits, mbits=mbits)
