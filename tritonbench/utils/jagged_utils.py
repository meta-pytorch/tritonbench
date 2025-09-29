"""
Utils for nested (jagged) tensor operators
e.g. jagged_sum, jagged_mean
"""

import argparse
import itertools
import math
import random

import torch


GIGABYTES_PER_BYTE = 1e-9
RANDOM_CHOICE_MARGIN = 0.3
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-3
EPSILON = 1e-6

PARSER_ARGS = {
    "B": (
        "--B",
        int,
        "[Optional] Size of dimension 0 in shape (B, *, M) (integer)",
        None,
    ),
    "M": (
        "--M",
        int,
        "[Optional] Size of dimension 2 in shape (B, *, M) (integer)",
        None,
    ),
    "seqlen": (
        "--seqlen",
        int,
        "[Optional] Maximum sequence length on ragged dimension (integer)",
        None,
    ),
    "sparsity": (
        "--sparsity",
        float,
        "[Optional] Average sparsity for nested tensor (float, (0.0-1.0))",
        None,
    ),
    "sum_then_buffer": (
        "--sum-then-buffer",
        int,  # 1: sum then buffer, 0: buffer then sum
        "[Optional] For Triton kernels, determines whether to sum individual blocks then add to a buffer or add to a buffer then sum; 1: sum then buffer, 0: buffer then sum; default 0",
        0,
    ),
    "plot_benchmarks": (
        "--plot-benchmarks",
        str,
        "[Optional] Determines which benchmarks to plot: all, torch, triton",
        "all",
    ),
}

STYLES = [
    ("blue", "-"),
    ("red", "-"),
    ("orange", "-"),
    ("green", "-"),
    ("magenta", "-"),
    ("purple", "-"),
]


def get_parse_op_args(*args):
    parser = argparse.ArgumentParser()
    for arg in args:
        if arg not in PARSER_ARGS:
            raise ValueError(f"jagged_utils: {arg} not in PARSER_ARGS")
        parser.add_argument(
            PARSER_ARGS[arg][0],
            type=PARSER_ARGS[arg][1],
            help=PARSER_ARGS[arg][2],
            default=PARSER_ARGS[arg][3],
        )
    return parser


def get_tensor_bytes_limit(test_only):
    if test_only:
        return (
            5 * 1e7
        )  # allocate tensors no greater than 50MB when running concurrent tests
    return 8 * 1e9  # allocate tensors no greater than 8GB


def get_dim_vals(sizes):
    vals = []
    vals.extend([2**n for n in sizes])
    vals.extend(
        [
            (n - 1) * (n + 1)
            for n in sizes
            if n - 1 > 0 and (n - 1) * (n + 1) not in vals
        ]
    )
    return vals


def generate_input_vals(B, M, max_seqlen, sparsity, sizes):
    """
    Generate values for input parameters B, M, max_seqlen, sparsity for
    nested tensor of logical shape (B, *, M) with maximum sequence length
    `max_seqlen` along the ragged dimension `*` and average sparsity `sparsity
    """

    B_vals, M_vals, seqlen_vals, sparsity_vals = [], [], [], []

    if B is None:
        B_vals.extend(get_dim_vals(sizes))
    else:
        B_vals.extend([B])

    if M is None:
        M_vals.extend([512])
    else:
        M_vals.extend([M])

    if max_seqlen is None:
        seqlen_vals.extend([512])
    else:
        seqlen_vals.extend([max_seqlen])

    if sparsity is None:
        sparsity_vals.extend((0.1, 0.25, 0.5, 0.75, 0.9))
    else:
        sparsity_vals.extend([sparsity])

    return B_vals, M_vals, seqlen_vals, sparsity_vals


def jagged_to_nested_tensor(values: torch.Tensor, offsets: list[torch.Tensor]):
    """
    Convert jagged tensor (values + offsets) to torch.nested.nested_tensor

    Args:
        values: Compressed values tensor
        offsets: List of offset tensors, indicating the starting position of each sequence

    Returns:
        Tensor in torch.nested.nested_tensor format
    """
    # Calculate the length of each sequence
    lengths = []
    for i in range(len(offsets)):
        if i == 0:
            # For the first layer, calculate the length of each batch
            batch_size = offsets[i].size(0) - 1
            batch_lengths = []
            for b in range(batch_size):
                batch_lengths.append(offsets[i][b + 1] - offsets[i][b])
            lengths.append(batch_lengths)
        else:
            # For deeper levels of nesting
            prev_lengths = lengths[i - 1]
            curr_lengths = []
            offset_idx = 0
            for prev_len in prev_lengths:
                seq_lengths = []
                for _ in range(prev_len):
                    seq_lengths.append(
                        offsets[i][offset_idx + 1] - offsets[i][offset_idx]
                    )
                    offset_idx += 1
                curr_lengths.append(seq_lengths)
            lengths.append(curr_lengths)

    # Build tensor list based on lengths and values
    tensor_list = []
    start_idx = 0
    for b in range(len(lengths[0])):
        length = lengths[0][b]
        end_idx = start_idx + length
        tensor_list.append(values[start_idx:end_idx])
        start_idx = end_idx

    # Create nested tensor
    return torch.nested.nested_tensor(
        tensor_list, layout=torch.jagged, device=values.device, dtype=values.dtype
    )


def get_size_in_bytes(shape, dtype) -> int:
    num_elements = math.prod(shape)
    element_size = dtype.itemsize
    return math.floor(num_elements * element_size)


def generate_random_nested_tensors(
    B_vals,
    M_vals,
    seqlen_vals,
    sparsity_vals,
    device,
    dtype,
    TENSOR_BYTES_LIMIT=8 * 1e9,
    RANDOM_CHOICE_MARGIN=0.3,
):
    """
    Generate random nested tensors of shape (B, *, M), where * is the ragged dimension
    with maximum sequence length `max_seqlen` and average sparsity `sparsity`
    """

    vals = itertools.product(B_vals, M_vals, seqlen_vals, sparsity_vals)

    for B, M, max_seqlen, sparsity in vals:
        if (
            get_size_in_bytes((B, M, max_seqlen), dtype) < TENSOR_BYTES_LIMIT
        ):  # ensure that GPU memory is not exceeded
            tensors = []

            # greater sparsity --> shorter sequence lengths on ragged dimension
            seqlen_avg = math.floor(
                max_seqlen * (1 - sparsity)
            )  # average sequence length across all tensors in nested tensor
            seqlen_margin = math.floor(
                max_seqlen * RANDOM_CHOICE_MARGIN
            )  # use margin to constrain sequence lengths to range [seqlen_avg - seqlen_margin, seqlen_avg + seqlen_margin] to approximate an average sequence length, which correlates with sparsity

            lower = max(seqlen_avg - seqlen_margin, 1)
            upper = min(seqlen_avg + seqlen_margin, max_seqlen)
            seqlens = [random.randint(lower, upper) for _ in range(B)]
            tensor_2d = torch.randn((sum(seqlens), M), device=device, dtype=dtype)
            for seqlen in seqlens:
                t, tensor_2d = tensor_2d[:seqlen, :], tensor_2d[seqlen:, :]
                tensors.append(t)
            del tensor_2d

            nt = torch.nested.nested_tensor(
                tensors,
                layout=torch.jagged,
                device=device,
                dtype=dtype,
            )
            yield (nt, B, M, max_seqlen, sparsity)

    # add 0-seqlen nested tensor
    if (
        len(seqlen_vals) > 1
    ):  # variable seqlen, in which case injecting a 0-seqlen tensor of sparsity 0.5 will not change existing values in the plot
        tensors = [
            torch.randn((seqlen_vals[0], M), device=device, dtype=dtype),
            torch.randn((0, M), device=device, dtype=dtype),
            torch.randn((seqlen_vals[0] // 2, M), device=device, dtype=dtype),
        ]
        nt = torch.nested.nested_tensor(
            tensors,
            layout=torch.jagged,
            device=device,
            dtype=dtype,
        )
        yield (nt, 3, M, seqlen_vals[0], 0.5)


# plot helper functions


def get_param_fstrings(B, M, max_seqlen, sparsity):
    str_B, str_M, str_max_seqlen, str_sparsity = (
        f"-B-{B}",
        f"-M-{M}",
        f"-seqlen-{max_seqlen}",
        f"-sparsity-{sparsity}",
    )
    if B is None:
        x_axis = "B"
        params = str_M + str_max_seqlen + str_sparsity
    elif M is None:
        x_axis = "M"
        params = str_B + str_max_seqlen + str_sparsity
    elif max_seqlen is None:
        x_axis = "seqlen"
        params = str_B + str_M + str_sparsity
    else:
        x_axis = "sparsity"
        params = str_B + str_M + str_max_seqlen

    return x_axis, params


def get_styles(num_styles):
    return STYLES[:num_styles]


def get_plot_args(
    plot_benchmarks, num_torch, line_vals_all, line_names_all, styles_all
):
    if plot_benchmarks == "all":
        line_vals, line_names, styles = line_vals_all, line_names_all, styles_all
    elif plot_benchmarks == "torch":
        line_vals = line_vals_all[:num_torch]
        line_names = line_names_all[:num_torch]
        styles = styles_all[:num_torch]
    else:
        line_vals = line_vals_all[num_torch:]
        line_names = line_names_all[num_torch:]
        styles = styles_all[num_torch:]

    return line_vals, line_names, styles
