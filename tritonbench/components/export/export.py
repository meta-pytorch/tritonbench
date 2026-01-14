"""
Serialize pickled tensors to directory.
"""

import pickle
from pathlib import Path
from typing import Any, Callable

from tritonbench.utils.input import input_cast


def get_input_gradients(inputs):
    all_input_grads = []
    input_cast(lambda x: True, lambda y: all_input_grads.append(y.grad), inputs)
    return all_input_grads


def export_data(
    x_val: str,
    inputs: Any,
    fn_mode: str,
    fn: Callable,
    export_type: str,
    export_dir: str,
):
    # pickle naming convention
    # x_<x_val>-input.pkl
    # x_<x_val>-<fn_name>-fwd-output.pkl
    # x_<x_val>-<fn_name>-bwd-grad.pkl
    assert export_dir, "Export dir must be specified."
    export_path = Path(export_dir)
    assert export_path.exists(), f"Export path {export_dir} must exist."
    if export_type == "input" or export_type == "both":
        input_file_name = f"x_{x_val}-input.pkl"
        input_file_path = export_path.joinpath(input_file_name)
        with open(input_file_path, "wb") as ifp:
            pickle.dump(inputs, ifp)
    if export_type == "output" or export_type == "both":
        if fn_mode == "fwd":
            output_type = "output"
            output = fn()
        elif fn_mode == "bwd":
            output_type = "grad"
            # output of the backward pass are the input gradients
            output = get_input_gradients(inputs)
        output_file_name = f"x_{x_val}-{fn._name}-{fn_mode}-{output_type}.pkl"
        output_file_path = export_path.joinpath(output_file_name)
        with open(output_file_path, "wb") as ofp:
            pickle.dump(output, ofp)
