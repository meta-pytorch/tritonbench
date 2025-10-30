import importlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from tritonbench.utils.env_utils import is_fbcode

SUPPORTED_INPUT_OPS = [
    "highway_self_gating",
    "grouped_gemm",
    "addmm",
    "bmm",
    "gemm",
    "jagged_dense_dense_sum",
]

INPUT_CONFIG_DIR = Path(__file__).parent.joinpath("input_configs")
INTERNAL_INPUT_CONFIG_DIR = (
    importlib.resources.files("tritonbench.data.input_configs.fb")
    if is_fbcode()
    else None
)


def get_input_config_path(input_config_short_path: str):
    if os.path.exists(input_config_short_path):
        input_file_path = Path(input_config_short_path)
    elif INPUT_CONFIG_DIR.joinpath(input_config_short_path).exists():
        input_file_path = INPUT_CONFIG_DIR.joinpath(input_config_short_path)
    elif INTERNAL_INPUT_CONFIG_DIR.joinpath(input_config_short_path).exists():
        input_file_path = INTERNAL_INPUT_CONFIG_DIR.joinpath(input_config_short_path)
    else:
        raise RuntimeError(f"Input file {input_config_short_path} does not exist.")
    return input_file_path


def get_input_loader(
    tritonbench_op: Any, input: Optional[str] = None, loader="builtin"
):
    """Dispatch input loader based on op name and loader type."""
    op_name = (
        tritonbench_op.aten_op_name
        if hasattr(tritonbench_op, "aten_op_name")
        else tritonbench_op.name
    )

    if hasattr(tritonbench_op, "aten_op_name"):
        loader = "aten"
    if loader == "jagged":
        # default config for all jagged inputs
        input = "durin_20250402/jagged_dense_dense_sum.json" if not input else input
    input_file_path = get_input_config_path(input)

    with open(input_file_path, "r") as f:
        input_config = json.load(f)

    # Sanity checks
    if "metadata" in input_config and "tritonbench_loader" in input_config["metadata"]:
        loader = input_config["metadata"]["tritonbench_loader"]
    if loader == "builtin":
        assert (
            hasattr(tritonbench_op, "aten_op_name") or op_name in SUPPORTED_INPUT_OPS
        ), f"Unsupported op by builtin loader: {op_name}. "

    # Load jagged inputs
    if loader == "jagged" and is_fbcode():
        from .input_loaders.fb.jagged import InputLoader

        return InputLoader(tritonbench_op, input_config).get_input_iter()
    # Load operator warehouse inputs
    elif loader == "operator_warehouse" and is_fbcode():
        from .input_loaders.fb.operator_warehouse import (
            get_input_iter as get_input_iter_ow,
        )

        return get_input_iter_ow(tritonbench_op, [input_config])

    op_module = ".".join(tritonbench_op.__module__.split(".")[:-1])
    generator_module = importlib.import_module(op_module)
    input_loader_cls = generator_module.InputLoader
    # Load aten inputs
    if loader == "aten":
        operator_inputs_loader = input_loader_cls(op_name, input_config)
        return operator_inputs_loader.get_input_iter()
    # Load builtin inputs
    elif loader == "builtin":
        input_loader = input_loader_cls(tritonbench_op, input_config)
        return input_loader.get_input_iter()
    else:
        raise ValueError(f"Unsupported input loader name: {loader}")
