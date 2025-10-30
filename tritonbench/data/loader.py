import importlib
import os
from pathlib import Path
from typing import Any

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


def get_input_loader(tritonbench_op: Any, op: str, input: str):
    assert (
        hasattr(tritonbench_op, "aten_op_name") or op in SUPPORTED_INPUT_OPS
    ), f"Unsupported op: {op}. "
    op_module = ".".join(tritonbench_op.__module__.split(".")[:-1])
    generator_module = importlib.import_module(op_module)
    input_loader_cls = generator_module.InputLoader
    if os.path.exists(input):
        input_file_path = Path(input)
    elif INPUT_CONFIG_DIR.joinpath(input).exists():
        input_file_path = INPUT_CONFIG_DIR.joinpath(input)
    elif INTERNAL_INPUT_CONFIG_DIR.joinpath(input).exists():
        input_file_path = INTERNAL_INPUT_CONFIG_DIR.joinpath(input)
    else:
        raise RuntimeError(f"Input file {input} does not exist.")
    input_loader = input_loader_cls(tritonbench_op, op, input_file_path)
    return input_loader.get_input_iter()
