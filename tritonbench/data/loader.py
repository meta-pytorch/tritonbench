import importlib
from pathlib import Path
from typing import Any

from tritonbench.utils.env_utils import is_fbcode

SUPPORTED_INPUT_OPS = ["highway_self_gating"]

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
    generator_module = importlib.import_module(f"{op_module}.input_loader")
    input_iter_getter = generator_module.get_input_iter
    input_iter = input_iter_getter(tritonbench_op, op, input)
    return input_iter
