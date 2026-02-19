import glob as glob_module
import importlib
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from tritonbench.utils.env_utils import is_fbcode

SUPPORTED_INPUT_OPS = [
    "highway_self_gating",
    "grouped_gemm",
    "addmm",
    "bmm",
    "gemm",
    "jagged_dense_dense_sum",
    "fp8_gemm",
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


def expand_input_config_paths(input_pattern: str) -> List[Path]:
    """
    Expand a path pattern (potentially with wildcards) to a list of actual file paths.
    Supports wildcards like *, ?, etc.
    """
    has_wildcard = any(c in input_pattern for c in ["*", "?", "[", "]"])

    if not has_wildcard:
        return [get_input_config_path(input_pattern)]

    matches = list(glob_module.glob(input_pattern))
    if matches:
        return [Path(m) for m in sorted(matches)]

    config_pattern = str(INPUT_CONFIG_DIR / input_pattern)
    matches = list(glob_module.glob(config_pattern))
    if matches:
        return [Path(m) for m in sorted(matches)]

    if INTERNAL_INPUT_CONFIG_DIR:
        internal_pattern = str(INTERNAL_INPUT_CONFIG_DIR / input_pattern)
        matches = list(glob_module.glob(internal_pattern))
        if matches:
            return [Path(m) for m in sorted(matches)]

    raise RuntimeError(f"No input files found matching pattern: {input_pattern}")


def load_and_merge_input_configs(
    file_paths: List[Path],
) -> Tuple[dict, Optional[str]]:
    """
    Load and merge multiple input config JSON files.
    Validates that all files have the same tritonbench_ops.
    Returns merged config and the common loader type.
    """
    if len(file_paths) == 1:
        with open(file_paths[0], "r") as f:
            config = json.load(f)
        loader = None
        if "metadata" in config and "tritonbench_loader" in config["metadata"]:
            loader = config["metadata"]["tritonbench_loader"]
        return config, loader

    configs = []
    ops_sets = []
    loaders = []

    for path in file_paths:
        with open(path, "r") as f:
            config = json.load(f)
        configs.append((path, config))

        if "metadata" in config and "tritonbench_ops" in config["metadata"]:
            ops = tuple(sorted(config["metadata"]["tritonbench_ops"]))
        else:
            ops = tuple(sorted(k for k in config.keys() if k != "metadata"))
        ops_sets.append(ops)

        if "metadata" in config and "tritonbench_loader" in config["metadata"]:
            loaders.append(config["metadata"]["tritonbench_loader"])

    unique_ops = set(ops_sets)
    if len(unique_ops) > 1:
        error_details = "\n".join(
            f"  - {path}: {ops}" for path, ops in zip(file_paths, ops_sets)
        )
        raise RuntimeError(
            f"All input config files must have the same tritonbench_ops, but found different ops:\n{error_details}"
        )

    unique_loaders = set(loaders)
    if len(unique_loaders) > 1:
        raise RuntimeError(
            f"All input config files must use the same tritonbench_loader, but found: {unique_loaders}"
        )

    common_loader = loaders[0] if loaders else None

    merged_config = {}
    _, first_config = configs[0]
    if "metadata" in first_config:
        merged_config["metadata"] = first_config["metadata"].copy()

    for _, config in configs:
        for key, inputs_list in config.items():
            if key == "metadata":
                continue
            if key not in merged_config:
                merged_config[key] = []

            existing_inputs = {inp["inputs"] for inp in merged_config[key]}
            for inp in inputs_list:
                if inp["inputs"] not in existing_inputs:
                    merged_config[key].append(inp)
                    existing_inputs.add(inp["inputs"])

    total_inputs = sum(len(v) for k, v in merged_config.items() if k != "metadata")
    print(
        f"[input-loader] Merged {len(file_paths)} input config files with {total_inputs} total unique inputs"
    )

    return merged_config, common_loader


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
        input = "durin_20250402/jagged_dense_dense_sum.json" if not input else input

    input_file_paths = expand_input_config_paths(input)
    input_config, config_loader = load_and_merge_input_configs(input_file_paths)

    if config_loader:
        loader = config_loader

    if loader == "builtin":
        assert (
            hasattr(tritonbench_op, "aten_op_name") or op_name in SUPPORTED_INPUT_OPS
        ), f"Unsupported op by builtin loader: {op_name}. "

    if loader == "jagged" and is_fbcode():
        from .input_loaders.fb.jagged import InputLoader

        return InputLoader(tritonbench_op, input_config).get_input_iter()
    elif loader == "operator_warehouse" and is_fbcode():
        from .input_loaders.fb.operator_warehouse import (
            get_input_iter as get_input_iter_ow,
        )

        return get_input_iter_ow(tritonbench_op, [input_config])

    op_module = ".".join(tritonbench_op.__module__.split(".")[:-1])
    generator_module = importlib.import_module(op_module)
    input_loader_cls = generator_module.InputLoader
    if loader == "aten":
        operator_inputs_loader = input_loader_cls(op_name, input_config)
        return operator_inputs_loader.get_input_iter()
    elif loader == "builtin":
        input_loader = input_loader_cls(tritonbench_op, input_config)
        return input_loader.get_input_iter()
    else:
        raise ValueError(f"Unsupported input loader name: {loader}")
