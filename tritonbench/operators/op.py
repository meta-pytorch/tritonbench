import importlib
import os
import pathlib
from typing import Dict, List

import yaml

OPBENCH_DIR = "operators"
INTERNAL_OPBENCH_DIR = "fb"


def _dir_contains_file(dir, file_name) -> bool:
    names = map(lambda x: x.name, filter(lambda x: x.is_file(), dir.iterdir()))
    return file_name in names


def _is_internal_operator(op_name: str) -> bool:
    p = (
        pathlib.Path(__file__)
        .parent.parent.joinpath(OPBENCH_DIR)
        .joinpath(INTERNAL_OPBENCH_DIR)
        .joinpath(op_name)
    )
    if p.exists() and p.joinpath("__init__.py").exists():
        return True
    return False


def _list_opbench_paths() -> List[str]:
    p = pathlib.Path(__file__).parent.parent.joinpath(OPBENCH_DIR)
    # Only load the model directories that contain a "__init.py__" file
    opbench = sorted(
        str(child.absolute())
        for child in p.iterdir()
        if child.is_dir() and _dir_contains_file(child, "__init__.py")
    )
    p = p.joinpath(INTERNAL_OPBENCH_DIR)
    if p.exists():
        o = sorted(
            str(child.absolute())
            for child in p.iterdir()
            if child.is_dir() and _dir_contains_file(child, "__init__.py")
        )
        opbench.extend(o)
        opbench = [
            op for op in opbench if not os.path.basename(op) == INTERNAL_OPBENCH_DIR
        ]
    return opbench


def list_operators() -> List[str]:
    return [os.path.basename(y) for y in _list_opbench_paths()]


def list_custom_triton_operators(
    custom_triton_yamls: List[str],
) -> Dict[str, List[str]]:
    """List all custom Triton operators"""
    import torch

    def is_fbcode() -> bool:
        return not hasattr(torch.version, "git_version")

    triton_ops = {}
    for custom_triton_yaml in custom_triton_yamls:
        with open(custom_triton_yaml, "r") as fp:
            custom_triton_metadata = yaml.safe_load(fp)
        if not is_fbcode() and "oss_only" in custom_triton_metadata:
            triton_ops.update(custom_triton_metadata["oss_only"])
        if is_fbcode() and "internal" in custom_triton_metadata:
            triton_ops.update(custom_triton_metadata["internal"])
        if "triton_ops" in custom_triton_metadata:
            triton_ops.update(custom_triton_metadata["triton_ops"])
    return triton_ops


def load_opbench_by_name(op_name: str):
    opbench_list = filter(
        lambda x: op_name.lower() == x.lower(),
        map(lambda y: os.path.basename(y), _list_opbench_paths()),
    )
    opbench_list = list(opbench_list)
    if not opbench_list:
        raise RuntimeError(f"{op_name} is not found in the Tritonbench operator list.")
    assert len(opbench_list) == 1, (
        f"Found more than one operators {opbench_list} matching the required name: {op_name}"
    )
    op_name = opbench_list[0]
    op_pkg = (
        op_name
        if not _is_internal_operator(op_name)
        else f"{INTERNAL_OPBENCH_DIR}.{op_name}"
    )
    module = importlib.import_module(f"..{op_pkg}", package=__name__)

    Operator = getattr(module, "Operator", None)
    if Operator is None:
        print(f"Warning: {module} does not define attribute Operator, skip it")
        return None
    if not hasattr(Operator, "name"):
        Operator.name = op_name
    return Operator
