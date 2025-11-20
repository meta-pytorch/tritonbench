from typing import List

from tritonbench.operators_collection.all import get_all_operators
from tritonbench.operators_collection.aten import get_aten_operators
from tritonbench.operators_collection.default import (
    get_operators as get_default_operators,
)
from tritonbench.operators_collection.liger import get_liger_operators
from tritonbench.operators_collection.triton import get_triton_operators
from tritonbench.utils.env_utils import is_fbcode


OP_COLLECTION_PATH = "operators_collection"

OP_COLLECTIONS = {
    "all": get_all_operators,
    "aten": get_aten_operators,
    "liger": get_liger_operators,
    "default": get_default_operators,
    "triton": get_triton_operators,
}

if is_fbcode():
    from tritonbench.operators_collection.fb.buck import (
        get_operators as get_buck_operators,
    )

    OP_COLLECTIONS["buck"] = get_buck_operators


def list_operator_collections() -> List[str]:
    """
    List the available operator collections.

    This function retrieves the list of available operator collections by scanning the directories
    in the current path that contain an "__init__.py" file.

    Returns:
        List[str]: A list of names of the available operator collections.
    """
    return OP_COLLECTIONS.keys()


def list_operators_by_collection(op_collection: str = "default") -> List[str]:
    """
    List the operators from the specified operator collections.

    This function retrieves the list of operators from the specified operator collections.
    If the collection name is "all", it retrieves operators from all available collections.
    If the collection name is not specified, it defaults to the "default" collection.

    Args:
        op_collection (str): Names of the operator collections to list operators from.
        It can be a single collection name or a comma-separated list of names.
        Special value "all" retrieves operators from all collections.

    Returns:
        List[str]: A list of operator names from the specified collection(s).

    """
    return OP_COLLECTIONS[op_collection]()
