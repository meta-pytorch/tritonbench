"""
Utilities for getting operator details.
"""

import sys
from typing import List, OrderedDict

from tritonbench.operators import load_opbench_by_name

from .triton_op import (
    BenchmarkOperatorBackend,
    OVERRIDDEN_METRICS,
    REGISTERED_BENCHMARKS,
    REGISTERED_METRICS,
)


def is_operator_loaded(operator_name: str) -> bool:
    """
    Check if an operator is loaded.

    Args:
        operator_name (str): The name of the operator.

    Returns:
        bool: True if the operator is loaded, False otherwise.
    """
    return operator_name in REGISTERED_BENCHMARKS.keys()


def get_all_loaded_operators() -> List[str]:
    """
    Get all loaded operators.

    Returns:
        List[str]: A list of operator names.
    """
    return list(REGISTERED_BENCHMARKS.keys())


def batch_load_operators(operators: List[str]) -> None:
    """
    Load a list of operators.

    Args:
        operators (List[str]): A list of operator names.
    """
    for op_name in operators:
        try:
            load_opbench_by_name(op_name)
        except Exception as e:
            print(f"Warning: Failed to load operator '{op_name}': {e}", file=sys.stderr)


def get_backends_for_operator(
    operator_name: str,
) -> OrderedDict[str, BenchmarkOperatorBackend]:
    """
    Get the backends of an operator.

    Args:
        operator_name (str): The name of the operator.

    Returns:
        OrderedDict[str, BenchmarkOperatorBackend]: A dictionary of backend names to backend configs.
    """
    if not is_operator_loaded(operator_name):
        raise ValueError(f"Operator {operator_name} is not loaded.")
    return REGISTERED_BENCHMARKS[operator_name]


def get_custom_metrics_for_operator(operator_name: str) -> List[str]:
    """Get custom metrics for specific operators."""
    if not is_operator_loaded(operator_name):
        raise ValueError(f"Operator {operator_name} is not loaded.")

    return REGISTERED_METRICS.get(operator_name, [])


def get_overridden_metrics_for_operator(operator_name: str) -> List[str]:
    """Get overridden metrics for specific operators."""
    if not is_operator_loaded(operator_name):
        raise ValueError(f"Operator {operator_name} is not loaded.")

    return OVERRIDDEN_METRICS.get(operator_name, [])
