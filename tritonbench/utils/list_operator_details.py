"""
Utilities for listing operator details including metrics and backends in tritonbench.
"""

from dataclasses import fields
from typing import Dict, List, Optional

# Indentation constants for consistent formatting
INDENT = "  "  # Base indentation unit (2 spaces)
INDENT2 = INDENT * 2  # Double indentation (4 spaces)

from tritonbench.utils.operator_utils import (
    batch_load_operators,
    get_backends_for_operator,
    get_custom_metrics_for_operator,
    get_overridden_metrics_for_operator,
    is_operator_loaded,
)
from tritonbench.utils.triton_op import BenchmarkOperatorMetrics


def get_builtin_metrics() -> List[str]:
    """Get all built-in metrics from BenchmarkOperatorMetrics dataclass."""
    return [
        field.name
        for field in fields(BenchmarkOperatorMetrics)
        if field.name != "extra_metrics"
    ]


def get_backends_for_operators(operators: List[str]) -> Dict[str, Dict[str, Dict]]:
    """Get backends for specific operators. Assumes operators are already loaded."""
    result = {}
    for op_name in operators:
        if not is_operator_loaded(op_name):
            continue

        backends = get_backends_for_operator(op_name)
        result[op_name] = {}

        for backend_name, backend_config in backends.items():
            result[op_name][backend_name] = {
                "name": backend_config.name,
                "label": backend_config.label,
                "baseline": backend_config.baseline,
                "enabled": backend_config.enabled,
                "fwd_only": backend_config.fwd_only,
                "ci": backend_config.ci,
            }

    return result


def get_metrics_for_operators(operators: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Get metrics for specific operators. Assumes operators are already loaded."""
    result = {}
    for op_name in operators:
        if not is_operator_loaded(op_name):
            continue

        result[op_name] = {
            "custom": get_custom_metrics_for_operator(op_name),
            "overridden": get_overridden_metrics_for_operator(op_name),
        }
    return result


def format_builtin_metrics_header(builtin_metrics: List[str]) -> List[str]:
    """Format the built-in metrics header section."""
    output = ["Built-in metrics (available for all operators):"]
    for metric in sorted(builtin_metrics):
        output.append(f"{INDENT}{metric}")
    return output


def format_backend_entry(backend_name: str, backend_info: Dict[str, any]) -> str:
    """Format a single backend entry with status indicators."""
    label = backend_info["label"]

    # Build status indicators
    status_indicators = []
    if backend_info["baseline"]:
        status_indicators.append("baseline")
    if not backend_info["enabled"]:
        status_indicators.append("disabled")
    if backend_info["fwd_only"]:
        status_indicators.append("fwd_only")
    if not backend_info["ci"]:
        status_indicators.append("no_ci")

    status_str = f" [{', '.join(status_indicators)}]" if status_indicators else ""

    if label != backend_name:
        return f"{INDENT2}{backend_name} (label: {label}){status_str}"
    else:
        return f"{INDENT2}{backend_name}{status_str}"


def format_metrics_section(
    op_name: str, metrics_data: Dict[str, Dict[str, List[str]]]
) -> List[str]:
    """Format the metrics section for a single operator."""
    output = []

    if op_name not in metrics_data:
        return output

    custom = metrics_data[op_name].get("custom", [])
    overridden = metrics_data[op_name].get("overridden", [])

    if custom:
        output.append(f"{INDENT}Custom metrics:")
        for metric in sorted(custom):
            output.append(f"{INDENT2}{metric}")

    if overridden:
        output.append(f"{INDENT}Overridden metrics:")
        for metric in sorted(overridden):
            output.append(f"{INDENT2}{metric}")

    return output


def format_backends_section(
    op_name: str, backends_data: Dict[str, Dict[str, Dict]]
) -> List[str]:
    """Format the backends section for a single operator."""
    output = []

    if op_name not in backends_data or not backends_data[op_name]:
        return output

    output.append(f"{INDENT}Backends:")
    backends = backends_data[op_name]

    for backend_name in sorted(backends.keys()):
        backend_info = backends[backend_name]
        output.append(format_backend_entry(backend_name, backend_info))

    return output


def format_operator_details(
    operators: List[str],
    builtin_metrics: List[str],
    metrics_data: Dict[str, Dict[str, List[str]]],
    backends_data: Dict[str, Dict[str, Dict]],
    show_metrics: bool,
    show_backends: bool,
) -> str:
    """Unified formatter for operator details with metrics and/or backends."""
    output = []

    # Add built-in metrics header if showing metrics
    if show_metrics:
        output.extend(format_builtin_metrics_header(builtin_metrics))

    # Process each operator
    for op_name in sorted(operators):
        # Check if operator has any relevant data
        has_metrics = (
            show_metrics
            and op_name in metrics_data
            and (
                metrics_data[op_name].get("custom", [])
                or metrics_data[op_name].get("overridden", [])
            )
        )
        has_backends = (
            show_backends and op_name in backends_data and backends_data[op_name]
        )

        if not has_metrics and not has_backends:
            continue

        # Add operator header
        output.append(f"\nOperator: {op_name}")

        # Add metrics section if requested and available
        if show_metrics:
            output.extend(format_metrics_section(op_name, metrics_data))

        # Add backends section if requested and available
        if show_backends:
            output.extend(format_backends_section(op_name, backends_data))

    return "\n".join(output)


def format_backends_output(
    operators: List[str],
    backends_data: Dict[str, Dict[str, Dict]],
) -> str:
    """Format backends output for specific operators."""
    return format_operator_details(
        operators=operators,
        builtin_metrics=[],
        metrics_data={},
        backends_data=backends_data,
        show_metrics=False,
        show_backends=True,
    )


def format_metrics_output(
    operators: List[str],
    builtin_metrics: List[str],
    metrics_data: Dict[str, Dict[str, List[str]]],
) -> str:
    """Format metrics output for specific operators."""
    return format_operator_details(
        operators=operators,
        builtin_metrics=builtin_metrics,
        metrics_data=metrics_data,
        backends_data={},
        show_metrics=True,
        show_backends=False,
    )


def format_combined_output(
    operators: List[str],
    builtin_metrics: List[str],
    metrics_data: Dict[str, Dict[str, List[str]]],
    backends_data: Dict[str, Dict[str, Dict]],
) -> str:
    """Format combined metrics and backends output for specific operators."""
    return format_operator_details(
        operators=operators,
        builtin_metrics=builtin_metrics,
        metrics_data=metrics_data,
        backends_data=backends_data,
        show_metrics=True,
        show_backends=True,
    )


def list_operator_details(
    operators: Optional[List[str]] = None,
    show_metrics: bool = False,
    show_backends: bool = False,
) -> str:
    """
    List operator details (metrics and/or backends) based on the provided operators.

    Args:
        operators: List of specific operators to show details for
        show_metrics: Whether to show metrics information
        show_backends: Whether to show backends information

    Returns:
        Formatted string with operator details
    """
    if not show_metrics and not show_backends:
        return "No details requested. Use --list-metrics and/or --list-backends."

    builtin_metrics = get_builtin_metrics()

    if operators:
        # Load operators once before getting their details
        batch_load_operators(operators)

        # Specific operators case
        metrics_data = get_metrics_for_operators(operators) if show_metrics else {}
        backends_data = get_backends_for_operators(operators) if show_backends else {}

        if show_metrics and show_backends:
            return format_combined_output(
                operators, builtin_metrics, metrics_data, backends_data
            )
        elif show_metrics:
            return format_metrics_output(operators, builtin_metrics, metrics_data)
        elif show_backends:
            return format_backends_output(operators, backends_data)
    else:
        # Global case
        if show_metrics and show_backends:
            # For global case with both, just show built-in metrics
            output = []
            output.append("Built-in metrics (available for all operators):")
            for metric in sorted(builtin_metrics):
                output.append(f"{INDENT}{metric}")
            output.append(
                "\nNote: Use --op or --op-collection to show operator-specific details."
            )
            return "\n".join(output)
        elif show_metrics:
            # Global metrics case - show built-in metrics only
            output = []
            output.append("Built-in metrics (available for all operators):")
            for metric in sorted(builtin_metrics):
                output.append(f"{INDENT}{metric}")
            return "\n".join(output)
        elif show_backends:
            # Global backends case - no global backends exist
            return "No global backends available. Use --op or --op-collection to show operator-specific backends."
