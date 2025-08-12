"""A/B testing utilities for tritonbench."""

import argparse
import shlex
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

from .parser import get_parser

from .triton_op import BenchmarkOperatorResult, REGISTERED_X_VALS


@dataclass
class MetricComparison:
    """Data class to store metric comparison results."""
    val_a: float
    val_b: float
    improvement_pct: float
    x_val: Any
    backend: str
    metric: str


def _extract_metric_value(metric_obj: Any) -> Optional[float]:
    """Extract and normalize metric value from metric objects.
    
    Only handles:
    - Objects with p50 attribute (percentile-based metrics)
    - Direct numeric values (int/float)
    
    Skips all other types (tuples, complex objects, etc.)
    """
    if metric_obj is None:
        return None
    
    # Handle objects with p50 attribute (percentile-based metrics)
    if hasattr(metric_obj, "p50"):
        return float(metric_obj.p50)
    
    # Handle direct numeric values only
    if isinstance(metric_obj, (int, float)):
        return float(metric_obj)
    
    # Skip all other types (tuples, complex objects, etc.)
    return None


def _calculate_improvement(val_a: float, val_b: float) -> float:
    """Calculate percentage improvement from val_a to val_b."""
    if val_a == 0:
        return 0.0
    return ((val_b - val_a) / val_a) * 100


def _get_comparable_data_points(
    result_a: BenchmarkOperatorResult,
    result_b: BenchmarkOperatorResult,
    common_x_vals: List,
    common_backends: List[str],
    metric: str,
) -> List[MetricComparison]:
    """Get all comparable data points for a specific metric."""
    # Create result dictionaries for easier lookup
    result_dict_a = {x_val: metrics_dict for x_val, metrics_dict in result_a.result}
    result_dict_b = {x_val: metrics_dict for x_val, metrics_dict in result_b.result}
    
    comparisons = []
    
    for backend in common_backends:
        for x_val in common_x_vals:
            if backend in result_dict_a[x_val] and backend in result_dict_b[x_val]:
                metrics_a = result_dict_a[x_val][backend]
                metrics_b = result_dict_b[x_val][backend]
                
                # Try to get the metric from direct attribute first
                raw_val_a = getattr(metrics_a, metric, None)
                raw_val_b = getattr(metrics_b, metric, None)
                
                # If not found, check in extra_metrics
                if raw_val_a is None and hasattr(metrics_a, 'extra_metrics') and metrics_a.extra_metrics:
                    raw_val_a = metrics_a.extra_metrics.get(metric, None)
                if raw_val_b is None and hasattr(metrics_b, 'extra_metrics') and metrics_b.extra_metrics:
                    raw_val_b = metrics_b.extra_metrics.get(metric, None)
                
                val_a = _extract_metric_value(raw_val_a)
                val_b = _extract_metric_value(raw_val_b)
                
                if val_a is not None and val_b is not None:
                    improvement_pct = _calculate_improvement(val_a, val_b)
                    comparisons.append(MetricComparison(
                        val_a=val_a,
                        val_b=val_b,
                        improvement_pct=improvement_pct,
                        x_val=x_val,
                        backend=backend,
                        metric=metric
                    ))
    
    return comparisons


def parse_ab_config(config_str: str) -> List[str]:
    """Parse A/B configuration string into argument list."""
    if not config_str:
        return []
    try:
        return shlex.split(config_str)
    except ValueError as e:
        raise ValueError(f"Failed to parse configuration string '{config_str}': {e}")


def separate_global_and_op_args(config_args: List[str]) -> Tuple[List[str], List[str]]:
    """Separate global tritonbench args from operator-specific args."""
    if not config_args:
        return [], []

    # Use a temporary parser to identify known global arguments
    temp_parser = get_parser()

    global_args = []
    op_args = []
    i = 0

    while i < len(config_args):
        arg = config_args[i]

        if arg.startswith("--"):
            # Check if this is a known global argument
            arg_name = arg.split("=")[0]  # Handle --arg=value format

            # Check if the argument is known to the parser
            is_known = False
            for action in temp_parser._actions:
                if arg_name in action.option_strings:
                    is_known = True
                    break

            if is_known:
                # This is a global argument
                if "=" in arg:
                    # --arg=value format
                    global_args.append(arg)
                    i += 1
                else:
                    # --arg value format (might need next argument)
                    global_args.append(arg)
                    # Check if next argument is a value (not starting with -)
                    if i + 1 < len(config_args) and not config_args[i + 1].startswith(
                        "-"
                    ):
                        global_args.append(config_args[i + 1])
                        i += 2
                    else:
                        i += 1
            else:
                # This is an operator-specific argument
                if "=" in arg:
                    # --arg=value format
                    op_args.append(arg)
                    i += 1
                else:
                    # --arg value format (might need next argument)
                    op_args.append(arg)
                    # Check if next argument is a value (not starting with -)
                    if i + 1 < len(config_args) and not config_args[i + 1].startswith(
                        "-"
                    ):
                        op_args.append(config_args[i + 1])
                        i += 2
                    else:
                        i += 1
        else:
            # Positional argument or value - add to operator args
            op_args.append(arg)
            i += 1

    return global_args, op_args


def update_args_with_global(
    base_args: argparse.Namespace, global_args: List[str]
) -> argparse.Namespace:
    """Update base args with global arguments from A/B config."""
    if not global_args:
        return argparse.Namespace(**vars(base_args))

    # Create a copy of base args
    updated_args = argparse.Namespace(**vars(base_args))

    # Parse global args and update the namespace
    temp_parser = get_parser()
    try:
        parsed_globals, _ = temp_parser.parse_known_args(global_args)

        # Update the namespace with new global values
        for key, value in vars(parsed_globals).items():
            if value is not None and key not in ["side_a", "side_b"]:
                setattr(updated_args, key, value)

    except SystemExit as e:
        # If parsing fails, keep original args
        print(
            f"WARNING: Failed to parse global arguments {global_args}, using original args: {e}"
        )
    except Exception as e:
        print(f"WARNING: Unexpected error parsing global arguments {global_args}: {e}")

    return updated_args


def _analyze_config_differences(
    config_a_args: List[str], config_b_args: List[str]
) -> Dict[str, Tuple[str, str]]:
    """Analyze differences between two configurations."""

    # Parse arguments into dictionaries
    def parse_config_to_dict(args):
        config_dict = {}
        i = 0
        while i < len(args):
            if args[i].startswith("--"):
                key = args[i][2:]  # Remove --
                if "=" in args[i]:
                    # Format: --key=value
                    key, value = args[i][2:].split("=", 1)
                    config_dict[key] = value
                    i += 1
                elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                    # Format: --key value
                    config_dict[key] = args[i + 1]
                    i += 2
                else:
                    # Flag without value
                    config_dict[key] = "True"
                    i += 1
            else:
                i += 1
        return config_dict

    config_a = parse_config_to_dict(config_a_args)
    config_b = parse_config_to_dict(config_b_args)

    # Find differences
    differences = {}
    all_keys = set(config_a.keys()) | set(config_b.keys())

    for key in all_keys:
        val_a = config_a.get(key, "default")
        val_b = config_b.get(key, "default")
        if val_a != val_b:
            differences[key] = (val_a, val_b)

    return differences


def _get_all_comparable_data_points(
    result_a: BenchmarkOperatorResult,
    result_b: BenchmarkOperatorResult,
    common_x_vals: List,
    common_backends: List[str],
) -> Dict[str, List[MetricComparison]]:
    """Get all comparable data points for all metrics at once."""
    all_comparisons = {}
    
    for metric in result_a.metrics:
        all_comparisons[metric] = _get_comparable_data_points(
            result_a, result_b, common_x_vals, common_backends, metric
        )
    
    return all_comparisons


def _calculate_performance_summary(
    all_comparisons: Dict[str, List[MetricComparison]],
    common_backends: List[str],
) -> Dict[str, Dict[str, float]]:
    """Calculate performance summary statistics from pre-computed comparisons."""
    summary = {}

    for backend in common_backends:
        backend_summary = {}

        for metric, comparisons in all_comparisons.items():
            # Filter for current backend
            backend_comparisons = [c for c in comparisons if c.backend == backend]
            improvements = [c.improvement_pct for c in backend_comparisons]

            if improvements:
                backend_summary[metric] = {
                    "avg_improvement": sum(improvements) / len(improvements),
                    "min_improvement": min(improvements),
                    "max_improvement": max(improvements),
                    "count": len(improvements),
                }

        summary[backend] = backend_summary

    return summary


def compare_ab_results(
    result_a: BenchmarkOperatorResult,
    result_b: BenchmarkOperatorResult,
    config_a_args: List[str],
    config_b_args: List[str],
):
    """Compare A/B test results"""
    if not result_a or not result_b:
        print("\n[A/B Comparison] ERROR: One or both results are invalid")
        return

    # Check if both results have data
    if not result_a.result or not result_b.result:
        print("ERROR: No benchmark data available for comparison")
        return

    # Get common data for analysis
    x_vals_a = {x_val for x_val, _ in result_a.result}
    x_vals_b = {x_val for x_val, _ in result_b.result}
    common_x_vals = sorted(x_vals_a.intersection(x_vals_b))

    if not common_x_vals:
        print("ERROR: No common input shapes found between configurations")
        return

    # Get common backends
    result_dict_a = {x_val: metrics_dict for x_val, metrics_dict in result_a.result}
    result_dict_b = {x_val: metrics_dict for x_val, metrics_dict in result_b.result}

    all_backends_a = set()
    all_backends_b = set()
    for x_val in common_x_vals:
        all_backends_a.update(result_dict_a[x_val].keys())
        all_backends_b.update(result_dict_b[x_val].keys())
    common_backends = sorted(all_backends_a.intersection(all_backends_b))

    if not common_backends:
        print("ERROR: No common backends found between configurations")
        return

    # ============================================================================
    # PRE-COMPUTE: Get all comparable data points once
    # ============================================================================
    all_comparisons = _get_all_comparable_data_points(
        result_a, result_b, common_x_vals, common_backends
    )
    

    # ============================================================================
    # SECTION 1: Configuration Analysis
    # ============================================================================
    print("\n" + "=" * 70)
    print(f"A/B Test Results: {result_a.op_name}")
    print("=" * 70)

    print("Configuration Differences:")
    try:
        differences = _analyze_config_differences(config_a_args, config_b_args)

        if differences:
            for param, (val_a, val_b) in differences.items():
                print(f"  {param:<15}: {val_a:<15} → {val_b}")
        else:
            print("  No configuration differences detected")
    except Exception as e:
        print(f"  ERROR: Failed to analyze configuration differences: {e}")

    print(
        f"\nTest Scope: {len(common_x_vals)} input shapes, {len(common_backends)} backends"
    )
    print(f"Metrics: {', '.join(result_a.metrics)}")

    # ============================================================================
    # SECTION 2: Performance Summary
    # ============================================================================
    print("\n" + "-" * 70)
    print("Performance Summary")
    print("-" * 70)

    summary = _calculate_performance_summary(all_comparisons, common_backends)

    for backend in common_backends:
        print(f"\n{backend}:")
        backend_data = summary.get(backend, {})

        if not backend_data:
            print("  No comparable data")
            continue

        for metric, stats in backend_data.items():
            avg_improvement = stats["avg_improvement"]
            min_improvement = stats["min_improvement"]
            max_improvement = stats["max_improvement"]

            print(
                f"  {metric:<12}: {avg_improvement:+5.1f}% avg [{min_improvement:+.1f}% to {max_improvement:+.1f}%]"
            )

    # ============================================================================
    # SECTION 3: Detailed Comparison (Compact)
    # ============================================================================
    print("\n" + "-" * 70)
    print("Detailed Comparison")
    print("-" * 70)

    x_val_name = REGISTERED_X_VALS.get(result_a.op_name, "x_val")

    # Show only metrics that have comparable data
    for metric in result_a.metrics:
        if metric not in all_comparisons or len(all_comparisons[metric]) == 0:
            continue  # Skip metrics with no comparable data
        print(f"\nMetric: {metric}")
        print("Backend".ljust(15), end="")
        print(x_val_name.ljust(20), end="")
        print("Config A".ljust(12), end="")
        print("Config B".ljust(12), end="")
        print("Difference".ljust(12))
        print("-" * 71)

        # Use pre-computed comparisons
        comparisons = all_comparisons.get(metric, [])

        # Group by backend for display
        backend_comparisons = {}
        for comp in comparisons:
            if comp.backend not in backend_comparisons:
                backend_comparisons[comp.backend] = []
            backend_comparisons[comp.backend].append(comp)

        for backend in common_backends:
            if backend not in backend_comparisons:
                continue
                
            first_row = True
            for comp in backend_comparisons[backend]:
                # Format values
                if isinstance(comp.val_a, float):
                    val_a_str = f"{comp.val_a:.3f}"
                    val_b_str = f"{comp.val_b:.3f}"
                else:
                    val_a_str = str(comp.val_a)
                    val_b_str = str(comp.val_b)

                # Print row
                backend_name = backend if first_row else ""
                print(
                    f"{backend_name:<15}{str(comp.x_val):<20}{val_a_str:<12}{val_b_str:<12}{comp.improvement_pct:+5.1f}%"
                )
                first_row = False

            if not first_row:  # Only print separator if we printed data
                print()


def run_ab_test(
    base_args: argparse.Namespace, base_extra_args: List[str], _run_func
) -> Tuple[BenchmarkOperatorResult, BenchmarkOperatorResult]:
    """Run A/B test with two configurations and return both results."""

    # Parse A and B configurations
    try:
        config_a_args = parse_ab_config(base_args.side_a)
    except ValueError as e:
        print(f"ERROR: Failed to parse Side A configuration: {e}")
        raise

    try:
        config_b_args = parse_ab_config(base_args.side_b)
    except ValueError as e:
        print(f"ERROR: Failed to parse Side B configuration: {e}")
        raise

    print(f"[A/B Test] Configuration A: {' '.join(config_a_args)}")
    print(f"[A/B Test] Configuration B: {' '.join(config_b_args)}")

    # Separate global and operator-specific arguments
    global_a_args, op_a_args = separate_global_and_op_args(config_a_args)
    global_b_args, op_b_args = separate_global_and_op_args(config_b_args)

    if global_a_args:
        print(f"[A/B Test] Global args A: {' '.join(global_a_args)}")
    if op_a_args:
        print(f"[A/B Test] Operator args A: {' '.join(op_a_args)}")
    if global_b_args:
        print(f"[A/B Test] Global args B: {' '.join(global_b_args)}")
    if op_b_args:
        print(f"[A/B Test] Operator args B: {' '.join(op_b_args)}")
    print()

    # Update args with global parameters
    args_a = update_args_with_global(base_args, global_a_args)
    args_b = update_args_with_global(base_args, global_b_args)

    # Combine extra_args with operator-specific args only
    extra_args_a = base_extra_args + op_a_args
    extra_args_b = base_extra_args + op_b_args

    print("=" * 60)
    print(f"Running Side A: {' '.join(config_a_args)}")
    if global_a_args:
        print(f"  Global args: {' '.join(global_a_args)}")
    if op_a_args:
        print(f"  Operator args: {' '.join(op_a_args)}")
    print("=" * 60)

    try:
        result_a = _run_func(args_a, extra_args_a)
        if not result_a:
            raise RuntimeError("Side A returned empty result")
    except Exception as e:
        print(f"ERROR: Side A failed to run: {e}")
        raise RuntimeError(f"A/B test failed - Side A error: {e}")

    print("\n" + "=" * 60)
    print(f"Running Side B: {' '.join(config_b_args)}")
    if global_b_args:
        print(f"  Global args: {' '.join(global_b_args)}")
    if op_b_args:
        print(f"  Operator args: {' '.join(op_b_args)}")
    print("=" * 60)

    try:
        result_b = _run_func(args_b, extra_args_b)
        if not result_b:
            raise RuntimeError("Side B returned empty result")
    except Exception as e:
        print(f"ERROR: Side B failed to run: {e}")
        raise RuntimeError(f"A/B test failed - Side B error: {e}")

    return result_a, result_b
