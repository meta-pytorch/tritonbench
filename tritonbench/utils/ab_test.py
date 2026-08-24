"""A/B testing utilities for tritonbench.

Side B is optional: with only ``--side-a`` we run that single configuration and
report the statistical analysis of its latency samples, which is useful to
gauge how noisy a configuration is before comparing anything against it.
"""

import argparse
import json
import logging
import math
import shlex
import statistics
import sys
from dataclasses import asdict
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

from tritonbench.components.do_bench.latency_analysis import (
    analyze_latency,
    format_latency_analysis,
    LatencyAnalysis,
    MIN_SAMPLE,
)

from .parser import get_parser
from .triton_op import BenchmarkOperatorResult, REGISTERED_X_VALS

logger = logging.getLogger(__name__)


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


def _iter_arg_groups(args: List[str]) -> Iterator[Tuple[Optional[str], List[str]]]:
    """Yield ``(option_name, tokens)`` for each option and the value it owns.

    ``option_name`` is None for a bare token that does not follow an option.
    Values are claimed the same way ``separate_global_and_op_args`` claims
    them, so both functions group a given command line identically.
    """
    i = 0
    while i < len(args):
        arg = args[i]
        i += 1
        if not arg.startswith("--"):
            yield None, [arg]
            continue
        group = [arg]
        if "=" not in arg and i < len(args) and not args[i].startswith("-"):
            group.append(args[i])
            i += 1
        yield arg.split("=")[0], group


# These select the configurations to compare; they belong to neither side.
_SIDE_OPTIONS = ("--side-a", "--side-b")
# --output-json is consumed by the A/B harness, which writes a single combined
# report for both sides, so it is not part of a side's configuration either.
_HARNESS_OPTIONS = _SIDE_OPTIONS + ("--output-json",)


def base_global_args(argv: Optional[List[str]] = None) -> List[str]:
    """Global tritonbench args the process was invoked with.

    Both sides inherit these, so a side's global configuration only reads
    correctly with them included: ``--side-a "--rep 1000"`` on an
    ``--op softmax --num-inputs 1`` run really means
    ``--op softmax --num-inputs 1 --rep 1000``.
    """
    argv = sys.argv[1:] if argv is None else argv
    global_args, _ = separate_global_and_op_args(argv)
    return [
        token
        for name, tokens in _iter_arg_groups(global_args)
        if name not in _HARNESS_OPTIONS
        for token in tokens
    ]


def merge_global_args(base_args: List[str], side_args: List[str]) -> List[str]:
    """Effective global args of one side, with its own args taking precedence.

    Mirrors :func:`update_args_with_global`, which applies the side's globals
    on top of the base namespace, so the printed args match what actually ran.
    An overridden option keeps the position it had on the command line.
    """
    merged: Dict[str, List[str]] = {}
    loose: List[str] = []
    for source in (base_args, side_args):
        for name, tokens in _iter_arg_groups(source):
            if name is None:
                loose.extend(tokens)
            else:
                merged[name] = tokens
    return [token for tokens in merged.values() for token in tokens] + loose


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
        logger.warning(
            f"Failed to parse global arguments {global_args}, using original args: {e}"
        )
    except Exception as e:
        logger.warning(f"Unexpected error parsing global arguments {global_args}: {e}")

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


def _calculate_performance_summary(
    result_a: BenchmarkOperatorResult,
    result_b: BenchmarkOperatorResult,
    common_x_vals: List,
    common_backends: List[str],
) -> Dict[str, Dict[str, float]]:
    """Calculate performance summary statistics."""
    summary = {}

    # Create result dictionaries for easier lookup
    result_dict_a = {x_val: metrics_dict for x_val, metrics_dict in result_a.result}
    result_dict_b = {x_val: metrics_dict for x_val, metrics_dict in result_b.result}

    for backend in common_backends:
        backend_summary = {}

        for metric in result_a.metrics:
            improvements = []

            for x_val in common_x_vals:
                if backend in result_dict_a[x_val] and backend in result_dict_b[x_val]:
                    metrics_a = result_dict_a[x_val][backend]
                    metrics_b = result_dict_b[x_val][backend]

                    val_a = getattr(metrics_a, metric, None)
                    val_b = getattr(metrics_b, metric, None)

                    if val_a is not None and val_b is not None:
                        # Handle different metric types
                        if hasattr(val_a, "p50"):
                            val_a_num = val_a.p50
                        else:
                            val_a_num = val_a

                        if hasattr(val_b, "p50"):
                            val_b_num = val_b.p50
                        else:
                            val_b_num = val_b

                        if val_a_num != 0:
                            improvement = ((val_b_num - val_a_num) / val_a_num) * 100
                            improvements.append(improvement)

            if improvements:
                backend_summary[metric] = {
                    "avg_improvement": sum(improvements) / len(improvements),
                    "min_improvement": min(improvements),
                    "max_improvement": max(improvements),
                    "count": len(improvements),
                }

        summary[backend] = backend_summary

    return summary


LATENCY_METRIC = "latency"


def _has_latency_metric(*results: Optional[BenchmarkOperatorResult]) -> bool:
    """True if any result collected the latency metric."""
    return any(
        result is not None and LATENCY_METRIC in (result.metrics or [])
        for result in results
    )


def _latency_samples(metrics_obj) -> Optional[List[float]]:
    """Raw per-iteration latency samples (in ms) held by a metrics entry."""
    latency = getattr(metrics_obj, LATENCY_METRIC, None)
    times = getattr(latency, "times", None)
    if not times:
        return None
    return [float(t) for t in times]


class LatencyEntry(NamedTuple):
    """One analyzed (backend, x_val) cell of the latency matrix."""

    backend: str
    x_val: Any
    analysis: LatencyAnalysis
    # True when side B ran this cell but kept no raw latency samples for it.
    b_missing: bool


def _collect_latency_analyses(
    result_dict_a: Dict,
    x_vals: List,
    backends: List[str],
    result_dict_b: Optional[Dict] = None,
) -> List[LatencyEntry]:
    """Analyze the raw latency samples of every (backend, x_val) cell.

    The analysis is bootstrap-heavy, so it is computed once here and shared by
    the log report and the JSON report.
    """
    entries = []
    for backend in backends:
        for x_val in x_vals:
            metrics_a = result_dict_a.get(x_val, {}).get(backend)
            if metrics_a is None:
                continue
            samples_a = _latency_samples(metrics_a)
            if not samples_a:
                continue

            samples_b = None
            if result_dict_b is not None:
                metrics_b = result_dict_b.get(x_val, {}).get(backend)
                samples_b = _latency_samples(metrics_b) if metrics_b else None

            analysis = analyze_latency(samples_a, samples_b)
            if analysis is None:
                continue
            entries.append(
                LatencyEntry(
                    backend=backend,
                    x_val=x_val,
                    analysis=analysis,
                    b_missing=result_dict_b is not None and samples_b is None,
                )
            )
    return entries


def _log_latency_analysis(
    entries: List[LatencyEntry],
    x_val_name: str,
    two_sided: bool,
    label_a: str = "Config A",
    label_b: str = "Config B",
):
    """Report the statistical analysis of the raw latency samples.

    With a single side this is descriptive statistics plus confidence
    intervals. With both sides it also runs the normality test, the hypothesis
    test it selects, and the percent change with its bootstrap CI.
    """
    lines = [
        "",
        "-" * 70,
        "Latency Analysis (All latencies in ms)",
        "-" * 70,
    ]
    if two_sided:
        lines.append(
            f"Percent change is {label_b} vs {label_a} (positive = B is slower)."
        )

    for entry in entries:
        lines.append(f"\n{entry.backend} @ {x_val_name}={entry.x_val}:")
        lines.extend(
            format_latency_analysis(
                entry.analysis, label_a=label_a, label_b=label_b, indent="  "
            )
        )
        if entry.b_missing:
            lines.append(f"  ({label_b} has no latency samples, side A only)")

    if not entries:
        lines.append(
            f"\nNo latency analysis available: needs at least {MIN_SAMPLE} raw "
            "latency samples per side"
        )
    logger.info("\n".join(lines))


# ============================================================================
# JSON report
# ============================================================================

SIDE_A_KEY = "side-a"
SIDE_B_KEY = "side-b"
AB_COMPARISON_KEY = "ab-comparison"


def _json_safe(value: Any) -> Any:
    """Coerce a value into something ``json.dump`` writes as valid JSON.

    NaN and +/-inf (which the statistics produce for degenerate samples) become
    null rather than the JSON-invalid ``NaN``/``Infinity`` literals, and x_vals
    of unsupported types (tuples of shapes, dtypes, ...) fall back to str.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _metric_key(op_name: str, op_mode: str, backend: str, x_val: Any) -> str:
    """Key of one (backend, x_val) cell in a side's ``metrics`` dict.

    Same shape as the keys a non-A/B ``--output-json`` writes, minus the metric
    name suffix -- here the metrics are the entry's own keys.
    """
    return f"tritonbench_{op_name}_{op_mode}[x_{x_val}-{backend}]"


def _cell_metrics(metrics_obj, metric_names: List[str]) -> Dict[str, Any]:
    """The metrics collected for one (backend, x_val) cell.

    Metrics live either on the dataclass or in ``extra_metrics``. Each is
    reported as a single p50 -- of a ``Latency``, or of a numeric sequence such
    as the ``(p50, low, high)`` triple a few operators return for ``gbps`` --
    so a cell is directly comparable across sides. For latency the full picture
    is the descriptive statistics :func:`_side_report` adds alongside it.
    """
    extra = getattr(metrics_obj, "extra_metrics", None) or {}
    values = {}
    for name in metric_names:
        value = extra[name] if name in extra else getattr(metrics_obj, name, None)
        if value is None:
            continue
        if hasattr(value, "p50"):
            value = value.p50
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (int, float)) for v in value
        ):
            value = statistics.median(value) if value else None
        values[name] = _json_safe(value)
    if getattr(metrics_obj, "error_msg", None):
        values["error_msg"] = metrics_obj.error_msg
    return values


def _side_report(
    result: BenchmarkOperatorResult,
    config_args: List[str],
    latency_entries: List[LatencyEntry],
    is_side_a: bool = True,
) -> Dict[str, Any]:
    """JSON report of one side: its configuration and its per-cell metrics.

    Each ``metrics`` entry is one (backend, x_val) cell, holding the metrics
    that were collected for it plus -- when the raw latency samples support it
    -- the descriptive statistics of those samples (mean, median, stddev, CV,
    IQR, confidence intervals). ``latency_entries`` are the shared analyses;
    ``is_side_a`` picks which half of each one to report.
    """
    global_args, op_args = separate_global_and_op_args(config_args)
    stats_by_cell = {}
    for entry in latency_entries:
        side_stats = entry.analysis.side_a if is_side_a else entry.analysis.side_b
        if side_stats is not None:
            stats_by_cell[(entry.backend, entry.x_val)] = side_stats

    metrics = {}
    for x_val, backend_metrics in result.result:
        for backend, metrics_obj in backend_metrics.items():
            cell = _cell_metrics(metrics_obj, result.metrics)
            side_stats = stats_by_cell.get((backend, x_val))
            if side_stats is not None:
                cell.update(_json_safe(asdict(side_stats)))
            key = _metric_key(result.op_name, result.op_mode, backend, x_val)
            metrics[key] = cell

    return {
        "config": list(config_args),
        "global_args": merge_global_args(base_global_args(), global_args),
        "op_args": op_args,
        "op_name": result.op_name,
        "op_mode": result.op_mode,
        "metrics": metrics,
    }


def report_single_side_results(
    result_a: BenchmarkOperatorResult,
    config_a_args: List[str],
) -> Optional[Dict[str, Any]]:
    """Report a single configuration (only --side-a was specified).

    Returns the JSON report, or None when there is nothing to report.
    """
    if not result_a or not result_a.result:
        logger.error("[A/B Comparison] No benchmark data available for Side A")
        return None

    x_vals = sorted({x_val for x_val, _ in result_a.result})
    result_dict_a = {x_val: metrics_dict for x_val, metrics_dict in result_a.result}
    backends = sorted(
        {backend for x_val in x_vals for backend in result_dict_a[x_val].keys()}
    )

    logger.info(
        "\n".join(
            [
                "",
                "=" * 70,
                f"Single-Side Test Results: {result_a.op_name}",
                "=" * 70,
                f"Configuration A: {' '.join(config_a_args) or '(defaults)'}",
                f"\nTest Scope: {len(x_vals)} input shapes, {len(backends)} backends",
                f"Metrics: {', '.join(result_a.metrics)}",
            ]
        )
    )

    x_val_name = REGISTERED_X_VALS.get(result_a.op_name, "x_val")
    latency_entries = []
    if _has_latency_metric(result_a):
        latency_entries = _collect_latency_analyses(result_dict_a, x_vals, backends)
        _log_latency_analysis(latency_entries, x_val_name, two_sided=False)

    return {SIDE_A_KEY: _side_report(result_a, config_a_args, latency_entries)}


def compare_ab_results(
    result_a: BenchmarkOperatorResult,
    result_b: Optional[BenchmarkOperatorResult],
    config_a_args: List[str],
    config_b_args: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Compare A/B test results.

    When ``result_b`` is None only side A was run, so we report that side on
    its own instead of a comparison.

    Returns the JSON report, or None when there is nothing to report.
    """
    if not result_a:
        logger.error("[A/B Comparison] Side A result is invalid")
        return None

    if result_b is None:
        return report_single_side_results(result_a, config_a_args)

    # Check if both results have data
    if not result_a.result or not result_b.result:
        logger.error("No benchmark data available for comparison")
        return None

    # Get common data for analysis
    x_vals_a = {x_val for x_val, _ in result_a.result}
    x_vals_b = {x_val for x_val, _ in result_b.result}
    common_x_vals = sorted(x_vals_a.intersection(x_vals_b))

    if not common_x_vals:
        logger.error("No common input shapes found between configurations")
        return None

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
        logger.error("No common backends found between configurations")
        return None

    # ============================================================================
    # SECTION 1: Configuration Analysis
    # ============================================================================
    lines = [
        "",
        "=" * 70,
        f"A/B Test Results: {result_a.op_name}",
        "=" * 70,
        "Configuration Differences:",
    ]
    differences = {}
    try:
        differences = _analyze_config_differences(config_a_args, config_b_args)

        if differences:
            for param, (val_a, val_b) in differences.items():
                lines.append(f"  {param:<15}: {val_a:<15} → {val_b}")
        else:
            lines.append("  No configuration differences detected")
    except Exception as e:
        lines.append(f"  ERROR: Failed to analyze configuration differences: {e}")

    lines.append(
        f"\nTest Scope: {len(common_x_vals)} input shapes, {len(common_backends)} backends"
    )
    lines.append(f"Metrics: {', '.join(result_a.metrics)}")
    logger.info("\n".join(lines))

    # ============================================================================
    # SECTION 2: Performance Summary
    # ============================================================================
    lines = ["", "-" * 70, "Performance Summary", "-" * 70]

    summary = _calculate_performance_summary(
        result_a, result_b, common_x_vals, common_backends
    )

    for backend in common_backends:
        lines.append(f"\n{backend}:")
        backend_data = summary.get(backend, {})

        if not backend_data:
            lines.append("  No comparable data")
            continue

        for metric, stats in backend_data.items():
            avg_improvement = stats["avg_improvement"]
            min_improvement = stats["min_improvement"]
            max_improvement = stats["max_improvement"]

            lines.append(
                f"  {metric:<12}: {avg_improvement:+5.1f}% avg [{min_improvement:+.1f}% to {max_improvement:+.1f}%]"
            )
    logger.info("\n".join(lines))

    # ============================================================================
    # SECTION 3: Detailed Comparison (Compact)
    # ============================================================================
    lines = ["", "-" * 70, "Detailed Comparison", "-" * 70]

    x_val_name = REGISTERED_X_VALS.get(result_a.op_name, "x_val")
    detailed_rows: List[Dict[str, Any]] = []

    # Show all metrics for detailed comparison
    for metric in result_a.metrics:
        lines.append(f"\nMetric: {metric}")
        lines.append(
            "Backend".ljust(15)
            + x_val_name.ljust(20)
            + "Config A".ljust(12)
            + "Config B".ljust(12)
            + "Difference".ljust(12)
        )
        lines.append("-" * 71)

        for backend in common_backends:
            first_row = True
            for x_val in common_x_vals:
                if (
                    backend not in result_dict_a[x_val]
                    or backend not in result_dict_b[x_val]
                ):
                    continue

                metrics_a = result_dict_a[x_val][backend]
                metrics_b = result_dict_b[x_val][backend]

                val_a = getattr(metrics_a, metric, None)
                val_b = getattr(metrics_b, metric, None)

                if val_a is not None and val_b is not None:
                    # Handle different data types
                    if hasattr(val_a, "p50"):
                        val_a_num = val_a.p50
                        val_b_num = val_b.p50
                    else:
                        val_a_num = val_a
                        val_b_num = val_b

                    if val_a_num != 0:
                        diff_pct = ((val_b_num - val_a_num) / val_a_num) * 100
                    else:
                        diff_pct = 0

                    # Format values
                    if isinstance(val_a_num, float):
                        val_a_str = f"{val_a_num:.3f}"
                        val_b_str = f"{val_b_num:.3f}"
                    else:
                        val_a_str = str(val_a_num)
                        val_b_str = str(val_b_num)

                    # Row of the comparison table
                    backend_name = backend if first_row else ""
                    lines.append(
                        f"{backend_name:<15}{str(x_val):<20}{val_a_str:<12}{val_b_str:<12}{diff_pct:+5.1f}%"
                    )
                    first_row = False
                    detailed_rows.append(
                        {
                            "metric": metric,
                            "backend": backend,
                            "x_val_name": x_val_name,
                            "x_val": _json_safe(x_val),
                            SIDE_A_KEY: _json_safe(val_a_num),
                            SIDE_B_KEY: _json_safe(val_b_num),
                            "pct_change": _json_safe(diff_pct),
                        }
                    )

            if not first_row:  # Only add a separator if we added data
                lines.append("")
    logger.info("\n".join(lines))

    # ============================================================================
    # SECTION 4: Latency Analysis
    # ============================================================================
    latency_entries = []
    if _has_latency_metric(result_a, result_b):
        latency_entries = _collect_latency_analyses(
            result_dict_a, common_x_vals, common_backends, result_dict_b=result_dict_b
        )
        _log_latency_analysis(latency_entries, x_val_name, two_sided=True)

    # ============================================================================
    # SECTION 5: JSON report
    # ============================================================================
    latency_comparison = [
        {
            "backend": entry.backend,
            "x_val_name": x_val_name,
            "x_val": _json_safe(entry.x_val),
            "comparison": _json_safe(asdict(entry.analysis.comparison)),
        }
        for entry in latency_entries
        if entry.analysis.comparison is not None
    ]
    comparison_report = {
        "op_name": result_a.op_name,
        "op_mode": result_a.op_mode,
        "x_val_name": x_val_name,
        "x_vals": _json_safe(common_x_vals),
        "backends": common_backends,
        "metrics": list(result_a.metrics),
        "config_differences": {
            param: {SIDE_A_KEY: val_a, SIDE_B_KEY: val_b}
            for param, (val_a, val_b) in differences.items()
        },
        "performance_summary": _json_safe(summary),
        "detailed_comparison": detailed_rows,
    }
    if latency_comparison:
        comparison_report["latency_comparison"] = latency_comparison

    return {
        SIDE_A_KEY: _side_report(
            result_a, config_a_args, latency_entries, is_side_a=True
        ),
        SIDE_B_KEY: _side_report(
            result_b, config_b_args or [], latency_entries, is_side_a=False
        ),
        AB_COMPARISON_KEY: comparison_report,
    }


def write_ab_json(output_json: str, report: Dict[str, Any]) -> None:
    """Write the A/B report to ``output_json``.

    The file always has a ``side-a`` key; ``side-b`` and ``ab-comparison`` are
    only present when ``--side-b`` was specified.
    """
    with open(output_json, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"[tritonbench] Output A/B result json to {output_json}")


def run_ab_test(
    base_args: argparse.Namespace, base_extra_args: List[str], _run_func
) -> Tuple[BenchmarkOperatorResult, Optional[BenchmarkOperatorResult]]:
    """Run the A/B test and return both results.

    Side B is optional: when ``--side-b`` is not specified only side A runs and
    the second result is None.
    """

    # Parse A and B configurations
    try:
        config_a_args = parse_ab_config(base_args.side_a)
    except ValueError as e:
        logger.error(f"Failed to parse Side A configuration: {e}")
        raise

    run_side_b = base_args.side_b is not None
    config_b_args = []
    if run_side_b:
        try:
            config_b_args = parse_ab_config(base_args.side_b)
        except ValueError as e:
            logger.error(f"Failed to parse Side B configuration: {e}")
            raise

    lines = [f"[A/B Test] Configuration A: {' '.join(config_a_args)}"]
    if run_side_b:
        lines.append(f"[A/B Test] Configuration B: {' '.join(config_b_args)}")
    else:
        lines.append("[A/B Test] Configuration B: (not specified, running side A only)")

    # Separate global and operator-specific arguments
    global_a_args, op_a_args = separate_global_and_op_args(config_a_args)
    global_b_args, op_b_args = separate_global_and_op_args(config_b_args)

    # Report the globals each side actually runs with: the ones on the command
    # line, overridden by the ones the side specifies.
    base_globals = base_global_args()
    effective_global_a = merge_global_args(base_globals, global_a_args)
    effective_global_b = merge_global_args(base_globals, global_b_args)

    if effective_global_a:
        lines.append(f"[A/B Test] Global args A: {' '.join(effective_global_a)}")
    if op_a_args:
        lines.append(f"[A/B Test] Operator args A: {' '.join(op_a_args)}")
    if run_side_b and effective_global_b:
        lines.append(f"[A/B Test] Global args B: {' '.join(effective_global_b)}")
    if op_b_args:
        lines.append(f"[A/B Test] Operator args B: {' '.join(op_b_args)}")
    logger.info("\n".join(lines))

    # Update args with global parameters
    args_a = update_args_with_global(base_args, global_a_args)

    # Combine extra_args with operator-specific args only
    extra_args_a = base_extra_args + op_a_args

    lines = ["", "=" * 60, f"Running Side A: {' '.join(config_a_args)}"]
    if effective_global_a:
        lines.append(f"  Global args: {' '.join(effective_global_a)}")
    if op_a_args:
        lines.append(f"  Operator args: {' '.join(op_a_args)}")
    lines.append("=" * 60)
    logger.info("\n".join(lines))

    try:
        result_a = _run_func(args_a, extra_args_a)
        if not result_a:
            raise RuntimeError("Side A returned empty result")
    except Exception as e:
        logger.error(f"Side A failed to run: {e}")
        raise RuntimeError(f"A/B test failed - Side A error: {e}")

    if not run_side_b:
        return result_a, None

    args_b = update_args_with_global(base_args, global_b_args)
    extra_args_b = base_extra_args + op_b_args

    lines = ["", "=" * 60, f"Running Side B: {' '.join(config_b_args)}"]
    if effective_global_b:
        lines.append(f"  Global args: {' '.join(effective_global_b)}")
    if op_b_args:
        lines.append(f"  Operator args: {' '.join(op_b_args)}")
    lines.append("=" * 60)
    logger.info("\n".join(lines))

    try:
        result_b = _run_func(args_b, extra_args_b)
        if not result_b:
            raise RuntimeError("Side B returned empty result")
    except Exception as e:
        logger.error(f"Side B failed to run: {e}")
        raise RuntimeError(f"A/B test failed - Side B error: {e}")

    return result_a, result_b
