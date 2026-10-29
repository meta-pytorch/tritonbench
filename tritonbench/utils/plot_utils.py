"""
Generic plotting utilities for tritonbench operators.

This module provides a generic implementation for plotting benchmark results
that can be used by any operator. It dynamically discovers providers from
benchmark results and generates bar charts comparing performance metrics.
"""

import csv
import math
import os
import subprocess
import sys
from typing import Any, Callable, List, Optional, Tuple

from tritonbench.utils.env_utils import is_fbcode


def _upload_plot(plot_file: str) -> None:
    try:
        result = subprocess.run(
            [
                "meta",
                "pixelcloud.image",
                "upload",
                "--file",
                f"file://{os.path.abspath(plot_file)}",
                "--no-truncate",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = (
            exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        )
        print(f"Pixelcloud upload failed: {detail}", file=sys.stderr)
        return
    print(result.stdout.strip())


def _metric_value(metrics: Any, y_metric: str) -> Optional[float]:
    """Return a provider's metric, or None if it is missing or errored."""
    if metrics is None or getattr(metrics, "error_msg", None):
        return None
    value = getattr(metrics, y_metric, None)
    if value is None:
        value = (getattr(metrics, "extra_metrics", None) or {}).get(y_metric)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _plot_name(op_name: str, tb_args: Any) -> Tuple[str, str]:
    """Plot file stem and title, unique per input source and mode."""
    input_loader = getattr(tb_args, "input_loader", None)
    suite = getattr(tb_args, "suite", None)
    if input_loader:
        source = os.path.basename(input_loader).replace(".json", "")
    elif suite:
        source = f"{op_name}-{suite}"
    else:
        source = f"{op_name}-performance"
    mode = getattr(tb_args, "mode", None)
    if mode and mode != "fwd":
        source = f"{source}-{mode}"
    return source, f"{op_name.upper()} Performance: {source}"


def plot_benchmark_results(
    output: Any,
    tb_args: Any,
    op_name: str,
    x_label: Optional[str] = None,
    y_metric: str = "tflops",
    y_label: str = "TFLOPS",
    plot_title: Optional[str] = None,
    x_val_formatter: Optional[Callable[[Any], str]] = None,
    save_subdir: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    colors: Optional[List[str]] = None,
    show_plot: bool = True,
    log_y: bool = False,
) -> Optional[str]:
    """
    Generic function to plot benchmark results for any tritonbench operator.

    This function dynamically discovers providers from the benchmark results
    and creates bar charts comparing the specified metric across providers.
    Inputs are plotted in run order, one bar group per input, so inputs that
    share an x_val (e.g. the same shape in two dtypes) stay distinct. Missing
    or errored results are marked "x" instead of drawn as zero bars, and
    results that failed the accuracy check are hatched. All inputs go in one
    plot; the figure widens with the input count.

    Args:
        output: The BenchmarkOperatorResult object containing benchmark data.
        tb_args: The tritonbench arguments object, used for the input source,
                 mode and output directory.
        op_name: The name of the operator (used for default titles and paths).
        x_label: Label for the x-axis. If None, uses "x_val".
        y_metric: The metric to plot on the y-axis (default: "tflops").
        y_label: Label for the y-axis (default: "TFLOPS").
        plot_title: Custom title for the plot. If None, auto-generated from
                    op_name and the input source.
        x_val_formatter: Optional function to format x_val as strings.
                        If None, uses str(x_val).
        save_subdir: Subdirectory under ~/tritonbench_plots/ to save plots.
                    If None, uses op_name. Ignored when --output-dir is set.
        figsize: Minimum figure size as (width, height) tuple.
        colors: List of colors for bars. If None, uses default color palette.
        show_plot: Whether to display the plot interactively (default: True).
        log_y: Use a log-scale y-axis.

    Returns:
        The path to the saved plot file, or None if no results to plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    rows = list(output.result)
    if not rows:
        print("No benchmark results to plot.")
        return None

    providers = list(dict.fromkeys(p for _, row in rows for p in row))
    if not providers:
        print("No providers found in benchmark results.")
        return None

    data = {
        p: [_metric_value(row.get(p), y_metric) for _, row in rows] for p in providers
    }
    inaccurate = {
        p: [getattr(row.get(p), "accuracy", None) is False for _, row in rows]
        for p in providers
    }
    format_x = x_val_formatter if x_val_formatter else str
    x_labels = [f"{i}: {format_x(x_val)}" for i, (x_val, _) in enumerate(rows)]

    plot_name, default_title = _plot_name(op_name, tb_args)
    if y_metric != "tflops":
        plot_name = f"{plot_name}-{y_metric}"
        default_title = f"{default_title} ({y_label})"
    final_title = plot_title if plot_title else default_title

    output_dir = getattr(tb_args, "output_dir", None)
    if output_dir:
        save_path = os.path.join(output_dir, "plots")
    else:
        subdir = save_subdir if save_subdir else op_name
        save_path = os.path.join(os.path.expanduser("~"), "tritonbench_plots", subdir)
    os.makedirs(save_path, exist_ok=True)

    default_colors = [
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    bar_colors = colors if colors else default_colors
    width = 0.8 / len(providers)

    x = np.arange(len(rows))
    horizontal = bool(getattr(tb_args, "suite", None)) and tb_args.suite.startswith(
        "sm90"
    )
    if horizontal:
        figsize = (18, max(6, len(rows) * 0.32))
    else:
        figsize = (max(figsize[0], 0.25 * len(providers) * len(x) + 3), figsize[1])
    fig, ax = plt.subplots(figsize=figsize)

    for i, provider in enumerate(providers):
        color = bar_colors[i % len(bar_colors)]
        offset = x + (i - len(providers) / 2 + 0.5) * width
        values = data[provider]
        heights = [np.nan if v is None else v for v in values]
        hatches = ["//" if bad else "" for bad in inaccurate[provider]]
        bar = ax.barh if horizontal else ax.bar
        bars = bar(offset, heights, width, label=provider, color=color)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        missing = [o for o, v in zip(offset, values) if v is None]
        # Axes-relative coordinates keep missing markers visible on log axes.
        ax.plot(
            [0.01] * len(missing) if horizontal else missing,
            missing if horizontal else [0.01] * len(missing),
            "x",
            color=color,
            markersize=6,
            transform=(
                ax.get_yaxis_transform() if horizontal else ax.get_xaxis_transform()
            ),
        )

    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="//"))
    labels = providers + ["inaccurate (hatched); x = missing/failed"]
    ax.legend(handles, labels, loc="upper right", fontsize=9)

    final_x_label = x_label if x_label else "x_val"
    ax.set_xlabel(y_label if horizontal else final_x_label, fontsize=11)
    ax.set_ylabel(final_x_label if horizontal else y_label, fontsize=11)
    ax.set_title(final_title, fontsize=13, fontweight="bold")
    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(x_labels, fontsize=8)
        ax.invert_yaxis()
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="x" if horizontal else "y", alpha=0.3)
    if log_y:
        (ax.set_xscale if horizontal else ax.set_yscale)("log")
    fig.tight_layout()

    plot_file = os.path.join(save_path, f"{plot_name}.png")
    fig.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")
    if not show_plot:
        plt.close(fig)

    csv_path = os.path.join(save_path, f"{plot_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_val"] + providers)
        for i, label in enumerate(x_labels):
            writer.writerow(
                [label] + ["" if data[p][i] is None else data[p][i] for p in providers]
            )
    print(f"Data saved to {csv_path}")

    if is_fbcode():
        _upload_plot(plot_file)

    if show_plot:
        plt.show()

    return plot_file
