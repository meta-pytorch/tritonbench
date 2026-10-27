"""
Generic plotting utilities for tritonbench operators.

This module provides a generic implementation for plotting benchmark results
that can be used by any operator. It dynamically discovers providers from
benchmark results and generates bar charts comparing performance metrics.
"""

import csv
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
) -> Optional[str]:
    """
    Generic function to plot benchmark results for any tritonbench operator.

    This function dynamically discovers providers from the benchmark results
    and creates a bar chart comparing the specified metric across providers.

    Args:
        output: The BenchmarkOperatorResult object containing benchmark data.
        tb_args: The tritonbench arguments object, used for input_loader path.
        op_name: The name of the operator (used for default titles and paths).
        x_label: Label for the x-axis. If None, auto-detected from x_vals type.
        y_metric: The metric to plot on the y-axis (default: "tflops").
        y_label: Label for the y-axis (default: "TFLOPS").
        plot_title: Custom title for the plot. If None, auto-generated from
                    op_name and input_loader.
        x_val_formatter: Optional function to format x_val as strings.
                        If None, uses default formatting based on type.
        save_subdir: Subdirectory under ~/tritonbench_plots/ to save plots.
                    If None, uses op_name.
        figsize: Figure size as (width, height) tuple (default: (12, 6)).
        colors: List of colors for bars. If None, uses default color palette.
        show_plot: Whether to display the plot interactively (default: True).

    Returns:
        The path to the saved plot file, or None if no results to plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Use the ordered result list: inputs sharing an x_val (same shape, other
    # dtype/strides) would collapse in the x_val-keyed result dict.
    rows = list(output.result)
    if not rows:
        print("No benchmark results to plot.")
        return None

    providers = list(rows[0][1].keys())

    if not providers:
        print("No providers found in benchmark results.")
        return None

    # None marks a missing or failed result; it is drawn as "n/a", not as 0.
    x_vals, seen = [], {}
    data = {provider: [] for provider in providers}

    for x_val, by_provider in rows:
        seen[x_val] = seen.get(x_val, 0) + 1
        x_vals.append(x_val if seen[x_val] == 1 else f"{x_val} #{seen[x_val]}")
        for provider in providers:
            m = by_provider.get(provider)
            if m is not None and not hasattr(m, y_metric):
                m = m.extra_metrics
            data[provider].append(
                m.get(y_metric) if isinstance(m, dict) else getattr(m, y_metric, None)
            )

    plot_name = f"{op_name}-performance"
    default_title = f"{op_name.upper()} Performance Comparison"

    if hasattr(tb_args, "input_loader") and tb_args.input_loader:
        input_loader_path = tb_args.input_loader
        base_name = os.path.basename(input_loader_path).replace(".json", "")
        plot_name = base_name
        default_title = f"{op_name.upper()} Performance: {base_name}"
    elif getattr(tb_args, "suite", None):
        plot_name = f"{op_name}-{tb_args.suite}-performance"
        default_title = f"{op_name.upper()} Performance: {tb_args.suite}"

    final_title = plot_title if plot_title else default_title

    _, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(x_vals))
    width = 0.8 / len(providers)
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

    for i, provider in enumerate(providers):
        offset = (i - len(providers) / 2 + 0.5) * width
        color = bar_colors[i % len(bar_colors)]
        vals = data[provider]
        ax.bar(
            x + offset,
            [np.nan if v is None else v for v in vals],
            width,
            label=provider,
            color=color,
        )
        for j, v in enumerate(vals):
            if v is None:
                ax.text(
                    x[j] + offset,
                    0,
                    "n/a",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=color,
                )

    if x_val_formatter:
        x_labels = [x_val_formatter(x_val) for x_val in x_vals]
    else:
        x_labels = [str(x_val) for x_val in x_vals]

    final_x_label = x_label if x_label else "x_val"

    ax.set_xlabel(final_x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(final_title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    home_dir = os.path.expanduser("~")
    subdir = save_subdir if save_subdir else op_name
    save_path = os.path.join(home_dir, "tritonbench_plots", subdir)
    os.makedirs(save_path, exist_ok=True)
    plot_file = os.path.join(save_path, f"{plot_name}.png")
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")

    csv_path = os.path.join(save_path, f"{plot_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_val"] + providers)
        for i, x_val in enumerate(x_vals):
            row = [str(x_val)] + [
                "" if data[p][i] is None else data[p][i] for p in providers
            ]
            writer.writerow(row)
    print(f"Data saved to {csv_path}")

    if is_fbcode():
        _upload_plot(plot_file)

    if show_plot:
        plt.show()

    return plot_file
