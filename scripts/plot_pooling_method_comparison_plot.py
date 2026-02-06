"""
Create a 2x2 grid of pooling-method line plots from fixed CSV inputs.

The script expects the CSV files listed in `METRIC_FILES` to be located in the
current working directory. Each CSV must contain a `Step` column and one or
more metric columns; all are plotted against `Step`, and the figure is saved as
`Pooling_Method_comparison_line_plot.svg`.

Additionaly, a bar plot of the last step of each metric is also created and 
saved as `Pooling_Method_comparison_barplot.svg`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.font_manager as font_manager
from Global_parameters import PROJ_HOME, AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE, LEGEND_FONT_SIZE
gill_sans_font = font_manager.FontProperties(family='Gill Sans')
plt.rcParams['font.family'] = gill_sans_font.get_name()

METRIC_FILES: Tuple[Tuple[str, str], ...] = (
    ("Binding Accuracy", "Pooling_Method_Binding_Accuracy.csv"),
    ("Exact Match Rate", "Pooling_Method_Exact_match_rate.csv"),
    ("F1 Score", "Pooling_Method_F1_score.csv"),
    ("Evaluation Loss", "Pooling_Method_Eval_loss.csv"),
)

LineStyleType = Union[str, Tuple[float, Tuple[float, ...]]]


def read_metric_csv(csv_path: Path) -> pd.DataFrame:
    """Return a dataframe with `Step` as the first column."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Step" not in df.columns:
        raise ValueError(f"`Step` column not found in {csv_path}")

    df = df.copy()
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df = df.dropna(subset=["Step"])
    return df


def plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    colors: Dict[str, Tuple[float, float, float, float]],
    line_styles: Dict[str, LineStyleType],
) -> Dict[str, plt.Line2D]:
    """Plot all metric columns against `Step`."""
    if len(df.columns) <= 1:
        raise ValueError(f"No metric columns found for {title}")

    lines: Dict[str, plt.Line2D] = {}
    for column in df.columns:
        if column == "Step":
            continue
        line, = ax.plot(
            df["Step"],
            df[column],
            label=column,
            color=colors[column],
            linestyle=line_styles.get(column, "-"),
        )
        lines[column] = line
    # set face color to whitesmoke
    ax.set_facecolor("whitesmoke")
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax.set_xlabel("Step", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel(title, fontsize=AXIS_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    add_top_margin(ax)
    return lines


def add_top_margin(ax: plt.Axes, fraction: float = 0.1) -> None:
    ymin, ymax = ax.get_ylim()
    if ymax == ymin:
        return
    ax.set_ylim(ymin, ymax + (ymax - ymin) * fraction)


def last_step_values(df: pd.DataFrame) -> pd.Series:
    """Return the metric values at the last (max) Step as a Series indexed by column.
    Values should not be N/A or NaN."""
    if "Step" not in df.columns:
        raise ValueError("`Step` column not found.")
    if len(df.columns) <= 1:
        raise ValueError("No metric columns found.")

    df_sorted = df.sort_values("Step")
    last_row = df_sorted.iloc[-1]
    metric_cols = [c for c in df_sorted.columns if c != "Step"]
    values = last_row[metric_cols].copy()
    values = pd.to_numeric(values, errors="coerce")
    values = values[values.notna()]
    return values


metric_data: Dict[str, pd.DataFrame] = {}
for metric_name, file_name in METRIC_FILES:
    csv_path = Path(file_name)
    metric_data[metric_name] = read_metric_csv(csv_path)

# Determine a consistent color per metric column across all plots
metric_columns: List[str] = []
for df in metric_data.values():
    for column in df.columns:
        if column == "Step" or column in metric_columns:
            continue
        metric_columns.append(column)


def generate_unique_line_styles(count: int) -> List[LineStyleType]:
    base_styles: List[LineStyleType] = ["-", "--", "-.", ":"]
    if count <= len(base_styles):
        return base_styles[:count]
    styles: List[LineStyleType] = base_styles.copy()
    for idx in range(count - len(base_styles)):
        dash_length = 1.5 + idx * 0.6
        gap_length = 0.8 + (idx % 3) * 0.3
        styles.append((0, (dash_length, gap_length)))
    return styles


cmap = plt.get_cmap("Set1", len(metric_columns))
metric_colors: Dict[str, Tuple[float, float, float, float]] = {
    column: cmap(idx % cmap.N) for idx, column in enumerate(metric_columns)
}
line_style_list = generate_unique_line_styles(len(metric_columns))
metric_line_styles: Dict[str, LineStyleType] = {
    column: line_style_list[idx]
    for idx, column in enumerate(metric_columns)
}

fig, axes = plt.subplots(2, 2, figsize=(30/2.54, 23.87/2.54), sharex=False)
axes_iter: Iterable[Tuple[str, plt.Axes]] = zip(
    metric_data.keys(), axes.flatten()
)

legend_handles: Dict[str, plt.Line2D] = {}
for metric_name, ax in axes_iter:
    lines = plot_metric(
        ax,
        metric_data[metric_name],
        metric_name,
        metric_colors,
        metric_line_styles,
    )
    for name, line in lines.items():
        if name not in legend_handles:
            legend_handles[name] = line

fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.legend(
    list(legend_handles.values()),
    list(legend_handles.keys()),
    loc="lower center",
    ncol=1,
    fontsize=LEGEND_FONT_SIZE,
)
output_dir = os.path.join(PROJ_HOME, "Performance/TargetScan_test")
os.makedirs(output_dir, exist_ok=True)

# output_path = os.path.join(output_dir, "Pooling_Method_comparison_lineplot.svg")
# fig.savefig(output_path, dpi=500)
# print(f"Saved plot to {output_path}")

# --- Bar plots (last step for each metric) ---
bar_fig, bar_axes = plt.subplots(2, 2, figsize=(30/2.54, 26.87/2.54), sharex=False)
bar_axes_iter: Iterable[Tuple[str, plt.Axes]] = zip(metric_data.keys(), bar_axes.flatten())

for metric_name, ax in bar_axes_iter:
    values = last_step_values(metric_data[metric_name])
    x_labels = list(values.index)
    y = values.values
    bar_colors = [metric_colors.get(name, (0.2, 0.2, 0.2, 1.0)) for name in x_labels]

    bar_container = ax.bar(x_labels, y, color=bar_colors)
    ax.set_facecolor("whitesmoke")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    ax.set_ylabel(metric_name, fontsize=AXIS_FONT_SIZE)
    ax.set_title(f"{metric_name}", fontsize=TITLE_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    # no x axis tick labels
    ax.set_xticks([])
    for rect, value in zip(bar_container, y):
        ax.annotate(
            f"{value:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=TICK_FONT_SIZE,
        )
    add_top_margin(ax)


bar_fig.tight_layout(rect=(0, 0.23, 1, 1))
legend_patches = [mpatches.Patch(color=metric_colors[name], label=name) for name in metric_columns]
bar_fig.legend(
    handles=legend_patches,
    labels=[p.get_label() for p in legend_patches],
    loc="lower center",
    ncol=1,
    fontsize=LEGEND_FONT_SIZE+2,
)

bar_output_path = os.path.join(output_dir, "Pooling_Method_comparison_barplot.svg")
bar_fig.savefig(bar_output_path, dpi=500)
print(f"Saved bar plot to {bar_output_path}")
