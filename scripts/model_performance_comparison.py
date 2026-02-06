import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager
from typing import List
from Global_parameters import PROJ_HOME, AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE, LEGEND_FONT_SIZE
# ======================
# Config & Fonts
# ======================
gill_sans_font = font_manager.FontProperties(family='Gill Sans')
plt.rcParams['font.family'] = gill_sans_font.get_name()

Performance_dir = os.path.join(PROJ_HOME,
                               "Performance/TargetScan_test")
os.makedirs(Performance_dir, exist_ok=True)

# ======================
# Data (from CSV file)
# ======================
csv_path = os.path.join(Performance_dir, "model_comparison_metrics.csv")
df = pd.read_csv(csv_path)

# ======================
# Colors
# ======================
colors = {
    "MiRformer": "#FF9F1C",   # orange
    "REPRESS": "#63CFEF",     # cyan
    "miTAR": "#A267F2",        # purple
    "Mimosa": "#54E3AF"        # green
}

# ======================
# Helpers
# ======================
def _subset_df_in_order(df: pd.DataFrame, models_in_order: List[str]) -> pd.DataFrame:
    present = set(df["Model"].astype(str))
    missing = [m for m in models_in_order if m not in present]
    if missing:
        raise ValueError(f"Missing model(s): {missing}. Present: {sorted(present)}")
    return (df[df["Model"].isin(models_in_order)]
            .set_index("Model")
            .reindex(models_in_order)
            .reset_index())

def _ensure_metrics_exist(df: pd.DataFrame, metrics: List[str]):
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metric column(s): {missing}. "
                         f"Available: {list(df.columns)}")

def _auto_ylim(values: np.ndarray, ceiling: float = 1.0, pad: float = 0.1):
    # values may contain NaN; ignore them for vmax and vmin
    if values.size == 0 or np.all(np.isnan(values)):
        return (0.0, ceiling)
    vmax = float(np.nanmax(values))
    vmin = float(np.nanmin(values))
    
    # Calculate top limit
    top = min(max(ceiling, vmax + pad), max(1.05, vmax + pad))
    
    # Calculate bottom limit
    if vmin < 0:
        bottom = vmin - pad * (vmax - vmin) if vmax != vmin else vmin - pad
    elif vmin < 0.1:
        bottom = 0.0
    else:
        range_pad = pad * (vmax - vmin) if vmax != vmin else pad
        bottom = max(0.0, vmin - range_pad)
    
    return (bottom, top)

def _bar_annot(ax, x, width, i, vals, ypad=0.01, fmt="{:.3f}"):
    vals = np.asarray(vals, dtype=float).ravel()
    for j, v in enumerate(vals):
        if np.isnan(v):
            continue
        ax.text(x[j] + i * width, v + ypad, fmt.format(v),
                ha='center', va='bottom', fontsize=10)

def _get_vals(sub: pd.DataFrame, model: str, metrics: List[str]) -> np.ndarray:
    row = sub.loc[sub["Model"] == model, metrics]
    arr = row.to_numpy(dtype=float)
    # Shape -> 1-D length == len(metrics)
    return np.atleast_1d(arr).reshape(1, -1).ravel()

def _plot_axis(ax, df, models, metric, title,
               ylim_ceiling=1.0, ylabel=None,
               n_max_models=None, max_bar_width=0.95,
               show_ylabel=True):
    """
    Plot a single-metric bar chart.
    `title` is repurposed: it becomes the y-axis label text
    instead of a subplot title.
    """
    _ensure_metrics_exist(df, [metric])
    sub = _subset_df_in_order(df, models)

    n_models = len(models)
    if n_max_models is None:
        n_max_models = n_models

    x = np.arange(n_models)

    # keep bar pixel width roughly constant across subplots
    width = max_bar_width * (n_models / n_max_models)

    # Bars - one bar per model
    for i, model in enumerate(models):
        row = sub.loc[sub["Model"] == model, metric]
        val = float(row.iloc[0]) if not pd.isna(row.iloc[0]) else np.nan
        if not np.isnan(val):
            ax.bar(x[i], val, width=width,
                   label=model, color=colors.get(model))
            # Place text slightly above the bar in *display* coordinates
            ax.annotate(
                f"{val:.3f}",
                xy=(x[i], val),          # bar top in data coords
                xytext=(0, 15),           # 6 points above in screen coords
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90,
                rotation_mode='anchor'
            )

    # Cosmetics
    # ax.set_xticks(x)
    # ax.set_xticklabels(models,
    #                    fontsize=TICK_LABEL_FONTSIZE-2,
    #                    rotation=90,
    #                    ha='right')
    # remove x ticks
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE-2)
    values = sub[metric].to_numpy(dtype=float)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.margins(y=0.3)

    # Use `title` text on the y-axis instead of as a subplot title
    if show_ylabel:
        ax.set_ylabel(title, fontsize=AXIS_FONT_SIZE-2)
    else:
        ax.set_ylabel("")

    return ax.get_legend_handles_labels()


# ======================
# Panels (per your spec)
# ======================
p1_models = ["MiRformer", "REPRESS", "miTAR", "Mimosa"]
p1_metric = "Binding Accuracy"

p2_models = ["MiRformer", "REPRESS", "miTAR", "Mimosa"]
p2_metric = "Seed Span AUROC"

p3_models = ["MiRformer", "REPRESS", "miTAR", "Mimosa"]
p3_metric = "Seed Span AUPRC"

p4_models = ["MiRformer", "REPRESS", "miTAR", "Mimosa"]
p4_metric = "Hit at 5"

p5_models = ["MiRformer", "REPRESS", "miTAR", "Mimosa"]
p5_metric = "Hit at 3"

p6_models = ["MiRformer", "REPRESS", "miTAR", "Mimosa"]
p6_metric = "Hit at 0"

all_model_lists = [p1_models, p2_models, p3_models, p4_models, p5_models, p6_models]
n_max_models = max(len(m) for m in all_model_lists)  # here: 4

# 2 rows x 3 columns, width fits A4 with 1" margins
fig, axes = plt.subplots(
    2, 3,
    figsize=(18/2.54, 12/2.54),
    dpi=500,
)

axes = axes.ravel()

h1, l1 = _plot_axis(axes[0], df, p1_models, p1_metric,
                    "Binding Accuracy",
                    n_max_models=n_max_models,
                    show_ylabel=True)

h2, l2 = _plot_axis(axes[1], df, p2_models, p2_metric,
                    "Seed Span AUROC",
                    n_max_models=n_max_models,
                    show_ylabel=True)

h3, l3 = _plot_axis(axes[2], df, p3_models, p3_metric,
                    "Seed Span AUPRC",
                    n_max_models=n_max_models,
                    show_ylabel=True)

h4, l4 = _plot_axis(axes[3], df, p4_models, p4_metric,
                    "Hit @ 5 Accuracy",
                    n_max_models=n_max_models,
                    show_ylabel=True)

h5, l5 = _plot_axis(axes[4], df, p5_models, p5_metric,
                    "Hit @ 3 Accuracy",
                    n_max_models=n_max_models,
                    show_ylabel=True)

h6, l6 = _plot_axis(axes[5], df, p6_models, p6_metric,
                    "Hit @ 0 Accuracy",
                    n_max_models=n_max_models,
                    show_ylabel=True)


# Shared legend
handles, labels = [], []
for H, L in [(h1, l1), (h2, l2), (h3, l3), (h4, l4), (h5, l5), (h6, l6)]:
    for h, l in zip(H, L):
        if l not in labels:
            handles.append(h)
            labels.append(l)

fig.legend(handles, labels,
    bbox_to_anchor=(0.85, 1.05),
    ncol=len(labels),
    frameon=False,
    fontsize=LEGEND_FONT_SIZE-2,
    borderaxespad=0.5)

fig.tight_layout(rect=[0, 0.12, 1, 1])

save_path = os.path.join(Performance_dir, "model_comparison_combined_horizontal.png")
plt.savefig(save_path, dpi=500, bbox_inches="tight")
plt.close(fig)
print(f"Combined figure saved to: {save_path}")

