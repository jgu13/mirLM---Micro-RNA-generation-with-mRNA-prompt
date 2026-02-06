import os
import numpy as np
import pandas as pd
import mpmath as mp
import matplotlib.pyplot as plt
from math import log, exp, sqrt
from scipy.stats import mannwhitneyu, ttest_ind
from Global_parameters import PROJ_HOME, AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE

def load_pred_probs(csv_path: str, column: str = "pred prob") -> pd.Series:
    """Load a CSV and return the specified prediction probability column."""
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")
    return df[column].dropna()


def build_box_stats(series: pd.Series) -> dict:
    """Return descriptive stats needed for drawing a boxplot."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    low_candidates = series[series >= lower_bound]
    high_candidates = series[series <= upper_bound]
    whislo = low_candidates.min() if not low_candidates.empty else series.min()
    whishi = high_candidates.max() if not high_candidates.empty else series.max()

    return {
        "med": series.median(),
        "q1": q1,
        "q3": q3,
        "whislo": whislo,
        "whishi": whishi,
        "fliers": [],
    }


perf_dir = os.path.join(
    PROJ_HOME, "Performance", "TargetScan_test", "TwoTowerTransformer", "30"
)

miniset_path = os.path.join(perf_dir, "negative_miniset_prediction.csv")
with_seed_path = os.path.join(perf_dir, "negative_with_seed_prediction.csv")

miniset_probs = load_pred_probs(miniset_path)
with_seed_probs = load_pred_probs(with_seed_path)


fig, ax = plt.subplots(figsize=(6, 3))
stats = [build_box_stats(miniset_probs), build_box_stats(with_seed_probs)]
for series, stat in zip((miniset_probs, with_seed_probs), stats):
    iqr = stat["q3"] - stat["q1"]
    lower = stat["q1"] - 1.5 * iqr
    upper = stat["q3"] + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    stat["fliers"] = outliers.tolist()
ax.set_xscale("log")
box = ax.bxp(
    stats,
    vert=False,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(linewidth=1.2, edgecolor="#4a4a4a"),
    whiskerprops=dict(linewidth=1.1, color="#4a4a4a"),
    capprops=dict(linewidth=1.1, color="#4a4a4a"),
    medianprops=dict(linewidth=2.0, color="#2c3e50"),
)

colors = ["#72B6A1", "#E99675"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

# axis facecolor set to whitesmoke
ax.set_facecolor("whitesmoke")

ax.set_yticks([1, 2])
ax.set_yticklabels(["Original Negative", "Negative w/ Seed"], fontsize=TICK_FONT_SIZE)
ax.set_xlabel("Predicted Probability (log10 scale)", fontsize=AXIS_FONT_SIZE)
ax.grid(axis="x", linestyle="--", alpha=0.4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ymin, ymax = ax.get_ylim()
if ymax != ymin:
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.1)

def log_p_two_sided(u, n1, n2):
    mean = n1 * n2 / 2
    std = mp.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mean - 0.5) / std
    # log(2) + log(erfc(|z| / sqrt(2)))
    return mp.log(2) + mp.log(mp.erfc(abs(z) / mp.sqrt(2)))

# Mann-Whitney test
result = mannwhitneyu(np.log10(miniset_probs), np.log10(with_seed_probs), method="exact")
log_p = log_p_two_sided(result.statistic, len(miniset_probs), len(with_seed_probs))
log10_p = log_p / mp.log(10)
p_value = mp.exp(log_p)
print(f"Mann-Whitney log10 p-value = {log10_p}")
print(f"Mann-Whitney p-value = {mp.nstr(mp.exp(log_p), 4)}")

# t-test
# test_result = ttest_ind(np.log10(miniset_probs), np.log10(with_seed_probs), equal_var=False)
# log_p = log_p_two_sided(test_result.statistic, len(miniset_probs), len(with_seed_probs))
# log10_p = log_p / mp.log(10)
# p_value = mp.exp(log_p)
# print(f"t-test log10 p-value = {log10_p}")
# print(f"t-test p-value = {mp.nstr(p_value, 4)}")

ax.text(
    0.5,
    1.02,
    f"Wilcoxon rank-sum p = {mp.nstr(p_value, 4)}",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=TICK_FONT_SIZE,
)

save_path = os.path.join(perf_dir, "negative_w_seed_boxplot.svg")
fig.tight_layout()
fig.savefig(save_path, dpi=500, bbox_inches="tight")
print(f"Boxplot saved to: {save_path}")
