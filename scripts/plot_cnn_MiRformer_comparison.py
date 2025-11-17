import os
import pandas as pd
import matplotlib.pyplot as plt
from Global_parameters import PROJ_HOME, AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE

# =========================
# File paths
# =========================
binding_file = os.path.join(PROJ_HOME, "Performance/TargetScan_test/TwoTowerTransformer/30/30nt-cnn-comparison-binding-accuracy.csv")
f1_file = os.path.join(PROJ_HOME, "Performance/TargetScan_test/TwoTowerTransformer/30/30nt-cnn-comparison-f1-score.csv")

# =========================
# Helper to load data
# =========================
def load_series(path):
    """
    Assumes:
      - first column: x-axis (e.g. Step)
      - second column: w CNN
      - third column: w/o CNN
    """
    df = pd.read_csv(path)

    # Use first three columns in order
    x_col = df.columns[0]
    w_cnn_col = df.columns[1]
    w_no_cnn_col = df.columns[2]

    x = df[x_col]
    w_cnn = df[w_cnn_col]
    w_no_cnn = df[w_no_cnn_col]

    return x, w_cnn, w_no_cnn, x_col

# Load both files
binding_x, binding_w_cnn, binding_w_no_cnn, binding_x_label = load_series(binding_file)
f1_x, f1_w_cnn, f1_w_no_cnn, f1_x_label = load_series(f1_file)

def remove_spines(ax):
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)

# =========================
# Matplotlib style
# =========================
# plt.rcParams["figure.facecolor"] = "lightgray"
plt.rcParams["axes.facecolor"] = "whitesmoke"

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

# --- Plot 1: Binding accuracy ---
ax1 = axes[0]
ax1.plot(binding_x, binding_w_cnn,
         label="w CNN",
         color="royalblue",
         linestyle="-")
ax1.plot(binding_x, binding_w_no_cnn,
         label="w/o CNN",
         color="limegreen",
         linestyle="--")

# ax1.set_title("Binding Accuracy: w CNN vs w/o CNN")
ax1.set_xlabel(binding_x_label, fontsize=AXIS_FONT_SIZE)
ax1.set_ylabel("Binding Accuracy", fontsize=AXIS_FONT_SIZE)
ax1.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
ax1.grid(True, color="white")
remove_spines(ax1)
ax1.legend(fontsize=TICK_FONT_SIZE)
# set the facecolor of the legend to white
ax1.legend(facecolor="white")

# --- Plot 2: F1-score ---
ax2 = axes[1]
ax2.plot(f1_x, f1_w_cnn,
         label="w CNN",
         color="royalblue",
         linestyle="-")
ax2.plot(f1_x, f1_w_no_cnn,
         label="w/o CNN",
         color="limegreen",
         linestyle="--")

# ax2.set_title("F1 Score: w CNN vs w/o CNN")
ax2.set_xlabel(f1_x_label, fontsize=AXIS_FONT_SIZE)
ax2.set_ylabel("F1 Score", fontsize=AXIS_FONT_SIZE)
ax2.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
ax2.grid(True, color="white")
remove_spines(ax2)
ax2.legend(fontsize=TICK_FONT_SIZE)
ax2.legend(facecolor="white")
fig.tight_layout()

# If you want to save instead of (or in addition to) showing:
file_name = os.path.join(PROJ_HOME,"Performance/TargetScan_test/TwoTowerTransformer/30/cnn_comparison.svg")
fig.savefig(file_name, dpi=500, bbox_inches="tight")
print(f"plot saved to {file_name}")
