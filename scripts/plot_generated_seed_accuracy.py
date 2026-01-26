import os
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from Global_parameters import PROJ_HOME


SEED_START_COL = "seed start"
SEED_END_COL = "seed end"
MIRNA_SEQ_COL = "miRNA sequence"
GENERATED_SEQ_COL = "generated_mirna"


def _clean_sequence(value: str) -> str:
    """Upper-case the sequence and strip whitespace so comparisons are uniform."""
    # convert all U to T in mirna sequence
    value = value.replace("U", "T")
    return "".join(str(value).upper().split()) if isinstance(value, str) else ""


def compute_accuracy(df: pd.DataFrame) -> tuple[dict[str, float], list[float]]:
    total_matches = total_bases = 0
    seed_matches = seed_bases = 0
    non_seed_matches = non_seed_bases = 0
    per_seed_accuracy: list[float] = []

    for _, row in df.iterrows():
        ref_seq = _clean_sequence(row.get(MIRNA_SEQ_COL, ""))
        gen_seq = _clean_sequence(row.get(GENERATED_SEQ_COL, ""))
        gen_seq = gen_seq[::-1] # because the generated miRNA is reversed
        if not ref_seq or not gen_seq:
            continue

        ref_len = len(ref_seq)
        gen_len = len(gen_seq)
        row_matches = 0

        for idx, base in enumerate(ref_seq):
            if idx < gen_len and base == gen_seq[idx]:
                row_matches += 1

        seed_start, seed_end = row.get(SEED_START_COL), row.get(SEED_END_COL)
        seed_len = seed_end - seed_start + 1
        row_seed_matches = 0

        mirna_start, mirna_end = 1, 1+seed_len
        
        if seed_len > 0:
            for idx in range(mirna_start, mirna_end):
                if idx < gen_len and ref_seq[idx] == gen_seq[idx]:
                    row_seed_matches += 1
            per_seed_accuracy.append(row_seed_matches / seed_len)
        else:
            per_seed_accuracy.append(math.nan)

        total_matches += row_matches
        total_bases += ref_len
        seed_matches += row_seed_matches
        seed_bases += seed_len

        non_seed_len = ref_len - seed_len
        non_seed_bases += non_seed_len
        non_seed_matches += row_matches - row_seed_matches

    metrics = {}
    if total_bases:
        metrics["Overall"] = total_matches / total_bases
    if seed_bases:
        metrics["Seed"] = seed_matches / seed_bases
    if non_seed_bases:
        metrics["Non-seed"] = non_seed_matches / non_seed_bases

    return metrics, [acc for acc in per_seed_accuracy if not math.isnan(acc)]


def plot_accuracy(metrics1: dict[str, float], metrics2: dict[str, float], color1: str, color2: str) -> None:
    """
    Plot the accuracy of the predicted miRNA seed and non-seed.
    """
    categories = ["Overall", "Seed", "Non-seed"]
    # Ensure values are in the correct order, default to 0 if missing
    values1 = [metrics1.get(cat, 0.0) for cat in categories]
    values2 = [metrics2.get(cat, 0.0) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    # Without CNN
    ax.bar(x - width/2, values1, width, color=color1, label="Full Cross Attention")
    # With CNN
    ax.bar(x + width/2, values2, width, color=color2, label="Sliding Window Cross Attention")
    
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Predicted miRNA seed vs non-seed Accuracy (randomized start validation)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    save_path = os.path.join(PROJ_HOME, "Performance", "TargetScan_test", "TwoTowerTransformer", "520", "generated_seed_and_non_seed_accuracy_500_randomized_start_validation_mini_full_and_cross_attn_comparison.png")
    # save the figure to the project home
    fig.savefig(save_path)
    print(f"Figure saved to {save_path}")

def plot_per_seed_accuracy(per_seed_accuracy1: list[float], per_seed_accuracy2: list[float], color1: str, color2: str) -> None:
    plt.figure(figsize=(4, 4))
    # plot boxplot of per_seed_accuracy with per-group colors
    bp = plt.boxplot(
        [per_seed_accuracy1, per_seed_accuracy2],
        tick_labels=["Full Cross Attention", "Sliding Window Cross Attention"],
        patch_artist=True,  # allows filled boxes
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.4),
    )
    for patch, c in zip(bp["boxes"], [color1, color2]):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
    plt.xticks(fontsize=6)
    plt.ylabel("Accuracy")
    plt.title("Predicted miRNA\naverage per-seed accuracy")
    save_path = os.path.join(PROJ_HOME, "Performance", "TargetScan_test", "TwoTowerTransformer", "520", "generated_mirna_average_per_seed_accuracy_500_randomized_start_validation_mini_full_and_cross_attn_comparison.png")
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

def main() -> None:
    DATASET_PATH1 = os.path.join(PROJ_HOME, "TargetScan_dataset", "generated_mirna_positive_primates_validation_500_randomized_start_mini_norm_by_key.csv")
    DATASET_PATH2 = os.path.join(PROJ_HOME, "TargetScan_dataset", "generated_mirna_positive_primates_validation_500_randomized_start_mini_cross_attn_norm_by_key.csv")
    
    dataframe1 = pd.read_csv(DATASET_PATH1)
    dataframe2 = pd.read_csv(DATASET_PATH2)
    metrics1, per_seed_accuracy1 = compute_accuracy(dataframe1)
    metrics2, per_seed_accuracy2 = compute_accuracy(dataframe2)
    if not metrics1 or not metrics2:
        raise RuntimeError("No valid miRNA / generated_mirna pairs were found.")
    plot_accuracy(metrics1, metrics2, color1="#A1A9AD", color2="#F07FDD")
    plot_per_seed_accuracy(per_seed_accuracy1, per_seed_accuracy2, color1="#A1A9AD", color2="#F07FDD")


if __name__ == "__main__":
    main()
