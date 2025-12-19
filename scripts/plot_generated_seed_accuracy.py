import os
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from Global_parameters import PROJ_HOME


DATASET_PATH = os.path.join(PROJ_HOME, "TargetScan_dataset", "generated_mirna_positive_samples_30_randomized_start_validation_random_samples.csv")
SEED_START_COL = "seed start"
SEED_END_COL = "seed end"
MIRNA_SEQ_COL = "miRNA sequence"
GENERATED_SEQ_COL = "generated_mirna"


def _clean_sequence(value: str) -> str:
    """Upper-case the sequence and strip whitespace so comparisons are uniform."""
    # convert all U to T in mirna sequence
    value = value.replace("U", "T")
    return "".join(str(value).upper().split()) if isinstance(value, str) else ""


def _seed_bounds(row_seed_start, row_seed_end, seq_len: int) -> tuple[int, int]:
    """Return zero-based start and end describing the seed span."""
    if pd.isna(row_seed_start) or pd.isna(row_seed_end):
        return 0, 0

    start = max(0, int(row_seed_start))
    end = max(start, int(row_seed_end))
    if start >= seq_len:
        return seq_len, seq_len
    return start, min(seq_len, end)


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


def plot_accuracy(metrics: dict[str, float]) -> None:
    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(labels)])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Predicted miRNA seed vs non-seed Accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    save_path = os.path.join(PROJ_HOME, "Performance", "TargetScan_test", "TwoTowerTransformer", "30", "generated_seed_and_non_seed_accuracy_30_randomized_start_validation_random_samples.png")
    # save the figure to the project home
    fig.savefig(save_path)
    print(f"Figure saved to {save_path}")

def plot_per_seed_accuracy(per_seed_accuracy: list[float]) -> None:
    plt.figure(figsize=(3, 4))
    # plot boxplot of per_seed_accuracy
    plt.boxplot(per_seed_accuracy)
    plt.ylabel("Accuracy")
    plt.title("Predicted miRNA\naverage per-seed accuracy")
    save_path = os.path.join(PROJ_HOME, "Performance", "TargetScan_test", "TwoTowerTransformer", "30", "generated_mirna_average_per_seed_accuracy_30_randomized_start_validation_random_samples.png")
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

def main() -> None:
    dataframe = pd.read_csv(DATASET_PATH)
    metrics, per_seed_accuracy = compute_accuracy(dataframe)
    if not metrics:
        raise RuntimeError("No valid miRNA / generated_mirna pairs were found.")
    plot_accuracy(metrics)
    plot_per_seed_accuracy(per_seed_accuracy)


if __name__ == "__main__":
    main()
