"""
Plot DTEA (MiRformer) scalability benchmark results.

Reads benchmark_results.tsv (with model column) and produces:
  1. Inference time comparison: sliding-window vs full attention
  2. Peak GPU memory comparison: sliding-window vs full attention

OOM points are marked with an "x" marker.

Usage:
    python plot_scalability.py --input benchmark_results.tsv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_results(path):
    """Load benchmark results. Separate valid data from OOM entries."""
    df = pd.read_csv(path, sep="\t")

    # Separate OOM rows
    oom_mask = df["median_time_s"] == "OOM"
    df_oom   = df[oom_mask].copy()
    df_valid = df[~oom_mask].copy()

    df_valid["median_time_s"] = df_valid["median_time_s"].astype(float)
    df_valid["std_time_s"]    = df_valid["std_time_s"].astype(float)
    df_valid["median_mem_MB"] = df_valid["median_mem_MB"].astype(float)
    df_valid["std_mem_MB"]    = df_valid["std_mem_MB"].astype(float)

    return df_valid, df_oom


def format_length_label(x):
    """500 -> '0.5k', 1000 -> '1k', 25000 -> '25k'."""
    if x >= 1000:
        return f"{int(x/1000)}k"
    elif x == 500:
        return "0.5k"
    return str(int(x))


# ── Style config ──
STYLES = {
    "Sliding-window": {
        "color": "#2C6FAC",
        "marker": "o",
        "label": "MiRformer (sliding-window)",
    },
    "Full attention": {
        "color": "#D94F3B",
        "marker": "s",
        "label": "Full attention",
    },
    "RNAhybrid": {
        "color": "#50C878",
        "marker": "^",
        "label": "RNAhybrid"
    }
}


def plot_time(df_valid, df_oom, all_lengths, output_path, dpi=300):
    """Line plot: inference time comparison."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for mode, style in STYLES.items():
        subset = df_valid[df_valid["model"] == mode]
        if subset.empty:
            continue

        lengths = subset["mrna_length"].values
        times   = subset["median_time_s"].values
        errs    = subset["std_time_s"].values

        ax.errorbar(lengths, times, yerr=errs,
                    fmt=f"-{style['marker']}", color=style["color"],
                    markersize=6, capsize=3, linewidth=1.8,
                    markerfacecolor="white", markeredgewidth=1.5,
                    label=style["label"])

        # Mark OOM: dashed extension from last valid point to first OOM length
        oom_subset = df_oom[df_oom["model"] == mode]
        if not oom_subset.empty and len(lengths) > 0:
            first_oom_len = int(oom_subset["mrna_length"].iloc[0])
            last_time     = times[-1]
            oom_y         = last_time * 1.35  # extrapolate upward
            ax.plot([lengths[-1], first_oom_len], [last_time, oom_y],
                    "--", color=style["color"], alpha=0.5, linewidth=1.5)
            ax.scatter([first_oom_len], [oom_y], marker="x",
                       color=style["color"], s=120, linewidths=2.5, zorder=5)
            ax.annotate("OOM", (first_oom_len, oom_y),
                        textcoords="offset points", xytext=(8, -2),
                        fontsize=9, color=style["color"], fontweight="bold")

    ax.set_xlabel("mRNA sequence length (nt)", fontsize=12)
    ax.set_ylabel("Inference time (seconds)",  fontsize=12)
    ax.set_title("Inference Time vs. mRNA Length", fontsize=13, pad=10)

    ax.set_xticks(all_lengths)
    ax.set_xticklabels([format_length_label(l) for l in all_lengths], fontsize=10)
    ax.set_xlim(left=-all_lengths.max() * 0.03)
    ax.set_ylim(bottom=0)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_memory(df_valid, df_oom, all_lengths, output_path, dpi=300):
    """Line plot: peak GPU memory comparison."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Decide units based on max memory across both modes
    max_mem = df_valid["median_mem_MB"].max() if not df_valid.empty else 0
    use_gb  = max_mem > 4000

    for mode, style in STYLES.items():
        subset = df_valid[df_valid["model"] == mode]
        if subset.empty:
            continue

        lengths = subset["mrna_length"].values
        mems    = subset["median_mem_MB"].values.copy()
        errs    = subset["std_mem_MB"].values.copy()

        if use_gb:
            mems = mems / 1024
            errs = errs / 1024

        ax.errorbar(lengths, mems, yerr=errs,
                    fmt=f"-{style['marker']}", color=style["color"],
                    markersize=6, capsize=3, linewidth=1.8,
                    markerfacecolor="white", markeredgewidth=1.5,
                    label=style["label"])

        # Mark OOM: dashed extension from last valid point to first OOM length
        oom_subset = df_oom[df_oom["model"] == mode]
        if not oom_subset.empty and len(lengths) > 0:
            first_oom_len = int(oom_subset["mrna_length"].iloc[0])
            last_mem      = mems[-1]
            oom_y         = last_mem * 1.35
            ax.plot([lengths[-1], first_oom_len], [last_mem, oom_y],
                    "--", color=style["color"], alpha=0.5, linewidth=1.5)
            ax.scatter([first_oom_len], [oom_y], marker="x",
                       color=style["color"], s=120, linewidths=2.5, zorder=5)
            ax.annotate("OOM", (first_oom_len, oom_y),
                        textcoords="offset points", xytext=(8, -2),
                        fontsize=9, color=style["color"], fontweight="bold")

    ylabel  = "Peak GPU/CPU memory (GB)" if use_gb else "Peak GPU/CPU memory (MB)"
    fmt_str = "%.1f" if use_gb else "%.0f"

    ax.set_xlabel("mRNA sequence length (nt)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Peak GPU/CPU Memory vs. mRNA Length", fontsize=13, pad=10)

    ax.set_xticks(all_lengths)
    ax.set_xticklabels([format_length_label(l) for l in all_lengths], fontsize=10)
    ax.set_xlim(left=-all_lengths.max() * 0.03)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt_str))

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    import os
    from Global_parameters import PROJ_HOME
    input_path = os.path.join(PROJ_HOME, "results", "benchmark_results.tsv")
    dpi = 300   # figure resolution
    df_valid, df_oom = load_results(input_path)
    print(f"Loaded {len(df_valid)} valid + {len(df_oom)} OOM data points from {input_path}\n")
    print(df_valid.to_string(index=False))
    if not df_oom.empty:
        print(f"\nOOM entries:")
        print(df_oom[["model", "mrna_length"]].to_string(index=False))
    print()

    # Collect all unique lengths for x-axis (union of both modes)
    all_lengths = np.sort(
        pd.concat([df_valid["mrna_length"], df_oom["mrna_length"].astype(int)])
        .unique()
    )

    base = input_path.replace(".tsv", "")
    plot_time(df_valid,   df_oom, all_lengths, f"{base}_time.svg",   dpi=dpi)
    plot_memory(df_valid, df_oom, all_lengths, f"{base}_memory.svg", dpi=dpi)


if __name__ == "__main__":
    main()
