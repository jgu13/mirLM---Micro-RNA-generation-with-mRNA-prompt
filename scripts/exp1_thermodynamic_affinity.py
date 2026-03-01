#!/usr/bin/env python3
"""
Experiment 1: Thermodynamic Binding Affinity (MFE Analysis)
============================================================

For each generated miRNA-mRNA pair in the 30nt, 100nt, and 500nt test sets:
  1. Compute MFE (ΔG) of generated miRNA : mRNA duplex via RNAduplex (ViennaRNA)
  2. Compute MFE of real (ground-truth) miRNA : mRNA duplex
  3. Compute MFE of random seed-match control miRNA : mRNA duplex
     (same seed complement, randomised flanking nucleotides)
  4. Compute MFE of shuffled mRNA control : generated miRNA duplex
     (dinucleotide shuffle of the mRNA preserving composition)
  5. Wilcoxon rank-sum test (generated vs. random seed-match)
  6. Plot MFE distribution curves

Outputs
-------
  results/exp1_mfe_comparison.csv
  plots/exp1_mfe_distribution.pdf

Requirements
------------
  RNAduplex (ViennaRNA ≥ 2.4) must be on PATH, or set RNADUPLEX_BIN below.
  Python packages: pandas, numpy, scipy, matplotlib
"""

import os
import sys
import random
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJ_HOME = os.path.expanduser("~/projects/MiRformer")

# ViennaRNA RNAduplex binary (update if not on PATH)
RNADUPLEX_BIN = "/home/mcb/users/jgu13/.conda/envs/HyenaDNA_clone2/bin/RNAduplex"

DATA_FILES = {
    "30nt": os.path.join(
        PROJ_HOME, "TargetScan_dataset",
        "generated_mirna_positive_samples_30_randomized_start_test.csv"
    ),
    "100nt": os.path.join(
        PROJ_HOME, "TargetScan_dataset",
        "generated_mirna_positive_primates_test_100_randomized_start_local_self_attn_full_cross_attn.csv"
    ),
    "500nt": os.path.join(
        PROJ_HOME, "TargetScan_dataset",
        "generated_mirna_positive_primates_test_500_randomized_start_local_self_attn_full_cross_attn.csv"
    ),
}

RESULTS_DIR = os.path.join(PROJ_HOME, "results")
PLOTS_DIR   = os.path.join(PROJ_HOME, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASES = list("ACGU")

# ── Helpers ────────────────────────────────────────────────────────────────────

def to_rna(seq: str) -> str:
    """Convert DNA/mixed to RNA (T→U, uppercase)."""
    return seq.upper().replace("T", "U")


def reverse_complement_rna(seq: str) -> str:
    """Return 5'→3' reverse complement of an RNA sequence."""
    comp = {"A": "U", "U": "A", "C": "G", "G": "C", "N": "N",
            "T": "A"}  # also handle T just in case
    return "".join(comp.get(b, "N") for b in seq.upper()[::-1])


def dinucleotide_shuffle(seq: str, rng: random.Random | None = None) -> str:
    """
    Dinucleotide shuffle preserving mono- and dinucleotide frequencies
    (Altschul & Erickson 1985 approach).
    Falls back to simple character shuffle if Eulerian path fails.
    """
    if rng is None:
        rng = random
    seq = seq.upper()
    n = len(seq)
    if n <= 2:
        s = list(seq); rng.shuffle(s); return "".join(s)

    # Build adjacency list of successors for each character
    adj: dict[str, list[str]] = defaultdict(list)
    for i in range(n - 1):
        adj[seq[i]].append(seq[i + 1])
    for k in adj:
        rng.shuffle(adj[k])

    # Hierholzer's Eulerian path algorithm
    start = seq[0]
    path: list[str] = []
    stack = [start]
    local_adj = {k: list(v) for k, v in adj.items()}
    while stack:
        v = stack[-1]
        if local_adj.get(v):
            stack.append(local_adj[v].pop())
        else:
            path.append(stack.pop())

    result = path[::-1]
    if len(result) != n:
        # Fallback: simple shuffle
        s = list(seq); rng.shuffle(s); return "".join(s)
    return "".join(result)


def make_random_seed_match_mirna(mirna_len: int, mrna_seq: str,
                                  seed_s: int, seed_e: int) -> str:
    """
    Build a random miRNA of the same length as the real one that
    contains a valid seed match to the mRNA.

    Strategy:
      - Seed positions of miRNA (1-indexed 2-8, 0-indexed 1:8) are set to
        the reverse complement of the mRNA seed window [seed_s:seed_e].
      - All other positions are filled with random RNA nucleotides.
    """
    mrna_seed_region = to_rna(mrna_seq[seed_s:seed_e])
    mirna_seed_frag  = reverse_complement_rna(mrna_seed_region)   # length = seed_e - seed_s

    # miRNA layout: [pos0] [seed pos 1-7] [pos8 .. end]
    pre_len  = 1                                          # position 0 (5'-most nt)
    post_len = mirna_len - pre_len - len(mirna_seed_frag) # remaining positions

    pre  = [random.choice(BASES) for _ in range(max(pre_len, 0))]
    post = [random.choice(BASES) for _ in range(max(post_len, 0))]

    rand_mirna = pre + list(mirna_seed_frag) + post
    # Ensure exact length (trim or pad)
    if len(rand_mirna) > mirna_len:
        rand_mirna = rand_mirna[:mirna_len]
    elif len(rand_mirna) < mirna_len:
        rand_mirna += [random.choice(BASES)] * (mirna_len - len(rand_mirna))

    return "".join(rand_mirna)


# ── RNAduplex wrapper ──────────────────────────────────────────────────────────

def parse_rnaduplex_output(line: str) -> float:
    """Parse a single RNAduplex output line and return the MFE value."""
    # Format: "structure  i1,j1  :  i2,j2  (mfe)"
    line = line.strip()
    if not line:
        return np.nan
    try:
        # Last token is "(mfe)" or "-inf"
        last = line.split()[-1]
        return float(last.strip("()"))
    except (ValueError, IndexError):
        return np.nan


def compute_mfe_batch(seq_pairs: list[tuple[str, str]]) -> list[float]:
    """
    Call RNAduplex once for a batch of (seq1, seq2) pairs.
    seq1 = miRNA (RNA), seq2 = mRNA (RNA)
    Returns a list of MFE floats (np.nan on failure).
    """
    if not seq_pairs:
        return []

    # Build stdin: alternating seq1 / seq2 lines, terminated by "@"
    stdin_lines = []
    for s1, s2 in seq_pairs:
        stdin_lines.append(to_rna(s1))
        stdin_lines.append(to_rna(s2))
    stdin_text = "\n".join(stdin_lines) + "\n@\n"

    try:
        result = subprocess.run(
            [RNADUPLEX_BIN, "--noLP"],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=600
        )
        output_lines = [l for l in result.stdout.splitlines() if l.strip()]
        mfes = [parse_rnaduplex_output(l) for l in output_lines]
        # Pad with NaN if output is shorter than expected
        while len(mfes) < len(seq_pairs):
            mfes.append(np.nan)
        return mfes[:len(seq_pairs)]
    except Exception as e:
        print(f"  [WARNING] RNAduplex subprocess failed: {e}")
        return [np.nan] * len(seq_pairs)


# ── Per-dataset processing ─────────────────────────────────────────────────────

def process_dataset(name: str, filepath: str,
                    batch_size: int = 200) -> pd.DataFrame:
    """
    For one length category, compute MFE for all four conditions
    (generated miRNA, real miRNA, random seed-match, shuffled mRNA).
    """
    print(f"\n{'='*60}")
    print(f"Processing {name}: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} rows")

    # Gather pairs for each condition
    gen_pairs  = []   # (generated_mirna, mRNA)
    real_pairs = []   # (real miRNA,      mRNA)
    rand_pairs = []   # (random seed-match, mRNA)
    shuf_pairs = []   # (generated_mirna,  shuffled mRNA)
    rand_mirnas = []

    for _, row in df.iterrows():
        mrna      = str(row["mRNA sequence"])
        gen_mirna = str(row["generated_mirna"])
        real_mirna= str(row["miRNA sequence"])
        seed_s    = int(row["seed start"])
        seed_e    = int(row["seed end"])

        gen_pairs.append((gen_mirna, mrna))
        real_pairs.append((real_mirna, mrna))

        rand_mirna = make_random_seed_match_mirna(
            len(gen_mirna), mrna, seed_s, seed_e
        )
        rand_pairs.append((rand_mirna, mrna))
        rand_mirnas.append(rand_mirna)

        shuf_mrna = dinucleotide_shuffle(mrna)
        shuf_pairs.append((gen_mirna, shuf_mrna))

    # Batch-compute MFEs
    def batch_mfe(pairs, label):
        mfes = []
        n = len(pairs)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            if start % 1000 == 0:
                print(f"  [{label}] {start}/{n} ...")
            batch = pairs[start:end]
            mfes.extend(compute_mfe_batch(batch))
        return mfes

    print("  Computing MFE for generated miRNA pairs ...")
    mfe_gen  = batch_mfe(gen_pairs,  "generated")
    print("  Computing MFE for real miRNA pairs ...")
    mfe_real = batch_mfe(real_pairs, "real")
    print("  Computing MFE for random seed-match controls ...")
    mfe_rand = batch_mfe(rand_pairs, "random")
    print("  Computing MFE for shuffled mRNA controls ...")
    mfe_shuf = batch_mfe(shuf_pairs, "shuffled")

    result_df = df[["Transcript ID", "miRNA ID", "seed start", "seed end"]].copy()
    result_df["mRNA_length"]           = name
    result_df["generated_mirna"]       = [p[0] for p in gen_pairs]
    result_df["real_mirna"]            = [p[0] for p in real_pairs]
    result_df["random_seed_mirna"]     = rand_mirnas
    result_df["MFE_generated"]         = mfe_gen
    result_df["MFE_real_mirna"]        = mfe_real
    result_df["MFE_random_seed_match"] = mfe_rand
    result_df["MFE_shuffled_mRNA"]     = mfe_shuf
    return result_df


# ── Statistics ─────────────────────────────────────────────────────────────────

def wilcoxon_less(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """
    Mann-Whitney U test: H1 = a < b (a has lower / more negative MFE).
    Returns (U-statistic, p-value).
    """
    a_clean = a.dropna().values
    b_clean = b.dropna().values
    if len(a_clean) < 5 or len(b_clean) < 5:
        return np.nan, np.nan
    stat, pval = stats.mannwhitneyu(a_clean, b_clean, alternative="less")
    return stat, pval


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_mfe_distributions(all_df: pd.DataFrame, out_pdf: str) -> None:
    """One figure with 3 sub-plots (one per mRNA length)."""
    lengths = ["30nt", "100nt", "500nt"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colours = {
        "MFE_generated":         ("royalblue",  "Generated miRNA"),
        "MFE_real_mirna":        ("forestgreen","Real miRNA (GT)"),
        "MFE_random_seed_match": ("darkorange",  "Random (seed-match ctrl)"),
        "MFE_shuffled_mRNA":     ("firebrick",   "Shuffled mRNA ctrl"),
    }

    for ax, name in zip(axes, lengths):
        sub = all_df[all_df["mRNA_length"] == name]
        for col, (colour, label) in colours.items():
            vals = sub[col].dropna()
            ax.hist(vals, bins=60, alpha=0.55, color=colour, label=label,
                    density=True)
            ax.axvline(vals.median(), color=colour, linestyle="--",
                       linewidth=1.2, alpha=0.85)

        # Wilcoxon annotation
        stat, pval = wilcoxon_less(sub["MFE_generated"],
                                   sub["MFE_random_seed_match"])
        pval_str = f"p={pval:.2e}" if not np.isnan(pval) else "p=N/A"

        ax.set_xlabel("MFE (kcal/mol)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"{name} mRNA\nGenerated vs Random ctrl: {pval_str}",
                     fontsize=12)
        ax.legend(fontsize=8)

    plt.suptitle("Experiment 1 – Thermodynamic Binding Affinity (ΔG)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out_pdf}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_dfs = []
    for name, filepath in DATA_FILES.items():
        df = process_dataset(name, filepath)
        all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp1_mfe_comparison.csv")
    all_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved → {csv_path}")

    # Statistical summary
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    for name in DATA_FILES:
        sub = all_df[all_df["mRNA_length"] == name]
        print(f"\n{name} (n={len(sub)}):")
        for col in ["MFE_generated", "MFE_real_mirna",
                    "MFE_random_seed_match", "MFE_shuffled_mRNA"]:
            vals = sub[col].dropna()
            print(f"  {col:35s}: mean={vals.mean():.3f}  "
                  f"median={vals.median():.3f}  std={vals.std():.3f}")
        stat, pval = wilcoxon_less(sub["MFE_generated"],
                                   sub["MFE_random_seed_match"])
        print(f"  Wilcoxon (generated < random seed-match): "
              f"U={stat:.0f}, p={pval:.4e}")
        stat2, pval2 = wilcoxon_less(sub["MFE_generated"],
                                     sub["MFE_shuffled_mRNA"])
        print(f"  Wilcoxon (generated < shuffled mRNA):     "
              f"U={stat2:.0f}, p={pval2:.4e}")

    # Plot
    plot_path = os.path.join(PLOTS_DIR, "exp1_mfe_distribution.pdf")
    plot_mfe_distributions(all_df, plot_path)

    print("\n" + "="*60)
    print("DONE")
    print(f"  CSV  → {csv_path}")
    print(f"  Plot → {plot_path}")


if __name__ == "__main__":
    main()
