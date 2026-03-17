"""
Compare seed-complementary k-mer counts, GC content, and AU/GC-rich region
proportions between positive and negative mRNA sequences from the TargetScan
dataset.  If distributions overlap substantially, motif density is not a
confound for the discriminator.

Outputs
-------
- results/kmer_seed_match_distributions.png
- results/gc_content_distributions.png
- results/kmer_frequency_spectrum.png
- logs/compute_GC_content_and_motif_density.log
"""

import csv
import logging
import os
import sys
from collections import Counter, defaultdict
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR, "TargetScan_dataset", "TargetScan_train_500_randomized_start.csv"
)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOGS_DIR, "compute_GC_content_and_motif_density.log")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
COMPLEMENT = str.maketrans("ACGUT", "TGCAA")


def reverse_complement(seq: str) -> str:
    """Return DNA reverse complement (U→A handled)."""
    return seq.upper().translate(COMPLEMENT)[::-1]


def count_kmer_occurrences(seq: str, kmer: str) -> int:
    """Count overlapping occurrences of *kmer* in *seq*."""
    seq = seq.upper()
    kmer = kmer.upper()
    n = 0
    start = 0
    while True:
        idx = seq.find(kmer, start)
        if idx == -1:
            return n
        n += 1
        start = idx + 1


def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq) if seq else 0.0


def au_rich_fraction(seq: str, window: int = 50, threshold: float = 0.65) -> float:
    """Fraction of sliding windows where AU content >= *threshold*."""
    seq = seq.upper()
    if len(seq) < window:
        au = sum(1 for c in seq if c in "ATU")
        return float(au / len(seq) >= threshold)
    au_wins = 0
    total_wins = 0
    for i in range(len(seq) - window + 1):
        w = seq[i : i + window]
        au = sum(1 for c in w if c in "ATU")
        if au / window >= threshold:
            au_wins += 1
        total_wins += 1
    return au_wins / total_wins if total_wins else 0.0


def count_all_kmers(seq: str, k: int) -> Counter:
    """Return Counter of all k-mers in *seq*."""
    seq = seq.upper()
    c = Counter()
    for i in range(len(seq) - k + 1):
        c[seq[i : i + k]] += 1
    return c


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(path: str, max_rows: int | None = None):
    """Return lists: mirna_seqs, mrna_seqs, labels."""
    mirna_seqs, mrna_seqs, labels = [], [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            mirna_seqs.append(row["miRNA sequence"].strip())
            mrna_seqs.append(row["mRNA sequence"].strip())
            labels.append(int(row["label"]))
    return mirna_seqs, mrna_seqs, labels


# ---------------------------------------------------------------------------
# Analysis 1 – seed-complementary k-mer counts
# ---------------------------------------------------------------------------
def analyse_seed_kmer_counts(mirna_seqs, mrna_seqs, labels, k=6):
    """
    For each (miRNA, mRNA) pair extract the miRNA seed (positions 2..2+k)
    and count how many times the reverse-complement of that seed appears in
    the mRNA.  Returns arrays for positives and negatives.
    """
    pos_counts, neg_counts = [], []
    for mirna, mrna, lab in zip(mirna_seqs, mrna_seqs, labels):
        seed = mirna[1 : 1 + k]  # canonical seed starts at position 2 (0-indexed: 1)
        seed_rc = reverse_complement(seed)
        n = count_kmer_occurrences(mrna, seed_rc)
        if lab == 1:
            pos_counts.append(n)
        else:
            neg_counts.append(n)
    return np.array(pos_counts), np.array(neg_counts)


# ---------------------------------------------------------------------------
# Analysis 2 – GC content & AU-rich fraction
# ---------------------------------------------------------------------------
def analyse_gc_and_au(mrna_seqs, labels):
    pos_gc, neg_gc = [], []
    pos_au, neg_au = [], []
    for mrna, lab in zip(mrna_seqs, labels):
        g = gc_content(mrna)
        a = au_rich_fraction(mrna)
        if lab == 1:
            pos_gc.append(g)
            pos_au.append(a)
        else:
            neg_gc.append(g)
            neg_au.append(a)
    return (np.array(pos_gc), np.array(neg_gc),
            np.array(pos_au), np.array(neg_au))


# ---------------------------------------------------------------------------
# Analysis 3 – overall k-mer frequency spectrum
# ---------------------------------------------------------------------------
def analyse_kmer_spectrum(mrna_seqs, labels, k=4):
    """
    Aggregate k-mer frequency vectors for positive vs negative mRNAs, then
    compute cosine similarity and per-kmer KL divergence.
    """
    pos_total = Counter()
    neg_total = Counter()
    for mrna, lab in zip(mrna_seqs, labels):
        c = count_all_kmers(mrna, k)
        if lab == 1:
            pos_total.update(c)
        else:
            neg_total.update(c)
    all_kmers = sorted(set(pos_total) | set(neg_total))
    pos_vec = np.array([pos_total[km] for km in all_kmers], dtype=float)
    neg_vec = np.array([neg_total[km] for km in all_kmers], dtype=float)
    pos_freq = pos_vec / pos_vec.sum()
    neg_freq = neg_vec / neg_vec.sum()
    return all_kmers, pos_freq, neg_freq


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
COLORS = {"pos": "#2166ac", "neg": "#b2182b"}


def plot_seed_kmer(pos_6, neg_6, pos_7, neg_7, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, pos, neg, k in zip(axes, [pos_6, pos_7], [neg_6, neg_7], [6, 7]):
        max_val = max(pos.max(), neg.max())
        bins = np.arange(-0.5, max_val + 1.5, 1)
        ax.hist(pos, bins=bins, density=True, alpha=0.55, label=f"Positive (n={len(pos)})",
                color=COLORS["pos"], edgecolor="white", linewidth=0.4)
        ax.hist(neg, bins=bins, density=True, alpha=0.55, label=f"Negative (n={len(neg)})",
                color=COLORS["neg"], edgecolor="white", linewidth=0.4)
        stat, pval = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        effect = abs(pos.mean() - neg.mean()) / np.sqrt((pos.std()**2 + neg.std()**2) / 2) if (pos.std() + neg.std()) > 0 else 0
        ax.set_title(f"{k}-mer seed-complement counts\n"
                     f"Mann-Whitney p={pval:.2e}, Cohen's d={effect:.3f}", fontsize=11)
        ax.set_xlabel(f"# seed-complement {k}-mer matches in mRNA")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved seed k-mer plot → %s", outpath)


def plot_gc_au(pos_gc, neg_gc, pos_au, neg_au, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # GC content
    ax = axes[0]
    bins = np.linspace(0, 1, 60)
    ax.hist(pos_gc, bins=bins, density=True, alpha=0.55, label=f"Positive (n={len(pos_gc)})",
            color=COLORS["pos"], edgecolor="white", linewidth=0.4)
    ax.hist(neg_gc, bins=bins, density=True, alpha=0.55, label=f"Negative (n={len(neg_gc)})",
            color=COLORS["neg"], edgecolor="white", linewidth=0.4)
    stat, pval = stats.mannwhitneyu(pos_gc, neg_gc, alternative="two-sided")
    effect = abs(pos_gc.mean() - neg_gc.mean()) / np.sqrt((pos_gc.std()**2 + neg_gc.std()**2) / 2)
    ax.set_title(f"GC Content Distribution\n"
                 f"Mann-Whitney p={pval:.2e}, Cohen's d={effect:.3f}", fontsize=11)
    ax.set_xlabel("GC fraction")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    # AU-rich fraction
    ax = axes[1]
    bins_au = np.linspace(0, 1, 60)
    ax.hist(pos_au, bins=bins_au, density=True, alpha=0.55, label=f"Positive (n={len(pos_au)})",
            color=COLORS["pos"], edgecolor="white", linewidth=0.4)
    ax.hist(neg_au, bins=bins_au, density=True, alpha=0.55, label=f"Negative (n={len(neg_au)})",
            color=COLORS["neg"], edgecolor="white", linewidth=0.4)
    stat, pval = stats.mannwhitneyu(pos_au, neg_au, alternative="two-sided")
    effect = abs(pos_au.mean() - neg_au.mean()) / np.sqrt((pos_au.std()**2 + neg_au.std()**2) / 2)
    ax.set_title(f"AU-rich Window Fraction (w=50, thr≥0.65)\n"
                 f"Mann-Whitney p={pval:.2e}, Cohen's d={effect:.3f}", fontsize=11)
    ax.set_xlabel("Fraction of AU-rich windows")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved GC / AU-rich plot → %s", outpath)


def plot_kmer_spectrum(all_kmers, pos_freq, neg_freq, k, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter pos vs neg frequencies
    ax = axes[0]
    ax.scatter(pos_freq, neg_freq, s=6, alpha=0.5, color="#333333")
    lim = max(pos_freq.max(), neg_freq.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.8)
    cos_sim = np.dot(pos_freq, neg_freq) / (np.linalg.norm(pos_freq) * np.linalg.norm(neg_freq))
    ax.set_title(f"{k}-mer Frequency: Positive vs Negative\n"
                 f"Cosine similarity = {cos_sim:.6f}", fontsize=11)
    ax.set_xlabel("Frequency in Positives")
    ax.set_ylabel("Frequency in Negatives")

    # Log-ratio distribution
    ax = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pseudo = 1e-8
        log_ratio = np.log2((pos_freq + pseudo) / (neg_freq + pseudo))
    ax.hist(log_ratio, bins=60, color="#636363", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_title(f"Log₂ Fold-Change of {k}-mer frequencies\n"
                 f"(Positive / Negative)", fontsize=11)
    ax.set_xlabel("log₂(freq_pos / freq_neg)")
    ax.set_ylabel("Count of k-mers")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved k-mer spectrum plot → %s", outpath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("Loading data from %s", DATA_PATH)
    mirna_seqs, mrna_seqs, labels = load_data(DATA_PATH)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    log.info("Loaded %d pairs  (positive=%d, negative=%d)", len(labels), n_pos, n_neg)

    # ---- Analysis 1: seed-complementary k-mer counts (6-mer and 7-mer) ----
    log.info("=" * 60)
    log.info("Analysis 1: Seed-complementary k-mer counts")
    log.info("=" * 60)
    for k in (6, 7):
        pos_k, neg_k = analyse_seed_kmer_counts(mirna_seqs, mrna_seqs, labels, k=k)
        stat, pval = stats.mannwhitneyu(pos_k, neg_k, alternative="two-sided")
        effect = (abs(pos_k.mean() - neg_k.mean()) /
                  np.sqrt((pos_k.std()**2 + neg_k.std()**2) / 2)) if (pos_k.std() + neg_k.std()) > 0 else 0
        log.info("[%d-mer]  pos mean=%.4f±%.4f  neg mean=%.4f±%.4f  "
                 "MWU p=%.4e  Cohen's d=%.4f",
                 k, pos_k.mean(), pos_k.std(), neg_k.mean(), neg_k.std(),
                 pval, effect)
        if k == 6:
            pos_6, neg_6 = pos_k, neg_k
        else:
            pos_7, neg_7 = pos_k, neg_k

    plot_seed_kmer(pos_6, neg_6, pos_7, neg_7,
                   os.path.join(RESULTS_DIR, "kmer_seed_match_distributions.png"))

    # ---- Analysis 2: GC content & AU-rich fraction ----
    log.info("=" * 60)
    log.info("Analysis 2: GC content & AU-rich fraction")
    log.info("=" * 60)
    pos_gc, neg_gc, pos_au, neg_au = analyse_gc_and_au(mrna_seqs, labels)
    for name, pos_arr, neg_arr in [("GC content", pos_gc, neg_gc),
                                    ("AU-rich frac", pos_au, neg_au)]:
        stat, pval = stats.mannwhitneyu(pos_arr, neg_arr, alternative="two-sided")
        effect = abs(pos_arr.mean() - neg_arr.mean()) / np.sqrt((pos_arr.std()**2 + neg_arr.std()**2) / 2)
        log.info("[%s]  pos mean=%.4f±%.4f  neg mean=%.4f±%.4f  "
                 "MWU p=%.4e  Cohen's d=%.4f",
                 name, pos_arr.mean(), pos_arr.std(),
                 neg_arr.mean(), neg_arr.std(), pval, effect)

    plot_gc_au(pos_gc, neg_gc, pos_au, neg_au,
               os.path.join(RESULTS_DIR, "gc_content_distributions.png"))

    # ---- Analysis 3: overall k-mer frequency spectrum (4-mer) ----
    log.info("=" * 60)
    log.info("Analysis 3: Overall k-mer frequency spectrum")
    log.info("=" * 60)
    for k in (4, 6):
        all_kmers, pos_freq, neg_freq = analyse_kmer_spectrum(mrna_seqs, labels, k=k)
        cos_sim = np.dot(pos_freq, neg_freq) / (np.linalg.norm(pos_freq) * np.linalg.norm(neg_freq))
        jsd = 0.5 * stats.entropy(pos_freq, 0.5*(pos_freq+neg_freq)) + \
              0.5 * stats.entropy(neg_freq, 0.5*(pos_freq+neg_freq))
        log.info("[%d-mer spectrum]  n_unique_kmers=%d  cosine_sim=%.6f  JSD=%.6f",
                 k, len(all_kmers), cos_sim, jsd)
        # Top 10 most differentially enriched k-mers (positive vs negative)
        with np.errstate(divide="ignore", invalid="ignore"):
            pseudo = 1e-8
            log_ratio = np.log2((pos_freq + pseudo) / (neg_freq + pseudo))
        top_pos_idx = np.argsort(log_ratio)[-5:][::-1]
        top_neg_idx = np.argsort(log_ratio)[:5]
        log.info("  Top 5 enriched in POSITIVE:")
        for idx in top_pos_idx:
            log.info("    %s  log2FC=%.4f  pos_freq=%.6f  neg_freq=%.6f",
                     all_kmers[idx], log_ratio[idx], pos_freq[idx], neg_freq[idx])
        log.info("  Top 5 enriched in NEGATIVE:")
        for idx in top_neg_idx:
            log.info("    %s  log2FC=%.4f  pos_freq=%.6f  neg_freq=%.6f",
                     all_kmers[idx], log_ratio[idx], pos_freq[idx], neg_freq[idx])

    plot_kmer_spectrum(all_kmers, pos_freq, neg_freq, k,
                       os.path.join(RESULTS_DIR, "kmer_frequency_spectrum.png"))

    # ---- Summary ----
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("All plots saved to %s", RESULTS_DIR)
    log.info("Log saved to %s", LOG_PATH)


if __name__ == "__main__":
    main()
