#!/usr/bin/env python3
"""
Experiment 2: Transcriptome-wide Off-Target Specificity
=========================================================

Steps:
  1. Extract 3'UTR sequences from the human cDNA FASTA (hg19_cDNA.fa)
     using the GENCODE v19 GTF annotation. Only 3'UTR exons are extracted.
     (Cached to hg19_3UTR_extracted.fa after first run.)
  2. Build a k-mer hash index (k=6,7,8) mapping each k-mer → set of
     transcript IDs containing it. O(total_UTR_length) to build, O(1) lookup.
  3. For each generated miRNA, extract the seed region (positions 2–8) and
     look up canonical matches (8-mer, 7mer-m8, 7mer-A1, 6-mer) in the index.
  4. Off-target count = |matching transcripts| − {true target transcript}.
  5. Specificity Index: SI = 1 / (off_target_count + 1)
  6. Compare: generated miRNA vs real (GT) miRNA vs random seed-match control.

Outputs
-------
  results/exp2_off_target_analysis.csv
  plots/exp2_off_target_histogram.pdf
  plots/exp2_specificity_comparison.pdf

Requirements
------------
  hg19_cDNA.fa   → miR_degradome_ago_clip_pairing_data/hg19_cDNA.fa
  gencode.v19.annotation.gtf  (already present in same directory)
  Python packages: pandas, numpy, scipy, matplotlib
"""

import os
import sys
import random
import gzip
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# Allow imports from scripts/
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from Global_parameters import PROJ_HOME

# ── Paths ──────────────────────────────────────────────────────────────────────
DEGR_DIR   = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data")
GTF_PATH   = os.path.join(DEGR_DIR,  "gencode.v19.annotation.gtf")
CDNA_FA    = os.path.join(DEGR_DIR,  "hg19_cDNA.fa")
UTR3_FASTA = os.path.join(DEGR_DIR,  "hg19_3UTR_extracted.fa")    # cached
KMER_INDEX = os.path.join(DEGR_DIR,  "hg19_3UTR_kmer_count.pkl")  # cached

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

BASES = list("ACGT")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Extract / load 3'UTR sequences
# ══════════════════════════════════════════════════════════════════════════════

def parse_gtf_utr3_coords(gtf_path: str) -> dict:
    """Parse GENCODE v19 GTF → {transcript_id: [(chrom, start, end, strand)]}
    for protein-coding 3'UTR exons only."""
    utr3: dict[str, list] = defaultdict(list)
    print(f"Parsing GTF: {gtf_path}", flush=True)
    opener = gzip.open if gtf_path.endswith(".gz") else open
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip().split("\t")
            if len(f) < 9 or f[2] != "UTR":
                continue
            gt = re.search(r'gene_type "([^"]+)"', f[8])
            if gt and gt.group(1) != "protein_coding":
                continue
            m = re.search(r'transcript_id "([^"]+)"', f[8])
            if not m:
                continue
            utr3[m.group(1)].append(
                (f[0], int(f[3]), int(f[4]), f[6])
            )
    print(f"  {len(utr3)} transcripts with 3'UTR annotations", flush=True)
    return utr3


def load_fasta(fasta_path: str) -> dict[str, str]:
    """Load FASTA → {seq_id_without_version: sequence}."""
    seqs: dict[str, str] = {}
    opener = gzip.open if fasta_path.endswith(".gz") else open
    cur_id, buf = None, []
    with opener(fasta_path, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(buf).upper()
                cur_id = line[1:].split()[0].split(".")[0]
                buf = []
            else:
                buf.append(line)
    if cur_id:
        seqs[cur_id] = "".join(buf).upper()
    print(f"  Loaded {len(seqs)} sequences from {fasta_path}", flush=True)
    return seqs


def extract_utr3_sequences(
    utr3_by_tx: dict, cdna_seqs: dict[str, str], out_fasta: str
) -> dict[str, str]:
    """
    Approximate 3'UTR extraction: take the last N nucleotides of the cDNA
    where N = sum of annotated UTR exon lengths for that transcript.
    Writes results to out_fasta (cached).
    """
    if os.path.exists(out_fasta):
        print(f"Loading cached 3'UTR FASTA: {out_fasta}", flush=True)
        return load_fasta(out_fasta)

    print("Extracting 3'UTR sequences ...", flush=True)
    utr3_seqs: dict[str, str] = {}
    for tx_id, coords in utr3_by_tx.items():
        tx_base  = tx_id.split(".")[0]
        cdna     = cdna_seqs.get(tx_base, "")
        utr_len  = min(sum(e - s + 1 for _, s, e, _ in coords), len(cdna))
        if utr_len < 6:
            continue
        utr3_seqs[tx_base] = cdna[-utr_len:]

    print(f"  {len(utr3_seqs)} 3'UTR sequences extracted", flush=True)
    with open(out_fasta, "w") as fh:
        for tid, seq in utr3_seqs.items():
            fh.write(f">{tid}\n{seq}\n")
    print(f"  Saved → {out_fasta}", flush=True)
    return utr3_seqs


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Build k-mer hash index  (k = 6, 7, 8)
# ══════════════════════════════════════════════════════════════════════════════

def build_kmer_count_index(
    utr3_seqs: dict[str, str],
    ks: tuple[int, ...] = (6, 7, 8),
    cache_path: str | None = None,
) -> dict[int, dict[str, int]]:
    """
    Build a count-based k-mer index: {k → {kmer: num_transcripts_containing_kmer}}.

    For each transcript, each unique k-mer within that transcript is counted
    once (so counts reflect transcript-level presence, not position count).
    This is memory-efficient (~2 MB total) versus storing full tx_id sets.

    True-target exclusion is handled at lookup time by checking the target's
    UTR sequence directly.

    Returns
    -------
    index : dict[k → dict[kmer_str → int]]
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached k-mer count index: {cache_path}", flush=True)
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    print(f"Building k-mer count index (k={ks}) over {len(utr3_seqs)} UTRs ...",
          flush=True)
    index: dict[int, dict[str, int]] = {k: {} for k in ks}
    n = len(utr3_seqs)

    for i, (tx_id, seq) in enumerate(utr3_seqs.items()):
        if i % 10000 == 0:
            print(f"  Indexed {i}/{n} ...", flush=True)
        for k in ks:
            seen: set[str] = set()
            kd = index[k]
            for j in range(len(seq) - k + 1):
                kmer = seq[j: j + k]
                if kmer not in seen:
                    seen.add(kmer)
                    kd[kmer] = kd.get(kmer, 0) + 1

    if cache_path:
        print(f"  Saving count index → {cache_path}", flush=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(index, fh, protocol=4)

    sizes = {k: len(v) for k, v in index.items()}
    total_mb = sum(len(v) * 30 for v in index.values()) / 1e6
    print(f"  Index sizes: {sizes}  (est. {total_mb:.0f} MB)", flush=True)
    return index


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Seed extraction & O(1) off-target lookup
# ══════════════════════════════════════════════════════════════════════════════

def dna_complement(seq: str) -> str:
    c = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A", "N": "N"}
    return "".join(c.get(b, "N") for b in seq.upper())


def to_dna(seq: str) -> str:
    return seq.upper().replace("U", "T")


def extract_seed_patterns(mirna: str) -> dict[str, str]:
    """
    Compute the four canonical seed-match target strings for a given miRNA.
    mirna is in 3'→5' storage order; we reverse to 5'→3' first.
    Returns DNA strings to search in mRNA / 3'UTR sequences.
    """
    fwd = to_dna(mirna[::-1])                # 5'→3' DNA
    seed7  = fwd[1:8]                        # positions 2-8
    seed7_rc = dna_complement(seed7)[::-1]   # RC

    seed6  = fwd[1:7]                        # positions 2-7
    seed6_rc = dna_complement(seed6)[::-1]

    return {
        "8-mer":    seed7_rc + "A",
        "7-mer-m8": seed7_rc,
        "7-mer-A1": seed6_rc + "A",
        "6-mer":    seed6_rc,
    }


def count_off_targets_indexed(
    mirna: str,
    true_tx_id: str,
    count_index: dict[int, dict[str, int]],
    utr3_seqs: dict[str, str],
) -> tuple[int, int, int, int]:
    """
    Return (total_off_targets, n_8mer, n_7m8, n_7a1) using count-based index.

    For each canonical seed pattern (8-mer, 7mer-m8, 7mer-A1, 6-mer):
      - Look up the transcript count from count_index
      - Subtract 1 if the true target's UTR also contains this pattern
        (done by direct string search on the cached UTR, very fast)

    NOTE: Counts are ADDITIVE across match types (one transcript may appear
    in multiple type counts). For a conservative total, we use the 6-mer
    count which is a superset of all stronger matches.
    """
    patterns = extract_seed_patterns(mirna)
    true_base = true_tx_id.split(".")[0]
    true_utr  = utr3_seqs.get(true_base, "")

    def adjusted(k, mtype):
        pat   = patterns[mtype]
        count = count_index[k].get(pat, 0)
        if pat in true_utr:
            count = max(0, count - 1)
        return count

    n_8   = adjusted(8, "8-mer")
    n_7m8 = adjusted(7, "7-mer-m8")
    n_7a1 = adjusted(7, "7-mer-A1")
    # Total = conservative union approximated as 6-mer count
    # (any transcript with a 6-mer also has all stronger matches counted above)
    total = adjusted(6, "6-mer")
    return total, n_8, n_7m8, n_7a1


def make_random_seed_match_mirna(mirna_len: int, mrna: str,
                                  seed_s: int, seed_e: int) -> str:
    """Generate a random miRNA of same length with a valid seed complement."""
    mrna_seed   = to_dna(mrna)[seed_s:seed_e]
    mirna_seed  = dna_complement(mrna_seed)[::-1]
    pre  = [random.choice(BASES) for _ in range(1)]
    post = [random.choice(BASES) for _ in range(max(mirna_len - 1 - len(mirna_seed), 0))]
    result = (pre + list(mirna_seed) + post)[:mirna_len]
    if len(result) < mirna_len:
        result += [random.choice(BASES)] * (mirna_len - len(result))
    return "".join(result)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_off_target_histogram(all_df: pd.DataFrame, out_pdf: str) -> None:
    lengths = ["30nt", "100nt", "500nt"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name in zip(axes, lengths):
        sub = all_df[all_df["mRNA_length"] == name]
        max_val = max(
            sub["gen_off_target_count"].max(),
            sub["real_off_target_count"].max(),
            sub["rand_off_target_count"].max(),
        )
        bins = min(80, int(max_val) + 2)
        ax.hist(sub["gen_off_target_count"],  bins=bins, alpha=0.6,
                color="royalblue",  label="Generated miRNA", density=True)
        ax.hist(sub["real_off_target_count"], bins=bins, alpha=0.6,
                color="forestgreen",label="Real miRNA (GT)", density=True)
        ax.hist(sub["rand_off_target_count"], bins=bins, alpha=0.6,
                color="darkorange", label="Random seed-match", density=True)
        u, p = stats.mannwhitneyu(
            sub["gen_off_target_count"], sub["real_off_target_count"],
            alternative="two-sided"
        )
        ax.set_title(f"{name}\nGen vs Real p={p:.2e}", fontsize=11)
        ax.set_xlabel("Off-target transcript count", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8)
    plt.suptitle("Experiment 2 – Transcriptome-wide Off-Target Count",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_pdf}", flush=True)


def plot_specificity_comparison(all_df: pd.DataFrame, out_pdf: str) -> None:
    lengths = ["30nt", "100nt", "500nt"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colours = ["royalblue", "forestgreen", "darkorange"]
    for ax, name in zip(axes, lengths):
        sub  = all_df[all_df["mRNA_length"] == name]
        data = [sub["gen_si"].values, sub["real_si"].values, sub["rand_si"].values]
        bp   = ax.boxplot(data, patch_artist=True)
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
        ax.set_xticklabels(["Generated", "Real miRNA", "Random"], fontsize=9)
        ax.set_title(f"{name}", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("Specificity Index (1/(off-targets+1))", fontsize=11)
    plt.suptitle("Experiment 2 – Specificity Index Comparison",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_pdf}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Validate input
    if not os.path.exists(CDNA_FA):
        raise FileNotFoundError(
            f"hg19_cDNA.fa not found at: {CDNA_FA}\n"
            "Please place it in miR_degradome_ago_clip_pairing_data/."
        )

    # ── 1. Build / load 3'UTR sequence database ────────────────────────────
    print("=" * 65, flush=True)
    print("STEP 1 – 3'UTR sequence database", flush=True)
    print("=" * 65, flush=True)

    if os.path.exists(UTR3_FASTA):
        print(f"Loading cached 3'UTR FASTA: {UTR3_FASTA}", flush=True)
        utr3_seqs = load_fasta(UTR3_FASTA)
    else:
        utr3_by_tx = parse_gtf_utr3_coords(GTF_PATH)
        cdna_seqs  = load_fasta(CDNA_FA)
        utr3_seqs  = extract_utr3_sequences(utr3_by_tx, cdna_seqs, UTR3_FASTA)

    print(f"3'UTR database: {len(utr3_seqs)} transcripts", flush=True)

    # ── 2. Build k-mer count index ─────────────────────────────────────────
    print("\n" + "=" * 65, flush=True)
    print("STEP 2 – Building k-mer count index (k=6,7,8)", flush=True)
    print("=" * 65, flush=True)
    index = build_kmer_count_index(utr3_seqs, ks=(6, 7, 8), cache_path=KMER_INDEX)

    # ── 3. Process each dataset ────────────────────────────────────────────
    print("\n" + "=" * 65, flush=True)
    print("STEP 3 – Off-target lookup per generated miRNA", flush=True)
    print("=" * 65, flush=True)

    all_rows = []
    for name, filepath in DATA_FILES.items():
        print(f"\nDataset: {name}", flush=True)
        df = pd.read_csv(filepath)
        print(f"  {len(df)} samples", flush=True)

        for i, row in df.iterrows():
            if i % 1000 == 0:
                print(f"  {i}/{len(df)} ...", flush=True)

            mrna       = str(row["mRNA sequence"])
            gen_mirna  = str(row["generated_mirna"])
            real_mirna = str(row["miRNA sequence"])
            tx_id      = str(row["Transcript ID"])
            seed_s     = int(row["seed start"])
            seed_e     = int(row["seed end"])
            rand_mirna = make_random_seed_match_mirna(
                len(gen_mirna), mrna, seed_s, seed_e
            )

            gen_ot,  g8, g7m, g7a = count_off_targets_indexed(gen_mirna,  tx_id, index, utr3_seqs)
            real_ot, r8, r7m, r7a = count_off_targets_indexed(real_mirna, tx_id, index, utr3_seqs)
            rand_ot, _,  _,   _   = count_off_targets_indexed(rand_mirna, tx_id, index, utr3_seqs)

            all_rows.append({
                "mRNA_length":        name,
                "Transcript_ID":      tx_id,
                "miRNA_ID":           row["miRNA ID"],
                "gen_off_target_count":  gen_ot,
                "gen_8mer_off":          g8,
                "gen_7m8_off":           g7m,
                "gen_7a1_off":           g7a,
                "real_off_target_count": real_ot,
                "real_8mer_off":         r8,
                "real_7m8_off":          r7m,
                "real_7a1_off":          r7a,
                "rand_off_target_count": rand_ot,
                "gen_si":   1 / (gen_ot  + 1),
                "real_si":  1 / (real_ot + 1),
                "rand_si":  1 / (rand_ot + 1),
            })

    all_df = pd.DataFrame(all_rows)

    # ── 4. Statistical summary ─────────────────────────────────────────────
    print("\n" + "=" * 65, flush=True)
    print("STATISTICAL SUMMARY", flush=True)
    print("=" * 65, flush=True)
    for name in DATA_FILES:
        sub = all_df[all_df["mRNA_length"] == name]
        print(f"\n{name} (n={len(sub)}):", flush=True)
        for col in ["gen_off_target_count", "real_off_target_count",
                    "rand_off_target_count"]:
            vals = sub[col]
            print(f"  {col:30s}: mean={vals.mean():.1f}  "
                  f"median={vals.median():.0f}  "
                  f"mean_SI={1/(vals.mean()+1):.4f}",
                  flush=True)
        u, pval = stats.mannwhitneyu(
            sub["gen_off_target_count"], sub["real_off_target_count"],
            alternative="two-sided"
        )
        print(f"  Wilcoxon (gen vs real):   U={u:.0f}, p={pval:.4e}",
              flush=True)
        u2, pval2 = stats.mannwhitneyu(
            sub["gen_off_target_count"], sub["rand_off_target_count"],
            alternative="two-sided"
        )
        print(f"  Wilcoxon (gen vs random): U={u2:.0f}, p={pval2:.4e}",
              flush=True)

    # ── 5. Save & plot ─────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "exp2_off_target_analysis.csv")
    all_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved → {csv_path}", flush=True)

    plot_off_target_histogram(
        all_df,
        os.path.join(PLOTS_DIR, "exp2_off_target_histogram.pdf")
    )
    plot_specificity_comparison(
        all_df,
        os.path.join(PLOTS_DIR, "exp2_specificity_comparison.pdf")
    )

    print("\n" + "=" * 65, flush=True)
    print("DONE – Experiment 2", flush=True)
    print(f"  {csv_path}", flush=True)
    print(f"  plots/exp2_off_target_histogram.pdf", flush=True)
    print(f"  plots/exp2_specificity_comparison.pdf", flush=True)


if __name__ == "__main__":
    main()
