#!/usr/bin/env python3
"""
Experiment 3: Functional Potency Prediction
============================================

Part A – Simplified Context++ Score (all 3 lengths: 30nt, 100nt, 500nt)
  For each generated miRNA-mRNA pair compute:
    1. Seed match type (8-mer, 7mer-m8, 7mer-A1, 6-mer, none)
       using the canonical TargetScan seed-match logic.
    2. Local AU content – fraction of A/U in the 30-nt window flanking
       the seed match site in the mRNA.
    3. 3' supplementary pairing – Watson-Crick pair count at miRNA
       positions 13-16 (0-indexed 12:16) versus the aligned mRNA segment.
    4. Simplified Context++ score (weighted sum calibrated to the Agarwal
       et al. 2015 eLife paper weights):
         score = w_site[seed_type] + w_au * local_au + w_3p * pairs_3p

Part B – Cross-Attention Correlation (100nt dataset)
  For the 100nt test set:
    5. Load the Longformer TargetGenerationModel (120-token mRNA window).
    6. Run batched inference; capture per-token cross-attention weights
       (decoder cross-attention of each generated miRNA token to every
       mRNA position) from the last decoder layer.
    7. Average the attention over the seed-generation steps (miRNA tokens
       2-8) → scalar "seed attention concentration" per pair.
    8. Correlation scatter: seed attention concentration vs Context++ score.

Outputs
-------
  results/exp3_potency_prediction.json   (full per-sample predictions)
  results/exp3_potency_summary.csv       (per-sample summary table)
  plots/exp3_seed_type_potency.pdf       (box-plots: potency by seed type)
  plots/exp3_attention_vs_potency.pdf    (scatter: attention vs potency)
  plots/exp3_lfc_distribution.pdf        (predicted ΔlFC distribution)

Requirements
------------
  Python packages: torch, pandas, numpy, scipy, matplotlib
  Model checkpoint (for Part B):
    checkpoints/TargetScan/TwoTowerTransformer/Longformer/
      TargetGeneration/120/full_cross_attn/best_token_accuracy_0.9552_epoch17.pth
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import torch

# Allow imports from the scripts/ directory
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from Global_parameters import PROJ_HOME

# ── Paths ──────────────────────────────────────────────────────────────────────
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

CKPT_PATH = os.path.join(
    PROJ_HOME,
    "checkpoints/TargetScan/TwoTowerTransformer/Longformer/",
    "TargetGeneration/120/full_cross_attn/best_token_accuracy_0.9552_epoch17.pth",
)

RESULTS_DIR = os.path.join(PROJ_HOME, "results")
PLOTS_DIR   = os.path.join(PROJ_HOME, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MRNA_MAX_LEN  = 120
MIRNA_MAX_LEN = 26   # BOS + 24 nt + EOS

# ── Simplified Context++ weights (Agarwal et al. 2015, Table S7) ──────────────
SITE_TYPE_SCORE = {
    "8-mer":   -0.310,
    "7-mer-m8": -0.161,
    "7-mer-A1": -0.099,
    "6-mer":   -0.015,
    None:       0.000,
}
W_LOCAL_AU = -0.380   # coefficient for local AU content
W_3PRIME   = -0.095   # coefficient per 3' supplementary pair (pos 13-16)


# ══════════════════════════════════════════════════════════════════════════════
# Part A helpers – sequence-based features
# ══════════════════════════════════════════════════════════════════════════════

def dna_complement(seq: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A",
            "U": "A", "N": "N"}
    return "".join(comp.get(b, "N") for b in seq.upper())


def to_dna(seq: str) -> str:
    return seq.upper().replace("U", "T")


def to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def canonical_seed_match(mrna: str, mirna: str) -> tuple[str | None, int]:
    """
    Canonical TargetScan seed-match search at miRNA positions 2-8.
    mirna is stored 3'→5' in the CSV; convert to 5'→3' first.

    Returns (match_type, mRNA_position) where mRNA_position is 0-based
    start of the match in the mRNA.
    """
    mrna  = to_dna(mrna)
    mirna = to_dna(mirna)
    mirna_fwd = mirna[::-1]   # 5'→3' of miRNA

    seed = mirna_fwd[1:8]                        # positions 2-8 (0-indexed 1:8)
    seed_rc = dna_complement(seed)[::-1]          # RC: complement then reverse

    # 8-mer: seed 2-8 RC + A anchor at mRNA position 1
    target_8mer = seed_rc + "A"
    if target_8mer in mrna:
        return "8-mer", mrna.find(target_8mer)

    # 7-mer-m8: seed 2-8 RC
    if seed_rc in mrna:
        return "7-mer-m8", mrna.find(seed_rc)

    # 7-mer-A1: seed 2-7 RC + A anchor
    seed6 = mirna_fwd[1:7]
    seed6_rc = dna_complement(seed6)[::-1]
    target_7a1 = seed6_rc + "A"
    if target_7a1 in mrna:
        return "7-mer-A1", mrna.find(target_7a1)

    # 6-mer: seed 2-7 RC
    if seed6_rc in mrna:
        return "6-mer", mrna.find(seed6_rc)

    return None, -1


def local_au_content(mrna: str, seed_pos: int, flank: int = 30) -> float:
    """AU content in a ±flank-nt window around the seed match site."""
    lo  = max(0, seed_pos - flank)
    hi  = min(len(mrna), seed_pos + flank)
    window = mrna[lo:hi].upper().replace("T", "U")
    if not window:
        return 0.5
    au = sum(1 for c in window if c in "AU")
    return au / len(window)


def three_prime_pairing(mirna: str, mrna: str, seed_pos: int) -> int:
    """
    Count Watson-Crick pairs at miRNA 3' supplementary region (positions 13-16,
    0-indexed 12:16 in 5'→3' direction).
    Aligns mirna positions 12-15 against the mRNA just upstream of the seed.
    Returns count (0–4).
    """
    mirna_fwd = to_dna(mirna[::-1])  # convert to 5'→3' DNA
    mirna_fwd = mirna_fwd.upper()
    mrna_dna  = to_dna(mrna).upper()

    supp_region = mirna_fwd[12:16]   # 4 nts
    if len(supp_region) < 4 or seed_pos < 4:
        return 0

    # mRNA target of 3' supplementary: immediately 5' of seed match
    # In the anti-parallel sense, this is mrna_pos - 4 ... mrna_pos-1 reversed
    mrna_target_start = max(0, seed_pos - 4)
    mrna_target = mrna_dna[mrna_target_start:seed_pos][::-1]  # reverse to align 5'→3'
    if len(mrna_target) < len(supp_region):
        return 0

    pairs = 0
    wc = {"A": "T", "T": "A", "C": "G", "G": "C"}
    for m, r in zip(supp_region, mrna_target):
        if wc.get(m) == r:
            pairs += 1
    return pairs


def context_plus_plus_score(seed_type: str | None, au: float, pairs: int) -> float:
    """Simplified Context++ predicted log fold change (more negative = stronger)."""
    return SITE_TYPE_SCORE[seed_type] + W_LOCAL_AU * au + W_3PRIME * pairs


def compute_features_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all Part A features for a dataset DataFrame."""
    rows = []
    for _, row in df.iterrows():
        mrna      = str(row["mRNA sequence"])
        gen_mirna = str(row["generated_mirna"])
        real_mirna= str(row["miRNA sequence"])

        # Seed match features for generated miRNA
        gen_seed_type, gen_seed_pos = canonical_seed_match(mrna, gen_mirna)
        gen_au    = local_au_content(mrna, gen_seed_pos) if gen_seed_pos >= 0 else 0.5
        gen_pairs = three_prime_pairing(gen_mirna, mrna, gen_seed_pos) if gen_seed_pos >= 0 else 0
        gen_score = context_plus_plus_score(gen_seed_type, gen_au, gen_pairs)

        # Seed match features for real (ground-truth) miRNA
        real_seed_type, real_seed_pos = canonical_seed_match(mrna, real_mirna)
        real_au    = local_au_content(mrna, real_seed_pos) if real_seed_pos >= 0 else 0.5
        real_pairs = three_prime_pairing(real_mirna, mrna, real_seed_pos) if real_seed_pos >= 0 else 0
        real_score = context_plus_plus_score(real_seed_type, real_au, real_pairs)

        rows.append({
            "Transcript_ID":       row["Transcript ID"],
            "miRNA_ID":            row["miRNA ID"],
            "gen_seed_type":       gen_seed_type if gen_seed_type else "None",
            "gen_seed_pos":        gen_seed_pos,
            "gen_local_au":        gen_au,
            "gen_3prime_pairs":    gen_pairs,
            "gen_context_score":   gen_score,
            "real_seed_type":      real_seed_type if real_seed_type else "None",
            "real_seed_pos":       real_seed_pos,
            "real_local_au":       real_au,
            "real_3prime_pairs":   real_pairs,
            "real_context_score":  real_score,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Part B helpers – cross-attention extraction
# ══════════════════════════════════════════════════════════════════════════════

def load_generation_model(ckpt_path: str, device: str):
    """Load TargetGenerationModel from checkpoint.

    The checkpoint was trained with mirna_max_len=MRNA_MAX_LEN (120) for the
    decoder rotary embedding buffers, so we must match that size here.
    Actual generation is limited to MIRNA_MAX_LEN tokens by the decoding loop.
    """
    import DTEA_model as dm
    model = dm.TargetGenerationModel(
        mirna_max_len  = MRNA_MAX_LEN,   # must match ckpt rotary buffer shape
        mrna_max_len   = MRNA_MAX_LEN,
        embed_dim      = 1024,
        ff_dim         = 4096,
        num_heads      = 8,
        num_layers     = 4,
        batch_size     = BATCH_SIZE,
        vocab_size     = 13,
        n_classes      = 13,
        device         = device,
        use_longformer = True,
        window_size    = 20,
    )
    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    state  = loaded.get("model_state_dict", loaded)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def extract_cross_attention_during_generation(
    model,
    mrna_seqs: list[str],
    batch_size: int = 32,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    Run greedy decoding for each mRNA and capture the cross-attention weights
    from the LAST decoder layer at each decoding step.

    For every sample we compute the average attention weight over
    miRNA generation steps that correspond to seed positions 2-8
    (decoder steps 2 through 8, 0-indexed).

    Returns
    -------
    attn_at_seed : np.ndarray, shape (N, mRNA_len)
        Mean attention weight (over seed-generation steps and heads)
        for each mRNA position.
    """
    from Data_pipeline import CharacterTokenizer

    tokenizer = model.tokenizer
    PAD = tokenizer.pad_token_id
    EOS = tokenizer.eos_token_id
    BOS = tokenizer.bos_token_id

    n = len(mrna_seqs)
    all_seed_attn = []

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = mrna_seqs[start:end]
        bsz   = len(batch)

        # Tokenise mRNAs
        mrna_ids_list = []
        for seq in batch:
            enc = tokenizer(
                seq.upper(),
                add_special_tokens=False,
                max_length=MRNA_MAX_LEN - 1,   # leave 1 slot for EOS
                truncation=True,
                padding=False,
            )
            ids = enc["input_ids"] + [EOS]
            if len(ids) < MRNA_MAX_LEN:
                ids = ids + [PAD] * (MRNA_MAX_LEN - len(ids))
            else:
                ids = ids[:MRNA_MAX_LEN]
            mrna_ids_list.append(ids)

        mrna_tokens = torch.tensor(mrna_ids_list, dtype=torch.long, device=device)
        mrna_mask   = (mrna_tokens != PAD).long()

        # Encode mRNA once
        with torch.no_grad():
            mrna_emb = model.sn_embedding(mrna_tokens)
            mrna_cnn = model.cnn_embedding(mrna_emb.transpose(-1, -2))
            mrna_emb = mrna_emb + mrna_cnn

            # Build Longformer mask
            lf_mask = torch.where(
                mrna_mask > 0,
                torch.zeros_like(mrna_mask),
                torch.full_like(mrna_mask, -1),
            )
            memory = model.mrna_encoder(mrna_emb, mask=lf_mask)  # (B, mRNA_len, D)

        # Per-sample seed-step attention accumulator: (bsz, num_heads, mRNA_len)
        # We'll collect attention at decoder steps 2..8 (seed positions)
        SEED_STEPS = list(range(1, 8))   # 0-indexed token positions being generated

        sample_attn = np.zeros((bsz, MRNA_MAX_LEN), dtype=np.float32)
        step_counts = np.zeros(bsz, dtype=np.int32)

        # Greedy decoding step by step
        tgt_ids = torch.full((bsz, 1), BOS, dtype=torch.long, device=device)

        for step in range(MIRNA_MAX_LEN - 1):
            with torch.no_grad():
                tgt_emb = model.sn_embedding(tgt_ids)
                tgt_cnn = model.cnn_embedding(tgt_emb.transpose(-1, -2))
                tgt_emb = tgt_emb + tgt_cnn

                # Causal mask for decoder self-attention
                t = tgt_ids.size(1)
                tgt_mask = torch.tril(torch.ones(t, t, device=device)).unsqueeze(0)

                # Run decoder layers manually to capture cross-attention
                x = tgt_emb
                for layer in model.mirna_decoder.layers:
                    # Self-attention
                    sa_out = layer.self_attn(x, x, x, mask=tgt_mask)
                    x = layer.norm1(x + layer.dropout(sa_out))
                    # Cross-attention (query=miRNA, key=value=mRNA)
                    ca_out = layer.cross_attn(x, memory, memory, mask=mrna_mask)
                    x = layer.norm2(x + layer.dropout(ca_out))
                    x = layer.norm3(x + layer.dropout(layer.feed_forward(x)))

                # Extract last_attention from the last decoder layer
                # Shape: (B, H, current_t, mRNA_len)
                last_cross_attn = model.mirna_decoder.layers[-1].cross_attn.last_attention
                # → (B, H, mRNA_len) at last query position
                attn_last_token = last_cross_attn[:, :, -1, :].cpu().numpy()  # (B, H, mRNA_len)

                # Accumulate if this step is in SEED_STEPS
                if step in SEED_STEPS:
                    # Mean over heads: (B, mRNA_len)
                    attn_mean = attn_last_token.mean(axis=1)
                    sample_attn += attn_mean
                    step_counts += 1

                # Next token prediction
                out = model.mirna_decoder.norm(x)
                out = model.mirna_decoder.dropout(out)
                logits  = model.predictor_head(out[:, -1, :])  # (B, vocab)
                next_tok = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
                tgt_ids  = torch.cat([tgt_ids, next_tok], dim=1)

                # Early stop if all sequences have generated EOS
                if (next_tok.squeeze(-1) == EOS).all():
                    break

        # Normalise by step count
        for i in range(bsz):
            if step_counts[i] > 0:
                sample_attn[i] /= step_counts[i]

        all_seed_attn.append(sample_attn)
        if start % 500 == 0:
            print(f"  Cross-attn extraction: {end}/{n}")

    return np.concatenate(all_seed_attn, axis=0)   # (N, mRNA_len)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_seed_type_potency(all_df: pd.DataFrame, out_pdf: str) -> None:
    """Box-plots of Context++ score per seed type, per mRNA length."""
    lengths = ["30nt", "100nt", "500nt"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    seed_types = ["8-mer", "7-mer-m8", "7-mer-A1", "6-mer", "None"]
    colours    = {"8-mer": "#1a6faf", "7-mer-m8": "#57a0d2",
                  "7-mer-A1": "#f4a460", "6-mer": "#e07b39", "None": "#bbb"}

    for ax, name in zip(axes, lengths):
        sub = all_df[all_df["mRNA_length"] == name]
        data   = [sub.loc[sub["gen_seed_type"] == st, "gen_context_score"].dropna().values
                  for st in seed_types]
        labels = [f"{st}\n(n={len(d)})" for st, d in zip(seed_types, data)]
        bp = ax.boxplot([d for d in data if len(d) > 0],
                        patch_artist=True, notch=False, vert=True)
        valid_labels = [l for d, l in zip(data, labels) if len(d) > 0]
        ax.set_xticklabels(valid_labels, rotation=20, ha="right", fontsize=9)
        for patch, st in zip(bp["boxes"],
                              [s for s, d in zip(seed_types, data) if len(d) > 0]):
            patch.set_facecolor(colours.get(st, "#ccc"))
        ax.set_title(f"{name} mRNA", fontsize=12)
        ax.set_xlabel("Seed Type", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Simplified Context++ Score", fontsize=11)

    plt.suptitle("Experiment 3 – Predicted Potency by Seed Type",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_pdf}")


def plot_attention_vs_potency(
    attn_seed_concentration: np.ndarray,
    potency_scores: np.ndarray,
    out_pdf: str,
) -> None:
    """Scatter + marginal histograms: mean seed-region cross-attention vs Context++ score.

    Cross-attention distribution (x-axis marginal) → orange (#e07b39).
    Context++ score distribution (y-axis marginal) → blue (#1a6faf).
    """
    CLR_ATTN    = "#e07b39"   # orange  — cross-attention
    CLR_SCORE   = "#1a6faf"   # blue    — Context++ score

    mask = np.isfinite(attn_seed_concentration) & np.isfinite(potency_scores)
    x = attn_seed_concentration[mask]
    y = potency_scores[mask]

    r, pval = stats.pearsonr(x, y)
    rho, pval_sp = stats.spearmanr(x, y)

    # Joint-distribution layout: main scatter + top/right marginal histograms
    fig = plt.figure(figsize=(8, 7))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=(5, 1.2), height_ratios=(1.2, 5),
        hspace=0.05, wspace=0.05,
    )
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)   # cross-attention hist
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)   # Context++ score hist

    # Main scatter
    ax_main.scatter(x, y, alpha=0.20, s=8, color="#888888", edgecolors="none")
    m, b = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 200)
    ax_main.plot(xr, m * xr + b, color="black", linewidth=1.8)
    ax_main.set_xlabel("Mean Cross-Attention at Seed Steps (pos 2–8)", fontsize=11)
    ax_main.set_ylabel("Simplified Context++ Score (log FC)",          fontsize=11)

    # Top marginal: cross-attention distribution (orange)
    ax_top.hist(x, bins=50, color=CLR_ATTN, alpha=0.85, density=True)
    ax_top.axvline(np.mean(x), color=CLR_ATTN, linestyle="--", linewidth=1.5,
                   label=f"mean={np.mean(x):.4f}")
    ax_top.set_ylabel("Density", fontsize=9)
    ax_top.legend(fontsize=8, loc="upper right")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Right marginal: Context++ score distribution (blue)
    ax_right.hist(y, bins=50, color=CLR_SCORE, alpha=0.85, density=True,
                  orientation="horizontal")
    ax_right.axhline(np.mean(y), color=CLR_SCORE, linestyle="--", linewidth=1.5,
                     label=f"mean={np.mean(y):.3f}")
    ax_right.set_xlabel("Density", fontsize=9)
    ax_right.legend(fontsize=8, loc="upper right")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.suptitle(
        f"Cross-Attention vs Predicted Potency (100nt)\n"
        f"Pearson r={r:.3f} (p={pval:.2e}),  Spearman ρ={rho:.3f} (p={pval_sp:.2e})",
        fontsize=11, y=1.01,
    )
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_pdf}")


def plot_lfc_distribution(all_df: pd.DataFrame, out_pdf: str) -> None:
    """Histogram of predicted log-fold change for generated vs real miRNA."""
    lengths = ["30nt", "100nt", "500nt"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, lengths):
        sub = all_df[all_df["mRNA_length"] == name]
        ax.hist(sub["gen_context_score"].dropna(),  bins=40, alpha=0.6,
                color="royalblue", label="Generated miRNA", density=True)
        ax.hist(sub["real_context_score"].dropna(), bins=40, alpha=0.6,
                color="forestgreen", label="Real miRNA (GT)", density=True)
        ax.axvline(sub["gen_context_score"].median(),  color="royalblue",
                   linestyle="--", linewidth=1.5)
        ax.axvline(sub["real_context_score"].median(), color="forestgreen",
                   linestyle="--", linewidth=1.5)
        u, pval = stats.mannwhitneyu(
            sub["gen_context_score"].dropna(),
            sub["real_context_score"].dropna(),
            alternative="two-sided",
        )
        ax.set_title(f"{name}\np={pval:.2e}", fontsize=12)
        ax.set_xlabel("Context++ Score", fontsize=11)
        ax.set_ylabel("Density",         fontsize=11)
        ax.legend(fontsize=9)
    plt.suptitle("Experiment 3 – Predicted Log Fold Change Distribution",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_pdf}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Part A: Context++ features for all 3 lengths ──────────────────────
    csv_path = os.path.join(RESULTS_DIR, "exp3_potency_summary.csv")

    if os.path.exists(csv_path):
        print(f"Loading cached Part A results from {csv_path}")
        all_df = pd.read_csv(csv_path)
    else:
        print("=" * 65)
        print("PART A – Simplified Context++ Score (all lengths)")
        print("=" * 65)

        all_feat_dfs = []
        for name, filepath in DATA_FILES.items():
            print(f"\nProcessing {name}: {filepath}")
            df  = pd.read_csv(filepath)
            print(f"  {len(df)} samples")
            feat = compute_features_for_df(df)
            feat.insert(0, "mRNA_length", name)
            all_feat_dfs.append(feat)
            print(f"  Done. Seed-match rate: "
                  f"{(feat['gen_seed_type'] != 'None').mean()*100:.1f}%")

        all_df = pd.concat(all_feat_dfs, ignore_index=True)
        # Save Part A results immediately so they survive a Part B crash
        all_df.to_csv(csv_path, index=False)
        print(f"\nPart A CSV cached → {csv_path}")

    # ── Print Part A summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PART A SUMMARY")
    print("=" * 65)
    for name in DATA_FILES:
        sub = all_df[all_df["mRNA_length"] == name]
        print(f"\n{name}  (n={len(sub)})")
        print(f"  Seed-match rate (generated): "
              f"{(sub['gen_seed_type']!='None').mean()*100:.1f}%")
        print(f"  Mean Context++ score – generated: "
              f"{sub['gen_context_score'].mean():.4f}")
        print(f"  Mean Context++ score – real miRNA: "
              f"{sub['real_context_score'].mean():.4f}")
        print(f"  Seed type distribution (generated):")
        for st, cnt in sub["gen_seed_type"].value_counts().items():
            print(f"    {st:12s}: {cnt:5d}  ({cnt/len(sub)*100:.1f}%)")

    # ── Part B: Cross-attention extraction (100nt model) ──────────────────
    print("\n" + "=" * 65)
    print("PART B – Cross-Attention Extraction (100nt model)")
    print("=" * 65)

    df_100 = pd.read_csv(DATA_FILES["100nt"])
    mrna_seqs_100 = df_100["mRNA sequence"].tolist()
    feat_100      = all_df[all_df["mRNA_length"] == "100nt"].reset_index(drop=True)

    print(f"Loading generation model on {DEVICE} ...")
    model = load_generation_model(CKPT_PATH, DEVICE)

    print(f"Extracting cross-attention for {len(mrna_seqs_100)} mRNAs ...")
    seed_attn = extract_cross_attention_during_generation(
        model, mrna_seqs_100, batch_size=BATCH_SIZE, device=DEVICE
    )
    # seed_attn: (N, mRNA_len) – mean attention over seed generation steps and heads

    # Seed concentration = mean attention over mRNA positions covered by seed match
    seed_pos_arr = feat_100["gen_seed_pos"].values
    seed_concentration = np.zeros(len(feat_100), dtype=np.float32)
    for i, sp in enumerate(seed_pos_arr):
        if sp >= 0:
            lo = max(0, sp)
            hi = min(MRNA_MAX_LEN, sp + 7)
            if lo < hi and hi <= seed_attn.shape[1]:
                seed_concentration[i] = seed_attn[i, lo:hi].mean()

    feat_100 = feat_100.copy()
    feat_100["seed_attn_concentration"] = seed_concentration

    potency_arr = feat_100["gen_context_score"].values
    r, pval = stats.pearsonr(
        seed_concentration[seed_pos_arr >= 0],
        potency_arr[seed_pos_arr >= 0],
    )
    print(f"\nPearson r (seed attn vs potency) = {r:.4f}  p = {pval:.4e}")

    # ── Save outputs ───────────────────────────────────────────────────────
    # JSON with full predictions
    json_path = os.path.join(RESULTS_DIR, "exp3_potency_prediction.json")
    records   = all_df.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nJSON saved → {json_path}")

    # CSV summary
    csv_path = os.path.join(RESULTS_DIR, "exp3_potency_summary.csv")
    all_df.to_csv(csv_path, index=False)
    print(f"CSV saved  → {csv_path}")

    # Cross-attention column: store only for 100nt samples
    cross_attn_csv = os.path.join(RESULTS_DIR, "exp3_cross_attention_100nt.csv")
    feat_100.to_csv(cross_attn_csv, index=False)
    print(f"CSV (cross-attn 100nt) → {cross_attn_csv}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_seed_type_potency(
        all_df,
        os.path.join(PLOTS_DIR, "exp3_seed_type_potency.pdf")
    )
    plot_attention_vs_potency(
        seed_concentration,
        potency_arr,
        os.path.join(PLOTS_DIR, "exp3_attention_vs_potency.pdf")
    )
    plot_lfc_distribution(
        all_df,
        os.path.join(PLOTS_DIR, "exp3_lfc_distribution.pdf")
    )

    print("\n" + "=" * 65)
    print("DONE – Experiment 3")
    print(f"  results/exp3_potency_prediction.json")
    print(f"  results/exp3_potency_summary.csv")
    print(f"  results/exp3_cross_attention_100nt.csv")
    print(f"  plots/exp3_seed_type_potency.pdf")
    print(f"  plots/exp3_attention_vs_potency.pdf")
    print(f"  plots/exp3_lfc_distribution.pdf")


if __name__ == "__main__":
    main()