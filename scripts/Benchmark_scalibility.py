"""
Benchmark DTEA (MiRformer) inference scalability: sliding-window vs full attention.

Loads generated RNA sequences from benchmark_sequences.tsv, tokenizes them
with CharacterTokenizer (matching SpanDataset convention: add_special_tokens=False),
and profiles wall-clock time and peak GPU memory for both attention modes.

Usage:
    python benchmark_scalability.py --device cuda:0 --sequences benchmark_sequences.tsv
"""

import torch
import torch.nn as nn
import numpy as np
import csv
import argparse
import time

from DTEA_model import DTEA
from Data_pipeline import CharacterTokenizer

# ──────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────
MIRNA_MAX_LEN   = 30          # fixed miRNA padded length (adjust to match your training config)
MAX_MRNA_LEN    = 25000       # build model once with this ceiling
WARMUP_RUNS     = 2           # discarded warmup runs per length
BATCH_SIZE      = 1

# Model hyperparameters (match your trained model)
EMBED_DIM       = 1024
NUM_HEADS       = 8
NUM_LAYERS      = 4
FF_DIM          = 4096
WINDOW_SIZE     = 20          # must match CrossAttentionPredictor default


def build_model(device, use_longformer, mrna_max_len):
    """Build DTEA model. use_longformer controls attention type."""
    model = DTEA(
        mrna_max_len=mrna_max_len,
        mirna_max_len=MIRNA_MAX_LEN,
        device=device,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        predict_span=True,
        predict_binding=True,
        predict_cleavage=True,
        use_longformer=use_longformer,
    )
    model.to(device)
    model.eval()
    return model


def load_sequences(tsv_path):
    """
    Load sequences from benchmark_sequences.tsv.
    Returns dict: {mrna_length: [(mirna_seq, mrna_seq), ...]}
    Converts U -> T to match CharacterTokenizer vocabulary (same as SpanDataset).
    """
    sequences = {}
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mrna_len = int(row["mrna_length"])
            mirna_seq = row["mirna_seq"].replace("U", "T")
            mrna_seq  = row["mrna_seq"].replace("U", "T")
            if mrna_len not in sequences:
                sequences[mrna_len] = []
            sequences[mrna_len].append((mirna_seq, mrna_seq))
    return sequences


def tokenize_sequence(seq, tokenizer, max_len):
    """
    Tokenize a single RNA sequence following SpanDataset convention:
      - add_special_tokens=False (no CLS/SEP)
      - padding="max_length"
      - truncation=True
    Returns (input_ids, attention_mask) as tensors of shape (1, max_len).
    """
    encoded = tokenizer(
        seq,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_attention_mask=True,
    )
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
    attn_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long)
    return input_ids, attn_mask


def _pad_to_window(length):
    """Round up to nearest multiple of 2*WINDOW_SIZE (required by sliding_chunks)."""
    chunk = 2 * WINDOW_SIZE
    return ((length + chunk - 1) // chunk) * chunk


def benchmark_length(model, sequences, tokenizer, mrna_len, device):
    """
    Run forward passes for all sequences at a given mRNA length.
    First WARMUP_RUNS are discarded.
    Returns list of (time_seconds, peak_memory_bytes) for measured runs.
    """
    padded_mrna_len = _pad_to_window(mrna_len)
    results = []

    for i, (mirna_seq, mrna_seq) in enumerate(sequences):
        mirna_ids, mirna_mask = tokenize_sequence(mirna_seq, tokenizer, MIRNA_MAX_LEN)
        mrna_ids,  mrna_mask  = tokenize_sequence(mrna_seq,  tokenizer, padded_mrna_len)

        mirna_ids  = mirna_ids.to(device)
        mirna_mask = mirna_mask.to(device)
        mrna_ids   = mrna_ids.to(device)
        mrna_mask  = mrna_mask.to(device)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        t_start = time.perf_counter()

        with torch.no_grad():
            _ = model(
                mirna=mirna_ids,
                mrna=mrna_ids,
                mrna_mask=mrna_mask,
                mirna_mask=mirna_mask,
            )

        torch.cuda.synchronize(device)
        t_end = time.perf_counter()

        peak_mem = torch.cuda.max_memory_allocated(device)
        elapsed  = t_end - t_start

        if i >= WARMUP_RUNS:
            results.append((elapsed, peak_mem))

        del mirna_ids, mrna_ids, mirna_mask, mrna_mask
        torch.cuda.empty_cache()

    return results


def run_benchmark(model, sequences, tokenizer, mrna_lengths, device, label):
    """Run benchmarks across all lengths for one model. Returns list of result dicts."""
    all_results = []

    for mrna_len in mrna_lengths:
        print(f"  [{label}] mRNA length = {mrna_len:>6} nt ... ", end="", flush=True)

        try:
            results = benchmark_length(
                model, sequences[mrna_len], tokenizer, mrna_len, device
            )

            times = [r[0] for r in results]
            mems  = [r[1] for r in results]

            median_time = np.median(times)
            std_time    = np.std(times)
            median_mem  = np.median(mems) / (1024**2)
            std_mem     = np.std(mems) / (1024**2)

            all_results.append({
                "attention_mode":   label,
                "mrna_length":      mrna_len,
                "median_time_s":    round(median_time, 4),
                "std_time_s":       round(std_time, 4),
                "median_mem_MB":    round(median_mem, 1),
                "std_mem_MB":       round(std_mem, 1),
                "num_runs":         len(results),
            })

            print(f"time = {median_time:.4f}s (±{std_time:.4f}), "
                  f"mem = {median_mem:.1f} MB (±{std_mem:.1f})")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM! Skipping this and longer lengths.")
                torch.cuda.empty_cache()
                all_results.append({
                    "attention_mode":   label,
                    "mrna_length":      mrna_len,
                    "median_time_s":    "OOM",
                    "std_time_s":       "OOM",
                    "median_mem_MB":    "OOM",
                    "std_mem_MB":       "OOM",
                    "num_runs":         0,
                })
                # Skip remaining lengths — they will also OOM
                for remaining_len in mrna_lengths[mrna_lengths.index(mrna_len) + 1:]:
                    all_results.append({
                        "attention_mode":   label,
                        "mrna_length":      remaining_len,
                        "median_time_s":    "OOM",
                        "std_time_s":       "OOM",
                        "median_mem_MB":    "OOM",
                        "std_mem_MB":       "OOM",
                        "num_runs":         0,
                    })
                break
            else:
                raise e

    return all_results


def main():
    import os
    from Global_parameters import PROJ_HOME

    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    sequences_path = os.path.join(PROJ_HOME, "TargetScan_dataset", "benchmark_sequences.tsv")
    output_path = os.path.join(PROJ_HOME, "results", "benchmark_results.tsv")

    # ── Load sequences ──
    print(f"Loading sequences from {sequences_path}...")
    sequences = load_sequences(sequences_path)
    mrna_lengths = sorted(sequences.keys())
    print(f"Found {len(mrna_lengths)} lengths: {mrna_lengths}")
    print(f"Sequences per length: {len(sequences[mrna_lengths[0]])}")

    # ── Initialize tokenizer ──
    tokenizer = CharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],
        model_max_length=MAX_MRNA_LEN,
        padding_side="right",
    )
    print(f"Tokenizer vocab: {tokenizer.get_vocab()}")

    # ══════════════════════════════════════════════════
    # 1) Benchmark: Sliding-window attention (Longformer)
    # ══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Benchmarking: Sliding-window attention (use_longformer=True)")
    print(f"{'='*60}")
    model_lf = build_model(device, use_longformer=True, mrna_max_len=MAX_MRNA_LEN)
    total_params = sum(p.numel() for p in model_lf.parameters())
    print(f"Total parameters: {total_params:,}\n")

    results_lf = run_benchmark(
        model_lf, sequences, tokenizer, mrna_lengths, device,
        label="Sliding-window"
    )

    # Free GPU memory before building the next model
    del model_lf
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════
    # 2) Benchmark: Full attention
    # ══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Benchmarking: Full attention (use_longformer=False)")
    print(f"{'='*60}")
    model_full = build_model(device, use_longformer=False, mrna_max_len=MAX_MRNA_LEN)
    total_params = sum(p.numel() for p in model_full.parameters())
    print(f"Total parameters: {total_params:,}\n")

    results_full = run_benchmark(
        model_full, sequences, tokenizer, mrna_lengths, device,
        label="Full attention"
    )

    del model_full
    torch.cuda.empty_cache()

    # ── Save combined results ──
    all_results = results_lf + results_full

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {output_path}")

    # ── Print summary table ──
    print(f"\n{'Mode':<20} {'mRNA len':>10} {'Time (s)':>12} {'Memory (MB)':>14}")
    print("-" * 60)
    for r in all_results:
        t = r['median_time_s']
        m = r['median_mem_MB']
        t_str = f"{t:.4f}" if isinstance(t, float) else t
        m_str = f"{m:.1f}" if isinstance(m, (float, int)) else m
        print(f"{r['attention_mode']:<20} {r['mrna_length']:>10} {t_str:>12} {m_str:>14}")


if __name__ == "__main__":
    main()