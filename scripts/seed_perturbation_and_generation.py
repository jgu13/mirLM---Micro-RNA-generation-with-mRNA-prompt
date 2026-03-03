"""
seed_perturbation_and_generation.py

For every sample in the validation set, perturb ALL bases inside the
seed match region simultaneously (positions seed_start … seed_end, inclusive),
replacing each base with a randomly-chosen alternative nucleotide in one go.
The output has the same number of rows as the input validation set.

Generated miRNAs are produced by the trained TargetGenerationModel from the
perturbed mRNAs and saved in the same format as the reference generation CSV.

Output CSV columns:
    Transcript ID, miRNA ID, miRNA sequence, mRNA sequence,
    seed start, seed end, label, perturbed_mRNA, generated_mirna

Run from the scripts/ directory:
    python seed_perturbation_and_generation.py
"""

import os
import sys
import random
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Allow imports from scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Global_parameters import PROJ_HOME
import DTEA_model as dm

# ─── Configuration ────────────────────────────────────────────────────────────

DEVICE        = "cuda:1"
BATCH_SIZE    = 64
MRNA_MAX_LEN  = 120
MIRNA_MAX_LEN = 24 + 2    # BOS + up to 24 nucleotides + EOS  →  26 tokens
EMBED_DIM     = 1024
FF_DIM        = 4096
NUM_HEADS     = 8
NUM_LAYERS    = 4
VOCAB_SIZE    = 13
N_CLASSES     = 13
USE_LONGFORMER = True

CKPT_PATH = os.path.join(
    PROJ_HOME,
    "checkpoints/TargetScan/TwoTowerTransformer/CNN-tokenized/TargetGeneration/",
    "520/full_cross_attn/best_token_accuracy_0.9376_epoch18.pth",
)

INPUT_CSV = os.path.join(
    PROJ_HOME,
    "TargetScan_dataset/Positive_primates_test_500_randomized_start.csv",
)

OUTPUT_CSV = os.path.join(
    PROJ_HOME,
    "TargetScan_dataset"
    "/generated_mirna_seed_perturbation_520_longformer_random_samples_test.csv",
)

BASES = ["A", "T", "C", "G"]

# ─── Dataset ──────────────────────────────────────────────────────────────────


class PerturbedMRNADataset(Dataset):
    """
    Tokenises a flat list of (already-perturbed) mRNA strings exactly as
    TargetPredictionDataset does for mRNA:
        char tokens  +  [EOS]  +  right-padding to MRNA_MAX_LEN
    """

    def __init__(self, mrna_seqs, tokenizer, mrna_max_len):
        self.mrna_seqs    = mrna_seqs
        self.tokenizer    = tokenizer
        self.mrna_max_len = mrna_max_len
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id

    def __len__(self):
        return len(self.mrna_seqs)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.mrna_seqs[idx],
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.mrna_max_len - 1,   # leave one slot for EOS
        )
        ids = encoded["input_ids"] + [self.eos]
        if len(ids) < self.mrna_max_len:
            ids = ids + [self.pad] * (self.mrna_max_len - len(ids))
        else:
            ids = ids[:self.mrna_max_len]
        return torch.tensor(ids, dtype=torch.long)


# ─── Perturbation ─────────────────────────────────────────────────────────────


def perturb_seed_region(mrna, seed_s, seed_e):
    """
    Replace every base in [seed_s, seed_e] (inclusive) simultaneously with a
    randomly-chosen alternative nucleotide.  Returns the perturbed string.
    """
    bases = list(mrna)
    for pos in range(seed_s, seed_e + 1):
        orig = bases[pos]
        alts = [b for b in BASES if b != orig]
        bases[pos] = random.choice(alts)
    return "".join(bases)


def build_perturbed_rows(df):
    """
    One perturbed-mRNA record per sample: all seed-region bases are mutated
    simultaneously.  Output row count equals input row count.

    Returns a list of dicts (later converted to a DataFrame).
    """
    rows = []
    for _, row in df.iterrows():
        mrna   = row["mRNA sequence"]
        seed_s = int(row["seed start"])
        seed_e = int(row["seed end"])
        perturbed_mrna = perturb_seed_region(mrna, seed_s, seed_e)
        rows.append({
            "Transcript ID":  row["Transcript ID"],
            "miRNA ID":       row["miRNA ID"],
            "miRNA sequence": row["miRNA sequence"],
            "mRNA sequence":  mrna,
            "seed start":     seed_s,
            "seed end":       seed_e,
            "label":          row["label"],
            "perturbed_mRNA": perturbed_mrna,
        })
    return rows


# ─── Model loading ────────────────────────────────────────────────────────────


def load_generation_model(ckpt_path, device):
    """
    Build TargetGenerationModel with the same hyper-parameters used during
    training and load the checkpoint (mirrors the predict branch of
    TargetGenerationModel.run).
    """
    model = dm.TargetGenerationModel(
        mrna_max_len   = MRNA_MAX_LEN,
        mirna_max_len  = MIRNA_MAX_LEN,
        device         = device,
        embed_dim      = EMBED_DIM,
        ff_dim         = FF_DIM,
        num_heads      = NUM_HEADS,
        num_layers     = NUM_LAYERS,
        batch_size     = BATCH_SIZE,
        vocab_size     = VOCAB_SIZE,
        n_classes      = N_CLASSES,
        lr             = 3e-5,
        seed           = 10020,
        use_longformer = USE_LONGFORMER,
    )

    # pad/bos/eos indices are already set correctly by TargetGenerationModel.__init__
    # via self.tokenizer.{pad,bos,eos}_token_id — no need to override them.

    # Load checkpoint weights (same logic as TargetGenerationModel.run predict)
    loaded  = torch.load(ckpt_path, map_location=device)
    current = model.state_dict()
    to_load = {}
    for k, v in loaded.items():
        if k not in current:
            continue
        if ("rotary.cos_emb" in k) or ("rotary.sin_emb" in k):
            if v.shape == current[k].shape:
                to_load[k] = v
            else:
                print(f"  Dropping mismatched rotary buffer {k}: "
                      f"ckpt {v.shape} vs model {current[k].shape}")
        else:
            to_load[k] = v
    model.load_state_dict(to_load, strict=False)
    print(f"Loaded checkpoint from {ckpt_path}")

    model.to(device)
    model.eval()
    return model


# ─── Generation ───────────────────────────────────────────────────────────────


def generate_for_mrnas(model, mrna_seqs, device, batch_size):
    """
    Run batched greedy decoding for a list of mRNA strings.
    Returns a list of decoded miRNA strings (DNA notation, 3'→5').
    """
    dataset    = PerturbedMRNADataset(mrna_seqs, model.tokenizer, MRNA_MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    generated_seqs = []
    n_done = 0
    for batch_tokens in dataloader:
        batch_tokens = batch_tokens.to(device)
        generated = model.greedy_generate(
            model       = model,
            mrna_tokens = batch_tokens,
            max_len     = MIRNA_MAX_LEN,
        )
        for i in range(generated.size(0)):
            seq_ids = generated[i].tolist()
            decoded = model.tokenizer.decode(seq_ids, skip_special_tokens=True)
            generated_seqs.append(decoded)
        n_done += batch_tokens.size(0)
        print(f"  Generated {n_done}/{len(dataset)} sequences", end="\r", flush=True)

    print()   # newline after progress indicator
    return generated_seqs


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    print(f"Input  : {INPUT_CSV}")
    print(f"Output : {OUTPUT_CSV}")
    print(f"Device : {DEVICE}\n")

    # 1. Load validation set
    print(f"Loading dataset …")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df)} samples.\n")

    # 2. Perturb entire seed region of each sample in one go
    print("Perturbing seed match regions …")
    rows         = build_perturbed_rows(df)
    perturbed_df = pd.DataFrame(rows)
    print(f"  {len(perturbed_df)} perturbed sequences (one per sample).\n")

    # 3. Load model
    print("Loading TargetGenerationModel …")
    model = load_generation_model(CKPT_PATH, DEVICE)
    print()

    # 4. Generate miRNAs for every perturbed mRNA
    print("Generating miRNAs …")
    mrna_list      = perturbed_df["perturbed_mRNA"].tolist()
    generated_seqs = generate_for_mrnas(model, mrna_list, DEVICE, BATCH_SIZE)
    perturbed_df["generated_mirna"] = generated_seqs
    print(f"  Done — {len(generated_seqs)} sequences generated.\n")

    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    perturbed_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
