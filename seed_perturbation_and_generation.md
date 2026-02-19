# Seed Perturbation and miRNA Generation

## Overview

`scripts/seed_perturbation_and_generation.py` simultaneously mutates every
nucleotide inside the **seed match region** of each mRNA sequence in the
30-nt validation set (one random alternative per base, all at once) and
generates a candidate miRNA from each perturbed mRNA using the trained
`TargetGenerationModel`.

---

## What the Script Does

### 1. Load the Validation Dataset

```
TargetScan_dataset/positive_samples_30_random_samples_validation.csv
```

19 951 positive miRNA–mRNA pairs with 30-nt mRNA windows.  Each row
contains the columns:

| Column | Description |
|---|---|
| `Transcript ID` | Ensembl transcript identifier |
| `miRNA ID` | miRBase identifier |
| `miRNA sequence` | Full miRNA sequence (RNA, 5'→3') |
| `mRNA sequence` | 30-nt 3'-UTR window (DNA, T-based) |
| `seed start` | 0-based start of the seed match in the mRNA window |
| `seed end` | 0-based end of the seed match (inclusive) |
| `label` | 1 = positive pair |

### 2. Build the Perturbed mRNA Catalogue

For each sample, **every base in positions `[seed_start, seed_end]`
(inclusive) is replaced simultaneously** with a randomly-chosen alternative
nucleotide (any of A, T, C, G excluding the original):

```
seed region positions:  seed_start, seed_start+1, …, seed_end
each position:          1 random alternative base
records per sample:     1  (entire seed region mutated in one pass)
total output rows:      19 951  (one-to-one with the input)
```

The perturbed DataFrame carries all original columns plus:

| New column | Description |
|---|---|
| `perturbed_mRNA` | Full 30-nt mRNA with all seed-region bases simultaneously substituted |

### 3. Load the TargetGenerationModel Checkpoint

Model configuration (identical to the training run in `transformer_model.py`):

| Hyper-parameter | Value |
|---|---|
| `mrna_max_len` | 30 |
| `mirna_max_len` | 26 (= 24 bases + BOS + EOS) |
| `embed_dim` | 256 |
| `ff_dim` | 512 |
| `num_layers` | 2 |
| `num_heads` | 2 |
| `vocab_size` / `n_classes` | 13 |
| `dropout_rate` | 0.1 |

Checkpoint path:

```
checkpoints/TargetScan/TwoTowerTransformer/CNN-tokenized/
    TargetGeneration/30/best_token_accuracy_0.9554_epoch19.pth
```

This checkpoint achieves **95.54 % token accuracy** on the validation set
after 19 training epochs.

### 4. Generate miRNAs (Greedy Decoding)

Each perturbed mRNA is tokenised with `CharacterTokenizer` (character-level,
T-based DNA, right-padded to length 30 with an appended `[EOS]` token) and
passed through the model's encoder–decoder in batches of 64.

Greedy decoding procedure (`TargetGenerationModel.greedy_generate`):

1. Encode the perturbed mRNA once with the CNN-augmented `TransformerEncoder`.
2. Start the decoder with a single `[BOS]` token.
3. At each step take the `argmax` of the last-position logits and append the
   predicted token.
4. Stop when all sequences in the batch have emitted `[EOS]` or the
   maximum length (26) is reached.
5. Strip `[BOS]` from the output; `[EOS]` and `[PAD]` tokens are removed
   during decoding (`skip_special_tokens=True`).

The resulting sequences are in **DNA notation, 3'→5' direction** (matching
the reference file convention; RNA `U` was converted to `T` and the
sequence was reversed 3'→5' during tokenisation).

### 5. Save Output

```
TargetScan_dataset/generated_mirna_seed_perturbation_30_random_samples_validation.csv
```

---

## Output Format

The output CSV has the same columns as the input plus two additional columns:

```
Transcript ID, miRNA ID, miRNA sequence, mRNA sequence,
seed start, seed end, label,
perturbed_mRNA, generated_mirna
```

Example row (seed region positions 16–23 fully mutated in one pass):

| Transcript ID | miRNA ID | miRNA sequence | mRNA sequence | seed start | seed end | label | perturbed_mRNA | generated_mirna |
|---|---|---|---|---|---|---|---|---|
| ENST00000324344.4 | hsa-miR-182-5p | UUUGGCAAUGGUAGAACUCACACU | GCTCTTTCCCCCATCTTTGCCAAATCTCAA | 16 | 23 | 1 | GCTCTTTCCCCCATCTTTACGTACATCTCAA | … |
| … | … | … | … | … | … | … | … | … |

---

## How to Run

```bash
cd /home/mcb/users/jgu13/projects/MiRformer/scripts
python seed_perturbation_and_generation.py
```

Adjust `DEVICE`, `BATCH_SIZE`, and `OUTPUT_CSV` at the top of the script
as needed.  The script prints progress to stdout and writes one CSV file
when finished.

---

## Design Choices

| Choice | Rationale |
|---|---|
| Inclusive seed window `[seed_start, seed_end]` | Matches TargetScan annotation convention used throughout the dataset |
| Simultaneous random perturbation of all seed bases | Simulates a fully disrupted seed match in one step; avoids the combinatorial explosion of exhaustive single-base mutagenesis |
| One output row per input sample | Keeps output size identical to the validation set (19 951 rows), making downstream comparisons straightforward |
| Greedy decoding | Deterministic; consistent with the original generation pipeline in `transformer_model.py` |
| Batch size 64 | Balances GPU memory and throughput on the 30-nt model |
| Output appended to existing columns | Preserves full traceability — each generated miRNA can be linked back to the exact substitution that produced it |
