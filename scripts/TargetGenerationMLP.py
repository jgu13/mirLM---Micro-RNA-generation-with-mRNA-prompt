"""
TargetGenerationWithMLP  v4 — decoder output → binding_output
=============================================================

Simplest and most principled approach:
  - decoder_output is the representation the Generator was pretrained to optimize
  - It already contains mRNA-miRNA interaction (via cross-attention inside decoder)
  - No need to modify TransformerDecoder at all
  - No fusion layer needed

Architecture:
  mRNA  ──► [Frozen Encoder] ──► memory
                                    │
  miRNA ──► [Trainable Decoder] ────┤──► decoder_output (B, L_mirna, D)
                  │                                │
                  ▼                                ▼ mean-pool
          predictor_head                     pooled (B, D)
          (gen CE loss)                            │
                                                   ▼
                                     [Pretrained binding_output]
                                        1024 → 4096 → 4096 → 1
                                                   │
                                               P(positive)

Freeze: sn_embedding, cnn_embedding, mrna_encoder
Train:  mirna_decoder, predictor_head, binding_output
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)

from DTEA_model import TargetGenerationModel, LinearHead
from Global_parameters import PROJ_HOME


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_VOCAB = [
    "three_prime_utr", "intron", "exon", "five_prime_utr",
    "exon,intron", "intron,exon",
    "five_prime_utr,three_prime_utr", "three_prime_utr,exon",
    "exon,three_prime_utr", "exon,five_prime_utr",
    "five_prime_utr,exon", "three_prime_utr,five_prime_utr",
    "three_prime_utr,intron", "intron,three_prime_utr",
    "intron,exon,three_prime_utr", "five_prime_utr,exon,intron",
    "exon,intron,three_prime_utr", "intron,five_prime_utr",
    "five_prime_utr,intron", "exon,five_prime_utr,intron",
    "intron,exon,five_prime_utr", "UNK",
]
FEATURE_TO_IDX = {f: i for i, f in enumerate(FEATURE_VOCAB)}


class eCLIPClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, mrna_max_len, mirna_max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mrna_max_len = mrna_max_len
        self.mirna_max_len = mirna_max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mrna_enc = self.tokenizer(
            str(row["gene"]), padding="max_length", truncation=True,
            max_length=self.mrna_max_len, return_tensors="pt",
        )
        mirna_enc = self.tokenizer(
            str(row["noncodingRNA"]), padding="max_length", truncation=True,
            max_length=self.mirna_max_len, return_tensors="pt",
        )
        feat_str = str(row["feature"]) if pd.notna(row["feature"]) else "UNK"
        feat_idx = FEATURE_TO_IDX.get(feat_str, FEATURE_TO_IDX["UNK"])
        feature_onehot = torch.zeros(len(FEATURE_VOCAB), dtype=torch.float)
        feature_onehot[feat_idx] = 1.0

        return {
            "mrna_input_ids":       mrna_enc["input_ids"].squeeze(0),
            "mrna_attention_mask":  mrna_enc["attention_mask"].squeeze(0),
            "mirna_input_ids":      mirna_enc["input_ids"].squeeze(0),
            "mirna_attention_mask": mirna_enc["attention_mask"].squeeze(0),
            "feature_onehot":       feature_onehot,
            "label":                torch.tensor(int(row["label"]), dtype=torch.float),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Model
# ══════════════════════════════════════════════════════════════════════════════

class TargetGenerationWithMLP(nn.Module):
    """
    v4: decoder_output → mean-pool → pretrained binding_output.
    No modifications to TransformerDecoder needed.
    """

    def __init__(
        self,
        pretrained_gen: TargetGenerationModel,
        pretrained_binding_output: LinearHead,
    ):
        super().__init__()
        self.gen_model = pretrained_gen
        self.binding_output = pretrained_binding_output
        self._freeze_encoder()

    def _freeze_encoder(self):
        for name, param in self.gen_model.named_parameters():
            if name.startswith(("sn_embedding.", "cnn_embedding.", "mrna_encoder.")):
                param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def pad_idx(self):
        return self.gen_model.pad_idx

    # ── encode (frozen) ───────────────────────────────────────────────────

    def encode_mrna(self, mrna_ids, mrna_mask):
        m = self.gen_model
        mrna_sn = m.sn_embedding(mrna_ids)
        mrna_cnn = m.cnn_embedding(mrna_sn.transpose(-1, -2))
        mrna_emb = mrna_sn + mrna_cnn

        if m.use_longformer:
            lf = mrna_mask
            if lf.dim() == 3 and lf.shape[1] == 1:
                lf = lf.squeeze(1)
            lf_mask = torch.where(
                lf > 0,
                torch.zeros_like(lf, dtype=torch.long),
                torch.full_like(lf, -1, dtype=torch.long),
            )
            return m.mrna_encoder(mrna_emb, mask=lf_mask)
        return m.mrna_encoder(mrna_emb, mask=mrna_mask)

    # ── decode (trainable) ────────────────────────────────────────────────

    def decode_mirna(self, mirna_ids, memory, mrna_mask, mirna_mask_causal):
        m = self.gen_model
        mirna_emb = m.sn_embedding(mirna_ids)

        if m.use_longformer:
            lf = mrna_mask
            if lf.dim() == 3 and lf.shape[1] == 1:
                lf = lf.squeeze(1)
            src_key_mask = lf.to(torch.uint8)
        else:
            src_key_mask = mrna_mask

        # Standard decoder forward — no modifications needed
        decoder_output = m.mirna_decoder(
            x=mirna_emb,
            memory=memory,
            src_mask=src_key_mask,
            tgt_mask=mirna_mask_causal,
        )
        gen_logits = m.predictor_head(decoder_output)
        return decoder_output, gen_logits

    # ── classify ──────────────────────────────────────────────────────────

    def classify(self, decoder_output, mirna_mask_1d):
        """
        decoder_output : (B, L_mirna, D)  — final decoder representation
        mirna_mask_1d  : (B, L_mirna)     — 1=valid, 0=pad
        """
        mask = mirna_mask_1d.unsqueeze(-1).float()                          # (B, L, 1)
        pooled = (decoder_output * mask).sum(1) / mask.sum(1).clamp(min=1e-8)  # (B, D)
        return self.binding_output(pooled).squeeze(-1)                      # (B,)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, mrna_ids, mrna_mask, mirna_ids, mirna_mask_causal, mirna_mask_1d):
        memory = self.encode_mrna(mrna_ids, mrna_mask)
        decoder_output, gen_logits = self.decode_mirna(
            mirna_ids, memory, mrna_mask, mirna_mask_causal,
        )
        cls_logit = self.classify(decoder_output, mirna_mask_1d)
        return gen_logits, cls_logit


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Weight loading
# ══════════════════════════════════════════════════════════════════════════════

def load_binding_output_from_dtea_ckpt(
    ckpt_path, embed_dim=1024, ff_dim=4096, n_classes=1,
    dropout=0.2, device="cpu",
):
    binding_head = LinearHead(
        input_size=embed_dim,
        hidden_sizes=[ff_dim, ff_dim],
        output_size=n_classes,
        dropout=dropout,
    )
    sd = torch.load(ckpt_path, map_location=device)

    prefix = "predictor.binding_output."
    binding_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    if not binding_sd:
        raise ValueError(
            f"No keys with prefix '{prefix}' in checkpoint.\n"
            f"Top-level prefixes: {sorted(set(k.split('.')[0] for k in sd))}"
        )

    missing, unexpected = binding_head.load_state_dict(binding_sd, strict=False)
    print(f"[load_binding_output] loaded {len(binding_sd)} tensors")
    if missing:
        print(f"  missing: {missing}")
    if unexpected:
        print(f"  unexpected: {unexpected}")
    return binding_head


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Trainer
# ══════════════════════════════════════════════════════════════════════════════

class JointTrainer:
    def __init__(self, model, device="cuda", lr=3e-5,
                 alpha=0.3, beta=1.0, max_grad_norm=1.0, seed=42):
        self.model = model
        self.device = torch.device(device)
        self.alpha = alpha
        self.beta = beta
        self.max_grad_norm = max_grad_norm

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=lr)
        self.gen_loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self._seed_everything(seed)

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _create_src_mask(self, t):
        return (t != self.model.pad_idx).to(torch.uint8)

    def _create_tgt_mask_causal(self, t):
        B, L = t.size()
        causal = torch.tril(torch.ones(L, L, dtype=torch.uint8, device=t.device)).unsqueeze(0)
        non_pad = (t != self.model.pad_idx).to(torch.uint8).unsqueeze(1).expand(B, L, L)
        return causal & non_pad

    def _create_tgt_mask_1d(self, t):
        return (t != self.model.pad_idx).to(torch.uint8)

    def _step(self, batch):
        mrna_ids  = batch["mrna_input_ids"].to(self.device)
        mirna_ids = batch["mirna_input_ids"].to(self.device)
        labels    = batch["label"].to(self.device)

        tgt_input  = mirna_ids[:, :-1]
        tgt_output = mirna_ids[:, 1:]

        src_mask   = self._create_src_mask(mrna_ids).to(self.device)
        tgt_causal = self._create_tgt_mask_causal(tgt_input).to(self.device)
        tgt_1d     = self._create_tgt_mask_1d(tgt_input).to(self.device)

        gen_logits, cls_logit = self.model(
            mrna_ids=mrna_ids, mrna_mask=src_mask,
            mirna_ids=tgt_input,
            mirna_mask_causal=tgt_causal, mirna_mask_1d=tgt_1d,
        )

        B, L, V = gen_logits.size()
        gen_loss = self.gen_loss_fn(gen_logits.view(B * L, V), tgt_output.reshape(B * L))
        cls_loss = self.cls_loss_fn(cls_logit, labels)
        return gen_loss, cls_loss, cls_logit, labels

    def train_epoch(self, dataloader, epoch, accumulation_step=1):
        self.model.train()
        total_gen, total_cls, total_loss = 0., 0., 0.
        self.optimizer.zero_grad()
        loss_buf = []

        for i, batch in enumerate(dataloader):
            gen_loss, cls_loss, _, _ = self._step(batch)
            loss = self.alpha * gen_loss + self.beta * cls_loss
            (loss / accumulation_step).backward()

            loss_buf.append(loss.item() / accumulation_step)
            total_gen += gen_loss.item()
            total_cls += cls_loss.item()
            total_loss += loss.item()

            if (i + 1) % accumulation_step == 0:
                clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                bs = batch["mrna_input_ids"].size(0)
                print(
                    f"Epoch {epoch} [{(i+1)*bs}/{len(dataloader.dataset)} "
                    f"({(i+1)*bs/len(dataloader.dataset)*100:.0f}%)] "
                    f"loss={sum(loss_buf)/len(loss_buf):.4f} "
                    f"gen={gen_loss.item():.4f} cls={cls_loss.item():.4f}",
                    flush=True,
                )
                loss_buf = []

        if (i + 1) % accumulation_step != 0:
            clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        n = len(dataloader)
        return total_loss / n, total_gen / n, total_cls / n

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_gen, total_cls = 0., 0.
        all_labels, all_probs = [], []

        for batch in dataloader:
            gen_loss, cls_loss, cls_logit, labels = self._step(batch)
            total_gen += gen_loss.item()
            total_cls += cls_loss.item()
            all_probs.append(torch.sigmoid(cls_logit).cpu())
            all_labels.append(labels.cpu())

        y = torch.cat(all_labels).numpy()
        p = torch.cat(all_probs).numpy()
        yhat = (p >= 0.5).astype(int)
        n = len(dataloader)

        return {
            "gen_loss":  total_gen / n,
            "cls_loss":  total_cls / n,
            "accuracy":  accuracy_score(y, yhat),
            "f1":        f1_score(y, yhat),
            "precision": precision_score(y, yhat),
            "recall":    recall_score(y, yhat),
            "auroc":     roc_auc_score(y, p),
            "auprc":     average_precision_score(y, p),
            "mcc":       matthews_corrcoef(y, yhat),
        }

    def run(self, train_loader, val_loader, epochs=20,
            accumulation_step=4, patience=5, save_dir="checkpoints/joint_v4"):
        os.makedirs(save_dir, exist_ok=True)
        self.model.to(self.device)
        best_auroc, counter = 0., 0

        for epoch in range(epochs):
            tl, tg, tc = self.train_epoch(train_loader, epoch, accumulation_step)
            m = self.evaluate(val_loader)
            print(
                f"\n═══ Epoch {epoch} ═══\n"
                f"  Train: total={tl:.4f}  gen={tg:.4f}  cls={tc:.4f}\n"
                f"  Val:   gen={m['gen_loss']:.4f}  cls={m['cls_loss']:.4f}\n"
                f"  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
                f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}\n"
                f"  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
                f"MCC={m['mcc']:.4f}\n", flush=True,
            )
            if m["auroc"] > best_auroc:
                best_auroc = m["auroc"]
                counter = 0
                path = os.path.join(save_dir, f"best_auroc_{best_auroc:.4f}_epoch{epoch}.pth")
                torch.save(self.model.state_dict(), path)
                print(f"  ★ Saved → {path}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        return best_auroc


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    MRNA_MAX_LEN  = 120
    MIRNA_MAX_LEN = 24 + 2
    EMBED_DIM     = 1024
    NUM_HEADS     = 8
    NUM_LAYERS    = 4
    FF_DIM        = 4096
    VOCAB_SIZE    = 13
    N_CLASSES     = 13
    BATCH_SIZE    = 64
    LR            = 3e-5
    SEED          = 10020
    DEVICE        = "cuda:0"
    EPOCHS        = 20
    ACCUM_STEPS   = 4
    ALPHA         = 0.3      # generation loss weight
    BETA          = 1.0      # classification loss weight
    PATIENCE      = 5

    # ── paths (update these) ──────────────────────────────────────────────
    GEN_CKPT = os.path.join(
        PROJ_HOME,
        "checkpoints/TargetScan/TwoTowerTransformer/Longformer/"
        "TargetGeneration/120/full_cross_attn/"
        "best_token_accuracy_XXXX_epochXX.pth",
    )
    DTEA_CKPT = os.path.join(
        PROJ_HOME,
        "checkpoints/DTEA/"
        "best_binding_XXXX_epochXX.pth",
    )
    ECLIP_DATA = os.path.join(PROJ_HOME, "AGO2_eCLIP_Manakov2022_test.tsv.gz")

    # ── 1. Load pretrained Generator ──────────────────────────────────────
    gen_model = TargetGenerationModel(
        mrna_max_len=MRNA_MAX_LEN, mirna_max_len=MIRNA_MAX_LEN,
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, batch_size=BATCH_SIZE, vocab_size=VOCAB_SIZE,
        n_classes=N_CLASSES, lr=LR, seed=SEED, device=DEVICE,
        use_longformer=True,
    )
    gen_model.load_state_dict(torch.load(GEN_CKPT, map_location=DEVICE), strict=False)
    print(f"Loaded Generator from {GEN_CKPT}")

    # ── 2. Load pretrained binding_output ─────────────────────────────────
    binding_head = load_binding_output_from_dtea_ckpt(
        DTEA_CKPT, embed_dim=EMBED_DIM, ff_dim=FF_DIM,
        n_classes=1, dropout=0.2, device=DEVICE,
    )
    print(f"Loaded binding_output from {DTEA_CKPT}")

    # ── 3. Build joint model ──────────────────────────────────────────────
    model = TargetGenerationWithMLP(
        pretrained_gen=gen_model,
        pretrained_binding_output=binding_head,
    )
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_frozen:,} frozen,  {n_train:,} trainable")

    # ── 4. Data ───────────────────────────────────────────────────────────
    tokenizer = gen_model.tokenizer
    df = pd.read_csv(ECLIP_DATA, compression="gzip", sep="\t")
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=SEED,
    )
    ds_train = eCLIPClassificationDataset(df_train, tokenizer, MRNA_MAX_LEN, MIRNA_MAX_LEN)
    ds_val   = eCLIPClassificationDataset(df_val,   tokenizer, MRNA_MAX_LEN, MIRNA_MAX_LEN)
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train: {len(ds_train):,}  Val: {len(ds_val):,}")

    # ── 5. Train ──────────────────────────────────────────────────────────
    trainer = JointTrainer(
        model=model, device=DEVICE, lr=LR,
        alpha=ALPHA, beta=BETA, seed=SEED,
    )
    best = trainer.run(
        train_loader, val_loader, epochs=EPOCHS,
        accumulation_step=ACCUM_STEPS, patience=PATIENCE,
        save_dir=os.path.join(PROJ_HOME, "checkpoints", "joint_gen_cls_v4"),
    )
    print(f"\nDone. Best AUROC = {best:.4f}")