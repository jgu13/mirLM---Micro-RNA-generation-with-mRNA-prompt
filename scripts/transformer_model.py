import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import math
import wandb
import random
import numpy as np
import pandas as pd
from time import time

from utils import load_dataset
from Data_pipeline import CharacterTokenizer, QuestionAnswerDataset, TargetPredictionDataset


class CNNTokenization(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        D = 2*embed_dim
        self.conv1 = nn.Conv1d(embed_dim, D, padding=2, kernel_size=5)
        self.conv2 = nn.Conv1d(embed_dim, D, padding=3, kernel_size=7)
        self.fc1 = nn.Linear(D, D)
        self.bn1 = nn.BatchNorm1d(D)
        self.fc2 = nn.Linear(D, embed_dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x1 = self.conv1(x) # (B, D, L)
        x2 = self.conv2(x)

        x1 = x1.transpose(-1, -2) # (B, L. D)
        x1 = self.act(self.fc1(x1)) # (B, L, D)
        x1 = x1.transpose(-1, -2) # (B, D, L)
        x1 = self.bn1(x1) # (B, D, L)
        x1 = x1.transpose(-1, -2) #(B, L, D)
        x1 = self.fc2(x1) # (B, L, embed_dim)

        x2 = x2.transpose(-1, -2) # (B, L. D)
        x2 = self.act(self.fc1(x2)) # (B, L, D)
        x2 = x2.transpose(-1, -2) # (B, D, L)
        x2 = self.bn1(x2) # (B, D, L)
        x2 = x2.transpose(-1, -2) #(B, L, D)
        x2 = self.fc2(x2) # (B, L, embed_dim)

        x = x1 + x2
        return x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        t = torch.arange(max_seq_len)
        # (max_seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  
        # interleave to (max_seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)        
        # register buffers so they move with .to(device)
        self.register_buffer("cos_emb", emb.cos()[None, None, :, :])  
        self.register_buffer("sin_emb", emb.sin()[None, None, :, :])  

    def forward(self, x):
        # x: (batch, heads, seq_len, head_dim)
        seq_len = x.shape[2]
        cos = self.cos_emb[:, :, :seq_len, :]
        sin = self.sin_emb[:, :, :seq_len, :]
        # rotate pairs
        x2 = torch.stack([-x[..., 1::2], x[..., 0::2]], -1).reshape_as(x)
        return x * cos + x2 * sin

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads,
                 device='cuda',
                 max_seq_len=10000,
                 cross_attn=False,):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device
        self.cross_attn = cross_attn

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        if not cross_attn:
            self.rotary = RotaryEmbedding(dim=self.head_dim, 
                                          max_seq_len=max_seq_len)

    def forward(self, 
                query, 
                key, 
                value, 
                mask=None,):
        batch_size = query.size(0)
        len_q, len_k, len_v = query.size(1), key.size(1), value.size(1)
        
        # Linear transformations and split into heads
        # [batchsize, seq_len, (num_heads*head_dim)]
        Q = self.query(query).view(batch_size, len_q, self.num_heads, self.head_dim) 
        K = self.key(key).view(batch_size, len_k, self.num_heads, self.head_dim)
        V = self.value(value).view(batch_size, len_v, self.num_heads, self.head_dim)

        # Transpose 
        # [batchsize, num_heads, seq_len, head_dim]
        Q = Q.transpose(1,2) 
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        if not self.cross_attn:
            Q = self.rotary(Q)
            K = self.rotary(K)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(2, 3)) * self.scale # (batchsize, num_head, q_len, k_len)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, Q.shape[2], -1) # (batchsize, head_dim, q_len, k_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            mask = mask.to(self.device)
            scores = scores.masked_fill(mask==0, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        self.last_attention = attention.detach().cpu()
        attention = F.dropout(attention, p=0.1)
        output = torch.matmul(attention, V) # [batchsize, num_heads, q_len, head_dim]

        # Concatenate heads and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim) # [batchsize, q_len, embed_dim]
        output = self.out(output) # [batchsize, q_len, embed_dim]

        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class AdditivePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # Learnable positional embedding: shape [max_len, d_model]
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        # Position indices: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)  # [batch_size, seq_len]

        # print("position_ids.max() =", position_ids.max().item())
        # print("embedding size =", self.pos_embedding.num_embeddings)
        
        # Get positional embeddings
        pos_emb = self.pos_embedding(position_ids)  # [batch_size, seq_len, d_model]
        return x + pos_emb

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, max_seq_len=10000, dropout=0.1, device='cuda'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim=embed_dim, 
                                            num_heads=num_heads, 
                                            device=device,
                                            max_seq_len=max_seq_len)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-Head Attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Position-wise Feed-Forward Network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, max_seq_len=10000, dropout=0.1, device='cuda'):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim=embed_dim, 
                                     num_heads=num_heads, 
                                     ff_dim=ff_dim, 
                                     max_seq_len=max_seq_len,
                                     dropout=dropout, 
                                     device=device) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, max_seq_len=10000, dropout=0.1, device='cuda', tgt_mask=None):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim=embed_dim, 
                                            num_heads=num_heads, 
                                            device=device,
                                            max_seq_len=max_seq_len,
                                            cross_attn=False,)
        self.cross_attn = MultiHeadAttention(embed_dim=embed_dim, 
                                             num_heads=num_heads, 
                                             device=device,
                                             cross_attn=True,
                                             max_seq_len=max_seq_len,)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn_output = self.cross_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, max_seq_len=10000, dropout=0.1, device='cuda'):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(embed_dim=embed_dim, 
                                     num_heads=num_heads, 
                                     ff_dim=ff_dim, 
                                     max_seq_len=max_seq_len,
                                     dropout=dropout, 
                                     device=device) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class LinearHead(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_sizes,
                 output_size,
                 dropout):
        super(LinearHead, self).__init__()
        self.activation = nn.ReLU()
        layers = []
        for h in hidden_sizes:
            layer = nn.Linear(input_size, h)
            layers += [
                layer,
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            input_size = h
        layers.append(nn.Linear(h, output_size))
        self.transform = nn.Sequential(*layers)
    def forward(self, x):
        return self.transform(x)

class CrossAttentionPredictor(nn.Module):
    def __init__(self,  
                 mirna_max_len:int,
                 mrna_max_len:int, 
                 vocab_size:int=12, # 7 special tokens + 5 bases
                 num_layers:int=2, 
                 embed_dim:int=256, 
                 num_heads:int=2, 
                 ff_dim:int=512,
                 hidden_sizes:list[int]=[512, 512],
                 n_classes:int=1, 
                 dropout_rate:float=0.1,
                 device:str='cuda',
                 predict_span=True,
                 predict_binding=False):
        super(CrossAttentionPredictor, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.device = device
        self.sn_embedding = nn.Embedding(vocab_size, embed_dim)
        self.cnn_embedding = CNNTokenization(embed_dim)

        self.mirna_encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim, 
            max_seq_len=mirna_max_len,
            device=device,
            dropout=dropout_rate,
        )
        self.mrna_encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim, 
            max_seq_len=mrna_max_len,
            device=device,
            dropout=dropout_rate,
        )
        self.cross_attn_layer = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
            cross_attn=True, # no positional encoding in cross attention
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.cross_norm = nn.LayerNorm(embed_dim) # normalize over embedding dimension
        self.qa_outputs = nn.Linear(embed_dim, 2) # Linear head instead of one Linear transformation
        self.binding_output = LinearHead(
            input_size=embed_dim, 
            hidden_sizes=hidden_sizes,
            output_size=n_classes,
            dropout=dropout_rate)
        self.predict_span = predict_span
        self.predict_binding = predict_binding
    
    def forward(self, mirna, mrna, mrna_mask, mirna_mask):
        mirna_sn_embedding = self.sn_embedding(mirna)
        mrna_sn_embedding = self.sn_embedding(mrna)
        
        # add N-gram CNN-encoded embedding
        mirna_cnn_embedding = self.cnn_embedding(mirna_sn_embedding.transpose(-1, -2)) # (batch_size, embed_dim, mirna_len)
        mrna_cnn_embedding = self.cnn_embedding(mrna_sn_embedding.transpose(-1, -2))  # (batch_size, embed_dim, mrna_len)
        mirna_embedding = mirna_sn_embedding + mirna_cnn_embedding # (batch_size, mirna_len, embed_dim)
        mrna_embedding = mrna_sn_embedding + mrna_cnn_embedding # (batch_size, mrna_len, embed_dim)

        mirna_embedding = self.mirna_encoder(mirna_embedding, mask=mirna_mask)  # (batch_size, mirna_len, embed_dim)
        mrna_embedding = self.mrna_encoder(mrna_embedding, mask=mrna_mask) # (batch_size, mrna_len, embed_dim)
        
        z = self.cross_attn_layer(query=mrna_embedding, 
                                  key=mirna_embedding,
                                  value=mirna_embedding,
                                  mask=mirna_mask) # pass key-mask
        self.cross_attn_output = z
        z_res = self.dropout(z) + mrna_embedding # residual connection
        z_norm = self.cross_norm(z_res)
        z_norm = z_norm.masked_fill(mrna_mask.unsqueeze(-1)==0, 0) # (batch_size, mrna_len, embed_dim)
        
        if self.predict_binding:
            valid_counts = mrna_mask.sum(dim=1, keepdim=True) # (batch_size)
            # avg pooling over seq_len
            z_norm_mean = z_norm.sum(dim=1) / (valid_counts + 1e-8) # (batch_size, embed_dim)
            # predict binding
            binding_logits = self.binding_output(z_norm_mean) # (batchsize, 1)
        else:
            binding_logits = None
        if self.predict_span:
            # predict start and end
            span_logits = self.qa_outputs(z_norm) # (batchsize, mrna_len, 2)
            start_logits, end_logits = span_logits[...,0], span_logits[...,1] # (batchsize, mrna_len)
        else:
            start_logits, end_logits = None, None
        return binding_logits, start_logits, end_logits
    
class QuestionAnsweringModel(nn.Module):
    def __init__(self,
                mrna_max_len,
                mirna_max_len,
                device: str='cuda',
                epochs:int=100,
                embed_dim=256,
                ff_dim:int=512,
                batch_size:int=32,
                lr=0.001,
                seed=42,
                predict_span=True,
                predict_binding=False,
                use_cross_attn=True,):
        super(QuestionAnsweringModel, self).__init__()
        self.mrna_max_len = mrna_max_len
        self.mirna_max_len = mirna_max_len
        self.device = device
        self.epochs = epochs
        self.ff_dim = ff_dim
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.predict_binding = predict_binding
        self.predict_span = predict_span
        if use_cross_attn:
            self.predictor = CrossAttentionPredictor(mrna_max_len=mrna_max_len,
                                                    mirna_max_len=mirna_max_len,
                                                    embed_dim = embed_dim,
                                                    ff_dim = ff_dim,
                                                    hidden_sizes = [ff_dim, ff_dim],
                                                    device=device,
                                                    predict_span=predict_span,
                                                    predict_binding=predict_binding)
    
    def forward(self, 
                mirna, 
                mrna, 
                mrna_mask, 
                mirna_mask,):
        return self.predictor(mirna=mirna, 
                              mrna=mrna, 
                              mrna_mask=mrna_mask,
                              mirna_mask=mirna_mask,)
    
    @staticmethod
    def compute_span_metrics(start_preds, end_preds, start_labels, end_labels):
        """
        计算 exact match 和 F1 score.
        输入张量都是 [B]，代表每个样本的 start/end.
        """
        exact_matches = 0
        f1_total = 0.0
        n = len(start_preds)

        for i in range(n):
            pred_start = int(start_preds[i])
            pred_end   = int(end_preds[i])
            true_start = int(start_labels[i])
            true_end   = int(end_labels[i])

            # Compute overlap
            overlap_start = max(pred_start, true_start)
            overlap_end   = min(pred_end, true_end)
            overlap       = max(0, overlap_end - overlap_start)

            pred_len = max(1, pred_end - pred_start)
            true_len = max(1, true_end - true_start)

            precision = overlap / pred_len
            recall = overlap / true_len
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            f1_total += f1

            # Exact match
            if pred_start == true_start and pred_end == true_end:
                exact_matches += 1

        return {
            "exact_match": exact_matches / n,
            "f1": f1_total / n,
        }
    
    def train_loop(self, 
              model, 
              dataloader, 
              loss_fn,
              optimizer, 
              device,
              epoch,
              accumulation_step=1,
              alpha1=1,
              alpha2=1):
        '''
        Training loop
        '''
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        loss_list = []
        for batch_idx, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)

            mirna_mask = batch["mirna_attention_mask"]
            mrna_mask  = batch["mrna_attention_mask"]
            
            outputs = model(
                mirna=batch["mirna_input_ids"],
                mrna=batch["mrna_input_ids"],
                mirna_mask=mirna_mask,
                mrna_mask=mrna_mask,
            )
            binding_logits, start_logits, end_logits = outputs  # [B, L]
               
            if self.predict_span:
                # mask padded output in start and end logits
                start_logits = start_logits.masked_fill(mrna_mask==0, float("-inf"))
                end_logits   = end_logits.masked_fill(mrna_mask==0, float("-inf"))
                start_positions = batch["start_positions"]
                end_positions   = batch["end_positions"]

            span_loss    = torch.tensor(0.0, device=device)
            binding_loss = torch.tensor(0.0, device=device)

            if self.predict_binding and binding_logits is not None:
                binding_targets = batch["target"]
                binding_loss_fn = nn.BCEWithLogitsLoss()
                binding_loss    = binding_loss_fn(binding_logits.squeeze(-1), binding_targets.view(-1).float())
                pos_mask        = binding_targets.view(-1).bool()
                if self.predict_span and pos_mask.any():
                    # only loss of positive pairs are counted
                    loss_start = loss_fn(start_logits[pos_mask,], start_positions[pos_mask]) # CrossEntropyLoss expects [B, L], labels as [B]
                    loss_end   = loss_fn(end_logits[pos_mask,], end_positions[pos_mask])
                    span_loss  = 0.5 * (loss_start + loss_end)
                    loss       = alpha1 * binding_loss + alpha2 * span_loss
                else:
                    loss       = binding_loss
            elif self.predict_span:
                # assume all mirna-mrna pairs are positive
                # CrossEntropyLoss expects [B, L], labels as [B]
                loss_start = loss_fn(start_logits, start_positions)
                loss_end   = loss_fn(end_logits, end_positions)
                span_loss  = 0.5 * (loss_start + loss_end)
                loss       = span_loss 

            loss = loss / accumulation_step # why do we divide by accumulation_step here?
            loss.backward()
            bs = batch["mrna_input_ids"].size(0)
            if accumulation_step != 1:
                loss_list.append(loss.item())
                if (batch_idx + 1) % accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print(
                        f"Train Epoch: {epoch} "
                        f"[{(batch_idx + 1) * bs}/{len(dataloader.dataset)} "
                        f"({(batch_idx + 1) * bs / len(dataloader.dataset) * 100:.0f}%)] "
                        f"Avg loss: {sum(loss_list) / len(loss_list):.6f}\n",
                        flush=True
                    )
                    loss_list = []
            else:
                optimizer.step()
                optimizer.zero_grad()
                print(
                    f"Train Epoch: {epoch} "
                    f"[{(batch_idx + 1) * bs}/{len(dataloader.dataset)} "
                    f"({(batch_idx + 1) * bs / len(dataloader.dataset) * 100:.0f}%)] "
                    f"Span Loss: {span_loss.item():.6f} "
                    f"Binding Loss: {binding_loss.item():.6f}\n",
                    flush=True
                ) 

            total_loss += loss.item() * accumulation_step
        # After the loop, if gradients remain (for non-divisible number of batches)
        if (batch_idx + 1) % accumulation_step != 0:
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def eval_loop(self, 
                  model, 
                  dataloader, 
                  device,
                  alpha1=1,
                  alpha2=0.25,
                  evaluation=False):
        model.eval()
        total_loss = 0.0 
        all_start_preds, all_end_preds        = [], []
        all_binding_preds, all_binding_labels = [], []
        all_binding_probs                     = []
        all_start_labels, all_end_labels      = [], []

        with torch.no_grad():
            for batch in dataloader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                mrna_mask  = batch["mrna_attention_mask"]
                mirna_mask = batch["mirna_attention_mask"]
                outputs    = model(
                    mirna=batch["mirna_input_ids"],
                    mrna=batch["mrna_input_ids"],
                    mirna_mask=mirna_mask,
                    mrna_mask=mrna_mask,
                )
                binding_logits, start_logits, end_logits = outputs

                if self.predict_span:
                    # mask padded mrna tokens
                    start_logits = start_logits.masked_fill(mrna_mask==0, float("-inf"))
                    end_logits   = end_logits.masked_fill(mrna_mask==0, float("-inf"))
                    start_positions = batch["start_positions"] # (batchsize, )
                    end_positions   = batch["end_positions"] # (batchsize, )

                # Compute loss
                loss_fn = nn.CrossEntropyLoss()
                loss    = 0.0 
                if self.predict_binding and binding_logits is not None:
                    # Compute binding loss if binding label is predicted
                    binding_targets = batch["target"] # (batchsize, )
                    binding_loss_fn = nn.BCEWithLogitsLoss()
                    binding_loss    = binding_loss_fn(binding_logits.squeeze(-1), 
                                                      binding_targets.view(-1).float())
                    # binding metric
                    binding_probs   = F.sigmoid(binding_logits)
                    binding_preds   = (binding_probs > 0.5).to(torch.int)
                    all_binding_probs.extend(binding_probs.cpu())
                    all_binding_preds.extend(binding_preds.cpu())
                    all_binding_labels.extend(binding_targets.view(-1).cpu())
                else:
                    binding_loss = None

                # span loss and predictions
                if self.predict_span and start_logits is not None and end_logits is not None:
                    # predict both binding and span
                    if self.predict_binding:
                        pos_mask    = binding_targets.view(-1).bool() # (batchsize, )
                    else: # not predicting binding, then assume all are positive samples
                        pos_mask    = torch.ones_like(start_positions, dtype=torch.bool, device=start_positions.device)

                    if pos_mask.any():
                        start_logits    = start_logits[pos_mask,]
                        end_logits      = end_logits[pos_mask,]
                        start_positions = start_positions[pos_mask]
                        end_positions   = end_positions[pos_mask]   
                    
                        # span loss 
                        loss_start  = loss_fn(start_logits, start_positions)
                        loss_end    = loss_fn(end_logits, end_positions)
                        span_loss   = 0.5 * (loss_start + loss_end)  

                        # predictions
                        start_preds = torch.argmax(start_logits, dim=-1) #(batch_size, )
                        end_preds   = torch.argmax(end_logits, dim=-1) #(batch_size, )
                        all_start_preds.extend(start_preds.cpu())
                        all_end_preds.extend(end_preds.cpu())
                        all_start_labels.extend(start_positions.cpu())
                        all_end_labels.extend(end_positions.cpu())
                    else:
                        span_loss = torch.tensor(0.0, device=start_positions.device) # no positive samples
                else: #
                    span_loss = None
                
                if binding_loss is not None:
                    loss += alpha1 * binding_loss
                if span_loss is not None:
                    loss += alpha2 * span_loss
                
                total_loss += loss.item()

        # if there are positive examples
        if len(all_start_preds) > 0:
            all_start_preds  = torch.stack(all_start_preds).detach().cpu().long()
            all_start_labels = torch.stack(all_start_labels).detach().cpu().long()
            all_end_preds    = torch.stack(all_end_preds).detach().cpu().long()
            all_end_labels   = torch.stack(all_end_labels).detach().cpu().long()
            acc_start        = (all_start_preds == all_start_labels).float().mean().item()
            acc_end          = (all_end_preds == all_end_labels).float().mean().item()
            span_metrics     = self.compute_span_metrics(
                all_start_preds, all_end_preds, all_start_labels, all_end_labels)
            exact_match      = span_metrics["exact_match"]
            f1               = span_metrics["f1"]
        else:
            print("No positive example in this epoch. No span metrics is measured.")
            acc_start   = 0.0
            acc_end     = 0.0
            exact_match = 0.0
            f1          = 0.0

        if self.predict_binding:
            all_binding_probs  = torch.tensor(all_binding_probs, dtype=torch.float)
            all_binding_labels = torch.tensor(all_binding_labels, dtype=torch.long)
            all_binding_preds  = torch.tensor(all_binding_preds, dtype=torch.long)
            acc_binding        = (all_binding_preds == all_binding_labels).float().mean().item()
        else:
            acc_binding        = 0.0

        avg_loss = total_loss / len(dataloader)
        
        print(f"Start Acc:   {acc_start*100}%\n"
              f"End Acc:     {acc_end*100}%\n"
              f"Span Exact Match: {exact_match*100}%\n"
              f"F1 Score:    {f1}\n"
              f"Binding Acc: {acc_binding*100}")
        
        if evaluation:
            if self.predict_binding:
                self.all_binding_probs = all_binding_probs.numpy()
                self.all_binding_preds = all_binding_preds.numpy()
            if self.predict_span:
                self.all_start_preds = all_start_preds.numpy()
                self.all_end_preds = all_end_preds.numpy()

        return avg_loss, acc_binding, acc_start, acc_end, exact_match, f1    

    def predict(self, 
                model, 
                dataloader,
                device):
        model.eval()
        all_binding_probs, all_start_preds, all_end_preds = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                mrna_mask  = batch["mrna_attention_mask"]
                mirna_mask = batch["mirna_attention_mask"]
                outputs    = model(
                    mirna=batch["mirna_input_ids"],
                    mrna=batch["mrna_input_ids"],
                    mirna_mask=mirna_mask,
                    mrna_mask=mrna_mask,
                )
                binding_logits, start_logits, end_logits = outputs
                if self.predict_binding:
                    binding_probs = F.sigmoid(binding_logits)
                    binding_probs = binding_probs.detach().cpu()
                    all_binding_probs.extend(binding_probs)
                if self.predict_span:
                    start_logits = start_logits.masked_fill(mrna_mask==0, float("-inf"))
                    end_logits   = end_logits.masked_fill(mrna_mask==0, float("-inf"))
                    # predictions
                    start_preds = torch.argmax(start_logits, dim=-1) #(batch_size, )
                    end_preds   = torch.argmax(end_logits, dim=-1) #(batch_size, )
                    start_preds = start_preds.detach().cpu()
                    end_preds = end_preds.detach().cpu()
                    all_start_preds.extend(start_preds)
                    all_end_preds.extend(end_preds)

        all_binding_probs = torch.stack(all_binding_probs)
        all_start_preds = torch.stack(all_start_preds)
        all_end_preds = torch.stack(all_end_preds)
        print(all_binding_probs.shape, all_start_preds.shape, all_end_preds.shape)
        # all_binding_probs = np.concatenate(all_binding_probs, axis=0).numpy()
        # all_start_preds = np.concatenate(all_start_preds, axis=0).numpy()
        # all_end_preds = np.concatenate(all_end_preds, axis=0).numpy()
        
        return all_binding_probs, all_start_preds, all_end_preds

    @staticmethod 
    def seed_everything(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    def run(self, 
            model,
            train_path="",
            valid_path="",
            test_path="",
            evaluation=False,
            predict=False,
            finetune=False,
            accumulation_step=1,
            ckpt_path="",
            ):
        tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                       add_special_tokens=False, 
                                       model_max_length=self.mrna_max_len,
                                       padding_side="right")
        if predict:
            D_test = load_dataset(test_path, sep=',')
            ds_test = QuestionAnswerDataset(data=D_test,
                                mrna_max_len=self.mrna_max_len,
                                mirna_max_len=self.mirna_max_len,
                                tokenizer=tokenizer,
                                seed_start_col="seed start",
                                seed_end_col="seed end",)
            test_loader = DataLoader(ds_test,
                                    batch_size=self.batch_size, 
                                    shuffle=False)
            loaded_data = torch.load(ckpt_path, map_location=model.device)
            current_state = model.state_dict()
            for key in list(loaded_data.keys()):
                if key not in current_state:
                    continue
                if loaded_data[key].shape == current_state[key].shape:
                    continue
                if "rotary.cos_emb" in key or "rotary.sin_emb" in key:
                    orig_shape = loaded_data[key].shape
                    loaded_data[key] = current_state[key].clone()
                    print(f"Replaced rotary buffer {key} with shape {orig_shape} to match model shape {current_state[key].shape}")
                else:
                    print(f"Dropping mismatched key {key}: checkpoint {loaded_data[key].shape}, model {current_state[key].shape}")
                    loaded_data.pop(key)
            model.load_state_dict(loaded_data, strict=False)
            print(f"Loaded checkpoint from {ckpt_path}")
            model.to(self.device)
            all_binding_probs, all_start_preds, all_end_preds = self.predict(model=model,
                                                            dataloader=test_loader,
                                                            device=self.device)
            D_test["binding_probs"] = all_binding_probs
            D_test["start_preds"] = all_start_preds
            D_test["end_preds"] = all_end_preds
            D_test.to_csv(os.path.join(os.path.dirname(test_path), "zero_shot_prediction_best_composite_1.0000_1.0000_epoch10.csv"), index=False)
            print(f"Zero shot prediction saved to {os.path.join(os.path.dirname(test_path), 'zero_shot_prediction_best_composite_1.0000_1.0000_epoch10.csv')}")
        elif evaluation:
            D_test = load_dataset(test_path, sep=',')
            ds_test = QuestionAnswerDataset(data=D_test,
                                mrna_max_len=self.mrna_max_len,
                                mirna_max_len=self.mirna_max_len,
                                tokenizer=tokenizer,
                                seed_start_col="seed start",
                                seed_end_col="seed end",)
            test_loader = DataLoader(ds_test,
                                    batch_size=self.batch_size, 
                                    shuffle=False)
            loaded_data = torch.load(ckpt_path, map_location=model.device)
            current_state = model.state_dict()
            for key in list(loaded_data.keys()):
                if key not in current_state:
                    continue
                if loaded_data[key].shape == current_state[key].shape:
                    continue
                if "rotary.cos_emb" in key or "rotary.sin_emb" in key:
                    orig_shape = loaded_data[key].shape
                    loaded_data[key] = current_state[key].clone()
                    print(f"Replaced rotary buffer {key} with shape {orig_shape} to match model shape {current_state[key].shape}")
                else:
                    print(f"Dropping mismatched key {key}: checkpoint {loaded_data[key].shape}, model {current_state[key].shape}")
                    loaded_data.pop(key)
            model.load_state_dict(loaded_data, strict=False)
            print(f"Loaded checkpoint from {ckpt_path}")
            model.to(self.device)
            self.eval_loop(model=model, 
                           dataloader=test_loader,
                           device=self.device,
                           evaluation=evaluation)
            D_test_w_pred = D_test.copy()
            if self.predict_binding:
                D_test_w_pred["pred label"] = self.all_binding_preds
                D_test_w_pred["pred prob"]  = self.all_binding_probs
                res_df = D_test_w_pred
            if self.predict_span:
                D_test_positive = D_test_w_pred.loc[D_test_w_pred["label"] == 1].copy()
                D_test_positive["pred start"] = self.all_start_preds
                D_test_positive["pred end"]   = self.all_end_preds
                # merge D_test_w_pred with D_test_positive
                cols = ['pred start', 'pred end']
                D_pred_se = D_test_positive[cols]

                # 2. 左连接（保留 D_test_w_pred 的所有行）
                D_merged = D_test_w_pred.join(D_pred_se, how='left')

                # 3. 将缺失的 pred start/end 填成 -1，并转成整数
                D_merged[['pred start', 'pred end']] = (
                    D_merged[['pred start', 'pred end']]
                    .fillna(-1)
                    .astype(int)
                )
                res_df = D_merged
            pred_df_path = os.path.join(os.path.join(PROJ_HOME, "Performance/TargetScan_test/TwoTowerTransformer"), str(mrna_max_len))
            os.makedirs(pred_df_path, exist_ok=True)
            res_df.to_csv(os.path.join(pred_df_path, "negative_with_seed_prediction.csv"), index=False)
            print(f"Prediction saved to {pred_df_path}")
        else:
            # weights and bias initialization
            wandb.login(key="600e5cca820a9fbb7580d052801b3acfd5c92da2")
            run = wandb.init(
                project="mirna-Question-Answering",
                name=f"CNN_binding-span-random-start-len:{self.mrna_max_len}-epoch:{self.epochs}-batchsize:{self.batch_size}-2layerTrans-{self.ff_dim}MLP_hidden", 
                config={
                    "batch_size": self.batch_size * accumulation_step,
                    "epochs": self.epochs,
                    "learning rate": self.lr,
                },
                tags=["binding-span", "primates", "CNN-5-7-kernel", "non-canonical", "finetune"],
                save_code=False,
                job_type="train"
            )
            self.seed_everything(seed=self.seed)
            # load dataset
            D_train  = load_dataset(train_path, sep=',')
            D_val    = load_dataset(valid_path, sep=',')
            ds_train = QuestionAnswerDataset(data=D_train,
                                            mrna_max_len=self.mrna_max_len,
                                            mirna_max_len=self.mirna_max_len,
                                            tokenizer=tokenizer,
                                            seed_start_col="seed start",
                                            seed_end_col="seed end",)
            ds_val = QuestionAnswerDataset(data=D_val,
                                        mrna_max_len=self.mrna_max_len,
                                        mirna_max_len=self.mirna_max_len,
                                        tokenizer=tokenizer,
                                        seed_start_col="seed start",
                                        seed_end_col="seed end",)
            train_loader = DataLoader(ds_train, 
                                batch_size=self.batch_size, 
                                shuffle=True)
            val_loader   = DataLoader(ds_val, 
                                    batch_size=self.batch_size, 
                                    shuffle=False)
            loss_fn   = nn.CrossEntropyLoss()

            if finetune:
                loaded_data = torch.load(ckpt_path, map_location=self.device)
                current_state = model.state_dict()
                for key in list(loaded_data.keys()):
                    if key not in current_state:
                        continue
                    if loaded_data[key].shape == current_state[key].shape:
                        continue
                    if "rotary.cos_emb" in key or "rotary.sin_emb" in key:
                        orig_shape = loaded_data[key].shape
                        loaded_data[key] = current_state[key].clone()
                        print(f"Replaced rotary buffer {key} with shape {orig_shape} to match model shape {current_state[key].shape}")
                    else:
                        print(f"Dropping mismatched key {key}: checkpoint {loaded_data[key].shape}, model {current_state[key].shape}")
                        loaded_data.pop(key)
                model.load_state_dict(loaded_data, strict=False)

            print(f"Loaded checkpoint from {ckpt_path}")
            model.to(self.device)
            
            if self.predict_binding and not self.predict_span:
                # freeze update of params in the span prediction head
                for p in model.predictor.qa_outputs.parameters():
                    p.requires_grad = False
            elif self.predict_span and not self.predict_binding:
                # freeze update of params in the binding prediction head
                for p in model.predictor.binding_output.parameters():
                    p.requires_grad = False

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=1e-2)

            start    = time()
            count    = 0
            patience = 10
            best_binding_acc = 0
            best_exact_match = 0
            best_f1_score    = 0
            best_composite_metric = 0
            model_checkpoints_dir = os.path.join(
                PROJ_HOME, 
                "checkpoints", 
                "TargetScan", 
                "TwoTowerTransformer", 
                "CNN-tokenized",
                str(self.mrna_max_len),
            )
            os.makedirs(model_checkpoints_dir, exist_ok=True)
            for epoch in range(self.epochs):
                train_loss = self.train_loop(model=model,
                        dataloader=train_loader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        device=self.device,
                        epoch=epoch,
                        accumulation_step=accumulation_step)
                eval_loss, acc_binding, acc_start, acc_end, exact_match, f1 = self.eval_loop(model=model,
                                                                                dataloader=val_loader,
                                                                                device=self.device,)
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "eval/loss": eval_loss,
                    "eval/binding accuracy": acc_binding,
                    "eval/start accuracy": acc_start,
                    "eval/end accuracy": acc_end,
                    "eval/exact match": exact_match,
                    "eval/F1 score": f1
                }, step=epoch)
                
                if self.predict_binding and self.predict_span:
                    composite_metric = f1 + acc_binding
                    if composite_metric > best_composite_metric:
                        best_composite_metric = composite_metric
                        best_binding_acc      = acc_binding
                        best_f1_score         = f1
                        count = 0
                        ckpt_name = f"finetuned_best_composite_{f1:.4f}_{acc_binding:.4f}_epoch{epoch}.pth"
                        ckpt_path = os.path.join(model_checkpoints_dir, ckpt_name)
                        torch.save(model.state_dict(), ckpt_path)
                        model_art = wandb.Artifact(
                            name="binding-span-model",
                            type="model",
                            metadata={ "epoch": epoch, "f1 + acc_binding": composite_metric }
                        )
                        model_art.add_file(ckpt_path)
                        run.log_artifact(model_art)
                        # mark as the “latest”
                        run.log_artifact(model_art).wait()
                    else:
                        count += 1
                        if count == patience:
                            print("Max patience reached with no improvement on accuracy. Early stop triggered.")
                            break
                elif self.predict_binding and not self.predict_span:
                    if acc_binding >= best_binding_acc:
                        best_binding_acc = acc_binding
                        count = 0
                        ckpt_name = f"primates_best_binding_acc_{acc_binding:.4f}_epoch{epoch}.pth"
                        ckpt_path = os.path.join(model_checkpoints_dir, ckpt_name)
                        torch.save(model.state_dict(), ckpt_path)
                        # create new artifact
                        model_art = wandb.Artifact(
                            name="mirna-binding-model",
                            type="model",
                            metadata={ "epoch": epoch, "binding_acc": acc_binding }
                        )
                        model_art.add_file(ckpt_path)
                        run.log_artifact(model_art)
                        # mark as the “latest”
                        run.log_artifact(model_art).wait()
                    else:
                        count += 1
                        if count == patience:
                            print("Max patience reached with no improvement on accuracy. Early stop triggered.")
                            break
                elif self.predict_span and not self.predict_binding:
                    if exact_match >= best_exact_match:
                        best_exact_match = exact_match
                        count = 0
                        ckpt_name = f"best_exact_match_{exact_match:.4f}_epoch{epoch}.pth"
                        ckpt_path = os.path.join(model_checkpoints_dir, ckpt_name)
                        torch.save(model.state_dict(), ckpt_path)
                        # create new artifact
                        model_art = wandb.Artifact(
                            name="mirna-span-model",
                            type="model",
                            metadata={ "epoch": epoch, "exact_match": exact_match}
                        )
                        model_art.add_file(ckpt_path)
                        run.log_artifact(model_art)
                        # mark as the “latest”
                        run.log_artifact(model_art).wait()
                    else:
                        count += 1
                        if count == patience:
                            print("Max patience reached with no improvement on accuracy. Early stop triggered.")
                            break
                cost = time() - start
                remain = cost/(epoch + 1) * (self.epochs - epoch - 1) /3600
                print(f'still remain: {remain} hrs.')

            cost = time() - start
            print(f"Training takes {cost / 3600} hours.")
            run.summary["best_binding_acc"] = best_binding_acc
            run.summary["best F1 score"]    = best_f1_score
            run.summary["best_epoch"]       = int(model_art.metadata["epoch"])
            run.finish()

class TargetGenerationModel(nn.Module):
    def __init__(self,
                 mirna_max_len:int,
                 mrna_max_len:int,
                 lr:float=1e-4,
                 batch_size:int=32,
                 vocab_size:int=13, # 8 special tokens + 5 bases
                 num_layers:int=2, 
                 embed_dim:int=256, 
                 num_heads:int=2, 
                 ff_dim:int=512,
                 hidden_sizes:list[int]=[512, 512],
                 n_classes:int=13, 
                 dropout_rate:float=0.1,
                 device:str='cuda',
                 predict_span=True,
                 predict_binding=False,
                 seed:int=42):
        super(TargetGenerationModel, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.mrna_max_len = mrna_max_len
        self.mirna_max_len = mirna_max_len
        self.sn_embedding = nn.Embedding(vocab_size, embed_dim)
        self.cnn_embedding = CNNTokenization(embed_dim=embed_dim)
        self.mrna_encoder = TransformerEncoder(
            num_layers=num_layers, 
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            max_seq_len=mrna_max_len, 
            dropout=dropout_rate, 
            device=device)
        self.mirna_decoder = TransformerDecoder(
            num_layers=num_layers, 
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            max_seq_len=mirna_max_len, 
            dropout=dropout_rate, 
            device=device)
        self.predictor_head = LinearHead(
            input_size=embed_dim, 
            hidden_sizes=hidden_sizes,
            output_size=n_classes,
            dropout=dropout_rate)
        
        self.tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                            add_special_tokens=False, 
                            model_max_length=self.mrna_max_len-2, # minus 2 for BOS and EOS tokens
                            padding_side="right")
        self.pad_idx = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.bos_idx = self.tokenizer.convert_tokens_to_ids("[BOS]")
        self.eos_idx = self.tokenizer.convert_tokens_to_ids("[EOS]")

    def forward(self,
                mirna,
                mrna,
                mrna_mask,
                mirna_mask,):
        mrna_sn_embedding = self.sn_embedding(mrna)
        mirna_sn_embedding = self.sn_embedding(mirna)
        
        mirna_embedding = mirna_sn_embedding # no CNN embedding for miRNA

        mrna_cnn_embedding = self.cnn_embedding(mrna_sn_embedding.transpose(-1, -2))
        mrna_embedding = mrna_sn_embedding + mrna_cnn_embedding
        mrna_embedding = self.mrna_encoder(mrna_embedding, mask=mrna_mask)

        mirna_embedding = self.mirna_decoder(x=mirna_embedding, memory=mrna_embedding, src_mask=mrna_mask, tgt_mask=mirna_mask)
        next_token_logits = self.predictor_head(mirna_embedding)
        return next_token_logits

    @staticmethod
    def generate_square_subsequent_mask(L, device=None):
        # shape: (1, L, L) with 1 on lower triangle
        mask = torch.tril(torch.ones(L, L, dtype=torch.uint8))
        mask = mask.unsqueeze(0)
        if device is not None:
            mask = mask.to(device)
        return mask

    def create_src_mask(self, src_tokens):
        """
        src_tokens: (B, L_src)
        Returns mask of shape (B, 1, L_src) with 1 for valid, 0 for pad.
        Here we only mask out padded positions as keys.
        """
        B, L_src = src_tokens.size()
        # src_key_padding_mask: (B, L_src) -> 1 if non-pad, 0 if pad
        non_pad = (src_tokens != self.pad_idx).to(torch.uint8) # (B, L_src)
        # Broadcast to (B, L_src, L_src): query positions always allowed,
        # key positions masked if pad.
        src_mask = non_pad.unsqueeze(1) # (B, 1, L_src)
        return src_mask

    def create_tgt_mask(self, tgt_input):
        """
        tgt_input: (B, L_tgt)
        Combines:
        - causal mask (no attending to future)
        - padding mask (PAD tokens shouldn't be attended to)
        Returns (B, L_tgt, L_tgt)
        """
        B, L_tgt = tgt_input.size()

        # causal: (1, L_tgt, L_tgt)
        causal = self.generate_square_subsequent_mask(L_tgt, device=tgt_input.device)

        # padding: (B, L_tgt, 1) broadcast to (B, L_tgt, L_tgt)
        non_pad = (tgt_input != self.pad_idx).to(torch.uint8)  # (B, L_tgt)
        non_pad = non_pad.unsqueeze(1)                    # (B, 1, L_tgt)
        non_pad = non_pad.repeat(1, L_tgt, 1)             # (B, L_tgt, L_tgt)

        # combine: valid if both not masked by causal and not PAD
        # causal: (1, L, L) -> (B, L, L) by broadcasting
        tgt_mask = causal & non_pad  # uint8 AND
        return tgt_mask

    def train_loop(self,
                   model,
                   dataloader,
                   loss_fn,
                   optimizer,
                   device,
                   epoch,
                   accumulation_step=1,):
        model.train()
        total_loss = 0
        loss_list = []
        for batch_idx, batch in enumerate(dataloader):
            mirna_input = batch["mirna_input_ids"].to(device)
            mrna_input = batch["mrna_input_ids"].to(device)
            # 1) Build decoder inputs/outputs by shifting
            # tgt_input: all but last token
            # tgt_output: all but first token
            tgt_input = mirna_input[:, :-1] # (B, L_mirna-1)
            tgt_output = mirna_input[:, 1:] # (B, L_mirna-1)
            src_input = mrna_input # (B, L_mrna)

            # 2) Masks
            src_mask = self.create_src_mask(mrna_input)    # (B, L_mrna, L_mrna)
            tgt_mask = self.create_tgt_mask(tgt_input)     # (B, L_mirna-1, L_mirna-1)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            logits = model(mirna=tgt_input, mrna=src_input, mrna_mask=src_mask, mirna_mask=tgt_mask)  # (B, L_mirna-1, n_classes)
            B, L, V = logits.size()
            loss = loss_fn(
                logits.view(B*L, V), 
                tgt_output.reshape(B*L,)) # (B*L, )

            loss = loss / accumulation_step 
            loss.backward()
            bs = batch["mrna_input_ids"].size(0)
            if accumulation_step != 1:
                loss_list.append(loss.item())
                if (batch_idx + 1) % accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print(
                        f"Train Epoch: {epoch} "
                        f"[{(batch_idx + 1) * bs}/{len(dataloader.dataset)} "
                        f"({(batch_idx + 1) * bs / len(dataloader.dataset) * 100:.0f}%)] "
                        f"Avg loss: {sum(loss_list) / len(loss_list):.6f}\n",
                        flush=True
                    )
                    loss_list = []
            else:
                optimizer.step()
                optimizer.zero_grad()
                print(
                    f"Train Epoch: {epoch} "
                    f"[{(batch_idx + 1) * bs}/{len(dataloader.dataset)} "
                    f"({(batch_idx + 1) * bs / len(dataloader.dataset) * 100:.0f}%)] "
                    f"Loss: {loss.item():.6f}\n",
                    flush=True
                ) 

            total_loss += loss.item() * accumulation_step
        # After the loop, if gradients remain (for non-divisible number of batches)
        if (batch_idx + 1) % accumulation_step != 0:
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def eval_loop(self,
                  model,
                  dataloader,
                  device,
                  epoch,
                  loss_fn,):
        model.eval()
        total_loss = 0
        total_correct_tokens = 0
        total_tokens = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                mirna_input = batch["mirna_input_ids"].to(device)
                mrna_input = batch["mrna_input_ids"].to(device)
                # 1) Build decoder inputs/outputs by shifting
                # tgt_input: all but last token
                # tgt_output: all but first token
                tgt_input = mirna_input[:, :-1] # (B, L_mirna-1)
                tgt_output = mirna_input[:, 1:] # (B, L_mirna-1)
                src_input = mrna_input # (B, L_mrna)

                # 2) Masks
                src_mask = self.create_src_mask(src_input)    # (B, L_mrna, L_mrna)
                tgt_mask = self.create_tgt_mask(tgt_input)     # (B, L_mirna-1, L_mirna-1)
                src_mask = src_mask.to(device)
                tgt_mask = tgt_mask.to(device)

                # 3) Forward pass
                logits = model(mirna=tgt_input, mrna=src_input, mrna_mask=src_mask, mirna_mask=tgt_mask)  # (B, L_mirna-1, n_classes)
                B, L, V = logits.size()
                valid_mask = (tgt_output != self.pad_idx)
                total_tokens += valid_mask.sum().item()
                loss = loss_fn(
                    logits.view(B*L, V), 
                    tgt_output.reshape(B*L,)) # (B*L, )
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                total_correct_tokens += ((preds == tgt_output) & valid_mask).sum().item() # only count real tokens
        avg_loss = total_loss / len(dataloader)
        avg_token_accuracy = total_correct_tokens / max(1, total_tokens)
        print(f"Total tokens: {total_tokens}")
        print(f"Total correct tokens: {total_correct_tokens}")
        print(
            f"Avg loss: {avg_loss:.6f} "
            f"Avg token accuracy: {avg_token_accuracy:.4f}\n",
            flush=True
        )
        return avg_loss, avg_token_accuracy

    def greedy_generate(self, model, src_tokens, max_len=40):
        """
        src_tokens: (1, L_src) single example
        Returns a list of token ids for the generated miRNA (excluding BOS).
        """
        model.eval()
        src_tokens = src_tokens.to(self.device)

        with torch.no_grad():
            # Encode mRNA once
            src_mask = self.create_src_mask(src_tokens).to(self.device)
            
            mrna_sn_embedding = model.sn_embedding(src_tokens)
            mrna_cnn_embedding = model.cnn_embedding(mrna_sn_embedding.transpose(-1, -2))
            src_embedding = mrna_sn_embedding + mrna_cnn_embedding
            
            memory = model.mrna_encoder(src_embedding, src_mask)

            # Start with BOS
            batch_size = src_tokens.size(0)
            generated = torch.full(
                (batch_size, 1), 
                self.bos_idx, 
                dtype=torch.long, 
                device=self.device
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            for _ in range(max_len):
                tgt_input = generated
                tgt_mask = self.create_tgt_mask(tgt_input).to(self.device)

                mirna_sn_embedding = model.sn_embedding(tgt_input)
                tgt_embedding = mirna_sn_embedding

                # Decode
                out = model.mirna_decoder(x=tgt_embedding, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
                logits = model.predictor_head(out)
                
                # Only last position logits used for next token
                next_token_logits = logits[:, -1, :]  # (B, vocab_size)
                next_token_id = next_token_logits.argmax(dim=-1) # [B]
                pad_fill = torch.full_like(next_token_id, self.pad_idx) # [B]
                next_token_id = torch.where(finished, pad_fill, next_token_id)
                generated = torch.cat([generated, next_token_id.unsqueeze(1)], dim=1) # [B, L_tgt+1]
                finished = finished | (next_token_id == self.eos_idx)

                if finished.all():
                    break

        # Remove BOS, keep everything until PAD (or full length)
        return generated[:, 1:]
    
    @staticmethod
    def partial_load_old_predictor_ckpt(
            model,
            ckpt_path: str,
            device: str,
            old_prefix: str = "predictor.",
            copy_rotary_buffers: bool = False,
            init_new_row_from: str = "PAD",  # "PAD" or "RANDOM"
        ):
        sd_old = torch.load(ckpt_path, map_location=device)

        sd_new = model.state_dict()
        to_load = {}

        # Helper: remap old key -> new key by stripping "predictor."
        def remap_key(k: str) -> str:
            if k.startswith(old_prefix):
                return k[len(old_prefix):] # strip "predictor."
            return k

        # 1) Load shape-matched weights for sn_embedding/cnn_embedding/mrna_encoder
        allowed_prefixes = ("sn_embedding.", "cnn_embedding.", "mrna_encoder.")
        for k_old, v_old in sd_old.items():
            k_new = remap_key(k_old)

            if not k_new.startswith(allowed_prefixes):
                continue
            if k_new not in sd_new:
                continue

            # Rotary buffers: either skip (recommended) or copy if shape matches
            if ("rotary.cos_emb" in k_new) or ("rotary.sin_emb" in k_new):
                if copy_rotary_buffers and v_old.shape == sd_new[k_new].shape:
                    to_load[k_new] = v_old
                continue

            # Normal shape match
            if v_old.shape == sd_new[k_new].shape:
                to_load[k_new] = v_old
            # sn_embedding.weight will mismatch (12 vs 13) -> handled next
            # Anything else mismatched -> skip
            else:
                continue

        missing, unexpected = model.load_state_dict(to_load, strict=False)
        print(f"[partial_load] loaded shape-matched tensors: {len(to_load)}")
        # print("missing:", missing)
        # print("unexpected:", unexpected)

        # 2) Special case: sn_embedding.weight row-wise partial copy
        old_w_key = old_prefix + "sn_embedding.weight"
        if old_w_key in sd_old:
            old_w = sd_old[old_w_key]  # (12, D)
            new_w = model.sn_embedding.weight.data  # (13, D)

            # Copy overlapping rows
            n_copy = min(old_w.shape[0], new_w.shape[0])
            new_w[:n_copy].copy_(old_w[:n_copy])
            print(f"[partial_load] sn_embedding: copied {n_copy} rows")

            # Initialize extra rows (e.g., EOS row)
            if new_w.shape[0] > old_w.shape[0]:
                start = old_w.shape[0]
                if init_new_row_from.upper() == "PAD":
                    pad_idx = getattr(model, "pad_idx", None)
                    if pad_idx is None:
                        raise ValueError("model.pad_idx not set, can't init new row from PAD")
                    new_w[start:].copy_(new_w[pad_idx].unsqueeze(0).repeat(new_w.shape[0] - start, 1)) # initia
                    print(f"[partial_load] initialized new rows {start}..{new_w.shape[0]-1} from PAD row {pad_idx}")
                else:
                    nn.init.normal_(new_w[start:], mean=0.0, std=0.02)
                    print(f"[partial_load] random-initialized new rows {start}..{new_w.shape[0]-1}")

        else:
            print("[partial_load] WARNING: old sn_embedding.weight not found in checkpoint")

        return model

    def seed_everything(self, seed:int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # for cudnn, if reproducibility is needed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def run(self,
            model,
            train_path="",
            valid_path="",
            test_path="",
            ckpt_path=None,
            evaluation=False,
            predict=False,
            finetune=False,
            accumulation_step=1,
            epochs=1,):
        if predict:
            D_test  = load_dataset(test_path, sep='\t')
            ds_test = TargetPredictionDataset(data=D_test,
                                    mrna_max_len=self.mrna_max_len,
                                    mirna_max_len=self.mirna_max_len,
                                    tokenizer=self.tokenizer,)
            test_loader = DataLoader(ds_test, 
                                batch_size=self.batch_size, 
                                shuffle=False)
            if ckpt_path is not None:
                loaded_data = torch.load(ckpt_path, map_location=model.device)
                current_state = model.state_dict()
                encoder_state = {}
                for key, value in loaded_data.items():
                    if key not in current_state:
                        continue
                    # Rotary buffers: either skip (recommended) or copy if shape matches
                    if ("rotary.cos_emb" in key) or ("rotary.sin_emb" in key):
                        if value.shape == current_state[key].shape:
                            encoder_state[key] = value
                        else:
                            print(f"Dropping mismatched key {key}: checkpoint {value.shape}, model {current_state[key].shape}")
                            continue
                    else:
                        encoder_state[key] = value
                model.load_state_dict(encoder_state, strict=False)

                print(f"Loaded checkpoint from {ckpt_path}")
                
            model.to(self.device)

            all_generated_seqs = []
            for batch in test_loader:
                src_tokens = batch["mrna_input_ids"].to(self.device)
                generated = self.greedy_generate(model=model, 
                                                src_tokens=src_tokens, 
                                                max_len=self.mirna_max_len)
                # Decode
                for i in range(generated.size(0)):
                    seq_ids = generated[i].tolist()
                    decoded = self.tokenizer.decode(seq_ids, skip_special_tokens=True)
                    all_generated_seqs.append(decoded)

            D_test["generated_mirna"] = all_generated_seqs
            save_path = os.path.join(os.path.dirname(test_path), "generated_mirna_positive_samples_30_randomized_start_validation_random_samples.csv")
            D_test.to_csv(save_path, index=False)
            print(f"Generated mirna saved to {save_path}")
        else:
            # weights and bias initialization
            wandb.login(key="600e5cca820a9fbb7580d052801b3acfd5c92da2")
            wandb.init(
                project="mirna-Generation",
                name=f"{self.mrna_max_len}-epoch:{epochs}-batchsize:{self.batch_size}-2layerTrans-{self.ff_dim}MLP_hidden", 
                config={
                    "batch_size": self.batch_size * accumulation_step,
                    "epochs": epochs,
                    "learning rate": self.lr,
                },
                tags=["finetune", "best_composite_0.9911_0.9977_epoch11"],
                save_code=False,
                job_type="train"
            )
            self.seed_everything(seed=self.seed)
            # load dataset
            D_train  = load_dataset(train_path, sep=',')
            D_val    = load_dataset(valid_path, sep=',')
            ds_train = TargetPredictionDataset(data=D_train,
                                            mrna_max_len=self.mrna_max_len,
                                            mirna_max_len=self.mirna_max_len,
                                            tokenizer=self.tokenizer,)
            ds_val = TargetPredictionDataset(data=D_val,
                                    mrna_max_len=self.mrna_max_len,
                                    mirna_max_len=self.mirna_max_len,
                                    tokenizer=self.tokenizer,)
            train_loader = DataLoader(ds_train, 
                                batch_size=self.batch_size, 
                                shuffle=True)
            val_loader   = DataLoader(ds_val, 
                                    batch_size=self.batch_size, 
                                    shuffle=False)
            loss_fn   = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            optimizer = AdamW(model.parameters(), lr=self.lr)

            if finetune:
                model = self.partial_load_old_predictor_ckpt(
                            model=model,
                            ckpt_path=ckpt_path,
                            device=model.device,
                            copy_rotary_buffers=False,
                            init_new_row_from="PAD",
                        )
                print(f"Loaded checkpoint from {ckpt_path}")

            model.to(self.device)

            best_token_accuracy = 0
            patience = 10
            counter = 0
            model_checkpoints_dir = os.path.join(
                PROJ_HOME, 
                "checkpoints", 
                "TargetScan", 
                "TwoTowerTransformer", 
                "CNN-tokenized",
                "TargetGeneration",
                str(self.mrna_max_len),
            )
            os.makedirs(model_checkpoints_dir, exist_ok=True)
            for epoch in range(epochs):
                train_loss = self.train_loop(model=model,
                                            dataloader=train_loader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=self.device,
                                            epoch=epoch,
                                            accumulation_step=accumulation_step)
                val_loss, token_accuracy = self.eval_loop(model=model,
                                            dataloader=val_loader,
                                            device=self.device,
                                            epoch=epoch,
                                            loss_fn=loss_fn)
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/token_accuracy": token_accuracy,
                })

                if token_accuracy > best_token_accuracy:
                    best_token_accuracy = token_accuracy
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_dir, f"best_token_accuracy_{best_token_accuracy:.4f}_epoch{epoch}.pth"))
                else:
                    counter +=1
                    if counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}. No improvement for {patience} consecutive epochs.")
                        break
            return train_loss, val_loss, best_token_accuracy

if __name__ == "__main__":
    torch.cuda.empty_cache() # clear crashed cache
    mrna_max_len = 30
    mirna_max_len = 24+2 # plus 2 for BOS and EOS tokens
    from Global_parameters import PROJ_HOME
    train_datapath = os.path.join(PROJ_HOME, "TargetScan_dataset/positive_samples_30_random_samples_train.csv")
    valid_datapath = os.path.join(PROJ_HOME, "TargetScan_dataset/positive_samples_30_random_samples_validation.csv")
    test_datapath  = os.path.join(PROJ_HOME, "TargetScan_dataset/positive_samples_30_randomized_start_validation_random_samples.csv")

    # train target generation model
    model = TargetGenerationModel(mrna_max_len=mrna_max_len,
                                  mirna_max_len=mirna_max_len,
                                  device='cuda:0',
                                  embed_dim=256,
                                  ff_dim=512,
                                  batch_size=32,
                                  vocab_size=13,
                                  n_classes=13,
                                  lr=1e-4,
                                  seed=54,)
    model.run(model=model,
              train_path=train_datapath,
              valid_path=valid_datapath,
              test_path =test_datapath,
              evaluation=False,
              predict=True,
              finetune=False,
              accumulation_step=8,
              epochs=20,
              ckpt_path=os.path.join(PROJ_HOME, "checkpoints/TargetScan/TwoTowerTransformer/CNN-tokenized/TargetGeneration/30/best_token_accuracy_0.9554_epoch19.pth"))
    
    # # train binding, span prediction model
    # model = QuestionAnsweringModel(mrna_max_len=mrna_max_len,
    #                             mirna_max_len=mirna_max_len,
    #                             device='cuda:0',
    #                             epochs=15,
    #                             embed_dim=256,
    #                             ff_dim=512,
    #                             batch_size=32,
    #                             lr=1e-4,
    #                             seed=54,
    #                             predict_span=True,
    #                             predict_binding=True)
    # # total_params = sum(param.numel() for param in model.parameters())
    # # print(f"Total Parameters: {total_params}")
    # model.run(model=model,
    #         train_path=train_datapath,
    #         valid_path=valid_datapath,
    #         test_path =test_datapath,
    #         evaluation=False,
    #         predict=False,
    #         finetune=True,
    #         accumulation_step=1,
    #         ckpt_path=os.path.join(PROJ_HOME, "checkpoints/TargetScan/TwoTowerTransformer/CNN-tokenized/30/best_composite_0.9911_0.9977_epoch11.pth"))
