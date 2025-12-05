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
from Data_pipeline import CharacterTokenizer, QuestionAnswerDataset


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
        x1 = x1.transpose(-1, -2) # (B, L. D)
        x1 = self.act(self.fc1(x1)) # (B, L, D)
        x1 = x1.transpose(-1, -2) # (B, D, L)
        x1 = self.bn1(x1) # (B, D, L)
        x1 = x1.transpose(-1, -2) #(B, L, D)
        x1 = self.fc2(x1) # (B, L, embed_dim)

        x2 = self.conv2(x)
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
                 cross_attn=False):
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
            mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, Q.shape[2], -1) # (batchsize, head_dim, q_len, k_len)
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
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sn_embedding = nn.Embedding(vocab_size, embed_dim)
        self.cnn_embedding = CNNTokenization(embed_dim)
        # self.mirna_positional_embedding = AdditivePositionalEncoding(max_len=mirna_max_len, d_model=embed_dim)
        # self.mrna_positional_embedding = AdditivePositionalEncoding(max_len=mrna_max_len, d_model=embed_dim)
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
        # mirna_embedding = self.mirna_positional_embedding(mirna_embedding) # (batch_size, mirna_len, embed_dim)
        mrna_sn_embedding = self.sn_embedding(mrna)
        # mrna_embedding = self.mrna_positional_embedding(mrna_embedding) # (batch_size, mrna_len, embed_dim)
        
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
            finetune=False,
            accumulation_step=1,
            ckpt_path=""):
        tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                       add_special_tokens=False, 
                                       model_max_length=self.mrna_max_len,
                                       padding_side="right")
        if evaluation:
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
            # D_test_w_pred = D_test.copy()
            # if self.predict_binding:
            #     D_test_w_pred["pred label"] = self.all_binding_preds
            #     D_test_w_pred["pred prob"]  = self.all_binding_probs
            #     res_df = D_test_w_pred
            # if self.predict_span:
            #     D_test_positive = D_test_w_pred.loc[D_test_w_pred["label"] == 1].copy()
            #     D_test_positive["pred start"] = self.all_start_preds
            #     D_test_positive["pred end"]   = self.all_end_preds
            #     # merge D_test_w_pred with D_test_positive
            #     cols = ['pred start', 'pred end']
            #     D_pred_se = D_test_positive[cols]

            #     # 2. 左连接（保留 D_test_w_pred 的所有行）
            #     D_merged = D_test_w_pred.join(D_pred_se, how='left')

            #     # 3. 将缺失的 pred start/end 填成 -1，并转成整数
            #     D_merged[['pred start', 'pred end']] = (
            #         D_merged[['pred start', 'pred end']]
            #         .fillna(-1)
            #         .astype(int)
            #     )
            #     res_df = D_merged
            # pred_df_path = os.path.join(os.path.join(PROJ_HOME, "Performance/TargetScan_test/TwoTowerTransformer"), str(mrna_max_len))
            # os.makedirs(pred_df_path, exist_ok=True)
            # res_df.to_csv(os.path.join(pred_df_path, "negative_with_seed_prediction.csv"), index=False)
            # print(f"Prediction saved to {pred_df_path}")
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

if __name__ == "__main__":
    torch.cuda.empty_cache() # clear crashed cache
    mrna_max_len = 40
    mirna_max_len = 24
    from Global_parameters import PROJ_HOME
    train_datapath = os.path.join(PROJ_HOME, "TargetScan_dataset/TargetScan_train_40_non_canonical.csv")
    valid_datapath = os.path.join(PROJ_HOME, "TargetScan_dataset/TargetScan_validation_40_non_canonical.csv")
    test_datapath  = os.path.join(PROJ_HOME, "Non_canonical_sites/mmu_mimosa_non_canonical_sites.csv")

    model = QuestionAnsweringModel(mrna_max_len=mrna_max_len,
                                   mirna_max_len=mirna_max_len,
                                   device='cuda:2',
                                   epochs=15,
                                   embed_dim=256,
                                   ff_dim=512,
                                   batch_size=32,
                                   lr=1e-4,
                                   seed=54,
                                   predict_span=True,
                                   predict_binding=True)
    # total_params = sum(param.numel() for param in model.parameters())
    # print(f"Total Parameters: {total_params}")
    model.run(model=model,
              train_path=train_datapath,
              valid_path=valid_datapath,
              test_path =test_datapath,
              evaluation=True,
              finetune=False,
              accumulation_step=1,
              ckpt_path=os.path.join(PROJ_HOME, "checkpoints/TargetScan/TwoTowerTransformer/CNN-tokenized/30/best_composite_0.9911_0.9977_epoch11.pth"))
