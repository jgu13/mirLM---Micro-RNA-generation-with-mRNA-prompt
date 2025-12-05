import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import os
import sys
import math
import wandb # uncomment to use wandb
import random
import numpy as np
import pandas as pd
from time import time
from itertools import chain
from wandb.sdk.wandb_settings import Settings # uncomment to use wandb

from utils import load_dataset
from ckpt_util import load_training_state, save_training_state
from Data_pipeline import SpanDataset, BatchStratifiedSampler, TokenClassificationDataset
from Data_pipeline import CharacterTokenizer

from diagonaled_mm_tvm import mask_invalid_locations
from sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
from sliding_chunks import sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv
from sliding_chunks import sliding_window_cross_attention, check_key_mask_rows
from Global_parameters import PROJ_HOME

data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")

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

class LongformerAttention(nn.Module):
    def __init__(self, 
                embed_dim, 
                num_heads, 
                window_size, 
                layer_id,
                max_seq_len=1000,
                dilation=1, 
                autoregressive=False,
                attention_mode="sliding_chunks", 
                dropout=0.2,
                device='cuda',
                cross_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out   = nn.Linear(embed_dim, embed_dim)

        self.rotary = RotaryEmbedding(dim=self.head_dim, 
                                      max_seq_len=max_seq_len)

        self.query_global = nn.Linear(embed_dim, embed_dim)
        self.key_global = nn.Linear(embed_dim, embed_dim)
        self.value_global = nn.Linear(embed_dim, embed_dim)

        self.dropout = dropout

        self.layer_id = layer_id
        self.attention_window = window_size
        self.attention_dilation = dilation
        self.attention_mode = attention_mode
        self.autoregressive = autoregressive
        self.device = device
        self.cross_attn = cross_attn

    def forward(self, 
                x=None, # only used in self-attention 
                query=None, # only used in cross attention when q != k
                key=None, # only used in cross attention when k != q
                value=None, # only used in cross attention when v != k
                attention_mask=None,
                query_attention_mask=None, # only used in cross attention when q != k 
                output_attentions=False,):
        if self.cross_attn:
            if attention_mask is not None:
                assert (attention_mask >= 0).all(), "attention_mask has values less than 0"
                assert (attention_mask > 0).any(), "attention_mask has no values greater than 0"
            if query_attention_mask is not None:
                assert (query_attention_mask >= 0).all(), "query_attention_mask has values less than 0"
                assert (query_attention_mask > 0).any(), "query_attention_mask has no values greater than 0"
            check_key_mask_rows(attention_mask)

            bsz, q_len, _ = query.shape
            _, k_len, _   = key.shape
            _, v_len, _   = value.shape
            
            # Process rest of query with sliding window cross attention
            q = self.query(query)  # (B, L_q, D)
            q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(2, 1).contiguous()  # (B, H, L_q, D)
            
            # Process key and value
            k = self.key(key)  # (B, L_k, D)
            v = self.value(value)  # (B, L_v, D)
            k = k.view(bsz, k_len, self.num_heads, self.head_dim).transpose(2, 1).contiguous()  # (B, H, L_k, D)
            v = v.view(bsz, v_len, self.num_heads, self.head_dim).transpose(2, 1).contiguous()  # (B, H, L_v, D)

            # Apply rotary embeddings
            q = self.rotary(q)
            k = self.rotary(k)
            
            # Handle attention masks
            if query_attention_mask is not None:
                assert attention_mask.shape == (bsz, k_len)
                assert query_attention_mask.shape == (bsz, q_len)
                
                attention_mask = (attention_mask > 0)  # bool
                query_attention_mask = (query_attention_mask > 0)  # bool
                # Create mask for cross attention (B, L_q, L_k)
                mask = attention_mask[:, None, :] & query_attention_mask[:, :, None]
                mask = mask[:, None, :, :].expand(bsz, self.num_heads, q_len, k_len) # (B, H, L_q, L_k)
            else:
                # If no query attention mask, create a default mask that allows all attention
                mask = torch.ones(bsz, self.num_heads, q_len, k_len, device=q.device, dtype=torch.bool)

            # check if the attention mask is all False
            if not mask.any():
                raise ValueError("Attention mask is all False")

            # 3) Sliding window cross attention for rest of query
            z, sliding_attn_weights = sliding_window_cross_attention(
                Q=q, K=k, V=v, 
                w=self.attention_window, 
                mask=mask, 
                norm_by_query=True,
                use_lse=True,)    # (B, H, L_q, D)
            
            # Reshape to final output format
            B, H, Lq, D = z.shape
            z = z.permute(0, 2, 1, 3).contiguous()  # (B, L_q, H, D)
            z = z.view(B, Lq, H*D)  # (B, L_q, embed_dim)
            z = self.out(z)  # (B, L_q, embed_dim)
            self.last_attention = sliding_attn_weights.detach().cpu()
            return (z, sliding_attn_weights)
        else:
            hidden_states = x
            bsz, seq_len, _ = x.shape
            q = self.query(hidden_states) # (B, L, D)
            k = self.key(hidden_states)
            v = self.value(hidden_states)

            q = q.view(bsz, seq_len, self.num_heads, self.head_dim) # (B, L, H, D)
            k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
            v = v.view(bsz, seq_len, self.num_heads, self.head_dim)

            q = self.rotary(q)
            k = self.rotary(k)
            q /= math.sqrt(self.head_dim)
            if attention_mask is not None:
                assert (attention_mask <= 0).all(), "attention_mask has values greater than 0"
                key_padding_mask = attention_mask < 0
                extra_attention_mask = attention_mask > 0
                remove_from_windowed_attention_mask = attention_mask != 0

                num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
                max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
                if max_num_extra_indices_per_batch <= 0:
                    extra_attention_mask = None
                else:
                    # To support the case of variable number of global attention in the rows of a batch,
                    # we use the following three selection masks to select global attention embeddings
                    # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                    # 1) selecting embeddings that correspond to global attention
                    extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                    zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
                                                        device=num_extra_indices_per_batch.device)
                    # mask indicating which values are actually going to be padding
                    selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                    # 2) location of the non-padding values in the selected global attention
                    selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                    # 3) location of the padding values in the selected global attention
                    selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
                assert extra_attention_mask is None, "extra_attention_mask is not None"
            else:
                remove_from_windowed_attention_mask = None
                extra_attention_mask = None
                key_padding_mask = None

            if self.attention_mode == "sliding_chunks":
                attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
            elif self.attention_mode == "sliding_chunks_no_overlap":
                attn_weights = sliding_chunks_no_overlap_matmul_qk(q, k, self.attention_window, padding_value=0)
            else:
                raise False
            mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
            if remove_from_windowed_attention_mask is not None:
                # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
                # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
                remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
                # cast to float/half then replace 1's with -inf
                float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
                repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
                float_mask = float_mask.repeat(1, 1, repeat_size, 1)
                ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
                # diagonal mask with zeros everywhere and -inf inplace of padding
                if self.attention_mode == "sliding_chunks":
                    d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)
                elif self.attention_mode == "sliding_chunks_no_overlap":
                    d_mask = sliding_chunks_no_overlap_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

                attn_weights += d_mask # apply per chunk mask
            assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
            assert attn_weights.size(dim=3) in [self.attention_window * 2 + 1, self.attention_window * 3]

            # the extra attention
            if extra_attention_mask is not None:
                selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
                selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
                # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
                selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
                selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
                # concat to attn_weights
                # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
                attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)

            attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
            if key_padding_mask is not None:
                # softmax sometimes inserts NaN if all positions are masked, replace them with 0
                attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            attn_weights = attn_weights_float.type_as(attn_weights)
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            attn = 0

            if extra_attention_mask is not None:
                selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
                selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
                selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
                # use `matmul` because `einsum` crashes sometimes with fp16
                # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
                attn = torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
                attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

            if self.attention_mode == "sliding_chunks":
                attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
            elif self.attention_mode == "sliding_chunks_no_overlap":
                attn += sliding_chunks_no_overlap_matmul_pv(attn_probs, v, self.attention_window)
            else:
                raise False

            attn = attn.type_as(hidden_states)
            assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
            attn = attn.reshape(bsz, seq_len, self.embed_dim).contiguous()

            # For this case, we'll just recompute the attention for these indices
            # and overwrite the attn tensor. TODO: remove the redundant computation
            if extra_attention_mask is not None:
                selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, self.embed_dim)
                selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[extra_attention_mask_nonzeros[::-1]]

                q = self.query_global(selected_hidden_states)
                k = self.key_global(hidden_states)
                v = self.value_global(hidden_states)
                q /= math.sqrt(self.head_dim)

                q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
                k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
                v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
                attn_weights = torch.bmm(q, k.transpose(1, 2))
                assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]

                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
                attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
                if key_padding_mask is not None:
                    attn_weights = attn_weights.masked_fill(
                        key_padding_mask.unsqueeze(1).unsqueeze(2),
                        -10000.0,
                    )
                attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
                attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
                attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
                selected_attn = torch.bmm(attn_probs, v)
                assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]

                selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
                nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
                attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1).type_as(hidden_states)

            context_layer = attn
            if output_attentions:
                if extra_attention_mask is not None:
                    # With global attention, return global attention probabilities only
                    # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
                    # which is the attention weights from tokens with global attention to all tokens
                    # It doesn't not return local attention
                    # In case of variable number of global attantion in the rows of a batch,
                    # attn_weights are padded with -10000.0 attention scores
                    attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
                else:
                    # without global attention, return local attention probabilities
                    # batch_size x num_heads x sequence_length x window_size
                    # which is the attention weights of every token attending to its neighbours
                    attn_weights = attn_weights.permute(0, 2, 1, 3)
            outputs = (context_layer, attn_weights) if output_attentions else (context_layer,)
            return outputs

class LongformerEncoderLayer(nn.Module):
    def __init__(self, 
                embed_dim, 
                num_heads, 
                layer_id, 
                ff_dim, 
                window_size=20, 
                dilation=1, 
                dropout=0.2,
                max_seq_len=1000,
                device='cuda',
                cross_attn=False):
        super().__init__()
        self.self_attn = LongformerAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            window_size=window_size, 
            dilation=dilation,
            autoregressive=False,
            layer_id=layer_id,
            dropout=dropout, 
            device=device,
            max_seq_len=max_seq_len,
            cross_attn=cross_attn)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, attention_mask=mask)[0]
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class LongformerEncoder(nn.Module):
    def __init__(self, 
                num_layers, 
                embed_dim, 
                num_heads, 
                ff_dim, 
                window_size, 
                dilation=1,
                max_seq_len=10000, 
                dropout=0.2, 
                device='cuda',
                cross_attn=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [LongformerEncoderLayer(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                layer_id=i,
                ff_dim=ff_dim, 
                window_size=window_size, 
                dilation=dilation,
                dropout=dropout, 
                device=device,
                max_seq_len=max_seq_len,
                cross_attn=cross_attn) for i in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        return x

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
        attention = F.dropout(attention, p=0.2)
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
    def __init__(self, embed_dim, num_heads, ff_dim, max_seq_len=10000, dropout=0.2, device='cuda'):
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
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, max_seq_len=10000, dropout=0.2, device='cuda'):
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

class BindingHead(nn.Module):
    """
    Binding head for MIL binding prediction.
    """
    def __init__(self, d_model, output_size, hidden_sizes=[1024, 1024, 1], tau=1.0):
        super().__init__()
        self.seed_scorer = LinearHead(input_size=d_model, hidden_sizes=hidden_sizes, output_size=output_size, dropout=0.2)
        self.tau = tau

    def forward(self, z_mrna, mrna_mask):
        """
        z_mrna: (B, Lm, D)  -- encoder/cross-attn output over mRNA tokens
        mrna_mask: (B, Lm)  -- 1 valid, 0 pad (CLS is valid=1)
        returns: binding_logit (B,), weights (B, Lm)
        """
        s = self.seed_scorer(z_mrna).squeeze(-1)  # (B, Lm)                                      
        s = s.masked_fill(mrna_mask == 0, -1e4)   # never use pads
        # LSE pooling (smooth max)
        x = s / self.tau
        m = x.max(dim=-1, keepdim=True).values
        lse = m + torch.log(torch.clamp(torch.exp(x - m).sum(dim=-1, keepdim=True), min=1e-20))
        binding_logit = (lse * self.tau).squeeze(-1)             # (B,)
        w = torch.softmax(s / self.tau, dim=-1) * (mrna_mask > 0)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
        return binding_logit, w

class CleavageHead(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super(CleavageHead, self).__init__()
        self.transform = LinearHead(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, dropout=dropout)
    def forward(self, x):
        return self.transform(x).squeeze(-1) # (batchsize, mrna_len)

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
    """
    Cross-attention predictor with MIL binding pooling for binding prediction.
    
    Expected input shapes:
    - mrna_ids: (B, Lm) with Lm including the new CLS position at index 0
    - longformer_attn_mask: (B, Lm) with {0,1,2}, position 0 is 2 (global)
    - z_mrna (MIL input): (B, Lm, D)
    - mrna_mask: (B, Lm) with 1 for valid tokens (including CLS), 0 for pad
    
    Outputs:
    - binding_logit: (B,) - MIL binding prediction (replaces old binding_logits)
    - binding_aux: dict with pos_weights and pos_logits for visualization
    - start_logits, end_logits: (B, Lm) or None
    """
    def __init__(self,  
                 mirna_max_len:int,
                 mrna_max_len:int, 
                 vocab_size:int=12, # Fallback if tokenizer not provided (7 special + 5 bases)
                 num_layers:int=2, 
                 embed_dim:int=256, 
                 num_heads:int=2, 
                 window_size:int=20,
                 ff_dim:int=512,
                 hidden_sizes:list[int]=[512, 512],
                 n_classes:int=1, 
                 dropout_rate:float=0.2,
                 device:str='cuda',
                 predict_span=True,
                 predict_binding=False,
                 predict_cleavage=False,
                 use_longformer=False):
        super(CrossAttentionPredictor, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.device = device        
            
        # Create embedding table with correct size
        self.sn_embedding = nn.Embedding(vocab_size, embed_dim)
        self.cnn_embedding = CNNTokenization(embed_dim)
        self.ln_merge = nn.LayerNorm(embed_dim)
        self.mirna_encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim, 
            max_seq_len=mirna_max_len,
            device=device,
            dropout=dropout_rate,
        )
        if use_longformer:
            self.mrna_encoder = LongformerEncoder(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                window_size=window_size,
                max_seq_len=mrna_max_len,
                dropout=dropout_rate,
                device=device,
            )
        else:
            self.mrna_encoder = TransformerEncoder(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim, 
                max_seq_len=mrna_max_len,
                device=device,
                dropout=dropout_rate,
            )
        if use_longformer:
            self.cross_attn_layer = LongformerAttention(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                window_size=window_size, 
                autoregressive=False,
                layer_id=None,
                max_seq_len=mrna_max_len,
                dropout=dropout_rate, 
                device=device,
                cross_attn=True
            )
        else:
            self.cross_attn_layer = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                device=device,
                cross_attn=True, # no positional encoding in cross attention
            )
        self.dropout = nn.Dropout(dropout_rate)
        self.cross_norm = nn.LayerNorm(embed_dim) # normalize over embedding dimension
        self.qa_outputs = nn.Linear(embed_dim, 2) # Linear head instead of one Linear transformation
        self.cleavage_head = CleavageHead(
            input_size=embed_dim, 
            hidden_sizes=hidden_sizes, 
            output_size=1,
            dropout=dropout_rate) # Linear head instead of one Linear transformation
        
        # Add MIL binding head
        self.binding_output = LinearHead(
            input_size=embed_dim, 
            hidden_sizes=hidden_sizes,
            output_size=n_classes,
            dropout=dropout_rate)
        
        self.predict_span = predict_span
        self.predict_binding = predict_binding
        self.predict_cleavage = predict_cleavage
        self.use_longformer = use_longformer
        
        # Initialize the new global token embedding
        self._init_global_token_embedding()
    
    def get_binding_attention_weights(self, z_mrna, mrna_mask):
        """Get MIL binding attention weights for visualization"""
        with torch.no_grad():
            binding_logit, binding_aux = self.binding_head(z_mrna, mrna_mask)
            return binding_aux["pos_weights"], binding_aux["pos_logits"]
    
    def _init_global_token_embedding(self):
        """Initialize the global token embedding row with Xavier initialization"""
        with torch.no_grad():
            # Get the original embedding weights (excluding the new row)
            original_weights = self.sn_embedding.weight[:-1]  # All but the last row
            # Initialize the new row with Xavier initialization
            nn.init.xavier_uniform_(self.sn_embedding.weight[-1:])
    
    def forward(self, mirna, mrna, mrna_mask, mirna_mask):
        mirna_sn_embedding = self.sn_embedding(mirna)
        mrna_sn_embedding = self.sn_embedding(mrna)
        
        # Create Longformer attention mask for mRNA
        if self.use_longformer:
            # Longformer convention: -1=pad, 0=local, 1=global
            # Convert from mrna_mask (0=pad, 1=valid) to Longformer format
            lf_mask = torch.where(
                mrna_mask > 0,
                torch.zeros_like(mrna_mask),  # Set all valid tokens to 0 (local attention)
                torch.full_like(mrna_mask, fill_value=-1)  # Set original 0s (pads) to -1
            )
            # check lf_mask has all values smaller or equal to 0
            assert (lf_mask <= 0).all(), "lf_mask has values greater than 0"
        
        # add N-gram CNN-encoded embedding
        mirna_cnn_embedding = self.cnn_embedding(mirna_sn_embedding.transpose(-1, -2)) # (batch_size, embed_dim, mirna_len)
        mrna_cnn_embedding  = self.cnn_embedding(mrna_sn_embedding.transpose(-1, -2))  # (batch_size, embed_dim, mrna_len)
        mirna_embedding     = mirna_sn_embedding + mirna_cnn_embedding # (batch_size, mirna_len, embed_dim)
        mrna_embedding      = mrna_sn_embedding + mrna_cnn_embedding # (batch_size, mrna_len, embed_dim)
        mirna_embedding = self.mirna_encoder(mirna_embedding, mask=mirna_mask)  # (batch_size, mirna_len, embed_dim)

        if self.use_longformer:
            # use lf_mask for mrna_encoder
            mrna_embedding = self.mrna_encoder(mrna_embedding, mask=lf_mask) # (batch_size, mrna_len, embed_dim) with global attention
        else:
            mrna_embedding = self.mrna_encoder(mrna_embedding, mask=mrna_mask) # (batch_size, mrna_len, embed_dim)
        
        if self.use_longformer:
            output = self.cross_attn_layer(
                query=mrna_embedding,
                key=mirna_embedding,
                value=mirna_embedding,
                attention_mask=mirna_mask,
                query_attention_mask=mrna_mask,  # Use mask: 1=unmasked, 0=masked for cross attention
            )[0]
            self.cross_attn_output = output
            z = output
        else: 
            z = self.cross_attn_layer(query=mrna_embedding, 
                                    key=mirna_embedding,
                                    value=mirna_embedding,
                                    mask=mirna_mask) # pass key-mask
            self.cross_attn_output = z
        z_res = self.dropout(z) + mrna_embedding # residual connection
        z_norm = self.cross_norm(z_res)
        z_norm = z_norm.masked_fill(mrna_mask.unsqueeze(-1)==0, 0) # (batch_size, mrna_len, embed_dim)
        
        # mean-pooled binding head on mRNA positions
        if self.predict_binding:
            valid_counts = mrna_mask.sum(dim=1, keepdim=True) # (batch_size)
            # avg pooling over seq_len
            z_norm_mean = z_norm.sum(dim=1) / (valid_counts + 1e-8) # (batch_size, embed_dim)
            # predict binding
            binding_logit = self.binding_output(z_norm_mean) # (batchsize, 1)
            binding_weights = None
        else:
            binding_logit, binding_weights = None, None

        if self.predict_span:
            # predict start and end
            span_logits = self.qa_outputs(z_norm) # (batchsize, mrna_len, 2)
            start_logits, end_logits = span_logits[...,0], span_logits[...,1] # (batchsize, mrna_len)
        else:
            start_logits, end_logits = None, None
        
        # Predict cleavage site using cross-attention hidden states
        cleavage_logits = self.cleavage_head(z_norm).squeeze(-1) # (batchsize, mrna_len)
        
        # Return MIL outputs along with existing outputs
        return binding_logit, binding_weights, start_logits, end_logits, cleavage_logits

def create_dataset(train_path, valid_path, tokenizer, mRNA_max_len):
    D_train = load_dataset(train_path, sep=',', parse_seeds=True)
    D_val = load_dataset(valid_path, sep=',', parse_seeds=True)
    
    # # Convert the 'seeds' column from string to list of tuples
    # D_train['seeds'] = D_train['seeds']
    # D_val['seeds'] = D_val['seeds']

    ds_train = TokenClassificationDataset(
        df=D_train,
        tokenizer=tokenizer,
        mrna_max_len=mRNA_max_len,
        mirna_max_len=mirna_max_len
    )
    ds_val = TokenClassificationDataset(
        df=D_val,
        tokenizer=tokenizer,
        mrna_max_len=mRNA_max_len,
        mirna_max_len=mirna_max_len
    )
    return ds_train, ds_val
    
class DTEA(nn.Module):
    def __init__(self,
                mrna_max_len,
                mirna_max_len,
                device: str=None,
                epochs:int=100,
                embed_dim=256,
                num_heads=2,
                num_layers=2,
                ff_dim:int=512,
                batch_size:int=32,
                lr=0.001,
                seed=42,
                predict_span=True,
                predict_binding=False,
                predict_cleavage=False,
                use_cross_attn=True,
                use_longformer=False):
        super(DTEA, self).__init__()
        self.mrna_max_len = mrna_max_len
        self.mirna_max_len = mirna_max_len
        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            def pick_device():
                if torch.cuda.is_available():
                    # With Slurm, CUDA_VISIBLE_DEVICES is already set, so "cuda" == the first allowed GPU
                    return torch.device("cuda")
                if torch.backends.mps.is_available():
                    return torch.device("mps")
                else:
                    return torch.device("cpu")
            self.device = pick_device()
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.predict_binding = predict_binding
        self.predict_span = predict_span
        self.predict_cleavage = predict_cleavage
        self.attn_cache = []
        if use_cross_attn:
            self.predictor = CrossAttentionPredictor(mrna_max_len=mrna_max_len,
                                                    mirna_max_len=mirna_max_len,
                                                    embed_dim = embed_dim,
                                                    num_heads=num_heads,
                                                    num_layers=num_layers,
                                                    ff_dim = ff_dim,
                                                    hidden_sizes = [ff_dim, ff_dim],
                                                    device=self.device,
                                                    predict_span=predict_span,
                                                    predict_binding=predict_binding,
                                                    predict_cleavage=predict_cleavage,
                                                    use_longformer=use_longformer)
    
    def forward(self, 
                mirna, 
                mrna, 
                mrna_mask, 
                mirna_mask):
        return self.predictor(mirna=mirna, 
                              mrna=mrna, 
                              mrna_mask=mrna_mask,
                              mirna_mask=mirna_mask)
    
    @staticmethod
    def compute_span_metrics(start_preds, end_preds, start_labels, end_labels):
        """
        Computes exact match and F1 score.
        Input tensors are all [B], representing the start/end of each sample.
        """
        exact_matches = 0
        f1_total = 0.0
        n = len(start_preds)

        for i in range(n):
            pred_start = int(start_preds[i])
            pred_end   = int(end_preds[i])
            true_start = int(start_labels[i])
            true_end = int(end_labels[i])

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

    def bce_with_soft_gaussian_loss(self, logits:torch.Tensor, soft_targets:torch.Tensor, pos_boost:float=1.5):
        '''
        logits: (batch_size, mrna_len)
        soft_targets: (batch_size, mrna_len)
        pos_boost: float, boost the loss at positive locations by this factor
        '''
        base = nn.BCEWithLogitsLoss(reduction='none')(logits, soft_targets) # (batch_size, mrna_len)
        if pos_boost != 1.0:
            # if pos_boost is greater than 1.0, boost the loss at positive locations
            # weights are multiplied to the base loss and normalized by the sum of weights
            # this way the loss is unaffected by the gaussian distributions between batches
            weights = 1.0 + (pos_boost - 1.0) * soft_targets # (batch_size, mrna_len)
            weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8) # (batch_size, 1)
            loss = (weights * base).sum(dim=1, keepdim=True) / weights_sum # (batch_size, 1)
            return loss.squeeze(-1).mean()  # (batch_size,) -> scalar
        else:
            loss = base.mean().squeeze(-1) # (batch_size, )
            return loss.mean()
        
    def train_loop(self, 
              model, 
              dataloader, 
              loss_fn,
              optimizer, 
              device,
              epoch,
              scheduler=None,
              accumulation_step=1,
              alpha1=1,
              alpha2=0.75,
              trainable_params=None):
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
            binding_logit, binding_weights, start_logits, end_logits, cleavage_logits = outputs 
               
            if self.predict_span:
                # mask padded output in start and end logits
                start_logits = start_logits.masked_fill(mrna_mask==0, float("-inf"))
                end_logits   = end_logits.masked_fill(mrna_mask==0, float("-inf"))
                start_positions = batch["start_positions"]
                end_positions   = batch["end_positions"]

            span_loss      = torch.tensor(0.0, device=device)
            binding_loss   = torch.tensor(0.0, device=device)
            cleavage_loss  = torch.tensor(0.0, device=device)
            loss           = torch.tensor(0.0, device=device)
            
            # Cleavage site prediction loss (if cleavage positions are provided)
            if self.predict_cleavage:
                cleavage_targets = batch["cleavage_soft_targets"]  # (batchsize,mrna_len)
                cleavage_loss = self.bce_with_soft_gaussian_loss(logits=cleavage_logits, soft_targets=cleavage_targets, pos_boost=1.5)
                loss          += cleavage_loss
                
            if self.predict_binding:
                binding_targets = batch["target"]
                # Use MIL binding predictions instead 
                binding_loss_fn = nn.BCEWithLogitsLoss()
                binding_loss    = binding_loss_fn(binding_logit.squeeze(-1), binding_targets.view(-1).float())
                pos_mask        = binding_targets.view(-1).bool()
                if self.predict_span and pos_mask.any():
                    # only loss of positive pairs are counted
                    loss_start = loss_fn(start_logits[pos_mask,], start_positions[pos_mask]) # CrossEntropyLoss expects [B, L], labels as [B]
                    loss_end   = loss_fn(end_logits[pos_mask,], end_positions[pos_mask])
                    span_loss  = 0.5 * (loss_start + loss_end)
                    loss       += alpha1 * binding_loss + alpha2 * span_loss  # binding_loss is now MIL binding loss
                else:
                    loss       += binding_loss  # binding_loss is now MIL binding loss
            elif self.predict_span:
                # assume all mirna-mrna pairs are positive
                # CrossEntropyLoss expects [B, L], labels as [B]
                loss_start = loss_fn(start_logits, start_positions)
                loss_end   = loss_fn(end_logits, end_positions)
                span_loss  = 0.5 * (loss_start + loss_end)
                loss       += span_loss
            
            # If no other losses are computed, ensure we have a valid loss
            if loss.item() == 0.0 and not (self.predict_binding or self.predict_span):
                # Only cleavage prediction enabled, loss is already computed above
                pass 

            loss = loss / accumulation_step
            loss.backward()
            bs = batch["mrna_input_ids"].size(0)
            trainable_params = model.parameters() if trainable_params is None else trainable_params
            if accumulation_step != 1:
                loss_list.append(loss.item())
                if (batch_idx + 1) % accumulation_step == 0:
                    clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    scheduler.step() if scheduler is not None else None
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
                clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step() if scheduler is not None else None
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
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step() if scheduler is not None else None
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def eval_loop(self, 
                  model, 
                  dataloader, 
                  device,
                  alpha1=1,
                  alpha2=0.75,
                  evaluation=False,
                  W_list=[3,5]):
        model.eval()
        total_loss = 0.0 
        all_start_preds, all_end_preds        = [], []
        all_binding_preds, all_binding_labels = [], []
        all_binding_probs                     = []
        all_start_labels, all_end_labels      = [], []
        all_cleavage_preds, all_cleavage_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                mrna_mask  = batch["mrna_attention_mask"]
                mirna_mask = batch["mirna_attention_mask"]
                outputs    = model(
                    mirna=batch["mirna_input_ids"],
                    mrna=batch["mrna_input_ids"],
                    mrna_mask=mrna_mask,
                    mirna_mask=mirna_mask,
                )
                binding_logit, binding_weights, start_logits, end_logits, cleavage_logits = outputs

                if self.predict_span:
                    # mask padded mrna tokens
                    start_logits = start_logits.masked_fill(mrna_mask==0, float("-inf"))
                    end_logits   = end_logits.masked_fill(mrna_mask==0, float("-inf"))
                    start_positions = batch["start_positions"] # (batchsize, )
                    end_positions   = batch["end_positions"] # (batchsize, )

                # Compute loss
                loss_fn = nn.CrossEntropyLoss()
                loss    = 0.0 

                if self.predict_binding: 
                    # Compute binding loss using MIL binding predictions
                    binding_targets = batch["target"] # (batchsize, )
                    binding_loss_fn = nn.BCEWithLogitsLoss()
                    binding_loss    = binding_loss_fn(binding_logit.squeeze(-1), binding_targets.view(-1).float())
                    loss += binding_loss
                    # binding metric using MIL binding predictions
                    binding_probs = torch.sigmoid(binding_logit)
                    binding_preds = (binding_probs > 0.5).to(torch.int)
                    all_binding_preds.extend(binding_preds.cpu())
                    all_binding_labels.extend(binding_targets.view(-1).cpu())
                else:
                    binding_loss = None

                # Cleavage site prediction loss and metrics
                cleavage_loss = torch.tensor(0.0, device=device)
                if self.predict_cleavage: 
                    cleavage_sites = batch["cleavage_sites"] # (batchsize, )
                    cleavage_targets = batch["cleavage_soft_targets"]  # (batchsize, mrna_len) - soft targets
                    cleavage_loss = self.bce_with_soft_gaussian_loss(
                        logits=cleavage_logits, soft_targets=cleavage_targets, pos_boost=1.5)
                    loss += cleavage_loss
                    
                    # Cleavage predictions
                    cleavage_preds = torch.argmax(cleavage_logits, dim=-1)
                    all_cleavage_preds.extend(cleavage_preds.cpu())
                    all_cleavage_labels.extend(cleavage_sites.cpu())

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
                
                # If no other losses are computed, ensure we have a valid loss
                if loss == 0.0 and not (self.predict_binding or self.predict_span):
                    # Only cleavage prediction enabled, loss is already computed above
                    pass
                
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

        # MIL binding accuracy
        if self.predict_binding:
            all_binding_probs  = torch.tensor(all_binding_probs, dtype=torch.float)
            all_binding_labels = torch.tensor(all_binding_labels, dtype=torch.long)
            all_binding_preds  = torch.tensor(all_binding_preds, dtype=torch.long)
            acc_binding        = (all_binding_preds == all_binding_labels).float().mean().item()
        else:
            acc_binding        = 0.0

        # Cleavage site accuracy
        if len(all_cleavage_preds) > 0:
            all_cleavage_preds  = torch.tensor(all_cleavage_preds, dtype=torch.long)
            all_cleavage_labels = torch.tensor(all_cleavage_labels, dtype=torch.long)
            acc_cleavage        = (all_cleavage_preds == all_cleavage_labels).float().mean().item()
            if W_list is not None:
                hit_at_w_list = {}
                for w in W_list:
                    hit_at_w = ((all_cleavage_preds - all_cleavage_labels).abs() <= w).float().mean().item()
                    hit_at_w_list[f"Hit at {w}"] = hit_at_w
            else:
                hit_at_w_list   = None
        else:
            acc_cleavage        = 0.0
            hit_at_w_list       = None

        avg_loss = total_loss / len(dataloader)
        
        if evaluation:
            if self.predict_binding:
                self.all_binding_probs = all_binding_probs.numpy()
                self.all_binding_preds = all_binding_preds.numpy()
            if self.predict_span:
                self.all_start_preds = all_start_preds.numpy()
                self.all_end_preds = all_end_preds.numpy()
            if len(all_cleavage_preds) > 0:
                self.all_cleavage_preds = all_cleavage_preds.numpy()
        
        print(f"Start Acc:   {acc_start*100}%\n"
              f"End Acc:     {acc_end*100}%\n"
              f"Span Exact Match: {exact_match*100}%\n"
              f"F1 Score:    {f1}\n"
              f"Binding Acc: {acc_binding*100}%\n"
              f"Cleavage Exact Match: {acc_cleavage*100}%")
        if hit_at_w_list is not None:
            for w, hit_at_w in hit_at_w_list.items():
                print(f"{w}: {hit_at_w*100}%")

        return avg_loss, acc_binding, acc_start, acc_end, exact_match, f1, acc_cleavage, hit_at_w_list     

    @staticmethod 
    def seed_everything(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    def train_on_BIO(self, model, dataloader, optimizer, device, epoch, accumulation_step=1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        loss_list = []
        binding_loss_fn = nn.BCEWithLogitsLoss()
        token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        for batch_idx, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)

            binding_logits, token_logits, attn_weights = model(
                mirna=batch["mirna_input_ids"], 
                mrna=batch["mrna_input_ids"],
                mrna_mask=batch["mrna_attention_mask"],
                mirna_mask=batch["mirna_attention_mask"]
            )

            binding_loss = binding_loss_fn(binding_logits.squeeze(-1), batch["binding_labels"].view(-1).float())
            token_loss = token_loss_fn(token_logits.view(-1, 3), batch["labels"].view(-1)) # num_labels = 3 (B, I, O)
            # reg_loss = kl_diag_seed_loss(
            #             attn=attn_weights,  # Use actual attention weights
            #             seed_q_start=batch["seed_start"],
            #             seed_q_end=batch["seed_end"],
            #             q_mask=batch["mrna_attention_mask"],
            #             k_mask=batch["mirna_attention_mask"],
            #             y_pos=batch["target"],
            #             sigma=1.0,  # Set sigma value
            #             k_seed_start=1)
            reg_loss = torch.tensor(0.0, device=binding_loss.device)  # Disable regularization for now

            loss = binding_loss + token_loss + reg_loss
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
        if (batch_idx + 1) % accumulation_step != 0:
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def eval_on_BIO(self, model, dataloader, device):
        model.eval()
        total_loss = 0.0
        all_token_preds = []
        all_token_labels = []
        all_binding_preds = []
        all_binding_labels = []

        binding_loss_fn = nn.BCEWithLogitsLoss()
        token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        with torch.no_grad():
            for batch in dataloader:
                for k in batch:
                    batch[k] = batch[k].to(device)

                binding_logits, token_logits = model(
                    mirna=batch["mirna_input_ids"],
                    mrna=batch["mrna_input_ids"],
                    mrna_mask=batch["mrna_attention_mask"],
                    mirna_mask=batch["mirna_attention_mask"]
                )

                binding_loss = binding_loss_fn(binding_logits.squeeze(-1), batch["binding_labels"].view(-1).float())
                token_loss = token_loss_fn(token_logits.view(-1, 3), batch["labels"].view(-1))

                loss = binding_loss + token_loss
                total_loss += loss.item()

                binding_preds = (torch.sigmoid(binding_logits) > 0.5).long()
                all_binding_preds.extend(binding_preds.cpu().numpy().flatten())
                all_binding_labels.extend(batch["binding_labels"].cpu().numpy().flatten())

                token_preds = torch.argmax(token_logits, dim=-1)
                all_token_preds.extend(token_preds.cpu().numpy().flatten())
                all_token_labels.extend(batch["labels"].cpu().numpy().flatten())

        avg_loss = total_loss / len(dataloader)
        print("dataset size: ",len(dataloader))
        binding_accuracy = (np.array(all_binding_preds) == np.array(all_binding_labels)).mean()
        
        # Filter out the ignored index (-100) for token accuracy calculation
        all_token_labels = np.array(all_token_labels)
        all_token_preds = np.array(all_token_preds)
        valid_indices = all_token_labels != -100
        token_accuracy = (all_token_preds[valid_indices] == all_token_labels[valid_indices]).mean()

        print(f"Validation Loss: {avg_loss*100}%\n"
              f"Binding Accuracy: {binding_accuracy*100}\n"
              f"Token Accuracy: {token_accuracy*100}\n")

        return avg_loss, binding_accuracy, token_accuracy

    def run(self, 
            model,
            train_path="",
            valid_path="",
            test_path="",
            evaluation=False,
            accumulation_step=1,
            ckpt_path="",
            training_mode="SPAN"):
        """
        model: nn.Module
            The model to train or evaluate.
        train_path: str
            The path to the training data.
        valid_path: str
            The path to the validation data.
        test_path: str
            The path to the test data.
        evaluation: bool
            If True, evaluate the model on the test data.
        accumulation_step: int
            The number of steps to accumulate the gradients.
        ckpt_path: str
            The path of the checkpoint file.
        training_mode: str
            "BIO": BIO tagging
            "QA": Question Answering
        """
        tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                       model_max_length=self.mrna_max_len,
                                       padding_side="right")      
        if training_mode == "BIO":
            ds_train, ds_val = create_dataset(train_path, valid_path, tokenizer, mRNA_max_len=self.mrna_max_len)
            train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            optimizer = AdamW(model.parameters(), lr=self.lr)

            # wandb.login(key="600e5cca820a9fbb7580d052801b3acfd5c92da2")
            settings = Settings(
                start_method="thread",   # avoid fork issues on HPC
                init_timeout=180,        # give it more time
                console="simple"         # quieter logging
            )
            run = wandb.init(
                project="mirna-token-classification",
                name=f"BIO-tagging-len:{self.mrna_max_len}-epoch:{self.epochs}", 
                config={
                    "batch_size": self.batch_size * accumulation_step,
                    "epochs": self.epochs,
                    "learning_rate": self.lr,
                },
                tags=["BIO-tagging", "sliding-window-local-attn"],
                save_code=True,
                job_type="train"
            )

            model.to(self.device)
            start = time()
            count = 0
            patience = 10
            best_accuracy = 0
            model_checkpoints_dir = os.path.join(
                PROJ_HOME, 
                "checkpoints", 
                "TargetScan", 
                "TokenClassification", 
                str(self.mrna_max_len),
            )
            os.makedirs(model_checkpoints_dir, exist_ok=True)
            for epoch in range(self.epochs):
                train_loss = self.train_on_BIO(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    device=self.device,
                    epoch=epoch,
                    accumulation_step=accumulation_step,
                )
                eval_loss, binding_accuracy, token_accuracy = self.eval_on_BIO(
                    model=model,
                    dataloader=val_loader,
                    device=self.device,
                )
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "eval/loss": eval_loss,
                    "eval/binding accuracy": binding_accuracy,
                    "eval/BIO accuracy": token_accuracy
                }, step=epoch)

                if token_accuracy > best_accuracy:
                    best_accuracy = token_accuracy
                    ckpt_name = f"best_accuracy_{best_accuracy:.4f}_epoch{epoch}.pth"
                    ckpt_path = os.path.join(model_checkpoints_dir, ckpt_name)
                    torch.save(model.state_dict(), ckpt_path)

                    model_art = wandb.Artifact(
                        name="BIO-tagging-model",
                        type="model",
                        metadata={
                            "epoch": epoch,
                            "accuracy": best_accuracy
                        }
                    )
                    model_art.add_file(ckpt_path)
                    try:
                        run.log_artifact(model_art, aliases=["best-bio"])
                    except Exception as e:
                        print(f"[W&B] artifact log failed at epoch {epoch}: {e}")
                else:
                    count += 1
                    if count >= patience:
                        print("Max patience reached with no improvement. Early stopping.")
                        break
                                    # ETA printout
                elapsed = time() - start
                remaining = elapsed / (epoch + 1) * (self.epochs - epoch - 1) / 3600
                print(f"Still remain: {remaining:.2f} hrs.")

        elif training_mode == "SPAN":
            if evaluation:
                D_test = load_dataset(test_path, sep=',')
                ds_test = SpanDataset(data=D_test,
                                    mrna_max_len=self.mrna_max_len,
                                    mirna_max_len=self.mirna_max_len,
                                    tokenizer=tokenizer,
                                    seed_start_col="seed start" if "seed start" in D_test.columns else None,
                                    seed_end_col="seed end" if "seed end" in D_test.columns else None,
                                    cleavage_site_col="cleave_site" if "cleave_site" in D_test.columns else None)
                test_loader = DataLoader(ds_test,
                                        batch_size=self.batch_size, 
                                        shuffle=False)
                ckpt_path = os.path.join(PROJ_HOME, 
                                "checkpoints", 
                                "TargetScan/TwoTowerTransformer",
                                "longformer",
                                str(model.mrna_max_len), 
                                ckpt_name)
                loaded_data = torch.load(ckpt_path, map_location=model.device)
                model.load_state_dict(loaded_data)
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

                    # 2. Left join (keep all rows of D_test_w_pred)
                    D_merged = D_test_w_pred.join(D_pred_se, how='left')

                # fll missing start/end positions with -1
                D_merged[['pred start', 'pred end']] = (
                    D_merged[['pred start', 'pred end']]
                    .fillna(-1)
                    .astype(int)
                )
                res_df = D_merged
                pred_df_path = os.path.join(os.path.join(PROJ_HOME, "Performance/TargetScan_test/TwoTowerTransformer"), str(self.mrna_max_len))
                os.makedirs(pred_df_path, exist_ok=True)
                res_df.to_csv(os.path.join(pred_df_path, "seed_prediction.csv"), index=False)
                print(f"Prediction saved to {pred_df_path}")
            else:
                # weights and bias initialization
                run = wandb.init(
                    project="mirna-cleavage-prediction",
                    name=f"CNN_len:{self.mrna_max_len}-epoch:{self.epochs}-MLP_hidden:{self.ff_dim}", 
                    config={
                        "batch_size": self.batch_size * accumulation_step,
                        "epochs": self.epochs,
                        "learning rate": self.lr,
                    },
                    tags=["cleavage-prediction", "continue-training", "best_composite_0.9042_0.9871_epoch12"],
                    save_code=True,
                    job_type="train",
                )
                self.seed_everything(seed=self.seed)
                # load dataset
                D_train  = load_dataset(train_path, sep='\t' if train_path.endswith('.tsv') else ',')
                D_val    = load_dataset(valid_path, sep='\t' if valid_path.endswith('.tsv') else ',')
                
                # Add default label column if it doesn't exist (for degradome data)
                if "label" not in D_train.columns:
                    D_train["label"] = 1  # All degradome data is positive
                    
                ds_train = SpanDataset(data=D_train,
                                                mrna_max_len=self.mrna_max_len,
                                                mirna_max_len=self.mirna_max_len,
                                                tokenizer=tokenizer,
                                                seed_start_col="seed start" if "seed start" in D_train.columns else None,
                                                seed_end_col="seed end" if "seed end" in D_train.columns else None,
                                                cleavage_site_col="cleave_site" if "cleave_site" in D_train.columns else None)
                ds_val = SpanDataset(data=D_val,
                                            mrna_max_len=self.mrna_max_len,
                                            mirna_max_len=self.mirna_max_len,
                                            tokenizer=tokenizer, 
                                            seed_start_col="seed start" if "seed start" in D_val.columns else None,
                                            seed_end_col="seed end" if "seed end" in D_val.columns else None,
                                            cleavage_site_col="cleave_site" if "cleave_site" in D_val.columns else None)
                # train_sampler = BatchStratifiedSampler(labels = [example["target"].item() for example in ds_train],
                                                # batch_size = self.batch_size)
                train_loader = DataLoader(ds_train, 
                                    batch_size=self.batch_size,
                                    # batch_sampler=train_sampler,
                                    shuffle=False)
                val_loader   = DataLoader(ds_val, 
                                        batch_size=self.batch_size,
                                        shuffle=False)
                loss_fn   = nn.CrossEntropyLoss()
                model.to(self.device)
                
                # if not self.predict_span:
                #     # freeze update of params in the span prediction head
                #     for p in model.predictor.qa_outputs.parameters():
                #         p.requires_grad = False
                # elif not self.predict_binding:
                #     # freeze update of params in the binding prediction head
                #     for p in model.predictor.binding_head.parameters():
                #         p.requires_grad = False
                # elif not self.predict_cleavage:
                #     # freeze update of params in the cleavage prediction head
                #     for p in model.predictor.cleavage_head.parameters():
                #         p.requires_grad = False

                optimizer = AdamW(model.parameters(), lr=self.lr)
                
                total_steps   = math.ceil(len(train_loader) * self.epochs / accumulation_step)
                warmup_steps  = int(0.05 * total_steps)  # 5% warmup (3–5% is typical)
                eta_min       = 3e-5                     # final floor  

                warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
                cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
                scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

                start    = time()
                count    = 0
                patience = 10
                best_binding_acc = 0
                best_exact_match = 0
                best_f1_score    = 0
                best_composite_metric = 0
                best_acc_and_hit = 0
                model_checkpoints_dir = os.path.join(
                    PROJ_HOME, 
                    "checkpoints", 
                    "TargetScan", 
                    "TwoTowerTransformer", 
                    "Longformer",
                    str(self.mrna_max_len),
                    "predict_cleavage",
                    "continue_training",
                    "UTR_windows_500",
                )
                os.makedirs(model_checkpoints_dir, exist_ok=True)

                if ckpt_path != "":
                    # resumed_data = load_training_state(
                    #     ckpt_path=ckpt_path, 
                    #     model=model, optimizer=None, scheduler=None, # do not load optimizer and scheduler
                    #     map_location=model.device)
                    model.load_state_dict(torch.load(ckpt_path, map_location=model.device), strict=False)
                    print(f"Loaded checkpoint from {ckpt_path}", flush=True)

                for epoch in range(self.epochs):
                    if epoch == 0:
                        # evaluate once on the validation set before training
                        eval_loss, acc_binding, acc_start, acc_end, exact_match, f1, acc_cleavage, hit_at_w_list = self.eval_loop(
                            model=model,
                            dataloader=val_loader,
                            device=self.device,
                            W_list=[3,5],
                        )
                        # log the evaluation results
                        log_dict = {
                            "eval/loss": eval_loss,
                            "eval/binding accuracy": acc_binding,
                            "eval/start accuracy": acc_start,
                            "eval/end accuracy": acc_end,
                            "eval/exact match": exact_match,
                            "eval/F1 score": f1,
                            "eval/cleavage accuracy": acc_cleavage,
                        }
                        if hit_at_w_list is not None:
                            log_dict.update({f"eval/{w}": hit_at_w for w, hit_at_w in hit_at_w_list.items()})
                        wandb.log(log_dict, step=epoch)
                    else:
                        # TRAINING
                        train_loss = self.train_loop(
                            model=model,
                            dataloader=train_loader,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            device=self.device,
                            epoch=epoch,
                            accumulation_step=accumulation_step,
                            trainable_params=None,
                        )

                        # EVALUATION
                        eval_loss, acc_binding, acc_start, acc_end, exact_match, f1, acc_cleavage, hit_at_w_list = self.eval_loop(
                            model=model,
                            dataloader=val_loader,
                            device=self.device,
                            W_list=[3,5],
                        )

                        # SAFE METRIC LOGGING
                        try:
                            log_dict = {
                                "epoch": epoch,
                                "train/loss": train_loss,
                                "eval/loss": eval_loss,
                                "eval/binding accuracy": acc_binding,
                                "eval/start accuracy": acc_start,
                                "eval/end accuracy": acc_end,
                                "eval/exact match": exact_match,
                                "eval/F1 score": f1,
                                "eval/cleavage accuracy": acc_cleavage,
                            }
                            if hit_at_w_list is not None:
                                log_dict.update({f"eval/{w}": hit_at_w for w, hit_at_w in hit_at_w_list.items()})
                            wandb.log(log_dict, step=epoch)
                        except Exception as e:
                            print(f"[W&B] log failed at epoch {epoch}: {e}")

                        # CHECK FOR IMPROVEMENT
                        if self.predict_binding and self.predict_span:
                            composite = f1 + acc_binding
                            improved = composite > best_composite_metric
                        elif self.predict_binding:
                            improved = acc_binding >= best_binding_acc
                        elif self.predict_span:
                            improved = exact_match >= best_exact_match
                        elif self.predict_cleavage:
                            acc_and_hit = acc_cleavage + sum([hit_at_w for w, hit_at_w in hit_at_w_list.items()])
                            improved = acc_and_hit > best_acc_and_hit

                        if improved:
                            # update bests & reset patience
                            best_composite_metric = composite if self.predict_binding and self.predict_span else best_composite_metric
                            best_binding_acc      = acc_binding   if self.predict_binding else best_binding_acc
                            best_f1_score         = f1            if self.predict_span    else best_f1_score
                            best_exact_match      = exact_match   if self.predict_span    else best_exact_match
                            best_acc_and_hit      = acc_cleavage + sum([hit_at_w for w, hit_at_w in hit_at_w_list.items()]) if self.predict_cleavage else best_acc_and_hit
                            count = 0

                            # save checkpoint
                            ckpt_name = (
                                f"best_composite_0.9042_0.9871_epoch12_best_composite_{best_f1_score:.4f}_{best_binding_acc:.4f}_epoch{epoch}.pth"
                                if (self.predict_binding and self.predict_span)
                                else f"best_composite_0.9042_0.9871_epoch12_best_binding_acc_{best_binding_acc:.4f}_epoch{epoch}.pth"
                                if self.predict_binding
                                else f"best_composite_0.9042_0.9871_epoch12_best_exact_match_{best_exact_match:.4f}_epoch{epoch}.pth"
                                if self.predict_span
                                else f"continue_training_best_composite_0.9042_0.9871_epoch12_best_acc_and_hit_{best_acc_and_hit:.4f}_epoch{epoch}.pth"
                            )
                            ckpt_path = os.path.join(model_checkpoints_dir, ckpt_name)

                            try:
                                torch.save(model.state_dict(), ckpt_path)
                                print(f"[CKPT] saved to {ckpt_path}", flush=True)
                            except Exception as e:
                                print(f"[CKPT][ERROR] failed to save {ckpt_path}: {e}", file=sys.stderr, flush=True)
                        
                            # create and log artifact with alias
                            model_art = wandb.Artifact(
                                name=(
                                    "best_composite_0.9042_0.9871_epoch12_binding-span-model" if (self.predict_binding and self.predict_span)
                                    else "best_composite_0.9042_0.9871_epoch12_mirna-binding-model" if self.predict_binding
                                    else "best_composite_0.9042_0.9871_epoch12_mirna-span-model"
                                ),
                                type="model",
                                metadata={
                                    "epoch": epoch,
                                    **({"best_composite_0.9042_0.9871_epoch12_f1+acc_binding": composite} if (self.predict_binding and self.predict_span) else {}),
                                    **({"best_composite_0.9042_0.9871_epoch12_binding_acc": acc_binding} if self.predict_binding and not self.predict_span else {}),
                                    **({"best_composite_0.9042_0.9871_epoch12_exact_match": exact_match} if self.predict_span and not self.predict_binding else {}),
                                }
                            )
                            model_art.add_file(ckpt_path)

                            try:
                                run.log_artifact(model_art, aliases=["predict_cleavage_best_composite_0.9042_0.9871_epoch12_500nt"])
                            except Exception as e:
                                print(f"[W&B] artifact log failed at epoch {epoch}: {e}")

                        else:
                            count += 1
                            if count >= patience:
                                print("Max patience reached with no improvement. Early stopping.")
                                break

                        # ETA printout
                        elapsed = time() - start
                        remaining = elapsed / (epoch + 1) * (self.epochs - epoch - 1) / 3600
                        print(f"Still remain: {remaining:.2f} hrs.")
        else:
            raise ValueError("training_mode must be one of 'QA' or 'BIO'")

if __name__ == "__main__":
    torch.cuda.empty_cache() # clear crashed cache
    mrna_max_len =40
    mirna_max_len = 24
    train_datapath = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/starbase_degradome_UTR_windows_500_train.csv")
    valid_datapath = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/starbase_degradome_UTR_windows_500_validation.csv")
    test_datapath  = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/starbase_degradome_UTR_windows_500_test.csv")
    ckpt_path = os.path.join(PROJ_HOME, "checkpoints/TargetScan/TwoTowerTransformer/Longformer/520/embed=1024d/norm_by_key/LSE/best_composite_0.9042_0.9871_epoch12.pth")
    model = DTEA(mrna_max_len=mrna_max_len,
                mirna_max_len=mirna_max_len,
                device="cuda:0",
                epochs=100,
                embed_dim=1024,
                num_heads=8,
                num_layers=4,
                ff_dim=4096,
                batch_size=32,
                lr=3e-5,
                seed=10020,
                predict_span=False,
                predict_binding=False,
                predict_cleavage=True,
                use_longformer=True)
    # total_params = sum(param.numel() for param in model.parameters())
    # print(f"Total Parameters: {total_params}")
    # trainable_params = [p for p in model.parameters() if p.requires_grad]
    # print(f"Total trainable parameters = ", len(trainable_params))
    model.run(model=model,
              train_path=train_datapath,
              valid_path=valid_datapath,
              accumulation_step=8,
              training_mode="SPAN",
              ckpt_path=ckpt_path
            )
