import os
import torch
import random
import numpy as np
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# local import
import transformer_model as tm
from Data_pipeline import CharacterTokenizer
from plot_transformer_heatmap import plot_heatmap

from Global_parameters import PROJ_HOME, AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE

data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")

def predict(model, 
            **kwargs
            ):
    model.eval()
    with torch.no_grad():
        mRNA_seq = kwargs["mRNA_seq"]
        miRNA_seq = kwargs["miRNA_seq"]
        mRNA_seq_mask = kwargs["mRNA_seq_mask"]
        miRNA_seq_mask = kwargs["miRNA_seq_mask"] # (B, L)
        binding_logits, _, _ = model(
                                mirna = miRNA_seq,
                                mrna = mRNA_seq,
                                mirna_mask = miRNA_seq_mask,
                                mrna_mask = mRNA_seq_mask
                            )
        # predicted embedding
        # layer norm
        # z = model.predictor.cross_attn_output # (B,L,D)
        # z_norm = torch.nn.functional.layer_norm(z, normalized_shape=(z.shape[2],)) # (B, L, D)
        # z_masked = z_norm.masked_fill(mRNA_seq_mask.unsqueeze(-1)==0, 0)
        # z_pooled = z_masked.sum(dim=-1) / (mRNA_seq_mask.sum(dim=1, keepdim=True)) # mean-pooled on embeding (B, L)
        # z_pooled = z_pooled[0].detach().cpu() # (L,)
        # predicted attention 
        attention_score = model.predictor.cross_attn_layer.last_attention[0] # (H, mrna_len, mirna_len)
        attention_score = torch.amax(attention_score, dim=(0,2))
        attention_score = attention_score.detach().cpu()
        # binding probability
        if model.predict_binding:
            binding_prob = torch.nn.functional.sigmoid(binding_logits)
            binding_prob = binding_prob.detach().cpu()
    return attention_score, binding_prob

def encode_seq(model, 
               tokenizer, 
               mRNA_seq, 
               miRNA_seq,
               ):
    mRNA_seq_encoding = tokenizer(
        mRNA_seq,
        add_special_tokens=False,
        padding="max_length",
        max_length=model.mrna_max_len,
        truncation=True,
        return_attention_mask=True,
    )
    mRNA_seq_tokens = mRNA_seq_encoding["input_ids"]  # get input_ids
    mRNA_seq_mask = mRNA_seq_encoding["attention_mask"]  # get attention mask
    # print("Tokenized sequence length = ", len(mRNA_seq_tokens))

    miRNA_seq_encoding = tokenizer(
        miRNA_seq,
        add_special_tokens=False,
        padding="max_length",
        max_length=model.mirna_max_len,
        truncation=True,
        return_attention_mask=True,
    )
    miRNA_seq_tokens = miRNA_seq_encoding["input_ids"]  # get input_ids
    miRNA_seq_mask = miRNA_seq_encoding["attention_mask"]  # get attention mask
    
    mRNA_seq_tokens = torch.tensor(mRNA_seq_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
    miRNA_seq_tokens = torch.tensor(miRNA_seq_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
    mRNA_seq_mask = torch.tensor(mRNA_seq_mask, dtype=torch.long, device=model.device).unsqueeze(0)
    miRNA_seq_mask = torch.tensor(miRNA_seq_mask, dtype=torch.long, device=model.device).unsqueeze(0)
    return {"mRNA_seq": mRNA_seq_tokens, 
            "miRNA_seq": miRNA_seq_tokens, 
            "mRNA_seq_mask": mRNA_seq_mask, 
            "miRNA_seq_mask": miRNA_seq_mask}
    
def load_model(ckpt_name,
               **args_dict):
    # load model checkpoint
    # model = mirLM.create_model(**args_dict)
    model = tm.QuestionAnsweringModel(**args_dict)
    ckpt_path = os.path.join(PROJ_HOME, 
                            "checkpoints", 
                            "TargetScan/TwoTowerTransformer",
                            "CNN-tokenized",
                            str(model.mrna_max_len), 
                            ckpt_name)
    loaded_data = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(loaded_data)
    print(f"Loaded checkpoint from {ckpt_path}")
    return model

def single_base_perturbation(seq, pos):
    bases = ['A','T','C','G']
    seq = list(seq)
    orig = seq[pos]
    seq[pos] = random.choice([base for base in bases if base != orig])
    return ''.join(seq)

def viz_sequence(seq, 
                 attn_changes,
                #  emb_changes,
                 prob_changes,
                 base_ax=None,             
                 seed_start=None, 
                 seed_end=None,
                 figsize=(20, 8),
                 logo_height_frac=0.22,
                 pad_frac=0.1,
                 file_name=None):
    '''
    Draw two sequence-logo plots, either on a new figure or
    stacked above an existing heatmap Axes.

    Parameters
    ----------
    seq : list or str
        Sequence of bases.
    attn_changes, prob_changes : array-like of length len(seq)
    seed_start, seed_end : int or None
    base_ax : matplotlib.axes.Axes, optional
        If provided, the two new logo Axes will be placed
        above this Axes instead of making a new figure.
    figsize : tuple, only used if base_ax is None
    logo_height_frac : float
        Fraction of the base Axes' height to use for each logo.
    pad_frac : float
        Fraction of the base Axes' height to leave as vertical padding.
    Returns
    -------
    fig : matplotlib.figure.Figure
    logo_axes : list of two matplotlib.axes.Axes
    '''
    # Create a matrix for the sequence logo
    attn_logo_matrix = lm.alignment_to_matrix([seq]).astype(float)
    # emb_logo_matrix  = lm.alignment_to_matrix([seq]).astype(float)
    prob_logo_matrix = lm.alignment_to_matrix([seq]).astype(float) 
    # Scale the matrix by the changes
    all_matrices = [attn_logo_matrix, prob_logo_matrix]
    all_changes  = [attn_changes, prob_changes]
    for i, changes in enumerate(all_changes):
        logo_matrix = all_matrices[i]
        for pos, base in enumerate(seq):
            scale = changes[pos]
            logo_matrix.loc[pos, base] *= scale # scale only the original base
    if base_ax is None:
        fig, logo_axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig = base_ax.figure
        fig.set_size_inches(*figsize, forward=True)
        
        # get base Axes position (in figure fraction)
        pos = base_ax.get_position()
        new_h = pos.height * 0.5 # shrink the heatmap to 50%
        
        # keep the bottom edge in the same place, only height changes
        base_ax.set_position([pos.x0, pos.y0, pos.width, new_h])
        
        # resize the colorbar to match the new heatmap height —
        quad = base_ax.collections[0]          # the QuadMesh returned by sns.heatmap
        cbar = quad.colorbar                   # its attached Colorbar
        cax  = cbar.ax                         # the Axes of the colorbar

        # get their new positions
        heat_pos = base_ax.get_position()      # Bbox(x0, y0, w, h) of heatmap
        cpos     = cax.get_position()          # Bbox(x0, y0, w, h) of colorbar

        # re-anchor the colorbar so its bar is the same height as the heatmap
        cax.set_position([
            cpos.x0,         # keep the same left edge
            heat_pos.y0,     # align bottom with heatmap
            cpos.width,      # keep the same width
            heat_pos.height  # match heatmap’s new height
        ])
        x0, y0, w, h = heat_pos.bounds
        logo_height  = h * logo_height_frac
        pad          = h * pad_frac

        # create two new Axes above the heatmap
        logo_ax1 = fig.add_axes([x0, 
                                y0 + h + pad,
                                w, 
                                logo_height],
                              sharex=base_ax)
        logo_ax2 = fig.add_axes([x0, 
                                y0 + h + pad + logo_height + pad,
                                w, 
                                logo_height],
                              sharex=base_ax)
        logo_axes = [logo_ax1, logo_ax2]

    # fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20,8))
    ax_titles = ["Attention\nChanges", "Probability\nChanges"]
    for logo_matrix, ax, ax_title in zip(
        [attn_logo_matrix, prob_logo_matrix],
        logo_axes,
        ax_titles):
        # Create the sequence logo
        logo = lm.Logo(logo_matrix, 
                       ax=ax,
                       color_scheme='classic',)

        # Style the plot
        logo.style_spines(visible=False)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(range(len(seq)))
        ax.set_xticklabels(list(seq))
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        # logo.ax.set_yscale('log')  # Log-transformed y-axis
        # logo.ax.set_ylim(1e-6, 1e-5)  # Example: 1e-5 to 0.1
        ax.set_ylabel(ax_title, fontsize=AXIS_FONT_SIZE)

        # Add rectangle around specific bases (highlight_region is a tuple: (start, end))
        if seed_start is not None and seed_end is not None:
            y0, y1 = ax.get_ylim()
            rect = patches.Rectangle(
                (seed_start - 0.5, y0),  # (x, y)
                seed_end - seed_start + 1,  # width
                y1 - y0, # height
                linewidth=2,
                edgecolor='orange',
                facecolor='none'
            )
            ax.add_patch(rect)
        
    if file_name:
        fig.savefig(file_name, dpi=500, bbox_inches='tight')  # Save as PNG
    print(f"Logo plot saved to {file_name}")
    return fig, logo_axes

def main():
    # parser = get_argument_parser()
    # parser.add_argument(
    #     "--save_plot_dir",
    #     type=str,
    #     default="sequence_visualization",
    #     required=False
    # )
    # args = parser.parse_args()
    # args_dict = vars(args)

    mirna_max_len   = 24
    mrna_max_len    = 30
    predict_span    = True
    predict_binding = True
    device          = "cuda:3" 
    args_dict = {"mirna_max_len": mirna_max_len,
                 "mrna_max_len": mrna_max_len,
                 "device": device,
                 "embed_dim": 256,
                 "ff_dim": 512,
                 "predict_span": predict_span,
                 "predict_binding": predict_binding,}
    print("Loading model ... ")
    model = load_model(ckpt_name="best_composite_0.9911_0.9977_epoch11.pth",
                       **args_dict)
    
    test_data_path = os.path.join(PROJ_HOME, data_dir, 
                                 "TargetScan_train_30_randomized_start.csv")
    test_data = pd.read_csv(test_data_path)
    mRNA_seqs = test_data[["mRNA sequence"]].values
    miRNA_seqs = test_data[["miRNA sequence"]].values

    save_plot_dir = os.path.join(PROJ_HOME, 
                                 f"Performance/TargetScan_test/viz_seq_perturbation/TwoTowerTransformer/{mrna_max_len}")
    os.makedirs(save_plot_dir, exist_ok=True)

    # Testing the first sequence
    i=203041
    mRNA_seq = mRNA_seqs[i][0]
    miRNA_seq = miRNA_seqs[i][0]
    miRNA_id = test_data[["miRNA ID"]].iloc[i,0]
    mRNA_id = test_data[["Transcript ID"]].iloc[i,0]
    seed_start = test_data[["seed start"]].iloc[i,0]
    seed_end = test_data[["seed end"]].iloc[i,0]
    print("miRNA id = ",miRNA_id)
    print("mRNA id = ", mRNA_id)
    # replace U with T and reverse miRNA to 3' to 5' 
    miRNA_seq = miRNA_seq.replace("U", "T")[::-1]
    
    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],  # add RNA characters, N is uncertain
        model_max_length=model.mrna_max_len,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="right",  # since HyenaDNA is causal, we pad on the left
    )
    encoded = encode_seq(
        model=model, 
        tokenizer=tokenizer,
        mRNA_seq=mRNA_seq,
        miRNA_seq=miRNA_seq,
    )

    model.to(args_dict["device"])
    wt_attn_score, wt_binding_prob = predict(model, **encoded)
    # print("Wild-type prediction = ", wt_prob)
    
    # convert mRNA seq tokens to mRNA ids
    mRNA_tokens_squeezed = encoded["mRNA_seq"].squeeze(0)
    mRNA_ids = [tokenizer._convert_id_to_token(d.item()) for d in mRNA_tokens_squeezed]
    # convert miRNA seq tokens to mRNA ids
    miRNA_tokens_squeezed = encoded["miRNA_seq"].squeeze(0)
    miRNA_ids = [tokenizer._convert_id_to_token(d.item()) for d in miRNA_tokens_squeezed]

    fig, ax_attn = plot_heatmap(model,
                 miRNA_seq=miRNA_ids,
                 mRNA_seq=mRNA_ids,
                 miRNA_id = miRNA_id,
                 mRNA_id = mRNA_id,
                 seed_start = seed_start,
                 seed_end = seed_end,) 

    # perturb mRNA sequence
    attn_deltas = []
    emb_deltas  = []
    prob_deltas = []
    print("Start perturbing mrna ...")
    for pos in range(len(mRNA_seq)):
        attn_score_delta_list = []
        # pooled_emb_delta_list = []
        binding_prob_delta_list = []
        # perturb the base 3 times and average the delta
        for _ in range(1):
            perturbed_mRNA = single_base_perturbation(seq=mRNA_seq, pos=pos)
            encoded = encode_seq(
                model=model,
                tokenizer=tokenizer,
                mRNA_seq=perturbed_mRNA,
                miRNA_seq=miRNA_seq,
            )
            attn_score,  binding_prob = predict(model, **encoded)
            attn_score_delta = abs(wt_attn_score - attn_score) # (L,)
            binding_prob_delta = abs(wt_binding_prob - binding_prob) 
            attn_score_delta_list.append(attn_score_delta)
            binding_prob_delta_list.append(binding_prob_delta)
        # only take the change at the current position
        all_attn_score_delta = torch.stack(attn_score_delta_list, dim=0)[:, pos] # (3, L) -> (3,)
        all_binding_prob_delta = torch.stack(binding_prob_delta_list) # (3, )
        attn_delta = (all_attn_score_delta.sum(dim=0) / len(all_attn_score_delta)).item()
        prob_delta = (all_binding_prob_delta.sum(dim=0) / len(all_binding_prob_delta)).item()
        attn_deltas.append(attn_delta) # (seed_len, )
        prob_deltas.append(prob_delta) # (seed_len, )

    print(len(prob_deltas))
    
    # print("Max in delta = ", max(deltas))
    print("plot changes on base logos ...")
    file_path = os.path.join(save_plot_dir, f"{mRNA_id}_{miRNA_id}_attn_perturbed.svg")
    fig, ax_viz = viz_sequence(seq=mRNA_seq, # visualize change on the original mRNA seq
                 attn_changes=attn_deltas,
                #  emb_changes=emb_deltas,
                 prob_changes=prob_deltas,
                 seed_start=seed_start,
                 seed_end=seed_end,
                 base_ax=ax_attn,
                 figsize=(12, 9),
                 file_name=file_path)

if __name__ == '__main__':
    main()
