import os
import torch
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import transformer_model as tm
import matplotlib.patches as patches
from matplotlib import colors
from Data_pipeline import CharacterTokenizer
from Global_parameters import PROJ_HOME, AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE

def load_model(
        ckpt_name,
        **args_dict):
    # load model checkpoint
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

def predict(model,
            mRNA_seq,
            miRNA_seq,
            mRNA_seq_mask,
            miRNA_seq_mask,
            seed_start=-1,
            seed_end=-1):
    model.eval()
    binding_prob, exact_match, f1 = None, None, None
    with torch.no_grad():
        output = model(
            mirna = miRNA_seq,
            mrna = mRNA_seq,
            mirna_mask = miRNA_seq_mask,
            mrna_mask = mRNA_seq_mask
        )
        binding_logits, start_logits, end_logits = output
        if model.predict_binding:
            binding_prob = F.sigmoid(binding_logits)
            binding_prob = binding_prob.detach().cpu().item()
        if model.predict_span:
            start_pred = torch.argmax(start_logits, dim=-1).detach().cpu().long()
            end_pred   = torch.argmax(end_logits, dim=-1).detach().cpu().long()
            if seed_start != -1 and seed_end != -1:
                span_metrics = model.compute_span_metrics(start_pred, end_pred, 
                                                            start_labels=[seed_start],
                                                            end_labels=[seed_end])
                f1 = span_metrics["f1"]
                exact_match = span_metrics["exact_match"]
            else:
                f1 = None,
                exact_match = None

        return binding_prob, exact_match, f1

def plot_heatmap(model,
                 mRNA_seq,
                 miRNA_seq,
                 mRNA_id,
                 miRNA_id,
                 seed_start,
                 seed_end,
                 figsize=(12,6),
                 metrics=None,
                 file_name=None,
                 save_plot_dir=os.getcwd()):
    attn_weights = model.predictor.cross_attn_layer.last_attention
    attn_weights = torch.amax(attn_weights[0], dim=0) # (mrna, mirna)
    attn_weights = attn_weights.transpose(0,1) # (mirna, mrna)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    a, b = 0.1, attn_weights.max().item()
    norm = colors.Normalize(vmin=a, vmax=b)
    # plot attn weights for each head
    # for h in range(attn_weights.shape[0]):
    w = attn_weights.detach().cpu() #[mirna, mrna]
    # 1) zero out anything below a
    w = torch.where(w < a, torch.zeros_like(w), w)
    # 2) clip anything above b
    w = torch.where(w > b, torch.full_like(w, b), w)
    im = sns.heatmap(w.numpy(), 
            ax=ax,
            cmap=sns.color_palette('mako', as_cmap=True), # "Blues"
            xticklabels=mRNA_seq,
            yticklabels=miRNA_seq,
            norm=norm,
            cbar=True
    )

    # get the Colorbar object
    cbar = im.collections[0].colorbar
    # set its tick‐label fontsize to 12
    cbar.ax.tick_params(labelsize=12)

    if seed_start != -1 and seed_end != -1:    
        # seed_start and seed_end are indices into the mRNA sequence (0-based)
        xs = seed_start
        xe = seed_end
        seed_len = xe - xs + 1
            
        # get where the seed starts and ends in miRNA
        # skip [PAD] tokens
        i = len(miRNA_seq) - 1
        token = miRNA_seq[i]
        while token == '[PAD]':
            i -= 1
            token = miRNA_seq[i]
        ys = i # seed ends at the 2nd last base
        ys = ys - seed_len

        # draw a red rectangle around those rows
        rect = patches.Rectangle(
            (xs, ys),  # lower-left corner in data coords
            seed_len,              # width = number of seed bases
            seed_len,             # height = number of seed bases
            linewidth=1,
            edgecolor="orange",
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_xlabel(mRNA_id, fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel(miRNA_id, fontsize=AXIS_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    # ax.set_title(f"Head {h+1}", fontsize=TITLE_FONT_SIZE-1)
        # fig.suptitle("miRNA-mRNA Cross-Attention Heatmap")
    # if metrics is not None:
    #     fig.text(0.5, 0.93, 
    #             f"Binding probability = {metrics['binding_prob']:.3f}, F1 Score = {metrics['f1']}", 
    #             fontsize=TITLE_FONT_SIZE, ha='center')
    if file_name is not None:
        fig.savefig(file_name, dpi=500, bbox_inches='tight')
    # else:
    #     file_name = os.path.join(save_plot_dir, f"binding_span_{mRNA_id}_{miRNA_id}_heatmap_w_CNN_regenerated.svg")
    #     fig.savefig(file_name, dpi=500, bbox_inches='tight')
    #     print(f"Heatmap is saved to {file_name}")
    return fig, ax

def main():
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
    
    data_dir = os.path.join(PROJ_HOME, 'TargetScan_dataset')
    test_datapath = os.path.join(PROJ_HOME, data_dir, 
                                 "TargetScan_train_30_randomized_start.csv")
    test_data  = pd.read_csv(test_datapath, sep=',')
    mRNA_seqs  = test_data[["mRNA sequence"]].values
    miRNA_seqs = test_data[["miRNA sequence"]].values
    miRNA_IDs   = test_data[["miRNA ID"]].values
    mRNA_IDs    = test_data[["Transcript ID"]].values
    labels      = test_data[["label"]].values
    seed_starts = test_data[["seed start"]].values
    seed_ends   = test_data[["seed end"]].values
    
    model = load_model(ckpt_name="best_composite_0.9911_0.9977_epoch11.pth",
                       **args_dict)

    # Testing the first sequence
    i=299857 # row number - 2
    mRNA_seq  = mRNA_seqs[i][0]
    miRNA_seq = miRNA_seqs[i][0]
    mRNA_ID   = mRNA_IDs[i][0]
    miRNA_ID  = miRNA_IDs[i][0]
    label     = labels[i][0]
    seed_start = seed_starts[i][0]
    seed_end   = seed_ends[i][0]
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
    # convert mRNA seq tokens to mRNA ids
    mRNA_tokens_squeezed = encoded["mRNA_seq"].squeeze(0)
    mRNA_ids = [tokenizer._convert_id_to_token(d.item()) for d in mRNA_tokens_squeezed]
    # convert miRNA seq tokens to mRNA ids
    miRNA_tokens_squeezed = encoded["miRNA_seq"].squeeze(0)
    miRNA_ids = [tokenizer._convert_id_to_token(d.item()) for d in miRNA_tokens_squeezed]

    model.to(args_dict["device"])
    binding_prob, exact_match, f1 = predict(
            model=model,
            mRNA_seq=encoded["mRNA_seq"],
            miRNA_seq=encoded["miRNA_seq"],
            mRNA_seq_mask=encoded["mRNA_seq_mask"],
            miRNA_seq_mask=encoded["miRNA_seq_mask"],
            seed_start=seed_start,
            seed_end=seed_end,
    )

    save_plot_dir = os.path.join(PROJ_HOME, "Performance/TargetScan_test", "TwoTowerTransformer", str(mrna_max_len))
    os.makedirs(save_plot_dir, exist_ok=True)
    plot_heatmap(model,
                 miRNA_seq=miRNA_ids,
                 mRNA_seq=mRNA_ids,
                 miRNA_id = miRNA_ID,
                 mRNA_id = mRNA_ID,
                 seed_start = seed_start,
                 seed_end = seed_end,
                 metrics = {"binding_prob": binding_prob, "f1": f1},
                 save_plot_dir=save_plot_dir)
    
if __name__ == '__main__':
    main()
