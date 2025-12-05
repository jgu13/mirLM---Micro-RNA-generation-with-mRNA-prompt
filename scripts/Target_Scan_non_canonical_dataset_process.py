import os
import time
import csv
import random
import gzip
import pandas as pd
import numpy as np
from Global_parameters import PROJ_HOME

data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")
predicted_targets_f = "Human_Predicted_Targets_Context_Scores.default_predictions.txt.zip"

# positive miRNA and mRNA pairs
path = os.path.join(data_dir, predicted_targets_f)
predicted_targets = pd.read_csv(path, sep='\t', compression="zip")
# filter for human (9606), mouse (10090)
tax_ids = [9606]
top_predicted_targets = predicted_targets[
    predicted_targets["Gene Tax ID"].isin(tax_ids) 
    ]
# filter out non-canonical sites
top_predicted_targets = top_predicted_targets.loc[top_predicted_targets["Site Type"].isin([-2,-3])]

positive_pairs = top_predicted_targets[[
     "miRNA",
     "Transcript ID",
     "UTR_start",
     "UTR_end"
]]
positive_pairs.columns = ["miRNA", "Transcript_ID", "UTR_start", "UTR_end"]
positive_pairs = positive_pairs.drop_duplicates(subset=["Transcript_ID", "miRNA", "UTR_start", "UTR_end"])

# Uncomment to build a single column that already holds the coordinate pair 
# positive_pairs["coords"] = list(zip(positive_pairs["UTR_start"], positive_pairs["UTR_end"]))
# Uncomment to group & aggregate into lists of tuples 
# positive_pairs = (positive_pairs
#                .groupby(["Transcript_ID", "miRNA"])["coords"]
#                .apply(list)                 # → list of tuples
#                .reset_index(name="UTR_coords"))

positive_pairs.loc[:, "label"] = 1
positive_pairs.to_csv(os.path.join(data_dir, "Positive_pairs_human_non_canonical.csv"), sep='\t', index=False)

print("Total predicted mirna-transcript pairs = ", len(positive_pairs))

# negative miRNA and mRNA pairs: select mRNA species that is not in positive pairs with the miRNA 
all_mrnas       = set(predicted_targets["Transcript ID"].unique())
print("Start generating an equal number of negative samples.")
with open(os.path.join(data_dir, "negative_pairs_human_non_canonical.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["miRNA", "Transcript_ID", "coords", "label"], delimiter="\t")
    writer.writeheader()
    for mirna, group in positive_pairs.groupby("miRNA"):
        pos_set       = set(group['Transcript_ID'].tolist())
        n_pos         = len(group['Transcript_ID'].tolist())
        neg_mrna_pool = list(all_mrnas - pos_set)
        if len(neg_mrna_pool) < n_pos:
            raise ValueError(ValueError(f"Warning: {mirna}: Pool of negative mrna ({len(neg_mrna_pool)}) is fewer than positive mrnas ({n_pos})!"))
        chosen_neg = random.sample(neg_mrna_pool, k=n_pos)
        for mrna in chosen_neg:
            writer.writerow(
                {"miRNA":        mirna,
                "Transcript_ID": mrna,
                "coords":        -1,
                "label":         0}
            )

print("Finished generating negative samples")

# mRNAseq_path = os.path.join(data_dir, "mouse_3utr_sequences.fa.gz")
# mRNA_seq_dict = []
# with gzip.open(mRNAseq_path, "rt") as handle:
#     for record in SeqIO.parse(handle, "fasta"):
#         # full UTR sequence
#         seq = str(record.seq)
#         # split off the coords, keep only the transcript ID
#         tran_id, _coords = record.id.split("::", 1)
#         # if you only need transcript → sequence mapping:
#         mRNA_seq_dict.append({
#             "Transcript ID": tran_id,
#             "mRNA sequence": seq
#         })
                
# mRNA_seq_df = pd.DataFrame(mRNA_seq_dict)
# print(mRNA_seq_df.head(n=10))
# mRNA_seq_df.to_csv(
#     os.path.join(os.path.join(data_dir, "mouse_mrna_seq.csv.gz")), 
#     sep='\t', 
#     index=False, 
#     compression='gzip')

# mRNA_df_path = os.path.join(data_dir, "mouse_3utr_sequences.txt.zip")
# utr_df = pd.read_csv(mRNA_df_path, sep='\t', compression='zip')
# # filter for mouse
# species_ids = [10090]
# utr_df = utr_df[utr_df['Species ID'].isin(species_ids)]
# def clean_seq(s):
#     return s.replace("-", "").upper().replace("U", "T")
# utr_df["UTR sequence"] = utr_df["UTR sequence"].apply(clean_seq)
# utr_df.columns = ['Transcript ID', 'Gene ID', 'Gene Symbol', 'Species ID', 'mRNA sequence']
# mrna_save_path = os.path.join(data_dir, "mouse_mrna_seq.csv.gz")
# utr_df.to_csv(mrna_save_path,
#               sep='\t',
#               index=False,
#               compression='gzip')

# mRNA_df_path = os.path.join(data_dir, "human_utr_sequences.txt.zip")
# utr_df = pd.read_csv(mRNA_df_path, sep='\t', compression='zip')
# # filter for mouse
# species_ids = [9606]
# utr_df = utr_df[utr_df['Species ID'].isin(species_ids)]
# def clean_seq(s):
#     return s.replace("-", "").upper().replace("U", "T")
# utr_df["UTR sequence"] = utr_df["UTR sequence"].apply(clean_seq)
# utr_df.columns = ['Transcript ID', 'Gene ID', 'Gene Symbol', 'Species ID', 'mRNA sequence']
# mrna_save_path = os.path.join(data_dir, "human_mrna_seq.csv.gz")
# utr_df.to_csv(mrna_save_path,
#               sep='\t',
#               index=False,
#               compression='gzip')