import os
import csv
import time
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
from Global_parameters import PROJ_HOME

data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")

def get_complementary_seq(seq, mRNA):
    d = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    try:
        s = ''.join([d[c] for c in seq]) # ignore N
        return s
    except KeyError:
        print(f"KeyError found in mRNA sequence: {mRNA}")

def check_complementarity(miRNA, mRNA, seg_length = 6):
    # check whether there are 6-consecutive complmentary segment between miRNA and mRNA
    miRNA = miRNA[::-1]
    miRNA = miRNA.replace('U', 'T')
    for start in range(0, len(mRNA)-seg_length+1):
        mRNA_seg = mRNA[start:start + seg_length]
        if 'N' in mRNA_seg:
            print("Found N in seed candidate, skipping")
            continue
        else:
            mRNA_seg_c = get_complementary_seq(mRNA_seg, mRNA)
            if mRNA_seg_c in miRNA:
                return True
    return False

# mRNA = "ACGCGCCGGCTCTGGCGGCGGGCTCCGGGCCCAGCCCCGCAGCGGCCAGGAGCACGGGCCGGCGATGCGGCTGCCGGCGCACCCCCGCCCGGGGCCGAGGTGCCTACGGGCGGGGGCTGCAGGGCCACGCCGCCACGGGGTCAGTGTTGCTACCGCCGCTGCCAGGCGCTCCGGAGCTGTCCGCTTCAGCACCACGCCGNCGCCGGTAGTGGTGGTGCGGACCAGCCCGGAGCCCGCCGACGCACTCATGCACTTTAGAACCTCGGGCCGCAGCTCCACCCCGCACACCGGAACCGACACAGAGACCCACGGCCCCCAGCCCGCCCCCTCCCGCACGGCTCCCCTGCCCCGCCCCTCCTCCGATCAATCTGGTCTGCAGCCCGGGCAGCTGCGGCGTCGGACTGCGCGGCCAGGGAGGTTTGGCTGCATATTTGCATGAGCTTCCTACCCACCTCAGCCTAGCCCTCCGGGCAATTCCTCAACTGTACCCCTGTCTGGCG"
# miRNA = "CAAAGUGCUUACAGUGCAGGUAG"
# match = check_complementarity(miRNA, mRNA)
# print(match)

# read positive pairs
# positive_pairs_human = pd.read_csv(os.path.join(data_dir, "Positive_pairs_human.csv"), sep='\t')
positive_pairs_mouse = pd.read_csv(os.path.join(data_dir, "Positive_pairs_mouse.csv"), sep='\t')
# negative_pairs_human = pd.read_csv(os.path.join(data_dir, "negative_pairs_human.csv"), sep='\t')
negative_pairs_mouse = pd.read_csv(os.path.join(data_dir, "negative_pairs_mouse.csv"), sep='\t')

# uncomment to construct a single column that holds the coordinate pairs
# positive_pairs_human["UTR_coords"] = positive_pairs_human["UTR_coords"].map(ast.literal_eval)

positive_pairs = positive_pairs_mouse.to_dict(orient = "records")
negative_pairs = negative_pairs_mouse.to_dict(orient = "records")

# read mrna 3utr
mRNA_df = pd.read_csv(os.path.join(data_dir, "mouse_mrna_seq.csv.gz"), sep='\t', compression='gzip')
# mRNA_df = pd.read_csv(os.path.join(data_dir, "mouse_mrna_seq.csv.gz"), sep="\t", compression="gzip")
# mRNA_df = pd.concat([human_mRNA_df, mouse_mRNA_df])

# read mature sequence of miRNA
path     = os.path.join(data_dir, "miR_Family_Info.txt")
miRNA_df = pd.read_csv(path, sep='\t')
# filter for human (9606), mouse (10090)
species_ids = [10090]
miRNA_df    = miRNA_df[miRNA_df["Species ID"].isin(species_ids)]

# get window-length mRNA segment
def get_window_len_mrna(mRNA_seq,
                         window_len,
                         seed_start,
                         seed_end):
    
    seq_len = len(mRNA_seq)
    if seq_len <= window_len:
        return {"mRNA sequence": mRNA_seq,
                "seed start": seed_start-1, # because TargetScan index starting from 1
                "seed end": seed_end-1}
    else:
        seed_len  = seed_end - seed_start + 1
        total_ext = window_len - seed_len

        # segment mRNA seq binding site
        seed_start   = seed_start - 1 # because TargetScan index starting from 1
        seed_end     = seed_end - 1
        ext_left_max = total_ext
        ext_left_min = 0
        ext_left     = random.randint(ext_left_min, ext_left_max)
        ext_left     = min(ext_left, seed_start)
        ext_right    = total_ext  - ext_left
        window_start = seed_start - ext_left
        window_end   = seed_end   + ext_right + 1 # because window end needs to include the last nucleotide
        if window_end > seq_len:
            window_end   = seq_len
            window_start = window_end - window_len # to ensure mRNA seq is `window_len` long
            ext_right    = seq_len    - seed_end
            ext_left     = total_ext  - ext_right # to ensure total extention = ext_right + ext_left
        new_seed_start = ext_left
        new_seed_end   = new_seed_start + seed_len - 1 # because seed start and seed end needs to be index 
        mRNA_seg = mRNA_seq[window_start:window_end]
        print(f"Length of mRNA_seg = ", len(mRNA_seg))
        if len(mRNA_seg) >= 6: # must be greater than the shortest seed length
            return {"mRNA sequence": mRNA_seg,
                    "seed start": new_seed_start,
                    "seed end":   new_seed_end}
        else:
            print("Sequence segment is shorter than the shortest seed length 6. Returning None")
            return {"mRNA sequence": None}

def get_window_len_multi_seeds(mRNA_seq,
                               window_len, 
                               seeds):
    seq_len   = len(mRNA_seq)
    if seq_len <= window_len:
        return [{"mRNA sequence": mRNA_seq,
                "seeds": [(s-1, e-1) for s, e in seeds]}] # because TargetScan index starting from 1
    if len(seeds) < 1:
        raise("Fewer than 1 seed is not allowed")
    elif len(seeds) == 1:
        # only having one seed within mRNA seq, expand on both sides
        seed = seeds[0]
        seg_dict = get_window_len_mrna(mRNA_seq=mRNA_seq, window_len=window_len, seed_start=seed[0], seed_end=seed[1])
        return [{
            "mRNA sequence": seg_dict["mRNA sequence"],
            "seeds": [(seg_dict["seed start"], seg_dict["seed end"])]
        }]
    else:
        results = [] # return list of valid segments and shifted seeds
        # slide a window across mRNA
        i = 0
        while i < len(seeds):
            first_seed_start = seeds[i][0]
            j = i
            while j < len(seeds) and seeds[j][1] - first_seed_start + 1 <= window_len:
                j += 1
            # now seeds[i:j] all fit; last seed is seeds[j-1]
            group = seeds[i:j]
            last_end = group[-1][1]

            # carve out window and compute new offsets
            window_dict = get_window_len_mrna(mRNA_seq=mRNA_seq, window_len=window_len, seed_start=first_seed_start, seed_end=last_end)
            subseq = window_dict["mRNA sequence"]
            new_s0 = window_dict["seed start"]
            if subseq is not None:
                # shift all group seeds by the same amount
                shift = first_seed_start - new_s0
                shifted = [ (s - shift, e - shift) for s,e in group ]

                results.append({"mRNA sequence": subseq, "seeds": shifted})
                # advance i to the next seed past this group
                i = j
            else:
                print(f"Cannot find valid segment for transcript {mRNA_seq}.", flush=True)
                i += 1 # advance to the next seed to avoid infinite loop
        return results

print("Putting together positive samples: ")
window_len = 100
miRNA_len  = 24
multi_seeds = False
start_time = time.time()

# assemble positive pairs
positive_samples = []
max_len = 0

# with open(os.path.join(data_dir, f"mouse_positive_samples_{str(window_len)}_randomized_start.csv"), "w", newline="") as f:
#     if multi_seeds:
#         fieldnames = ["Transcript ID", 
#                        "miRNA ID", 
#                        "miRNA sequence", 
#                        "mRNA sequence", 
#                        "seeds",
#                        "label"]
#     else:
#         fieldnames = ["Transcript ID", 
#                       "miRNA ID", 
#                       "miRNA sequence", 
#                       "mRNA sequence", 
#                       "seed start",
#                       "seed end",
#                       "label"]
#     writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
#     writer.writeheader()
#     for positive_pair in positive_pairs:
#         miRNA_ID       = positive_pair.get("miRNA", "")
#         tran_id        = positive_pair.get("Transcript_ID", "")
#         # if (miRNA_ID, tran_id) in done:
#         #     continue
#         if miRNA_ID: 
#             miRNA_seq = miRNA_df.loc[miRNA_df["MiRBase ID"] == miRNA_ID, "Mature sequence"]
#             miRNA_seq = miRNA_seq.values[0]
#         if tran_id:
#             mRNA_seq   = mRNA_df.loc[(mRNA_df["Transcript ID"] == tran_id), "mRNA sequence"]
#             if multi_seeds:
#                 orig_seeds = positive_pair["UTR_coords"]
#                 orig_seeds = sorted(orig_seeds)
#                 orig_seed_start = orig_seeds[0][0] # first seed start
#                 orig_seed_end   = orig_seeds[-1][1] # last seed end
#             else:
#                 orig_seed_start = positive_pair["UTR_start"]
#                 orig_seed_end   = positive_pair["UTR_end"]
#             if (len(mRNA_seq) > 0) and (len(mRNA_seq.values[0]) > orig_seed_start) and (len(mRNA_seq.values[0]) > orig_seed_end):    
#                 mRNA_seq = mRNA_seq.values[0]
#                 if multi_seeds:
#                     # return a list of mRNA seq segments containing shifted seeds
#                     res_list = get_window_len_multi_seeds(mRNA_seq=mRNA_seq,
#                                                     window_len=window_len,
#                                                     seeds = orig_seeds,)
#                     for res in res_list:
#                         valid_seeds = []
#                         for seed_start, seed_end in res["seeds"]:
#                             seed = res["mRNA sequence"][seed_start:seed_end+1]
#                             if check_complementarity(miRNA=miRNA_seq, mRNA=seed): # check seed complementarilty in mRNA segments
#                                 max_len = max(max_len, len(res["mRNA sequence"]))
#                                 valid_seeds.append((seed_start, seed_end))
#                         if len(valid_seeds) > 0:
#                             writer.writerow({
#                                             "Transcript ID":  tran_id,
#                                             "miRNA ID":       miRNA_ID,
#                                             "miRNA sequence": miRNA_seq,
#                                             "mRNA sequence":  res["mRNA sequence"],
#                                             "seeds":          valid_seeds,
#                                             "label":          1
#                                             })
#                         else:
#                             raise RuntimeError(f"No complementary seed found in miRNA:{miRNA_ID}: {miRNA_seq} \
#                                 and mRNA id:{tran_id}: {mRNA_seq}\noriginal seed start:{orig_seed_start}, \
#                                 original seed end:{orig_seed_end}\nmRNA segment: {res['mRNA sequence']}\n.")
#                 else:
#                     res = get_window_len_mrna(mRNA_seq=mRNA_seq,
#                                             window_len=window_len,
#                                             seed_start=orig_seed_start,
#                                             seed_end=orig_seed_end)
#                     if res["mRNA sequence"] is not None:
#                         seed_region = res["mRNA sequence"][res["seed start"]:res["seed end"]+1]
#                         if check_complementarity(miRNA=miRNA_seq, mRNA=seed_region): # check seed complementarilty in mRNA segments
#                             max_len = max(max_len, len(res["mRNA sequence"]))
#                             writer.writerow({
#                                             "Transcript ID":  tran_id,
#                                             "miRNA ID":       miRNA_ID,
#                                             "miRNA sequence": miRNA_seq,
#                                             "mRNA sequence":  res["mRNA sequence"],
#                                             "seed start":     res["seed start"],
#                                             "seed end":       res["seed end"],
#                                             "label":          1
#                                             })
#                         else:
#                             raise RuntimeError(f"No complementary seed found in miRNA:{miRNA_ID}: {miRNA_seq} \
#                                 and mRNA id:{tran_id}: {mRNA_seq}\noriginal seed start:{orig_seed_start}, \
#                                 original seed end:{orig_seed_end}\nmRNA segment: {res['mRNA sequence']}\n.")
#                     else:
#                         print(f"Cannot find valid segment for transcript [{tran_id}] and miRNA: [{miRNA_ID}].", flush=True)
#             else:
#                 print(f"One of the following happened to transcript [{tran_id}]:"
#                     f"1. No such gene is found OR "
#                     f"2. mRNA length <= seed start OR "
#                     f"3. mRNA length <= seed end", flush=True)       
#         else:
#             print(f"Cannot find mRNA seq: [{tran_id}].", flush=True)

# print("Time taken to put together positive samples = ", (time.time() - start_time) / 60)
# print(f"Maximum mRNA length = {max_len}.")


# # assemble negative pairs
# print("Putting together negative samples: ")
# start_time = time.time()
# seg_len = window_len
# print(f"Segment length = {seg_len}")
# negative_samples = []
# max_length = 0

# with open(os.path.join(data_dir, f"mouse_negative_samples_{str(window_len)}.csv"), "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=["Transcript ID", 
#                                            "miRNA ID", 
#                                            "miRNA sequence", 
#                                            "mRNA sequence", 
#                                            "seed start",
#                                            "seed end",
#                                            "label"], delimiter="\t")
#     writer.writeheader()
#     for negative_pair in negative_pairs:
#         miRNA_ID       = negative_pair.get("miRNA", "")
#         tran_id        = negative_pair.get("Transcript_ID", "")
#         if miRNA_ID: 
#             miRNA_seq  = miRNA_df.loc[miRNA_df["MiRBase ID"] == miRNA_ID, "Mature sequence"]
#             miRNA_seq  = miRNA_seq.values[0]
#         if tran_id:
#             mRNA_seq = mRNA_df.loc[(mRNA_df["Transcript ID"] == tran_id), "mRNA sequence"]
#             if len(mRNA_seq) > 0:    
#                 mRNA_seq = mRNA_seq.values[0]
#                 # randomly select seg_len-nt mRNA segment
#                 if len(mRNA_seq) < seg_len:
#                     if not check_complementarity(miRNA=miRNA_seq, mRNA=mRNA_seq):
#                         writer.writerow({
#                                         "Transcript ID":  tran_id,
#                                         "miRNA ID":       miRNA_ID,
#                                         "miRNA sequence": miRNA_seq,
#                                         "mRNA sequence":  mRNA_seq,
#                                         "seed start":     -1,
#                                         "seed end":       -1,
#                                         "label":          0
#                                         })
#                         max_length = max(max_length, len(mRNA_seq))
#                         continue
#                 else:
#                     start    = random.randint(0, len(mRNA_seq)-seg_len)
#                     end      = min(start + seg_len, len(mRNA_seq))
#                     mRNA_seg = mRNA_seq[start:end]
#                     match    = check_complementarity(miRNA_seq, mRNA_seg)
#                     if match:
#                         print("Found complementarity in negative samples, re-sampling mRNA segment...")
#                         possible_starts    = len(mRNA_seq) - seg_len + 1
#                         n_samples_per_pair = 10
#                         max_tries          = min(n_samples_per_pair, possible_starts) # choose between N samples or max possible segments per sample
#                         tried_starts       = set()

#                         while len(tried_starts) < max_tries:
#                             start = random.randint(0, possible_starts - 1)
#                             if start in tried_starts:
#                                 # Already tried this start; draw again
#                                 continue
#                             tried_starts.add(start)
#                             end = start + seg_len
#                             mRNA_seg = mRNA_seq[start:end]
#                             if not check_complementarity(miRNA_seq, mRNA_seg):
#                                 writer.writerow({
#                                     "Transcript ID":    tran_id,
#                                     "miRNA ID":         miRNA_ID,
#                                     "miRNA sequence":   miRNA_seq,
#                                     "mRNA sequence":    mRNA_seg,
#                                     "seed start":       -1,
#                                     "seed end":         -1,
#                                     "label":            0
#                                 })
#                     else:
#                         writer.writerow({ 
#                                         "Transcript ID":    tran_id,
#                                         "miRNA ID":         miRNA_ID,
#                                         "miRNA sequence":   miRNA_seq,
#                                         "mRNA sequence":    mRNA_seg,
#                                         "seed start":       -1,
#                                         "seed end":         -1,
#                                         "label":            0
#                                         })
#                     max_length = max(max_length, len(mRNA_seg))    
#             else:
#                 print(f"Cannot find mRNA seq: [{tran_id}].", flush=True)

# print("Maximum mRNA length = ", max_length)
# print(f"Time taken for putting together negative samples = {(time.time() - start_time)/60} min")

positive_human_samples = pd.read_csv(os.path.join(data_dir, f"human_positive_samples_{str(window_len)}_randomized_start.csv"), sep='\t')
# negative_human_samples = pd.read_csv(os.path.join(data_dir, f"human_negative_samples_{str(window_len)}.csv"), sep='\t')
positive_mouse_samples = pd.read_csv(os.path.join(data_dir, f"mouse_positive_samples_{str(window_len)}_randomized_start.csv"), sep='\t')
# negative_mouse_samples = pd.read_csv(os.path.join(data_dir, f"mouse_negative_samples_{str(window_len)}.csv"), sep='\t')
samples = pd.concat([positive_human_samples, positive_mouse_samples], axis=0)
# samples = pd.concat([positive_human_samples, positive_mouse_samples, negative_human_samples, negative_mouse_samples], axis=0) 
ds_train, ds_rem = train_test_split(samples, test_size=0.1, random_state=42, shuffle=True)
ds_val, ds_test = train_test_split(ds_rem, test_size=0.2, shuffle=False)
ds_train.to_csv(os.path.join(data_dir, f"Positive_primates_train_{str(window_len)}_randomized_start.csv"), sep=',', index=False)
ds_val.to_csv(os.path.join(data_dir, f"Positive_primates_validation_{str(window_len)}_randomized_start.csv"), sep=',', index=False)
ds_test.to_csv(os.path.join(data_dir, f"Positive_primates_test_{str(window_len)}_randomized_start.csv"), sep=',', index=False)
