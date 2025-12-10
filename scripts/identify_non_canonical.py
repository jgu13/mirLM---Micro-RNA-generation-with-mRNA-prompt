import pandas as pd
import os
from Global_parameters import PROJ_HOME

data_dir = os.path.join(PROJ_HOME, "Mimosa_dataset")

# Updated WC rules for miRNA base -> Target base (miRNA base is key)
# Assuming Target Site uses A, T, G, C
WC_rules = {"U": "A", "A": "T", "C": "G", "G": "C"}

def is_wc_match_aligned(miRNA_segment, target_segment):
    """Checks for a Watson-Crick match between two equal-length strings."""
    if len(miRNA_segment) != len(target_segment):
        return False
        
    for i in range(len(miRNA_segment)):
        miRNA_base = miRNA_segment[i]
        target_base = target_segment[i]
        
        if WC_rules.get(miRNA_base) != target_base:
            return False
            
    return True

# --- Main script logic ---
try:
    mmu_mimosa = pd.read_csv(os.path.join(data_dir, "mmu_mimosa.csv"))
except FileNotFoundError:
    print("Error: mmu_mimosa.csv not found.")
    exit()

mmu_mimosa["canonical_site"] = 0 # Initialize as non-canonical (0)
mmu_mimosa["seed start"] = -1
mmu_mimosa["seed end"] = -1

for index, row in mmu_mimosa.iterrows():
    miRNA_seq = row["miRNA sequence"]
    # 1. Reverse the mRNA target site for correct 5'-5' alignment
    target_rev_seq = row["mRNA sequence"][::-1] 
    
    # Iterate through all possible starting positions (i) in the reversed target sequence
    # to check the maximum possible seed length (8 bases).
    max_len = 8
    
    for i in range(len(target_rev_seq) - max_len + 1): 
        # The potential match in the reversed target sequence
        target_match_region = target_rev_seq[i:i+max_len]
        
        # --- Check the 7 canonical site types in order of stringency (longer/more specific first) ---

        # 1. 8mer (p1-p8) WC
        # miRNA[0:8] vs Target_rev[i:i+8]
        if is_wc_match_aligned(miRNA_seq[:8], target_match_region[:8]):
            mmu_mimosa.loc[index, "canonical_site"] = 1
            # add matched seed site (start and end indices) to the dataframe
            mmu_mimosa.loc[index, "seed start"] = i
            mmu_mimosa.loc[index, "seed end"] = i + 8
            break

        # 2. 8mer-A1: p2-p8 WC + 'A' at p1 (Target_rev[i] is 'A')
        # miRNA[1:8] vs Target_rev[i+1:i+8] AND Target_rev[i] == 'A'
        if (target_match_region[0] == 'A' and # Check 'A' at p1 (Target_rev[i])
            is_wc_match_aligned(miRNA_seq[1:8], target_match_region[1:8])):
            mmu_mimosa.loc[index, "canonical_site"] = 1
            # add matched seed site (start and end indices) to the dataframe
            mmu_mimosa.loc[index, "seed start"] = i + 1
            mmu_mimosa.loc[index, "seed end"] = i + 8
            break
            
        # 3. 7mer-m8: p2-p8 WC
        # miRNA[1:8] vs Target_rev[i+1:i+8] (This is structurally identical to 8mer-A1 but without the 'A' requirement)
        # We only need to check this if the 8mer match did not already classify it.
        # Note: If target_rev[i] is not 'A', an 8mer-A1 is not formed. This check handles the general p2-p8 WC.
        if is_wc_match_aligned(miRNA_seq[1:8], target_match_region[1:8]):
            mmu_mimosa.loc[index, "canonical_site"] = 1
            # add matched seed site (start and end indices) to the dataframe
            mmu_mimosa.loc[index, "seed start"] = i + 1
            mmu_mimosa.loc[index, "seed end"] = i + 8
            break

        # Check for 7-mer and 6-mer types starting at p1/p2/p3
        
        # We need to ensure we can check at least the shortest 6-mer
        if len(target_rev_seq) < i + 6:
            continue
            
        # 4. 7mer-A1: p2-p7 WC + 'A' at p1 (Target_rev[i] is 'A')
        # miRNA[1:7] vs Target_rev[i+1:i+7] AND Target_rev[i] == 'A'
        if (target_match_region[0] == 'A' and 
            is_wc_match_aligned(miRNA_seq[1:7], target_match_region[1:7])):
            mmu_mimosa.loc[index, "canonical_site"] = 1
            # add matched seed site (start and end indices) to the dataframe
            mmu_mimosa.loc[index, "seed start"] = i + 1
            mmu_mimosa.loc[index, "seed end"] = i + 7
            break

        # 5. 6mer (p1-p6) WC
        # miRNA[0:6] vs Target_rev[i:i+6]
        if is_wc_match_aligned(miRNA_seq[:6], target_match_region[:6]):
            mmu_mimosa.loc[index, "canonical_site"] = 1
            # add matched seed site (start and end indices) to the dataframe
            mmu_mimosa.loc[index, "seed start"] = i
            mmu_mimosa.loc[index, "seed end"] = i + 6
            break

        # 6. 6mer (p2-p7) WC
        # miRNA[1:7] vs Target_rev[i+1:i+7]
        if is_wc_match_aligned(miRNA_seq[1:7], target_match_region[1:7]):
            mmu_mimosa.loc[index, "canonical_site"] = 1
            # add matched seed site (start and end indices) to the dataframe
            mmu_mimosa.loc[index, "seed start"] = i + 1
            mmu_mimosa.loc[index, "seed end"] = i + 7
            break

        # 7. 6mer (p3-p8) WC
        # miRNA[2:8] vs Target_rev[i+2:i+8]
        if is_wc_match_aligned(miRNA_seq[2:8], target_match_region[2:8]):
            mmu_mimosa.loc[index, "canonical_site"] = 1
            # add matched seed site (start and end indices) to the dataframe
            mmu_mimosa.loc[index, "seed start"] = i + 2
            mmu_mimosa.loc[index, "seed end"] = i + 8
            break

            
# --- Final Count and Save ---
mmu_mimosa.to_csv(os.path.join(data_dir, "mmu_mimosa.csv"), index=False)
non_canonical_sites = mmu_mimosa[mmu_mimosa["canonical_site"] == 0]
# save non-canonical sites to a csv file
non_canonical_sites.to_csv(os.path.join(data_dir, "mmu_mimosa_non_canonical_sites.csv"), index=False)
print(f"Number of non-canonical sites (Final Fixed Logic): {len(non_canonical_sites)}")