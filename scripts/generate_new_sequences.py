"""
Generate random miRNA and mRNA sequences for scalability benchmarking.
- Fixed miRNA length: 22 nt (typical miRNA length)
- Variable mRNA lengths: 500, 1k, 2k, 4k, 6k, 8k, 10k, 25k
- 20 sequences per length
- Output: Tab-delimited file
"""

import random
import os

random.seed(42)

NUCLEOTIDES = ['A', 'C', 'G', 'T']
MIRNA_LENGTH = 24
MRNA_LENGTHS = [500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 25000]
SEQS_PER_LENGTH = 20


def generate_random_rna(length):
    return ''.join(random.choices(NUCLEOTIDES, k=length))


def main():
    output_path = f"TargetScan_dataset/benchmark_sequences.tsv"
    with open(output_path, 'w') as f:
        # Header
        f.write("id\tmirna_seq\tmrna_seq\tmrna_length\n")
        
        for mrna_len in MRNA_LENGTHS:
            for i in range(SEQS_PER_LENGTH):
                seq_id = f"mRNA_{mrna_len}nt_pair_{i+1}"
                mirna = generate_random_rna(MIRNA_LENGTH)
                mrna = generate_random_rna(mrna_len)
                f.write(f"{seq_id}\t{mirna}\t{mrna}\t{mrna_len}\n")

            print(f"Generated {SEQS_PER_LENGTH} pairs for mRNA length {mrna_len:>6} nt")
            print(f"  mRNA length:    {mrna_len} nt")
            print(f"  miRNA length:    {MIRNA_LENGTH} nt")

    total_pairs = len(MRNA_LENGTHS) * SEQS_PER_LENGTH
    print(f"\nSummary:")
    print(f"  Total pairs:     {total_pairs}")
    print(f"  Seqs per length: {SEQS_PER_LENGTH}")


if __name__ == "__main__":
    main()