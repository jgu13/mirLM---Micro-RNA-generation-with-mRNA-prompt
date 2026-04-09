"""
Build mRNA windows around degradome cleavage sites with randomized positioning.

Inputs:
  - Degradome TSV with at least: transcript, cleaveLocus, miRNAname, miRNAseq, 
      region, fdr, degraIDnum
      e.g., cleaveLocus like "chr10:90767475:+"
  - GENCODE v19 GTF (GRCh37/hg19)
  - transcripts FASTA (from: gffread GTF -g genome.fa -w transcripts.fa)
  - (optional) transcripts.fa.fai and samtools for fast random access

Filters applied:
  * Only 3'UTR regions
  * fdr <= 0.01
  * degraIDnum >= 2
  * Drops miRNA-transcript pairs with multiple peaks <50 nt apart

Output TSV columns:
  miRNAid   transcript   cleave_site   mrna_window   miRNAseq

Notes:
  * cleave_site is 0-based index within the window (randomized 0 to window_size-1).
  * mrna_window has length 30 by default; cleavage base position is randomized.
  * If the cleavage maps to intron (mismatch / bad row), the row is skipped.
"""

import sys, os, re, argparse, gzip, csv, shutil, subprocess
import numpy as np
from collections import defaultdict

def smart_open(path, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

def parse_gtf_attrs(attr):
    d = {}
    for k, v in re.findall(r'(\S+)\s+"([^"]+)"', attr):
        d[k] = v
    return d

def load_gtf_min(gtf_path):
    """Load only exon features (and strand/chrom) into a per-transcript model."""
    exons_by_tx = defaultdict(list)
    strand_by_tx = {}
    chrom_by_tx = {}
    base_to_full = {}

    with smart_open(gtf_path, "rt") as g:
        for line in g:
            if not line or line.startswith("#"): 
                continue
            toks = line.rstrip("\n").split("\t")
            if len(toks) < 9: 
                continue
            chrom, source, feature, start, end, score, strand, frame, attr = toks
            if feature != "exon":
                continue
            a = parse_gtf_attrs(attr)
            tx = a.get("transcript_id")
            if not tx: 
                continue
            start0 = int(start) - 1   # 0-based inclusive
            end0   = int(end)         # half-open

            exons_by_tx[tx].append((start0, end0))
            # setonce
            if tx not in strand_by_tx:
                strand_by_tx[tx] = strand
                chrom_by_tx[tx] = chrom
    # order exons per transcript in transcript order (5'->3')
    tx_models = {}
    for tx, exs in exons_by_tx.items():
        strand = strand_by_tx[tx]
        # genomic sort
        exs_sorted = sorted(exs, key=lambda x: (x[0], x[1]))
        if strand == "-":
            exs_sorted = list(reversed(exs_sorted))
        # cumulative starts in transcript coords
        exon_lens = [e - s for s, e in exs_sorted]
        cum = [0]
        for L in exon_lens[:-1]:
            cum.append(cum[-1] + L)
        tx_models[tx] = {
            "chrom": chrom_by_tx[tx],
            "strand": strand,
            "exons_tx_order": exs_sorted,
            "cum": cum,
            "exon_lens": exon_lens
        }
        base_to_full.setdefault(tx.split(".")[0], tx)
    return tx_models, base_to_full

def harmonize_chrom_names(ch_gtf, ch_tsv):
    """
    Make chrom styles comparable:
      - if one starts with 'chr' and the other doesn't, strip/add for compare only.
    Returns (norm_gtf, norm_tsv)
    """
    def norm(c):
        if c.startswith("chr"):
            return c[3:]
        return c
    return norm(ch_gtf), norm(ch_tsv)

def genome_to_tx_index(tx_model, chrom, pos_1based, strand):
    """Map genomic 1-based position to 0-based transcript index using exon order."""
    if tx_model is None:
        return None, "no_tx_model"

    # Chrom/strand consistency check (with tolerant name harmonization)
    gtf_ch = tx_model["chrom"]; gtf_st = tx_model["strand"]
    n1, n2 = harmonize_chrom_names(gtf_ch, chrom)
    if n1 != n2:
        return None, "chrom_mismatch"
    if gtf_st != strand:
        return None, "strand_mismatch"

    pos0 = pos_1based - 1
    exs = tx_model["exons_tx_order"]
    cum = tx_model["cum"]
    tx_strand = tx_model["strand"]

    # walk exons in transcript order; compute offset within the exon
    for i, (s0, e0) in enumerate(exs):
        if s0 <= pos0 < e0:
            if tx_strand == "+":
                off = pos0 - s0
            else:
                off = (e0 - 1) - pos0
            return cum[i] + off, None
    return None, "intronic_or_uncovered"

def parse_degradome_row(line, header_map):
    """Extract needed fields from the degradome TSV row."""
    transcript = line[header_map["transcript"]]
    cleaveLocus = line[header_map["cleaveLocus"]]  # "chr:pos:strand"
    miRNAid = line[header_map.get("miRNAname","miRNAname")] if "miRNAname" in header_map else ""
    miRNAseq = line[header_map.get("miRNAseq","miRNAseq")] if "miRNAseq" in header_map else ""
    # parse cleaveLocus
    try:
        chrom, pos, strand = cleaveLocus.split(":")
        pos = int(pos)
    except Exception:
        return None, "bad_cleaveLocus", None, None, None
    return transcript, None, chrom, pos, strand, miRNAid, miRNAseq

def read_fai_lengths(fai_path):
    """Parse .fai to get transcript lengths."""
    lens = {}
    if not os.path.exists(fai_path):
        return lens
    with open(fai_path, "rt") as f:
        for line in f:
            if not line.strip(): 
                continue
            toks = line.rstrip("\n").split("\t")
            # fai: name, length, offset, line_blen, line_len
            name = toks[0]; length = int(toks[1])
            lens[name] = length
    return lens

def fetch_fasta_slice_with_samtools(fasta, name, start0, end0):
    """
    Use `samtools faidx <fasta> name:start-end` to fetch a slice.
    start0/end0 are 0-based half-open; samtools expects 1-based inclusive.
    """
    start1 = start0 + 1
    end1   = end0
    region = f"{name}:{start1}-{end1}"
    try:
        out = subprocess.check_output(["samtools", "faidx", fasta, region], text=True)
    except Exception as e:
        return None
    seq = "".join([ln.strip() for ln in out.splitlines() if not ln.startswith(">")]).upper()
    return seq

def stream_load_needed_transcripts(fasta_path, needed_ids):
    """
    Scan the transcripts FASTA once and load sequences for IDs in needed_ids.
    Headers may contain version suffix; we accept exact and versionless matches.
    """
    seqs = {}
    need_base = {tid.split(".")[0] for tid in needed_ids}
    current_id = None
    keep = False
    chunks = []
    with smart_open(fasta_path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                # commit previous
                if current_id is not None and keep:
                    seqs[current_id] = "".join(chunks).upper()
                # new header
                head = line[1:].strip().split()[0]
                base = head.split(".")[0]
                keep = (head in needed_ids) or (base in need_base)
                current_id = head if keep else None
                chunks = []
            else:
                if keep:
                    chunks.append(line.strip())
        # commit last
        if current_id is not None and keep:
            seqs[current_id] = "".join(chunks).upper()
    return seqs

def main():
    ap = argparse.ArgumentParser(description="Map degradome loci to transcript coords and extract 500-nt windows.")
    ap.add_argument("--gtf", required=True, help="GENCODE v19 GTF (hg19/GRCh37)")
    ap.add_argument("--transcripts-fa", required=True, help="transcripts FASTA from gffread -w")
    ap.add_argument("--degradome-tsv", required=True, help="input degradome TSV")
    ap.add_argument("--out-tsv", required=True, help="output TSV path")
    ap.add_argument("--samtools", type=str, required=False,default=None, help="use samtools faidx to fetch sequences")
    ap.add_argument("--window", type=int, default=30, help="window length (default 30)")
    ap.add_argument("--skip-header", action="store_true", help="set if degradome TSV has no header row")
    ap.add_argument("--report-every", type=int, default=50000, help="progress report frequency")
    args = ap.parse_args()

    # 1) Load transcript models (exons, strand, chrom)
    sys.stderr.write("[info] loading GTF exon models...\n")
    tx_models, base_to_full = load_gtf_min(args.gtf)
    sys.stderr.write(f"[info] transcripts in GTF: {len(tx_models)}\n")

    # 2) First pass: filter degradome rows and collect data
    sys.stderr.write("[info] scanning degradome and applying filters (3'UTR, fdr<=0.01, degraIDnum>=2)...\n")
    needed = set()
    filtered_rows = []
    peak_data = []  # For close peak filtering
    n_total = 0
    n_region_filter = 0
    n_fdr_filter = 0
    n_degra_filter = 0
    
    with smart_open(args.degradome_tsv, "rt") as fin:
        reader = csv.reader(fin, delimiter="\t")
        header = next(reader)
        header_map = {h: i for i, h in enumerate(header)}
        if args.skip_header:
            sys.stderr.write("[warn] --skip-header given; expecting named columns present.\n")
        
        for row in reader:
            n_total += 1
            if not row or len(row) < 1:
                continue
            
            # Apply filters
            region = row[header_map.get("region", "")] if "region" in header_map else ""
            if region != "3'UTR":
                n_region_filter += 1
                continue
            
            fdr_val = float(row[header_map.get("fdr", "1.0")]) if "fdr" in header_map else 1.0
            if fdr_val > 0.01:
                n_fdr_filter += 1
                continue
            
            degra_num = int(row[header_map.get("degraIDnum", "0")]) if "degraIDnum" in header_map else 0
            if degra_num < 2:
                n_degra_filter += 1
                continue
            
            raw_tx = row[header_map["transcript"]]
            tx_key = raw_tx
            if tx_key not in tx_models:
                tx_key = base_to_full.get(raw_tx.split(".")[0])
            if tx_key is None or tx_key not in tx_models:
                continue
            
            needed.add(tx_key)
            filtered_rows.append(row)
            
            # Extract position for close peak filtering
            miRNAid = row[header_map.get("miRNAname", "")] if "miRNAname" in header_map else ""
            cleaveLocus = row[header_map["cleaveLocus"]]
            try:
                chrom, pos_str, strand = cleaveLocus.split(":")
                pos1 = int(pos_str)
                peak_data.append((miRNAid, tx_key, pos1, row))
            except:
                pass

    sys.stderr.write(f"[info] total rows: {n_total}\n")
    sys.stderr.write(f"[info] filtered by region (not 3'UTR): {n_region_filter}\n")
    sys.stderr.write(f"[info] filtered by fdr (>0.01): {n_fdr_filter}\n")
    sys.stderr.write(f"[info] filtered by degraIDnum (<2): {n_degra_filter}\n")
    sys.stderr.write(f"[info] rows passing initial filters: {len(filtered_rows)}\n")
    
    # Final filtered set
    final_rows = []
    for row in filtered_rows:
        miRNAid = row[header_map.get("miRNAname", "")] if "miRNAname" in header_map else ""
        raw_tx = row[header_map["transcript"]]
        tx_key = raw_tx if raw_tx in tx_models else base_to_full.get(raw_tx.split(".")[0])
        final_rows.append(row)
    
    sys.stderr.write(f"[info] final rows after all filters: {len(final_rows)}\n")
    sys.stderr.write(f"[info] unique transcripts needed: {len(needed)}\n")

    # 3) Prepare sequence access
    tx_lengths = {}
    tx_seqs = {}
    if args.samtools is not None:
        sys.stderr.write("[info] using samtools faidx on transcripts.fa for on-demand slices\n")
        tx_lengths = read_fai_lengths(args.samtools)
    else:
        sys.stderr.write("[info] samtools/faidx not available; loading needed transcripts into memory...\n")
        tx_seqs = stream_load_needed_transcripts(args.transcripts_fa, needed)
        for k, v in tx_seqs.items():
            tx_lengths[k] = len(v)
        # also map versionless to versioned if only one present
        base_map = {}
        for tid in needed:
            if tid not in tx_seqs:
                base = tid.split(".")[0]
                # find any key that matches base
                for k in tx_seqs:
                    if k.split(".")[0] == base:
                        tx_seqs[tid] = tx_seqs[k]
                        tx_lengths[tid] = len(tx_seqs[k])
                        break

    # 4) Process filtered degradome rows and write output
    n_in = 0
    n_ok = 0
    n_skip = 0
    with open(args.out_tsv, "wt", newline="") as fout:
        w = csv.writer(fout, delimiter="\t")
        w.writerow(["miRNA", "Transcript_ID", "cleave_site", "mRNA sequence", "miRNA sequence"])

        for row in final_rows:
            n_in += 1
            # parse required fields
            transcript = row[header_map["transcript"]]
            cleaveLocus = row[header_map["cleaveLocus"]]
            miRNAid = row[header_map.get("miRNAname","miRNAname")] if "miRNAname" in header_map else ""
            miRNAseq = row[header_map.get("miRNAseq","miRNAseq")] if "miRNAseq" in header_map else ""
            tx_key = base_to_full.get(transcript.split(".")[0])
            miRNAseq = miRNAseq.replace("-", "")

            try:
                chrom, pos_str, strand = cleaveLocus.split(":")
                pos1 = int(pos_str)
            except Exception:
                n_skip += 1
                continue

            tx_model = tx_models.get(tx_key)
            if tx_model is None:
                # try versionless
                base = tx_key.split(".")[0]
                tx_model = tx_models.get(base)
                if tx_model is None:
                    n_skip += 1
                    continue

            tx_idx, err = genome_to_tx_index(tx_model, chrom, pos1, strand)
            if tx_idx is None:
                n_skip += 1
                continue

            # transcript length
            tlen = tx_lengths.get(tx_key)
            if tlen is None:
                # try versionless
                tlen = tx_lengths.get(tx_key.split(".")[0])
            if tlen is None:
                # last resort: if we loaded all seqs, try getting it
                if tx_key in tx_seqs:
                    tlen = len(tx_seqs[tx_key])
                elif tx_key.split(".")[0] in tx_seqs:
                    tlen = len(tx_seqs[tx_key.split(".")[0]])
                else:
                    n_skip += 1
                    continue
                
            if tlen <= 0:
                n_skip += 1
                continue
            if tlen < args.window:
                cleave_site = tx_idx -1 # cleave site is 0-based index
                w.writerow([miRNAid, tx_key, cleave_site, tx_seqs[tx_key], miRNAseq])
                n_ok += 1
            else:  
                # Place window around cleave site with random positioning
                # Cleave site can be at any position from 0 to (window-1) within the window
                cleave_site_in_window = np.random.randint(0, args.window)
                
                # Calculate window bounds
                win_start = tx_idx - cleave_site_in_window -1 # cleave site is 0-based index
                win_end = win_start + args.window
                
                # Clamp to transcript bounds
                if win_start < 0:
                    win_start = 0
                    win_end = min(args.window, tlen)
                    cleave_site_in_window = tx_idx -1  # recalculate position in clamped window
                elif win_end > tlen:
                    win_end = tlen
                    win_start = max(0, tlen - args.window)
                    cleave_site_in_window = tx_idx - win_start -1  # recalculate position in clamped window
                
                # fetch sequence slice
                if args.samtools is not None:
                    # which name to use in FAI? prefer exact; else try versionless
                    tname = tx_key
                    if tname not in tx_lengths:
                        base = tx_key.split(".")[0]
                        if base in tx_lengths:
                            tname = base
                    slice_seq = fetch_fasta_slice_with_samtools(args.transcripts_fa, tname, win_start, win_end)
                    if slice_seq is None:
                        n_skip += 1
                        continue
                else:
                    # in-memory
                    s = tx_seqs.get(tx_key) or tx_seqs.get(tx_key.split(".")[0])
                    if s is None:
                        n_skip += 1
                        continue
                    slice_seq = s[win_start:win_end]

                # write row
                w.writerow([miRNAid, tx_key, cleave_site_in_window, slice_seq, miRNAseq])
                n_ok += 1

            if args.report_every and (n_in % args.report_every == 0):
                sys.stderr.write(f"[progress] processed={n_in} ok={n_ok} skipped={n_skip}\n")

    sys.stderr.write(f"[done] total rows={n_in}, ok={n_ok}, skipped={n_skip}\n")

if __name__ == "__main__":
    main()