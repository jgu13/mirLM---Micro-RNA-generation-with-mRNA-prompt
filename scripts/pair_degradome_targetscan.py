import argparse, sys, re
from collections import defaultdict, namedtuple
import pandas as pd

UTRBlock = namedtuple("UTRBlock", ["chrom","start","end","strand"])

def parse_args():
    ap = argparse.ArgumentParser(description="Pair degradome cleavage sites to TargetScan seed sites in 3'UTR.")
    ap.add_argument("--degradome", required=True, help="Degradome TSV (hg19)")
    ap.add_argument("--ts-gff", required=True, help="TargetScan TS7 hg19 3'UTR blocks GFF")
    ap.add_argument("--ts-sites", required=True, help="TargetScan seed sites table (UTR 1-based coords)")
    ap.add_argument("--mir-family", default=None, help="miR_Family_Info.txt to map MIMAT -> miR_Family (optional)")
    ap.add_argument("--out", required=True, help="Output TSV")
    ap.add_argument("--transcript-col", default="transcript", help="Degradome column for transcript ID")
    ap.add_argument("--cleave-col", default="cleaveLocus", help="Degradome column with 'chr:pos:strand'")
    ap.add_argument("--region-col", default="region", help="Degradome region column (filter 3'UTR)")
    ap.add_argument("--mirnaid-col", default="miRNAid", help="Degradome miRNA id column (MIMAT)")
    ap.add_argument("--utr-region-token", default="3'UTR", help="Substring in region that indicates 3'UTR (e.g., 3'UTR)")
    ap.add_argument("--strip-transcript-version", action="store_true",
                    help="If set, drop .version from ENST in degrado & TS sites to join")
    ap.add_argument("--mirname-col", default="miRNAname",
                    help="Degradome column holding mature miRNA name (e.g., hsa-let-7a-3p)")
    ap.add_argument("--strict", action="store_true",
                    help="If set, drop rows with found_block=False or roundtrip_ok=False")
    return ap.parse_args()

def load_ts_utr_blocks(ts_gff, strip_version=False):
    """
    Returns:
      blocks_by_enst: dict ENST -> list[UTRBlock] (unsorted)
      strand_by_enst: dict ENST -> '+'/'-'
    """
    blocks_by_enst = defaultdict(list)
    strand_by_enst = {}
    with open(ts_gff, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#") or line.startswith("browser") or line.startswith("track"):
                continue
            # GFF columns: chrom, source, feature, start, end, score, strand, frame, attributes (here: transcript id)
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9: 
                continue
            chrom, src, feat, start, end, score, strand, frame, col9 = parts
            if feat.upper() != "UTR":
                continue
            try:
                start = int(start)
                end = int(end)
            except:
                continue
            enst = col9.strip()
            if strip_version:
                enst = enst.split(".")[0]
            blocks_by_enst[enst].append(UTRBlock(chrom, start, end, strand))
            strand_by_enst[enst] = strand
    return blocks_by_enst, strand_by_enst

def sort_blocks_transcript_order(blocks, strand):
    """Sort UTR blocks in the transcript 5'->3' direction."""
    if strand == '+':
        return sorted(blocks, key=lambda b: b.start)
    else:
        # minus strand: transcript 5'->3' runs from high->low genomic
        return sorted(blocks, key=lambda b: b.start, reverse=True)

def map_genome_to_utr(enst, chrom, pos, strand, blocks_by_enst):
    """
    Map genomic (hg19) cleavage pos to UTR 1-based position in TS space.
    Returns: (utr_pos_1based, total_utr_len, found_block, roundtrip_ok)
    """
    blocks = blocks_by_enst.get(enst, [])
    if not blocks:
        return (None, 0, False, False)
    # order in transcript direction
    blocks_ord = sort_blocks_transcript_order(blocks, blocks[0].strand)
    # total UTR len
    total_len = sum(b.end - b.start + 1 for b in blocks_ord)
    cum = 0
    utr_pos = None
    for b in blocks_ord:
        if b.chrom != chrom or b.strand != strand:
            cum += (b.end - b.start + 1)
            continue
        if b.start <= pos <= b.end:
            offset = (pos - b.start) if strand == '+' else (b.end - pos)
            utr_pos = cum + offset + 1  # 1-based
            break
        else:
            cum += (b.end - b.start + 1)
    if utr_pos is None:
        return (None, total_len, False, False)

    # round-trip check: map UTR pos back to genome
    gpos = utr_to_genome(utr_pos, blocks_ord, strand)
    roundtrip_ok = (gpos == pos)
    return (utr_pos, total_len, True, roundtrip_ok)

def utr_to_genome(utr_pos_1based, blocks_ord, strand):
    """Inverse mapping: TS UTR 1-based -> genomic coordinate."""
    remaining = int(utr_pos_1based) - 1  # to 0-based
    for b in blocks_ord:
        blen = b.end - b.start + 1
        if remaining < blen:
            # inside this block
            if strand == '+':
                return b.start + remaining
            else:
                return b.end - remaining
        remaining -= blen
    return None

def parse_cleavelocus(s):
    # "chr19:8027503:-"
    m = re.match(r'^(chr[^:]+):(\d+):([+-])$', str(s).strip())
    if not m:
        return (None, None, None)
    return m.group(1), int(m.group(2)), m.group(3)

def load_mir_family_map(path):
    """
    Parse TargetScan miR_Family_Info.txt variants.

    Expected columns (case/space insensitive):
      - 'miR family'            -> family label (e.g., 'let-7', 'miR-23-3p')
      - 'MiRBase Accession'     -> accession (e.g., 'MIMAT0000062')
      - 'MiRBase ID'            -> mature name (e.g., 'hsa-let-7a-3p')

    Returns:
      dict mapping:
        MIMAT (MiRBase Accession) -> miR family
      (If a row lacks accession, we additionally keep a secondary map
       from normalized MiRBase ID -> family and will merge those keys.)
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")

    # normalize header names (lowercase + squeeze spaces + strip)
    def norm(s): return s.lower().replace("\u00a0", " ").strip()
    colmap = {norm(c): c for c in df.columns}

    fam_col = colmap.get("miR family".lower()) or colmap.get("mir family") or colmap.get("family")
    acc_col = colmap.get("mirbase accession") or colmap.get("mimat") or colmap.get("accession")
    id_col  = colmap.get("mirbase id") or colmap.get("mature id") or colmap.get("mature_id") or colmap.get("mature")

    if not fam_col or (not acc_col and not id_col):
        raise ValueError(
            f"Could not find needed columns in {path}. "
            f"Have: {df.columns.tolist()} ; "
            f"Need at least 'miR family' and one of 'MiRBase Accession' or 'MiRBase ID'."
        )

    # Clean up strings
    df[fam_col] = df[fam_col].astype(str).str.strip()
    if acc_col:
        df[acc_col] = df[acc_col].astype(str).str.strip()
    if id_col:
        df[id_col]  = df[id_col].astype(str).str.strip()

    # Primary map: MIMAT accession -> family
    acc_ok = []
    if acc_col:
        acc_ok = df[[acc_col, fam_col]].dropna()
        acc_ok = acc_ok[acc_ok[acc_col].str.len() > 0]
    mimat_to_family = {row[acc_col]: row[fam_col] for _, row in acc_ok.iterrows()}

    # Secondary: MiRBase ID -> family (normalize by stripping species prefix like 'hsa-','mmu-','mml-','cfa-')
    def normalize_mirbase_id(x: str) -> str:
        x = x.strip()
        # strip leading species prefix (two/three letters + '-' commonly)
        # examples: hsa-let-7a-3p -> let-7a-3p
        parts = x.split('-', 1)
        if len(parts) == 2 and len(parts[0]) in (3, 4):  # crude but effective
            return parts[1]
        return x

    id_ok = []
    if id_col:
        id_ok = df[[id_col, fam_col]].dropna()
        id_ok = id_ok[id_ok[id_col].str.len() > 0]
    name_to_family = {normalize_mirbase_id(row[id_col]): row[fam_col] for _, row in id_ok.iterrows()}

    # Merge (MIMAT map is authoritative; name map is a fallback)
    # We still return a single dict for the pipeline: keys are MIMAT if present,
    # plus normalized IDs for rows lacking accession.
    for k, v in name_to_family.items():
        if k not in mimat_to_family:
            mimat_to_family[k] = v

    print(f"[miR-family] Loaded {len(mimat_to_family)} mappings "
          f"(MIMAT accessions + normalized MiRBase IDs) from {path}")
    return mimat_to_family

def load_ts_sites(ts_sites_path, strip_version=False):
    """
    Load TargetScan sites (UTR 1-based), pairing strictly by 'miRNA' exact match.
    Expected headers include:
      'Transcript ID','miRNA','UTR_start','UTR_end','Site Type', ...
    Returns: dict transcript_id -> list of dicts:
      {start, end, type, mirna_name}
    """
    try:
        df = pd.read_csv(ts_sites_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(ts_sites_path, sep="\t")

    name_map = {c.lower().strip(): c for c in df.columns}

    tx_col    = name_map.get("transcript id") or name_map.get("transcript_id") or name_map.get("transcript")
    start_col = name_map.get("utr_start") or name_map.get("site_start") or name_map.get("utr start")
    end_col   = name_map.get("utr_end")   or name_map.get("site_end")   or name_map.get("utr end")
    type_col  = name_map.get("site type") or name_map.get("site_type")  or name_map.get("type")
    mirna_col = name_map.get("mirna")

    if not (tx_col and start_col and end_col and mirna_col):
        raise ValueError(
            f"Missing one of Transcript ID / UTR_start / UTR_end / miRNA in {ts_sites_path}. "
            f"Columns={df.columns.tolist()}"
        )

    df["ts_transcript_id"] = df[tx_col].astype(str).str.strip()
    if strip_version:
        df["ts_transcript_id"] = df["ts_transcript_id"].str.split(".").str[0]
    df["utr_start_1b"] = pd.to_numeric(df[start_col], errors="coerce").astype("Int64")
    df["utr_end_1b"]   = pd.to_numeric(df[end_col],   errors="coerce").astype("Int64")
    df["site_type_str"]= df[type_col].astype(str).str.strip() if type_col else ""
    df["mirna_name"]   = df[mirna_col].astype(str).str.strip()

    df = df.dropna(subset=["utr_start_1b","utr_end_1b"])

    from collections import defaultdict
    sites_by_tx = defaultdict(list)
    for rec in df.to_dict(orient="records"):
        sites_by_tx[rec["ts_transcript_id"]].append({
            "start": int(rec["utr_start_1b"]),
            "end":   int(rec["utr_end_1b"]),
            "type":  rec["site_type_str"],
            "mirna_name": rec["mirna_name"],
        })

    for k in sites_by_tx:
        sites_by_tx[k].sort(key=lambda d: d["start"])
    return sites_by_tx

def nearest_seed(utr_pos, sites, family=None):
    """
    Find nearest seed site. If family is provided, filter by family first.
    Returns dict: {start,end,family,type,distance,overlap}
    """
    if not sites:
        return None
    cand = sites
    if family:
        famset = {family}
        cand = [s for s in sites if s["family"] in famset]
        if not cand:
            cand = sites  # fall back to any
    best = None
    for s in cand:
        # distance to interval
        if s["start"] <= utr_pos <= s["end"]:
            dist = 0
            overlap = True
        elif utr_pos < s["start"]:
            dist = s["start"] - utr_pos
            overlap = False
        else:
            dist = utr_pos - s["end"]
            overlap = False
        if (best is None) or (dist < best["distance"]):
            best = {
                "start": s["start"], "end": s["end"],
                "family": s["family"], "type": s["type"],
                "distance": dist, "overlap": overlap
            }
    return best

def main():
    args = parse_args()

    # Load TargetScan UTR blocks
    blocks_by_enst, strand_by_enst = load_ts_utr_blocks(args.ts_gff, strip_version=args.strip_transcript_version)

    # Load TS seed sites
    sites_by_tx = load_ts_sites(args.ts_sites, strip_version=args.strip_transcript_version)

    # Optional miR family map
    mirfam_map = {}
    if args.mir_family:
        mirfam_map = load_mir_family_map(args.mir_family)

    # Load degradome
    degr = pd.read_csv(args.degradome, sep='\t', engine="python")
    # normalize column names (keep originals for output)
    tcol = args.transcript_col
    ccol = args.cleave_col
    rcol = args.region_col
    mcol = args.mirnaid_col
    for need in [tcol, ccol, rcol]:
        if need not in degr.columns:
            raise ValueError(f"Missing column '{need}' in {args.degradome}. Columns={degr.columns.tolist()}")

    # Optionally strip transcript versions to join with TS if needed
    tx_norm = degr[tcol].astype(str)
    if args.strip_transcript_version:
        tx_norm = tx_norm.str.split(".").str[0]
    degr["_tx_norm"] = tx_norm

    out_rows = []
    for rec in degr.to_dict(orient="records"):
        tx = rec[tcol]
        tx_norm = rec["_tx_norm"]
        region = rec[rcol]
        cleave = rec[ccol]
        mirnaid = rec[mcol] if mcol in degr.columns else ""
        mirna_name = rec[args.mirname_col] if args.mirname_col in degr.columns else ""
        # Only map 3'UTR rows
        if not isinstance(region, str) or (args.utr_region_token not in region):
            continue

        chrom, pos, strand = parse_cleavelocus(cleave)
        if chrom is None and args.strict:
            continue
        elif chrom is None and not args.strict:
            out_rows.append({
                "transcript": tx,
                "cleaveLocus": cleave,
                "utr_pos_ts_1based": None,
                "total_utr_len": None,
                "found_block": False,
                "roundtrip_ok": False,
                "miRNAid": mirnaid,
                "miRNAname": mirna_name,
                "paired_seed_start": None,
                "paired_seed_end": None,
                "paired_seed_type": "",
                "nearest_seed_dist_nt": None,
                "overlaps_seed": None,
                "note": "bad_cleaveLocus_format"
            })
            continue

        # Map genome -> TS UTR
        enst_key = tx_norm if args.strip_transcript_version else tx
        if enst_key not in blocks_by_enst:
            # try without version if not already stripped
            if not args.strip_transcript_version and (tx.split(".")[0] in blocks_by_enst):
                enst_key = tx.split(".")[0]
            else:
                if args.strict:
                    continue
                else:
                    out_rows.append({
                        "transcript": tx,
                        "cleaveLocus": cleave,
                        "utr_pos_ts_1based": None,
                        "total_utr_len": None,
                        "found_block": False,
                        "roundtrip_ok": False,
                        "miRNAid": mirnaid,
                        "miRNAname": mirna_name,
                        "paired_seed_start": None,
                        "paired_seed_end": None,
                        "paired_seed_type": "",
                        "nearest_seed_dist_nt": None,
                        "overlaps_seed": None,
                        "note": "no_ts_utr_blocks_for_transcript"
                    })
                    continue

        utr_pos, total_len, found_block, roundtrip_ok = map_genome_to_utr(
            enst_key, chrom, pos, strand, blocks_by_enst
        )
        # STRICT FILTER: drop rows that are not clean mappings
        if args.strict and (not found_block or not roundtrip_ok):
            continue

        # sites on this transcript
        sites = sites_by_tx.get(enst_key, [])

        nearest = None
        if (utr_pos is not None) and sites:
            # STRICT NAME MATCH ONLY
            # exact string equality after .strip(); no normalization, no family fallback
            cand = [s for s in sites if s["mirna_name"] == mirna_name.strip()]
            if cand:
                # choose the closest site among exact-name matches
                best = None
                for s in cand:
                    if s["start"] <= utr_pos <= s["end"]:
                        dist, overlap = 0, True
                    elif utr_pos < s["start"]:
                        dist, overlap = s["start"] - utr_pos, False
                    else:
                        dist, overlap = utr_pos - s["end"], False
                    if (best is None) or (dist < best["distance"]):
                        best = {
                            "start": s["start"], "end": s["end"],
                            "type":  s.get("type",""),
                            "distance": dist, "overlap": overlap,
                            "mirna_name": s["mirna_name"]
                        }
                nearest = best
            else:
                # no exact-name site on that transcript -> leave unpaired
                nearest = None
                
        if args.strict and (nearest is None):
            continue
        # drop mapping that is too far away
        if nearest["distance"] > 5.0:
            continue
        else:
            out_rows.append({
                "transcript": tx,
                "cleaveLocus": cleave,
                "utr_pos_ts_1based": int(utr_pos) if utr_pos is not None else None,
                "total_utr_len": int(total_len) if total_len else None,
                "found_block": bool(found_block),
                "roundtrip_ok": bool(roundtrip_ok),
                "miRNAid": mirnaid,
                "miRNAname": mirna_name,
                "paired_seed_start": (nearest["start"] if nearest else None),
                "paired_seed_end":   (nearest["end"]   if nearest else None),
                "paired_seed_type":  (nearest["type"]  if nearest else ""),
                "nearest_seed_dist_nt": (nearest["distance"] if nearest else None),
                "overlaps_seed": (nearest["overlap"] if nearest else None),
                "note": "" if (found_block and roundtrip_ok) else ("no_block" if not found_block else "roundtrip_failed")
            })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"[OK] Wrote {len(out_df)} rows -> {args.out}")

if __name__ == "__main__":
    main()