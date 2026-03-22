"""
Pair degradome cleavage sites to TargetScan seed sites by matching
(transcript_id, miRNA_name) pairs. No genomic coordinate mapping needed.
"""
import argparse

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(
        description="Match (transcript_id, miRNA_name) pairs between degradome and TargetScan."
    )
    ap.add_argument("--degradome", required=True, help="Degradome TSV")
    ap.add_argument("--ts-sites", required=True, help="TargetScan seed sites table")
    ap.add_argument("--out", required=True, help="Output TSV")
    ap.add_argument("--transcript-col", default="transcript",
                    help="Degradome column for transcript ID")
    ap.add_argument("--cleave-col", default="cleaveLocus",
                    help="Degradome column with cleavage locus")
    ap.add_argument("--region-col", default="region",
                    help="Degradome region column (used to filter 3'UTR rows)")
    ap.add_argument("--mirnaid-col", default="miRNAid",
                    help="Degradome miRNA id column (MIMAT accession)")
    ap.add_argument("--mirname-col", default="miRNAname",
                    help="Degradome column holding mature miRNA name (e.g., hsa-let-7a-3p)")
    ap.add_argument("--utr-region-token", default="3'UTR",
                    help="Substring in region that indicates 3'UTR")
    ap.add_argument("--strip-transcript-version", action="store_true",
                    help="Drop .version suffix from ENST IDs before joining")
    return ap.parse_args()


def load_ts_sites(path, strip_version=False):
    """
    Load TargetScan sites table as a DataFrame with normalised column names.
    Returns DataFrame with columns: ts_tx, ts_mirna, site_start, site_end, site_type
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")

    name_map = {c.lower().strip(): c for c in df.columns}

    tx_col    = name_map.get("transcript id") or name_map.get("transcript_id") or name_map.get("transcript")
    start_col = name_map.get("utr_start") or name_map.get("site_start") or name_map.get("utr start")
    end_col   = name_map.get("utr_end")   or name_map.get("site_end")   or name_map.get("utr end")
    type_col  = name_map.get("site type") or name_map.get("site_type")  or name_map.get("type")
    mirna_col = name_map.get("mirna")

    if not (tx_col and mirna_col):
        raise ValueError(
            f"Missing 'Transcript ID' or 'miRNA' column in {path}. "
            f"Columns: {df.columns.tolist()}"
        )

    out = pd.DataFrame()
    out["ts_tx"]    = df[tx_col].astype(str).str.strip()
    out["ts_mirna"] = df[mirna_col].astype(str).str.strip()
    out["site_type"]  = df[type_col].astype(str).str.strip() if type_col else ""
    out["site_start"] = pd.to_numeric(df[start_col], errors="coerce").astype("Int64") if start_col else pd.NA
    out["site_end"]   = pd.to_numeric(df[end_col],   errors="coerce").astype("Int64") if end_col   else pd.NA

    if strip_version:
        out["ts_tx"] = out["ts_tx"].str.split(".").str[0]

    return out


def load_degradome_utr(path, tcol, ccol, rcol, mcol, ncol,
                       utr_token, strip_version=False):
    """
    Load degradome, keep only 3'UTR rows, and normalise the transcript ID column.
    Adds _degr_idx (0-based row index in the full file) for counting matches vs len(full).

    Returns:
        n_full: int — len(degradome) before 3'UTR filter
        df: filtered DataFrame (3'UTR rows only)
    """
    df = pd.read_csv(path, sep=None, engine="python")
    n_full = len(df)
    df["_degr_idx"] = range(n_full)

    for need in [tcol, ccol, rcol]:
        if need not in df.columns:
            raise ValueError(f"Missing column '{need}' in {path}. Columns={df.columns.tolist()}")

    df = df[df[rcol].astype(str).str.contains(utr_token, regex=False)].copy()

    df["_tx_norm"] = df[tcol].astype(str).str.strip()
    if strip_version:
        df["_tx_norm"] = df["_tx_norm"].str.split(".").str[0]

    if mcol not in df.columns:
        df[mcol] = ""
    if ncol not in df.columns:
        df[ncol] = ""

    df[ncol] = df[ncol].astype(str).str.strip()

    return n_full, df


def main():
    args = parse_args()

    # ── Load inputs ────────────────────────────────────────────────────────────
    ts = load_ts_sites(args.ts_sites, strip_version=args.strip_transcript_version)
    print(f"[TS sites]  {len(ts):>7,} rows  |  "
          f"{ts[['ts_tx','ts_mirna']].drop_duplicates().shape[0]:>7,} unique (tx, miRNA) pairs")

    n_degr_full, degr = load_degradome_utr(
        args.degradome,
        tcol=args.transcript_col,
        ccol=args.cleave_col,
        rcol=args.region_col,
        mcol=args.mirnaid_col,
        ncol=args.mirname_col,
        utr_token=args.utr_region_token,
        strip_version=args.strip_transcript_version,
    )
    degr_pairs = degr[["_tx_norm", args.mirname_col]].drop_duplicates()
    print(
        f"[Degradome] {n_degr_full:>7,} total rows in file  |  "
        f"{len(degr):>7,} 3'UTR rows  |  "
        f"{len(degr_pairs):>7,} unique (tx, miRNA) pairs"
    )

    # ── Transcript-level match ─────────────────────────────────────────────────
    # If version stripping is off, try a fallback: strip version only when the
    # full ID is absent from TS.
    if not args.strip_transcript_version:
        ts_tx_set = set(ts["ts_tx"])
        degr["_tx_norm"] = degr["_tx_norm"].apply(
            lambda x: x.split(".")[0] if x not in ts_tx_set and x.split(".")[0] in ts_tx_set else x
        )

    n_degr_tx   = degr["_tx_norm"].nunique()
    ts_tx_set   = set(ts["ts_tx"])
    tx_matched  = degr["_tx_norm"].isin(ts_tx_set)
    n_tx_match  = degr.loc[tx_matched, "_tx_norm"].nunique()
    tx_match_pct = 100 * n_tx_match / n_degr_tx if n_degr_tx else 0.0
    print(f"\n[tx match]  {n_tx_match} / {n_degr_tx} unique transcripts in degradome "
          f"found in TargetScan ({tx_match_pct:.1f}%)")

    # ── (tx, miRNA) pair match via inner join ──────────────────────────────────
    merged = degr.merge(
        ts,
        left_on=["_tx_norm", args.mirname_col],
        right_on=["ts_tx",   "ts_mirna"],
        how="inner",
    )

    n_degr_pairs  = len(degr_pairs)
    n_matched_pairs = merged[["_tx_norm", args.mirname_col]].drop_duplicates().shape[0]
    pair_match_pct  = 100 * n_matched_pairs / n_degr_pairs if n_degr_pairs else 0.0

    # Rows in full degradome file that matched (at least one TS site for same tx + miRNA)
    n_matched_rows = merged["_degr_idx"].nunique()
    full_dataset_pct = 100 * n_matched_rows / n_degr_full if n_degr_full else 0.0

    # ── Build output ───────────────────────────────────────────────────────────
    out_cols = {
        "transcript":       merged[args.transcript_col],
        "cleaveLocus":      merged[args.cleave_col],
        "miRNAid":          merged[args.mirnaid_col],
        "miRNAname":        merged[args.mirname_col],
        "ts_transcript_id": merged["ts_tx"],
        "site_start":       merged["site_start"],
        "site_end":         merged["site_end"],
        "site_type":        merged["site_type"],
    }
    out_df = pd.DataFrame(out_cols)
    out_df.to_csv(args.out, sep="\t", index=False)

    print(
        f"[pair match] {n_matched_pairs} / {n_degr_pairs} unique (tx, miRNA) pairs matched "
        f"({pair_match_pct:.1f}%)\n"
        f"[full file]  {n_matched_rows:,} / {n_degr_full:,} degradome rows matched "
        f"({full_dataset_pct:.1f}% of entire dataset)\n"
        f"[output]     {len(out_df):,} rows written -> {args.out}"
    )


if __name__ == "__main__":
    main()
