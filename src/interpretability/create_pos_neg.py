# parse_table_s7_filter_lncrna_noncoding.py
# Usage:
#   python create_pos_neg.py \
#       --xlsx 12915_2024_1959_MOESM7_ESM.xlsx \
#       --sheet "Table S7" \
#       --outdir out_s7
#
# What it does:
# 1) reads "Table S7" from the Excel
# 2) keeps only genebiotype in {lncRNA, non_coding}
# 3) makes "pos" (padj<0.05 & |log2FoldChange|>1) and "neg" (others)
# 4) saves CSVs (+ optional FASTA if you later add sequences)

import argparse
import os
import re
import pandas as pd

def _normalise_col(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"\s+", " ", c)
    return c

def _coerce_numeric(s: pd.Series) -> pd.Series:
    # handles "3.18219E-37" and other scientific notation strings
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")

def load_sheet_any_header(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Some supplementary Excels have a few title rows before the real header.
    This tries header=0..30 and picks the first that contains the key columns.
    """
    required = {"StringTie_geneID", "genebiotype", "log2FoldChange", "padj"}
    for header_row in range(0, 31):
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
        df.columns = [_normalise_col(c) for c in df.columns]
        cols = set(df.columns)
        if required.issubset(cols):
            return df
    raise ValueError(
        f"Couldn't find expected columns {sorted(required)} in sheet={sheet_name}. "
        "Try opening the Excel and confirm the header row / column names."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to the supplementary Excel file containing Table S7")
    ap.add_argument("--sheet", default="Table S7", help="Sheet name for Table S7 (default: 'Table S7')")
    ap.add_argument("--outdir", default="out_s7", help="Output directory")
    ap.add_argument("--padj", type=float, default=0.05, help="padj threshold (default 0.05)")
    ap.add_argument("--logfc", type=float, default=1.0, help="abs(log2FoldChange) threshold (default 1.0)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load Table S7
    df = load_sheet_any_header(args.xlsx, args.sheet)

    # 2) Clean / coerce key columns
    for col in ["StringTie_geneID", "ref_stable_geneid", "genename", "genebiotype"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["log2FoldChange"] = _coerce_numeric(df["log2FoldChange"])
    df["padj"] = _coerce_numeric(df["padj"])
    if "pvalue" in df.columns:
        df["pvalue"] = _coerce_numeric(df["pvalue"])

    # Drop rows missing essentials
    df = df.dropna(subset=["StringTie_geneID", "genebiotype", "log2FoldChange", "padj"]).copy()

    # 3) Filter ncRNAs: genebiotype in {lncRNA, non_coding}
    keep_biotypes = {"lncRNA", "non_coding"}
    nc = df[df["genebiotype"].isin(keep_biotypes)].copy()

    # 4) Create pos/neg labels based on thresholds
    pos_mask = (nc["padj"] < args.padj) & (nc["log2FoldChange"].abs() > args.logfc)
    pos = nc[pos_mask].copy()
    neg = nc[~pos_mask].copy()

    # Add explicit label column
    pos.insert(0, "label", 1)
    neg.insert(0, "label", 0)

    # 5) Save outputs
    nc.to_csv(os.path.join(args.outdir, "table_s7_ncRNAs_all.csv"), index=False)
    pos.to_csv(os.path.join(args.outdir, "table_s7_ncRNAs_pos.csv"), index=False)
    neg.to_csv(os.path.join(args.outdir, "table_s7_ncRNAs_neg.csv"), index=False)

    # Also save a compact manifest with just IDs + stats (handy for sequence fetching later)
    cols_wanted = [c for c in ["label", "StringTie_geneID", "ref_stable_geneid", "genename", "genebiotype",
                              "log2FoldChange", "pvalue", "padj"] if c in pos.columns]
    pd.concat([pos[cols_wanted], neg[cols_wanted]], axis=0)\
      .to_csv(os.path.join(args.outdir, "table_s7_ncRNAs_manifest.csv"), index=False)

    print(f"Saved:\n"
          f"  {args.outdir}/table_s7_ncRNAs_all.csv  (n={len(nc)})\n"
          f"  {args.outdir}/table_s7_ncRNAs_pos.csv  (n={len(pos)})\n"
          f"  {args.outdir}/table_s7_ncRNAs_neg.csv  (n={len(neg)})\n"
          f"  {args.outdir}/table_s7_ncRNAs_manifest.csv")

if __name__ == "__main__":
    main()
