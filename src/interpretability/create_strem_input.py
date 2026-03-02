import pandas as pd
from datasets import load_from_disk

SPANS_CSV = "ig_outputs_all/ig_per_sequence_top_spans_with_nt.csv"
DATA_DIR  = "lncrna_balanced_hf"

OUT_FORE = "foreground_rank1_pos.fa"
OUT_BACK = "background_rank1_neg.fa"

# load dataset labels
ds = load_from_disk(DATA_DIR)
print(ds[0:2])

base = ds["train"] if hasattr(ds, "keys") and "train" in ds else (ds if not hasattr(ds, "keys") else None)
if base is None:
    from datasets import concatenate_datasets
    base = concatenate_datasets([ds[k] for k in ds.keys()])

df_lab = base.to_pandas()
label_col = "label" if "label" in df_lab.columns else "labels"
id2lab = dict(zip(df_lab["id"], df_lab[label_col]))

# load spans
sp = pd.read_csv(SPANS_CSV)

# keep only rank 1
sp = sp[sp["rank"] == 1].copy()

# attach label
sp["label"] = sp["id"].map(id2lab)
sp = sp.dropna(subset=["label"])

pos = sp[sp["label"] == 1].copy()
neg = sp[sp["label"] == 0].copy()

print("Rank1 pos spans:", len(pos))
print("Rank1 neg spans:", len(neg))

def write_fasta(df, out_fa):
    with open(out_fa, "w") as f:
        for i, r in df.iterrows():
            hdr = f">{r['id']}|rank={r['rank']}|nt={r['nt_start']}-{r['nt_end']}|score={r['span_score']:.4f}"
            f.write(hdr + "\n")
            f.write(str(r["nt_span_seq"]) + "\n")
    print("Wrote:", out_fa)

write_fasta(pos, OUT_FORE)
write_fasta(neg, OUT_BACK)
