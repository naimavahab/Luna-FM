import re
from datasets import load_from_disk
from datasets import concatenate_datasets

DATA_DIR = "lncrna_balanced_hf"
OUT_FA = "full_positive_sequences.fa"

# ----------------------------
# Load dataset
# ----------------------------
ds = load_from_disk(DATA_DIR)

# Handle DatasetDict vs Dataset
if hasattr(ds, "keys"):   # DatasetDict
    base = concatenate_datasets([ds[k] for k in ds.keys()])
else:
    base = ds

df = base.to_pandas()

# Detect label column automatically
label_col = None
for col in ["label", "labels"]:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    raise ValueError(f"No label column found. Columns: {df.columns}")

print("Using label column:", label_col)

# ----------------------------
# Filter positives
# ----------------------------
df_pos = df[df[label_col] == 1].copy()
print("Number of positive sequences:", len(df_pos))

# ----------------------------
# Clean RNA function
# ----------------------------
def clean_rna(seq: str) -> str:
    seq = str(seq).upper().replace("T", "U")
    seq = re.sub(r"[^ACGU]", "", seq)
    return seq

# ----------------------------
# Write FASTA
# ----------------------------
n_written = 0

with open(OUT_FA, "w", newline="\n") as f:
    for _, row in df_pos.iterrows():
        seq_id = row["id"]
        seq = clean_rna(row["sequence"])

        if len(seq) == 0:
            continue

        f.write(f">{seq_id}\n")

        # wrap at 60 characters per line
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

        n_written += 1

print(f"Wrote {n_written} sequences to {OUT_FA}")
