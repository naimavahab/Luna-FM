# filename: ig_batch_export_v2.py
# pip install captum datasets transformers pandas numpy
# IG export for both pos and neg, with:
# (1) non-overlapping token spans per sequence
# (2) +/- 20 nt context around each span

import os
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BEST_DIR = "./preeclampsia_best_model"
DATA_DIR = "lncrna_balanced_hf"

MAX_LENGTH = 512
N_STEPS = 30
TOP_K_SPANS = 10
SPAN_WINDOW = 8
MAX_CLASS1 = 200
OUTDIR = "./ig_outputs_all"
os.makedirs(OUTDIR, exist_ok=True)

# NEW: context for motif discovery
NT_CONTEXT = 20

tokenizer = AutoTokenizer.from_pretrained(BEST_DIR, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    BEST_DIR, trust_remote_code=True
).to(DEVICE)
model.eval()

# ----------------------------
# Helpers
# ----------------------------
def norm_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")

def encode(seq: str):
    seq = norm_rna(seq)
    enc = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )
    enc.pop("token_type_ids", None)
    return {k: v.to(DEVICE) for k, v in enc.items()}

def offsets_for_seq(seq: str):
    """
    Returns (normalised_seq, offset_mapping) aligned with encode() settings.
    Requires a fast tokenizer.
    """
    seq = norm_rna(seq)
    enc = tokenizer(
        seq,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    return seq, enc["offset_mapping"]

def forward_func(input_ids, attention_mask=None):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits

def get_emb_layer():
    emb = model.get_input_embeddings()
    if emb is None:
        raise RuntimeError("model.get_input_embeddings() is None; cannot run IG.")
    return emb

def ig_for_seq(seq: str, target_class: int = 1):
    batch = encode(seq)
    input_ids = batch["input_ids"]
    attn_mask = batch.get("attention_mask", None)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids = torch.full_like(input_ids, fill_value=pad_id)

    lig = LayerIntegratedGradients(forward_func, get_emb_layer())

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attn_mask,),
        target=target_class,
        n_steps=N_STEPS,
        return_convergence_delta=True,
    )

    token_attr = attributions.squeeze(0).detach().cpu().numpy()   # (L, H)
    scores = np.linalg.norm(token_attr, axis=1)                   # (L,)
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0).detach().cpu().tolist()
    )

    with torch.no_grad():
        probs = torch.softmax(model(**batch).logits, dim=-1).squeeze(0).detach().cpu().numpy()

    return tokens, scores, probs, float(delta.detach().cpu().item())

def token_summary(tokens, scores):
    special = set(tokenizer.all_special_tokens)
    out = []
    for t, s in zip(tokens, scores):
        if t in special:
            continue
        out.append((t, float(s)))
    return out

def token_span_to_char_span(offsets, start_tok, end_tok):
    """
    Convert token span [start_tok, end_tok) -> char span [start_char, end_char)
    offsets: list[(start_char, end_char)] length = num_tokens
    """
    start_char = offsets[start_tok][0]
    end_char = offsets[end_tok - 1][1]
    return int(start_char), int(end_char)

def all_candidate_spans(tokens, scores, window=SPAN_WINDOW):
    """
    Generate ALL valid windowed spans with scores.
    """
    special = set(tokenizer.all_special_tokens)
    spans = []
    L = len(tokens)
    for i in range(0, L - window + 1):
        span_toks = tokens[i:i+window]
        if any(t in special for t in span_toks):
            continue
        span_score = float(np.sum(scores[i:i+window]))
        spans.append((span_score, i, i+window, span_toks))
    spans.sort(key=lambda x: x[0], reverse=True)
    return spans

def select_non_overlapping_spans(sorted_spans, k=TOP_K_SPANS):
    """
    NEW: Greedy selection of top spans ensuring no token index overlap.
    """
    chosen = []
    used = set()  # token indices already used
    for span_score, start_tok, end_tok, span_toks in sorted_spans:
        if any(ti in used for ti in range(start_tok, end_tok)):
            continue
        chosen.append((span_score, start_tok, end_tok, span_toks))
        used.update(range(start_tok, end_tok))
        if len(chosen) >= k:
            break
    return chosen

def add_nt_context(seq_norm, nt_start, nt_end, context=NT_CONTEXT):
    """
    NEW: Extend nucleotide span by +/- context, clipped to sequence bounds.
    """
    L = len(seq_norm)
    ctx_start = max(0, nt_start - context)
    ctx_end = min(L, nt_end + context)
    return ctx_start, ctx_end, seq_norm[ctx_start:ctx_end]


# -----------------------
# Load dataset + filter class 1 / 0
# -----------------------
ds = load_from_disk(DATA_DIR)
base = ds["train"] if hasattr(ds, "keys") and "train" in ds else (ds if not hasattr(ds, "keys") else None)

if base is None:
    from datasets import concatenate_datasets
    base = concatenate_datasets([ds[k] for k in ds.keys()])

df = base.to_pandas()

label_col = "label" if "label" in df.columns else ("labels" if "labels" in df.columns else None)
if label_col is None:
    raise ValueError(f"No label column found. Columns: {df.columns}")

df_pos = df[df[label_col] == 1].copy()
df_neg = df[df[label_col] == 0].copy()

MAX_PER_CLASS = 200
df_pos = df_pos.sample(n=min(MAX_PER_CLASS, len(df_pos)), random_state=42).reset_index(drop=True)
df_neg = df_neg.sample(n=min(MAX_PER_CLASS, len(df_neg)), random_state=42).reset_index(drop=True)

df_run = pd.concat([df_pos, df_neg], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
print("Running IG on:", len(df_run), "sequences  (pos:", len(df_pos), ", neg:", len(df_neg), ")")


# -----------------------
# Run IG for each sequence
# -----------------------
rows_spans = []
token_rows = []

for idx, r in df_run.iterrows():
    seq_id = r["id"]
    seq = r["sequence"]
    true_label = int(r[label_col])

    try:
        tokens, scores, probs, delta = ig_for_seq(seq, target_class=1)
        seq_norm, offsets = offsets_for_seq(seq)
    except Exception as e:
        print(f"[WARN] failed {seq_id}: {e}")
        continue

    # NEW: generate all spans -> pick non-overlapping top-K
    cand = all_candidate_spans(tokens, scores, window=SPAN_WINDOW)
    spans = select_non_overlapping_spans(cand, k=TOP_K_SPANS)

    for rank, (span_score, start_tok, end_tok, span_toks) in enumerate(spans, start=1):
        nt_start, nt_end = token_span_to_char_span(offsets, start_tok, end_tok)
        nt_seq = seq_norm[nt_start:nt_end]

        # NEW: +/- context
        ctx_nt_start, ctx_nt_end, nt_seq_ctx = add_nt_context(seq_norm, nt_start, nt_end, context=NT_CONTEXT)

        rows_spans.append({
            "id": seq_id,
            "rank": rank,
            "label": true_label,
            "p_class1": float(probs[1]) if len(probs) > 1 else float(probs[0]),
            "delta": delta,
            "span_score": span_score,

            # token-level indices
            "tok_start": start_tok,
            "tok_end": end_tok,
            "span_tokens": " ".join(span_toks),

            # nucleotide-level span
            "nt_start": nt_start,
            "nt_end": nt_end,
            "nt_span_seq": nt_seq,

            # NEW: nucleotide-level context span
            "ctx_nt_start": ctx_nt_start,
            "ctx_nt_end": ctx_nt_end,
            "nt_span_seq_ctx": nt_seq_ctx,
        })

    for t, s in token_summary(tokens, scores):
        token_rows.append({"id": seq_id, "token": t, "ig_score": s})

    if (idx + 1) % 10 == 0:
        print(f"Processed {idx+1}/{len(df_run)}")

# -----------------------
# Save per-sequence spans
# -----------------------
spans_df = pd.DataFrame(rows_spans)

# NEW filename to make it obvious it includes non-overlap + context
spans_path = os.path.join(OUTDIR, "ig_per_sequence_top_spans_nonoverlap_ctx20.csv")
spans_df.to_csv(spans_path, index=False)
print("Saved:", spans_path)

# -----------------------
# Aggregate tokens (unchanged)
# -----------------------
tok_df = pd.DataFrame(token_rows)

agg = (tok_df.groupby("token")
       .agg(
           n=("ig_score", "size"),
           mean_ig=("ig_score", "mean"),
           max_ig=("ig_score", "max"),
           sum_ig=("ig_score", "sum"),
       )
       .reset_index()
       .sort_values(["sum_ig", "mean_ig"], ascending=False))

tok_path = os.path.join(OUTDIR, "ig_token_enrichment.csv")
agg.to_csv(tok_path, index=False)
print("Saved:", tok_path)

long_path = os.path.join(OUTDIR, "ig_token_scores_long.csv")
tok_df.to_csv(long_path, index=False)
print("Saved:", long_path)
