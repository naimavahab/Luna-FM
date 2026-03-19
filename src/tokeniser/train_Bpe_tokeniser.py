# Train HuggingFace-compatible BPE tokenizer (tokenizers.SentencePieceBPETokenizer)
# from an EXISTING corpus.txt (streaming, huge-corpus safe).

# nohup python train_Bpe_tokeniser.py   --corpus final.fasta   --out_dir ./tokenizer_bpe_4k_final   --vocab_size 4096   --model_length 2000 > log4k.log 2>&1 & 

import os
import argparse
from pathlib import Path
from typing import Iterator, Optional

from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]


def sanity_check_corpus(corpus_txt: str, min_lines: int = 200):
    if not os.path.exists(corpus_txt):
        raise FileNotFoundError(f"Corpus not found: {corpus_txt}")

    size = os.path.getsize(corpus_txt)
    if size == 0:
        raise RuntimeError("Corpus file is empty.")

    bad = 0
    n = 0
    with open(corpus_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            if any(c not in "ACGUN" for c in line):
                bad += 1
            if n >= min_lines:
                break

    if n == 0:
        raise RuntimeError("Corpus contains no valid non-empty lines.")

    print(f"[OK] corpus size={size/1e6:.2f}MB | sampled_lines={n} | bad_sampled={bad}")
    if bad > 0:
        print("[WARN] Some lines contain characters outside A/C/G/U/N.")


def print_head(corpus_txt: str, n: int = 3):
    print("---- corpus preview ----")
    with open(corpus_txt, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(line.strip())
    print("-------------------------")


def iter_corpus_lines(corpus_txt: str, limit: Optional[int] = None) -> Iterator[str]:
    """Yield non-empty lines (RNA sequences) from corpus.txt."""
    n = 0
    with open(corpus_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield s
            n += 1
            if limit is not None and n >= limit:
                break


def train_bpe_tokenizer_from_corpus(
    corpus_txt: str,
    out_dir: str,
    vocab_size: int = 4000,
    min_frequency: int = 2,
    model_length: int = 256,
    preview_lines: int = 3,
    iterator_limit: Optional[int] = None,  # set e.g. 5_000_000 for faster dev runs
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Checking corpus...")
    sanity_check_corpus(corpus_txt)
    print_head(corpus_txt, n=preview_lines)

    print("[2/3] Training tokenizers.SentencePieceBPETokenizer (BPE)...")
    tk_tokenizer = SentencePieceBPETokenizer()  # BPE

    # Stream corpus to avoid RAM blowups
    text_iter = iter_corpus_lines(corpus_txt, limit=iterator_limit)

    tk_tokenizer.train_from_iterator(
        iterator=text_iter,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
    )

    # Save tokenizer artifacts (tokenizers format)
    # - tokenizer.json is easiest for PreTrainedTokenizerFast
    tokenizer_json = out_dir / "tokenizer.json"
    tk_tokenizer.save(str(tokenizer_json))

    # Also save vocab.json + merges.txt (optional but often nice to keep)
    # This creates: <out_dir>/rna_bpe-vocab.json and <out_dir>/rna_bpe-merges.txt
    tk_tokenizer.save_model(str(out_dir), "rna_bpe")

    print("[3/3] Converting to HF PreTrainedTokenizerFast + saving save_pretrained...")
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        model_max_length=model_length,
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>",
    )

    # Save in HF format: tokenizer_config.json, special_tokens_map.json, tokenizer.json, etc.
    hf_tok.save_pretrained(str(out_dir))

    # sanity ids
    print("\nSaved HF tokenizer to:", str(out_dir))
    print("Sanity token ids:")
    for t in SPECIAL_TOKENS:
        print(f"  {t:>6} -> {hf_tok.convert_tokens_to_ids(t)}")
    print("\nYou can now load with:")
    print(f'  from transformers import AutoTokenizer')
    print(f'  tok = AutoTokenizer.from_pretrained("{out_dir}")')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Path to corpus.txt (1 sequence per line)")
    ap.add_argument("--out_dir", required=True, help="Where to save the HF tokenizer folder")
    ap.add_argument("--vocab_size", type=int, default=4000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--model_length", type=int, default=256)
    ap.add_argument("--preview_lines", type=int, default=3)
    ap.add_argument("--iterator_limit", type=int, default=None,
                    help="Optional: limit number of lines for quick tests")
    args = ap.parse_args()

    train_bpe_tokenizer_from_corpus(
        corpus_txt=args.corpus,
        out_dir=args.out_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        model_length=args.model_length,
        preview_lines=args.preview_lines,
        iterator_limit=args.iterator_limit,
    )


if __name__ == "__main__":
    main()