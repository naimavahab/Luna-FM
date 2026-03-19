# filename: train_sentencepiece_unigram_from_corpus.py
# Train SentencePiece Unigram tokenizer from an EXISTING corpus.txt (huge corpus-safe).
#
# Key fixes for std::bad_alloc on large corpora:
# - input_sentence_size: sample only N lines (e.g., 2M) instead of loading 17.5M
# - train_extremely_large_corpus=True: SentencePiece mode for very large corpora
# - max_sentence_length: cap very long RNAs (e.g., 2048)
# - hard_vocab_limit=False: avoid crash if exact vocab can't be met
# - sanity check + preview before training

import os
import argparse
from pathlib import Path
import sentencepiece as spm


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
        print("[WARN] Some lines contain characters outside A/C/G/U/N. Consider cleaning corpus.txt.")


def print_head(corpus_txt: str, n: int = 3):
    print("---- corpus preview ----")
    with open(corpus_txt, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(line.strip())
    print("-------------------------")


def train_sentencepiece_unigram(
    corpus_txt: str,
    out_dir: str,
    model_prefix: str,
    vocab_size: int,
    input_sentence_size: int,
    max_sentence_length: int,
    character_coverage: float = 1.0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / model_prefix)

    spm.SentencePieceTrainer.Train(
    input=corpus_txt,
    model_prefix=prefix,
    model_type="unigram",
    vocab_size=vocab_size,
    character_coverage=character_coverage,

    hard_vocab_limit=False,
    train_extremely_large_corpus=True,
    input_sentence_size=input_sentence_size,
    shuffle_input_sentence=True,

    # Train tokenizer in the same length regime as your task
    max_sentence_length=512,

    # IMPORTANT for no-space RNA lines
    split_by_whitespace=True ,#False,

    # HARD CAP token length (prevents “too long pieces”)
    max_sentencepiece_length=12, #8,

    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
)
    print('split_by_whitespace=True and  max_sentencepiece_length=12')

    return prefix + ".model", prefix + ".vocab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Path to existing corpus.txt (one sequence per line)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--model_prefix", default="rna_unigram", help="Output prefix (without extension)")
    ap.add_argument("--vocab_size", type=int, default=8000, help="Vocab size (recommend 6k–10k for RNA)")
    ap.add_argument("--input_sentence_size", type=int, default=2000000,
                    help="How many lines to sample from corpus (2M is safe for 64GB)")
    ap.add_argument("--max_sentence_length", type=int, default=2048,
                    help="Cap sequence length used by SentencePiece")
    ap.add_argument("--preview_lines", type=int, default=3, help="How many corpus lines to print as preview")
    args = ap.parse_args()

    print("[1/3] Checking corpus...")
    sanity_check_corpus(args.corpus)
    print_head(args.corpus, n=args.preview_lines)

    print("[2/3] Training SentencePiece Unigram...")
    try:
        model_path, vocab_path = train_sentencepiece_unigram(
            corpus_txt=args.corpus,
            out_dir=args.out_dir,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            input_sentence_size=args.input_sentence_size,
            max_sentence_length=args.max_sentence_length,
        )
    except Exception as e:
        print("\nSentencePiece failed:")
        print(repr(e))
        print("\nTry these safer settings:")
        print("  --vocab_size 6000 --input_sentence_size 1000000 --max_sentence_length 1024")
        raise

    print("[3/3] Done ✓")
    print("Saved:")
    print(" ", model_path)
    print(" ", vocab_path)


if __name__ == "__main__":
    main()
