# Example:
# python train_rna_mlm.py \
#   --fasta_file final.fasta \
#   --dataset_dir rna_Seqs \
#   --tokenizer_path tokenizer_bpe_2k_final \
#   --tokenized_dir tokenised_rna_BPE2k \
#   --output_dir ./RNA_FM_models_BPE2k \
#   --save_model_path Model_RNA_BPE2k \
#   --wandb_project RNA_Pretrain \
#   --wandb_name Pretraining_BPE \
#   --run_name RNA_modernBERT \
#   --max_len 128 \
#   --num_train_epochs 20 \
#   --per_device_train_batch_size 96 \
#   --gradient_accumulation_steps 2 \
#   --learning_rate 5e-5 \
#   --adam_beta1 0.9 \
#   --adam_beta2 0.98 \
#   --adam_epsilon 1e-6 \
#   --weight_decay 0.01 \
#   --warmup_ratio 0.06 \
#   --mlm_probability 0.20 \
#   --save_total_limit 20 \
#   --logging_steps 10 \
#   --logging_dir ./logs \
#   --fp16

import os
os.environ["PYTORCH_ENABLE_DYNAMO"] = "0"

import argparse
import time
import torch
import wandb

from Bio import SeqIO
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    ModernBertConfig,
    ModernBertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)

torch.cuda.empty_cache()

if torch.cuda.is_available():
    print("CUDA is available. Number of GPUs:", torch.cuda.device_count())
else:
    print("CUDA is not available. No GPUs detected.")

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file", type=str, default="final.fasta")
    parser.add_argument("--dataset_dir", type=str, default="rna_Seqs")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer_bpe_2k_final")
    parser.add_argument("--tokenized_dir", type=str, default="tokenised_rna_BPE2k")
    parser.add_argument("--output_dir", type=str, default="./RNA_FM_models_BPE2k")
    parser.add_argument("--save_model_path", type=str, default="Model_RNA_BPE2k")
    parser.add_argument("--wandb_project", type=str, default="RNA_Pretrain")
    parser.add_argument("--wandb_name", type=str, default="Pretraining_BPE")
    parser.add_argument("--run_name", type=str, default="RNA_modernBERT")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=96)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--mlm_probability", type=float, default=0.20)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def load_sp_tokenizer(sp_model_path: str):
    tok = AutoTokenizer.from_pretrained(sp_model_path)

    if tok.unk_token is None:
        tok.unk_token = "<unk>"
    if tok.pad_token is None:
        tok.pad_token = "<pad>"
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})

    special_add = {}
    if tok.cls_token is None:
        special_add["cls_token"] = "<cls>"
    if tok.sep_token is None:
        special_add["sep_token"] = "<sep>"
    if special_add:
        tok.add_special_tokens(special_add)

    for t in ["<pad>", "<cls>", "<sep>", "<mask>", "<unk>"]:
        print(t, tok.convert_tokens_to_ids(t))
    print("vocab_size:", tok.vocab_size)
    return tok


def load_fasta_biopython(fasta_path, name):
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append({"text": str(record.seq)})
    dataset = Dataset.from_list(sequences)
    dataset.save_to_disk(name)


def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    required_tokens = ["<pad>", "<cls>", "<sep>", "<mask>", "<unk>", "<s>", "</s>"]

    print("\nChecking special tokens:\n")
    for token in required_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id and token != tokenizer.unk_token:
            print(f"Missing token: {token}")
        else:
            print(f"{token} ID: {token_id}")
        print()

    return tokenizer


def tokenize_dataset(dataset, tokenizer, max_len=128):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000
    )

    print("Tokenization complete.\n")
    return tokenized


def compute_metrics(eval_preds: EvalPrediction):
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    ppl = torch.exp(loss).item() if loss.item() < 100 else float("inf")
    return {"perplexity": ppl}


def main():
    args = parse_args()
    start_time = time.time()

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        print("Logged into Weights & Biases.\n")

    if os.path.exists(args.dataset_dir):
        print("Data Folder exists")
        dataset = load_from_disk(args.dataset_dir)
    else:
        print("Data Folder does NOT exist, creating the data folder")
        load_fasta_biopython(args.fasta_file, args.dataset_dir)
        dataset = load_from_disk(args.dataset_dir)

    tokenizer = load_tokenizer(args.tokenizer_path)

    if os.path.exists(args.tokenized_dir):
        print("Tokenised Folder exists")
        tokenized_dataset = load_from_disk(args.tokenized_dir)
    else:
        print("Tokenised Folder does NOT exist, tokenising the data")
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_len=args.max_len)
        tokenized_dataset.save_to_disk(args.tokenized_dir)

    split = tokenized_dataset

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability
    )

    config = ModernBertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        reference_compile=False
    )

    model = ModernBertForMaskedLM(config=config)
    print("Model initialized.\n")
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        fp16=args.fp16,
        torch_compile=False,
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=args.run_name,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting training...\n")
    if hasattr(trainer.args, "_n_gpu"):
        print(f"Hugging Face Trainer is configured to use {trainer.args._n_gpu} GPUs.")
    else:
        print("Could not determine GPU usage from Trainer arguments directly.")

    trainer.train()
    trainer.save_model(args.save_model_path)

    print(f"Model saved to: {args.save_model_path}\n")

    end_time = time.time()
    total_time_sec = end_time - start_time
    hours = int(total_time_sec // 3600)
    minutes = int((total_time_sec % 3600) // 60)
    seconds = int(total_time_sec % 60)
    print(f"Total training time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()