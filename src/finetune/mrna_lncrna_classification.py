# Example:
# python finetune_mrna_lncrna_binary_cls_optuna.py \
#   --dataset_path mrna_lncrna_binary \
#   --model_name /path/to/model \
#   --tokenizer_name /path/to/tokenizer \
#   --task_name mrna_lncrna \
#   --max_length 256 \
#   --seed 42 \
#   --tune_fraction 0.20 \
#   --early_stop_patience 2 \
#   --map_num_proc 10 \
#   --n_trials 20 \
#   --lr_min 1e-6 \
#   --lr_max 5e-4 \
#   --batch_choices 4 \
#   --epoch_min 2 \
#   --epoch_max 20 \
#   --weight_decay 0.01 \
#   --warmup_ratio 0.1 \
#   --dropout 0.1 \
#   --results_dir ./results

import os
import shutil
import random
import argparse
import time
import numpy as np
import optuna
import torch
import torch.nn as nn

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
    AutoModel,
)

from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="mrna_lncrna_binary")
    parser.add_argument("--task_name", type=str, default="mrna_lncrna")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune_fraction", type=float, default=0.20)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--map_num_proc", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_storage", type=str, default=None)
    parser.add_argument("--direction", type=str, default="maximize")
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--lr_max", type=float, default=5e-4)
    parser.add_argument("--batch_choices", type=int, nargs="+", default=[4])
    parser.add_argument("--epoch_min", type=int, default=2)
    parser.add_argument("--epoch_max", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--results_dir", type=str, default="./results")
    return parser.parse_args()


class BinaryCLSHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, cls_vec: torch.Tensor) -> torch.Tensor:
        return self.net(cls_vec).squeeze(-1)


class RiNALMoBinaryCLS(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config = config
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = self.backbone.config.hidden_size

        self.classifier = BinaryCLSHead(hidden_size, dropout=dropout)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            labels = labels.float().view(-1)
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def build_tokenizer(tokenizer_name: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tok is None:
        raise ValueError("Tokenizer failed to load.")
    return tok


def tokenize_function(tokenizer, examples, max_length):
    return tokenizer(
        examples["sequence"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def cast_labels_binary_float(examples):
    examples["labels"] = [float(int(x)) for x in examples["label"]]
    return examples


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "mcc": matthews_corrcoef(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def convert_T_to_U_in_column(ds, seq_col="sequence", map_num_proc=1):
    if seq_col not in ds.column_names:
        raise ValueError(f"'{seq_col}' not found. Columns: {ds.column_names}")

    def _map_fn(examples):
        examples[seq_col] = [
            s.replace("T", "U").replace("t", "u") for s in examples[seq_col]
        ]
        return examples

    return ds.map(
        _map_fn,
        batched=True,
        num_proc=map_num_proc,
        load_from_cache_file=False,
        desc=f"Converting T->U in {seq_col}",
    )


def stratified_subsample(ds, frac, label_col="label", seed=42):
    rng = np.random.default_rng(seed)

    if label_col in ds.column_names:
        col = label_col
    elif "labels" in ds.column_names:
        col = "labels"
    else:
        n = max(1, int(len(ds) * frac))
        return ds.shuffle(seed=seed).select(range(n))

    y_list = ds[col]
    try:
        y = np.array(list(y_list), dtype=np.int64)
    except TypeError:
        y = np.array([int(v) for v in y_list], dtype=np.int64)

    idx = np.arange(len(ds))
    keep = []

    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        k = max(1, int(len(cls_idx) * frac))
        keep.append(rng.choice(cls_idx, size=k, replace=False))

    keep = np.concatenate(keep)
    keep = keep[rng.permutation(len(keep))]
    return ds.select(keep.tolist())


def prepare_data(args):
    print("Loading dataset from disk...")
    full = load_from_disk(args.dataset_path)

    for split_name in full.keys():
        if "sequence" in full[split_name].column_names:
            full[split_name] = convert_T_to_U_in_column(
                full[split_name],
                seq_col="sequence",
                map_num_proc=args.map_num_proc
            )

    if "train" not in full or "validation" not in full:
        raise ValueError("Expected DatasetDict with at least 'train' and 'validation' splits.")

    print("Loading tokenizer...")
    tokenizer = build_tokenizer(args.tokenizer_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Tokenizing...")
    tokenized = full.map(
        lambda x: tokenize_function(tokenizer, x, args.max_length),
        batched=True,
        num_proc=args.map_num_proc,
        remove_columns=["sequence"],
        load_from_cache_file=False,
    )

    tokenized = tokenized.map(
        cast_labels_binary_float,
        batched=True,
        num_proc=args.map_num_proc,
        load_from_cache_file=False,
    )

    keep_cols = ["input_ids", "attention_mask", "labels"]
    tokenized.set_format(type="torch", columns=keep_cols)

    train_dataset = tokenized["train"]
    valid_dataset = tokenized["validation"]
    test_dataset = tokenized["test"] if "test" in tokenized else None

    tune_train_dataset = stratified_subsample(
        train_dataset,
        frac=args.tune_fraction,
        label_col="labels",
        seed=args.seed
    )

    print(f"HPO train subset size: {len(tune_train_dataset)} / {len(train_dataset)} ({args.tune_fraction * 100:.1f}%)")

    return tokenizer, data_collator, train_dataset, tune_train_dataset, valid_dataset, test_dataset


def make_args(args, output_dir, lr, bs, epochs, save_total_limit=1):
    return TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=min(bs, 32),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=save_total_limit,
    )


def objective(trial: optuna.Trial, args, data_collator, tune_train_dataset, valid_dataset):
    set_seed(args.seed)

    lr = trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True)
    bs = trial.suggest_categorical("per_device_train_batch_size", args.batch_choices)
    epochs = trial.suggest_int("num_train_epochs", args.epoch_min, args.epoch_max)

    out_dir = os.path.join(args.results_dir, args.task_name, "optuna", f"trial_{trial.number}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    model = RiNALMoBinaryCLS(args.model_name, dropout=args.dropout)
    training_args = make_args(args, out_dir, lr=lr, bs=bs, epochs=epochs, save_total_limit=1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tune_train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=valid_dataset)
    return float(metrics["eval_f1_macro"])


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    study_name = f"optuna_{args.task_name}_binary_cls"

    tokenizer, data_collator, train_dataset, tune_train_dataset, valid_dataset, test_dataset = prepare_data(args)

    print(f"Starting Optuna study: {study_name} (n_trials={args.n_trials})")
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        direction=args.direction,
        study_name=study_name,
        sampler=sampler,
        storage=args.study_storage,
        load_if_exists=bool(args.study_storage),
    )

    study.optimize(
        lambda t: objective(t, args, data_collator, tune_train_dataset, valid_dataset),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    print("\nBest trial:")
    print("  value (eval_f1_macro):", study.best_value)
    print("  params:", study.best_params)

    best = study.best_params
    final_out = os.path.join(args.results_dir, args.task_name, "best_run")
    if os.path.exists(final_out):
        shutil.rmtree(final_out, ignore_errors=True)
    os.makedirs(final_out, exist_ok=True)

    set_seed(args.seed)
    final_model = RiNALMoBinaryCLS(args.model_name, dropout=args.dropout)

    final_args = make_args(
        args,
        final_out,
        lr=float(best["learning_rate"]),
        bs=int(best["per_device_train_batch_size"]),
        epochs=int(best["num_train_epochs"]),
        save_total_limit=2,
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )

    print("\nTraining final model on 100% training data with best hyperparameters...")
    start = time.time()
    final_trainer.train()
    end = time.time()
    print("Total time", (end - start))

    print("\nFinal validation metrics (best checkpoint):")
    print(final_trainer.evaluate(eval_dataset=valid_dataset))

    if test_dataset is not None:
        print("\nTest metrics (evaluate once):")
        print(final_trainer.evaluate(eval_dataset=test_dataset))

    print("Done.")


if __name__ == "__main__":
    main()