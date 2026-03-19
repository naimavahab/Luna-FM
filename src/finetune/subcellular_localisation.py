# Example:
# python finetune_subcellular_multiclass_cls_optuna.py \
#   --dataset_path ../Data/subcellular_hf \
#   --model_name /home/ubuntu/Pretrain/RNA_FM_models_BPE2k/checkpoint-295520 \
#   --tokenizer_name /home/ubuntu/Pretrain/RNA_FM_models_BPE2k/checkpoint-295520 \
#   --task_name subcellular_localisation \
#   --max_length 1024 \
#   --seed 42 \
#   --mlp_hidden 512 \
#   --dropout 0.1 \
#   --tune_fraction 0.20 \
#   --n_trials 30 \
#   --early_stop_patience 2 \
#   --lr_min 5e-6 \
#   --lr_max 3e-4 \
#   --batch_choices 1 2 \
#   --epoch_min 3 \
#   --epoch_max 20 \
#   --grad_accum_choices 1 2 4 8 \
#   --map_num_proc 1 \
#   --study_dir ./results \
#   --fp16

import os
import shutil
import random
import argparse
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="subcellular_localisation")
    parser.add_argument("--dataset_path", type=str, default="../Data/subcellular_hf")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp_hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tune_fraction", type=float, default=0.20)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--lr_min", type=float, default=5e-6)
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--batch_choices", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epoch_min", type=int, default=3)
    parser.add_argument("--epoch_max", type=int, default=20)
    parser.add_argument("--grad_accum_choices", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--map_num_proc", type=int, default=1)
    parser.add_argument("--study_dir", type=str, default="./results")
    parser.add_argument("--study_storage", type=str, default=None)
    parser.add_argument("--direction", type=str, default="maximize")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        tgt_logp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        tgt_p = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - tgt_p).pow(self.gamma)
        loss = -focal_factor * tgt_logp

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(logits.device).gather(0, targets)
            else:
                alpha_t = torch.full_like(loss, float(self.alpha))
            loss = alpha_t * loss

        if self.ignore_index is not None and self.ignore_index >= 0:
            valid = targets != self.ignore_index
            loss = loss[valid]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiClassCLS(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_classes: int,
        mlp_hidden: int = 512,
        dropout: float = 0.1,
        focal_gamma: float = 2.0,
        focal_alpha=None,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config = config
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, n_classes),
        )

        self.criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="mean")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))

        loss = None
        if labels is not None:
            labels = labels.long().view(-1)
            loss = self.criterion(logits, labels)

        return {"loss": loss, "logits": logits}


def build_tokenizer(tokenizer_name):
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


def cast_label_to_int(examples):
    examples["labels"] = [int(x) for x in examples["label"]]
    return examples


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1).astype(int)
    labels = labels.astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "mcc": matthews_corrcoef(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def stratified_subsample(ds, frac, label_col="labels", seed=42):
    if label_col not in ds.column_names:
        n = max(1, int(len(ds) * frac))
        return ds.shuffle(seed=seed).select(range(n))

    labels = np.asarray(ds[label_col], dtype=np.int64)
    idx = np.arange(len(ds))
    rng = np.random.default_rng(seed)

    keep = []
    for y in np.unique(labels):
        y_idx = idx[labels == y]
        k = max(1, int(round(len(y_idx) * frac)))
        if k >= len(y_idx):
            chosen = y_idx
        else:
            chosen = rng.choice(y_idx, size=k, replace=False)
        keep.append(chosen)

    keep = np.concatenate(keep)
    keep = keep[rng.permutation(len(keep))]
    return ds.select(keep.tolist())


def prepare_data(args):
    print(f"Loading DatasetDict from disk: {args.dataset_path}")
    full = load_from_disk(args.dataset_path)

    missing = [k for k in ["train", "validation", "test"] if k not in full]
    if missing:
        raise ValueError(f"Expected splits train/validation/test, missing: {missing}. Found: {list(full.keys())}")

    train_labels_raw = full["train"]["label"]
    train_labels_int = [int(x) for x in train_labels_raw]
    n_classes = len(set(train_labels_int))
    print(f"Detected n_classes (from train): {n_classes}")

    tokenizer = build_tokenizer(args.tokenizer_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized = full.map(
        lambda x: tokenize_function(tokenizer, x, args.max_length),
        batched=True,
        num_proc=args.map_num_proc,
        remove_columns=["sequence"],
        load_from_cache_file=False,
        desc="Tokenizing",
    )

    tokenized = tokenized.map(
        cast_label_to_int,
        batched=True,
        num_proc=args.map_num_proc,
        load_from_cache_file=False,
        desc="Casting labels",
    )

    keep_cols = [c for c in ["input_ids", "attention_mask", "labels"] if c in tokenized["train"].column_names]
    tokenized.set_format(type="torch", columns=keep_cols)

    train_dataset = tokenized["train"]
    valid_dataset = tokenized["validation"]
    test_dataset = tokenized["test"]

    tune_train_dataset = stratified_subsample(
        train_dataset,
        frac=args.tune_fraction,
        label_col="labels",
        seed=args.seed
    )
    print(f"HPO train subset size: {len(tune_train_dataset)} / {len(train_dataset)} ({args.tune_fraction * 100:.1f}%)")

    return n_classes, data_collator, train_dataset, tune_train_dataset, valid_dataset, test_dataset


def make_training_args(args, output_dir, lr, bs, epochs, grad_accum, seed=None, save_total_limit=1):
    if seed is None:
        seed = args.seed

    return TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=min(bs, 16),
        gradient_accumulation_steps=grad_accum,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        fp16=args.fp16,
        report_to="none",
        save_total_limit=save_total_limit,
        dataloader_pin_memory=True,
    )


def objective(trial: optuna.Trial, args, n_classes, data_collator, tune_train_dataset, valid_dataset):
    set_seed(args.seed)

    lr = trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True)
    bs = trial.suggest_categorical("per_device_train_batch_size", args.batch_choices)
    epochs = trial.suggest_int("num_train_epochs", args.epoch_min, args.epoch_max)
    grad_accum = trial.suggest_categorical("gradient_accumulation_steps", args.grad_accum_choices)

    out_dir = os.path.join(args.study_dir, args.task_name, "optuna", f"trial_{trial.number}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    model = MultiClassCLS(
        args.model_name,
        n_classes=n_classes,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    )

    training_args = make_training_args(
        args,
        out_dir,
        lr=lr,
        bs=bs,
        epochs=epochs,
        grad_accum=grad_accum,
        save_total_limit=1,
    )

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

    n_classes, data_collator, train_dataset, tune_train_dataset, valid_dataset, test_dataset = prepare_data(args)

    study_name = f"optuna_{args.task_name}_cls"
    print(f"\nStarting Optuna study: {study_name} (trials={args.n_trials})")
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        direction=args.direction,
        study_name=study_name,
        sampler=sampler,
        storage=args.study_storage,
        load_if_exists=bool(args.study_storage),
    )

    study.optimize(
        lambda t: objective(t, args, n_classes, data_collator, tune_train_dataset, valid_dataset),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    print("\nBest trial:")
    print("  value (eval_f1_macro):", study.best_value)
    print("  params:", study.best_params)

    best = study.best_params

    final_out = os.path.join(args.study_dir, args.task_name, "best_run")
    if os.path.exists(final_out):
        shutil.rmtree(final_out, ignore_errors=True)
    os.makedirs(final_out, exist_ok=True)

    set_seed(args.seed)
    final_model = MultiClassCLS(
        args.model_name,
        n_classes=n_classes,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    )

    final_args = make_training_args(
        args,
        final_out,
        lr=float(best["learning_rate"]),
        bs=int(best["per_device_train_batch_size"]),
        epochs=int(best["num_train_epochs"]),
        grad_accum=int(best["gradient_accumulation_steps"]),
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

    print("\nTraining final model on 100% train with best hyperparameters...")
    final_trainer.train()

    print("\nFinal validation metrics (best checkpoint):")
    print(final_trainer.evaluate(eval_dataset=valid_dataset))

    print("\nTest metrics (once):")
    print(final_trainer.evaluate(eval_dataset=test_dataset))

    print("\nDone.")


if __name__ == "__main__":
    main()