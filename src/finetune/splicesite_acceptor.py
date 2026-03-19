import os
import shutil
import random
import time
import numpy as np
import optuna

import torch
import torch.nn as nn

from datasets import load_dataset
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


# ----------------------------
# Config
# ----------------------------
MODEL_NAME =  "LunaFM" #replace model name here
TASK_NAME = "luna_acceptor"   
TASK_NAME_org = "splice_site_acceptor"                    # binary task from genbio-ai/rna-downstream-tasks
MAX_LENGTH = 256                                  # adjust if needed
SEED = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# --- HPO policy ---
TUNE_FRACTION = 0.15          
EARLY_STOP_PATIENCE = 2         # epochs without improvement
MAP_NUM_PROC = 10               # set lower if HPC FS is flaky

# Optuna config
N_TRIALS = 20
STUDY_NAME = f"optuna_{TASK_NAME}_binary_cls"
STUDY_STORAGE = None
DIRECTION = "maximize"

# HPO ranges
LR_RANGE = (1e-6, 5e-4)
BATCH_CHOICES = [4, 8, 16, 32]  # adjust for GPU memory
EPOCH_RANGE = (2, 20)

# Fixed training defaults
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.1


# ----------------------------
# Model: base backbone + CLS head
# ----------------------------
class CLSHead(nn.Module):
    """
    Base encoder -> take CLS token embedding -> MLP -> 1 logit.
    Uses BCEWithLogitsLoss.
    """
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config = config

        # Load base model (NOT sequence prediction head)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True,reference_compile=False)

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        cls = out.last_hidden_state[:, 0, :]     # (B, H)
        logits = self.classifier(cls).squeeze(-1)  # (B,)

        loss = None
        if labels is not None:
            labels = labels.float().view(-1)
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# ----------------------------
# Tokenization + label casting
# ----------------------------
def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,reference_compile=False)
    if tok is None:
        raise ValueError("Tokenizer failed to load.")
    return tok


def tokenize_function(tokenizer, examples):
    # RNA downstream task uses "sequences" column
    return tokenizer(
        examples["sequences"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # DataCollator handles padding
    )


def cast_labels_binary_float(examples):
    # Ensure labels are float {0,1} for BCEWithLogitsLoss
    examples["labels"] = [float(int(x)) for x in examples["labels"]]
    return examples


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    return {
        "f1": f1_score(labels, preds, average="binary", pos_label=1),
        "mcc": matthews_corrcoef(labels, preds),
        "precision": precision_score(labels, preds, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, average="binary", pos_label=1, zero_division=0),
    }


# ----------------------------
# Stratified subsample for HPO
# ----------------------------
def stratified_subsample(ds, frac, label_col="labels", seed=42):
    """
    Stratified subsample that works reliably with Hugging Face Datasets.
    """
    rng = np.random.default_rng(seed)

    if label_col not in ds.column_names:
        n = max(1, int(len(ds) * frac))
        return ds.shuffle(seed=seed).select(range(n))

    y_list = ds[label_col]
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

def convert_T_to_U_in_column(ds, seq_col="sequence"):
    """
    Replace T/t with U/u in a string sequence column.
    Works on a Hugging Face Dataset (single split).
    """
    if seq_col not in ds.column_names:
        raise ValueError(f"'{seq_col}' not found. Columns: {ds.column_names}")

    def _map_fn(examples):
        out = []
        for s in examples[seq_col]:
            s = s.replace("T", "U").replace("t", "u")
            out.append(s) #" ".join(list(s)))  # spaces between characters
        examples[seq_col] = out
#        examples[seq_col] = [
 #           s.replace("T", "U").replace("t", "u") for s in examples[seq_col]
  #      ]
   #     examples[seq_col] = ' '.join(examples[seq_col])
        return examples

    return ds.map(
        _map_fn,
        batched=True,
        num_proc=MAP_NUM_PROC,
        load_from_cache_file=False,   # safer on shared/HPC FS
        desc=f"Converting T->U in {seq_col}",
    )
# ----------------------------
# Data prep
# ----------------------------
def prepare_data():
    print("Loading dataset...")
    full = load_dataset("genbio-ai/rna-downstream-tasks", TASK_NAME_org)
    for split_name in full.keys():
        if "sequences" in full[split_name].column_names:
            full[split_name] = convert_T_to_U_in_column(full[split_name], seq_col="sequences")
    print(full['train'][0])

    if "train" not in full or "validation" not in full:
        raise ValueError("Expected splits at least: train, validation.")

    print("Loading tokenizer...")
    tokenizer = build_tokenizer(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Tokenizing...")
    tokenized = full.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
        num_proc=MAP_NUM_PROC,
        remove_columns=["sequences"],
        load_from_cache_file=False,
    )

    tokenized = tokenized.map(
        cast_labels_binary_float,
        batched=True,
        num_proc=MAP_NUM_PROC,
        load_from_cache_file=False,
    )

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = tokenized["train"]
    valid_dataset = tokenized["validation"]

    # Optional test sets in this dataset card
    eval_sets = {}
    for k in ["test", "test_danio", "test_fly", "test_thaliana", "test_worm"]:
        if k in tokenized:
            eval_sets[k] = tokenized[k]

    tune_train_dataset = stratified_subsample(train_dataset, frac=TUNE_FRACTION, label_col="labels", seed=SEED)
    print(f"HPO train subset size: {len(tune_train_dataset)} / {len(train_dataset)} ({TUNE_FRACTION*100:.1f}%)")

    return tokenizer, data_collator, train_dataset, tune_train_dataset, valid_dataset, eval_sets


# ----------------------------
# TrainingArguments builder
# ----------------------------
def make_args(output_dir, lr, bs, epochs, save_total_limit=1):
    return TrainingArguments(
        output_dir=output_dir,
        seed=SEED,

        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,

        per_device_train_batch_size=bs,
        per_device_eval_batch_size=min(bs, 32),

        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,

        fp16=torch.cuda.is_available(),
        report_to="none",

        save_total_limit=save_total_limit,
    )


# ----------------------------
# Optuna objective
# ----------------------------
def objective(trial: optuna.Trial, data_collator, tune_train_dataset, valid_dataset):
    set_seed(SEED)

    lr = trial.suggest_float("learning_rate", LR_RANGE[0], LR_RANGE[1], log=True)
    bs = trial.suggest_categorical("per_device_train_batch_size", BATCH_CHOICES)
    epochs = trial.suggest_int("num_train_epochs", EPOCH_RANGE[0], EPOCH_RANGE[1])

    out_dir = f"./results/{TASK_NAME}/optuna/trial_{trial.number}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    model = CLSHead(MODEL_NAME, dropout=DROPOUT)
    training_args = make_args(out_dir, lr=lr, bs=bs, epochs=epochs, save_total_limit=1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tune_train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=valid_dataset)
    return float(metrics["eval_f1"])


# ----------------------------
# Main: HPO -> Final train -> Evaluate tests once
# ----------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tokenizer, data_collator, train_dataset, tune_train_dataset, valid_dataset, eval_sets = prepare_data()

    print(f"\nStarting Optuna study: {STUDY_NAME} (n_trials={N_TRIALS})")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction=DIRECTION,
        study_name=STUDY_NAME,
        sampler=sampler,
        storage=STUDY_STORAGE,
        load_if_exists=bool(STUDY_STORAGE),
    )

    study.optimize(
        lambda t: objective(t, data_collator, tune_train_dataset, valid_dataset),
        n_trials=N_TRIALS,
        gc_after_trial=True,
    )

    print("\nBest trial:")
    print("  value (eval_f1):", study.best_value)
    print("  params:", study.best_params)

    # Final training with best HPs on full train
    best = study.best_params
    final_out = f"./results/{TASK_NAME}/best_run"
    if os.path.exists(final_out):
        shutil.rmtree(final_out, ignore_errors=True)
    os.makedirs(final_out, exist_ok=True)

    set_seed(SEED)
    final_model = CLSHead(MODEL_NAME, dropout=DROPOUT)

    final_args = make_args(
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    print("\nTraining final model on 100% training data with best hyperparameters...")
    start = time.time()
    final_trainer.train()
    end = time.time()
    print("Total time:", (end - start), "seconds")

    print("\nFinal validation metrics (best checkpoint):")
    print(final_trainer.evaluate(eval_dataset=valid_dataset))

    if len(eval_sets) > 0:
        print("\nTest metrics (evaluate once per split):")
        for name, ds in eval_sets.items():
            print(f"{name}: {final_trainer.evaluate(eval_dataset=ds)}")

    print("Done.")


if __name__ == "__main__":
    main()
