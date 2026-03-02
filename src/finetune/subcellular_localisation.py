import os
import shutil
import random
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
from multimolecule import RnaTokenizer, RiNALMoModel,RiNALMoForSequencePrediction

from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
)

# ----------------------------
# Environment
# ----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"  # uncomment / set as needed

# ----------------------------
# Config
# ----------------------------
TASK_NAME = "subcellular_localisation"  # rename freely
DATASET_PATH = "subcellular_hf"         # <-- path to the DatasetDict saved_to_disk (train/validation/test)
MODEL_NAME = "../Luna_FM"   # <-- your base model
TOKENIZER = "../Luna_FM"  
#MODEL_NAME = "Lancelot53/birnabert-2ep"
#TOKENIZER = "buetnlpbio/birna-tokenizer"  # <-- tokenizer path (often same as model)

#MODEL_NAME = "multimolecule/rinalmo-mega"
#TOKENIZER = "multimolecule/rinalmo-mega"
MAX_LENGTH = 1024
SEED = 42

# Head config
MLP_HIDDEN = 512
DROPOUT = 0.1

# HPO policy
TUNE_FRACTION = 0.20          # 0.15â€“0.20 recommended
N_TRIALS = 30
EARLY_STOP_PATIENCE = 2

# HPO ranges (biased for small data + long seqs)
LR_RANGE = (5e-6, 3e-4)       # slightly narrower than (1e-6,5e-4)
BATCH_CHOICES = [1, 2] #,16]  # long seqs usually force small batches
EPOCH_RANGE = (3, 20)

# Reduce memory pressure for long sequences
GRAD_ACCUM_CHOICES = [1, 2, 4, 8]  # tune effective batch without OOM

# Data map multiprocessing
MAP_NUM_PROC = 1  # safest on shared FS; increase only if stable

STUDY_NAME = f"optuna_{TASK_NAME}_cls"
STUDY_STORAGE = None
DIRECTION = "maximize"


# ----------------------------
# Model
# ----------------------------
class MultiClassCLS(nn.Module):
    """
    Base encoder -> CLS embedding -> dropout -> Linear -> GELU -> Linear -> n_classes logits.
    CrossEntropyLoss for single-label multi-class classification.
    """
    def __init__(self, model_name: str, n_classes: int, mlp_hidden: int = 512, dropout: float = 0.1):
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]            # (B, H)
        logits = self.classifier(self.dropout(cls))     # (B, C)

        loss = None
        if labels is not None:
            labels = labels.long().view(-1)
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


# ----------------------------
# Tokenization
# ----------------------------
def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    if tok is None:
        raise ValueError("Tokenizer failed to load.")
    return tok

def tokenize_function(tokenizer, examples):
    # Your dataset column is "sequence"
    return tokenizer(
        examples["sequence"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

def cast_label_to_int(examples):
    # Input column: "label" -> Trainer expects "labels"
    # Handles cases where label is already int/str
    print(examples["label"])
    examples["labels"] = [int(x) for x in examples["label"]]
    return examples


# ----------------------------
# Metrics
# ----------------------------
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


# ----------------------------
# Subsampling for HPO (15â€“20% of train)
# - robust to tiny classes: if a class has too few samples, we still keep >=1
# ----------------------------
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
            # if class is very small, keep all (prevents empty selection / replace=False errors)
            chosen = y_idx
        else:
            chosen = rng.choice(y_idx, size=k, replace=False)
        keep.append(chosen)

    keep = np.concatenate(keep)
    keep = keep[rng.permutation(len(keep))]
    return ds.select(keep.tolist())


# ----------------------------
# Data prep (load_from_disk)
# ----------------------------
def prepare_data():
    print(f"Loading DatasetDict from disk: {DATASET_PATH}")
    full = load_from_disk(DATASET_PATH)
    
   # full = full.class_encode_column("label")


    # Expect train/validation/test
    missing = [k for k in ["train", "validation", "test"] if k not in full]
    if missing:
        raise ValueError(f"Expected splits train/validation/test, missing: {missing}. Found: {list(full.keys())}")

    # Infer n_classes from TRAIN labels only (avoid peeking, but class set should match anyway)
    train_labels_raw = full["train"]["label"]
    print(full["train"][0])
    print(train_labels_raw)
    print('train_labels_raw')
    train_labels_int = [int(x) for x in train_labels_raw]
    n_classes = len(set(train_labels_int))
    print(f"Detected n_classes (from train): {n_classes}")

    tokenizer = build_tokenizer()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print(full["train"][0])
    print("Tokenizing (map)...")
    tokenized = full.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
        num_proc=MAP_NUM_PROC,
        remove_columns=["sequence"],           # keep id, label, source if present
        load_from_cache_file=False,            # safer on shared FS
        desc="Tokenizing",
    )

    tokenized = tokenized.map(
        cast_label_to_int,
        batched=True,
        num_proc=MAP_NUM_PROC,
        load_from_cache_file=False,
        desc="Casting labels",
    )

    # Keep only columns Trainer needs (plus id if you want to keep it; Trainer will ignore extras)
    keep_cols = [c for c in ["input_ids", "attention_mask", "labels"] if c in tokenized["train"].column_names]
    tokenized.set_format(type="torch", columns=keep_cols)

    train_dataset = tokenized["train"]
    valid_dataset = tokenized["validation"]
    test_dataset = tokenized["test"]

    tune_train_dataset = stratified_subsample(train_dataset, frac=TUNE_FRACTION, label_col="labels", seed=SEED)
    print(f"HPO train subset size: {len(tune_train_dataset)} / {len(train_dataset)} ({TUNE_FRACTION*100:.1f}%)")

    return n_classes, data_collator, train_dataset, tune_train_dataset, valid_dataset, test_dataset


# ----------------------------
# TrainingArguments builder
# ----------------------------
def make_training_args(output_dir, lr, bs, epochs, grad_accum, seed=SEED, save_total_limit=1):
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

        fp16=True, #torch.cuda.is_available(),
        report_to="none",

        save_total_limit=save_total_limit,

        # keeps evaluation simpler / stable on small sets
        dataloader_pin_memory=True,
    )


# ----------------------------
# Optuna objective
# ----------------------------
def objective(trial: optuna.Trial, n_classes, data_collator, tune_train_dataset, valid_dataset):
    set_seed(SEED)

    lr = trial.suggest_float("learning_rate", LR_RANGE[0], LR_RANGE[1], log=True)
    bs = trial.suggest_categorical("per_device_train_batch_size", BATCH_CHOICES)
    epochs = trial.suggest_int("num_train_epochs", EPOCH_RANGE[0], EPOCH_RANGE[1])
    grad_accum = trial.suggest_categorical("gradient_accumulation_steps", GRAD_ACCUM_CHOICES)

    out_dir = f"./results/{TASK_NAME}/optuna/trial_{trial.number}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    model = MultiClassCLS(
        MODEL_NAME,
        n_classes=n_classes,
        mlp_hidden=MLP_HIDDEN,
        dropout=DROPOUT,
    )

    training_args = make_training_args(
        out_dir,
        lr=lr,
        bs=bs,
        epochs=epochs,
        grad_accum=grad_accum,
        save_total_limit=1#, fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tune_train_dataset,   # HPO subset only
        eval_dataset=valid_dataset,         # choose HPs by validation
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=valid_dataset)
    return float(metrics["eval_f1_macro"])


# ----------------------------
# Main: HPO -> Final train -> Test once
# ----------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    n_classes, data_collator, train_dataset, tune_train_dataset, valid_dataset, test_dataset = prepare_data()

    print(f"\nStarting Optuna study: {STUDY_NAME} (trials={N_TRIALS})")
    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        direction=DIRECTION,
        study_name=STUDY_NAME,
        sampler=sampler,
        storage=STUDY_STORAGE,
        load_if_exists=bool(STUDY_STORAGE),
    )

    study.optimize(
        lambda t: objective(t, n_classes, data_collator, tune_train_dataset, valid_dataset),
        n_trials=N_TRIALS,
        gc_after_trial=True,
    )

    print("\nBest trial:")
    print("  value (eval_f1_macro):", study.best_value)
    print("  params:", study.best_params)

    best = study.best_params

    # ----------------------------
    # Final training on 100% train
    # ----------------------------
    final_out = f"./results/{TASK_NAME}/best_run"
    if os.path.exists(final_out):
        shutil.rmtree(final_out, ignore_errors=True)
    os.makedirs(final_out, exist_ok=True)

    set_seed(SEED)
    final_model = MultiClassCLS(
        MODEL_NAME,
        n_classes=n_classes,
        mlp_hidden=MLP_HIDDEN,
        dropout=DROPOUT,
    )

    final_args = make_training_args(
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
        train_dataset=train_dataset,     # 100% train
        eval_dataset=valid_dataset,      # same validation split
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    print("\nTraining final model on 100% train with best hyperparameters...")
    final_trainer.train()

    print("\nFinal validation metrics (best checkpoint):")
    print(final_trainer.evaluate(eval_dataset=valid_dataset))

    # ----------------------------
    # Evaluate on test ONCE
    # ----------------------------
    print("\nTest metrics (once):")
    print(final_trainer.evaluate(eval_dataset=test_dataset))

    print("\nDone.")


if __name__ == "__main__":
    main()
