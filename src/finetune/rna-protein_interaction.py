
import os
import gc
import random
import shutil
import numpy as np
from tqdm import tqdm

import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset

import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_constant_schedule_with_warmup

from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, recall_score, precision_score


# =========================
# Repro + device
# =========================
SEED = 42
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_all_seeds(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Base config
# =========================
MODEL_NAME = "LunaFM" #"buetnlpbio/birna-bert"
TOKENIZER_NAME = "LunaFM" #"buetnlpbio/birna-tokenizer"


dataset_name = "data/AKAP1/AKAP1"
MAX_TOKENS = 1022

ONLY_BPE = False
ONLY_NUC = False

# ---------------- HPO policy ----------------
TUNE_FRACTION = 0.20          # ✅ 0.15–0.20 recommended
EARLY_STOP_PATIENCE = 2

# ---------------- Optuna config ----------------
N_TRIALS = 20
STUDY_NAME = f"optuna_meanpool_binary_{os.path.basename(dataset_name)}_frac{int(TUNE_FRACTION*100)}"
STUDY_STORAGE = None
DIRECTION = "maximize"

# HPO ranges (adjust to your GPU)
LR_RANGE = (1e-7, 5e-4)                  # log-uniform
BATCH_CHOICES = [ 8, 16,32,64]            # remove 16 if OOM
EPOCH_RANGE = (2, 10)                    # keep smaller during HPO
DROPOUT_RANGE = (0.0, 0.3)
MLP_HIDDEN_CHOICES = [64, 128, 256, 512]
WARMUP_RATIO_RANGE = (0.0, 0.15)
WEIGHT_DECAY = 0.01
GRAD_ACCUM_CHOICES = [1, 2, 4]


# =========================
# Helpers
# =========================
def dynamic_tokenize_preprocessing(seq: str) -> str:
    if ONLY_NUC:
        return " ".join(seq)
    elif ONLY_BPE:
        return seq
    if len(seq) < MAX_TOKENS:
        #seq = " ".join(seq[:MAX_TOKENS])
        return seq

def tensor_to_seq(t):
    t = t.numpy()[1:-1]  # drop special tokens
    s = []
    for v in t:
        if v == 5: s.append("A")
        elif v == 6: s.append("U")
        elif v == 7: s.append("C")
        elif v == 8: s.append("G")
    return "".join(s)

def collate_fn_factory(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate_fn(batch):
        input_ids = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = (input_ids != pad_id).long()
        labels = torch.stack(labels).float()
        return input_ids, attention_mask, labels

    return collate_fn

def compute_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, thr: float = 0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_pred = (probs > thr).astype(int)
    y_true = labels.detach().cpu().numpy().astype(int)

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# =========================
# Dataset
# =========================
class RNA_M6_Site(Dataset):
    def __init__(self, seqs, labels, tokenizer):
        self.seqs = seqs
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx].replace("T", "U").upper()
        seq = dynamic_tokenize_preprocessing(seq)

        tok = self.tokenizer(
            seq,
            return_tensors="pt",
            max_length=MAX_TOKENS,
            truncation=True,
            padding=False
        )
        input_ids = tok["input_ids"].squeeze(0)  # CPU tensor
        label_tensor = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return input_ids, label_tensor


# =========================
# Model: masked mean pool -> MLP
# =========================
class MeanPoolBinaryHead(nn.Module):
    def __init__(self, backbone_mlm, hidden_size, mlp_hidden=256, dropout=0.1, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone_mlm
        self.hidden_size = hidden_size
        self.freeze_backbone = freeze_backbone

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(mlp_hidden, 1)

    def _last_hidden(self, input_ids, attention_mask):
        # For MLM models, encoder is usually `.bert`
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        else:
            out = self.backbone.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        if isinstance(out, tuple):
            return out[0]
        return out.last_hidden_state

    @staticmethod
    def _masked_mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()   # (B,L,1)
        summed = (last_hidden * mask).sum(dim=1)      # (B,H)
        denom = mask.sum(dim=1).clamp(min=1e-6)       # (B,1)
        return summed / denom                         # (B,H)

    def forward(self, input_ids, attention_mask):
        last_hidden = self._last_hidden(input_ids, attention_mask)           # (B,L,H)
        mean_vec = self._masked_mean_pool(last_hidden, attention_mask)       # (B,H)

        x = self.dropout(mean_vec)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.fc2(x).squeeze(-1)                                     # (B,)
        return logits


# =========================
# Train/Eval loops
# =========================
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    all_logits, all_labels = [], []

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device).long()
        attention_mask = attention_mask.to(device).long()
        labels = labels.to(device).float()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics_from_logits(all_logits, all_labels)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics

def train_one_epoch(model, loader, optimizer, scheduler, criterion, grad_accum_steps=1):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    losses = []
    all_logits, all_labels = [], []

    for step, (input_ids, attention_mask, labels) in enumerate(loader):
        input_ids = input_ids.to(device).long()
        attention_mask = attention_mask.to(device).long()
        labels = labels.to(device).float()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels) / float(grad_accum_steps)
        loss.backward()

        losses.append(loss.item() * float(grad_accum_steps))
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics_from_logits(all_logits, all_labels)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


# =========================
# Data prep (run once)
# =========================
def prepare_data():
    # Load tensors -> strings
    test_file_x  = torch.load(f"{dataset_name}_test_x_tensors")
    valid_file_x = torch.load(f"{dataset_name}_valid_x_tensors")
    train_file_x = torch.load(f"{dataset_name}_train_x_tensors")

    label_test  = np.loadtxt(f"{dataset_name}_test_y")
    label_valid = np.loadtxt(f"{dataset_name}_valid_y")
    label_train = np.loadtxt(f"{dataset_name}_train_y")

    train_seqs = [tensor_to_seq(train_file_x[i]) for i in range(len(train_file_x))]
    valid_seqs = [tensor_to_seq(valid_file_x[i]) for i in range(len(valid_file_x))]
    test_seqs  = [tensor_to_seq(test_file_x[i])  for i in range(len(test_file_x))]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    collate_fn = collate_fn_factory(tokenizer)

    train_full = RNA_M6_Site(train_seqs, label_train, tokenizer)
    valid_full = RNA_M6_Site(valid_seqs, label_valid, tokenizer)
    test_ds    = RNA_M6_Site(test_seqs,  label_test,  tokenizer)

    # ✅ Subsample for HPO
    rng = np.random.default_rng(SEED)

    n_train = max(1, int(len(train_full) * TUNE_FRACTION))
    n_valid = max(1, int(len(valid_full) * TUNE_FRACTION))

    train_idx = rng.choice(len(train_full), n_train, replace=False)
    valid_idx = rng.choice(len(valid_full), n_valid, replace=False)

    train_subset = Subset(train_full, train_idx.tolist())
    valid_subset = Subset(valid_full, valid_idx.tolist())

    return tokenizer, collate_fn, train_subset, valid_subset, test_ds, train_full, valid_full


# =========================
# Optuna objective (tune on subsets only)
# =========================
def objective(trial: optuna.Trial, collate_fn, train_subset, valid_subset):
    set_all_seeds(SEED)

    lr = trial.suggest_float("learning_rate", LR_RANGE[0], LR_RANGE[1], log=True)
    batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    epochs = trial.suggest_int("epochs", EPOCH_RANGE[0], EPOCH_RANGE[1])
    dropout = trial.suggest_float("dropout", DROPOUT_RANGE[0], DROPOUT_RANGE[1])
    mlp_hidden = trial.suggest_categorical("mlp_hidden", MLP_HIDDEN_CHOICES)
    warmup_ratio = trial.suggest_float("warmup_ratio", WARMUP_RATIO_RANGE[0], WARMUP_RATIO_RANGE[1])
    freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, True, False])  # bias True
    grad_accum_steps = trial.suggest_categorical("grad_accum_steps", GRAD_ACCUM_CHOICES)

    model = None
    backbone = None
    optimizer = None
    scheduler = None

    try:
        # Load backbone per trial
        config = transformers.BertConfig.from_pretrained(MODEL_NAME)
        backbone = AutoModelForMaskedLM.from_pretrained(
            MODEL_NAME, config=config, trust_remote_code=True
        ).to(device)

        # Freeze/unfreeze
        for p in backbone.parameters():
            p.requires_grad = (not freeze_backbone)

        hidden = backbone.config.hidden_size

        model = MeanPoolBinaryHead(
            backbone_mlm=backbone,
            hidden_size=hidden,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0, pin_memory=torch.cuda.is_available()
        )
        valid_loader = DataLoader(
            valid_subset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0, pin_memory=torch.cuda.is_available()
        )

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        num_steps = len(train_loader) * epochs
        num_warmup_steps = int(warmup_ratio * num_steps)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

        best_val_f1 = -1.0
        bad_epochs = 0

        for ep in range(1, epochs + 1):
            _ = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, grad_accum_steps=grad_accum_steps)
            val_metrics = evaluate(model, valid_loader, criterion)

            trial.report(val_metrics["f1"], step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= EARLY_STOP_PATIENCE:
                    break

        return float(best_val_f1)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise optuna.TrialPruned()
        raise

    finally:
        # critical cleanup between trials
        for name in ["model", "backbone", "optimizer", "scheduler", "train_loader", "valid_loader"]:
            if name in locals():
                try:
                    del locals()[name]
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# =========================
# Final training (full train/valid), test once
# =========================
def train_final(best_params, collate_fn, train_full, valid_full, test_ds):
    set_all_seeds(SEED)

    lr = float(best_params["learning_rate"])
    batch_size = int(best_params["batch_size"])
    epochs = int(best_params["epochs"])
    dropout = float(best_params["dropout"])
    mlp_hidden = int(best_params["mlp_hidden"])
    warmup_ratio = float(best_params["warmup_ratio"])
    freeze_backbone = bool(best_params["freeze_backbone"])
    grad_accum_steps = int(best_params["grad_accum_steps"])

    config = transformers.BertConfig.from_pretrained(MODEL_NAME)
    backbone = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, config=config, trust_remote_code=True
    ).to(device)

    for p in backbone.parameters():
        p.requires_grad = (not freeze_backbone)

    hidden = backbone.config.hidden_size
    model = MeanPoolBinaryHead(
        backbone_mlm=backbone,
        hidden_size=hidden,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_full, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    num_steps = len(train_loader) * epochs
    num_warmup_steps = int(warmup_ratio * num_steps)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    best_state = None
    best_val_f1 = -1.0

    for ep in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, grad_accum_steps=grad_accum_steps)
        val_metrics = evaluate(model, valid_loader, criterion)

        print(
            f"Epoch {ep:02d} | train_f1={train_metrics['f1']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_mcc={val_metrics['mcc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nFinal validation metrics (full valid):")
    print(evaluate(model, valid_loader, criterion))

    print("\nTest metrics:")
    print(evaluate(model, test_loader, criterion))


# =========================
# Main
# =========================
def main():
    tokenizer, collate_fn, train_subset, valid_subset, test_ds, train_full, valid_full = prepare_data()

    print(f"Full sizes:   train={len(train_full)}, valid={len(valid_full)}, test={len(test_ds)}")
    print(f"HPO sizes:    train={len(train_subset)} ({TUNE_FRACTION:.2f}), "
          f"valid={len(valid_subset)} ({TUNE_FRACTION:.2f})")

    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        direction=DIRECTION,
        study_name=STUDY_NAME,
        sampler=sampler,
        storage=STUDY_STORAGE,
        load_if_exists=bool(STUDY_STORAGE),
    )

    study.optimize(
        lambda t: objective(t, collate_fn, train_subset, valid_subset),
        n_trials=N_TRIALS,
        gc_after_trial=True,  # python gc; cuda cleanup is in finally
    )

    print("\nBest trial:")
    print("  value (val_f1 on subset):", study.best_value)
    print("  params:", study.best_params)

    print("\nTraining final model on FULL train/valid and evaluating on test once...")
    train_final(study.best_params, collate_fn, train_full, valid_full, test_ds)

    print("\nDone.")


if __name__ == "__main__":
    main()
