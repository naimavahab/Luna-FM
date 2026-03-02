# filename: finetune_archiveii_9class_then_tsne.py
# Fine-tune BiRNA-BERT on multimolecule/archiveii (9-class) + extract embeddings + t-SNE

import os
import numpy as np
import torch
from datasets import load_dataset, ClassLabel,load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
os.environ["PYTORCH_ENABLE_DYNAMO"] = "0"
torch._dynamo.disable()
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_ENABLE_DYNAMO"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# ----------------------------
# Config
# ----------------------------
set_seed(42)

MODEL_NAME =  ""
TOKENIZER_NAME = ""
DATASET_NAME = "../../Data/lncrna_preeclampsia" #rfam_longest_50k" #"multimolecule/archiveii"

# If you want to pin a GPU:
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LABELS = 2

# For RNA/DNA tokenizers, max length is often ~1024 tokens for BERT-like
MAX_LENGTH = 1024  # adjust if your model config has smaller max_position_embeddings
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 8

OUT_DIR = ".archiveii_9class"
LOG_DIR = ".log_archiveii_9class"

# ----------------------------
# Load dataset
# ----------------------------
# archiveii has columns like: sequence, family
# We'll create label ids from 'family' strings.
raw = load_from_disk(DATASET_NAME)
print(raw)

families = sorted(set(raw["label"])) #)family"]))
print(families)
label2id = {f:i for i, f in enumerate(families)}
id2label = {i:f for f,i in label2id.items()}

# 2) Add integer labels column
def add_labels(ex):
    ex["labels"] = label2id[ex["label"]]
    return ex

raw = raw.map(add_labels)

# 3) Cast labels to ClassLabel (this is the key!)
raw = raw.cast_column("labels", ClassLabel(num_classes=len(families), names=families))


tmp = raw.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")
tmp2 = tmp["train"].train_test_split(test_size=0.125, seed=42, stratify_by_column="labels")
ds_train, ds_val, ds_test = tmp2["train"], tmp2["test"], tmp["test"]

# Build label mapping from all splits to be safe
all_families = set(ds_train["labels"]) | set(ds_val["labels"]) | set(ds_test["labels"])
label_list = sorted(list(all_families))
label2id = {lab: i for i, lab in enumerate(label_list)}
id2label = {i: lab for lab, i in label2id.items()}

print("Num classes:", len(label_list))
print("Classes:", label_list)
assert len(label_list) == NUM_LABELS, f"Expected {NUM_LABELS} classes, got {len(label_list)}"

# Add numeric labels
def add_label(example):
    example["labels"] = label2id[example["labels"]]
    return example

ds_train = ds_train.map(add_label)
ds_val   = ds_val.map(add_label)
ds_test  = ds_test.map(add_label)

# ----------------------------
# Tokenize
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

def preprocess(batch):
    # Many RNA tokenizers are char-level; some require spacing like "A U G C".
    # BiRNA tokenizer typically works directly on raw sequence, so keep as-is.
    # If you need spaced, uncomment the next line:
    # batch["sequence"] = [" ".join(list(s)) for s in batch["sequence"]]

    return tokenizer(
        batch["sequence"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # let data collator pad dynamically
    )

train_tok = ds_train.map(preprocess, batched=True, remove_columns=[c for c in ds_train.column_names if c not in ("labels",)])
val_tok   = ds_val.map(preprocess, batched=True, remove_columns=[c for c in ds_val.column_names if c not in ("labels",)])
test_tok  = ds_test.map(preprocess, batched=True, remove_columns=[c for c in ds_test.column_names if c not in ("labels",)])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------
# Model
# ----------------------------
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    num_labels=NUM_LABELS,
    label2id=label2id,
    id2label=id2label,
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    trust_remote_code=True#,reference_compile = True
).to(DEVICE)

# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    f1w = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w}

OUT_DIR = "./runs"
BEST_DIR = "./best_model"

# ----------------------------
# Training
# ----------------------------
args = TrainingArguments(
    output_dir=OUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",                # save every epoch
    save_total_limit=2,

    load_best_model_at_end=True,          #  automatically reload best
    metric_for_best_model="f1_macro",     #  choose metric
    greater_is_better=True,               #  maximize

    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,

    fp16=torch.cuda.is_available(),
    warmup_ratio=0.1,
    logging_steps=100,
    report_to="none",
    torch_compile=False

)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
print(trainer.evaluate(test_tok))


print("Best checkpoint:", trainer.state.best_model_checkpoint)

trainer.save_model(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)

print(f"Best model saved to: {BEST_DIR}")

