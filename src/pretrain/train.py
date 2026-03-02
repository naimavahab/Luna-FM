
import torch
torch.cuda.empty_cache()
from Bio import SeqIO
import os
os.environ["PYTORCH_ENABLE_DYNAMO"] = "0"

import math
import time
import pandas as pd
import numpy as np
import wandb
from datetime import datetime
from datasets import load_dataset,load_from_disk

from datasets import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    ModernBertConfig,
    ModernBertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EvalPrediction
)


import torch
if torch.cuda.is_available():
    print("CUDA is available. Number of GPUs:", torch.cuda.device_count())
else:
    print("CUDA is not available. No GPUs detected.")
    
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()

def convert_fasta_to_dataset(dataset_name):
  
    dataset = load_dataset("multimolecule/rnacentral", split="train", streaming=True)
   # dataset = dataset.take(1000)
    dataset = dataset.rename_column("sequence", "text")
    return dataset  #.select(range(min(max_samples, len(dataset))))

def load_fasta_biopython(fasta_path,name):
    sequences = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append({"text": str(record.seq)})

    dataset = Dataset.from_list(sequences)
    dataset.save_to_disk(name)
  #  return dataset
def load_tokenizer(tokenizer_path):
 
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        special_tokens=["<unk>", "<pad>", "<cls>", "<sep>", "<mask>"],
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
    )
    required_tokens = ["<pad>", "<cls>", "<sep>", "<mask>", "<unk>", "<s>", "</s>"]
    for token in required_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            print(f"Missing token: {token}")
            print('\n')
        else:
            print(f"{token} ID: {token_id}")
            print('\n')
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

    print("Tokenization complete.")
    print('\n')
    return tokenized


def compute_metrics(eval_preds: EvalPrediction):
    predictions, labels = eval_preds.predictions, eval_preds.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    logits = torch.tensor(predictions)
    labels = torch.tensor(labels)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss).item() if loss.item() < 100 else float("inf")


    print('\n')

    return {"eval_loss": loss.item(), "perplexity": perplexity}


def main():
    # Configuration
    folder_path ="data" #subset_100_truncated_len3000.csv"
    tokenizer_path ="rna_tokenizer_final.json"
    output_dir = "./RNA_FM_models"
    save_model_path = "Model_RNA"
    
    # Login to Weights & Biases

    wandb.init(project="RNA_Pretrain", name="Pretraining")
    print(" Logged into Weights & Biases.")
    print('\n')

    dataset = convert_fasta_to_dataset(folder_path)
    fasta_file = '../final.fasta'
    folder_path = 'rna_Seqs'


    if os.path.exists(folder_path):
        print("Data Folder exists")
        dataset = load_from_disk(folder_path)
    else:
        print("Data Folder does NOT exist,creating the data folder")
        load_fasta_biopython(fasta_file,folder_path)
        dataset = load_from_disk(folder_path)
    

    tokenizer = load_tokenizer(tokenizer_path)
    
    tokenised_folder = 'tokenised_rna'
    if os.path.exists(tokenised_folder):
        print("Tokenised Folder exists")
        tokenized_dataset = load_from_disk(tokenised_folder)
    else:
        print("Tokenised Folder does NOT exist, tokenising the data")
       
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        
        tokenized_dataset.save_to_disk(tokenised_folder) # Specify a directory to save to


    split = tokenized_dataset # tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.20
    )

    # Model Config and Initialization
    config = ModernBertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
      #  global_rope_theta=10000,
        reference_compile=False
    )



    model = ModernBertForMaskedLM(config=config)
    print("Model initialized.")
    print('\n')
    model.resize_token_embeddings(len(tokenizer))

   
    training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=40,
  #  max_steps=50 , #500000,

    per_device_train_batch_size=96 , #64,             
    #per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,
    #eval_accumulation_steps=2,
    learning_rate=5e-5,                           
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    lr_scheduler_type="cosine",                   
    warmup_ratio=0.06,
    save_strategy="epoch",       # save at the end of each epoch
    save_total_limit=2,          # keep only the last 2 checkpoints
    logging_steps=10,
    logging_dir='./logs',
   # eval_strategy="epoch", # also evaluate at the end of each epoch
  #  fp16=True,
    torch_compile=False,
    no_cuda=False,
    report_to=["wandb"],
    run_name="RNA_modernBERT",
    # load_best_model_at_end=True,      # useful for best validation F1
    metric_for_best_model="loss",       # important to guide saving
    greater_is_better=False,           # since higher F1 is better
   #  bf16 = True    
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split,
       # eval_dataset=split["test"].shuffle(seed=42),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print("Starting training...")
    print('\n')
        # Access the number of GPUs configured for the Trainer
    if hasattr(trainer.args, '_n_gpu'):
        print(f"Hugging Face Trainer is configured to use {trainer.args._n_gpu} GPUs.")
    else:
        print("Could not determine GPU usage from Trainer arguments directly (e.g., using DistributedDataParallel).")

        
    trainer.train()
    

    trainer.save_model(save_model_path)
    print(f"Model saved to: {save_model_path}")
    print('\n')


    end_time = time.time()

    # Print total training time
    total_time_sec = end_time - start_time
    hours = int(total_time_sec // 3600)
    minutes = int((total_time_sec % 3600) // 60)
    seconds = int(total_time_sec % 60)
    print(f"Total training time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    
    main()

