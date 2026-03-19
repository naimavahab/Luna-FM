<img width="2852" height="1936" alt="methodology" src="https://github.com/user-attachments/assets/fd42436b-0cd7-4d70-8273-c0d64608907b" /># Luna-FM (BPE): RNA Foundation Model for Long-Sequence Understanding

---

## Overview

Luna-FM (BPE) is a large-scale RNA foundation model designed to model long and complex RNA sequences using subword tokenisation and an optimised BERT-style transformer encoder.

Unlike nucleotide-level RNA models, Luna-FM uses Byte Pair Encoding (BPE) to compress RNA sequences into biologically meaningful subunits such as motifs and conserved regions. This enables efficient modelling of long transcripts while preserving functional signals.



---

## Key Features

- Subword-based RNA representation using BPE  
- Efficient modelling of long RNA sequences  
- Hardware-optimised transformer architecture  
- Strong performance across multiple downstream tasks  
- Biologically meaningful embeddings validated via interpretability  

---

## Data Processing

- Source: RNAcentral  
- Initial dataset: approximately 34 million RNA sequences  
- Redundancy reduction: CD-HIT-EST at 80 percent sequence identity  
- Final dataset used for training: approximately 28 million sequences  

This preprocessing reduces bias, removes redundancy, and improves generalisation.

---

## Tokenisation

A Byte Pair Encoding (BPE) tokeniser was trained to learn variable-length RNA sub-sequences.

### Vocabulary Analysis

| Vocabulary Size | Effect |
|----------------|--------|
| 2k             | Low compression, high resolution |
| 4k             | Balanced compression and biological signal (default) |
| 8k             | Higher compression with longer tokens |
| 15k            | Excessive compression and sparsity |

Default configuration uses a vocabulary size of 4096 tokens.

### Tokeniser Design

- Special tokens: `[CLS], [SEP], [MASK], [UNK]`  
- Maximum token length: 16 bases  
- No padding at tokenisation stage  
- Padding handled internally using unpadding  

The BPE tokeniser captures recurring RNA patterns while maintaining flexibility for rare sequences.

to create a new BPE tokeniser from a fasta file run :
```bash
nohup python src/tokeniser/train_Bpe_tokeniser.py   --corpus final.fasta   --out_dir ./tokenizer_bpe_4k_final   --vocab_size 4096   --model_length 2000 > log4k.log 2>&1 & 

to create a new sentence-piece unigram tokeniser from a fasta file run :
```bash
python train_sentencepiece_unigram_from_corpus.py \
  --corpus corpus.txt \
  --out_dir tokenizer_unigram_8k \
  --model_prefix rna_unigram_8k \
  --vocab_size 8000 \
  --input_sentence_size 2000000 \
  --max_sentence_length 2048


---

## Model Architecture

Luna-FM is an encoder-only transformer with the following configuration:

- Layers: 22  
- Hidden size: 768  
- Attention heads: 12  
- Feedforward: GeGLU with dimension 2304  
- Total parameters: approximately 113 million  

### Architectural Improvements

**Alternating Attention**  
Every third layer uses global attention, while intermediate layers use local sliding-window attention with a window size of 128 tokens.

**Rotary Positional Embeddings (RoPE)**  
Supports variable-length sequences without fixed positional limits.

**Unpadding**  
Padding tokens are removed before attention computation, improving efficiency.

**Flash Attention**  
FlashAttention v2 and v3 are used to accelerate attention operations.

---

## Pretraining Setup

| Parameter | Value |
|----------|------|
| Training steps | 30,000 |
| Batch size | 96 |
| Learning rate | 5 × 10⁻⁵ |
| Optimiser | AdamW |
| Mask ratio | 20 percent |
| Max sequence length | 128 |
| Hardware | 4 NVIDIA A10 GPUs |

---

---

## Usage

### Pretraining

Run the pretraining script with BPE tokenisation:

```bash
python src/pretrain/train_BPE.py \
  --fasta_file final.fasta \
  --dataset_dir rna_Seqs \
  --tokenizer_path ../tokeniser/tokenizer_bpe_4k_final \
  --tokenized_dir tokenised_rna_BPE4k \
  --output_dir ./RNA_FM_models \
  --save_model_path Model_RNA_BPE4k \
  --max_len 128 \
  --num_train_epochs 20 \
  --per_device_train_batch_size 96 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --fp16

## Downstream Tasks
Codes available under /src/finetune 
Data available under src/data

### mRNA vs lncRNA Classification

- Sequence length up to 26,831 bases  
- F1 score: 0.9293  
- Approximately 15 to 16 percent improvement over prior models  

---

### Subcellular Localisation

- Extremely long sequences (up to 551,120 bases)  
- Limited training data  
- F1 score: 0.5068  
- Strong improvement over existing models  

---

### Splice Site Prediction (Cross-species)

- Dataset: Spliceator  
- Evaluated on multiple species  
- Consistent second-best performance  
- Strong generalisation to mRNA despite ncRNA-focused pretraining  

---

## Interpretability

Model interpretability was assessed using:

- Integrated Gradients for identifying important sequence regions  
- XSTREME for motif enrichment analysis  

### Key Findings

- Motifs present in 43.0 percent of positive regions vs 28.6 percent in negative regions  
- Significant enrichment with odds ratio of 1.89  
- Identified motifs correspond to RNA-binding proteins such as ELAVL1, HNRNPC, and PCBP1  

These findings demonstrate that Luna-FM captures biologically meaningful sequence features.

---

## Model Advantages

Compared to existing RNA foundation models:

- Uses subword tokenisation instead of nucleotide-level encoding  
- Efficiently handles long RNA sequences  
- Supports flexible sequence lengths via RoPE  
- Improved computational efficiency using unpadding and FlashAttention  
- Produces interpretable and biologically relevant embeddings  

---

## Applications

Luna-FM can be used for:

- RNA classification tasks  
- Functional annotation  
- Subcellular localisation prediction  
- Splice site prediction  
- Motif discovery  
- Embedding generation for downstream machine learning tasks  

---

## Model Storage and Usage

Hugging face model checkpoints available at : https://huggingface.co/NaimaVahab/luna-fm_BPE_2k, https://huggingface.co/NaimaVahab/luna-fm_BPE_4k, https://huggingface.co/NaimaVahab/luna-fm_BPE_15k

---

## Reproducibility

- Hyperparameters tuned using Bayesian optimisation (Optuna)  
- Same training setup used for both BPE and Unigram variants  

---

## Summary

Luna-FM provides a scalable and efficient framework for RNA sequence modelling by combining subword tokenisation with a modern transformer architecture. It demonstrates strong performance on long RNA tasks and generalises effectively across diverse RNA prediction problems.

---

## Citation

```bibtex
@article{luna_fm_2026,
  title={Luna-FM: Subword-based RNA Foundation Model for Long-Sequence Understanding},
  author={Naima Vahab,Estrid He,Tabinda Sarwar and Sonika Tyagi},
  journal={ Bioinformatics (Oxford University Press) },
  year={2026}
}
