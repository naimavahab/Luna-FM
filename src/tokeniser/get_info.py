from transformers import AutoTokenizer

# Load your saved tokenizer directory
tokenizer = AutoTokenizer.from_pretrained("4k_BPE")

# Method 1 (recommended)
print("Vocab size:", tokenizer.vocab_size)

# Method 2 (includes added tokens if any)
print("Full vocab size (with added tokens):", len(tokenizer))

# Optional: inspect first few tokens
vocab = tokenizer.get_vocab()
print("First 10 tokens:", list(vocab.keys())[:10])