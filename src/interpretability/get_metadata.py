# count_fasta_stats.py

from statistics import mean

fasta_file = "background_neg.fa"   # change this

seq_lengths = []
current_seq = []

with open(fasta_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq:
                seq_lengths.append(len("".join(current_seq)))
                current_seq = []
        else:
            current_seq.append(line)

# add last sequence
if current_seq:
    seq_lengths.append(len("".join(current_seq)))

num_seqs = len(seq_lengths)
avg_len = mean(seq_lengths) if seq_lengths else 0

print("Number of sequences:", num_seqs)
print("Average length:", round(avg_len, 2))
print("Min length:", min(seq_lengths) if seq_lengths else 0)
print("Max length:", max(seq_lengths) if seq_lengths else 0)
