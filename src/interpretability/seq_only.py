inp = "ig_outputs_all/foreground_ctx20_pos.fa"
out = "foreground_pos.fa"
'''
with open(inp) as f, open(out, "w") as g:
    for line in f:
        if not line.startswith(">"):
            g.write(line)

print("Saved:", out)

inp = "ig_outputs_all/foreground_ctx20_pos.fa"
out = "foreground_pos.fa"

with open(inp) as f, open(out, "w") as g:
    for line in f:
        if line.startswith(">"):
            # keep only gene ID (before first "|")
            gene_id = line.strip().split("|")[0]
            g.write(gene_id + "\n")
        else:
            g.write(line.strip() + "\n")

print("Saved:", out)
'''
inp = "ig_outputs_all/background_ctx20_neg.fa"
out = "background_neg.fa"

seen = {}

with open(inp) as f, open(out, "w") as g:
    for line in f:
        if line.startswith(">"):
            base_id = line.strip().split("|")[0].lstrip(">")
            seen[base_id] = seen.get(base_id, 0) + 1
            g.write(f">{base_id}_{seen[base_id]}\n")
        else:
            seq = line.strip()
            if seq:
                g.write(seq + "\n")

print("Saved:", out)


