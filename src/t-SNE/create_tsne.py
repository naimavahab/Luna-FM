import torch
import numpy as np
from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

BEST_DIR = "./rfam_best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BEST_DIR)
model = AutoModelForSequenceClassification.from_pretrained(BEST_DIR).to(device)
model.eval()

ds = load_dataset("multimolecule/archiveii")["test"]
#ds = load_from_disk("../../Data/rfam_longest_50k")
def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1)

embs, labels = [], []

with torch.no_grad():
    for seq, lab in zip(ds["sequence"], ds["family"]):
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=248)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs, output_hidden_states=True, return_dict=True)

        h = out.hidden_states[-2]   # best layer for clustering
        pooled = mean_pool(h, inputs["attention_mask"])

        embs.append(pooled.squeeze(0).cpu().numpy())
        labels.append(lab)

X = np.stack(embs)

tsne = TSNE(n_components=2, perplexity=50, random_state=42)
Z = tsne.fit_transform(X)

import matplotlib.cm as cm


labels = np.array(labels)

families = sorted(np.unique(labels))
n_classes = len(families)

# Use a wide colour palette (covers whole spectrum)
cmap = cm.get_cmap("tab20", n_classes)   # good for <= 20 classes

plt.figure(figsize=(10, 8))

for i, fam in enumerate(families):
    idx = labels == fam
    plt.scatter(
        Z[idx, 0],
        Z[idx, 1],
        s=10,
        color=cmap(i),
        label=fam,
        alpha=0.8
    )

plt.title("t-SNE of fine-tuned BiRNA-BERT embeddings (ArchiveII)")
plt.legend(
    title="Families",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    markerscale=2,
    fontsize=9
)

plt.tight_layout()
plt.savefig("rfam.png", dpi=300, bbox_inches="tight")

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import silhouette_score
le = LabelEncoder()
y = le.fit_transform(labels)

sil_tsne = silhouette_score(Z, y, metric="euclidean")
print("Silhouette score (t-SNE 2D):", sil_tsne)
