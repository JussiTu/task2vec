print("STEP 0: Script started")

import random
from collections import defaultdict

print("STEP 1: Imports (sklearn, numpy) ...")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("STEP 2: Importing sentence-transformers ...")
from sentence_transformers import SentenceTransformer

print("STEP 3: Importing hdbscan ...")
import hdbscan

print("STEP 4: Importing matplotlib ...")
import matplotlib.pyplot as plt

random.seed(42)

MISCONCEPTIONS = {
    "overfitting": [
        "Data leakage is basically overfitting; the model memorizes training data.",
        "Leakage happens when a model is too complex and overfits."
    ],
    "label_only": [
        "Leakage only happens if the label is included as a feature.",
        "If the target column is removed, leakage cannot occur."
    ],
    "shuffle_ritual": [
        "Randomly shuffling and splitting once prevents all leakage.",
        "If you do a random split, leakage is impossible."
    ],
    "privacy": [
        "Data leakage is about privacy—when sensitive data is exposed; encryption prevents leakage.",
        "Leakage means someone can steal the dataset; access control solves it."
    ],
    "any_reuse": [
        "Leakage is any time you reuse data, like cross-validation, which makes results optimistic.",
        "Using a validation set is leakage."
    ],
    "preprocessing": [
        "Normalizing features before training is leakage.",
        "Dropout and batch normalization cause leakage."
    ]
}

def generate_wrong(n=60):
    wrong = []
    keys = list(MISCONCEPTIONS.keys())
    for _ in range(n):
        k = random.choice(keys)
        t = random.choice(MISCONCEPTIONS[k])
        wrong.append((k, t))
    return wrong

print("STEP 5: Generating synthetic wrong answers ...")
wrong = generate_wrong(60)
texts = [t for _, t in wrong]
print("Generated:", len(texts), "answers")

print("STEP 6: Loading embedding model (first run may download) ...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded")

print("STEP 7: Encoding sentences ...")
embeddings = model.encode(texts)
print("Embeddings shape:", embeddings.shape)

print("STEP 8: Scaling embeddings ...")
scaler = StandardScaler()
E = scaler.fit_transform(embeddings)

print("STEP 9: Clustering with HDBSCAN ...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1)
labels = clusterer.fit_predict(E)
print("Unique cluster labels:", sorted(set(labels)))

print("STEP 10: Cluster report ...")
cluster_report = defaultdict(list)
for lbl, txt in zip(labels, texts):
    cluster_report[lbl].append(txt)

for cid, exs in sorted(cluster_report.items(), key=lambda kv: len(kv[1]), reverse=True):
    print(f"\nCluster {cid}: {len(exs)} samples")
    for e in exs[:3]:
        print(" -", e)

print("\nSTEP 11: PCA -> 2D plot (optional) ...")
pca = PCA(n_components=2)
pts = pca.fit_transform(E)

plt.scatter(pts[:, 0], pts[:, 1], c=labels, s=40, alpha=0.7)
plt.title("HDBSCAN clusters of synthetic wrong answers (PCA projection)")
plt.show()

print("DONE")

