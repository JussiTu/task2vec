import random
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Synthetic Data
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

# Generate synthetic wrong answers
def generate_wrong(n=60):
    wrong = []
    keys = list(MISCONCEPTIONS.keys())
    for _ in range(n):
        k = random.choice(keys)
        t = random.choice(MISCONCEPTIONS[k])
        wrong.append((k, t))
    return wrong

# Get 60 wrong answers
wrong = generate_wrong(60)
texts = [t for _, t in wrong]

# --- Step 1: Generate Embeddings using Sentence-BERT
print("Generating sentence embeddings using Sentence-BERT...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight, high-performance model
embeddings = model.encode(texts)

# --- Step 2: Cluster with HDBSCAN
print("Clustering using HDBSCAN...")
# Standardize data (HDBSCAN is sensitive to scale)
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Apply HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1)  # Min samples is 1 so that noise can be detected
labels = clusterer.fit_predict(embeddings_scaled)

# --- Step 3: Visualize clusters (2D PCA projection for simplicity)
print("Visualizing clusters...")

# Reduce to 2D for plotting
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', s=50, alpha=0.7)

# Color legend for clusters
for i in range(len(set(labels))):
    plt.scatter([], [], c=plt.cm.Spectral(i / len(set(labels))), label=f"Cluster {i}")

plt.title('HDBSCAN Clustering of Synthetic Wrong Answers')
plt.legend()
plt.show()

# --- Step 4: Print cluster report
print("\n=== Cluster Analysis ===")
cluster_report = defaultdict(list)

for i, (label, text) in enumerate(zip(labels, texts)):
    cluster_report[label].append(text)

# Print out the cluster details
for cluster_id, examples in cluster_report.items():
    print(f"\nCluster {cluster_id} - {len(examples)} samples")
    for example in examples[:3]:  # Print top 3 examples
        print(f"  - {example}")
