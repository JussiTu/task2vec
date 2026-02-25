import random
from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

print("=== START DEBUG ANALYSIS ===")

random.seed(42)

QUESTION = "Explain data leakage and give two examples: one obvious and one subtle."

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
        "Data leakage is about privacy breaches; encryption prevents it.",
        "Leakage means sensitive data is exposed."
    ],
    "any_reuse": [
        "Any reuse of data like cross-validation is leakage.",
        "Using a validation set is leakage."
    ],
    "preprocessing": [
        "Normalizing features before training is leakage.",
        "Dropout and batch normalization cause leakage."
    ]
}

def generate_wrong(n=40):
    out = []
    keys = list(MISCONCEPTIONS.keys())
    for _ in range(n):
        k = random.choice(keys)
        t = random.choice(MISCONCEPTIONS[k])
        out.append((k, t))
    return out

wrong = generate_wrong(40)
texts = [t for _, t in wrong]

print("\nFirst 10 wrong answers:")
for i, (k, t) in enumerate(wrong[:10], 1):
    print(f"{i:02d}. [{k}] {t}")

print("\nVectorizing...")
vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(texts)

print("Clustering...")
km = KMeans(n_clusters=6, random_state=42, n_init=10)
labels = km.fit_predict(X)

clusters = defaultdict(list)
for (k, t), c in zip(wrong, labels):
    clusters[c].append((k, t))

print("\n=== ANALYSIS REPORT ===\n")
for cid, items in clusters.items():
    print(f"CLUSTER {cid} (size {len(items)})")
    print("Families:", Counter(k for k, _ in items))
    for _, t in items[:3]:
        print(" -", t)
    print()

print("=== DONE ===")
