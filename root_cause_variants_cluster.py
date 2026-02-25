import random
import re
from dataclasses import dataclass
from typing import Callable, List, Dict
from collections import defaultdict, Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
import hdbscan

random.seed(7)

QUESTION = "Explain data leakage and give two examples: one obvious and one subtle."
GOLD = (
    "Data leakage is when information that would not be available at prediction time influences training, "
    "feature engineering, or model selection, inflating evaluation. Obvious: including the target label (or a "
    "direct proxy) as a feature. Subtle: fitting preprocessing/feature selection on the full dataset before "
    "splitting, or using future information in time-series."
)

# ---------- Helpers ----------
def polish(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def shorten(s: str, n_words=22) -> str:
    w = s.split()
    return " ".join(w[:n_words]) + ("…" if len(w) > n_words else "")

def extract_obvious(gold: str) -> str:
    m = re.search(r"Obvious:\s*([^\.]+)\.", gold, re.IGNORECASE)
    return m.group(1).strip() if m else "including the label as a feature"

def extract_subtle(gold: str) -> str:
    m = re.search(r"Subtle:\s*([^\.]+)\.", gold, re.IGNORECASE)
    return m.group(1).strip() if m else "fitting preprocessing on all data before splitting"

# This “variation engine” is the key: it increases linguistic diversity *without* an LLM.
STARTERS = [
    "In simple terms, ",
    "Basically, ",
    "A common view is that ",
    "From a practical standpoint, ",
    "In many projects, "
]
HEDGES = ["often", "usually", "in practice", "typically", "most of the time"]
CONNECTORS = ["For example,", "In particular,", "That is,", "So,", "As a result,"]
ENDINGS = [
    "This is why evaluations can be misleading.",
    "That’s the main issue.",
    "This affects reported performance.",
    "This is a common pitfall.",
    "This can make results look better than they are."
]

def vary(text: str) -> str:
    """Create a paraphrase-ish variant using surface transformations."""
    t = text

    # Swap a few phrases to create diversity
    swaps = [
        (r"\bis\b", random.choice(["is", "is basically", "can be seen as"])),
        (r"\bprevent\b", random.choice(["prevent", "avoid", "eliminate"])),
        (r"\bimpossible\b", random.choice(["impossible", "not possible", "ruled out"])),
        (r"\bonly\b", random.choice(["only", "solely", "exclusively"])),
        (r"\bcauses\b", random.choice(["causes", "leads to", "creates"])),
        (r"\binfluences\b", random.choice(["influences", "affects", "biases"]))
    ]
    for pattern, repl in swaps:
        if random.random() < 0.25:
            t = re.sub(pattern, repl, t, flags=re.IGNORECASE)

    # Add a random starter/connector/ending sometimes
    if random.random() < 0.50:
        t = random.choice(STARTERS) + t[0].lower() + t[1:]
    if random.random() < 0.35:
        t += " " + random.choice(CONNECTORS) + " " + random.choice(ENDINGS)
    if random.random() < 0.25:
        t = t.replace(" is ", f" {random.choice(HEDGES)} is ", 1)

    return polish(t)

# ---------- Root causes (generators) ----------
@dataclass
class RootCause:
    id: str
    name: str
    generate_base: Callable[[str, str], str]
    feedback: str

def rc1_category_confusion(q: str, gold: str) -> str:
    return (
        "Data leakage is basically the same as overfitting: the model memorizes the training set and fails to generalize. "
        "Obvious example: training too long. Subtle example: using a model that is too complex."
    )

def rc2_single_case(q: str, gold: str) -> str:
    obvious = extract_obvious(gold)
    return (
        f"Data leakage only happens when {obvious}. If you avoid that one mistake, leakage cannot occur. "
        "So the key is simply to ensure the target column never enters the feature set."
    )

def rc3_ritual(q: str, gold: str) -> str:
    return (
        "You prevent leakage by following the standard checklist: shuffle the data and do a clean train/test split once. "
        "If you do that, leakage is impossible. The subtle mistake is forgetting to shuffle enough."
    )

def rc4_surface_fixation(q: str, gold: str) -> str:
    return (
        "Leakage is mainly caused by preprocessing choices. For example, normalizing features before training can leak information. "
        "A subtle leakage source is using dropout or batch normalization, which can distort evaluation results."
    )

def rc5_terminology_collision(q: str, gold: str) -> str:
    return (
        "Data leakage means sensitive data leaks out of the system (a privacy/security breach). "
        "Obvious example: exposing user records. Subtle example: attackers reconstructing data from logs. "
        "You prevent leakage using encryption and access control."
    )

def rc6_causal_inversion(q: str, gold: str) -> str:
    return (
        "Because evaluation results look good, this causes leakage during training. "
        "In other words, inflated accuracy creates leakage rather than leakage inflating evaluation."
    )

def rc7_boundary_ignorance(q: str, gold: str) -> str:
    return (
        "Train, validation, and test are all part of the same dataset lifecycle, so it’s fine to use all of them during development. "
        "Leakage is only a concern if you literally copy the answers into the training data."
    )

def rc8_tool_absolutism(q: str, gold: str) -> str:
    return (
        "Cross-validation is leakage by definition because you reuse data across folds. "
        "Any method that evaluates multiple times on the same dataset contaminates the result."
    )

def rc9_scale_blindness(q: str, gold: str) -> str:
    return (
        "Leakage is mostly theoretical; in practice it doesn’t depend on timing or deployment conditions. "
        "If offline evaluation is strong, it will transfer to production regardless of how data arrives over time."
    )

def rc10_optimization_myopia(q: str, gold: str) -> str:
    return (
        "If you optimize the training loss well enough, leakage stops mattering. "
        "A model with very low loss is robust, so evaluation contamination is not a real concern once you train properly."
    )

ROOT_CAUSES: List[RootCause] = [
    RootCause("RC1", "Category confusion", rc1_category_confusion,
              "Leakage is eval-boundary contamination, not overfitting."),
    RootCause("RC2", "Single-case overgeneralization", rc2_single_case,
              "Label-in-feature is one leakage case; pipeline/time leakage also exist."),
    RootCause("RC3", "Ritual compliance", rc3_ritual,
              "Random split helps but doesn’t prevent leakage from pipelines/time."),
    RootCause("RC4", "Surface-level feature fixation", rc4_surface_fixation,
              "Preprocessing isn’t leakage unless fit using eval data."),
    RootCause("RC5", "Terminology collision", rc5_terminology_collision,
              "Privacy leakage ≠ ML data leakage (eval contamination)."),
    RootCause("RC6", "Causal inversion", rc6_causal_inversion,
              "Leakage causes inflated evaluation, not vice versa."),
    RootCause("RC7", "Boundary ignorance", rc7_boundary_ignorance,
              "Val is for tuning; test is for final reporting; mixing leaks info."),
    RootCause("RC8", "Tool/method absolutism", rc8_tool_absolutism,
              "Cross-validation can be valid; leakage is unintended contamination."),
    RootCause("RC9", "Scale blindness", rc9_scale_blindness,
              "Time/scale matter; leakage can invalidate offline → prod assumptions."),
    RootCause("RC10", "Optimization myopia", rc10_optimization_myopia,
              "Low loss doesn’t fix invalid evaluation caused by leakage."),
]

# ---------- Generate N variants per root cause ----------
N_PER_ROOT_CAUSE = 25  # change to 50 if you want a bigger demo

records = []
for rc in ROOT_CAUSES:
    base = polish(rc.generate_base(QUESTION, GOLD))
    for i in range(N_PER_ROOT_CAUSE):
        variant = vary(base)
        records.append({
            "root_cause_id": rc.id,
            "root_cause_name": rc.name,
            "text": variant
        })

texts = [r["text"] for r in records]
true_labels = [r["root_cause_id"] for r in records]

print("QUESTION:", QUESTION)
print("GOLD (short):", shorten(GOLD))
print(f"\nGenerated {len(texts)} synthetic wrong answers "
      f"({N_PER_ROOT_CAUSE} variants × {len(ROOT_CAUSES)} root causes).")

print("\nSample variants (first 6):")
for r in records[:6]:
    print(f"- [{r['root_cause_id']}] {shorten(r['text'], 18)}")

# ---------- Embeddings ----------
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoding...")
E = model.encode(texts, show_progress_bar=True)
print("Embeddings:", E.shape)

# ---------- Scale + HDBSCAN ----------
E_scaled = StandardScaler().fit_transform(E)

print("\nClustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=2)
cluster_labels = clusterer.fit_predict(E_scaled)

print("Unique cluster labels:", sorted(set(cluster_labels)))
print("Noise count (-1):", int(np.sum(cluster_labels == -1)))

# ---------- Report: for each discovered cluster, which root causes appear? ----------
clusters: Dict[int, List[int]] = defaultdict(list)
for idx, c in enumerate(cluster_labels):
    clusters[int(c)].append(idx)

print("\n=== CLUSTER PURITY REPORT ===")
for c, idxs in sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True):
    rc_counts = Counter(true_labels[i] for i in idxs)
    total = len(idxs)
    top_rc, top_n = rc_counts.most_common(1)[0]
    purity = top_n / total
    print(f"\nCluster {c:>2} | size={total:>3} | top={top_rc} ({top_n}) | purity={purity:.2f}")
    print("Root causes in cluster:", dict(rc_counts.most_common(5)))
    print("Examples:")
    for i in idxs[:3]:
        print(f" - [{true_labels[i]}] {texts[i]}")

# ---------- Plot (PCA projection for visualization only) ----------
print("\nPlotting PCA projection...")
P = PCA(n_components=2).fit_transform(E_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(P[:, 0], P[:, 1], c=cluster_labels, s=35, alpha=0.75)
plt.title("HDBSCAN clusters of root-cause synthetic wrong answers (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
