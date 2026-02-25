import random, re
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

random.seed(42)

QUESTION = "Explain data leakage and give two examples: one obvious and one subtle."

GOLD = (
    "Data leakage is when information that would not be available at prediction time influences training, "
    "feature engineering, or model selection, inflating evaluation. Obvious: including the target label (or a "
    "direct proxy) as a feature. Subtle: fitting preprocessing/feature selection on the full dataset before "
    "splitting, or using future information in time-series."
)

# Misconception “templates” (these are the knobs you’ll later replace with real LLM calls)
MISCONCEPTIONS = {
    "M1_overfitting": [
        "Data leakage is basically overfitting; the model memorizes training data and fails on test.",
        "Leakage happens when you train too long and the model overfits.",
        "Leakage is when the model is too complex and doesn't generalize."
    ],
    "M2_label_only": [
        "Leakage only happens if the label (or target) is included as a feature; otherwise there is no leakage.",
        "If you remove the target column from the inputs, leakage can’t happen.",
        "Leakage means the answer is accidentally included in the features."
    ],
    "M3_shuffle_ritual": [
        "As long as you shuffle randomly and split once into train/test, leakage is impossible.",
        "Leakage is prevented by random splitting; the only issue is not shuffling enough.",
        "If you do a random split, you're safe from leakage."
    ],
    "M4_privacy_confusion": [
        "Data leakage is about privacy—when sensitive data is exposed; encryption prevents leakage.",
        "Leakage means someone can steal the dataset; access control solves it.",
        "Leakage is a security issue, not a modeling issue."
    ],
    "M5_any_reuse": [
        "Leakage is any time you reuse data, like cross-validation, which makes results optimistic.",
        "If you evaluate more than once on a dataset, that's leakage.",
        "Using a validation set is leakage because you looked at results before the final test."
    ],
    "M6_preprocessing_misunderstood": [
        "Leakage is when you normalize features before training; normalization is always leakage.",
        "Using dropout or batch norm can cause leakage.",
        "Leakage is caused by regularization choices like dropout."
    ],
}

NOISE_PHRASES = [
    "So it's best to just train longer.",
    "This mainly depends on the learning rate.",
    "The fix is to use a bigger model.",
    "This is why accuracy may go up.",
    "Therefore cross-validation is bad.",
]

def generate_wrong_answers(n=40):
    """Generate n wrong answers by sampling misconception families + adding small variation."""
    wrong = []
    keys = list(MISCONCEPTIONS.keys())
    for _ in range(n):
        m = random.choice(keys)
        base = random.choice(MISCONCEPTIONS[m])
        # small random variation
        base = base.replace("leakage", random.choice(["leakage", "data leakage", "evaluation leakage"]))
        if random.random() < 0.35:
            base += " " + random.choice(NOISE_PHRASES)
        wrong.append((m, base))
    return wrong

wrong = generate_wrong_answers(n=60)
texts = [t for _, t in wrong]

print("\nQUESTION:\n", QUESTION)
print("\nGOLD ANSWER (reference):\n", GOLD)
print("\n--- Generated WRONG answers (first 10) ---")
for i, (m, t) in enumerate(wrong[:10], 1):
    print(f"{i:02d}. [{m}] {t}")

# ---- Analysis: cluster wrong answers by text similarity ----
# (In a real system you’d use embeddings; TF-IDF works for a runnable demo.)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
X = vectorizer.fit_transform(texts)

k = 6  # choose 5–8 typically
km = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = km.fit_predict(X)

clusters = defaultdict(list)
for (m, t), c in zip(wrong, labels):
    clusters[c].append((m, t))

# ---- Label clusters (simple heuristic) ----
def label_cluster(examples):
    joined = " ".join(t.lower() for _, t in examples)
    if "encrypt" in joined or "security" in joined or "access control" in joined:
        return "Confuses evaluation leakage with privacy/security leakage"
    if "overfit" in joined or "memor" in joined or "complex" in joined:
        return "Confuses leakage with overfitting/generalization"
    if "shuffle" in joined or "random split" in joined:
        return "Believes random split alone prevents leakage"
    if "label" in joined or "target column" in joined:
        return "Believes leakage only means label-in-feature"
    if "cross-validation" in joined or "reuse data" in joined or "validation set is leakage" in joined:
        return "Calls any data reuse/leaking evaluation (incl. CV) as leakage"
    if "dropout" in joined or "batch norm" in joined or "regularization" in joined:
        return "Misattributes leakage to training tricks (dropout/bn/regularization)"
    return "Mixed / unclear misconception cluster"

print("\n\n================ ANALYSIS REPORT ================\n")
for cid, items in sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True):
    print(f"CLUSTER {cid} — size {len(items)}")
    print("Label:", label_cluster(items))
    fams = Counter(m for m, _ in items)
    print("Underlying generator families:", dict(fams))
    print("\nTop 3 example answers:")
    for ex in items[:3]:
        print(" -", ex[1])
    print("\n")

# ---- Pick “best” distractor per cluster (simple heuristic: shortest plausible) ----
def plausibility_score(text):
    # crude heuristic: penalize obviously meta / too short, prefer concrete terms
    score = 0
    score += 2 if len(text.split()) >= 10 else -2
    score += 2 if any(w in text.lower() for w in ["train", "test", "label", "target", "split", "overfit", "privacy"]) else 0
    score -= 2 if "therefore" in text.lower() and "bad" in text.lower() else 0
    score -= 2 if "learning rate" in text.lower() else 0
    return score

best = {}
for cid, items in clusters.items():
    ranked = sorted(items, key=lambda it: plausibility_score(it[1]), reverse=True)
    best[cid] = ranked[0][1]

print("\n\n============= PROPOSED MCQ DISTRACTORS =============\n")
print("Correct option (shortened):", " ".join(GOLD.split()[:28]) + "…\n")
for cid, txt in best.items():
    print(f"Cluster {cid} distractor:", txt)
