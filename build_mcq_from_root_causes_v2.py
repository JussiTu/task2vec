import random, re, json
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple
from collections import defaultdict
import numpy as np

from sentence_transformers import SentenceTransformer

random.seed(7)

QUESTION = "Explain data leakage and give two examples: one obvious and one subtle."
GOLD = (
    "Data leakage is when information that would not be available at prediction time influences training, "
    "feature engineering, or model selection, inflating evaluation. "
    "Obvious: including the target label (or a direct proxy) as a feature. "
    "Subtle: fitting preprocessing/feature selection on the full dataset before splitting, "
    "or using future information in time-series."
)

# ---------- helpers ----------
def polish(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def shorten(s: str, n_words=36) -> str:
    w = s.split()
    return " ".join(w[:n_words]) + ("…" if len(w) > n_words else "")

def extract_obvious(gold: str) -> str:
    m = re.search(r"Obvious:\s*([^\.]+)\.", gold, re.IGNORECASE)
    return m.group(1).strip() if m else "including the label as a feature"

STARTERS = [
    "In simple terms, ", "Basically, ", "A common view is that ",
    "From a practical standpoint, ", "In many projects, "
]
CONNECTORS = ["For example,", "In particular,", "That is,", "So,", "As a result,"]
ENDINGS = [
    "This is why evaluations can be misleading.",
    "This affects reported performance.",
    "This is a common pitfall."
]

def vary(text: str) -> str:
    t = text
    if random.random() < 0.55:
        t = random.choice(STARTERS) + t[0].lower() + t[1:]
    if random.random() < 0.35:
        t += " " + random.choice(CONNECTORS) + " " + random.choice(ENDINGS)
    swaps = [
        (r"\bonly\b", random.choice(["only", "solely", "exclusively"])),
        (r"\bprevent\b", random.choice(["prevent", "avoid", "eliminate"])),
        (r"\bimpossible\b", random.choice(["impossible", "not possible"])),
        (r"\bbasically\b", random.choice(["basically", "essentially"])),
    ]
    for pat, rep in swaps:
        if random.random() < 0.25:
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return polish(t)

BAD_TAIL_PATTERNS = [
    r"(?:As a result,|For example,|In particular,|That is,|So,)\s*This (?:is a common pitfall|affects reported performance|is why evaluations can be misleading)\.?$"
]

def finalize_option_text(s: str) -> str:
    s = polish(s)
    for pat in BAD_TAIL_PATTERNS:
        s = re.sub(pat, "", s).strip()
    s = re.sub(r"\s+\.$", ".", s)
    return s

# ---------- root causes ----------
@dataclass
class RootCause:
    id: str
    name: str
    generate_base: Callable[[str, str], str]
    feedback: str
    remediation: str

def rc1(q: str, gold: str) -> str:
    return ("Data leakage is basically the same as overfitting: the model memorizes the training set and fails to generalize. "
            "Obvious example: training too long. Subtle example: using a model that is too complex.")

def rc2(q: str, gold: str) -> str:
    obvious = extract_obvious(gold)
    return (f"Data leakage only happens when {obvious}. If you avoid that one mistake, leakage cannot occur. "
            "So the key is simply to ensure the target column never enters the feature set.")

def rc3(q: str, gold: str) -> str:
    return ("You prevent leakage by following the standard checklist: shuffle the data and do a clean train/test split once. "
            "If you do that, leakage is impossible. The subtle mistake is forgetting to shuffle enough.")

def rc5(q: str, gold: str) -> str:
    return ("Data leakage means sensitive data leaks out of the system (a privacy/security breach). "
            "Obvious example: exposing user records. Subtle example: attackers reconstructing data from logs. "
            "You prevent leakage using encryption and access control.")

def rc4(q: str, gold: str) -> str:
    return ("Leakage is mainly caused by preprocessing choices. For example, normalizing features before training can leak information. "
            "A subtle leakage source is using dropout or batch normalization, which can distort evaluation results.")

def rc6(q: str, gold: str) -> str:
    return ("Because evaluation results look good, this causes leakage during training. "
            "In other words, inflated accuracy creates leakage rather than leakage inflating evaluation.")

def rc7(q: str, gold: str) -> str:
    return ("Train, validation, and test are all part of the same dataset lifecycle, so it’s fine to use all of them during development. "
            "Leakage is only a concern if you literally copy the answers into the training data.")

def rc8(q: str, gold: str) -> str:
    return ("Cross-validation is leakage by definition because you reuse data across folds. "
            "Any method that evaluates multiple times on the same dataset contaminates the result.")

def rc9(q: str, gold: str) -> str:
    return ("Leakage is mostly theoretical; in practice it doesn’t depend on timing or deployment conditions. "
            "If offline evaluation is strong, it will transfer to production regardless of how data arrives over time.")

def rc10(q: str, gold: str) -> str:
    return ("If you optimize the training loss well enough, leakage stops mattering. "
            "A model with very low loss is robust, so evaluation contamination is not a real concern once you train properly.")

ROOT_CAUSES: List[RootCause] = [
    RootCause("RC1","Category confusion",rc1,
              "You’re describing overfitting. Leakage is about evaluation info contaminating training/selection.",
              "Study: overfitting vs. leakage; leakage via preprocessing/time."),
    RootCause("RC2","Single-case overgeneralization",rc2,
              "Label-in-feature is one leakage case, but leakage also happens via preprocessing, target encoding, time leakage, and selection bias.",
              "Study: leakage beyond the target column (pipelines, time-series splits)."),
    RootCause("RC3","Ritual compliance",rc3,
              "A random split helps, but doesn’t prevent leakage if preprocessing/feature selection uses all data or if time order matters.",
              "Practice: fit pipelines on train only; use time-based splits for temporal data."),
    RootCause("RC4","Surface-level fixation",rc4,
              "Normalization/dropout aren’t leakage by themselves. Leakage is eval info influencing training/selection.",
              "Study: when preprocessing becomes leakage (fit on full data)."),
    RootCause("RC5","Terminology collision",rc5,
              "That’s privacy/security leakage. In ML, data leakage means evaluation contamination inflating metrics.",
              "Study: ML evaluation leakage vs privacy leakage; separate mitigations."),
    RootCause("RC6","Causal inversion",rc6,
              "Leakage causes inflated evaluation; good-looking metrics do not ‘create’ leakage.",
              "Practice: audit data flow; compare clean vs contaminated pipelines."),
    RootCause("RC7","Boundary ignorance",rc7,
              "Validation is for tuning; test is for final reporting. Mixing them leaks evaluation information into decisions.",
              "Practice: strict train/val/test separation; never iterate on test."),
    RootCause("RC8","Tool absolutism",rc8,
              "Cross-validation can be valid. Leakage is unintended contamination, not simply reusing data correctly inside CV.",
              "Study: proper CV and nested CV for model selection."),
    RootCause("RC9","Scale blindness",rc9,
              "Timing and deployment conditions matter. Time leakage and distribution shift can break offline-to-prod transfer.",
              "Practice: time-aware evaluation; monitor drift; respect prediction-time availability."),
    RootCause("RC10","Optimization myopia",rc10,
              "Low loss doesn’t fix invalid evaluation. Leakage can make metrics look great while being invalid.",
              "Practice: add leakage checks; focus on evaluation validity, not just metrics.")
]

# ---------- generate variants & pick best per root cause ----------
N_PER_RC = 25
variants_by_rc: Dict[str, List[str]] = defaultdict(list)

def plausibility_score(s: str) -> float:
    n = len(s.split())
    score = 0.0
    if 18 <= n <= 70: score += 2.0
    if n < 10 or n > 110: score -= 2.0
    if any(w in s.lower() for w in ["obvious", "subtle", "train/test", "validation", "encryption", "shuffle"]): score += 0.6
    # penalize awkward endings if any remain
    if re.search(BAD_TAIL_PATTERNS[0], s): score -= 1.0
    return score

def best_variant(rc_id: str) -> str:
    ranked = sorted(variants_by_rc[rc_id], key=plausibility_score, reverse=True)
    return finalize_option_text(ranked[0])

for rc in ROOT_CAUSES:
    base = polish(rc.generate_base(QUESTION, GOLD))
    for _ in range(N_PER_RC):
        variants_by_rc[rc.id].append(vary(base))

best_distractor = {rc.id: best_variant(rc.id) for rc in ROOT_CAUSES}

# ---------- auto-select K distractors by semantic diversity ----------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return 1.0 - float(np.dot(a, b))

def select_diverse(rc_ids: List[str], k: int, model: SentenceTransformer) -> List[str]:
    embs = {rc_id: model.encode(best_distractor[rc_id]) for rc_id in rc_ids}
    ids = list(rc_ids)

    # start with farthest pair
    best_pair = None
    best_d = -1.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = cosine_distance(embs[ids[i]], embs[ids[j]])
            if d > best_d:
                best_d = d
                best_pair = (ids[i], ids[j])

    chosen = [best_pair[0], best_pair[1]]

    # greedy add: maximize min-distance to chosen set
    while len(chosen) < k:
        remaining = [x for x in ids if x not in chosen]
        best_cand = None
        best_score = -1.0
        for cand in remaining:
            mind = min(cosine_distance(embs[cand], embs[c]) for c in chosen)
            if mind > best_score:
                best_score = mind
                best_cand = cand
        chosen.append(best_cand)

    return chosen

print("Loading embedding model for distractor selection...")
model = SentenceTransformer("all-MiniLM-L6-v2")

all_rc_ids = [rc.id for rc in ROOT_CAUSES]
DISTRACTOR_RC_IDS = select_diverse(all_rc_ids, k=4, model=model)
# (You’ll get a diverse set; it might differ from RC2/RC3/RC5/RC1.)
# If you want to exclude some root causes from being distractors, we can do that too.

# ---------- build MCQ JSON ----------
options: List[Tuple[str, str, str]] = []
options.append(("A", shorten(GOLD, 45), "CORRECT"))

letter = "B"
for rc_id in DISTRACTOR_RC_IDS:
    options.append((letter, best_distractor[rc_id], rc_id))
    letter = chr(ord(letter) + 1)

answer_key = "A"

feedback_by_option = {}
for opt_id, _, tag in options:
    if tag == "CORRECT":
        feedback_by_option[opt_id] = "Correct. Leakage is evaluation contamination: info from eval influences training/selection."
    else:
        rc = next(r for r in ROOT_CAUSES if r.id == tag)
        feedback_by_option[opt_id] = rc.feedback

mcq_item = {
    "item_id": "ai_dl_001",
    "title": "Data leakage basics",
    "question": QUESTION,
    "options": [{"id": oid, "text": txt} for oid, txt, _ in options],
    "answer_key": answer_key,
    "diagnostics": {
        "option_to_root_cause": {oid: tag for oid, _, tag in options if tag != "CORRECT"},
        "feedback_by_option": feedback_by_option,
        "remediation_by_root_cause": {rc.id: rc.remediation for rc in ROOT_CAUSES},
    },
    "metadata": {
        "domain": "AI driving license",
        "concepts": ["data leakage", "evaluation validity", "train/val/test boundary"],
        "generation": {
            "variants_per_root_cause": N_PER_RC,
            "distractors_selected_by": "greedy_maximin_cosine_distance",
            "distractor_root_causes_used": DISTRACTOR_RC_IDS,
        }
    }
}

print("\n=== MCQ ITEM (JSON) ===\n")
print(json.dumps(mcq_item, indent=2, ensure_ascii=False))
