import random
import re
from dataclasses import dataclass
from typing import Callable, List

random.seed(7)

QUESTION = "Explain data leakage and give two examples: one obvious and one subtle."
GOLD = (
    "Data leakage is when information that would not be available at prediction time influences training, "
    "feature engineering, or model selection, inflating evaluation. Obvious: including the target label (or a "
    "direct proxy) as a feature. Subtle: fitting preprocessing/feature selection on the full dataset before "
    "splitting, or using future information in time-series."
)

# -----------------------------
# Helpers
# -----------------------------
def _extract_obvious_example(gold: str) -> str:
    m = re.search(r"Obvious:\s*([^\.]+)\.", gold, re.IGNORECASE)
    return m.group(1).strip() if m else "including the label as a feature"

def _extract_subtle_example(gold: str) -> str:
    m = re.search(r"Subtle:\s*([^\.]+)\.", gold, re.IGNORECASE)
    return m.group(1).strip() if m else "fitting preprocessing on all data before splitting"

def _shorten(s: str, n_words: int = 28) -> str:
    w = s.split()
    return " ".join(w[:n_words]) + ("…" if len(w) > n_words else "")

def _polish(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------------
# Root-cause generators (ALL take: (question, gold))
# -----------------------------
def _category_confusion(question: str, gold: str) -> str:
    # Confuse leakage with overfitting/generalization
    return (
        "Data leakage is basically the same as overfitting: the model memorizes the training set and fails to generalize. "
        "Obvious example: training too long. Subtle example: using a model that is too complex."
    )

def _single_case_overgeneralization(question: str, gold: str) -> str:
    obvious = _extract_obvious_example(gold)
    return (
        f"Data leakage only happens when {obvious}. "
        "If you avoid that one specific mistake, leakage cannot occur. "
        "So the key is simply to ensure the target column never enters the feature set."
    )

def _ritual_compliance(question: str, gold: str) -> str:
    return (
        "You prevent leakage by following the standard checklist: shuffle the data and do a clean train/test split once. "
        "If you do that, leakage is impossible. An obvious mistake is forgetting to shuffle; a subtle mistake is shuffling too little."
    )

def _surface_level_fixation(question: str, gold: str) -> str:
    return (
        "Leakage is mainly caused by preprocessing choices. For example, normalizing features before training can leak information. "
        "A subtle leakage source is using dropout or batch normalization, which can distort evaluation results."
    )

def _terminology_collision(question: str, gold: str) -> str:
    return (
        "Data leakage means sensitive data leaks out of the system (a privacy/security breach). "
        "Obvious example: exposing user records. Subtle example: attackers reconstructing data from logs. "
        "You prevent leakage using encryption and access control."
    )

def _causal_inversion(question: str, gold: str) -> str:
    return (
        "Because evaluation results look good, this causes leakage during training. "
        "In other words, inflated accuracy creates leakage rather than leakage inflating evaluation."
    )

def _boundary_ignorance(question: str, gold: str) -> str:
    return (
        "Train, validation, and test are all part of the same dataset lifecycle, so it’s fine to use all of them during development. "
        "If you later report the final performance, it will still be representative. "
        "Leakage is only a concern if you literally copy the answers into the training data."
    )

def _tool_absolutism(question: str, gold: str) -> str:
    return (
        "Cross-validation is leakage by definition, because you reuse data across folds. "
        "Any method that evaluates multiple times on the same dataset contaminates the result, so it should be avoided."
    )

def _scale_blindness(question: str, gold: str) -> str:
    return (
        "Leakage is mostly a theoretical issue; in practice it doesn’t depend on timing or deployment conditions. "
        "If your offline evaluation is strong, it will transfer to production regardless of how data arrives over time."
    )

def _optimization_myopia(question: str, gold: str) -> str:
    return (
        "If you optimize the training loss well enough, leakage stops mattering. "
        "A model with very low loss is robust, so evaluation contamination is not a real concern once you train properly."
    )

# -----------------------------
# RootCause registry
# -----------------------------
@dataclass
class RootCause:
    id: str
    name: str
    description: str
    generate: Callable[[str, str], str]  # (question, gold) -> wrong answer
    feedback: str

ROOT_CAUSES: List[RootCause] = [
    RootCause(
        id="RC1",
        name="Category confusion",
        description="Confuse a related concept with the target concept.",
        generate=_category_confusion,
        feedback="Leakage is an information-boundary violation (eval info influencing training/selection), not overfitting."
    ),
    RootCause(
        id="RC2",
        name="Single-case overgeneralization",
        description="Assume the most common example is the only case.",
        generate=_single_case_overgeneralization,
        feedback="Label-in-feature is one leakage case; pipeline steps and time leakage are also common."
    ),
    RootCause(
        id="RC3",
        name="Ritual compliance",
        description="Assume a standard checklist step guarantees correctness.",
        generate=_ritual_compliance,
        feedback="Random splitting helps but does not prevent pipeline leakage or time leakage."
    ),
    RootCause(
        id="RC4",
        name="Surface-level feature fixation",
        description="Blame a low-level operation instead of the real system mechanism.",
        generate=_surface_level_fixation,
        feedback="Preprocessing isn’t leakage by itself; leakage is when preprocessing is fit using evaluation data."
    ),
    RootCause(
        id="RC5",
        name="Terminology collision",
        description="Use everyday meaning instead of technical meaning.",
        generate=_terminology_collision,
        feedback="That’s privacy leakage; ML data leakage refers to evaluation contamination."
    ),
    RootCause(
        id="RC6",
        name="Causal inversion",
        description="Swap cause and effect.",
        generate=_causal_inversion,
        feedback="Leakage causes inflated evaluation, not the other way around."
    ),
    RootCause(
        id="RC7",
        name="Boundary ignorance",
        description="Ignore interfaces/phases; collapse train/val/test boundaries.",
        generate=_boundary_ignorance,
        feedback="Validation is for tuning; test is for final reporting. Mixing them leaks evaluation information."
    ),
    RootCause(
        id="RC8",
        name="Tool/method absolutism",
        description="Treat a method as universally safe/unsafe.",
        generate=_tool_absolutism,
        feedback="Cross-validation can be valid; leakage is unintended eval contamination, not mere reuse."
    ),
    RootCause(
        id="RC9",
        name="Scale blindness",
        description="Ignore time/scale/operational constraints.",
        generate=_scale_blindness,
        feedback="Time dependence matters (e.g., time leakage). Offline results may not transfer if evaluation is contaminated."
    ),
    RootCause(
        id="RC10",
        name="Optimization myopia",
        description="Assume improving one metric solves the whole system.",
        generate=_optimization_myopia,
        feedback="Low loss doesn’t fix evaluation contamination; leakage can make metrics look good while being invalid."
    ),
]

# -----------------------------
# Generate 10 incorrect open answers (1 per root cause)
# -----------------------------
def generate_wrong_answers(question: str, gold: str):
    out = []
    for rc in ROOT_CAUSES:
        wrong = _polish(rc.generate(question, gold))
        out.append({
            "root_cause_id": rc.id,
            "root_cause_name": rc.name,
            "wrong_open_answer": wrong,
            "diagnostic_feedback": rc.feedback
        })
    return out

# -----------------------------
# Run demo
# -----------------------------
wrong_set = generate_wrong_answers(QUESTION, GOLD)

print("\nQUESTION:\n", QUESTION)
print("\nGOLD (short):\n", _shorten(GOLD), "\n")

print("=== SYNTHETIC WRONG OPEN ANSWERS (10 root causes) ===\n")
for item in wrong_set:
    print(f"[{item['root_cause_id']}] {item['root_cause_name']}")
    print("Wrong answer:", item["wrong_open_answer"])
    print("Feedback:", item["diagnostic_feedback"])
    print("-" * 80)
