import json, textwrap, random
from collections import defaultdict

random.seed(7)

# -----------------------------
# STEP 0 — Input (what you scraped from a page)
# -----------------------------
SOURCE_TEXT = """
An AI system’s behavior depends on data, model, and objective. Training minimizes a loss on training data; generalization is performance on unseen data. Overfitting happens when a model fits noise or spurious patterns in training data. Train/validation/test splits help estimate generalization and tune models without leaking test information. Data leakage occurs when information from the evaluation set influences training or decisions, producing overly optimistic results.

Evaluation metrics must match the task and impact: accuracy can be misleading with class imbalance; precision/recall trade off; calibration measures whether predicted probabilities reflect true frequencies.

Large language models can hallucinate: they may produce plausible but incorrect statements, especially when asked for specific facts beyond context. Retrieval and verification can reduce risk.

Fairness concerns arise when models perform differently across groups. Mitigations include auditing, improving data, and constraining objectives, but tradeoffs exist. Privacy risks include memorization and re-identification; techniques like minimization, access control, and differential privacy can help.
""".strip()

def pretty(obj): return json.dumps(obj, indent=2, ensure_ascii=False)

print("STEP 0 — Input text (excerpt)\n")
print(textwrap.fill(SOURCE_TEXT[:420] + " ...", width=100))
print("\n" + "="*100 + "\n")


# -----------------------------
# STEP 1 — Concept extraction (simulated)
# -----------------------------
CONCEPT_CANDIDATES = [
    ("Generalization vs. overfitting", ["generalization", "overfitting", "spurious", "noise", "unseen"]),
    ("Train/validation/test splits", ["train", "validation", "test", "split"]),
    ("Data leakage", ["leakage", "evaluation", "influences", "optimistic"]),
    ("Metrics under class imbalance", ["accuracy", "imbalance", "precision", "recall"]),
    ("Calibration", ["calibration", "probabilities", "frequencies"]),
    ("LLM hallucinations", ["hallucinate", "plausible", "incorrect"]),
    ("Retrieval & verification", ["retrieval", "verification", "reduce", "risk"]),
    ("Fairness", ["fairness", "groups", "tradeoffs"]),
    ("Privacy in ML/LLMs", ["privacy", "memorization", "re-identification", "differential"])
]

def extract_concepts(text, top_n=6):
    text_l = text.lower()
    scored = []
    for name, kws in CONCEPT_CANDIDATES:
        score = sum(text_l.count(k.lower()) for k in kws)
        if score:
            scored.append((score, name))
    scored.sort(reverse=True)
    picked = [name for _, name in scored[:top_n]]

    notes = []
    for name in picked:
        if name == "Generalization vs. overfitting":
            notes.append({"concept_name": name,
                          "why_it_is_important": "Determines whether a model will work on new, real-world data rather than only the training set.",
                          "why_learners_fail": "People confuse high training performance with real-world success and miss spurious correlations."})
        elif name == "Train/validation/test splits":
            notes.append({"concept_name": name,
                          "why_it_is_important": "Separates learning, tuning, and final evaluation so you can estimate performance on unseen data.",
                          "why_learners_fail": "Learners accidentally tune on the test set or reuse information across splits."})
        elif name == "Data leakage":
            notes.append({"concept_name": name,
                          "why_it_is_important": "Prevents inflated evaluation results that can lead to unsafe deployment decisions.",
                          "why_learners_fail": "They think leakage only means 'label in features' or confuse it with privacy leaks."})
        elif name == "Metrics under class imbalance":
            notes.append({"concept_name": name,
                          "why_it_is_important": "Choosing the wrong metric can hide critical failures in rare but important cases.",
                          "why_learners_fail": "Accuracy feels intuitive, so learners ignore precision/recall and base-rate effects."})
        elif name == "Calibration":
            notes.append({"concept_name": name,
                          "why_it_is_important": "When probabilities drive decisions, calibration determines whether 0.8 really means ~80% chance.",
                          "why_learners_fail": "They assume model confidence is automatically trustworthy."})
        elif name == "LLM hallucinations":
            notes.append({"concept_name": name,
                          "why_it_is_important": "Hallucinations create trust and safety risks because outputs can sound confident but be false.",
                          "why_learners_fail": "Fluency is mistaken for truth; users don’t add grounding or abstention behavior."})
        elif name == "Retrieval & verification":
            notes.append({"concept_name": name,
                          "why_it_is_important": "Grounding answers in sources and verifying claims reduces hallucination risk.",
                          "why_learners_fail": "They treat the model as a 'knowledge oracle' and skip system-level safeguards."})
        elif name == "Fairness":
            notes.append({"concept_name": name,
                          "why_it_is_important": "Performance differences across groups can cause harm and legal/compliance issues.",
                          "why_learners_fail": "They expect one universal fairness metric and overlook tradeoffs."})
        elif name == "Privacy in ML/LLMs":
            notes.append({"concept_name": name,
                          "why_it_is_important": "Models can leak sensitive information via memorization or re-identification.",
                          "why_learners_fail": "They assume training data use is automatically safe and underestimate linkage risks."})
        else:
            notes.append({"concept_name": name, "why_it_is_important": "Key AI concept.", "why_learners_fail": "Common misunderstandings exist."})
    return notes

concepts = extract_concepts(SOURCE_TEXT, top_n=6)
print("STEP 1 — Extract concepts (result)\n")
print(pretty(concepts))
print("\n" + "="*100 + "\n")


# -----------------------------
# STEP 2 — Open question generation (simulated)
# -----------------------------
QUESTION_BANK = {
    "Generalization vs. overfitting": [
        "Why can a model have excellent training performance but fail in production? Give two concrete mechanisms.",
        "Name two practical steps to detect and reduce overfitting during model development."
    ],
    "Train/validation/test splits": [
        "How should train/validation/test sets be used when selecting hyperparameters and reporting final performance?",
        "What is the purpose of a validation set versus a test set?"
    ],
    "Data leakage": [
        "Explain data leakage and give two examples: one obvious and one subtle.",
        "In a time-series prediction task, what is 'time leakage' and how do you prevent it?"
    ],
    "Metrics under class imbalance": [
        "Why is accuracy often misleading for imbalanced classification, and what metrics would you use instead?",
        "Give an example where a model with 99% accuracy can still be useless."
    ],
    "Calibration": [
        "What does it mean for a model to be calibrated, and why might calibration matter in decision-making?",
        "How can you check calibration, and name one method to improve it?"
    ],
    "LLM hallucinations": [
        "What is an LLM hallucination, and why does it happen even when responses sound confident?",
        "Name two system-level techniques to reduce hallucination risk in an LLM application and explain how they help."
    ],
    "Retrieval & verification": [
        "Explain how retrieval-augmented generation reduces hallucinations and one limitation it still has.",
        "What are two ways to verify LLM outputs in production systems?"
    ],
    "Fairness": [
        "Describe one fairness risk in ML systems and a mitigation approach, including a tradeoff.",
        "Why can improving overall accuracy worsen fairness for a minority group?"
    ],
    "Privacy in ML/LLMs": [
        "Give two privacy risks specific to ML/LLMs and one mitigation strategy for each.",
        "What is memorization in language models and why is it risky?"
    ]
}

def generate_open_questions(concepts, per_concept=2):
    out = []
    for c in concepts:
        name = c["concept_name"]
        qs = QUESTION_BANK.get(name, [])[:per_concept]
        out.append({"concept": name, "questions": qs})
    return out

open_q = generate_open_questions(concepts, per_concept=2)
print("STEP 2 — Generate open questions (result)\n")
print(pretty(open_q))
print("\n" + "="*100 + "\n")

questions = [{"concept": item["concept"], "question": q}
             for item in open_q for q in item["questions"]]


# -----------------------------
# STEP 3 — Gold answers (simulated)
# -----------------------------
GOLD_ANSWERS = {
    "Explain data leakage and give two examples: one obvious and one subtle.": (
        "Data leakage is when information that would not be available at prediction time influences training, "
        "feature engineering, or model selection, inflating evaluation. Obvious: including the target label (or a "
        "direct proxy) as a feature. Subtle: fitting preprocessing/feature selection on the full dataset before "
        "splitting, or using future information in time-series (e.g., aggregates that include post-prediction data)."
    ),
    "Name two system-level techniques to reduce hallucination risk in an LLM application and explain how they help.": (
        "1) Retrieval-augmented generation: fetch relevant sources and ground generation in them. "
        "2) Verification/guardrails: enforce citations, run fact-check steps/tool calls, and allow abstention when uncertain."
    ),
    "Why is accuracy often misleading for imbalanced classification, and what metrics would you use instead?": (
        "With imbalance, a model can predict the majority class and still get high accuracy while missing rare but critical cases. "
        "Use precision/recall, F1, ROC-AUC/PR-AUC, and evaluate per-class performance; choose metrics aligned with costs."
    ),
    "What does it mean for a model to be calibrated, and why might calibration matter in decision-making?": (
        "Calibration means predicted probabilities match observed frequencies (e.g., among 0.7 predictions, ~70% are positive). "
        "It matters when decisions depend on risk thresholds, expected value, or resource allocation."
    )
}

def generate_gold_answers(questions):
    out = []
    for q in questions:
        stem = q["question"]
        ans = GOLD_ANSWERS.get(stem, "A concise, technically correct explanation with an example and correct use of terms.")
        out.append({"question": stem, "correct_answer": ans})
    return out

gold = generate_gold_answers(questions)
gold_map = {x["question"]: x["correct_answer"] for x in gold}

print("STEP 3 — Generate gold answers (result excerpt: 3 items)\n")
print(pretty(gold[:3]))
print("\n" + "="*100 + "\n")


# -----------------------------
# STEP 4 — Wrong answer generation (misconception simulator)
# -----------------------------
MISCONCEPTION_TEMPLATES = [
    ("Confusing related concepts", [
        "this is basically the same as {confused_with}",
        "it means {confused_with} happens during training"
    ]),
    ("Overgeneralizing a valid principle", [
        "it happens whenever you {overgen_rule}, so it's always a problem",
        "any time you {overgen_rule}, that's essentially it"
    ]),
    ("Ignoring necessary conditions or factors", [
        "it only happens when {only_case}; otherwise it can't happen",
        "as long as you avoid {only_case}, you're safe"
    ]),
    ("Reversing cause and effect", [
        "{effect} causes {cause}, so that's the key issue",
        "because {effect} happens, it leads to {cause}"
    ]),
    ("Using memorized rules incorrectly", [
        "if you {ritual}, then it's impossible, so you're done",
        "the correct rule is: just {ritual}"
    ])
]

DEFAULT_PARAMS = dict(
    confused_with="a different ML issue",
    overgen_rule="do anything more than once",
    only_case="make a single obvious mistake",
    effect="a metric changes",
    cause="the result becomes invalid",
    ritual="follow a simple checklist step"
)

PARAMS_BY_CONCEPT = {
    "Data leakage": dict(
        confused_with="overfitting or having too little data",
        overgen_rule="reuse data or do cross-validation",
        only_case="including labels as features",
        effect="the model overfits",
        cause="the test set becomes contaminated",
        ritual="shuffle and randomly split the data once"
    ),
    "LLM hallucinations": dict(
        confused_with="model creativity or paraphrasing",
        overgen_rule="ask for facts",
        only_case="the prompt is too short",
        effect="the model sounds confident",
        cause="the answer becomes true",
        ritual="add 'be accurate' to the prompt"
    ),
    "Metrics under class imbalance": dict(
        confused_with="overfitting or bad training",
        overgen_rule="get high accuracy",
        only_case="the dataset is small",
        effect="accuracy is high",
        cause="the model must be good",
        ritual="increase the dataset size and keep using accuracy"
    ),
    "Calibration": dict(
        confused_with="high accuracy",
        overgen_rule="use softmax probabilities",
        only_case="the model is a neural network",
        effect="probabilities look smooth",
        cause="they must be correct",
        ritual="pick the max probability class and ignore thresholds"
    ),
    "Generalization vs. overfitting": dict(
        confused_with="training convergence",
        overgen_rule="train longer",
        only_case="the learning rate is too high",
        effect="training loss decreases",
        cause="the model generalizes",
        ritual="just stop when training accuracy is high"
    ),
    "Train/validation/test splits": dict(
        confused_with="data augmentation",
        overgen_rule="look at the test set during development",
        only_case="you don't have a validation set",
        effect="the test score improves",
        cause="the model is better",
        ritual="tune hyperparameters on the test set to maximize score"
    ),
    "Retrieval & verification": dict(
        confused_with="adding more parameters to the model",
        overgen_rule="trust the model output",
        only_case="you forgot to set temperature to zero",
        effect="the answer sounds fluent",
        cause="it must be correct",
        ritual="tell the model 'don't hallucinate'"
    )
}

def wrong_answers_for(question, concept, n=12):
    params = dict(DEFAULT_PARAMS)
    params.update(PARAMS_BY_CONCEPT.get(concept, {}))

    out = []
    for _ in range(n):
        mtype, patterns = random.choice(MISCONCEPTION_TEMPLATES)
        pattern = random.choice(patterns)
        txt = pattern.format(**params)

        if concept == "Data leakage":
            answer = f"Data leakage is when {txt}. For example, {random.choice(['using dropout', 'having class imbalance', 'training longer', 'getting high accuracy'])}."
        else:
            answer = txt

        out.append({"misconception_type": mtype, "answer": answer})
    return out

wrong_by_question = {q["question"]: wrong_answers_for(q["question"], q["concept"], n=12) for q in questions}

sample_q = "Explain data leakage and give two examples: one obvious and one subtle."
print("STEP 4 — Generate wrong answers (result for ONE question)\n")
print("Question:", sample_q, "\n")
print(pretty(wrong_by_question[sample_q][:6]))
print("\n" + "="*100 + "\n")


# -----------------------------
# STEP 5 — Misconception mining (simulated clustering → labels)
# -----------------------------
def mine_misconceptions(question, wrong_answers):
    buckets = defaultdict(list)
    for wa in wrong_answers:
        buckets[wa["misconception_type"]].append(wa["answer"])

    labels = []
    for key, answers in buckets.items():
        if "leakage" in question.lower():
            if key == "Confusing related concepts":
                label = "Confuses evaluation leakage with overfitting/dataset size issues"
                expl = "Leakage is about information crossing the train/test boundary, not simply poor generalization."
            elif key == "Ignoring necessary conditions or factors":
                label = "Believes leakage only occurs when labels (or proxies) are used as features"
                expl = "Leakage also happens via preprocessing, target encoding, time leakage, and model selection decisions."
            elif key == "Using memorized rules incorrectly":
                label = "Assumes a single ritual (shuffle/split) prevents leakage in all settings"
                expl = "Random split helps but doesn't prevent leakage from pipeline steps or time-dependent structure."
            elif key == "Overgeneralizing a valid principle":
                label = "Calls any data reuse 'leakage' even when it is valid evaluation practice"
                expl = "Cross-validation can be valid; leakage is specifically unintended evaluation contamination."
            else:
                label = "Treats leakage vaguely rather than as an evaluation-boundary violation"
                expl = "Leakage is defined by forbidden information influencing training or selection."
        else:
            label = f"{key} misconception"
            expl = "This bucket groups similar plausible learner errors."

        labels.append({
            "misconception_type": key,
            "misconception_label": label,
            "explanation": expl,
            "examples": answers[:2]
        })
    return labels

misconceptions = mine_misconceptions(sample_q, wrong_by_question[sample_q])
print("STEP 5 — Mine misconceptions (result)\n")
print(pretty(misconceptions))
print("\n" + "="*100 + "\n")


# -----------------------------
# STEP 6 — Build MCQ using misconceptions as distractors
# -----------------------------
def build_mcq(question, correct_answer, misconceptions, num_options=4):
    correct_opt = " ".join(correct_answer.split()[:34]).rstrip(",") + "…"
    picked = misconceptions[: num_options-1]

    distractors = []
    for m in picked:
        lab = m["misconception_label"].lower()
        if "overfitting" in lab:
            txt = "Data leakage is basically overfitting or having too little data, which makes test results unstable."
        elif "labels" in lab:
            txt = "Data leakage only happens if labels (or direct proxies) are included as features; otherwise it can’t occur."
        elif "ritual" in lab or "shuffle" in lab:
            txt = "If you shuffle and randomly split once, leakage is impossible, so you’re safe."
        else:
            txt = "Data leakage is any data reuse that makes evaluation optimistic, including cross-validation."
        distractors.append({"text": txt, "is_correct": False, "misconception": m["misconception_label"]})

    options = [{"text": correct_opt, "is_correct": True, "misconception": None}] + distractors
    random.shuffle(options)
    return {"question": question, "options": options}

mcq = build_mcq(sample_q, gold_map[sample_q], misconceptions, num_options=4)
print("STEP 6 — MCQ (result)\n")
print(pretty(mcq))
print("\n" + "="*100 + "\n")


# -----------------------------
# STEP 7 — Feedback per distractor (diagnostic)
# -----------------------------
def feedback_for(misconception_label):
    base = "This choice reflects a common misconception. "
    lab = misconception_label.lower()
    if "overfitting" in lab:
        return base + "You’re describing overfitting/variance. Leakage is when evaluation-set information influences training or selection (e.g., preprocessing on all data, time leakage)."
    if "labels" in lab:
        return base + "Label-in-feature is one case, but leakage also happens via preprocessing, feature selection, target encoding, or time leakage. Fit the whole pipeline only on training."
    if "ritual" in lab or "shuffle" in lab:
        return base + "Random split helps but doesn’t guarantee safety. Pipeline steps can leak and time-series needs time-aware splits. Treat leakage as information flow across the boundary."
    if "reuse" in lab:
        return base + "Cross-validation can be valid; leakage is unintended contamination from evaluation into training/model selection."
    return base + "Compare with the correct definition: leakage is evaluation contamination via forbidden information flow."

feedbacks = []
for opt in mcq["options"]:
    if not opt["is_correct"]:
        feedbacks.append({
            "option_text": opt["text"],
            "misconception": opt["misconception"],
            "feedback": feedback_for(opt["misconception"])
        })

print("STEP 7 — Feedback for incorrect options (result)\n")
print(pretty(feedbacks))
print("\n" + "="*100 + "\n")


# -----------------------------
# FINAL — "Published" payload you would serve via /public/{slug}
# -----------------------------
published = {
    "assessment": {
        "name": "AI Driving License — Mini Exam (Demo)",
        "concepts": [c["concept_name"] for c in concepts]
    },
    "item": {
        "concept": "Data leakage",
        "mcq": mcq,
        "feedback": feedbacks
    }
}

print("FINAL — Example published payload\n")
print(pretty(published))
