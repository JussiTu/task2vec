"""
mcq_generator.py

Rule-based MCQ generator for broad-audience (office worker) security scenarios.

Goal:
- Create 1 "gold" option + N distractors using generic "root causes of error"
- Keep all options same structure and ~same length (word count window)
- Avoid giveaway words like "only/always/never" unless you want them

How it works (simple + controllable):
1) You write:
   - stem (scenario)
   - an "acceptance invariant" (the real rule)
   - a short gold heuristic (correct condition phrased like a peer heuristic)
2) You select root causes to use as distractors.
3) The generator:
   - expands each into an option sentence with consistent phrasing
   - generates a few variants per option
   - picks best variants that match target length and avoid bad cues
"""

import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

random.seed(7)

# ---------------------------
# Text utilities
# ---------------------------

def polish(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def word_count(s: str) -> int:
    return len(s.split())

def has_giveaway_words(s: str) -> bool:
    # You can tune this list for your exam style
    giveaways = ["always", "never", "guarantee", "impossible", "only if", "must"]
    low = s.lower()
    return any(g in low for g in giveaways)

def normalize_punctuation(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+\.", ".", s)
    s = re.sub(r"\s+,", ",", s)
    if not s.endswith("."):
        s += "."
    return s

def sentenceize(prefix: str, condition: str) -> str:
    # Keep same form across options (very important!)
    # Example: "It is acceptable to comply if <condition>."
    return normalize_punctuation(polish(f"{prefix} {condition}"))

# ---------------------------
# Root cause model
# ---------------------------

@dataclass
class RootCause:
    id: str
    name: str
    condition_templates: List[str]
    # Optional: short diagnostic text you can attach later
    feedback: str = ""

def fill_template(t: str, ctx: Dict[str, str]) -> str:
    # Simple {placeholder} fill
    out = t
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", v)
    return out

def generate_option_variants(
    condition_templates: List[str],
    ctx: Dict[str, str],
    n_variants: int = 8
) -> List[str]:
    variants = []
    for _ in range(n_variants):
        t = random.choice(condition_templates)
        variants.append(polish(fill_template(t, ctx)))
    return variants

def pick_best_variant(
    variants: List[str],
    prefix: str,
    target_words: int,
    tol: int,
    avoid_giveaways: bool = True
) -> str:
    """
    Pick a variant whose final option sentence length is close to target_words.
    """
    scored: List[Tuple[float, str]] = []
    for v in variants:
        sent = sentenceize(prefix, v)
        wc = word_count(sent)
        length_penalty = abs(wc - target_words)

        # Strong penalty if too far
        if abs(wc - target_words) > tol:
            length_penalty += 6.0

        giveaway_penalty = 0.0
        if avoid_giveaways and has_giveaway_words(sent):
            giveaway_penalty += 4.0

        # mild penalty if it's vague “seems”
        vague_penalty = 0.7 if "seems" in sent.lower() else 0.0

        score = length_penalty + giveaway_penalty + vague_penalty
        scored.append((score, sent))

    scored.sort(key=lambda x: x[0])
    return scored[0][1]

# ---------------------------
# MCQ generator
# ---------------------------

def build_mcq(
    stem: str,
    question: str,
    gold_condition: str,
    distractor_root_causes: List[RootCause],
    ctx: Dict[str, str],
    prefix: str = "It is acceptable to comply with the request if",
    n_variants: int = 10,
    target_words: Optional[int] = None,
    tol: int = 2,
    shuffle: bool = False
) -> Dict:
    """
    Returns a dict with question + options (A..).
    gold_condition should already be phrased at the same level as distractors.
    """
    # 1) Build raw gold sentence
    gold_sentence = sentenceize(prefix, gold_condition)

    # Choose target length from gold unless user sets it
    if target_words is None:
        target_words = word_count(gold_sentence)

    # 2) Build distractor sentences by RC templates -> variants -> best match
    distractors: List[Tuple[str, str]] = []  # (rc_id, sentence)
    for rc in distractor_root_causes:
        variants = generate_option_variants(rc.condition_templates, ctx, n_variants=n_variants)
        best = pick_best_variant(
            variants,
            prefix=prefix,
            target_words=target_words,
            tol=tol,
            avoid_giveaways=True
        )
        distractors.append((rc.id, best))

    # 3) Option assembly
    # Keep gold as A by default (easier for debugging); can shuffle later.
    options = [("A", "GOLD", gold_sentence)]
    letters = ["B", "C", "D", "E", "F", "G"]
    for i, (rc_id, sent) in enumerate(distractors):
        options.append((letters[i], rc_id, sent))

    if shuffle:
        # If you want real exam behavior, shuffle while keeping answer_key correct
        # We'll implement a safe shuffle that tracks answer key.
        pool = options[:]
        random.shuffle(pool)
        # re-letter
        relabeled = []
        for i, (_, tag, sent) in enumerate(pool):
            relabeled.append((chr(ord("A") + i), tag, sent))
        options = relabeled

    answer_key = next(letter for (letter, tag, _) in options if tag == "GOLD")

    # 4) Equalize length (optional “micro padding”)
    # If you want strict equal word counts, you can pad short options with neutral phrases.
    # I keep it OFF by default because padding can add artifacts.
    # If you want strict equalization, tell me and I’ll enable a safe version.

    return {
        "stem": stem,
        "question": question,
        "options": [{"id": letter, "text": sent} for (letter, _, sent) in options],
        "answer_key": answer_key,
        "meta": {
            "target_words": target_words,
            "tolerance": tol,
            "root_causes_used": [rc.id for rc in distractor_root_causes],
        }
    }

# ---------------------------
# Example: your exact question
# ---------------------------

if __name__ == "__main__":
    STEM = (
        "You receive an unexpected call from someone claiming to be company IT support "
        "and asking for remote access."
    )
    QUESTION = "Under which condition is it acceptable to comply with the request?"

    # Gold condition: keep it short and “peer-heuristic style”
    GOLD_CONDITION = (
        "you personally confirm it using a company contact you already trust"
    )

    # Context placeholders if you want to reuse templates across scenarios
    CTX = {
        "tool": "an official remote support tool provided by the company",
        "no_secrets": "the request does not require sharing passwords or authentication codes",
        "low_risk": "the access is limited, temporary, and related to a routine technical issue",
    }

    # Define root causes with multiple templates each (gives variety without an LLM)
    RC_TOOL_FIXATION = RootCause(
        id="RC2",
        name="Surface feature fixation (tool looks official)",
        condition_templates=[
            "the request uses {tool}",
            "the caller asks you to use {tool}",
            "the remote support session is done using {tool}",
        ],
        feedback="Official tools can be abused; tool branding is not verification.",
    )

    RC_PARTIAL_RULE = RootCause(
        id="RC8",
        name="Risk minimization framing (no password means safe)",
        condition_templates=[
            "{no_secrets}",
            "you are not asked to share passwords or authentication codes",
            "the caller does not ask for any passwords or authentication codes",
        ],
        feedback="Not sharing a password is good, but remote access or resets can still compromise you.",
    )

    RC_LOW_RISK = RootCause(
        id="RC3",
        name="Partial condition acceptance (limited access feels safe)",
        condition_templates=[
            "{low_risk}",
            "the access is limited and temporary for a routine technical issue",
            "the action is limited to temporary access to resolve a routine issue",
        ],
        feedback="‘Limited’ access can still be enough for compromise; verification is required.",
    )

    mcq = build_mcq(
        stem=STEM,
        question=QUESTION,
        gold_condition=GOLD_CONDITION,
        distractor_root_causes=[RC_TOOL_FIXATION, RC_PARTIAL_RULE, RC_LOW_RISK],
        ctx=CTX,
        prefix="It is acceptable to comply with the request if",
        n_variants=12,
        tol=2,
        shuffle=False
    )

    print("\n=== MCQ (JSON) ===\n")
    import json
    print(json.dumps(mcq, indent=2, ensure_ascii=False))

    print("\n=== Pretty print ===\n")
    print(STEM)
    print(QUESTION)
    for opt in mcq["options"]:
        print(f'{opt["id"]}. {opt["text"]}')
    print(f"\nAnswer key: {mcq['answer_key']}")
