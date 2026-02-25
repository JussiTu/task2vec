"""
task2vec Ticket Advisor API
POST /api/analyze  — embed ticket text, find similar, call LLM for advice
POST /api/score    — score a ticket using outcome signals (no LLM)
GET  /api/health   — readiness check
"""
import json
import os
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
CACHE = BASE / ".cache"

SEARCH_INDEX = CACHE / "search_index.npy"
SEARCH_KEYS  = CACHE / "search_keys.npy"
SEARCH_META  = CACHE / "search_meta.json"
APP_DATA     = CACHE / "umap_app_data.json"

TOP_SIMILAR    = 20   # neighbours for expert detection
TOP_DISPLAY    = 5    # similar tickets shown to user
TOP_LLM        = 10   # tickets given to LLM for context
TOP_SCORE      = 20   # neighbours considered for scoring
EMBED_MODEL    = "text-embedding-3-large"
ADVICE_MODEL   = "gpt-4o-mini"

OUTCOME_SIGNALS = CACHE / "outcome_signals.json"

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

client = OpenAI()   # reads OPENAI_API_KEY from env

# ── Global data (loaded once at startup) ────────────────────────────────────
print("Loading search index …", flush=True)
t0 = time.time()
_vecs = np.load(str(SEARCH_INDEX))          # (N, 3072) float32, L2-normalised
_keys = np.load(str(SEARCH_KEYS))           # (N,) str
with open(SEARCH_META, encoding="utf-8") as f:
    _meta = json.load(f)                    # list[{key,summary,assignee,year,x,y,cluster}]
with open(APP_DATA, encoding="utf-8") as f:
    _app  = json.load(f)
print(f"  Loaded {len(_meta):,} tickets in {time.time()-t0:.1f}s", flush=True)

# Build key → meta-index lookup for fast access
_key_to_idx: dict = {_meta[i].get("key", ""): i for i in range(len(_meta))}

# Load outcome signals (optional — may not exist yet)
_outcome_signals: dict = {}
_outcome_calibration: dict = {}
if OUTCOME_SIGNALS.exists():
    print("Loading outcome signals …", flush=True)
    with open(OUTCOME_SIGNALS, encoding="utf-8") as f:
        _outcome_data = json.load(f)
    _outcome_signals    = _outcome_data.get("signals", {})
    _outcome_calibration = _outcome_data.get("calibration", {})
    print(f"  Loaded {len(_outcome_signals):,} outcome signals.", flush=True)
else:
    print("  outcome_signals.json not found — /api/score will return empty results.", flush=True)

_strategy = _app.get("strategy", {})
_cluster_labels = {
    str(k): v["label"]
    for k, v in _app.get("cluster_labels", {}).items()
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _embed(text: str) -> np.ndarray:
    """Return L2-normalised embedding vector."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec  = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _top_similar(vec: np.ndarray, n: int = TOP_SIMILAR):
    """Return indices of n most-similar tickets (dot product on normed vecs)."""
    sims = _vecs @ vec                        # (N,)
    idx  = np.argpartition(sims, -n)[-n:]     # unordered top-n
    idx  = idx[np.argsort(sims[idx])[::-1]]   # sorted best-first
    return idx, sims[idx]


def _estimate_umap(top5_idx) -> dict:
    xs = [_meta[i]["x"] for i in top5_idx if _meta[i].get("x") is not None]
    ys = [_meta[i]["y"] for i in top5_idx if _meta[i].get("y") is not None]
    if not xs:
        return {}
    return {"x": float(np.mean(xs)), "y": float(np.mean(ys))}


def _majority_cluster(top5_idx) -> str:
    clusters = [str(_meta[i].get("cluster", "")) for i in top5_idx
                if _meta[i].get("cluster") is not None]
    if not clusters:
        return ""
    most_common = Counter(clusters).most_common(1)[0][0]
    return _cluster_labels.get(most_common, most_common)


def _strategy_context() -> str:
    early  = [t.get("label","") for t in _strategy.get("early_top",  [])[:2]]
    recent = [t.get("label","") for t in _strategy.get("recent_top", [])[:2]]
    early_str  = " and ".join(early)  or "early focus areas"
    recent_str = " and ".join(recent) or "recent focus areas"
    narr = _strategy.get("narrative", "")
    return (
        f"The project shifted from {early_str} toward {recent_str}. "
        + (narr if narr else "")
    )


def _llm_advice(ticket_text: str, similar_tickets: list, strat_ctx: str) -> str:
    ticket_block = "\n".join(
        f'- [{t["key"]}] {t["summary"]}' for t in similar_tickets
    )
    prompt = (
        "You are an engineering advisor helping a developer understand how a new ticket "
        "fits into the broader project landscape.\n\n"
        f"PROJECT DIRECTION:\n{strat_ctx}\n\n"
        f"NEW TICKET:\n{ticket_text}\n\n"
        f"MOST SIMILAR HISTORICAL TICKETS:\n{ticket_block}\n\n"
        "Please provide a concise (3–5 sentence) analysis:\n"
        "1. What semantic area of the project does this ticket belong to?\n"
        "2. Is it aligned with the current project direction, or does it address an older concern?\n"
        "3. Any patterns in the similar tickets that are worth noting?\n"
        "Keep the tone technical but clear."
    )
    resp = client.chat.completions.create(
        model=ADVICE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "indexed": len(_meta)})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "No ticket text provided"}), 400
    ticket_text = (data.get("text") or "").strip()
    if not ticket_text:
        return jsonify({"error": "No ticket text provided"}), 400
    if len(ticket_text) > 8000:
        ticket_text = ticket_text[:8000]

    try:
        vec = _embed(ticket_text)
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {e}"}), 500

    # Similarity search
    top_idx, top_sims = _top_similar(vec, TOP_SIMILAR)

    # UMAP position estimate from nearest 5
    umap_pos = _estimate_umap(top_idx[:5])

    # Cluster via majority vote of nearest 5
    cluster_label = _majority_cluster(top_idx[:5])

    # Similar tickets for display (top 5)
    similar_display = []
    for i, sim in zip(top_idx[:TOP_DISPLAY], top_sims[:TOP_DISPLAY]):
        m = _meta[i]
        similar_display.append({
            "key":      m.get("key", ""),
            "summary":  m.get("summary", ""),
            "assignee": m.get("assignee", ""),
            "year":     m.get("year"),
            "cluster":  _cluster_labels.get(str(m.get("cluster","")), ""),
            "similarity": round(float(sim), 3),
        })

    # Experts: assignees most frequent in top-20
    assignees = [_meta[i].get("assignee","") for i in top_idx if _meta[i].get("assignee")]
    experts = [{"name": name, "count": cnt}
               for name, cnt in Counter(assignees).most_common(3)
               if name]

    # LLM advice using top-10 for context
    similar_for_llm = []
    for i in top_idx[:TOP_LLM]:
        m = _meta[i]
        similar_for_llm.append({"key": m.get("key",""), "summary": m.get("summary","")})
    strat_ctx = _strategy_context()
    try:
        advice = _llm_advice(ticket_text, similar_for_llm, strat_ctx)
    except Exception as e:
        advice = f"(LLM unavailable: {e})"

    return jsonify({
        "umap_pos":        umap_pos,
        "cluster":         cluster_label,
        "top_similarity":  round(float(top_sims[0]), 3),
        "similar":         similar_display,
        "experts":         experts,
        "advice":          advice,
    })


@app.route("/api/score", methods=["POST"])
def score():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "No ticket text provided"}), 400
    ticket_text = (data.get("text") or "").strip()
    if not ticket_text:
        return jsonify({"error": "No ticket text provided"}), 400
    if len(ticket_text) > 8000:
        ticket_text = ticket_text[:8000]

    try:
        vec = _embed(ticket_text)
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {e}"}), 500

    # Find top-20 similar tickets
    top_idx, top_sims = _top_similar(vec, TOP_SCORE)

    # For each of top-10, look up outcome signals by key
    TIERS = ("Automate", "Assist", "Escalate")
    weighted: dict[str, float] = {t: 0.0 for t in TIERS}
    total_weight = 0.0
    evidence = []

    for i, sim in zip(top_idx[:10], top_sims[:10]):
        m   = _meta[i]
        key = m.get("key", "")
        sig = _outcome_signals.get(key)
        if sig is None:
            continue
        w = max(0.0, float(sim))
        weighted[sig["label"]] += w
        total_weight            += w
        evidence.append({
            "key":        key,
            "summary":    m.get("summary", ""),
            "similarity": round(float(sim), 3),
            "days":       sig["days"],
            "watches":    sig["watches"],
            "label":      sig["label"],
        })

    coverage = len(evidence)

    if total_weight == 0:
        # No outcome data found — fall back to uniform distribution
        probs = {t: round(1/3, 3) for t in TIERS}
        tier  = "Assist"
        confidence = 0.0
    else:
        probs = {t: round(weighted[t] / total_weight, 3) for t in TIERS}
        # Normalise to sum exactly 1
        s = sum(probs.values())
        if s > 0:
            probs = {t: round(probs[t] / s, 3) for t in probs}
        tier       = max(probs, key=probs.__getitem__)
        confidence = round(probs[tier], 3)

    # Average resolution days (evidence only)
    avg_days = None
    if evidence:
        avg_days = round(sum(e["days"] for e in evidence) / len(evidence), 1)

    return jsonify({
        "tier":          tier,
        "confidence":    confidence,
        "probabilities": probs,
        "evidence":      evidence,
        "avg_days":      avg_days,
        "coverage":      coverage,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
