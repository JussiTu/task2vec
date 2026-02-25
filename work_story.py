# work_story.py
#
# Produce a verbal "work history" narrative for a Jira assignee:
# - Embed tickets from JSONL (cached — no repeat API calls)
# - Cluster with KMeans (for themes)
# - Auto-label clusters using an LLM (short theme names)
# - Build an assignee timeline (early/mid/recent) of themes
# - Detect shifts
# - Compute alignment with overall direction
# - Output:
#     - assignee_work_story.md
#     - assignee_work_story.json
#     - cluster_labels.json
#
# Run:
#   python work_story.py --input jira_1000.jsonl --limit 1000 --kmeans_k 24 --assignee_rank 1

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dateutil import parser as dateparser
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

from ticketing_intel.config import cfg
from ticketing_intel.etl.pipeline import run_pipeline


# -----------------------------
# OpenAI LLM helpers (kept as-is)
# -----------------------------
def llm_cluster_label(samples: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
    client = OpenAI()
    lines = []
    for s in samples[:8]:
        k = s.get("key", "").strip()
        t = (s.get("summary") or s.get("text") or "").strip().replace("\n", " ")[:180]
        lines.append(f"- {k}: {t}")
    payload = "\n".join(lines)
    msg = (
        "You are labeling a cluster of Jira tickets. "
        "Return ONE concise theme label (3–6 words). "
        "No punctuation at the end. No quotes.\n\n"
        f"Tickets:\n{payload}\n\nLabel:"
    )
    resp = OpenAI().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": msg}],
        temperature=0.1,
    )
    label = (resp.choices[0].message.content or "").strip()
    return " ".join(label.split()[:8])


def llm_write_story(assignee_id: str, stats: Dict[str, Any], model: str = "gpt-4o-mini") -> str:
    compact = json.dumps(stats, ensure_ascii=False)
    msg = (
        "Write a concise work-history narrative for one anonymized assignee based on the provided structured data.\n"
        "Requirements:\n"
        "- 6–10 sentences max.\n"
        "- Use plain English.\n"
        "- Mention the timeline (early → middle → recent).\n"
        "- Mention 2–4 theme labels and include 1–2 example ticket keys per theme.\n"
        "- Include the alignment statement as: 'X% of steps aligned with the overall direction.'\n"
        "- Do NOT mention embeddings, PCA, cosine, vectors.\n"
        "- Do NOT invent facts outside the provided data.\n\n"
        f"Assignee: {assignee_id}\n"
        f"Data (JSON): {compact}\n\n"
        "Narrative:"
    )
    resp = OpenAI().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": msg}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


# -----------------------------
# Clustering / themes
# -----------------------------
def kmeans_labels(X: np.ndarray, k: int):
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = km.fit_predict(X)  # X already L2-normalised from pipeline
    return labels, km.cluster_centers_


def representative_indices_to_centroid(X: np.ndarray, idxs: List[int], centroid: np.ndarray, top_n: int = 6) -> List[int]:
    if not idxs:
        return []
    Xn = normalize(X[idxs], norm="l2")
    cn = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = cosine_similarity(Xn, cn.reshape(1, -1)).reshape(-1)
    return [idxs[i] for i in np.argsort(-sims)[:top_n]]


def top_keywords_for_indices(texts: List[str], idxs: List[int], top_n: int = 8) -> List[str]:
    if not idxs:
        return []
    subset = [texts[i] for i in idxs]
    v = TfidfVectorizer(max_features=4000, stop_words="english", ngram_range=(1, 2), min_df=2)
    try:
        M = v.fit_transform(subset)
    except ValueError:
        return []
    mean_w = np.asarray(M.mean(axis=0)).reshape(-1)
    terms = np.array(v.get_feature_names_out())
    return [t for t in terms[np.argsort(-mean_w)[:top_n]].tolist() if t.strip()]


# -----------------------------
# Trajectory + alignment
# -----------------------------
def step_alignment_positive_rate(X_path: np.ndarray) -> Tuple[float, List[float]]:
    if X_path.shape[0] < 2:
        return float("nan"), []
    d = X_path[-1] - X_path[0]
    dn = d / (np.linalg.norm(d) + 1e-12)
    steps = X_path[1:] - X_path[:-1]
    stepsn = steps / (np.linalg.norm(steps, axis=1, keepdims=True) + 1e-12)
    cos = (stepsn @ dn).reshape(-1)
    return float(np.mean(cos > 0)), cos.tolist()


def split_into_terciles(sorted_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    n = len(sorted_items)
    if n == 0:
        return {"early": [], "middle": [], "recent": []}
    a, b = n // 3, 2 * n // 3
    return {"early": sorted_items[:a], "middle": sorted_items[a:b], "recent": sorted_items[b:]}


def theme_distribution(items: List[Dict[str, Any]]) -> Counter:
    return Counter(it["cluster"] for it in items)


def top_themes(counter: Counter, k: int = 3) -> List[Tuple[int, int]]:
    return counter.most_common(k)


def detect_shifts(dists: Dict[str, Counter], topk: int = 2) -> List[Dict[str, Any]]:
    shifts = []
    prev = None
    for ph in ["early", "middle", "recent"]:
        top = [cid for cid, _ in dists[ph].most_common(topk)]
        if prev is not None and top and top != prev:
            shifts.append({"from_top": prev, "to_top": top, "phase": ph})
        prev = top
    return shifts


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="JSONL path (overrides JIRA_DUMP_PATH in .env)")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--chat_model", default="gpt-4o-mini")
    ap.add_argument("--include_comments", action="store_true")
    ap.add_argument("--kmeans_k", type=int, default=24)
    ap.add_argument("--min_tickets", type=int, default=8)
    ap.add_argument("--assignee_id", default="", help="Optional exact assignee ID")
    ap.add_argument("--assignee_rank", type=int, default=1, help="If no assignee_id, pick Nth by ticket count")
    ap.add_argument("--out_md", default="assignee_work_story.md")
    ap.add_argument("--out_json", default="assignee_work_story.json")
    ap.add_argument("--out_cluster_labels", default="cluster_labels.json")
    args = ap.parse_args()

    if args.input:
        cfg.jira_dump_path = args.input

    keys, X, store = run_pipeline(cfg, limit=args.limit, include_comments=args.include_comments)
    all_meta = {t["key"]: t for t in store.all_tickets()}
    store.close()

    # Reconstruct per-ticket lists aligned with keys/X
    summaries = [(all_meta.get(k, {}).get("summary") or "").replace("\n", " ").strip() for k in keys]
    texts_for_tfidf = [
        f"{all_meta.get(k, {}).get('summary', '')} {all_meta.get(k, {}).get('description', '')}"
        for k in keys
    ]
    assignee = [all_meta.get(k, {}).get("assignee_id", "") for k in keys]
    created_str = [all_meta.get(k, {}).get("created", "") for k in keys]

    def parse_ts(s: str) -> Optional[float]:
        try:
            return dateparser.parse(s).timestamp() if s else None
        except Exception:
            return None

    created_ts = [parse_ts(s) for s in created_str]
    created_iso = created_str  # already ISO strings from Jira

    # Choose target assignee
    counts = Counter(a for a in assignee if a)
    if not counts:
        raise SystemExit("No assignee IDs found.")

    if args.assignee_id:
        target = args.assignee_id
        if target not in counts:
            raise SystemExit(f"assignee_id not found: {target}")
    else:
        ranked = [a for a, n in counts.most_common() if n >= args.min_tickets]
        if not ranked:
            raise SystemExit(f"No assignees with >= {args.min_tickets} tickets.")
        target = ranked[max(1, min(args.assignee_rank, len(ranked))) - 1]

    # Cluster all tickets (themes are globally comparable)
    cluster, centroids = kmeans_labels(X, k=args.kmeans_k)

    # Prepare cluster labels (LLM)
    cluster_to_indices = defaultdict(list)
    for i, c in enumerate(cluster):
        cluster_to_indices[int(c)].append(i)

    cluster_labels: Dict[str, Any] = {}
    for c in range(args.kmeans_k):
        idxs = cluster_to_indices.get(c, [])
        rep = representative_indices_to_centroid(X, idxs, centroids[c], top_n=6)
        samples = [{"key": keys[i], "summary": summaries[i] or texts_for_tfidf[i][:180]} for i in rep]
        label = llm_cluster_label(samples, model=args.chat_model) if samples else f"Cluster {c}"
        kw = top_keywords_for_indices(texts_for_tfidf, idxs, top_n=10)
        cluster_labels[str(c)] = {
            "label": label,
            "keywords": kw,
            "size": len(idxs),
            "examples": [{"key": keys[i], "summary": summaries[i]} for i in rep[:3]],
        }

    with open(args.out_cluster_labels, "w", encoding="utf-8") as f:
        json.dump(cluster_labels, f, ensure_ascii=False, indent=2)

    # Build target assignee trajectory
    t_idxs = [i for i, a in enumerate(assignee) if a == target and created_ts[i] is not None]
    if len(t_idxs) < args.min_tickets:
        raise SystemExit(f"Target assignee has only {len(t_idxs)} tickets with timestamps; need >= {args.min_tickets}.")

    t_idxs.sort(key=lambda i: created_ts[i])
    items = [{
        "idx": i,
        "key": keys[i],
        "created_iso": created_iso[i],
        "created_ts": created_ts[i],
        "cluster": int(cluster[i]),
        "cluster_label": cluster_labels[str(int(cluster[i]))]["label"],
    } for i in t_idxs]

    bins = split_into_terciles(items)
    dists = {ph: theme_distribution(bins[ph]) for ph in ["early", "middle", "recent"]}
    shifts = detect_shifts(dists, topk=2)
    overall_dist = theme_distribution(items)

    X_path = X[t_idxs]
    pos_rate, cos_list = step_alignment_positive_rate(X_path)
    pos_pct = int(round(100.0 * pos_rate)) if pos_rate == pos_rate else None

    def phase_themes(phase: str, topn: int = 3) -> List[Dict[str, Any]]:
        total = max(1, sum(dists[phase].values()))
        return [
            {
                "cluster": int(cid),
                "label": cluster_labels[str(cid)]["label"],
                "share": cnt / total,
                "count": cnt,
                "example_keys": [it["key"] for it in bins[phase] if it["cluster"] == cid][:3],
            }
            for cid, cnt in dists[phase].most_common(topn)
        ]

    overall_total = max(1, sum(overall_dist.values()))
    overall_themes = [
        {
            "cluster": int(cid),
            "label": cluster_labels[str(cid)]["label"],
            "share": cnt / overall_total,
            "count": cnt,
            "example_keys": [it["key"] for it in items if it["cluster"] == cid][:3],
        }
        for cid, cnt in overall_dist.most_common(5)
    ]

    labeled_shifts = [
        {
            "phase": s["phase"],
            "from": [{"cluster": cid, "label": cluster_labels[str(cid)]["label"]} for cid in s["from_top"]],
            "to": [{"cluster": cid, "label": cluster_labels[str(cid)]["label"]} for cid in s["to_top"]],
        }
        for s in shifts
    ]

    story_data = {
        "assignee_id": target,
        "tickets": {"count": len(items), "first_created": items[0]["created_iso"], "last_created": items[-1]["created_iso"]},
        "themes": {"overall_top": overall_themes, "by_phase": {ph: phase_themes(ph) for ph in ["early", "middle", "recent"]}},
        "shifts": labeled_shifts,
        "alignment": {"positive_rate": pos_rate, "positive_percent": pos_pct, "steps": len(items) - 1},
        "evidence": {
            "sample_tickets_early": [{"key": it["key"], "label": it["cluster_label"]} for it in bins["early"][:5]],
            "sample_tickets_recent": [{"key": it["key"], "label": it["cluster_label"]} for it in bins["recent"][-5:]],
        },
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(story_data, f, ensure_ascii=False, indent=2)

    narrative = llm_write_story(target, story_data, model=args.chat_model)

    facts = [
        f"# Work history summary (assignee {target})", "",
        f"- Tickets analyzed: **{len(items)}**",
        f"- Time range: **{items[0]['created_iso']} → {items[-1]['created_iso']}**",
    ]
    if pos_pct is not None:
        facts.append(f"- Alignment: **{pos_pct}% of steps aligned with the overall direction**")
    facts += ["", "## Narrative", narrative, "", "## Themes (overall top 5)"]
    for t in overall_themes:
        share = int(round(100 * t["share"]))
        facts.append(f"- **{t['label']}** — {t['count']} tickets (~{share}%) — examples: {', '.join(t['example_keys'])}")
    facts += ["", "## Themes by phase"]
    for ph in ["early", "middle", "recent"]:
        facts.append(f"### {ph.capitalize()}")
        for t in phase_themes(ph):
            share = int(round(100 * t["share"]))
            facts.append(f"- **{t['label']}** — {t['count']} tickets (~{share}%) — examples: {', '.join(t['example_keys'])}")
        facts.append("")
    facts.append("## Detected shifts")
    if labeled_shifts:
        for sh in labeled_shifts:
            frm = "; ".join(x["label"] for x in sh["from"])
            to = "; ".join(x["label"] for x in sh["to"])
            facts.append(f"- In **{sh['phase']}**, top themes changed from **{frm}** → **{to}**")
    else:
        facts.append("- No strong theme shift detected.")
    facts.append("")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(facts))

    print(f"Target assignee: {target} (tickets={len(items)})")
    print(f"Wrote: {args.out_md}")
    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_cluster_labels}")
    print("Done.")


if __name__ == "__main__":
    main()
