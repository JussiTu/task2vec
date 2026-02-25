"""
run_all_work_stories.py

Generates a work-history story for every qualifying assignee efficiently:
  - Pipeline runs once (embeddings from cache)
  - Clusters computed once
  - Cluster labels generated once (24 LLM calls)
  - One narrative LLM call per assignee

Outputs go to stories/ directory:
  stories/<short_id>.md
  stories/<short_id>.json
  stories/cluster_labels.json
  stories/all_stories_summary.md

Run (JSONL):
  python run_all_work_stories.py --input jira_1000.jsonl --limit 1000 --kmeans_k 24 --min_tickets 8

Run (MongoDB):
  python run_all_work_stories.py --collection Spring --min_tickets 30 --out_dir stories_spring
  python run_all_work_stories.py --collection Apache --projects KAFKA ZOOKEEPER --min_tickets 20
"""

import json
import argparse
import os
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
from ticketing_intel.etl.pipeline import run_pipeline, run_pipeline_from_tickets


# -----------------------------------------------------------------------
# Reused from work_story.py (shared logic)
# -----------------------------------------------------------------------

def llm_cluster_label(samples: List[Dict[str, str]], model: str) -> str:
    lines = [f"- {s.get('key','')}: {(s.get('summary') or '')[:180].replace(chr(10),' ')}" for s in samples[:8]]
    msg = (
        "You are labeling a cluster of Jira tickets. "
        "Return ONE concise theme label (3-6 words). "
        "No punctuation at the end. No quotes.\n\n"
        f"Tickets:\n{chr(10).join(lines)}\n\nLabel:"
    )
    resp = OpenAI().chat.completions.create(
        model=model, messages=[{"role": "user", "content": msg}], temperature=0.1
    )
    label = (resp.choices[0].message.content or "").strip()
    return " ".join(label.split()[:8])


def llm_write_story(assignee_id: str, stats: Dict[str, Any], model: str) -> str:
    msg = (
        "Write a concise work-history narrative for one anonymized assignee based on the provided structured data.\n"
        "Requirements:\n"
        "- 6-10 sentences max.\n"
        "- Use plain English.\n"
        "- Mention the timeline (early, middle, recent).\n"
        "- Mention 2-4 theme labels and include 1-2 example ticket keys per theme.\n"
        "- Include the alignment statement as: 'X% of steps aligned with the overall direction.'\n"
        "- Do NOT mention embeddings, PCA, cosine, vectors.\n"
        "- Do NOT invent facts outside the provided data.\n\n"
        f"Assignee: {assignee_id}\n"
        f"Data (JSON): {json.dumps(stats, ensure_ascii=False)}\n\n"
        "Narrative:"
    )
    resp = OpenAI().chat.completions.create(
        model=model, messages=[{"role": "user", "content": msg}], temperature=0.2
    )
    return (resp.choices[0].message.content or "").strip()


def representative_indices(X: np.ndarray, idxs: List[int], centroid: np.ndarray, top_n: int = 6) -> List[int]:
    if not idxs:
        return []
    Xn = normalize(X[idxs], norm="l2")
    cn = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = cosine_similarity(Xn, cn.reshape(1, -1)).reshape(-1)
    return [idxs[i] for i in np.argsort(-sims)[:top_n]]


def top_keywords(texts: List[str], idxs: List[int], top_n: int = 8) -> List[str]:
    if not idxs:
        return []
    v = TfidfVectorizer(max_features=4000, stop_words="english", ngram_range=(1, 2), min_df=2)
    try:
        M = v.fit_transform([texts[i] for i in idxs])
    except ValueError:
        return []
    terms = np.array(v.get_feature_names_out())
    return terms[np.argsort(-np.asarray(M.mean(axis=0)).reshape(-1))[:top_n]].tolist()


def step_alignment_positive_rate(X_path: np.ndarray) -> Tuple[float, List[float]]:
    if X_path.shape[0] < 2:
        return float("nan"), []
    d = X_path[-1] - X_path[0]
    dn = d / (np.linalg.norm(d) + 1e-12)
    steps = X_path[1:] - X_path[:-1]
    stepsn = steps / (np.linalg.norm(steps, axis=1, keepdims=True) + 1e-12)
    cos = (stepsn @ dn).reshape(-1)
    return float(np.mean(cos > 0)), cos.tolist()


def split_terciles(items: List) -> Dict:
    n = len(items)
    a, b = n // 3, 2 * n // 3
    return {"early": items[:a], "middle": items[a:b], "recent": items[b:]}


def theme_dist(items: List[Dict]) -> Counter:
    return Counter(it["cluster"] for it in items)


def detect_shifts(dists: Dict[str, Counter], topk: int = 2) -> List[Dict]:
    shifts, prev = [], None
    for ph in ["early", "middle", "recent"]:
        top = [cid for cid, _ in dists[ph].most_common(topk)]
        if prev is not None and top and top != prev:
            shifts.append({"from_top": prev, "to_top": top, "phase": ph})
        prev = top
    return shifts


def build_story_for_assignee(
    target: str,
    keys: List[str],
    X: np.ndarray,
    assignee_list: List[str],
    created_ts_list: List[Optional[float]],
    created_iso_list: List[str],
    cluster: np.ndarray,
    cluster_labels: Dict[str, Any],
    chat_model: str,
    min_tickets: int,
) -> Optional[Dict[str, Any]]:
    """Build story data and narrative for one assignee. Returns None if not enough tickets."""
    t_idxs = [
        i for i, a in enumerate(assignee_list)
        if a == target and created_ts_list[i] is not None
    ]
    if len(t_idxs) < min_tickets:
        return None

    t_idxs.sort(key=lambda i: created_ts_list[i])
    items = [{
        "idx": i,
        "key": keys[i],
        "created_iso": created_iso_list[i],
        "created_ts": created_ts_list[i],
        "cluster": int(cluster[i]),
        "cluster_label": cluster_labels[str(int(cluster[i]))]["label"],
    } for i in t_idxs]

    bins = split_terciles(items)
    dists = {ph: theme_dist(bins[ph]) for ph in ["early", "middle", "recent"]}
    shifts = detect_shifts(dists, topk=2)
    overall_dist = theme_dist(items)

    X_path = X[t_idxs]
    pos_rate, _ = step_alignment_positive_rate(X_path)
    pos_pct = int(round(100.0 * pos_rate)) if pos_rate == pos_rate else None

    def phase_themes(phase: str, topn: int = 3) -> List[Dict]:
        total = max(1, sum(dists[phase].values()))
        return [{
            "cluster": int(cid),
            "label": cluster_labels[str(cid)]["label"],
            "share": cnt / total,
            "count": cnt,
            "example_keys": [it["key"] for it in bins[phase] if it["cluster"] == cid][:3],
        } for cid, cnt in dists[phase].most_common(topn)]

    overall_total = max(1, sum(overall_dist.values()))
    overall_themes = [{
        "cluster": int(cid),
        "label": cluster_labels[str(cid)]["label"],
        "share": cnt / overall_total,
        "count": cnt,
        "example_keys": [it["key"] for it in items if it["cluster"] == cid][:3],
    } for cid, cnt in overall_dist.most_common(5)]

    labeled_shifts = [{
        "phase": s["phase"],
        "from": [{"cluster": cid, "label": cluster_labels[str(cid)]["label"]} for cid in s["from_top"]],
        "to":   [{"cluster": cid, "label": cluster_labels[str(cid)]["label"]} for cid in s["to_top"]],
    } for s in shifts]

    story_data = {
        "assignee_id": target,
        "tickets": {"count": len(items), "first_created": items[0]["created_iso"], "last_created": items[-1]["created_iso"]},
        "themes": {"overall_top": overall_themes, "by_phase": {ph: phase_themes(ph) for ph in ["early", "middle", "recent"]}},
        "shifts": labeled_shifts,
        "alignment": {"positive_rate": pos_rate, "positive_percent": pos_pct, "steps": len(items) - 1},
        "evidence": {
            "sample_tickets_early":  [{"key": it["key"], "label": it["cluster_label"]} for it in bins["early"][:5]],
            "sample_tickets_recent": [{"key": it["key"], "label": it["cluster_label"]} for it in bins["recent"][-5:]],
        },
    }

    narrative = llm_write_story(target, story_data, model=chat_model)
    story_data["narrative"] = narrative
    return story_data


def story_to_md(story: Dict[str, Any]) -> str:
    target = story["assignee_id"]
    items_count = story["tickets"]["count"]
    first = story["tickets"]["first_created"]
    last  = story["tickets"]["last_created"]
    pos_pct = story["alignment"]["positive_percent"]
    overall_themes = story["themes"]["overall_top"]
    by_phase = story["themes"]["by_phase"]
    shifts = story["shifts"]
    narrative = story.get("narrative", "")

    lines = [
        f"# Work history: {target}", "",
        f"- Tickets analyzed: **{items_count}**",
        f"- Time range: **{first} - {last}**",
    ]
    if pos_pct is not None:
        lines.append(f"- Alignment: **{pos_pct}% of steps aligned with the overall direction**")
    lines += ["", "## Narrative", narrative, "", "## Themes (overall top 5)"]
    for t in overall_themes:
        share = int(round(100 * t["share"]))
        lines.append(f"- **{t['label']}** - {t['count']} tickets (~{share}%) - examples: {', '.join(t['example_keys'])}")
    lines += ["", "## Themes by phase"]
    for ph in ["early", "middle", "recent"]:
        lines.append(f"### {ph.capitalize()}")
        for t in by_phase.get(ph, []):
            share = int(round(100 * t["share"]))
            lines.append(f"- **{t['label']}** - {t['count']} tickets (~{share}%) - examples: {', '.join(t['example_keys'])}")
        lines.append("")
    lines.append("## Detected shifts")
    if shifts:
        for sh in shifts:
            frm = "; ".join(x["label"] for x in sh["from"])
            to  = "; ".join(x["label"] for x in sh["to"])
            lines.append(f"- In **{sh['phase']}**: **{frm}** -> **{to}**")
    else:
        lines.append("- No strong theme shift detected.")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Source: JSONL (legacy) or MongoDB
    ap.add_argument("--input",       help="JSONL path (overrides JIRA_DUMP_PATH in .env)")
    ap.add_argument("--collection",  help="MongoDB collection name (e.g. Spring, Apache)")
    ap.add_argument("--projects",    nargs="+", help="Project key(s) to filter in MongoDB collection")
    ap.add_argument("--since",       help="Only tickets on/after this date (YYYY-MM-DD)")
    ap.add_argument("--limit",       type=int, default=None)
    ap.add_argument("--include_comments", action="store_true")
    # Analysis
    ap.add_argument("--kmeans_k",    type=int, default=24)
    ap.add_argument("--min_tickets", type=int, default=30)
    ap.add_argument("--chat_model",  default="gpt-4o-mini")
    ap.add_argument("--out_dir",     default="stories")
    ap.add_argument("--cluster_labels_json", default="",
                    help="Path to existing cluster_labels.json to skip re-labeling")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Pipeline (cached embeddings)
    if args.collection:
        from ticketing_intel.etl.mongo_loader import MongoLoader
        loader = MongoLoader(uri=cfg.mongo_uri, db=cfg.mongo_db)
        tickets = loader.load_tickets(
            collection=args.collection,
            projects=args.projects or None,
            since=args.since,
            limit=args.limit,
            include_comments=args.include_comments,
        )
        loader.close()
        cfg.validate()
        keys, X, store = run_pipeline_from_tickets(tickets, cfg)
    else:
        if args.input:
            cfg.jira_dump_path = args.input
        cfg.validate()
        keys, X, store = run_pipeline(cfg, limit=args.limit, include_comments=args.include_comments)

    all_meta = {t["key"]: t for t in store.all_tickets()}
    # Only keep metadata for keys in this run (DB may have tickets from other runs)
    all_meta = {k: all_meta[k] for k in keys if k in all_meta}
    store.close()

    # Per-ticket aligned lists
    summaries    = [(all_meta.get(k, {}).get("summary") or "").replace("\n", " ").strip() for k in keys]
    texts_tfidf  = [f"{all_meta.get(k,{}).get('summary','')} {all_meta.get(k,{}).get('description','')}" for k in keys]
    assignee_list = [all_meta.get(k, {}).get("assignee_id", "") for k in keys]
    created_str  = [all_meta.get(k, {}).get("created", "") for k in keys]

    def parse_ts(s: str) -> Optional[float]:
        try: return dateparser.parse(s).timestamp() if s else None
        except: return None

    created_ts_list  = [parse_ts(s) for s in created_str]
    created_iso_list = created_str

    # 2. Cluster once
    print(f"\n[batch] Clustering {len(keys)} tickets into {args.kmeans_k} clusters...")
    km = KMeans(n_clusters=args.kmeans_k, n_init="auto", random_state=0)
    cluster = km.fit_predict(X)
    centroids = km.cluster_centers_
    cluster_to_indices = defaultdict(list)
    for i, c in enumerate(cluster):
        cluster_to_indices[int(c)].append(i)

    # 3. Label clusters (once) — reuse existing file if provided
    cluster_labels_path = args.cluster_labels_json or os.path.join(args.out_dir, "cluster_labels.json")
    if args.cluster_labels_json and os.path.exists(args.cluster_labels_json):
        print(f"[batch] Loading existing cluster labels from {args.cluster_labels_json}")
        with open(args.cluster_labels_json, encoding="utf-8") as f:
            cluster_labels = json.load(f)
    else:
        print(f"[batch] Labeling {args.kmeans_k} clusters with LLM ({args.chat_model})...")
        cluster_labels: Dict[str, Any] = {}
        for c in range(args.kmeans_k):
            idxs = cluster_to_indices.get(c, [])
            rep  = representative_indices(X, idxs, centroids[c], top_n=6)
            samples = [{"key": keys[i], "summary": summaries[i]} for i in rep]
            label = llm_cluster_label(samples, model=args.chat_model) if samples else f"Cluster {c}"
            kw = top_keywords(texts_tfidf, idxs, top_n=10)
            cluster_labels[str(c)] = {"label": label, "keywords": kw, "size": len(idxs),
                                       "examples": [{"key": keys[i], "summary": summaries[i]} for i in rep[:3]]}
            print(f"  Cluster {c:>2}: {label}  (n={len(idxs)})")

        with open(cluster_labels_path, "w", encoding="utf-8") as f:
            json.dump(cluster_labels, f, ensure_ascii=False, indent=2)
        print(f"[batch] Saved cluster labels -> {cluster_labels_path}")

    # 4. Get qualifying assignees from store
    assignee_counts = Counter(a for a in assignee_list if a)
    targets = [(aid, n) for aid, n in assignee_counts.most_common() if n >= args.min_tickets]
    print(f"\n[batch] Generating stories for {len(targets)} assignees (>= {args.min_tickets} tickets)...\n")

    all_stories = []
    for rank, (target, ticket_count) in enumerate(targets, 1):
        print(f"[{rank}/{len(targets)}] {target[:50]}  ({ticket_count} tickets)")
        story = build_story_for_assignee(
            target=target,
            keys=keys,
            X=X,
            assignee_list=assignee_list,
            created_ts_list=created_ts_list,
            created_iso_list=created_iso_list,
            cluster=cluster,
            cluster_labels=cluster_labels,
            chat_model=args.chat_model,
            min_tickets=args.min_tickets,
        )
        if story is None:
            print(f"  Skipped (not enough tickets with timestamps)")
            continue

        short_id = target.replace("<<|author_key|", "").replace("|>>", "")[:36]
        md_path   = os.path.join(args.out_dir, f"{short_id}.md")
        json_path = os.path.join(args.out_dir, f"{short_id}.json")

        with open(md_path,   "w", encoding="utf-8") as f: f.write(story_to_md(story))
        with open(json_path, "w", encoding="utf-8") as f: json.dump(story, f, ensure_ascii=False, indent=2)

        pct = story["alignment"]["positive_percent"]
        top_theme = story["themes"]["overall_top"][0]["label"] if story["themes"]["overall_top"] else "-"
        print(f"  Top theme: {top_theme}  |  Alignment: {pct}%")
        all_stories.append({"rank": rank, "assignee_id": target, "ticket_count": ticket_count,
                             "alignment_pct": pct, "top_theme": top_theme,
                             "md": md_path, "json": json_path})

    # 5. Write summary
    summary_path = os.path.join(args.out_dir, "all_stories_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# All assignee work stories summary\n\n")
        f.write(f"| Rank | Tickets | Alignment% | Top theme | File |\n")
        f.write(f"|------|---------|------------|-----------|------|\n")
        for s in all_stories:
            fname = os.path.basename(s["md"])
            f.write(f"| {s['rank']} | {s['ticket_count']} | {s['alignment_pct']}% | {s['top_theme']} | {fname} |\n")
    print(f"\n[batch] Summary -> {summary_path}")
    print(f"[batch] Done. {len(all_stories)} stories written to {args.out_dir}/")


if __name__ == "__main__":
    main()
