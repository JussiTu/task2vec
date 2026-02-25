# embed_cluster_plot.py
# Embed Jira issues, cluster, project to 2D, plot,
# then label clusters automatically using an LLM.
#
# Run:
#   python embed_cluster_plot.py
#   python embed_cluster_plot.py --input jira_100.jsonl --limit 100
#
# Embeddings are cached in .cache/embeddings.npz — no API calls on repeat runs.

import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

from openai import OpenAI

from ticketing_intel.config import cfg
from ticketing_intel.etl.pipeline import run_pipeline


# ----------------------------
# Clustering + 2D projection
# ----------------------------
def cluster_embeddings(X: np.ndarray):
    """X is already L2-normalised from the pipeline."""
    if HAS_HDBSCAN:
        Xr = PCA(n_components=min(64, X.shape[1]), random_state=0).fit_transform(X)
        labels = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2).fit_predict(Xr)
        return labels, X

    labels = KMeans(n_clusters=6, n_init="auto", random_state=0).fit_predict(X)
    return labels, X


def project_2d(X: np.ndarray) -> np.ndarray:
    if HAS_UMAP:
        return umap.UMAP(n_neighbors=10, min_dist=0.15, metric="cosine", random_state=0).fit_transform(X)
    return PCA(n_components=2, random_state=0).fit_transform(X)


# ----------------------------
# Cluster labeling (LLM)
# ----------------------------
def closest_to_centroid(X: np.ndarray, idxs: np.ndarray, topk: int = 12) -> np.ndarray:
    C = X[idxs].mean(axis=0)
    C = C / (np.linalg.norm(C) + 1e-12)
    sims = X[idxs] @ C
    return idxs[np.argsort(-sims)[:topk]]


def label_clusters_llm(
    store_meta: Dict[str, Dict],   # key → {summary, description, issuetype, status, project_key}
    keys: List[str],
    labels: np.ndarray,
    X: np.ndarray,
    topk: int = 12,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    client = OpenAI()
    clusters = defaultdict(list)
    for i, lab in enumerate(labels):
        clusters[int(lab)].append(i)

    out: Dict[str, Any] = {}
    for lab, idx_list in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        if lab == -1:
            continue
        idxs = np.array(idx_list, dtype=int)
        reps = closest_to_centroid(X, idxs, topk=topk)

        evidence_lines = []
        for i in reps:
            key = keys[i]
            meta = store_meta.get(key, {})
            summary = (meta.get("summary") or "").replace("\n", " ")[:120]
            desc = (meta.get("description") or "").replace("\n", " ")[:300]
            itype = meta.get("issuetype", "")
            status = meta.get("status", "")
            proj = meta.get("project_key", "")
            evidence_lines.append(
                f"- {key} | {proj} | {itype} | {status}\n"
                f"  Summary: {summary}\n"
                f"  Desc: {desc}"
            )
        evidence = "\n".join(evidence_lines)

        prompt = f"""
You are labeling a cluster of Jira issues to create an Orthobullets-style learning topic.

Given the example issues below, produce:
1) title: <= 8 words, specific but not overly narrow
2) keywords: 5–10 comma-separated terms
3) description: 2–4 sentences describing the common problem pattern
4) typical_signals: 3–6 bullet points (symptoms/logs/context)
5) common_fixes: 3–6 bullet points (typical remediation patterns)
6) out_of_scope: 2–4 bullet points (similar issues that do NOT belong)

Be concrete and avoid generic phrases like "various issues".
Return strict JSON only.

EXAMPLES (cluster members):
{evidence}
""".strip()

        resp = client.responses.create(model=model, input=prompt, temperature=0.2)
        text = (resp.output_text or "").strip()
        try:
            data = json.loads(text)
        except Exception:
            data = {"raw": text}

        data["cluster_id"] = lab
        data["n_issues"] = int(len(idxs))
        out[str(lab)] = data

        print(f"\n=== Cluster {lab} (n={len(idxs)}) ===")
        if "title" in data:
            print("Title:", data["title"])
            print("Keywords:", data.get("keywords", ""))
        else:
            print("LLM output (raw):", data.get("raw", "")[:400])

    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="JSONL path (overrides JIRA_DUMP_PATH in .env)")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--include_comments", action="store_true")
    ap.add_argument("--llm_model", default="gpt-4.1-mini")
    args = ap.parse_args()

    if args.input:
        cfg.jira_dump_path = args.input

    keys, X, store = run_pipeline(cfg, limit=args.limit, include_comments=args.include_comments)
    store_meta = {t["key"]: t for t in store.all_tickets()}

    labels, Xn = cluster_embeddings(X)
    X2 = project_2d(Xn)

    # Plot
    plt.figure(figsize=(9, 7))
    plt.scatter(X2[:, 0], X2[:, 1], s=40)
    plt.title("Jira issues (embeddings) — 2D projection")

    u, c = np.unique(labels, return_counts=True)
    print("\nCluster counts:")
    for uu, cc in zip(u, c):
        print(f"  {uu}: {cc}")

    print("\nExamples per cluster:")
    for uu in u:
        idxs = np.where(labels == uu)[0]
        print(f"\n=== Cluster {uu} (n={len(idxs)}) ===")
        for j in idxs[:5]:
            meta = store_meta.get(keys[j], {})
            print("-", (meta.get("summary") or "")[:140].replace("\n", " "))

    print("\nLabeling clusters with LLM...")
    cluster_meta = label_clusters_llm(
        store_meta=store_meta,
        keys=keys,
        labels=labels,
        X=Xn,
        topk=10,
        model=args.llm_model,
    )

    with open("cluster_labels.json", "w", encoding="utf-8") as f:
        json.dump(cluster_meta, f, ensure_ascii=False, indent=2)

    print("\nSaved cluster labels to cluster_labels.json")
    store.close()
    plt.show()


if __name__ == "__main__":
    main()
