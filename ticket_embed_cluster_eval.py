# ticket_embed_cluster_eval.py
#
# Goal:
# 1) Embed tickets (from cache if available - no repeat API calls)
# 2) Cluster (HDBSCAN if available, else KMeans)
# 3) Evaluate clustering vs metadata fields (purity + NMI + ARI)
# 4) Save a report JSON + print a readable summary
#
# Run:
#   python ticket_embed_cluster_eval.py --input jira_100.jsonl --limit 100
#   python ticket_embed_cluster_eval.py --input jira_1000.jsonl --limit 1000 --cluster_method kmeans --kmeans_k 12

import json
import argparse
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

from ticketing_intel.config import cfg
from ticketing_intel.etl.pipeline import run_pipeline
from ticketing_intel.etl.loader import stream_jsonl


# -----------------------------
# Metadata extraction (from raw JSONL - cheap, no embedding)
# -----------------------------
def _safe_str(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    try: return json.dumps(x, ensure_ascii=False)
    except Exception: return str(x)


def extract_metadata_fields(doc: Dict[str, Any]) -> Dict[str, str]:
    fields = doc.get("fields", {}) or {}
    md: Dict[str, str] = {}
    md["issuetype"] = _safe_str((fields.get("issuetype") or {}).get("name", "")).strip()
    md["status"] = _safe_str((fields.get("status") or {}).get("name", "")).strip()
    md["project_key"] = _safe_str((fields.get("project") or {}).get("key", "")).strip()
    md["priority"] = _safe_str((fields.get("priority") or {}).get("name", "")).strip()
    md["resolution"] = _safe_str((fields.get("resolution") or {}).get("name", "")).strip()
    comps = fields.get("components") or []
    if isinstance(comps, list) and comps:
        names = [_safe_str(c["name"]) for c in comps[:3] if isinstance(c, dict) and c.get("name")]
        md["component"] = ";".join(names).strip()
    else:
        md["component"] = ""
    labels = fields.get("labels") or []
    md["label_top"] = _safe_str(labels[0]) if isinstance(labels, list) and labels else ""
    return md


def load_metadata(path: str, limit: Optional[int]) -> Dict[str, Dict[str, str]]:
    """Read JSONL for metadata only (no embedding). Returns {key: metadata_dict}."""
    meta = {}
    for doc in stream_jsonl(path, limit=limit):
        key = str(doc.get("key") or doc.get("id") or "").strip()
        if key:
            meta[key] = extract_metadata_fields(doc)
    return meta


# -----------------------------
# Clustering
# -----------------------------
def cluster_embeddings(X: np.ndarray, method: str = "hdbscan", k: int = 12, pca_dims: int = 64) -> np.ndarray:
    """X is already L2-normalised from the pipeline."""
    if method == "hdbscan":
        if not HAS_HDBSCAN:
            raise RuntimeError("HDBSCAN requested but not installed. pip install hdbscan")
        Xr = PCA(n_components=min(pca_dims, X.shape[1]), random_state=0).fit_transform(X)
        return hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3, metric="euclidean").fit_predict(Xr)
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(X)
    raise ValueError(f"Unknown clustering method: {method}")


# -----------------------------
# Evaluation vs metadata
# -----------------------------
def purity_score(cluster_labels: np.ndarray, y: List[str]) -> float:
    pairs = [(c, lbl) for c, lbl in zip(cluster_labels, y) if lbl != ""]
    if not pairs:
        return float("nan")
    clusters = defaultdict(list)
    for c, lbl in pairs:
        clusters[int(c)].append(lbl)
    correct = sum(Counter(lbls).most_common(1)[0][1] for lbls in clusters.values())
    return correct / len(pairs)


def nmi_score(cluster_labels: np.ndarray, y: List[str]) -> float:
    idx = [i for i, lbl in enumerate(y) if lbl != ""]
    if len(idx) < 2:
        return float("nan")
    c = cluster_labels[idx]
    yy = [y[i] for i in idx]
    mapping = {s: j for j, s in enumerate(sorted(set(yy)))}
    return float(normalized_mutual_info_score([mapping[s] for s in yy], c))


def ari_score(cluster_labels: np.ndarray, y: List[str]) -> float:
    idx = [i for i, lbl in enumerate(y) if lbl != ""]
    if len(idx) < 2:
        return float("nan")
    yy = [y[i] for i in idx]
    if len(set(yy)) < 2:
        return float("nan")
    mapping = {s: j for j, s in enumerate(sorted(set(yy)))}
    return float(adjusted_rand_score([mapping[s] for s in yy], cluster_labels[idx]))


def top_enrichments(cluster_labels: np.ndarray, y: List[str], topn: int = 5) -> Dict[int, List[Tuple[str, int, float]]]:
    clusters = defaultdict(list)
    for c, lbl in zip(cluster_labels, y):
        if lbl != "":
            clusters[int(c)].append(lbl)
    return {
        c: [(lab, n, n / len(lbls)) for lab, n in Counter(lbls).most_common(topn)]
        for c, lbls in clusters.items()
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="JSONL path (overrides JIRA_DUMP_PATH in .env)")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--include_comments", action="store_true")
    ap.add_argument("--cluster_method", choices=["hdbscan", "kmeans"], default="hdbscan")
    ap.add_argument("--kmeans_k", type=int, default=12)
    ap.add_argument("--pca_dims", type=int, default=64)
    ap.add_argument("--out_report", default="cluster_eval_report.json")
    args = ap.parse_args()

    if args.input:
        cfg.jira_dump_path = args.input

    # Embeddings (cached)
    keys, X, store = run_pipeline(cfg, limit=args.limit, include_comments=args.include_comments)

    # Metadata (re-read JSONL for full fields including priority/resolution/component)
    raw_meta = load_metadata(cfg.jira_dump_path, limit=args.limit)
    md_rows = [raw_meta.get(key, {}) for key in keys]

    store.close()

    labels = cluster_embeddings(X, method=args.cluster_method, k=args.kmeans_k, pca_dims=args.pca_dims)

    fields_to_check = ["issuetype", "status", "project_key", "priority", "resolution", "component", "label_top"]

    report: Dict[str, Any] = {
        "n_docs": len(keys),
        "cluster_method": args.cluster_method,
        "clusters": dict(Counter(map(int, labels))),
        "metrics": {},
        "enrichments": {},
    }

    print("\nCluster sizes:")
    for cid, n in Counter(map(int, labels)).most_common():
        print(f"  {cid:>3}: {n}")

    print("\nMetadata agreement (content-only embedding -> clustering vs Jira metadata):")
    for f in fields_to_check:
        y = [r.get(f, "") for r in md_rows]
        pur = purity_score(labels, y)
        nmi = nmi_score(labels, y)
        ari = ari_score(labels, y)
        present = sum(1 for v in y if v != "")
        unique = len(set(v for v in y if v != ""))

        report["metrics"][f] = {"present": present, "unique": unique, "purity": pur, "nmi": nmi, "ari": ari}

        print(f"\n- {f}")
        print(f"  present: {present}/{len(y)}  unique: {unique}")
        print(f"  purity:  {pur:.3f}" if pur == pur else "  purity:  NaN")
        print(f"  NMI:     {nmi:.3f}" if nmi == nmi else "  NMI:     NaN")
        print(f"  ARI:     {ari:.3f}" if ari == ari else "  ARI:     NaN")

        enrich = top_enrichments(labels, y, topn=3)
        report["enrichments"][f] = {
            str(cid): [{"label": lab, "count": int(cnt), "share": float(share)} for (lab, cnt, share) in lst]
            for cid, lst in enrich.items() if cid != -1
        }

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved report -> {args.out_report}")

    scored = [
        (m["nmi"], m["purity"], f, m["present"], m["unique"])
        for f, m in report["metrics"].items()
        if m["nmi"] == m["nmi"] and m["purity"] == m["purity"]
    ]
    scored.sort(reverse=True)
    if scored:
        print("\nTop metadata fields aligned with content-only clusters (higher is better):")
        for nmi, pur, f, present, unique in scored[:5]:
            print(f"  {f:12s}  NMI={nmi:.3f}  purity={pur:.3f}  present={present}  unique={unique}")


if __name__ == "__main__":
    main()
