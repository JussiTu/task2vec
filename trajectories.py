# trajectories.py
#
# Run:
#   python trajectories.py --input jira_1000.jsonl --limit 1000 --kmeans_k 24 --assignee_top 10 --plot_assignee_rank 1
#
# Outputs:
#   - assignee_trajectories.csv
#   - trajectory.png
#
# Embeddings are cached in .cache/embeddings.npz - no API calls on repeat runs.

import csv
import argparse
from collections import Counter
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser as dateparser
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from ticketing_intel.config import cfg
from ticketing_intel.etl.pipeline import run_pipeline


# -----------------------------
# Clustering + strategy
# -----------------------------
def kmeans_labels(X: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    return km.fit_predict(X)  # X already L2-normalised from pipeline


def strategy_direction(X: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=1, random_state=0)
    pc1 = pca.fit(X).components_[0]
    return (pc1 / (np.linalg.norm(pc1) + 1e-12)).astype(np.float32)


def alignment_score(path_vecs: np.ndarray, s: np.ndarray) -> float:
    if path_vecs.size == 0:
        return float("nan")
    sims = cosine_similarity(path_vecs, s.reshape(1, -1)).reshape(-1)
    return float(np.mean(sims))


# -----------------------------
# Plotting
# -----------------------------
def plot_space_with_trajectory(
    X: np.ndarray,
    assignee_points: List[int],
    out_png: str,
    title: str,
):
    Z = PCA(n_components=2, random_state=0).fit_transform(X)
    plt.figure(figsize=(10, 7))
    plt.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.35)
    if assignee_points:
        path = Z[assignee_points]
        plt.plot(path[:, 0], path[:, 1], linewidth=2)
        plt.scatter(path[:, 0], path[:, 1], s=22)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="JSONL path (overrides JIRA_DUMP_PATH in .env)")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--include_comments", action="store_true")
    ap.add_argument("--kmeans_k", type=int, default=24)
    ap.add_argument("--min_tickets_per_assignee", type=int, default=5)
    ap.add_argument("--assignee_top", type=int, default=10)
    ap.add_argument("--plot_assignee_rank", type=int, default=1)
    ap.add_argument("--out_csv", default="assignee_trajectories.csv")
    ap.add_argument("--out_png", default="trajectory.png")
    args = ap.parse_args()

    if args.input:
        cfg.jira_dump_path = args.input

    keys, X, store = run_pipeline(cfg, limit=args.limit, include_comments=args.include_comments)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    cl = kmeans_labels(X, k=args.kmeans_k)
    s = strategy_direction(X)

    # Build ticket rows from store metadata
    all_meta = {t["key"]: t for t in store.all_tickets()}
    ticket_rows = []
    assignee_counts = Counter()

    for i, key in enumerate(keys):
        meta = all_meta.get(key, {})
        aid = meta.get("assignee_id", "")
        created = meta.get("created", "")
        ts = None
        try:
            ts = dateparser.parse(created).timestamp() if created else None
        except Exception:
            pass
        if aid:
            assignee_counts[aid] += 1
        ticket_rows.append({
            "idx": i,
            "assignee_id": aid,
            "created_ts": ts,
            "created_iso": created,
            "key": key,
            "cluster": int(cl[i]),
        })

    assignees = [
        a for a, n in assignee_counts.most_common()
        if n >= args.min_tickets_per_assignee
    ]
    if not assignees:
        raise SystemExit("No assignees with sufficient tickets found.")

    print("\nTop assignees by ticket count:")
    for rank, a in enumerate(assignees[:args.assignee_top], 1):
        print(f"  {rank:>2}. {a}  ({assignee_counts[a]} tickets)")

    # Write trajectories CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["assignee_id", "rank_by_count", "ticket_key", "created_iso", "cluster", "alignment_to_strategy"])
        for rank, a in enumerate(assignees, 1):
            items = [r for r in ticket_rows if r["assignee_id"] == a and r["created_ts"] is not None]
            items.sort(key=lambda r: r["created_ts"])
            idxs = [r["idx"] for r in items]
            score = alignment_score(X[idxs], s) if idxs else float("nan")
            for r in items:
                w.writerow([a, rank, r["key"], r["created_iso"], r["cluster"], score])
    print(f"Wrote trajectories CSV -> {args.out_csv}")

    # Plot one assignee trajectory
    plot_rank = max(1, min(args.plot_assignee_rank, len(assignees)))
    chosen = assignees[plot_rank - 1]
    chosen_items = [r for r in ticket_rows if r["assignee_id"] == chosen and r["created_ts"] is not None]
    chosen_items.sort(key=lambda r: r["created_ts"])
    chosen_idxs = [r["idx"] for r in chosen_items]

    plot_space_with_trajectory(
        X=X,
        assignee_points=chosen_idxs,
        out_png=args.out_png,
        title=f"Task space + trajectory for assignee #{plot_rank} (tickets={len(chosen_idxs)})",
    )
    print(f"Wrote trajectory plot -> {args.out_png}")

    # Alignment leaderboard
    print("\nAlignment to strategy axis:")
    scores = []
    for a in assignees[:args.assignee_top]:
        items = [r for r in ticket_rows if r["assignee_id"] == a and r["created_ts"] is not None]
        idxs = [r["idx"] for r in items]
        sc = alignment_score(X[idxs], s) if idxs else float("nan")
        scores.append((sc, a, len(idxs)))
    scores.sort(reverse=True)
    for sc, a, n in scores:
        print(f"  {sc:>7.3f}  {a}  (n={n})")

    store.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
