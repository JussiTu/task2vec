# 2trajectories.py (PCA + arrows + directions + trends)
#
# Run:
#   python 2trajectories.py --input jira_1000.jsonl --limit 1000 --kmeans_k 24 --plot_assignee_rank 1 --trend_window 5
#
# Outputs:
#   - assignee_trajectories.csv
#   - assignee_alignment_trend.csv
#   - trajectory_arrows.png
#   - alignment_trend.png
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


def strategy_direction_pc1(X: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=1, random_state=0)
    s = pca.fit(X).components_[0]
    return (s / (np.linalg.norm(s) + 1e-12)).astype(np.float32)


def step_alignments_to_strategy(X_path: np.ndarray, s: np.ndarray) -> np.ndarray:
    if X_path.shape[0] < 2:
        return np.array([], dtype=np.float32)
    dx = X_path[1:] - X_path[:-1]
    dxn = normalize(dx, norm="l2")
    sims = cosine_similarity(dxn, s.reshape(1, -1)).reshape(-1)
    return sims.astype(np.float32)


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size == 0:
        return x.copy()
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.size):
        lo = max(0, i - w + 1)
        out[i] = float(np.mean(x[lo:i + 1]))
    return out


# -----------------------------
# Plotting
# -----------------------------
def plot_pca_with_arrows(
    X: np.ndarray,
    points_idx_path: List[int],
    out_png: str,
    title: str,
    arrow_stride: int = 1,
    arrow_scale: float = 1.0,
    dir_scale: float = 1.0,
    early_k: int = 6,
    recent_k: int = 6,
):
    Z = PCA(n_components=2, random_state=0).fit_transform(X)

    plt.figure(figsize=(10, 7))
    plt.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.30)

    if not points_idx_path or len(points_idx_path) < 2:
        plt.title(title + " (not enough points for arrows)")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()
        return

    path = Z[points_idx_path]
    plt.plot(path[:, 0], path[:, 1], linewidth=1.8, alpha=0.9)
    plt.scatter(path[:, 0], path[:, 1], s=26, alpha=0.95)

    dx = path[1:] - path[:-1]
    bases = path[:-1]
    if arrow_stride > 1:
        bases = bases[::arrow_stride]
        dx = dx[::arrow_stride]

    plt.quiver(
        bases[:, 0], bases[:, 1], dx[:, 0], dx[:, 1],
        angles="xy", scale_units="xy", scale=(1.0 / max(1e-9, arrow_scale)),
        width=0.003, alpha=0.65,
    )

    start, end = path[0], path[-1]
    net = end - start
    plt.quiver(
        [start[0]], [start[1]], [net[0]], [net[1]],
        angles="xy", scale_units="xy", scale=(1.0 / max(1e-9, dir_scale)),
        width=0.006, alpha=0.95,
    )

    def avg_dir(vecs: np.ndarray) -> np.ndarray:
        if vecs.shape[0] == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    k1 = min(early_k, path.shape[0] - 1)
    k2 = min(recent_k, path.shape[0] - 1)
    early_vec = avg_dir(path[1:1 + k1] - path[:k1]) if k1 > 0 else np.zeros(2)
    recent_vec = avg_dir(path[-k2:] - path[-k2 - 1:-1]) if k2 > 0 else np.zeros(2)

    for anchor, vec in [(start, early_vec), (end, recent_vec)]:
        plt.quiver(
            [anchor[0]], [anchor[1]], [vec[0]], [vec[1]],
            angles="xy", scale_units="xy", scale=(1.0 / max(1e-9, dir_scale)),
            width=0.006, alpha=0.85,
        )

    plt.title(title)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()


def plot_alignment_trend(
    created_iso: List[str],
    align_step: np.ndarray,
    align_roll: np.ndarray,
    out_png: str,
    title: str,
):
    if align_step.size == 0:
        plt.figure(figsize=(10, 4))
        plt.title(title + " (not enough points)")
        plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()
        return

    x = np.arange(align_step.size)
    plt.figure(figsize=(10, 4.6))
    plt.plot(x, align_step, linewidth=1.2, alpha=0.65, label="step alignment")
    plt.plot(x, align_roll, linewidth=2.2, alpha=0.95, label="rolling mean")

    if created_iso:
        ticks = np.linspace(0, len(created_iso) - 2, num=min(6, len(created_iso) - 1), dtype=int)
        plt.xticks(ticks, [created_iso[t] for t in ticks], rotation=20, ha="right")

    plt.ylim(-1.05, 1.05)
    plt.axhline(0.0, linewidth=1.0, alpha=0.35)
    plt.title(title)
    plt.xlabel("Step (ordered by created time)")
    plt.ylabel("cos(step, strategy)")
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()


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
    ap.add_argument("--trend_window", type=int, default=5)
    ap.add_argument("--arrow_stride", type=int, default=1)
    ap.add_argument("--out_csv", default="assignee_trajectories.csv")
    ap.add_argument("--out_trend_csv", default="assignee_alignment_trend.csv")
    ap.add_argument("--out_png", default="trajectory_arrows.png")
    ap.add_argument("--out_trend_png", default="alignment_trend.png")
    args = ap.parse_args()

    if args.input:
        cfg.jira_dump_path = args.input

    keys, X, store = run_pipeline(cfg, limit=args.limit, include_comments=args.include_comments)

    clusters = kmeans_labels(X, k=args.kmeans_k)
    s = strategy_direction_pc1(X)

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
            "cluster": int(clusters[i]),
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
        w.writerow(["assignee_id", "rank_by_count", "ticket_key", "created_iso", "cluster"])
        for rank, a in enumerate(assignees, 1):
            items = [r for r in ticket_rows if r["assignee_id"] == a and r["created_ts"] is not None]
            items.sort(key=lambda r: r["created_ts"])
            for r in items:
                w.writerow([a, rank, r["key"], r["created_iso"], r["cluster"]])
    print(f"Wrote trajectories CSV -> {args.out_csv}")

    # Choose one assignee to plot + trend
    plot_rank = max(1, min(args.plot_assignee_rank, len(assignees)))
    chosen = assignees[plot_rank - 1]
    chosen_items = [r for r in ticket_rows if r["assignee_id"] == chosen and r["created_ts"] is not None]
    chosen_items.sort(key=lambda r: r["created_ts"])
    chosen_idxs = [r["idx"] for r in chosen_items]
    chosen_times = [r["created_iso"] for r in chosen_items]

    plot_pca_with_arrows(
        X=X,
        points_idx_path=chosen_idxs,
        out_png=args.out_png,
        title=f"PCA task space + trajectory vectors (assignee #{plot_rank}, n={len(chosen_idxs)})",
        arrow_stride=max(1, args.arrow_stride),
    )
    print(f"Wrote trajectory arrows PNG -> {args.out_png}")

    X_path = X[chosen_idxs] if chosen_idxs else np.zeros((0, X.shape[1]), dtype=np.float32)
    align_step = step_alignments_to_strategy(X_path, s)
    align_roll = rolling_mean(align_step, args.trend_window)

    with open(args.out_trend_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["assignee_id", "step_index", "from_ticket_key", "to_ticket_key", "to_created_iso", "step_alignment", "rolling_alignment"])
        for j in range(len(chosen_idxs) - 1):
            w.writerow([
                chosen, j,
                chosen_items[j]["key"], chosen_items[j + 1]["key"],
                chosen_items[j + 1]["created_iso"],
                float(align_step[j]), float(align_roll[j]),
            ])
    print(f"Wrote alignment trend CSV -> {args.out_trend_csv}")

    plot_alignment_trend(
        created_iso=chosen_times,
        align_step=align_step,
        align_roll=align_roll,
        out_png=args.out_trend_png,
        title=f"Alignment trend to strategy axis (assignee #{plot_rank}, window={args.trend_window})",
    )
    print(f"Wrote alignment trend PNG -> {args.out_trend_png}")

    if align_step.size:
        print("\nAlignment summary:")
        print(f"  mean(step alignment):   {float(np.mean(align_step)):.3f}")
        print(f"  recent rolling value:   {float(align_roll[-1]):.3f}")
        print(f"  min / max:              {float(np.min(align_step)):.3f} / {float(np.max(align_step)):.3f}")

    store.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
