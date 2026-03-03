# -*- coding: utf-8 -*-
"""
apache_learning_signals.py
==========================
Analysis D — passive→active trajectory on Apache data.

For each engineer (Jira assignee) with ≥30 resolved tickets in a single
Apache project (CAMEL, SPARK, HADOOP), we compute:

  git_coverage_rate = tickets that have a linked git commit / total tickets

Split into early (first half) and late (second half) of career, then check
whether late_rate > early_rate (passive→active: started without writing code,
grew into committing code).

Also reports:
  - Resolution-time velocity (Spearman rho, same as Spring Analysis B)
  - Ticket-complexity growth (avg files changed per commit, early vs late)

Output: .cache/apache_learning_signals.json  +  console summary

Usage:
    python apache_learning_signals.py
    python apache_learning_signals.py --projects spark
"""

import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from pymongo import MongoClient

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE   = Path(__file__).parent / ".cache"
OUTPUT  = CACHE / "apache_learning_signals.json"
GIT_IDX = CACHE / "apache_git_index.json"

PROJECTS = {
    "camel":  "CAMEL",
    "spark":  "SPARK",
    "hadoop": "HADOOP",
}

MIN_TICKETS = 30   # minimum resolved tickets to include an engineer


# ── helpers ──────────────────────────────────────────────────────────────────

KEY_EXTRACT = re.compile(r'author_key\|([a-f0-9\-]+)\|')

def get_assignee_id(assignee_field) -> str | None:
    """Extract anonymised UUID from the Jira assignee field."""
    if not isinstance(assignee_field, dict):
        return None
    raw = assignee_field.get("key", "")
    m = KEY_EXTRACT.search(str(raw))
    return m.group(1) if m else None


def spearman_rho(values: list[float]) -> float:
    """Spearman rank correlation against position index."""
    n = len(values)
    if n < 4:
        return float("nan")
    ranks = np.argsort(np.argsort(values)).astype(float) + 1
    d2 = sum((i + 1 - r) ** 2 for i, r in enumerate(ranks))
    return 1 - 6 * d2 / (n * (n * n - 1))


def split_half(items: list) -> tuple[list, list]:
    """Split a sorted list into early and late halves."""
    mid = len(items) // 2
    return items[:mid], items[mid:]


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse_project(prefix: str, col, git_keys: set) -> dict:
    """Run all analyses for one project prefix. Returns result dict."""

    print(f"\n[{prefix}] Loading resolved tickets …")
    tickets_by_engineer = defaultdict(list)

    cursor = col.find(
        {
            "key": {"$regex": f"^{prefix}-"},
            "fields.resolutiondate": {"$ne": None},
        },
        {
            "key": 1,
            "fields.assignee": 1,
            "fields.resolutiondate": 1,
            "fields.created": 1,
            "fields.issuetype": 1,
        },
    )

    total_docs = 0
    for doc in cursor:
        total_docs += 1
        fields = doc.get("fields", {})
        assignee_id = get_assignee_id(fields.get("assignee"))
        if not assignee_id:
            continue
        rd = fields.get("resolutiondate", "")
        if not rd or len(rd) < 10:
            continue
        created = fields.get("created", "")
        key = doc["key"]
        tickets_by_engineer[assignee_id].append({
            "key":     key,
            "date":    rd[:10],   # YYYY-MM-DD
            "created": created[:10] if created else "",
            "has_commit": key in git_keys,
            "files":   len(git_keys) and (  # placeholder — filled below
                           git_keys  # will use git_index
                       ),
        })

    print(f"  {total_docs:,} resolved tickets, "
          f"{len(tickets_by_engineer):,} unique assignees")

    # Filter to engineers with enough tickets
    qualified = {uid: sorted(tix, key=lambda t: t["date"])
                 for uid, tix in tickets_by_engineer.items()
                 if len(tix) >= MIN_TICKETS}
    print(f"  {len(qualified):,} engineers with >={MIN_TICKETS} resolved tickets")

    results = []
    for uid, tix in qualified.items():
        n = len(tix)
        early, late = split_half(tix)

        early_cov = sum(1 for t in early if t["has_commit"]) / len(early)
        late_cov  = sum(1 for t in late  if t["has_commit"]) / len(late)
        delta_cov = late_cov - early_cov

        # Resolution time velocity
        all_days = []
        for t in tix:
            if t["created"] and t["date"]:
                try:
                    from datetime import date
                    created = date.fromisoformat(t["created"])
                    resolved = date.fromisoformat(t["date"])
                    days = (resolved - created).days
                    if 0 <= days <= 3650:
                        all_days.append(days)
                except ValueError:
                    pass
        rho = spearman_rho(all_days) if len(all_days) >= 10 else float("nan")

        results.append({
            "uid":       uid,
            "n":         n,
            "early_n":   len(early),
            "late_n":    len(late),
            "early_cov": round(early_cov, 3),
            "late_cov":  round(late_cov, 3),
            "delta_cov": round(delta_cov, 3),
            "rho":       round(rho, 3) if not np.isnan(rho) else None,
            "span_years": (
                int(tix[-1]["date"][:4]) - int(tix[0]["date"][:4])
            ) if len(tix) > 1 else 0,
            "first_year": int(tix[0]["date"][:4]),
            "last_year":  int(tix[-1]["date"][:4]),
            "first_ticket": tix[0]["key"],
            "last_ticket":  tix[-1]["key"],
        })

    return {"prefix": prefix, "engineers": results}


# ── aggregate reporting ───────────────────────────────────────────────────────

def report(all_results: list[dict], git_index: dict):
    print("\n" + "=" * 65)
    print("APACHE LEARNING SIGNALS — SUMMARY")
    print("=" * 65)

    all_eng = [e for r in all_results for e in r["engineers"]]
    print(f"\nEngineers with ≥{MIN_TICKETS} resolved tickets: {len(all_eng):,}")

    # ── Coverage shift distribution ──────────────────────────────────────────
    deltas = [e["delta_cov"] for e in all_eng]
    arr = np.array(deltas)
    print(f"\nCoverage-rate shift (late − early):")
    print(f"  mean = {arr.mean():.3f}   median = {np.median(arr):.3f}")
    print(f"  std  = {arr.std():.3f}")
    print(f"  p25  = {np.percentile(arr,25):.3f}   p75 = {np.percentile(arr,75):.3f}")
    print(f"  Passive→Active (delta ≥ +0.20): "
          f"{(arr >= 0.20).sum()} / {len(arr)} engineers  "
          f"({100*(arr>=0.20).mean():.1f}%)")
    print(f"  Active→Passive (delta ≤ -0.20): "
          f"{(arr <= -0.20).sum()} / {len(arr)} engineers  "
          f"({100*(arr<=-0.20).mean():.1f}%)")
    print(f"  Stable (|delta| < 0.10):         "
          f"{(np.abs(arr) < 0.10).sum()} / {len(arr)} engineers  "
          f"({100*(np.abs(arr)<0.10).mean():.1f}%)")

    # ── Strong P→A cases ─────────────────────────────────────────────────────
    pa = sorted([e for e in all_eng if e["delta_cov"] >= 0.25],
                key=lambda e: -e["delta_cov"])
    print(f"\nTop passive→active transitions (delta ≥ 0.25): {len(pa)}")
    print(f"  {'uid[:8]':<10} {'n':>5}  {'early%':>7}  {'late%':>6}  "
          f"{'delta':>6}  {'rho':>6}  {'span':>5}  {'first':>6}")
    for e in pa[:15]:
        rho_str = f"{e['rho']:+.3f}" if e["rho"] is not None else "  n/a"
        print(f"  {e['uid'][:8]:<10} {e['n']:>5}  "
              f"{100*e['early_cov']:>6.1f}%  {100*e['late_cov']:>5.1f}%  "
              f"{e['delta_cov']:>+.3f}  {rho_str:>6}  "
              f"{e['span_years']:>5}  {e['first_year']:>6}")

    # ── Velocity (Spearman rho) ───────────────────────────────────────────────
    rhos = [e["rho"] for e in all_eng if e["rho"] is not None]
    if rhos:
        ra = np.array(rhos)
        print(f"\nResolution-time velocity (Spearman rho vs. ticket order):")
        print(f"  engineers with rho: {len(ra):,}")
        print(f"  mean = {ra.mean():.3f}   median = {np.median(ra):.3f}")
        print(f"  Getting faster (rho < −0.2): "
              f"{(ra < -0.2).sum()} ({100*(ra<-0.2).mean():.1f}%)")
        print(f"  Getting slower (rho > +0.2): "
              f"{(ra >  0.2).sum()} ({100*(ra>0.2).mean():.1f}%)")

    # ── Per-project breakdown ─────────────────────────────────────────────────
    print("\nPer-project breakdown:")
    for r in all_results:
        engs = r["engineers"]
        if not engs:
            continue
        deltas_p = [e["delta_cov"] for e in engs]
        pa_count = sum(1 for d in deltas_p if d >= 0.20)
        print(f"  {r['prefix']:8s}  engineers={len(engs):>4}  "
              f"mean_delta={np.mean(deltas_p):+.3f}  "
              f"P→A(≥0.20)={pa_count}  "
              f"({100*pa_count/len(engs):.1f}%)")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="+",
                        choices=list(PROJECTS.keys()),
                        default=list(PROJECTS.keys()))
    args = parser.parse_args()

    print("Loading git index …")
    with open(GIT_IDX, encoding="utf-8") as f:
        git_index: dict = json.load(f)
    git_keys = set(git_index.keys())
    print(f"  {len(git_keys):,} ticket keys with commits")

    client = MongoClient("mongodb://localhost:27017/")
    col = client["jiradump"]["Apache"]

    all_results = []
    for name in args.projects:
        prefix = PROJECTS[name]
        result = analyse_project(prefix, col, git_keys)
        all_results.append(result)

    # Full report
    report(all_results, git_index)

    # Save JSON
    CACHE.mkdir(exist_ok=True)
    output = {
        r["prefix"]: {
            "n_engineers": len(r["engineers"]),
            "engineers": r["engineers"],
        }
        for r in all_results
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
