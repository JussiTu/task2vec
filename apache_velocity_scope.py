# -*- coding: utf-8 -*-
"""
apache_velocity_scope.py
========================
For Apache engineers with >=10 git-linked tickets, compute:

  velocity_rho  = Spearman rank correlation of resolution-time vs. ticket order
                  negative = getting FASTER (learning)

  scope_rho     = Spearman rank correlation of files-changed vs. ticket order
                  positive = taking on BROADER changes over time

Cross-tabulate into 4 quadrants:
  Fast + Broad   (velocity_rho < -0.2, scope_rho > +0.2)  -> true skill growth
  Fast + Precise (velocity_rho < -0.2, scope_rho < -0.2)  -> surgical efficiency
  Slow + Broad   (velocity_rho > +0.2, scope_rho > +0.2)  -> harder work, still learning
  Slow + Precise (velocity_rho > +0.2, scope_rho < -0.2)  -> retreating / stuck
  Mixed          (everything else)

Also reports:
  - issuetype breakdown per quadrant
  - project breakdown
  - early vs late scope shift (do engineers take broader or narrower work over career?)

Output: .cache/apache_velocity_scope.json  +  console report

Usage:
    python apache_velocity_scope.py
    python apache_velocity_scope.py --min-tickets 15
"""

import argparse, json, re, sys
from collections import defaultdict, Counter
from pathlib import Path
from datetime import date as Date
import numpy as np
from pymongo import MongoClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE   = Path(__file__).parent / ".cache"
GIT_IDX = CACHE / "apache_git_index.json"
OUTPUT  = CACHE / "apache_velocity_scope.json"

PROJECTS = {"camel": "CAMEL", "spark": "SPARK", "hadoop": "HADOOP"}
MIN_TICKETS = 10

KEY_EXTRACT = re.compile(r'author_key\|([a-f0-9\-]+)\|')

THRESHOLD = 0.2   # rho threshold for "clear" direction


# ── helpers ──────────────────────────────────────────────────────────────────

def assignee_id(field) -> str | None:
    if not isinstance(field, dict):
        return None
    m = KEY_EXTRACT.search(str(field.get("key", "")))
    return m.group(1) if m else None


def spearman(values: list) -> float:
    n = len(values)
    if n < 5:
        return float("nan")
    ranks = np.argsort(np.argsort(values)).astype(float) + 1
    d2 = sum((i + 1 - r) ** 2 for i, r in enumerate(ranks))
    denom = n * (n * n - 1)
    return float(1 - 6 * d2 / denom) if denom else float("nan")


def quadrant(vrho: float, srho: float) -> str:
    fast = vrho < -THRESHOLD
    slow = vrho >  THRESHOLD
    broad   = srho >  THRESHOLD
    precise = srho < -THRESHOLD
    if fast and broad:   return "Fast+Broad"
    if fast and precise: return "Fast+Precise"
    if slow and broad:   return "Slow+Broad"
    if slow and precise: return "Slow+Precise"
    return "Mixed"


def days_between(created: str, resolved: str) -> float | None:
    try:
        c = Date.fromisoformat(created[:10])
        r = Date.fromisoformat(resolved[:10])
        d = (r - c).days
        return float(d) if 0 <= d <= 3650 else None
    except ValueError:
        return None


# ── load data ─────────────────────────────────────────────────────────────────

def load_linked_tickets(col, git_index: dict) -> dict:
    """
    For every Apache ticket that has a git commit, fetch its Jira fields.
    Returns {ticket_key: {assignee_id, days, files_count, issuetype, date, project}}.
    """
    git_keys = set(git_index.keys())
    print(f"Loading Jira data for {len(git_keys):,} git-linked tickets...")

    linked = {}
    cursor = col.find(
        {"key": {"$in": list(git_keys)},
         "fields.resolutiondate": {"$ne": None}},
        {"key": 1,
         "fields.assignee": 1,
         "fields.created": 1,
         "fields.resolutiondate": 1,
         "fields.issuetype": 1},
    )

    for doc in cursor:
        key    = doc["key"]
        fields = doc.get("fields", {})
        uid    = assignee_id(fields.get("assignee"))
        if not uid:
            continue
        created  = fields.get("created", "")
        resolved = fields.get("resolutiondate", "")
        days = days_between(created, resolved) if created and resolved else None
        if days is None:
            continue
        issuetype_raw = fields.get("issuetype", {})
        issuetype = (issuetype_raw.get("name", "Unknown")
                     if isinstance(issuetype_raw, dict) else "Unknown")
        git_rec = git_index[key]
        files_count = len(git_rec.get("files", []))
        if files_count == 0:
            continue  # shouldn't happen (we filtered in build step) but guard

        project = key.split("-")[0]
        linked[key] = {
            "uid":        uid,
            "days":       days,
            "files":      files_count,
            "issuetype":  issuetype,
            "date":       resolved[:10],
            "project":    project,
        }

    print(f"  {len(linked):,} tickets with Jira+git data and valid resolution time")
    return linked


# ── per-engineer analysis ────────────────────────────────────────────────────

def analyse_engineers(linked: dict, min_tickets: int) -> list[dict]:
    # Group by engineer
    by_eng = defaultdict(list)
    for key, rec in linked.items():
        by_eng[rec["uid"]].append({**rec, "key": key})

    # Filter and sort
    qualified = {uid: sorted(tix, key=lambda t: t["date"])
                 for uid, tix in by_eng.items()
                 if len(tix) >= min_tickets}
    print(f"\n  {len(qualified):,} engineers with >={min_tickets} git-linked tickets")

    results = []
    for uid, tix in qualified.items():
        days_series  = [t["days"]  for t in tix]
        files_series = [t["files"] for t in tix]

        vrho = spearman(days_series)
        srho = spearman(files_series)

        if np.isnan(vrho) or np.isnan(srho):
            continue

        quad = quadrant(vrho, srho)

        # Early vs late scope (first/last third)
        n = len(tix)
        third = max(1, n // 3)
        early_files = np.mean([t["files"] for t in tix[:third]])
        late_files  = np.mean([t["files"] for t in tix[-third:]])
        scope_delta = late_files - early_files

        # Issuetype distribution
        types = Counter(t["issuetype"] for t in tix)
        dominant_type = types.most_common(1)[0][0]

        # Project distribution
        projects = Counter(t["project"] for t in tix)
        dominant_project = projects.most_common(1)[0][0]

        results.append({
            "uid":             uid,
            "n":               n,
            "vrho":            round(vrho, 3),
            "srho":            round(srho, 3),
            "quadrant":        quad,
            "early_files":     round(early_files, 2),
            "late_files":      round(late_files, 2),
            "scope_delta":     round(scope_delta, 2),
            "median_days":     round(float(np.median(days_series)), 1),
            "median_files":    round(float(np.median(files_series)), 1),
            "dominant_type":   dominant_type,
            "dominant_project":dominant_project,
            "type_counts":     dict(types),
            "first_date":      tix[0]["date"],
            "last_date":       tix[-1]["date"],
            "first_key":       tix[0]["key"],
        })

    return results


# ── reporting ─────────────────────────────────────────────────────────────────

QUAD_DESC = {
    "Fast+Broad":   "getting faster AND taking on more files — skill growth",
    "Fast+Precise": "getting faster AND smaller changes — surgical efficiency",
    "Slow+Broad":   "slower AND broader — taking harder work, still learning",
    "Slow+Precise": "slower AND narrower — may be retreating or stuck",
    "Mixed":        "no clear trend in either dimension",
}

def report(results: list[dict]):
    n = len(results)
    print(f"\n{'='*65}")
    print("VELOCITY x SCOPE ANALYSIS — APACHE ENGINEERS")
    print(f"{'='*65}")
    print(f"\nEngineers analysed: {n}")

    # ── Quadrant distribution ─────────────────────────────────────────────────
    quad_counts = Counter(r["quadrant"] for r in results)
    print(f"\nQUADRANT DISTRIBUTION")
    print(f"  {'Quadrant':<16} {'Count':>5}  {'%':>5}   Description")
    print(f"  {'-'*64}")
    for q in ["Fast+Broad","Fast+Precise","Slow+Broad","Slow+Precise","Mixed"]:
        c = quad_counts.get(q, 0)
        print(f"  {q:<16} {c:>5}  {100*c/n:>4.1f}%   {QUAD_DESC[q]}")

    # ── Scope delta ───────────────────────────────────────────────────────────
    deltas = [r["scope_delta"] for r in results]
    arr = np.array(deltas)
    print(f"\nSCOPE SHIFT (late avg files - early avg files):")
    print(f"  mean = {arr.mean():.2f}   median = {np.median(arr):.2f}")
    print(f"  Growing broader (delta > +1):  {(arr > 1).sum()} ({100*(arr>1).mean():.1f}%)")
    print(f"  Growing precise (delta < -1):  {(arr < -1).sum()} ({100*(arr<-1).mean():.1f}%)")

    # ── Velocity rho distribution ─────────────────────────────────────────────
    vrhos = np.array([r["vrho"] for r in results])
    print(f"\nVELOCITY (Spearman rho on resolution days):")
    print(f"  mean = {vrhos.mean():.3f}   median = {np.median(vrhos):.3f}")
    print(f"  Getting faster (rho < -0.2): {(vrhos < -0.2).sum()} ({100*(vrhos<-0.2).mean():.1f}%)")
    print(f"  Getting slower (rho > +0.2): {(vrhos >  0.2).sum()} ({100*(vrhos>0.2).mean():.1f}%)")

    # ── Scope rho distribution ────────────────────────────────────────────────
    srhos = np.array([r["srho"] for r in results])
    print(f"\nSCOPE (Spearman rho on files changed):")
    print(f"  mean = {srhos.mean():.3f}   median = {np.median(srhos):.3f}")
    print(f"  Growing broader (rho > +0.2): {(srhos > 0.2).sum()} ({100*(srhos>0.2).mean():.1f}%)")
    print(f"  Growing precise (rho < -0.2): {(srhos < -0.2).sum()} ({100*(srhos<-0.2).mean():.1f}%)")

    # ── Top Fast+Broad examples ───────────────────────────────────────────────
    fb = sorted([r for r in results if r["quadrant"] == "Fast+Broad"],
                key=lambda r: r["vrho"] - r["srho"])
    print(f"\nTOP 'Fast+Broad' engineers (vrho < -0.2, srho > +0.2): {len(fb)} total")
    print(f"  {'uid[:8]':<10} {'n':>4}  {'vrho':>6}  {'srho':>6}  "
          f"{'files_early':>11}  {'files_late':>10}  {'proj':<6}  {'type'}")
    for r in fb[:12]:
        print(f"  {r['uid'][:8]:<10} {r['n']:>4}  {r['vrho']:>+.3f}  {r['srho']:>+.3f}  "
              f"{r['early_files']:>11.1f}  {r['late_files']:>10.1f}  "
              f"{r['dominant_project']:<6}  {r['dominant_type']}")

    # ── Top Fast+Precise examples ─────────────────────────────────────────────
    fp = sorted([r for r in results if r["quadrant"] == "Fast+Precise"],
                key=lambda r: r["vrho"])
    print(f"\nTOP 'Fast+Precise' engineers (vrho < -0.2, srho < -0.2): {len(fp)} total")
    for r in fp[:8]:
        print(f"  {r['uid'][:8]:<10} {r['n']:>4}  {r['vrho']:>+.3f}  {r['srho']:>+.3f}  "
              f"{r['early_files']:>11.1f}  {r['late_files']:>10.1f}  "
              f"{r['dominant_project']:<6}  {r['dominant_type']}")

    # ── Issuetype breakdown per quadrant ──────────────────────────────────────
    print(f"\nDOMINANT ISSUETYPE PER QUADRANT:")
    for q in ["Fast+Broad","Fast+Precise","Slow+Broad","Slow+Precise"]:
        group = [r for r in results if r["quadrant"] == q]
        if not group:
            continue
        types = Counter(r["dominant_type"] for r in group)
        top3 = ", ".join(f"{t}({c})" for t, c in types.most_common(3))
        print(f"  {q:<16}: {top3}")

    # ── Project breakdown ─────────────────────────────────────────────────────
    print(f"\nPER-PROJECT QUADRANT BREAKDOWN:")
    for proj in ["CAMEL","SPARK","HADOOP"]:
        grp = [r for r in results if r["dominant_project"] == proj]
        if not grp:
            continue
        qc = Counter(r["quadrant"] for r in grp)
        fb_pct = 100 * qc.get("Fast+Broad",0) / len(grp)
        fp_pct = 100 * qc.get("Fast+Precise",0) / len(grp)
        print(f"  {proj:<8}: n={len(grp)}  "
              f"Fast+Broad={qc.get('Fast+Broad',0)}({fb_pct:.0f}%)  "
              f"Fast+Precise={qc.get('Fast+Precise',0)}({fp_pct:.0f}%)  "
              f"Mixed={qc.get('Mixed',0)}")

    # ── Median files by quadrant ──────────────────────────────────────────────
    print(f"\nMEDIAN FILES CHANGED BY QUADRANT:")
    for q in ["Fast+Broad","Fast+Precise","Slow+Broad","Slow+Precise","Mixed"]:
        grp = [r for r in results if r["quadrant"] == q]
        if not grp:
            continue
        mf = np.median([r["median_files"] for r in grp])
        md = np.median([r["median_days"]  for r in grp])
        print(f"  {q:<16}: median_files={mf:.1f}  median_days={md:.0f}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-tickets", type=int, default=MIN_TICKETS)
    args = parser.parse_args()

    print("Loading git index...")
    with open(GIT_IDX, encoding="utf-8") as f:
        git_index = json.load(f)
    print(f"  {len(git_index):,} git-linked ticket keys")

    client = MongoClient("mongodb://localhost:27017/")
    col = client["jiradump"]["Apache"]

    linked = load_linked_tickets(col, git_index)
    results = analyse_engineers(linked, args.min_tickets)

    report(results)

    CACHE.mkdir(exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT}  ({len(results)} engineers)")


if __name__ == "__main__":
    main()
