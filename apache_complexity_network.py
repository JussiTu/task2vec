# -*- coding: utf-8 -*-
"""
apache_complexity_network.py
============================
Combines three signals into a unified task understanding:

1. COMPOSITE COMPLEXITY SCORE (from git)
   score = 0.4 * n_modules  +  0.3 * log(n_files+1)  +  0.3 * core_pct
   normalised 0-1 across all tickets

2. NETWORK SIGNALS (from Jira)
   - comment_count  : discussion depth (more = more complex / contentious)
   - watch_count    : social visibility (more = more people care)
   - vote_count     : community demand
   - has_links      : connected to other tickets (dependency network)
   - cross_assigned : reporter != assignee (hand-off / collaboration)
   - priority       : Blocker/Critical/Major/Minor/Trivial

3. FILE CO-CHANGE NETWORK (from git)
   Files that appear together in many commits are structurally coupled.
   For each ticket: avg coupling degree of its changed files
   (how "central" each file is in the codebase coupling graph)

Then tracks how these signals evolve across engineer careers.

Output: .cache/apache_complexity_network.json  +  console report
"""

import json, re, sys, math
from collections import defaultdict, Counter
from pathlib import Path
from datetime import date as Date
import numpy as np
from pymongo import MongoClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE   = Path(__file__).parent / ".cache"
GIT_IDX = CACHE / "apache_git_index.json"
CMPLX   = CACHE / "apache_task_complexity.json"
OUTPUT  = CACHE / "apache_complexity_network.json"

MIN_TICKETS = 10
KEY_EXTRACT = re.compile(r'author_key\|([a-f0-9\-]+)\|')

PRIORITY_RANK = {"Blocker": 5, "Critical": 4, "Major": 3, "Minor": 2, "Trivial": 1}

CORE_MODULES = {
    "CAMEL": {"camel-core","camel-api","camel-support","camel-base",
              "camel-core-model","camel-core-engine","camel-core-processor"},
    "SPARK": {"core","common"},
    "HADOOP": {"hadoop-common","hadoop-auth","src"},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def uid(field):
    if not isinstance(field, dict): return None
    m = KEY_EXTRACT.search(str(field.get("key", "")))
    return m.group(1) if m else None


def extract_module(filepath, project):
    parts = filepath.replace("\\", "/").split("/")
    if project == "CAMEL":
        return parts[1] if len(parts) >= 2 else parts[0]
    elif project == "SPARK":
        return parts[0]
    elif project == "HADOOP":
        if parts[0].startswith("hadoop-") and len(parts) >= 2:
            return parts[1]
        return parts[0]
    return parts[0]


def is_core(module, project):
    return module in CORE_MODULES.get(project, set())


def days_between(created, resolved):
    try:
        c = Date.fromisoformat(str(created)[:10])
        r = Date.fromisoformat(str(resolved)[:10])
        d = (r - c).days
        return float(d) if 0 <= d <= 3650 else None
    except ValueError:
        return None


def spearman(values):
    n = len(values)
    if n < 5: return float("nan")
    ranks = np.argsort(np.argsort(values)).astype(float) + 1
    d2 = sum((i + 1 - r) ** 2 for i, r in enumerate(ranks))
    denom = n * (n * n - 1)
    return float(1 - 6 * d2 / denom) if denom else float("nan")


# ── Step 1: Build file co-change coupling graph ───────────────────────────────

def build_cochange_graph(git_index):
    """
    Count how often each file appears with every other file in the same commit.
    Returns: file -> co-change degree (sum of co-occurrences with other files)
    """
    print("Building file co-change graph...")
    cochange_degree = Counter()
    pair_counts = Counter()
    total_commits = 0

    for key, rec in git_index.items():
        files = rec.get("files", [])
        if len(files) < 2:
            # Single-file commits still contribute to individual file frequency
            if files:
                cochange_degree[files[0]] += 0
            continue
        total_commits += 1
        for i, fa in enumerate(files):
            for fb in files[i+1:]:
                pair = tuple(sorted([fa, fb]))
                pair_counts[pair] += 1
                cochange_degree[fa] += 1
                cochange_degree[fb] += 1

    print(f"  {total_commits:,} multi-file commits, "
          f"{len(cochange_degree):,} unique files, "
          f"{len(pair_counts):,} co-change pairs")

    # Top coupled file pairs
    print("  Top 10 most co-changed file pairs:")
    for (fa, fb), c in pair_counts.most_common(10):
        fa_short = fa.split("/")[-1]
        fb_short = fb.split("/")[-1]
        print(f"    {c:>4}x  {fa_short}  <->  {fb_short}")

    return cochange_degree, pair_counts


# ── Step 2: Load Jira network signals ────────────────────────────────────────

def load_jira_network(col, git_keys):
    """Load comment count, watch count, links, priority, cross-assign for git-linked tickets."""
    print("\nLoading Jira network signals...")
    network = {}

    cursor = col.find(
        {"key": {"$in": list(git_keys)}, "fields.resolutiondate": {"$ne": None}},
        {"key": 1,
         "fields.assignee": 1, "fields.reporter": 1,
         "fields.comments": 1, "fields.watches": 1, "fields.votes": 1,
         "fields.issuelinks": 1, "fields.priority": 1,
         "fields.created": 1, "fields.resolutiondate": 1,
         "fields.issuetype": 1, "fields.components": 1},
    )

    for doc in cursor:
        key    = doc["key"]
        fields = doc.get("fields", {})

        assignee_uid = uid(fields.get("assignee"))
        reporter_uid = uid(fields.get("reporter"))
        cross_assigned = (
            assignee_uid is not None and reporter_uid is not None
            and assignee_uid != reporter_uid
        )

        comments = fields.get("comments") or []
        comment_count = len(comments)

        # Unique comment authors (network breadth)
        comment_author_uids = set()
        for c in comments:
            au = uid(c.get("author"))
            if au:
                comment_author_uids.add(au)
        comment_authors = len(comment_author_uids)

        watches = (fields.get("watches") or {})
        watch_count = watches.get("watchCount", 0) if isinstance(watches, dict) else 0

        votes_f = (fields.get("votes") or {})
        vote_count = votes_f.get("votes", 0) if isinstance(votes_f, dict) else 0

        links = fields.get("issuelinks") or []
        has_links = len(links) > 0
        link_count = len(links)

        priority_name = ""
        pf = fields.get("priority")
        if isinstance(pf, dict):
            priority_name = pf.get("name", "")
        priority_rank = PRIORITY_RANK.get(priority_name, 0)

        created  = fields.get("created", "") or ""
        resolved = fields.get("resolutiondate", "") or ""
        days     = days_between(created, resolved)

        issuetype_raw = fields.get("issuetype", {})
        issuetype = (issuetype_raw.get("name", "Unknown")
                     if isinstance(issuetype_raw, dict) else "Unknown")

        components = fields.get("components") or []
        comp_count = len(components) if isinstance(components, list) else 0

        project = key.split("-")[0]

        network[key] = {
            "uid":             assignee_uid,
            "project":         project,
            "issuetype":       issuetype,
            "date":            resolved[:10] if resolved else "",
            "days":            days,
            "cross_assigned":  cross_assigned,
            "comment_count":   comment_count,
            "comment_authors": comment_authors,
            "watch_count":     watch_count,
            "vote_count":      vote_count,
            "has_links":       has_links,
            "link_count":      link_count,
            "priority_rank":   priority_rank,
            "priority":        priority_name,
            "comp_count":      comp_count,
        }

    print(f"  {len(network):,} tickets with network data")
    return network


# ── Step 3: Composite complexity score ───────────────────────────────────────

def compute_complexity_scores(git_index, network):
    """Add complexity vector + composite score to each ticket."""
    raw_scores = {}

    for key, net in network.items():
        if key not in git_index:
            continue
        files   = git_index[key].get("files", [])
        project = net["project"]

        n_files = len(files)
        if n_files == 0:
            continue

        modules = [extract_module(f, project) for f in files]
        n_mods  = len(set(modules))
        core_ct = sum(1 for m in modules if is_core(m, project))
        core_pct = core_ct / n_files

        # Raw composite (un-normalised)
        raw = (0.4 * n_mods +
               0.3 * math.log(n_files + 1) +
               0.3 * core_pct * 5)   # scale core_pct to ~same range

        raw_scores[key] = {
            "n_files": n_files,
            "n_mods":  n_mods,
            "core_pct": round(core_pct, 3),
            "raw_complexity": raw,
        }

    # Normalise 0-1
    vals = [v["raw_complexity"] for v in raw_scores.values()]
    lo, hi = min(vals), max(vals)
    for v in raw_scores.values():
        v["complexity"] = round((v["raw_complexity"] - lo) / (hi - lo), 3)

    return raw_scores


# ── Step 4: Merge and enrich ──────────────────────────────────────────────────

def merge(network, complexity, cochange_degree, git_index):
    """Merge all signals into one record per ticket."""
    merged = {}
    for key, net in network.items():
        if key not in complexity:
            continue
        cv  = complexity[key]
        files = git_index[key].get("files", [])
        # Average co-change degree of this ticket's files
        avg_coupling = (
            np.mean([cochange_degree.get(f, 0) for f in files])
            if files else 0.0
        )
        merged[key] = {**net, **cv,
                       "avg_coupling": round(float(avg_coupling), 1)}
    return merged


# ── Step 5: Analysis ──────────────────────────────────────────────────────────

def analyse(merged):
    tickets = list(merged.values())
    print(f"\n{'='*65}")
    print("COMPLEXITY + NETWORK ANALYSIS")
    print(f"{'='*65}")
    print(f"\n{len(tickets):,} tickets with full data")

    # ── Complexity score distribution ─────────────────────────────────────────
    scores = [t["complexity"] for t in tickets]
    arr = np.array(scores)
    print(f"\nCOMPOSITE COMPLEXITY SCORE (0-1):")
    print(f"  mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
          f"p75={np.percentile(arr,75):.3f}  p90={np.percentile(arr,90):.3f}")

    # By issuetype
    print(f"\n  By issuetype:")
    by_type = defaultdict(list)
    for t in tickets:
        by_type[t["issuetype"]].append(t["complexity"])
    print(f"  {'Type':<22} {'n':>5}  {'median':>7}  {'mean':>7}")
    for itype, vals in sorted(by_type.items(), key=lambda x: -len(x[1])):
        if len(vals) < 50: continue
        print(f"  {itype:<22} {len(vals):>5}  {np.median(vals):>7.3f}  "
              f"{np.mean(vals):>7.3f}")

    # ── Network signals by complexity tier ────────────────────────────────────
    p33 = np.percentile(arr, 33)
    p67 = np.percentile(arr, 67)
    low    = [t for t in tickets if t["complexity"] <= p33]
    mid    = [t for t in tickets if p33 < t["complexity"] <= p67]
    high   = [t for t in tickets if t["complexity"] > p67]

    print(f"\nNETWORK SIGNALS BY COMPLEXITY TIER:")
    print(f"  {'Signal':<22} {'Low':>8}  {'Mid':>8}  {'High':>8}")
    print(f"  {'-'*52}")
    for label, fn in [
        ("comment_count",   lambda t: t["comment_count"]),
        ("comment_authors", lambda t: t["comment_authors"]),
        ("watch_count",     lambda t: t["watch_count"]),
        ("vote_count",      lambda t: t["vote_count"]),
        ("link_count",      lambda t: t["link_count"]),
        ("avg_coupling",    lambda t: t["avg_coupling"]),
        ("days",            lambda t: t["days"] or 0),
    ]:
        lo_m = np.median([fn(t) for t in low])
        mi_m = np.median([fn(t) for t in mid])
        hi_m = np.median([fn(t) for t in high])
        print(f"  {label:<22} {lo_m:>8.2f}  {mi_m:>8.2f}  {hi_m:>8.2f}")

    # Cross-assign rate
    for label, grp in [("Low", low), ("Mid", mid), ("High", high)]:
        rate = np.mean([t["cross_assigned"] for t in grp])
        print(f"  cross_assign ({label:4s})       {100*rate:>7.1f}%")

    # ── Priority × complexity ─────────────────────────────────────────────────
    print(f"\nPRIORITY vs MEDIAN COMPLEXITY:")
    by_priority = defaultdict(list)
    for t in tickets:
        if t["priority"]:
            by_priority[t["priority"]].append(t["complexity"])
    for p in ["Blocker","Critical","Major","Minor","Trivial"]:
        vals = by_priority.get(p, [])
        if not vals: continue
        print(f"  {p:<10}: n={len(vals):>5}  median={np.median(vals):.3f}  "
              f"mean={np.mean(vals):.3f}")

    # ── Engineer career arcs ──────────────────────────────────────────────────
    print(f"\nENGINEER CAREER ARCS (complexity + network over time)")
    by_eng = defaultdict(list)
    for key, t in merged.items():
        if t["uid"] and t["date"] and t["days"] is not None:
            by_eng[t["uid"]].append({**t, "key": key})

    qualified = {uid: sorted(tix, key=lambda x: x["date"])
                 for uid, tix in by_eng.items()
                 if len(tix) >= MIN_TICKETS}
    print(f"  {len(qualified):,} engineers with >={MIN_TICKETS} tickets")

    eng_results = []
    for uid, tix in qualified.items():
        n = len(tix)
        half = n // 2
        early, late = tix[:half], tix[half:]

        def med(lst, k): return float(np.median([t[k] for t in lst]))
        def avg(lst, k): return float(np.mean([t[k] for t in lst]))

        # Spearman for key signals over career
        crho  = spearman([t["complexity"]     for t in tix])
        ccrho = spearman([t["comment_count"]  for t in tix])
        wrho  = spearman([t["watch_count"]    for t in tix])
        vrho  = spearman([t["days"] or 0      for t in tix])

        project = Counter(t["project"] for t in tix).most_common(1)[0][0]

        eng_results.append({
            "uid": uid, "n": n, "project": project,
            "early_complexity": round(med(early, "complexity"), 3),
            "late_complexity":  round(med(late,  "complexity"), 3),
            "early_comments":   round(med(early, "comment_count"), 2),
            "late_comments":    round(med(late,  "comment_count"), 2),
            "early_watches":    round(med(early, "watch_count"), 2),
            "late_watches":     round(med(late,  "watch_count"), 2),
            "early_days":       round(med(early, "days"), 1) if all(t["days"] for t in early) else None,
            "late_days":        round(med(late,  "days"), 1) if all(t["days"] for t in late)  else None,
            "complexity_rho":   round(crho,  3) if not np.isnan(crho)  else None,
            "comment_rho":      round(ccrho, 3) if not np.isnan(ccrho) else None,
            "watch_rho":        round(wrho,  3) if not np.isnan(wrho)  else None,
            "velocity_rho":     round(vrho,  3) if not np.isnan(vrho)  else None,
        })

    # Aggregate
    cx_delta = np.array([r["late_complexity"] - r["early_complexity"]
                         for r in eng_results])
    cm_delta = np.array([r["late_comments"]   - r["early_comments"]
                         for r in eng_results])
    wt_delta = np.array([r["late_watches"]    - r["early_watches"]
                         for r in eng_results])

    print(f"\n  Career arc delta (late half median - early half median):")
    print(f"  {'Signal':<22} {'mean':>7}  {'median':>7}  "
          f"{'rising':>8}  {'falling':>8}")
    for label, arr2, th in [
        ("complexity",     cx_delta, 0.02),
        ("comment_count",  cm_delta, 0.3),
        ("watch_count",    wt_delta, 0.5),
    ]:
        rising  = (arr2 > th).sum()
        falling = (arr2 < -th).sum()
        print(f"  {label:<22} {arr2.mean():>+7.3f}  {np.median(arr2):>+7.3f}  "
              f"{rising:>4} ({100*rising/len(arr2):.0f}%)  "
              f"{falling:>4} ({100*falling/len(arr2):.0f}%)")

    # Engineers whose complexity grows fastest
    rising_cx = sorted([r for r in eng_results
                        if r["complexity_rho"] is not None and r["complexity_rho"] > 0.2],
                       key=lambda r: -r["complexity_rho"])
    print(f"\n  Engineers with rising complexity (rho > 0.2): {len(rising_cx)}")
    print(f"  {'uid[:8]':<10} {'n':>4}  {'cx_rho':>7}  {'v_rho':>7}  "
          f"{'early_cx':>8}  {'late_cx':>8}  {'proj'}")
    for r in rising_cx[:12]:
        vr = f"{r['velocity_rho']:+.3f}" if r["velocity_rho"] is not None else "   n/a"
        print(f"  {r['uid'][:8]:<10} {r['n']:>4}  {r['complexity_rho']:>+7.3f}  "
              f"{vr:>7}  {r['early_complexity']:>8.3f}  "
              f"{r['late_complexity']:>8.3f}  {r['project']}")

    # Cross-tab: complexity rising + velocity falling (taking harder work, getting slower)
    both = [r for r in eng_results
            if (r["complexity_rho"] is not None and r["complexity_rho"] > 0.2
                and r["velocity_rho"] is not None and r["velocity_rho"] > 0.2)]
    fast_complex = [r for r in eng_results
                    if (r["complexity_rho"] is not None and r["complexity_rho"] > 0.2
                        and r["velocity_rho"] is not None and r["velocity_rho"] < -0.2)]
    print(f"\n  Rising complexity + getting SLOWER (expected growth path): {len(both)}")
    print(f"  Rising complexity + getting FASTER (mastery signal):        {len(fast_complex)}")

    return eng_results


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading git index...")
    with open(GIT_IDX, encoding="utf-8") as f:
        git_index = json.load(f)
    git_keys = set(git_index.keys())
    print(f"  {len(git_keys):,} keys")

    client = MongoClient("mongodb://localhost:27017/")
    col = client["jiradump"]["Apache"]

    cochange_degree, pair_counts = build_cochange_graph(git_index)
    network    = load_jira_network(col, git_keys)
    complexity = compute_complexity_scores(git_index, network)
    merged     = merge(network, complexity, cochange_degree, git_index)
    eng_results = analyse(merged)

    # Save
    CACHE.mkdir(exist_ok=True)
    output = {
        "n_tickets": len(merged),
        "engineer_arcs": eng_results,
        "top_coupled_files": [
            {"fa": fa, "fb": fb, "count": c}
            for (fa, fb), c in pair_counts.most_common(100)
        ],
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
