# -*- coding: utf-8 -*-
"""
apache_task_complexity.py
=========================
Compute a three-dimensional complexity vector for each Apache ticket
that has a git commit:

  n_files      = raw file count (proxy for blast radius)
  n_modules    = distinct modules touched (proxy for coupling)
  core_pct     = fraction of files in core modules (proxy for criticality)

Then for engineers with >=10 git-linked tickets, track how the complexity
vector evolves over their career (early half vs late half).

Also cross-references with velocity to answer:
  "Are engineers who handle more complex tasks getting faster or slower?"

Output: .cache/apache_task_complexity.json  +  console report

Usage:
    python apache_task_complexity.py
"""

import json, re, sys
from collections import defaultdict, Counter
from pathlib import Path
from datetime import date as Date
import numpy as np
from pymongo import MongoClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE   = Path(__file__).parent / ".cache"
GIT_IDX = CACHE / "apache_git_index.json"
OUTPUT  = CACHE / "apache_task_complexity.json"

MIN_TICKETS = 10

KEY_EXTRACT = re.compile(r'author_key\|([a-f0-9\-]+)\|')


# ── Module classification ─────────────────────────────────────────────────────

def extract_module(filepath: str, project: str) -> str:
    """Return a canonical module name from a file path."""
    parts = filepath.replace("\\", "/").split("/")
    if not parts:
        return "unknown"

    if project == "CAMEL":
        # paths: core/camel-api/..., components/camel-aws/..., dsl/camel-yaml-dsl/...
        # module = second component (e.g. 'camel-api', 'camel-aws')
        if len(parts) >= 2:
            return parts[1]
        return parts[0]

    elif project == "SPARK":
        # paths: sql/catalyst/..., core/src/..., python/pyspark/..., mllib/src/...
        # module = first component
        return parts[0]

    elif project == "HADOOP":
        # paths: hadoop-common-project/hadoop-common/..., hadoop-tools/hadoop-aws/...
        #         src/java/..., src/core/...  (old structure)
        if parts[0].startswith("hadoop-") and len(parts) >= 2:
            return parts[1]   # e.g. 'hadoop-common', 'hadoop-aws'
        return parts[0]       # 'src' for old structure

    return parts[0]


# Core modules per project — these are the foundational modules everything
# else depends on. Changes here have highest blast radius / strategic weight.
CORE_MODULES = {
    "CAMEL": {
        "camel-core", "camel-api", "camel-support", "camel-base",
        "camel-core-model", "camel-core-engine", "camel-core-processor",
        "camel-core-languages",
    },
    "SPARK": {
        "core",       # spark-core (RDD, SparkContext, etc.)
        "common",     # shared utilities
        "sql",        # SQL/DataFrame — but sql has sub-modules so we flag at top level
                      # refined below
    },
    "HADOOP": {
        "hadoop-common",      # filesystem APIs, security, config
        "hadoop-auth",        # authentication
        "src",                # old monolithic src structure
    },
}

# Spark sub-module refinement: catalyst is deeper core than, say, streaming
SPARK_CORE_MODULES = {"core", "common", "common/src"}
SPARK_NEAR_CORE    = {"sql"}   # important but domain-specific


def is_core(module: str, project: str) -> bool:
    if project == "SPARK":
        return module in SPARK_CORE_MODULES
    return module in CORE_MODULES.get(project, set())


# ── Complexity vector for one ticket ─────────────────────────────────────────

def complexity_vector(files: list, project: str) -> dict:
    """Compute (n_files, n_modules, core_pct) for a list of changed files."""
    if not files:
        return {"n_files": 0, "n_modules": 0, "core_pct": 0.0, "modules": []}

    modules = [extract_module(f, project) for f in files]
    unique_modules = set(modules)
    core_count = sum(1 for m in modules if is_core(m, project))

    return {
        "n_files":   len(files),
        "n_modules": len(unique_modules),
        "core_pct":  core_count / len(files),
        "modules":   sorted(unique_modules),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def assignee_id(field) -> str | None:
    if not isinstance(field, dict):
        return None
    m = KEY_EXTRACT.search(str(field.get("key", "")))
    return m.group(1) if m else None


def days_between(created: str, resolved: str) -> float | None:
    try:
        c = Date.fromisoformat(created[:10])
        r = Date.fromisoformat(resolved[:10])
        d = (r - c).days
        return float(d) if 0 <= d <= 3650 else None
    except ValueError:
        return None


def spearman(values: list) -> float:
    n = len(values)
    if n < 5:
        return float("nan")
    ranks = np.argsort(np.argsort(values)).astype(float) + 1
    d2 = sum((i + 1 - r) ** 2 for i, r in enumerate(ranks))
    denom = n * (n * n - 1)
    return float(1 - 6 * d2 / denom) if denom else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading git index...")
    with open(GIT_IDX, encoding="utf-8") as f:
        git_index: dict = json.load(f)
    print(f"  {len(git_index):,} tickets")

    client = MongoClient("mongodb://localhost:27017/")
    col = client["jiradump"]["Apache"]

    # ── Build per-ticket complexity ───────────────────────────────────────────
    print("Computing complexity vectors...")
    ticket_data = {}   # key -> full record

    cursor = col.find(
        {"key": {"$in": list(git_index.keys())},
         "fields.resolutiondate": {"$ne": None}},
        {"key": 1, "fields.assignee": 1, "fields.created": 1,
         "fields.resolutiondate": 1, "fields.issuetype": 1},
    )

    for doc in cursor:
        key    = doc["key"]
        fields = doc.get("fields", {})
        uid    = assignee_id(fields.get("assignee"))
        if not uid:
            continue
        created  = fields.get("created", "") or ""
        resolved = fields.get("resolutiondate", "") or ""
        days = days_between(created, resolved)
        if days is None:
            continue
        issuetype_raw = fields.get("issuetype", {})
        issuetype = (issuetype_raw.get("name", "Unknown")
                     if isinstance(issuetype_raw, dict) else "Unknown")
        project  = key.split("-")[0]
        git_rec  = git_index[key]
        files    = git_rec.get("files", [])
        cv       = complexity_vector(files, project)
        if cv["n_files"] == 0:
            continue

        ticket_data[key] = {
            "uid":       uid,
            "project":   project,
            "issuetype": issuetype,
            "date":      resolved[:10],
            "days":      days,
            **cv,
        }

    print(f"  {len(ticket_data):,} tickets with full data")

    # ── Overall distributions ─────────────────────────────────────────────────
    print("\n=== OVERALL COMPLEXITY DISTRIBUTIONS ===")
    for project in ["CAMEL", "SPARK", "HADOOP"]:
        tickets = [v for v in ticket_data.values() if v["project"] == project]
        if not tickets:
            continue
        nf = [t["n_files"]   for t in tickets]
        nm = [t["n_modules"] for t in tickets]
        cp = [t["core_pct"]  for t in tickets]
        print(f"\n{project}  (n={len(tickets):,})")
        print(f"  n_files   : median={np.median(nf):.0f}  mean={np.mean(nf):.1f}  "
              f"p75={np.percentile(nf,75):.0f}  p90={np.percentile(nf,90):.0f}")
        print(f"  n_modules : median={np.median(nm):.0f}  mean={np.mean(nm):.1f}  "
              f"p75={np.percentile(nm,75):.0f}  p90={np.percentile(nm,90):.0f}")
        print(f"  core_pct  : median={np.median(cp):.2f}  mean={np.mean(cp):.2f}  "
              f"% tickets with any core file: {100*np.mean([c>0 for c in cp]):.0f}%")

        # Most common modules
        all_modules = []
        for t in tickets:
            all_modules.extend(t["modules"])
        top_modules = Counter(all_modules).most_common(8)
        print(f"  Top modules: {', '.join(f'{m}({c})' for m,c in top_modules)}")

    # ── Complexity by issuetype ───────────────────────────────────────────────
    print("\n=== COMPLEXITY BY ISSUETYPE (all projects) ===")
    by_type = defaultdict(list)
    for t in ticket_data.values():
        by_type[t["issuetype"]].append(t)
    print(f"  {'Issuetype':<22} {'n':>5}  {'med_files':>9}  {'med_modules':>11}  "
          f"{'med_days':>8}  {'core%':>5}")
    for itype, tix in sorted(by_type.items(), key=lambda x: -len(x[1])):
        if len(tix) < 50:
            continue
        nf = np.median([t["n_files"]   for t in tix])
        nm = np.median([t["n_modules"] for t in tix])
        nd = np.median([t["days"]      for t in tix])
        cp = np.mean(  [t["core_pct"]  for t in tix])
        print(f"  {itype:<22} {len(tix):>5}  {nf:>9.0f}  {nm:>11.0f}  "
              f"{nd:>8.0f}  {100*cp:>4.0f}%")

    # ── Per-engineer career complexity arc ────────────────────────────────────
    print("\n=== ENGINEER CAREER COMPLEXITY ARC ===")
    by_eng = defaultdict(list)
    for key, t in ticket_data.items():
        by_eng[t["uid"]].append({**t, "key": key})

    qualified = {uid: sorted(tix, key=lambda x: x["date"])
                 for uid, tix in by_eng.items()
                 if len(tix) >= MIN_TICKETS}
    print(f"  {len(qualified):,} engineers with >={MIN_TICKETS} git-linked tickets")

    eng_results = []
    for uid, tix in qualified.items():
        n = len(tix)
        half = n // 2
        early, late = tix[:half], tix[half:]

        def avg(lst, key): return float(np.mean([t[key] for t in lst]))

        early_files   = avg(early, "n_files")
        late_files    = avg(late,  "n_files")
        early_modules = avg(early, "n_modules")
        late_modules  = avg(late,  "n_modules")
        early_core    = avg(early, "core_pct")
        late_core     = avg(late,  "core_pct")
        early_days    = avg(early, "days")
        late_days     = avg(late,  "days")

        vrho = spearman([t["days"]     for t in tix])
        frho = spearman([t["n_files"]  for t in tix])
        mrho = spearman([t["n_modules"]for t in tix])
        crho = spearman([t["core_pct"] for t in tix])

        project = Counter(t["project"] for t in tix).most_common(1)[0][0]

        eng_results.append({
            "uid": uid, "n": n, "project": project,
            "early_files": round(early_files, 2),
            "late_files":  round(late_files,  2),
            "early_modules": round(early_modules, 2),
            "late_modules":  round(late_modules,  2),
            "early_core": round(early_core, 3),
            "late_core":  round(late_core,  3),
            "early_days": round(early_days, 1),
            "late_days":  round(late_days,  1),
            "vrho":  round(vrho, 3) if not np.isnan(vrho) else None,
            "frho":  round(frho, 3) if not np.isnan(frho) else None,
            "mrho":  round(mrho, 3) if not np.isnan(mrho) else None,
            "crho":  round(crho, 3) if not np.isnan(crho) else None,
        })

    # Aggregate career arc
    files_delta   = np.array([r["late_files"]   - r["early_files"]   for r in eng_results])
    modules_delta = np.array([r["late_modules"] - r["early_modules"] for r in eng_results])
    core_delta    = np.array([r["late_core"]    - r["early_core"]    for r in eng_results])
    days_delta    = np.array([r["late_days"]    - r["early_days"]    for r in eng_results])

    print(f"\n  Career arc deltas (late - early half average):")
    print(f"  {'Dimension':<16} {'mean':>7}  {'median':>7}  "
          f"{'growing (>+0.5)':>15}  {'shrinking (<-0.5)':>17}")
    for label, arr, thresh in [
        ("n_files",    files_delta,   0.5),
        ("n_modules",  modules_delta, 0.3),
        ("core_pct",   core_delta,    0.05),
        ("days",       days_delta,    2.0),
    ]:
        g = (arr > thresh).sum()
        s = (arr < -thresh).sum()
        print(f"  {label:<16} {arr.mean():>+7.3f}  {np.median(arr):>+7.3f}  "
              f"{g:>8} ({100*g/len(arr):.0f}%)  {s:>8} ({100*s/len(arr):.0f}%)")

    # ── Complexity vs velocity correlation ────────────────────────────────────
    print(f"\n=== COMPLEXITY vs VELOCITY ===")
    valid = [r for r in eng_results
             if r["vrho"] is not None and r["frho"] is not None]
    vrhos = np.array([r["vrho"] for r in valid])
    frhos = np.array([r["frho"] for r in valid])
    mrhos = np.array([r["mrho"] for r in valid if r["mrho"] is not None])

    corr_vf = np.corrcoef(vrhos, frhos)[0,1]
    print(f"  Pearson r(velocity_rho, scope_rho) = {corr_vf:.3f}")
    print(f"  (negative = getting faster correlates with broader scope)")

    # Engineers getting faster (vrho < -0.2): what's their median complexity?
    fast_eng  = [r for r in valid if r["vrho"] < -0.2]
    slow_eng  = [r for r in valid if r["vrho"] >  0.2]
    mixed_eng = [r for r in valid if -0.2 <= r["vrho"] <= 0.2]

    print(f"\n  Median complexity by velocity group:")
    print(f"  {'Group':<20} {'n':>4}  {'med_files':>9}  {'med_mods':>8}  "
          f"{'med_core%':>9}  {'med_days':>8}")
    for label, grp in [
        ("Getting faster",  fast_eng),
        ("Mixed",           mixed_eng),
        ("Getting slower",  slow_eng),
    ]:
        if not grp:
            continue
        mf = np.median([r["early_files"] for r in grp])
        mm = np.median([r["early_modules"] for r in grp])
        mc = np.median([r["early_core"] for r in grp])
        md = np.median([r["early_days"] for r in grp])
        print(f"  {label:<20} {len(grp):>4}  {mf:>9.1f}  {mm:>8.1f}  "
              f"{100*mc:>8.0f}%  {md:>8.0f}")

    # ── Top complex tickets (high n_modules) ──────────────────────────────────
    print(f"\n=== MOST COMPLEX TICKETS (by n_modules) ===")
    top_complex = sorted(ticket_data.values(),
                         key=lambda t: (t["n_modules"], t["n_files"]),
                         reverse=True)[:15]
    print(f"  {'key':<14} {'files':>5}  {'mods':>4}  {'core%':>5}  "
          f"{'days':>5}  {'type':<18}  modules (first 4)")
    for t in top_complex:
        mods_str = ", ".join(t["modules"][:4])
        if len(t["modules"]) > 4:
            mods_str += f" +{len(t['modules'])-4}"
        print(f"  {t.get('key','?'):<14}" if "key" in t else "", end="")

    # Re-do with keys
    top_complex2 = sorted(
        [(k,v) for k,v in ticket_data.items()],
        key=lambda x: (x[1]["n_modules"], x[1]["n_files"]),
        reverse=True
    )[:15]
    print(f"\n  {'key':<14} {'files':>5}  {'mods':>4}  {'core%':>5}  "
          f"{'days':>5}  {'type':<18}  modules (first 4)")
    for key, t in top_complex2:
        mods_str = ", ".join(t["modules"][:4])
        if len(t["modules"]) > 4:
            mods_str += f" +{len(t['modules'])-4}"
        print(f"  {key:<14} {t['n_files']:>5}  {t['n_modules']:>4}  "
              f"{100*t['core_pct']:>4.0f}%  {t['days']:>5.0f}  "
              f"{t['issuetype']:<18}  {mods_str}")

    # ── Save ──────────────────────────────────────────────────────────────────
    CACHE.mkdir(exist_ok=True)
    output = {
        "ticket_complexity": {
            k: {fld: v[fld] for fld in
                ["project","issuetype","date","days","n_files","n_modules","core_pct"]}
            for k, v in ticket_data.items()
        },
        "engineer_arcs": eng_results,
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f)
    print(f"\nSaved: {OUTPUT}")
    print(f"  {len(ticket_data):,} tickets  |  {len(eng_results):,} engineer arcs")


if __name__ == "__main__":
    main()
