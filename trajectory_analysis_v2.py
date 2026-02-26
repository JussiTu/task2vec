"""
trajectory_analysis_v2.py
─────────────────────────
Expanded trajectory analysis on Spring Framework contributors.

Key improvements over v1:
  - Lower ticket threshold (>=10) gives 3x more engineers
  - Separates "joined as junior" from "joined as expert" (avoids the
    Year-1 confound where senior engineers join mid-career)
  - Counts engineers showing clear upward Escalate progression
  - Computes "time to 50% Escalate" as a proxy for expertise development
  - Looks at cluster-level detail: which ticket types appear early vs late
    in a growing engineer's career?
"""

import json
from collections import defaultdict
from pathlib import Path

CACHE = Path('.cache')

CLUSTER_TIER = {
    12: 'Automate', 25: 'Automate', 13: 'Automate',
     2: 'Automate', 21: 'Automate', 17: 'Automate', 22: 'Automate',
    23: 'Assist',    3: 'Assist',   16: 'Assist',   24: 'Assist',
     5: 'Assist',   11: 'Assist',   14: 'Assist',   18: 'Assist',
    26: 'Assist',   30: 'Assist',
     4: 'Escalate',  7: 'Escalate',  6: 'Escalate', 15: 'Escalate',
     0: 'Escalate',  1: 'Escalate',  8: 'Escalate',  9: 'Escalate',
    10: 'Escalate', 19: 'Escalate', 20: 'Escalate', 27: 'Escalate',
    28: 'Escalate', 29: 'Escalate', 31: 'Escalate',
}

CLUSTER_LABEL = {
    12: 'Documentation Corrections',    25: 'Dependency Upgrades',
    13: 'Deprecation Updates',           2: 'Release Version Updates',
    21: 'Codebase Consistency',         17: 'Acceptance Test Failures',
    22: 'Build Improvements',           23: 'RequestMapping Enhancements',
     3: 'Content Negotiation',          16: 'Transaction Management',
    24: 'MongoDB Mapping',               5: 'Spring IDE Tooling',
    11: 'Messaging Configuration',      14: 'Redis Improvements',
    18: 'Bean Configuration',           26: 'Spring Web Flow',
    30: 'MongoDB Features',              4: 'JDBC Compatibility',
     7: 'JMS Message Listener',          6: 'Authentication Framework',
    15: 'JPA Entity Management',         0: 'Validation & Data Binding',
     1: 'Spring Roo Errors',             8: 'JPA & Entity Mapping',
     9: 'Autowiring & Bean Config',     10: 'Workspace & Class Loading',
    19: 'Step Scope & Tasklet',         20: 'Spring XD Module Dev',
    27: 'QueryDSL & Projections',       28: 'JSF & Spring Web Flow',
    29: 'Session Authentication',       31: 'OSGi Manifest & Deps',
}

TIERS = ['Automate', 'Assist', 'Escalate']

def tier_pcts(counts):
    total = sum(counts.get(t, 0) for t in TIERS)
    if total == 0: return None
    return {t: counts.get(t, 0) / total * 100 for t in TIERS}

def bar(pct, width=15):
    filled = round(pct / 100 * width)
    return '#' * filled + '.' * (width - filled)

# ── Load ────────────────────────────────────────────────────────────────────
print("Loading data...")
with open(CACHE / 'search_meta.json', encoding='utf-8', errors='replace') as f:
    meta = json.load(f)

tickets = []
for m in meta:
    if not m.get('assignee') or not m.get('year'): continue
    tier = CLUSTER_TIER.get(m.get('cluster'))
    if tier is None: continue
    tickets.append({
        'key': m['key'], 'assignee': m['assignee'],
        'year': m['year'], 'tier': tier, 'cluster': m['cluster'],
    })

print(f"  Usable tickets: {len(tickets):,}")

# ── Build career data ────────────────────────────────────────────────────────
career  = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
cluster_by_year = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
first_year = {}

for t in tickets:
    a, yr, tier, cl = t['assignee'], t['year'], t['tier'], t['cluster']
    career[a][yr][tier] += 1
    cluster_by_year[a][yr][cl] += 1
    if a not in first_year or yr < first_year[a]:
        first_year[a] = yr

totals = {a: sum(sum(y.values()) for y in yrs.values()) for a, yrs in career.items()}

# ── Filter: >=10 tickets, multi-year careers ────────────────────────────────
MIN_TICKETS = 10
MIN_YEARS   = 2

qualified = {}
for a, n in totals.items():
    if n < MIN_TICKETS: continue
    years = sorted(career[a].keys())
    if len(years) < MIN_YEARS: continue
    qualified[a] = n

print(f"  Engineers (>={MIN_TICKETS} tickets, >={MIN_YEARS} yr career): {len(qualified):,}")

# ── Classify each engineer: junior-start vs expert-start ────────────────────
# "Junior start"  = first-year Escalate% < 40%  AND career span >= 3 years
# "Expert start"  = first-year Escalate% >= 60%
# "Unclear"       = everything else

juniors  = []  # (assignee, first_esc%, last_esc%, span, total_n)
experts  = []
unclear  = []

for a in qualified:
    fy    = first_year[a]
    years = sorted(career[a].keys())
    span  = years[-1] - years[0] + 1

    # First-year tier mix
    p1 = tier_pcts(career[a][fy])
    if not p1: continue

    # Last 2 years tier mix
    late_counts = defaultdict(int)
    for yr in years[-2:]:
        for tier, n in career[a][yr].items():
            late_counts[tier] += n
    pl = tier_pcts(late_counts)
    if not pl: continue

    row = (a, p1['Escalate'], pl['Escalate'], span, totals[a])

    if p1['Escalate'] < 40 and span >= 3:
        juniors.append(row)
    elif p1['Escalate'] >= 60:
        experts.append(row)
    else:
        unclear.append(row)

print(f"\n  Junior starters (<40% Esc yr1, >=3yr span): {len(juniors)}")
print(f"  Expert starters (>=60% Esc yr1):             {len(experts)}")
print(f"  Unclear:                                      {len(unclear)}")

# ── Analysis 1: Do junior starters grow into escalate work? ─────────────────
print("\n" + "=" * 72)
print("  ANALYSIS 1 — JUNIOR STARTERS: does Escalate% grow over career?")
print("=" * 72)

# Bucket junior starters by career span
buckets = [(3,5,'3-5 yr'), (6,9,'6-9 yr'), (10,99,'10+ yr')]
print(f"\n  {'Career':8}  {'n':>4}  {'Avg start Esc%':>14}  {'Avg end Esc%':>12}  {'Change':>7}")
print("  " + "-" * 55)
for lo, hi, label in buckets:
    group = [r for r in juniors if lo <= r[3] <= hi]
    if not group: continue
    avg_start = sum(r[1] for r in group) / len(group)
    avg_end   = sum(r[2] for r in group) / len(group)
    delta = avg_end - avg_start
    marker = '  <-- growth' if delta > 10 else ''
    print(f"  {label:8}  {len(group):>4}  {avg_start:>13.1f}%  {avg_end:>11.1f}%  {delta:>+6.1f}pp{marker}")

# How many junior starters show clear upward progression?
grew     = [r for r in juniors if r[2] - r[1] > 15]  # >15pp increase
stayed   = [r for r in juniors if abs(r[2] - r[1]) <= 15]
declined = [r for r in juniors if r[2] - r[1] < -15]

print(f"\n  Of {len(juniors)} junior starters:")
print(f"    Grew into harder work (Esc +>15pp):  {len(grew):>3}  ({100*len(grew)/len(juniors):.0f}%)")
print(f"    Stayed similar (+/-15pp):            {stayed.__len__():>3}  ({100*len(stayed)/len(juniors):.0f}%)")
print(f"    Moved to easier work (Esc <-15pp):   {len(declined):>3}  ({100*len(declined)/len(juniors):.0f}%)")

# ── Analysis 2: Time to reach 50% Escalate ──────────────────────────────────
print("\n" + "=" * 72)
print("  ANALYSIS 2 — TIME TO EXPERTISE")
print("  For junior starters who crossed 50% Escalate: how long did it take?")
print("=" * 72)

time_to_50 = []
for a, first_esc, last_esc, span, n in juniors:
    if last_esc < 50: continue   # never reached 50%
    fy    = first_year[a]
    years = sorted(career[a].keys())
    cumulative = defaultdict(int)
    for yr in years:
        for tier, cnt in career[a][yr].items():
            cumulative[tier] += cnt
        p = tier_pcts(cumulative)
        if p and p['Escalate'] >= 50:
            yrs_taken = yr - fy + 1
            time_to_50.append((a, yrs_taken, n, first_esc, p['Escalate']))
            break

if time_to_50:
    avg_time = sum(r[1] for r in time_to_50) / len(time_to_50)
    print(f"\n  Engineers who reached cumulative 50% Escalate: {len(time_to_50)}")
    print(f"  Average years to reach 50% Escalate: {avg_time:.1f} years")
    print(f"\n  Distribution:")
    for lo, hi, label in [(1,2,'1-2 yr'),(3,4,'3-4 yr'),(5,6,'5-6 yr'),(7,10,'7-10 yr'),(11,99,'11+ yr')]:
        group = [r for r in time_to_50 if lo <= r[1] <= hi]
        if group:
            pct = 100 * len(group) / len(time_to_50)
            print(f"    {label}: {len(group):>3} engineers  {bar(pct, 10)}")

# ── Analysis 3: What ticket types appear early vs late? ─────────────────────
print("\n" + "=" * 72)
print("  ANALYSIS 3 — EARLY vs LATE TICKET TYPES for growing engineers")
print("  (junior starters who grew by >15pp Escalate)")
print("=" * 72)

early_cluster = defaultdict(int)
late_cluster  = defaultdict(int)

for a, first_esc, last_esc, span, n in grew:
    fy    = first_year[a]
    years = sorted(career[a].keys())
    for yr in years:
        career_yr = yr - fy + 1
        for cl, cnt in cluster_by_year[a][yr].items():
            if career_yr <= 2:
                early_cluster[cl] += cnt
            elif career_yr >= max(span - 1, 3):
                late_cluster[cl] += cnt

early_total = sum(early_cluster.values())
late_total  = sum(late_cluster.values())

if early_total and late_total:
    print(f"\n  Top cluster types in EARLY career (yr 1-2) of growing engineers:")
    early_sorted = sorted(early_cluster.items(), key=lambda x: x[1], reverse=True)[:8]
    for cl, n in early_sorted:
        label = CLUSTER_LABEL.get(cl, f'Cluster {cl}')
        tier  = CLUSTER_TIER.get(cl, '?')
        pct   = 100 * n / early_total
        print(f"    {tier:10s}  {pct:4.1f}%  {label}")

    print(f"\n  Top cluster types in LATE career (last 2 yr) of growing engineers:")
    late_sorted = sorted(late_cluster.items(), key=lambda x: x[1], reverse=True)[:8]
    for cl, n in late_sorted:
        label = CLUSTER_LABEL.get(cl, f'Cluster {cl}')
        tier  = CLUSTER_TIER.get(cl, '?')
        pct   = 100 * n / late_total
        print(f"    {tier:10s}  {pct:4.1f}%  {label}")

# ── Analysis 4: Expert starters — consistent deep specialists? ──────────────
print("\n" + "=" * 72)
print("  ANALYSIS 4 — EXPERT STARTERS: do they stay Escalate-heavy?")
print("=" * 72)

stayed_esc = [r for r in experts if r[2] >= 50]
dropped    = [r for r in experts if r[2] < 50]
print(f"\n  Of {len(experts)} expert starters (>=60% Esc yr1):")
print(f"    Stayed Escalate-heavy (late >=50%): {len(stayed_esc):>3}  ({100*len(stayed_esc)/len(experts):.0f}%)")
print(f"    Moved to mixed/easier work:         {len(dropped):>3}  ({100*len(dropped)/len(experts):.0f}%)")
if experts:
    avg_start = sum(r[1] for r in experts) / len(experts)
    avg_end   = sum(r[2] for r in experts) / len(experts)
    print(f"    Avg Esc%: start={avg_start:.0f}%  end={avg_end:.0f}%  change={avg_end-avg_start:+.0f}pp")

# ── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY — KEY NUMBERS FOR THE POST")
print("=" * 72)
print()
print(f"  Total engineers analysed:           {len(qualified):,}")
print(f"  Identified as junior starters:      {len(juniors)}")
print(f"  Identified as expert starters:      {len(experts)}")
print()
print(f"  Junior starters who grew (>15pp):   {len(grew)}  ({100*len(grew)/len(juniors):.0f}%)")
print(f"  Junior starters who stayed flat:    {len(stayed)}")
print(f"  Junior starters who moved easier:   {len(declined)}")
if time_to_50:
    print(f"  Avg years to reach 50% Escalate:   {avg_time:.1f}")
print()

# ── Career arc for all junior starters (aggregate by career year) ────────────
print("=" * 72)
print("  AGGREGATE CAREER ARC — junior starters by relative career year")
print("  (n per year band shown; small years excluded)")
print("=" * 72)

arc = defaultdict(lambda: defaultdict(int))
for a, first_esc, last_esc, span, n in juniors:
    fy = first_year[a]
    for yr, yc in career[a].items():
        cyr = yr - fy + 1
        for tier, cnt in yc.items():
            arc[cyr][tier] += cnt

print(f"\n  {'CY':>4}  {'n':>5}  {'Auto%':>6}  {'Assist%':>7}  {'Esc%':>6}  Escalate bar")
print("  " + "-" * 60)
for cyr in sorted(arc.keys()):
    p = tier_pcts(arc[cyr])
    n = sum(arc[cyr].values())
    if p and n >= 20:
        print(f"  {cyr:>4}  {n:>5,}  {p['Automate']:>5.1f}%  {p['Assist']:>6.1f}%  {p['Escalate']:>5.1f}%  {bar(p['Escalate'])}")
print()
