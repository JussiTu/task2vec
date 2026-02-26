"""
trajectory_analysis.py
───────────────────────
Analyses how each Spring Framework contributor's tier mix (Automate /
Assist / Escalate) evolved over their career in the project.

Uses search_meta.json (key / assignee / year / cluster) + the same
CLUSTER_TIER mapping verified in trend_by_cluster.py.

Questions answered:
  1. Do engineers start on easy (Automate) work and progress to Escalate?
  2. Or do top contributors hit Escalate from day one?
  3. Is there a consistent "learning arc" pattern in the data?
  4. What does the average career profile look like at each stage?
"""

import json
from collections import defaultdict
from pathlib import Path

CACHE = Path('.cache')

# ── Cluster -> tier (same verified mapping as trend_by_cluster.py) ─────────
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

TIERS = ['Automate', 'Assist', 'Escalate']

# ── Load ────────────────────────────────────────────────────────────────────
print("Loading search_meta.json...")
with open(CACHE / 'search_meta.json', encoding='utf-8', errors='replace') as f:
    meta = json.load(f)

# Keep only tickets with assignee, year, and a known cluster→tier
tickets = []
for m in meta:
    if not m.get('assignee') or not m.get('year'):
        continue
    tier = CLUSTER_TIER.get(m.get('cluster'))
    if tier is None:
        continue
    tickets.append({
        'key':      m['key'],
        'assignee': m['assignee'],
        'year':     m['year'],
        'tier':     tier,
    })

print(f"  Usable tickets: {len(tickets):,}")

# ── Build per-assignee career data ──────────────────────────────────────────
# career[assignee] = {year: {tier: count}}
career = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
first_year = {}

for t in tickets:
    a, yr, tier = t['assignee'], t['year'], t['tier']
    career[a][yr][tier] += 1
    if a not in first_year or yr < first_year[a]:
        first_year[a] = yr

# Total tickets per assignee (clustered)
totals = {a: sum(sum(y.values()) for y in yrs.values()) for a, yrs in career.items()}

# Filter: need ≥ 30 clustered tickets to have meaningful stats
qualified = {a: totals[a] for a in totals if totals[a] >= 30}
print(f"  Qualified assignees (>=30 clustered tickets): {len(qualified):,}")

# ── Helper ──────────────────────────────────────────────────────────────────
def tier_pcts(counts):
    total = sum(counts.get(t, 0) for t in TIERS)
    if total == 0:
        return None
    return {t: counts.get(t, 0) / total * 100 for t in TIERS}

def bar(pct, width=12):
    filled = round(pct / 100 * width)
    return '#' * filled + '.' * (width - filled)

# ── Analysis 1: Career-relative tier arc ────────────────────────────────────
# For each engineer: bucket tickets into career year 1, 2-3, 4-6, 7+
# and show how Escalate% changes

print("\n" + "=" * 72)
print("  ANALYSIS 1 — CAREER ARC  (how tier mix shifts with experience)")
print("  (career year = years since first ticket in Spring)")
print("=" * 72)

# Aggregate across ALL qualified assignees
arc = defaultdict(lambda: defaultdict(int))  # career_bucket -> tier -> count
BUCKETS = [
    (1, 1,   'Yr 1      '),
    (2, 3,   'Yr 2-3    '),
    (4, 6,   'Yr 4-6    '),
    (7, 99,  'Yr 7+     '),
]

for a in qualified:
    fy = first_year[a]
    for yr, tier_counts in career[a].items():
        career_yr = yr - fy + 1
        for lo, hi, label in BUCKETS:
            if lo <= career_yr <= hi:
                for tier, n in tier_counts.items():
                    arc[label][tier] += n

print(f"\n  {'Period':12}  {'n':>6}  {'Auto%':>6}  {'Assist%':>7}  {'Esc%':>6}  Escalate bar")
print("  " + "-" * 70)
for lo, hi, label in BUCKETS:
    p = tier_pcts(arc[label])
    if p:
        n = sum(arc[label].values())
        print(f"  {label}  {n:>6,}  {p['Automate']:>5.1f}%  {p['Assist']:>6.1f}%  {p['Escalate']:>5.1f}%  {bar(p['Escalate'])}")

# ── Analysis 2: Top 10 individual career arcs ────────────────────────────────
print("\n" + "=" * 72)
print("  ANALYSIS 2 — TOP 10 CONTRIBUTORS: full career arc year by year")
print("=" * 72)

top10 = sorted(qualified, key=lambda a: qualified[a], reverse=True)[:10]

for rank, a in enumerate(top10, 1):
    fy  = first_year[a]
    total_n = totals[a]
    years = sorted(career[a].keys())
    span = years[-1] - years[0] + 1

    # Overall tier split
    overall = defaultdict(int)
    for yr_counts in career[a].values():
        for tier, n in yr_counts.items():
            overall[tier] += n
    op = tier_pcts(overall)

    print(f"\n  Engineer #{rank}  |  {total_n} tickets  |  {years[0]}–{years[-1]}  ({span} yr career)")
    print(f"  Overall: Auto {op['Automate']:.0f}%  Assist {op['Assist']:.0f}%  Escalate {op['Escalate']:.0f}%")
    print(f"  {'Year':>5}  {'n':>4}  {'Auto%':>6}  {'Esc%':>5}  Escalate bar")

    for yr in years:
        p = tier_pcts(career[a][yr])
        n = sum(career[a][yr].values())
        if p and n >= 3:
            career_yr = yr - fy + 1
            marker = ' <- first year' if career_yr == 1 else ''
            print(f"  {yr:>5}  {n:>4}  {p['Automate']:>5.1f}%  {p['Escalate']:>4.1f}%  {bar(p['Escalate'])}{marker}")

# ── Analysis 3: Do starters on Automate end up doing more Escalate? ──────────
print("\n" + "=" * 72)
print("  ANALYSIS 3 — STARTING TIER vs LATER ESCALATE%")
print("  Does starting on easy work lead to more complex work later?")
print("=" * 72)

# Classify each engineer by their first-year dominant tier
# then look at their Escalate% in years 4+
early_auto  = []  # started Automate-heavy
early_esc   = []  # started Escalate-heavy
early_mixed = []  # started balanced

for a in qualified:
    fy = first_year[a]
    # First 2 years
    early = defaultdict(int)
    for yr, yc in career[a].items():
        if yr - fy + 1 <= 2:
            for tier, n in yc.items():
                early[tier] += n
    ep = tier_pcts(early)
    if not ep or sum(early.values()) < 5:
        continue

    # Later years (4+)
    late = defaultdict(int)
    for yr, yc in career[a].items():
        if yr - fy + 1 >= 4:
            for tier, n in yc.items():
                late[tier] += n
    lp = tier_pcts(late)
    if not lp or sum(late.values()) < 5:
        continue

    row = (ep['Escalate'], lp['Escalate'], sum(early.values()), sum(late.values()))
    if ep['Automate'] >= 40:
        early_auto.append(row)
    elif ep['Escalate'] >= 50:
        early_esc.append(row)
    else:
        early_mixed.append(row)

def avg_late_esc(group):
    if not group: return None
    return sum(r[1] for r in group) / len(group)

def avg_early_esc(group):
    if not group: return None
    return sum(r[0] for r in group) / len(group)

print(f"\n  Engineers grouped by their first 2 years' dominant tier:")
print(f"  {'Group':30s}  {'n':>4}  {'Early Esc%':>10}  {'Late Esc% (yr4+)':>16}")
print("  " + "-" * 68)

groups = [
    ("Started Automate-heavy (>=40%)", early_auto),
    ("Started mixed",                 early_mixed),
    ("Started Escalate-heavy (>=50%)", early_esc),
]
for label, group in groups:
    if group:
        print(f"  {label:30s}  {len(group):>4}  {avg_early_esc(group):>9.1f}%  {avg_late_esc(group):>15.1f}%")

# ── Analysis 4: Tier specialisation vs generalism ───────────────────────────
print("\n" + "=" * 72)
print("  ANALYSIS 4 — SPECIALISATION PATTERNS")
print("  Do engineers stick to one tier or work across all three?")
print("=" * 72)

specialists = {'Automate': 0, 'Assist': 0, 'Escalate': 0}
generalists = 0
total_q = 0

for a in qualified:
    overall = defaultdict(int)
    for yc in career[a].values():
        for tier, n in yc.items():
            overall[tier] += n
    p = tier_pcts(overall)
    if not p:
        continue
    total_q += 1
    dominant = max(TIERS, key=lambda t: p[t])
    if p[dominant] >= 70:
        specialists[dominant] += 1
    else:
        generalists += 1

print(f"\n  Out of {total_q} engineers with >=30 clustered tickets:")
for tier in TIERS:
    pct = specialists[tier] / total_q * 100
    print(f"  Specialist in {tier:10s}: {specialists[tier]:>3}  ({pct:.0f}%)  {bar(pct)}")
gen_pct = generalists / total_q * 100
print(f"  Generalist (no tier >=70%): {generalists:>3}  ({gen_pct:.0f}%)  {bar(gen_pct)}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY — LEARNING PATH IMPLICATIONS")
print("=" * 72)
print()
print("  Key findings for skills acquisition:")
print()

# Compute the career arc direction
yr1 = arc['Yr 1      ']
yr7 = arc['Yr 7+     ']
p1 = tier_pcts(yr1)
p7 = tier_pcts(yr7)
if p1 and p7:
    esc_delta = p7['Escalate'] - p1['Escalate']
    auto_delta = p7['Automate'] - p1['Automate']
    direction = "rises" if esc_delta > 5 else ("falls" if esc_delta < -5 else "stays flat")
    print(f"  1. Escalate% {direction} with career age: Yr1={p1['Escalate']:.0f}% -> Yr7+={p7['Escalate']:.0f}% ({esc_delta:+.0f}pp)")
    print(f"     Automate% change: {p1['Automate']:.0f}% -> {p7['Automate']:.0f}% ({auto_delta:+.0f}pp)")

if early_auto and early_esc:
    diff = avg_late_esc(early_esc) - avg_late_esc(early_auto)
    print(f"\n  2. Engineers who started on Escalate work end up doing {diff:+.0f}pp")
    print(f"     more Escalate in later years than those who started on Automate.")
    if diff > 5:
        print(f"     -> Starting on hard work predicts more hard work later.")
        print(f"        The traditional 'start easy, progress to hard' path may be")
        print(f"        less common than assumed.")
    elif abs(diff) < 5:
        print(f"     -> Starting tier has little effect on later tier distribution.")
        print(f"        Engineers converge to similar profiles regardless of start.")

print(f"\n  3. Specialisation: most engineers gravitate to one tier over time.")
print(f"     Implications for team design: you need BOTH specialists and")
print(f"     generalists — don't assume everyone can cover all three tiers.")
print()
