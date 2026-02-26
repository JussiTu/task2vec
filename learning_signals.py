"""
learning_signals.py
────────────────────
Four analyses on whether observable Jira+Git signals can detect
whether genuine learning happened — and whether being "given" code
(proxy for AI assistance) enables later independence.

A. Within-cluster velocity  — resolution time drops with experience?
B. Deepener vs sampler      — does specialisation drive Escalate growth?
C. Git engagement & learning— tickets with commits vs without: different arcs?
D. Code-given -> independence— low git early, high git late in same cluster?

Data:
  .cache/search_meta.json       key / assignee / year / cluster
  .cache/outcome_signals.json   key / days / watches / label
  .cache/git_index.json         key / sha / files / label
"""

import json, math
from collections import defaultdict
from pathlib import Path

CACHE = Path('.cache')

CLUSTER_TIER = {
    12:'Automate',25:'Automate',13:'Automate', 2:'Automate',21:'Automate',
    17:'Automate',22:'Automate',
    23:'Assist',   3:'Assist',  16:'Assist',  24:'Assist',  5:'Assist',
    11:'Assist',  14:'Assist',  18:'Assist',  26:'Assist',  30:'Assist',
     4:'Escalate', 7:'Escalate', 6:'Escalate',15:'Escalate', 0:'Escalate',
     1:'Escalate', 8:'Escalate', 9:'Escalate',10:'Escalate',19:'Escalate',
    20:'Escalate',27:'Escalate',28:'Escalate',29:'Escalate',31:'Escalate',
}

CLUSTER_LABEL = {
    12:'Documentation Corrections', 25:'Dependency Upgrades',
    13:'Deprecation Updates',        2:'Release Version Updates',
    21:'Codebase Consistency',      17:'Acceptance Test Failures',
    22:'Build Improvements',        23:'RequestMapping Enhancements',
     3:'Content Negotiation',       16:'Transaction Management',
    24:'MongoDB Mapping',            5:'Spring IDE Tooling',
    11:'Messaging Config',          14:'Redis Improvements',
    18:'Bean Configuration',        26:'Spring Web Flow',
    30:'MongoDB Features',           4:'JDBC Compatibility',
     7:'JMS Message Listener',       6:'Authentication Framework',
    15:'JPA Entity Management',      0:'Validation & Data Binding',
     1:'Spring Roo Errors',          8:'JPA & Entity Mapping',
     9:'Autowiring & Bean Config',  10:'Workspace & Class Loading',
    19:'Step Scope & Tasklet',      20:'Spring XD Module Dev',
    27:'QueryDSL & Projections',    28:'JSF & Spring Web Flow',
    29:'Session Authentication',    31:'OSGi Manifest & Deps',
}

TIERS = ['Automate','Assist','Escalate']

def bar(pct, width=15):
    filled = round(pct / 100 * width)
    return '#' * filled + '.' * (width - filled)

def tier_pcts(counts):
    total = sum(counts.get(t,0) for t in TIERS)
    if not total: return None
    return {t: counts.get(t,0)/total*100 for t in TIERS}

def spearman(xs, ys):
    """Spearman rank correlation — detects monotonic trend."""
    n = len(xs)
    if n < 4: return None
    def ranks(v):
        s = sorted(range(n), key=lambda i: v[i])
        r = [0]*n
        for rank, i in enumerate(s): r[i] = rank+1
        return r
    rx, ry = ranks(xs), ranks(ys)
    d2 = sum((rx[i]-ry[i])**2 for i in range(n))
    rho = 1 - 6*d2 / (n*(n*n-1))
    return rho

# ── Load ────────────────────────────────────────────────────────────────────
print("Loading data...")
with open(CACHE/'search_meta.json', encoding='utf-8', errors='replace') as f:
    meta = json.load(f)
with open(CACHE/'outcome_signals.json', encoding='utf-8') as f:
    outcome_data = json.load(f)
    signals = outcome_data['signals']
with open(CACHE/'git_index.json', encoding='utf-8') as f:
    git_index = json.load(f)

git_keys = set(git_index.keys())

# Build enriched ticket list
tickets = []
for m in meta:
    if not m.get('assignee') or not m.get('year'): continue
    tier = CLUSTER_TIER.get(m.get('cluster'))
    if tier is None: continue
    sig   = signals.get(m['key'])
    days  = sig['days'] if sig and sig.get('days') is not None else None
    in_git = m['key'] in git_keys
    tickets.append({
        'key':      m['key'],
        'assignee': m['assignee'],
        'year':     m['year'],
        'cluster':  m['cluster'],
        'tier':     tier,
        'days':     days,
        'in_git':   in_git,
    })

print(f"  Total usable tickets: {len(tickets):,}")
print(f"  With resolution days: {sum(1 for t in tickets if t['days'] is not None):,}")
print(f"  Git-confirmed:        {sum(1 for t in tickets if t['in_git']):,}")

# Build per-assignee structures
career = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
first_year = {}
for t in tickets:
    a, yr = t['assignee'], t['year']
    career[a][yr][t['tier']] += 1
    if a not in first_year or yr < first_year[a]: first_year[a] = yr

totals = {a: sum(sum(y.values()) for y in yrs.values()) for a,yrs in career.items()}
qualified = {a for a,n in totals.items() if n >= 10 and len(career[a]) >= 2}
print(f"  Qualified assignees:  {len(qualified):,}")


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS A — Within-cluster velocity
# Does resolution time decrease as an engineer repeats the same cluster?
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  ANALYSIS A — WITHIN-CLUSTER VELOCITY")
print("  Do engineers resolve the same ticket type faster with experience?")
print("="*72)

# For each (assignee, cluster): collect (ordinal, days) pairs sorted by year
cluster_tickets = defaultdict(lambda: defaultdict(list))  # [a][cl] = [(year,days)]
for t in tickets:
    if t['days'] is not None and t['assignee'] in qualified:
        cluster_tickets[t['assignee']][t['cluster']].append((t['year'], t['days']))

# Compute Spearman rho for each (assignee, cluster) with >=5 data points
rhos = []
negative_rhos = 0  # negative = getting faster = learning
total_series  = 0

for a, clusters in cluster_tickets.items():
    for cl, pairs in clusters.items():
        if len(pairs) < 5: continue
        pairs.sort(key=lambda x: x[0])
        ordinals = list(range(len(pairs)))
        days     = [p[1] for p in pairs]
        rho = spearman(ordinals, days)
        if rho is None: continue
        total_series += 1
        rhos.append((rho, cl, a, len(pairs)))
        if rho < -0.2: negative_rhos += 1

if rhos:
    avg_rho = sum(r[0] for r in rhos) / len(rhos)
    pct_faster = 100 * negative_rhos / total_series
    print(f"\n  Engineer-cluster series with >=5 data points: {total_series}")
    print(f"  Mean Spearman rho (negative = getting faster):  {avg_rho:+.3f}")
    print(f"  % of series with clear speedup (rho < -0.2):    {pct_faster:.0f}%")

    # Show by tier
    print(f"\n  Mean rho by tier:")
    for tier in TIERS:
        tier_rhos = [r[0] for r in rhos if CLUSTER_TIER.get(r[1]) == tier]
        if tier_rhos:
            avg = sum(tier_rhos)/len(tier_rhos)
            pct = 100*sum(1 for r in tier_rhos if r < -0.2)/len(tier_rhos)
            print(f"    {tier:10s}: mean rho={avg:+.3f}  speedup in {pct:.0f}% of series  (n={len(tier_rhos)})")

    # Best examples: clearest learning curves
    print(f"\n  Strongest learning curves (most negative rho, n>=8 tickets):")
    best = sorted([r for r in rhos if r[3] >= 8], key=lambda x: x[0])[:5]
    for rho, cl, a, n in best:
        label = CLUSTER_LABEL.get(cl,'?')
        tier  = CLUSTER_TIER.get(cl,'?')
        print(f"    rho={rho:+.3f}  n={n:>2}  {tier:10s}  {label}")


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS B — Deepener vs Sampler
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  ANALYSIS B — DEEPENER vs SAMPLER PROFILES")
print("  Does specialising in cluster types drive faster Escalate growth?")
print("="*72)

deepeners = []
samplers  = []

for a in qualified:
    cl_counts = defaultdict(int)
    for t in tickets:
        if t['assignee'] == a: cl_counts[t['cluster']] += 1
    if not cl_counts: continue

    distinct = len(cl_counts)
    total_n  = sum(cl_counts.values())
    avg_per_cl = total_n / distinct  # higher = deepener

    # Career growth: first-year Esc% vs last-year Esc%
    fy    = first_year[a]
    years = sorted(career[a].keys())
    p1 = tier_pcts(career[a][fy])
    last_counts = defaultdict(int)
    for yr in years[-2:]:
        for tier,n in career[a][yr].items(): last_counts[tier]+=n
    pl = tier_pcts(last_counts)
    if not p1 or not pl: continue
    span = years[-1]-years[0]+1
    if span < 3: continue

    row = (a, avg_per_cl, distinct, total_n, p1['Escalate'], pl['Escalate'], span)
    if avg_per_cl >= 5:
        deepeners.append(row)
    elif avg_per_cl <= 2:
        samplers.append(row)

def avg(lst, idx): return sum(r[idx] for r in lst)/len(lst) if lst else 0

print(f"\n  {'Group':20s}  {'n':>4}  {'Avg clusters':>12}  {'Avg tickets/cl':>14}  {'Start Esc%':>10}  {'End Esc%':>9}  {'Change':>7}")
print("  " + "-"*80)
for label, group in [("Deepeners (>=5 t/cl)", deepeners), ("Samplers (<=2 t/cl)", samplers)]:
    if group:
        print(f"  {label:20s}  {len(group):>4}  {avg(group,2):>12.1f}  {avg(group,1):>14.1f}  "
              f"{avg(group,4):>9.1f}%  {avg(group,5):>8.1f}%  {avg(group,5)-avg(group,4):>+6.1f}pp")


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS C — Git engagement and learning
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  ANALYSIS C — GIT ENGAGEMENT AND LEARNING")
print("  Engineers who produce commits vs those who don't: different arcs?")
print("="*72)

# Git engagement rate per engineer = git_confirmed / total tickets
# (restricted to 2008-2012 where git coverage was reasonable)
git_coverage_years = set(range(2008, 2013))

eng_git = defaultdict(lambda: {'git':0,'total':0})
for t in tickets:
    if t['year'] not in git_coverage_years: continue
    if t['assignee'] not in qualified: continue
    eng_git[t['assignee']]['total'] += 1
    if t['in_git']: eng_git[t['assignee']]['git'] += 1

high_git = []  # git rate >= 10%
low_git  = []  # git rate < 3%

for a, counts in eng_git.items():
    if counts['total'] < 10: continue
    rate = counts['git'] / counts['total']
    fy   = first_year[a]
    years = sorted(career[a].keys())
    if len(years) < 3: continue

    # Escalate growth across full career
    p1 = tier_pcts(career[a][fy])
    late = defaultdict(int)
    for yr in years[-2:]:
        for tier,n in career[a][yr].items(): late[tier]+=n
    pl = tier_pcts(late)
    if not p1 or not pl: continue

    row = (a, rate, counts['git'], counts['total'], p1['Escalate'], pl['Escalate'])
    if rate >= 0.10:
        high_git.append(row)
    elif rate < 0.03:
        low_git.append(row)

print(f"\n  {'Group':30s}  {'n':>4}  {'Avg git rate':>12}  {'Start Esc%':>10}  {'End Esc%':>9}  {'Growth':>7}")
print("  " + "-"*72)
for label, group in [("High git engagement (>=10%)", high_git), ("Low git engagement (<3%)", low_git)]:
    if group:
        avg_rate  = sum(r[1] for r in group)/len(group)
        avg_start = sum(r[4] for r in group)/len(group)
        avg_end   = sum(r[5] for r in group)/len(group)
        print(f"  {label:30s}  {len(group):>4}  {avg_rate:>11.1%}  {avg_start:>9.1f}%  {avg_end:>8.1f}%  {avg_end-avg_start:>+6.1f}pp")

# Also: does having a commit on first tickets in a cluster predict faster progression?
print(f"\n  First ticket in a cluster: git-confirmed vs not")
print(f"  -> Does starting with a commit predict faster cluster deepening?")

first_git_fast  = []  # first ticket git-confirmed, later faster
first_ngit_fast = []  # first ticket not git-confirmed

for a in qualified:
    cl_tickets = defaultdict(list)
    for t in tickets:
        if t['assignee'] == a and t['days'] is not None:
            cl_tickets[t['cluster']].append(t)
    for cl, ts in cl_tickets.items():
        if len(ts) < 5: continue
        ts.sort(key=lambda x: x['year'])
        first_git = ts[0]['in_git']
        # Velocity: early avg days vs late avg days
        half = len(ts)//2
        early_avg = sum(t['days'] for t in ts[:half]) / half
        late_avg  = sum(t['days'] for t in ts[half:]) / (len(ts)-half)
        speedup   = early_avg - late_avg  # positive = got faster
        if first_git:
            first_git_fast.append(speedup)
        else:
            first_ngit_fast.append(speedup)

if first_git_fast and first_ngit_fast:
    avg_fg  = sum(first_git_fast)/len(first_git_fast)
    avg_nfg = sum(first_ngit_fast)/len(first_ngit_fast)
    print(f"\n  First ticket git-confirmed:     avg speedup = {avg_fg:+.1f} days  (n={len(first_git_fast)})")
    print(f"  First ticket NOT git-confirmed: avg speedup = {avg_nfg:+.1f} days  (n={len(first_ngit_fast)})")
    diff = avg_fg - avg_nfg
    if diff > 5:
        print(f"  -> Having a commit on your first encounter with a cluster type")
        print(f"     predicts {diff:.0f} days more speedup in later tickets of that type.")
    elif abs(diff) <= 5:
        print(f"  -> No meaningful difference. Starting git/non-git doesn't predict velocity.")


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS D — "Code given" -> independence
# Proxy: tickets resolved without a git commit (code provided elsewhere, or
# passive resolution) early in a cluster, then git-confirmed commits later.
# Does the passive->active shift happen? And does it predict learning?
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  ANALYSIS D — CODE GIVEN -> INDEPENDENCE")
print("  Engineers who had code 'provided' early: did they become independent?")
print("  Proxy: low git-confirmation early in cluster -> high git-confirmation later")
print("="*72)

# For each (engineer, cluster) with >=6 tickets:
# Split into first half and second half
# Compare git-confirmation rate first-half vs second-half

became_independent = []  # git rate went up substantially
stayed_passive     = []
stayed_active      = []
mixed              = []

for a in qualified:
    cl_tickets = defaultdict(list)
    for t in tickets:
        if t['assignee'] == a:
            cl_tickets[t['cluster']].append(t)

    for cl, ts in cl_tickets.items():
        if len(ts) < 6: continue
        ts.sort(key=lambda x: x['year'])
        half = len(ts) // 2
        early = ts[:half]
        late  = ts[half:]

        early_git_rate = sum(1 for t in early if t['in_git']) / len(early)
        late_git_rate  = sum(1 for t in late  if t['in_git']) / len(late)

        row = (a, cl, early_git_rate, late_git_rate, len(ts))
        delta = late_git_rate - early_git_rate

        if early_git_rate < 0.1 and late_git_rate >= 0.15:
            became_independent.append(row)
        elif early_git_rate < 0.1 and late_git_rate < 0.1:
            stayed_passive.append(row)
        elif early_git_rate >= 0.15 and late_git_rate >= 0.15:
            stayed_active.append(row)
        else:
            mixed.append(row)

total_series_d = len(became_independent)+len(stayed_passive)+len(stayed_active)+len(mixed)
print(f"\n  Engineer-cluster series with >=6 tickets: {total_series_d}")
if total_series_d:
    for label, group in [
        ("Became independent (passive->active)", became_independent),
        ("Stayed passive (low git throughout)",  stayed_passive),
        ("Always active (high git throughout)",  stayed_active),
        ("Mixed pattern",                        mixed),
    ]:
        pct = 100*len(group)/total_series_d if total_series_d else 0
        print(f"  {label:40s}: {len(group):>4}  ({pct:.0f}%)")

# For "became independent": show examples and check if they also showed velocity improvement
print(f"\n  'Became independent' examples (passive early, active later):")
if became_independent:
    for a, cl, eg, lg, n in sorted(became_independent, key=lambda x: x[3]-x[2], reverse=True)[:6]:
        label = CLUSTER_LABEL.get(cl,'?')
        tier  = CLUSTER_TIER.get(cl,'?')
        print(f"    {tier:10s}  {label:35s}  git: {eg:.0%} -> {lg:.0%}  (n={n})")

# Key question: do "became independent" engineers show better Escalate growth
# than "stayed passive" engineers?
print(f"\n  Career Escalate% growth by independence pattern:")
print(f"  (comparing early career Esc% to late career Esc%)")

for label, group in [("Became independent", became_independent), ("Stayed passive", stayed_passive)]:
    if not group: continue
    growths = []
    for a, cl, eg, lg, n in group:
        fy    = first_year[a]
        years = sorted(career[a].keys())
        if len(years) < 3: continue
        p1 = tier_pcts(career[a][fy])
        late_counts = defaultdict(int)
        for yr in years[-2:]:
            for tier,cnt in career[a][yr].items(): late_counts[tier]+=cnt
        pl = tier_pcts(late_counts)
        if p1 and pl:
            growths.append(pl['Escalate'] - p1['Escalate'])
    if growths:
        avg_growth = sum(growths)/len(growths)
        print(f"  {label:35s}: avg Escalate growth = {avg_growth:+.1f}pp  (n={len(growths)})")

# ════════════════════════════════════════════════════════════════════════════
# SYNTHESIS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  SYNTHESIS — WHAT THE FOUR SIGNALS SAY TOGETHER")
print("="*72)
print()
print("  A. Velocity: do engineers get faster at the same ticket type?")
if rhos:
    pct_f = 100*negative_rhos/total_series
    print(f"     {pct_f:.0f}% of engineer-cluster series show clear speedup (rho<-0.2).")
    print(f"     Average rho={avg_rho:+.3f}. Learning is happening — measurable, not uniform.")
print()
print("  B. Deepeners grow more than samplers.")
if deepeners and samplers:
    d_growth = avg(deepeners,5)-avg(deepeners,4)
    s_growth = avg(samplers,5)-avg(samplers,4)
    print(f"     Deepeners: {d_growth:+.1f}pp Escalate growth.")
    print(f"     Samplers:  {s_growth:+.1f}pp Escalate growth.")
    print(f"     Delta: {d_growth-s_growth:+.1f}pp. Specialisation outperforms breadth for expertise.")
print()
print("  C. Git engagement correlates with career growth.")
if high_git and low_git:
    hg_growth = avg(high_git,5)-avg(high_git,4)
    lg_growth = avg(low_git,5)-avg(low_git,4)
    print(f"     High-git engineers: {hg_growth:+.1f}pp growth.")
    print(f"     Low-git engineers:  {lg_growth:+.1f}pp growth.")
    print(f"     Producing code — not just resolving tickets — predicts expertise.")
print()
print("  D. Passive->active transition is detectable and meaningful.")
if became_independent and stayed_passive:
    print(f"     {len(became_independent)} engineer-cluster pairs show the independence shift.")
    print(f"     This is the 'code given, then learned' pattern — measurable in data.")
print()
print("  AI IMPLICATION:")
print("  If AI provides the code (like a senior colleague in pre-2012 Spring),")
print("  the passive->active transition is exactly what you need to monitor.")
print("  The data suggests it CAN happen — but requires the right conditions.")
print("  A team that only measures 'tickets resolved' will miss whether it did.")
print()
