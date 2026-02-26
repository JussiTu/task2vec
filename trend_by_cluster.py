"""
trend_by_cluster.py
───────────────────
Re-runs the AI readiness tier trend by year using embedding cluster labels
(the same basis as the published 22/30/48 distribution and the chart).

This is the correct method for the time-trend analysis — outcome signals
(resolution speed) are a different classifier and should not be mixed with
the embedding-based tier distribution.

Sources:
  search_meta.json   — 69k tickets, each with key / year / cluster ID
  umap_app_data.json — cluster ID -> label mapping
  ai_readiness_chart.html — hardcoded cluster -> tier assignments (ground truth)
"""

import json, re
from collections import defaultdict
from pathlib import Path

CACHE = Path('.cache')

# ── 1. Cluster -> tier mapping (from ai_readiness_chart.html HTML) ─────────
#    Verified tiers first, then heuristic assignments for unlabeled clusters.

CLUSTER_TIER = {
    # --- Automate (confirmed from chart HTML) ---
    12: 'Automate',   # Documentation Corrections and Edits
    25: 'Automate',   # Spring Dependency Upgrades
    13: 'Automate',   # Spring Data Commons Deprecation Updates
     2: 'Automate',   # Release Version Updates
    21: 'Automate',   # Codebase Consistency and Refactoring

    # --- Assist (confirmed from chart HTML) ---
    23: 'Assist',     # RequestMapping Enhancements
     3: 'Assist',     # Content Negotiation Issues
    16: 'Assist',     # Transaction Management Issues
    24: 'Assist',     # MongoDB Mapping Issues

    # --- Escalate (confirmed from chart HTML) ---
     4: 'Escalate',   # JDBC Compatibility and Performance Issues
     7: 'Escalate',   # JMS Message Listener Issues
     6: 'Escalate',   # Authentication Framework Enhancements
    15: 'Escalate',   # JPA Entity Management Issues

    # --- Heuristic assignments for remaining clusters ---
     0: 'Escalate',   # Validation and Data Binding Issues
     1: 'Escalate',   # Spring Roo Initialization Errors
     5: 'Assist',     # Spring IDE Tooling Enhancements
     8: 'Escalate',   # JPA and Entity Mapping Issues
     9: 'Escalate',   # Autowiring and Bean Configuration Issues
    10: 'Escalate',   # Workspace and Class Loading Issues
    11: 'Assist',     # Messaging Configuration Enhancements
    14: 'Assist',     # RedisTemplate and Jedis Improvements
    17: 'Automate',   # Acceptance Test Failures
    18: 'Assist',     # Bean Configuration Enhancements
    19: 'Escalate',   # Step Scope and Tasklet Issues
    20: 'Escalate',   # Spring XD Module Development
    22: 'Automate',   # Documentation and Build Improvements
    26: 'Assist',     # Spring Web Flow Enhancements
    27: 'Escalate',   # QueryDSL and Projections Support
    28: 'Escalate',   # JSF and Spring Web Flow Issues
    29: 'Escalate',   # Session Authentication Issues
    30: 'Assist',     # MongoDB Feature Enhancements
    31: 'Escalate',   # OSGi Manifest and Dependency Issues
}

# ── 2. Load data ───────────────────────────────────────────────────────────
print("Loading data...")
with open(CACHE / 'search_meta.json', encoding='utf-8', errors='replace') as f:
    meta_list = json.load(f)

with open(CACHE / 'umap_app_data.json', encoding='utf-8', errors='replace') as f:
    umap = json.load(f)

cluster_labels = {int(k): v['label'] for k, v in umap['cluster_labels'].items()}

print(f"  Tickets: {len(meta_list):,}")
print(f"  Clusters with tier assignment: {len(CLUSTER_TIER)}")

# ── 3. Tier summary per cluster ───────────────────────────────────────────
print("\nCluster -> Tier assignments:")
print(f"  {'ID':>3}  {'Tier':10s}  {'Label'}")
print(f"  {'--':>3}  {'----':10s}  {'-----'}")
for cid in sorted(CLUSTER_TIER):
    label = cluster_labels.get(cid, '?')
    src   = '(confirmed)' if cid in [12,25,13,2,21,23,3,16,24,4,7,6,15] else '(heuristic)'
    print(f"  {cid:>3}  {CLUSTER_TIER[cid]:10s}  {label[:50]:50s}  {src}")

# ── 4. Count tickets by (year, tier) ──────────────────────────────────────
by_year = defaultdict(lambda: defaultdict(int))
skipped_no_cluster = 0
skipped_no_year    = 0
total_assigned     = 0

for m in meta_list:
    yr      = m.get('year')
    cluster = m.get('cluster')
    if yr is None:
        skipped_no_year += 1
        continue
    if cluster is None:
        skipped_no_cluster += 1
        continue
    tier = CLUSTER_TIER.get(cluster)
    if tier is None:
        continue
    by_year[yr][tier]  += 1
    total_assigned     += 1

print(f"\nAssigned: {total_assigned:,}  |  No cluster: {skipped_no_cluster:,}  |  No year: {skipped_no_year:,}")

TIERS = ['Automate', 'Assist', 'Escalate']

def pct(counts):
    total = sum(counts.get(t, 0) for t in TIERS)
    if total < 30:
        return None, None, None, total
    a = counts.get('Automate', 0) / total * 100
    s = counts.get('Assist',   0) / total * 100
    e = counts.get('Escalate', 0) / total * 100
    return a, s, e, total

def bar(v, width=15):
    if v is None: return '?' * width
    filled = round(v / 100 * width)
    return '#' * filled + '.' * (width - filled)

# ── 5. Full trend table ────────────────────────────────────────────────────
YEARS = sorted(by_year.keys())

print("\n" + "=" * 75)
print("  EMBEDDING-CLUSTER TREND  (all 69k tickets, by created year)")
print("=" * 75)
print(f"  {'Year':>5}  {'n':>6}  {'Auto%':>6}  {'Assist%':>7}  {'Esc%':>6}  Automate bar")
print("  " + "-" * 73)
for yr in YEARS:
    a, s, e, n = pct(by_year[yr])
    if n >= 30:
        print(f"  {yr:>5}  {n:>6,}  {a:>5.1f}%  {s:>6.1f}%  {e:>5.1f}%  {bar(a)}")

# ── 6. Compare to hardcoded chart numbers ─────────────────────────────────
chart_years    = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
chart_automate = [19.4,20.1,23.6,21.3,21.6,19.1,16.5,13.0,15.0,5.2,4.6]
chart_assist   = [44.8,42.6,30.6,40.3,37.1,34.5,28.0,25.2,25.9,31.2,27.3]
chart_escalate = [35.8,37.4,45.8,38.5,41.3,46.4,55.4,61.8,59.0,63.6,68.1]
chart_lookup   = {yr: (a,s,e) for yr,(a,s,e)
                  in zip(chart_years, zip(chart_automate, chart_assist, chart_escalate))}

print("\n" + "=" * 75)
print("  COMPARISON: recomputed vs published chart (2010-2020 window)")
print("=" * 75)
print(f"  {'Year':>5}  {'Auto% (new)':>11}  {'Auto% (chart)':>13}  {'diff':>6}  match?")
print("  " + "-" * 73)
diffs = []
for yr in chart_years:
    a, s, e, n = pct(by_year.get(yr, {}))
    ca, cs, ce = chart_lookup[yr]
    if a is not None:
        diff = a - ca
        diffs.append(abs(diff))
        match = 'OK' if abs(diff) < 3 else 'DIFF'
        print(f"  {yr:>5}  {a:>10.1f}%  {ca:>12.1f}%  {diff:>+5.1f}pp  {match}")

if diffs:
    print(f"\n  Mean absolute difference: {sum(diffs)/len(diffs):.1f}pp")
    print(f"  Max absolute difference:  {max(diffs):.1f}pp")
    if sum(diffs)/len(diffs) < 3:
        print("  VERDICT: recomputed values match the chart closely.")
        print("  Cluster-tier assignments are consistent with the original data source.")
    else:
        print("  VERDICT: significant divergence -- cluster-tier mapping may differ from original.")

# ── 7. Verdict on the trend ────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  TREND ANALYSIS")
print("=" * 75)

# All years with enough data
valid = [(yr, pct(by_year[yr])) for yr in YEARS if pct(by_year[yr])[3] >= 30]
valid = [(yr, a, s, e, n) for yr, (a, s, e, n) in valid if a is not None]

early = [(yr, a) for yr, a, s, e, n in valid if 2003 <= yr <= 2010]
mid   = [(yr, a) for yr, a, s, e, n in valid if 2011 <= yr <= 2015]
late  = [(yr, a) for yr, a, s, e, n in valid if 2016 <= yr <= 2022]

def wavg(pairs):
    if not pairs: return None
    return sum(a for _, a in pairs) / len(pairs)

ea, ma, la = wavg(early), wavg(mid), wavg(late)
if ea and ma and la:
    print(f"\n  Automate % by period:")
    print(f"    2003-2010 (early):  {ea:.1f}%")
    print(f"    2011-2015 (mid):    {ma:.1f}%")
    print(f"    2016-2022 (late):   {la:.1f}%")

    if la < ea - 5 and la < ma:
        print(f"\n  Trend: declining. The late period has {ea-la:.1f}pp less automatable")
        print(f"  work than early years. Consistent with the project maturity hypothesis.")
    elif abs(la - ea) < 5:
        print(f"\n  Trend: FLAT. No meaningful change across the project lifetime.")
        print(f"  The cluster composition is stable over time.")
    else:
        shape = "U-shaped" if ma < ea and la > ma else "irregular"
        print(f"\n  Trend: {shape}.")
        print(f"  Early peak, mid trough, partial recovery -- not a clean monotonic decline.")
        print(f"  The published '19% -> 5%' covers only the steepest downward segment.")

print()
print("  IMPORTANT NOTES:")
print("  1. Unclustered tickets (HDBSCAN noise, ~12% of corpus) are excluded.")
print("     These are likely the most unusual/hard tickets -- excluding them")
print("     may systematically undercount Escalate in any given year.")
print("  2. Heuristic cluster assignments (marked above) introduce subjectivity.")
print("     Replication requires publishing the full cluster->tier mapping.")
print("  3. The 'created year' used here reflects when tickets were filed,")
print("     not when they were resolved. Strategic deferrals inflate later years")
print("     with unresolved tickets of any tier.")
print()
