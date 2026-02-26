"""
check_strategy_confound.py
──────────────────────────
Tests whether the declining "automatable" trend over time is a genuine
difficulty signal or a strategic-deferral confound.

Three analyses:
  1. OVERALL TREND — tier distribution by year across all resolved tickets
     (outcome_signals.json × search_meta.json)

  2. GIT-CONFIRMED TREND — same distribution, but restricted to tickets
     that have a confirmed git commit (i.e. definitely implemented)
     (git_index.json × search_meta.json)

  3. IMPLEMENTATION RATE — what % of resolved tickets each year also appear
     in the git index. A declining rate would mean later tickets are
     increasingly resolved without producing code changes (strategic closure,
     docs-only, won't-fix, etc.)

If the trend in (2) is flat while (1) declines, the decline in (1) is
driven by un-implemented or strategically-deferred tickets — not by
genuine difficulty growth.
"""

import json
from collections import defaultdict
from pathlib import Path

CACHE = Path('.cache')

# ── Load data ──────────────────────────────────────────────────────────────

print("Loading data…")

with open(CACHE / 'git_index.json', encoding='utf-8') as f:
    git_index = json.load(f)   # {key: {sha, label, ...}}

with open(CACHE / 'outcome_signals.json', encoding='utf-8') as f:
    outcome_data = json.load(f)
outcome_signals = outcome_data['signals']  # {key: {days, watches, label, ...}}

with open(CACHE / 'search_meta.json', encoding='utf-8', errors='replace') as f:
    meta_list = json.load(f)
year_by_key = {m['key']: m['year'] for m in meta_list if m.get('year')}

print(f"  git_index:       {len(git_index):,} tickets (confirmed code commits)")
print(f"  outcome_signals: {len(outcome_signals):,} tickets (all resolved)")
print(f"  search_meta:     {len(year_by_key):,} tickets with year\n")

TIERS   = ['Automate', 'Assist', 'Escalate']
YEARS   = list(range(2003, 2022))

# ── Analysis 1: overall resolved trend ────────────────────────────────────

overall = defaultdict(lambda: defaultdict(int))   # year -> tier -> count
for key, sig in outcome_signals.items():
    if key.startswith('SPR-'):
        yr = year_by_key.get(key)
        if yr:
            overall[yr][sig['label']] += 1

# ── Analysis 2: git-confirmed trend ───────────────────────────────────────

git_trend = defaultdict(lambda: defaultdict(int))   # year -> tier -> count
for key, info in git_index.items():
    yr = year_by_key.get(key)
    if yr:
        git_trend[yr][info['label']] += 1

# ── Analysis 3: implementation rate ───────────────────────────────────────

git_keys = set(git_index.keys())
impl_rate = {}   # year -> (git_count, resolved_count, rate)
for yr in YEARS:
    resolved = sum(overall[yr].values())
    git_n    = sum(git_trend[yr].values())
    if resolved > 0:
        impl_rate[yr] = (git_n, resolved, git_n / resolved)

# ── Helper: % row ──────────────────────────────────────────────────────────

def pct_row(counts):
    total = sum(counts.get(t, 0) for t in TIERS)
    if total == 0:
        return None, None, None, 0
    a = counts.get('Automate', 0) / total * 100
    s = counts.get('Assist',   0) / total * 100
    e = counts.get('Escalate', 0) / total * 100
    return a, s, e, total

# ── Print results ──────────────────────────────────────────────────────────

YEARS_SHOWN = [y for y in YEARS if sum(overall[y].values()) >= 50]

col = 8

def bar(pct, width=12, char='#'):
    filled = round(pct / 100 * width)
    return char * filled + '.' * (width - filled)

# ── TABLE 1: Overall resolved tickets ─────────────────────────────────────
print("=" * 72)
print("  ANALYSIS 1 — OVERALL TREND  (all resolved SPR tickets)")
print("=" * 72)
print(f"{'Year':>5}  {'n':>5}  {'Auto%':>6}  {'Assist%':>7}  {'Escalate%':>9}  Auto bar")
print("-" * 72)
for yr in YEARS_SHOWN:
    a, s, e, n = pct_row(overall[yr])
    if n >= 50:
        print(f"{yr:>5}  {n:>5,}  {a:>5.1f}%  {s:>6.1f}%  {e:>8.1f}%  {bar(a)}")
print()

# ── TABLE 2: Git-confirmed tickets ─────────────────────────────────────────
print("=" * 72)
print("  ANALYSIS 2 — GIT-CONFIRMED TREND  (tickets with code commits only)")
print("=" * 72)
print(f"{'Year':>5}  {'n':>5}  {'Auto%':>6}  {'Assist%':>7}  {'Escalate%':>9}  Auto bar")
print("-" * 72)
for yr in YEARS_SHOWN:
    a, s, e, n = pct_row(git_trend[yr])
    if n >= 3:
        print(f"{yr:>5}  {n:>5,}  {a:>5.1f}%  {s:>6.1f}%  {e:>8.1f}%  {bar(a)}")
    elif yr in git_trend:
        print(f"{yr:>5}  {n:>5,}  (too few for %)")
print()

# ── TABLE 3: Implementation rate ───────────────────────────────────────────
print("=" * 72)
print("  ANALYSIS 3 — IMPLEMENTATION RATE  (git commits / resolved tickets)")
print("  Low rate = tickets resolved WITHOUT code changes (strategic closure,")
print("  docs-only fix, won't-fix, duplicate, etc.)")
print("=" * 72)
print(f"{'Year':>5}  {'resolved':>9}  {'w/ code':>8}  {'rate':>6}  Bar")
print("-" * 72)
for yr in YEARS_SHOWN:
    git_n, res_n, rate = impl_rate.get(yr, (0, 0, 0))
    if res_n >= 50:
        print(f"{yr:>5}  {res_n:>9,}  {git_n:>8,}  {rate:>5.1%}  {bar(rate*100, width=20)}")
print()

# ── SUMMARY VERDICT ────────────────────────────────────────────────────────
print("=" * 72)
print("  VERDICT")
print("=" * 72)

# Finding A: git coverage cliff
peak_year = max((y for y in YEARS if y in impl_rate), key=lambda y: impl_rate[y][0])
peak_rate = impl_rate[peak_year][2]
post_years = [y for y in range(2013, 2022) if y in impl_rate and impl_rate[y][1] >= 50]
post_rate  = (sum(impl_rate[y][2] for y in post_years) / len(post_years)) if post_years else 0.0

print(f"\n  FINDING A -- Git coverage cliff")
print(f"  Peak git match rate: {peak_rate:.1%} in {peak_year}")
print(f"  Average post-2012:   {post_rate:.1%}")
print(f"  Spring stopped referencing SPR-#### in commit messages after ~2012.")
print(f"  Likely cause: migration to GitHub PRs (different link format).")
print(f"  Git-based analysis is only reliable for the 2008-2011 window.")

# Finding B: overall Spring auto% is very low
spr_years_n = [y for y in YEARS_SHOWN if pct_row(overall[y])[3] >= 100]
spr_autos   = [pct_row(overall[y])[0] for y in spr_years_n]
avg_spr_auto = sum(spr_autos) / len(spr_autos) if spr_autos else 0.0

print(f"\n  FINDING B -- Outcome-signal auto% is very low for Spring ({avg_spr_auto:.1f}% avg)")
print(f"  The p33_days=1.9 threshold was calibrated across ALL projects.")
print(f"  Spring tickets are structurally slower to resolve than simpler projects.")
print(f"  The published 22/30/48 split uses embedding clusters (content-based),")
print(f"  not outcome signals (speed-based). These are different classifiers.")

# Finding C: trend is U-shaped, not monotonically declining
early_years = [y for y in range(2005, 2011) if sum(overall[y].values()) >= 50]
mid_years   = [y for y in range(2011, 2015) if sum(overall[y].values()) >= 50]
late_years  = [y for y in range(2015, 2020) if sum(overall[y].values()) >= 50]

def wavg_auto(years):
    pts = [(pct_row(overall[y])[0], pct_row(overall[y])[3]) for y in years]
    pts = [(a, n) for a, n in pts if a is not None and n >= 50]
    if not pts: return None
    return sum(a * n for a, n in pts) / sum(n for _, n in pts)

oa_early = wavg_auto(early_years)
oa_mid   = wavg_auto(mid_years)
oa_late  = wavg_auto(late_years)

print(f"\n  FINDING C -- Trend shape (outcome-signal auto%)")
if oa_early is not None and oa_mid is not None and oa_late is not None:
    print(f"  Early 2005-2010: {oa_early:.1f}%")
    print(f"  Mid   2011-2014: {oa_mid:.1f}%   <-- trough (Spring 3.x/4.x major releases)")
    print(f"  Late  2015-2019: {oa_late:.1f}%")
    if oa_late > oa_mid:
        print(f"  The trend is U-shaped, not monotonically declining.")
        print(f"  The 19%->5% figure in the research covers only the downward segment")
        print(f"  (approximately 2010-2020). A fuller view shows recovery after 2015.")

print(f"\n  RECOMMENDATIONS")
print(f"  ---------------")
print(f"  1. Qualify the trend claim: 'declined during 2010-2014, inconsistent")
print(f"     before and after'. Do not claim a universal monotonic decline.")
print(f"  2. Re-run the time trend using embedding cluster labels (not outcome")
print(f"     signals) -- that is what the published 22/30/48 is based on.")
print(f"  3. Compare Apache trend using the same embedding method to see if")
print(f"     the difference is real or a method artefact.")
print(f"  4. Note the post-2012 git coverage gap as a limitation in the methods paper.")
print()
