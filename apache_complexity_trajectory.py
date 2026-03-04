"""
apache_complexity_trajectory.py
================================
Re-runs trajectory classification using per-project-normalised complexity
scores, and compares with the existing global-normalised version and the
resolution-days-based rho already stored in the cache.

Complexity arc sign convention
  rho > +0.2  → engineer takes on HARDER tickets over time  → Growing
  rho < -0.2  → engineer takes on EASIER tickets over time  → Coasting
  |rho| ≤ 0.2 → No change

Resolution-days arc sign convention (opposite, for reference)
  rho < -0.2  → resolves FASTER over time                   → Growing
  rho > +0.2  → resolves SLOWER over time                   → Coasting
"""

import sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from collections import defaultdict
from pathlib import Path
import numpy as np
from pymongo import MongoClient

CACHE      = Path('.cache')
CMPLX_FILE = CACHE / 'apache_task_complexity.json'
SIG_FILE   = CACHE / 'apache_learning_signals.json'
OUT_FILE   = CACHE / 'apache_complexity_trajectory.json'

PROJECTS    = ['CAMEL', 'SPARK', 'HADOOP']
MIN_TICKETS = 15     # minimum matched (complexity-available) tickets per engineer
MIN_TOTAL   = 30     # must also have >=30 total resolved tickets (from signals cache)

KEY_EXTRACT = re.compile(r'author_key\|([a-f0-9\-]+)\|')

# ── helpers ───────────────────────────────────────────────────────────────────

def get_assignee_id(assignee_field):
    if not isinstance(assignee_field, dict):
        return None
    m = KEY_EXTRACT.search(str(assignee_field.get('key', '')))
    return m.group(1) if m else None

def spearman_rho(values):
    n = len(values)
    if n < 4:
        return float('nan')
    ranks = np.argsort(np.argsort(values)).astype(float) + 1
    d2 = sum((i + 1 - r) ** 2 for i, r in enumerate(ranks))
    return 1.0 - 6 * d2 / (n * (n * n - 1))

def label(rho, positive_is_growing=True):
    """Classify a rho value. positive_is_growing=True for complexity arc."""
    if positive_is_growing:
        if rho >  0.2: return 'Growing'
        if rho < -0.2: return 'Coasting'
    else:
        if rho < -0.2: return 'Growing'
        if rho >  0.2: return 'Coasting'
    return 'No change'

def pct_breakdown(labels_list):
    from collections import Counter
    c = Counter(labels_list)
    n = len(labels_list)
    return {k: (c[k], 100*c[k]/n) for k in ['Growing', 'Coasting', 'No change']}

# ── 1. Load & score complexity cache ─────────────────────────────────────────
print('Loading complexity cache …')
with open(CMPLX_FILE, encoding='utf-8') as f:
    raw = json.load(f)

tickets_cx = raw.get('ticket_complexity', raw)
print(f'  {len(tickets_cx):,} tickets in complexity cache')

raw_score = {}   # key → float
proj_of   = {}   # key → project prefix
for key, t in tickets_cx.items():
    proj = t.get('project', key.split('-')[0])
    nm   = t.get('n_modules', 1)
    nf   = t.get('n_files',   1)
    cp   = t.get('core_pct',  0.0)
    raw_score[key] = 0.4 * nm + 0.3 * np.log1p(nf) + 0.3 * 5 * cp
    proj_of[key]   = proj

# ── 2a. GLOBAL normalisation (for comparison) ─────────────────────────────────
all_vals = np.array(list(raw_score.values()))
g_mn, g_mx = all_vals.min(), all_vals.max()
norm_global = {k: (v - g_mn) / (g_mx - g_mn + 1e-9) for k, v in raw_score.items()}

# ── 2b. PER-PROJECT normalisation ─────────────────────────────────────────────
print('\nPer-project normalization ranges:')
proj_min_max = {}
for proj in PROJECTS:
    vals = [v for k, v in raw_score.items() if proj_of.get(k) == proj]
    mn, mx = min(vals), max(vals)
    proj_min_max[proj] = (mn, mx)
    print(f'  {proj}: n={len(vals):,}  raw=[{mn:.4f}, {mx:.4f}]')

norm_perproj = {}
for key, v in raw_score.items():
    p = proj_of.get(key)
    if p in proj_min_max:
        mn, mx = proj_min_max[p]
        norm_perproj[key] = (v - mn) / (mx - mn + 1e-9)
    else:
        norm_perproj[key] = norm_global[key]

# ── 3. Load qualified-engineer UIDs from signals cache ────────────────────────
print('\nLoading existing signals cache …')
with open(SIG_FILE, encoding='utf-8') as f:
    signals = json.load(f)

# uid → rho_days (resolution-days arc)
rho_days_by_proj = {}
qualified_uids   = {}
for proj in PROJECTS:
    rho_days_by_proj[proj] = {}
    qualified_uids[proj]   = set()
    for eng in signals.get(proj, {}).get('engineers', []):
        if eng.get('rho') is not None:
            rho_days_by_proj[proj][eng['uid']] = eng['rho']
            qualified_uids[proj].add(eng['uid'])
    print(f'  {proj}: {len(qualified_uids[proj]):,} engineers with rho_days')

# ── 4. Query MongoDB: engineer → ordered (date, key, complexity) ──────────────
print('\nQuerying MongoDB for tickets with complexity data …')
client = MongoClient('mongodb://localhost:27017/')
col    = client['jiradump']['Apache']

eng_tix = {}   # proj → uid → [(date, global_score, perproj_score)]
for proj in PROJECTS:
    uid_set = qualified_uids[proj]
    eng_tix[proj] = defaultdict(list)

    cursor = col.find(
        {
            'key':                     {'$regex': f'^{proj}-'},
            'fields.resolutiondate':   {'$ne': None},
        },
        {'key': 1, 'fields.assignee': 1, 'fields.resolutiondate': 1},
        batch_size=500
    )

    found = 0
    for doc in cursor:
        fields = doc.get('fields', {})
        uid    = get_assignee_id(fields.get('assignee'))
        if uid not in uid_set:
            continue
        rd  = fields.get('resolutiondate', '')
        if not rd or len(rd) < 10:
            continue
        key = doc['key']
        if key not in norm_global:
            continue
        eng_tix[proj][uid].append(
            (rd[:10], norm_global[key], norm_perproj[key])
        )
        found += 1

    # sort each engineer's tickets by date
    for uid in eng_tix[proj]:
        eng_tix[proj][uid].sort(key=lambda x: x[0])

    total_matched = sum(len(v) for v in eng_tix[proj].values())
    print(f'  {proj}: {total_matched:,} matched tickets across '
          f'{len(eng_tix[proj]):,} engineers')

# ── 5. Compute complexity rho (global & per-project) per engineer ─────────────
print('\nComputing complexity arc rho …')

# results per engineer
per_eng = {}   # proj → uid → {rho_global, rho_perproj, rho_days, n_matched}
for proj in PROJECTS:
    per_eng[proj] = {}
    for uid, tix in eng_tix[proj].items():
        if len(tix) < MIN_TICKETS:
            continue
        g_vals  = [t[1] for t in tix]
        pp_vals = [t[2] for t in tix]
        rho_g  = spearman_rho(g_vals)
        rho_pp = spearman_rho(pp_vals)
        per_eng[proj][uid] = {
            'rho_global':   round(rho_g,  4) if not np.isnan(rho_g)  else None,
            'rho_perproj':  round(rho_pp, 4) if not np.isnan(rho_pp) else None,
            'rho_days':     rho_days_by_proj[proj].get(uid),
            'n_matched':    len(tix),
        }
    print(f'  {proj}: {len(per_eng[proj]):,} engineers with enough matched tickets')

# ── 6. Summary table ──────────────────────────────────────────────────────────
print('\n' + '='*70)
print('TRAJECTORY COMPARISON  (Growing / Coasting / No change)')
print('  Complexity arc: rho >+0.2 → Growing, rho <-0.2 → Coasting')
print('  Resolution arc: rho <-0.2 → Growing, rho >+0.2 → Coasting')
print('='*70)

output_data = {}
for proj in PROJECTS:
    engs = per_eng[proj]
    if not engs:
        print(f'\n{proj}: no data')
        continue

    rg = [e['rho_global']  for e in engs.values() if e['rho_global']  is not None]
    rp = [e['rho_perproj'] for e in engs.values() if e['rho_perproj'] is not None]
    rd = [e['rho_days']    for e in engs.values() if e['rho_days']    is not None]

    def breakdown(rho_list, pos_growing):
        labels = [label(r, pos_growing) for r in rho_list]
        return pct_breakdown(labels)

    bk_g  = breakdown(rg, True)
    bk_pp = breakdown(rp, True)
    bk_d  = breakdown(rd, False)

    print(f'\n{proj}  (n engineers: global={len(rg)}, per-proj={len(rp)}, days={len(rd)})')

    for method, bk, n in [
        ('Resolution-days      ', bk_d,  len(rd)),
        ('Complexity-global    ', bk_g,  len(rg)),
        ('Complexity-per-proj  ', bk_pp, len(rp)),
    ]:
        g = bk['Growing'];   c = bk['Coasting'];  nc = bk['No change']
        print(f'  {method}  '
              f'Growing={g[0]:3d}({g[1]:4.1f}%)  '
              f'Coasting={c[0]:3d}({c[1]:4.1f}%)  '
              f'No change={nc[0]:3d}({nc[1]:4.1f}%)')

    # agreement: per-proj vs resolution-days
    both_pp_d = [(uid, e['rho_perproj'], e['rho_days'])
                 for uid, e in engs.items()
                 if e['rho_perproj'] is not None and e['rho_days'] is not None]
    if both_pp_d:
        agree = sum(
            1 for _, rpp, rd_ in both_pp_d
            if label(rpp, True) == label(rd_, False)
        )
        print(f'  Agreement (per-proj complexity vs days): '
              f'{agree}/{len(both_pp_d)} ({100*agree/len(both_pp_d):.1f}%)')

    output_data[proj] = {
        'resolution_days': {uid: e['rho_days']   for uid, e in engs.items()},
        'complexity_global':  {uid: e['rho_global']  for uid, e in engs.items()},
        'complexity_perproj': {uid: e['rho_perproj'] for uid, e in engs.items()},
        'n_matched_tickets':  {uid: e['n_matched']   for uid, e in engs.items()},
    }

# ── 7. Save ───────────────────────────────────────────────────────────────────
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2)
print(f'\nSaved → {OUT_FILE}')
