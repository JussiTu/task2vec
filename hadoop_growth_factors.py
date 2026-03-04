"""
hadoop_growth_factors.py
========================
Comprehensive multi-factor comparison: what drives higher Growth rates in Hadoop?

Uses 'Growing' = rho_days > +0.2 (network-plot convention: positive rho means
resolution time rising → engineer is taking on progressively harder work).

Data sources:
  .cache/apache_learning_signals.json  — rho, delta_cov, span, first_year
  .cache/apache_task_complexity.json   — early/late files/modules/days, arcs
  .cache/apache_velocity_scope.json    — type_counts, median_days, scope metrics
"""

import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu

CACHE = Path('.cache')

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading caches …')
with open(CACHE / 'apache_learning_signals.json', encoding='utf-8') as f:
    signals = json.load(f)
with open(CACHE / 'apache_task_complexity.json', encoding='utf-8') as f:
    cx_data = json.load(f)
with open(CACHE / 'apache_velocity_scope.json', encoding='utf-8') as f:
    vel_scope = json.load(f)

# ── Build engineer feature matrix ─────────────────────────────────────────────
# index complexity arcs by uid
arc_by_uid = {a['uid']: a for a in cx_data.get('engineer_arcs', [])}
# index velocity/scope by uid
vs_by_uid  = {v['uid']: v for v in vel_scope}

PROJECTS = ['CAMEL', 'SPARK', 'HADOOP']

def trajectory(rho):
    if rho is None: return 'Unknown'
    if rho >  0.2: return 'Growing'
    if rho < -0.2: return 'Coasting'
    return 'No change'

records = []
for proj in PROJECTS:
    for eng in signals[proj]['engineers']:
        uid  = eng['uid']
        rho  = eng.get('rho')
        traj = trajectory(rho)
        arc  = arc_by_uid.get(uid, {})
        vs   = vs_by_uid.get(uid, {})

        # ticket type ratios
        tc     = vs.get('type_counts', {})
        n_tix  = eng['n']
        bug_r  = tc.get('Bug', 0)        / n_tix if n_tix else 0
        imp_r  = tc.get('Improvement', 0)/ n_tix if n_tix else 0
        sub_r  = tc.get('Sub-task', 0)   / n_tix if n_tix else 0
        new_r  = tc.get('New Feature', 0)/ n_tix if n_tix else 0

        # files arc delta
        ef = arc.get('early_files', np.nan)
        lf = arc.get('late_files',  np.nan)
        file_delta = lf - ef if (not np.isnan(ef) and not np.isnan(lf)) else np.nan

        # core pct arc delta
        ec = arc.get('early_core', np.nan)
        lc = arc.get('late_core',  np.nan)
        core_delta = lc - ec if (not np.isnan(ec) and not np.isnan(lc)) else np.nan

        # days arc delta
        ed = arc.get('early_days', np.nan)
        ld = arc.get('late_days',  np.nan)
        days_delta = ld - ed if (not np.isnan(ed) and not np.isnan(ld)) else np.nan

        records.append({
            'uid':         uid,
            'project':     proj,
            'trajectory':  traj,
            'rho':         rho,
            # career
            'n':           n_tix,
            'span_years':  eng.get('span_years', 0),
            'first_year':  eng.get('first_year', 0),
            # git coverage
            'early_cov':   eng.get('early_cov', np.nan),
            'late_cov':    eng.get('late_cov',  np.nan),
            'delta_cov':   eng.get('delta_cov', np.nan),
            # ticket types
            'bug_rate':    bug_r,
            'imp_rate':    imp_r,
            'sub_rate':    sub_r,
            'new_rate':    new_r,
            # complexity arc
            'early_files': ef,
            'late_files':  lf,
            'file_delta':  file_delta,
            'early_core':  ec,
            'late_core':   lc,
            'core_delta':  core_delta,
            'frho':        arc.get('frho', np.nan),
            'mrho':        arc.get('mrho', np.nan),
            'crho':        arc.get('crho', np.nan),
            # velocity / scope
            'early_days':  ed,
            'late_days':   ld,
            'days_delta':  days_delta,
            'median_days': vs.get('median_days', np.nan),
            'scope_delta': vs.get('scope_delta', np.nan),
            'srho':        vs.get('srho', np.nan),
        })

print(f'  {len(records)} engineers total')
for p in PROJECTS:
    sub  = [r for r in records if r['project'] == p]
    grow = [r for r in sub if r['trajectory'] == 'Growing']
    print(f'  {p}: {len(sub)} engineers, {len(grow)} Growing ({100*len(grow)/len(sub):.1f}%)')

# ── Helpers ───────────────────────────────────────────────────────────────────

def vals(recs, key):
    return [r[key] for r in recs if r[key] is not None and not
            (isinstance(r[key], float) and np.isnan(r[key]))]

def mwu_p(a, b):
    """Mann-Whitney U p-value; returns 1.0 if insufficient data."""
    if len(a) < 3 or len(b) < 3: return 1.0
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    return p

def sig_star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

# ── Console summary ───────────────────────────────────────────────────────────
print('\n' + '='*72)
print('FACTOR ANALYSIS: Growing vs Non-Growing, by project')
print('  p-values: Mann-Whitney U  (*p<.05  **p<.01  ***p<.001)')
print('='*72)

METRICS = [
    ('n',           'Tickets (total)',           False),
    ('span_years',  'Career span (years)',        False),
    ('first_year',  'First year',                 False),
    ('delta_cov',   'Delta git-coverage',         False),
    ('early_cov',   'Early git-coverage',         False),
    ('late_cov',    'Late git-coverage',          False),
    ('bug_rate',    'Bug ratio',                  True),
    ('imp_rate',    'Improvement ratio',          True),
    ('sub_rate',    'Sub-task ratio',             True),
    ('new_rate',    'New Feature ratio',          True),
    ('median_days', 'Median resolution days',     False),
    ('early_days',  'Early career days',          False),
    ('late_days',   'Late career days',           False),
    ('days_delta',  'Days delta (late-early)',     False),
    ('early_files', 'Early career files/commit',  False),
    ('late_files',  'Late career files/commit',   False),
    ('file_delta',  'Files delta (late-early)',    False),
    ('frho',        'File-complexity arc (frho)', False),
    ('mrho',        'Module-complexity arc (mrho)',False),
    ('crho',        'Core-pct arc (crho)',        False),
    ('scope_delta', 'Scope delta',                False),
]

for proj in PROJECTS:
    sub   = [r for r in records if r['project'] == proj]
    grow  = [r for r in sub if r['trajectory'] == 'Growing']
    ngrow = [r for r in sub if r['trajectory'] != 'Growing']
    print(f'\n{proj}  (Growing n={len(grow)}, Other n={len(ngrow)})')
    print(f'  {"Metric":<30} {"Growing":>10} {"Other":>10}  p')
    for key, label, is_pct in METRICS:
        g_v  = vals(grow,  key)
        ng_v = vals(ngrow, key)
        if not g_v or not ng_v: continue
        gm   = np.median(g_v)
        ngm  = np.median(ng_v)
        p    = mwu_p(g_v, ng_v)
        star = sig_star(p)
        fmt  = '{:.1%}' if is_pct else '{:.2f}'
        print(f'  {label:<30} {fmt.format(gm):>10} {fmt.format(ngm):>10}  '
              f'{p:.3f}{star}')

# ── Cross-project comparison of Growing engineers ─────────────────────────────
print('\n' + '='*72)
print('GROWING ENGINEERS COMPARED ACROSS PROJECTS')
print('  (what makes Hadoop Growing engineers different?)')
print('='*72)

growing = {p: [r for r in records if r['project'] == p and r['trajectory'] == 'Growing']
           for p in PROJECTS}

print(f'\n  {"Metric":<30} {"CAMEL":>10} {"SPARK":>10} {"HADOOP":>10}  H vs S (p)')
for key, label, is_pct in METRICS:
    rows = {p: vals(growing[p], key) for p in PROJECTS}
    if not all(rows.values()): continue
    fmt = '{:.1%}' if is_pct else '{:.2f}'
    p_hs = mwu_p(rows['HADOOP'], rows['SPARK'])
    star = sig_star(p_hs)
    meds = {p: np.median(v) if v else float('nan') for p, v in rows.items()}
    print(f'  {label:<30} '
          f'{fmt.format(meds["CAMEL"]):>10} '
          f'{fmt.format(meds["SPARK"]):>10} '
          f'{fmt.format(meds["HADOOP"]):>10}  '
          f'{p_hs:.3f}{star}')

# ── Plot ──────────────────────────────────────────────────────────────────────
print('\nPlotting …')

PROJ_COLOR = {'CAMEL': '#f97316', 'SPARK': '#7b68ee', 'HADOOP': '#10b981'}
DARK_BG    = '#0d0d0d'
PANEL_BG   = '#111116'

PLOT_METRICS = [
    ('n',           'Tickets',                  False),
    ('span_years',  'Career span (yrs)',         False),
    ('first_year',  'First year',                False),
    ('delta_cov',   'Δ git coverage',            False),
    ('bug_rate',    'Bug ratio',                 True),
    ('imp_rate',    'Improvement ratio',         True),
    ('median_days', 'Median days',               False),
    ('days_delta',  'Days Δ (late−early)',        False),
    ('early_files', 'Files/commit (early)',       False),
    ('late_files',  'Files/commit (late)',        False),
    ('file_delta',  'Files Δ (late−early)',       False),
    ('frho',        'File-complexity arc ρ',     False),
]

NCOLS = 4
NROWS = 3
fig, axes_arr = plt.subplots(NROWS, NCOLS, figsize=(22, 10))
fig.patch.set_facecolor(DARK_BG)
plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.11,
                    hspace=0.65, wspace=0.38)
axes = axes_arr.flatten()

for ax, (key, lbl, is_pct) in zip(axes, PLOT_METRICS):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors='#888', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    x_pos  = 0
    xticks = []
    xlbls  = []

    for proj in PROJECTS:
        sub   = [r for r in records if r['project'] == proj]
        grow  = vals([r for r in sub if r['trajectory'] == 'Growing'],  key)
        ngrow = vals([r for r in sub if r['trajectory'] != 'Growing'], key)

        col = PROJ_COLOR[proj]

        for grp_vals, hatch, label_suffix in [
            (ngrow, '//',  'Other'),
            (grow,  '',   'Growing'),
        ]:
            if grp_vals:
                bp = ax.boxplot(
                    grp_vals,
                    positions=[x_pos],
                    widths=0.55,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#666'),
                    capprops=dict(color='#666'),
                    boxprops=dict(facecolor=col,
                                  alpha=0.85 if not hatch else 0.30,
                                  edgecolor=col,
                                  hatch=hatch),
                )
            xticks.append(x_pos)
            xlbls.append(f'{proj[:1]}\n{label_suffix[:3]}')
            x_pos += 1
        x_pos += 0.6   # gap between projects

        # significance annotation
        p = mwu_p(grow, ngrow)
        star = sig_star(p)
        if star:
            mid = x_pos - 1.3
            all_v = grow + ngrow
            top   = np.percentile(all_v, 90) if all_v else 1
            ax.text(mid, top * 1.05, star, ha='center', color='#ffdd88',
                    fontsize=9, fontweight='bold')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlbls, color='#999', fontsize=7)
    ax.set_title(lbl, color='#ddd', fontsize=9, pad=6)
    if is_pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

# legend
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor='#aaa', edgecolor='#aaa', label='Non-Growing'),
    Patch(facecolor='#aaa', edgecolor='#aaa', alpha=0.85, label='Growing'),
    Patch(facecolor=PROJ_COLOR['CAMEL'],  label='CAMEL'),
    Patch(facecolor=PROJ_COLOR['SPARK'],  label='SPARK'),
    Patch(facecolor=PROJ_COLOR['HADOOP'], label='HADOOP'),
]
fig.legend(handles=legend_els, loc='lower center', ncol=5,
           framealpha=0.2, facecolor='#111', edgecolor='#333',
           labelcolor='white', fontsize=10, bbox_to_anchor=(0.5, 0.005))

fig.suptitle(
    'What drives Growth?  —  Growing vs Non-Growing engineers by project\n'
    'Solid = Growing, Hatched = Other  ·  * p<.05  ** p<.01  *** p<.001',
    color='#eee', fontsize=13, y=0.97)

# subplots_adjust already called above
out = CACHE / 'hadoop_growth_factors.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved → {out}')
plt.show()
