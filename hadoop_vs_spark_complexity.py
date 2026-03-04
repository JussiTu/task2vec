"""
Complexity score distribution: Hadoop vs Spark vs Camel.
Uses apache_task_complexity.json cache.
"""

import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── load cache ───────────────────────────────────────────────────────
print("Loading complexity cache …")
with open('.cache/apache_task_complexity.json') as f:
    raw = json.load(f)

tickets = raw.get('ticket_complexity', raw)   # handle both structures
print(f"  {len(tickets):,} tickets")

# ── compute complexity score ─────────────────────────────────────────
# complexity_raw = 0.4×n_modules + 0.3×log(n_files+1) + 0.3×5×core_pct
scores = {}
for key, t in tickets.items():
    proj = t.get('project', key.split('-')[0])
    nm   = t.get('n_modules', 1)
    nf   = t.get('n_files',   1)
    cp   = t.get('core_pct',  0.0)
    raw_score = 0.4 * nm + 0.3 * np.log1p(nf) + 0.3 * 5 * cp
    scores[key] = {'proj': proj, 'raw': raw_score, 'days': t.get('days', 0),
                   'type': t.get('issuetype', '')}

# global normalisation
all_raw = np.array([v['raw'] for v in scores.values()])
mn, mx  = all_raw.min(), all_raw.max()
for v in scores.values():
    v['score'] = (v['raw'] - mn) / (mx - mn + 1e-9)

# split by project
proj_scores = defaultdict(list)
proj_days   = defaultdict(list)
proj_types  = defaultdict(list)
for v in scores.values():
    p = v['proj']
    if p in ('SPARK', 'HADOOP', 'CAMEL'):
        proj_scores[p].append(v['score'])
        if v['days'] > 0:
            proj_days[p].append(min(v['days'], 500))   # cap for viz
        proj_types[p].append(v['type'])

for p in ('SPARK', 'HADOOP', 'CAMEL'):
    arr = np.array(proj_scores[p])
    print(f"\n{p}  n={len(arr):,}")
    print(f"  complexity: mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
          f"p25={np.percentile(arr,25):.3f}  p75={np.percentile(arr,75):.3f}")
    from collections import Counter
    tc = Counter(proj_types[p])
    print(f"  top types: {tc.most_common(4)}")

# ── plot ─────────────────────────────────────────────────────────────
COLORS = {'SPARK': '#7b68ee', 'HADOOP': '#10b981', 'CAMEL': '#f97316'}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d0d0d')
gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # top: overlaid KDE
ax2 = fig.add_subplot(gs[1, 0])   # bottom-left: violin
ax3 = fig.add_subplot(gs[1, 1])   # bottom-right: cumulative distribution

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#111116')
    ax.tick_params(colors='#888', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

# ── panel 1: KDE ─────────────────────────────────────────────────────
from scipy.stats import gaussian_kde

bins = np.linspace(0, 1, 200)
for proj in ['SPARK', 'CAMEL', 'HADOOP']:
    arr = np.array(proj_scores[proj])
    kde = gaussian_kde(arr, bw_method=0.08)
    y   = kde(bins)
    ax1.fill_between(bins, y, alpha=0.25, color=COLORS[proj])
    ax1.plot(bins, y, color=COLORS[proj], lw=2.0, label=f'{proj} (n={len(arr):,})')
    # median line
    med = np.median(arr)
    ax1.axvline(med, color=COLORS[proj], lw=1.0, linestyle='--', alpha=0.7)

ax1.set_title('Complexity Score Distribution (KDE)  —  dashed = median',
              color='#ddd', fontsize=12, pad=10)
ax1.set_xlabel('Normalised complexity score (0 = simplest, 1 = hardest)', color='#888', fontsize=10)
ax1.set_ylabel('Density', color='#888', fontsize=10)
ax1.legend(fontsize=10, framealpha=0.2, facecolor='#111', edgecolor='#333', labelcolor='white')

# ── panel 2: violin ───────────────────────────────────────────────────
vdata  = [proj_scores[p] for p in ('CAMEL', 'SPARK', 'HADOOP')]
vlabels = ['CAMEL', 'SPARK', 'HADOOP']
vcolors = [COLORS[p] for p in vlabels]

parts = ax2.violinplot(vdata, positions=[1, 2, 3], showmedians=True, showextrema=False)
for i, (pc, col) in enumerate(zip(parts['bodies'], vcolors)):
    pc.set_facecolor(col)
    pc.set_alpha(0.6)
parts['cmedians'].set_color('white')
parts['cmedians'].set_linewidth(2)

ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(vlabels, color='#ccc', fontsize=11)
ax2.set_ylabel('Complexity score', color='#888', fontsize=10)
ax2.set_title('Distribution Shape (Violin)', color='#ddd', fontsize=12, pad=10)

# add median annotations
for i, proj in enumerate(vlabels):
    med = np.median(proj_scores[proj])
    p75 = np.percentile(proj_scores[proj], 75)
    ax2.text(i+1, p75 + 0.04, f'med={med:.2f}', ha='center', color='#ccc', fontsize=8)

# ── panel 3: CDF ──────────────────────────────────────────────────────
for proj in ['SPARK', 'CAMEL', 'HADOOP']:
    arr = np.sort(proj_scores[proj])
    cdf = np.arange(1, len(arr)+1) / len(arr)
    ax3.plot(arr, cdf, color=COLORS[proj], lw=2.2, label=proj)

# shade the "low complexity zone" (bottom 30%)
ax3.axhline(0.30, color='#444', lw=0.8, linestyle=':')
ax3.axhline(0.70, color='#444', lw=0.8, linestyle=':')
ax3.text(0.82, 0.31, '30th pct', color='#555', fontsize=8)
ax3.text(0.82, 0.71, '70th pct', color='#555', fontsize=8)

ax3.set_xlabel('Complexity score', color='#888', fontsize=10)
ax3.set_ylabel('Cumulative fraction of tickets', color='#888', fontsize=10)
ax3.set_title('Cumulative Distribution (CDF)', color='#ddd', fontsize=12, pad=10)
ax3.legend(fontsize=10, framealpha=0.2, facecolor='#111', edgecolor='#333', labelcolor='white')

fig.suptitle(
    'Ticket Complexity: Apache Spark vs Hadoop vs Camel\n'
    'Does Hadoop\'s higher Growing rate come from a harder ticket floor?',
    color='#eee', fontsize=13, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = '.cache/hadoop_spark_complexity.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved → {out}")
plt.show()
