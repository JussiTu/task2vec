"""
Apache people map — 101k contributors positioned by project-affiliation features.
Features: [log(camel_comments+1), log(spark_comments+1), log(hadoop_comments+1)]
UMAP → 2D scatter, coloured by dominant project, sized by total comment volume.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re
from collections import defaultdict
import numpy as np
from pymongo import MongoClient
import umap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

KEY_RE  = re.compile(r'author_key\|([a-f0-9\-]+)\|')
PROJ_RE = re.compile(r'^(CAMEL|SPARK|HADOOP)-')
PROJ_IDX = {'CAMEL': 0, 'SPARK': 1, 'HADOOP': 2}

def extract_key(field):
    if not isinstance(field, dict): return None
    m = KEY_RE.search(str(field.get('key', '')))
    return m.group(1) if m else None

# ── 1. scan all tickets ──────────────────────────────────────────────
print("Scanning tickets …")
client = MongoClient('mongodb://localhost:27017/')
col    = client['jiradump']['Apache']

# node_feats[id] = [camel_cnt, spark_cnt, hadoop_cnt]
node_feats = defaultdict(lambda: [0, 0, 0])

scanned = 0
for doc in col.find(
        {'fields.comments': {'$exists': True, '$not': {'$size': 0}}},
        {'key': 1, 'fields.comments': 1}, batch_size=500):
    key  = doc.get('key', '')
    m    = PROJ_RE.match(key)
    if not m:
        continue
    pidx = PROJ_IDX[m.group(1)]
    for c in (doc['fields'].get('comments') or []):
        ak = extract_key(c.get('author'))
        if ak:
            node_feats[ak][pidx] += 1
    scanned += 1
    if scanned % 200_000 == 0:
        print(f"  … {scanned:,} tickets")

print(f"  {len(node_feats):,} unique commenters found")

# ── 2. build feature matrix ──────────────────────────────────────────
ids    = list(node_feats.keys())
X_raw  = np.array([node_feats[i] for i in ids], dtype=np.float32)
totals = X_raw.sum(axis=1)                        # total comment count

# drop commenters with zero CAMEL+SPARK+HADOOP comments (shouldn't exist here)
mask = totals > 0
ids, X_raw, totals = [x[mask] if isinstance(x, np.ndarray) else [v for v, k in zip(x, mask) if k]
                      for x in [ids, X_raw, totals]]
ids    = [ids[i] for i, k in enumerate(mask) if k]
X_raw  = X_raw[mask]
totals = totals[mask]

print(f"  {len(ids):,} commenters after filtering")

X_log = np.log1p(X_raw)

# ── 3. UMAP ─────────────────────────────────────────────────────────
print("Running UMAP …")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    low_memory=True,
    verbose=True
)
embedding = reducer.fit_transform(X_log)
print(f"  embedding shape: {embedding.shape}")

# ── 4. colour by dominant project ────────────────────────────────────
PROJ_NAMES  = ['CAMEL', 'SPARK', 'HADOOP']
PROJ_COLORS = ['#f97316', '#7b68ee', '#10b981']   # orange / purple / green

dominant = np.argmax(X_raw, axis=1)   # 0=CAMEL, 1=SPARK, 2=HADOOP
colors   = [PROJ_COLORS[d] for d in dominant]

# size: log(total) normalised to 1–15
sizes_log  = np.log1p(totals)
sizes_norm = 1.0 + 14.0 * (sizes_log - sizes_log.min()) / (sizes_log.max() - sizes_log.min() + 1e-9)

# ── 5. plot ──────────────────────────────────────────────────────────
print("Plotting …")
fig, ax = plt.subplots(figsize=(18, 16))
fig.patch.set_facecolor('#0d0d0d')
ax.set_facecolor('#0d0d0d')

# draw in order: small dots first, large on top
order = np.argsort(sizes_norm)
emb_o  = embedding[order]
col_o  = [colors[i] for i in order]
siz_o  = sizes_norm[order]

ax.scatter(emb_o[:, 0], emb_o[:, 1],
           s=siz_o,
           c=col_o,
           alpha=0.35,
           linewidths=0)

# overlay: top 50 commenters larger + more opaque
top50_idx = np.argsort(totals)[-50:]
ax.scatter(embedding[top50_idx, 0], embedding[top50_idx, 1],
           s=sizes_norm[top50_idx] * 4,
           c=[colors[i] for i in top50_idx],
           alpha=0.9,
           linewidths=0.4,
           edgecolors='white',
           zorder=5)

# legend
from matplotlib.lines import Line2D
legend_els = [Line2D([0], [0], marker='o', color='w',
                     markerfacecolor=PROJ_COLORS[i], markersize=10,
                     label=PROJ_NAMES[i], linestyle='None')
              for i in range(3)]
ax.legend(handles=legend_els, loc='lower left',
          framealpha=0.2, facecolor='#111', edgecolor='#333',
          labelcolor='white', fontsize=12,
          title='Dominant project', title_fontsize=11)

ax.set_title(
    f'Apache Contributor Map  ·  {len(ids):,} people\n'
    f'Position = project-affiliation similarity (UMAP)  ·  size = comment volume',
    color='#eee', fontsize=13, pad=16)
ax.axis('off')

plt.tight_layout()
out = '.cache/apache_people_map.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
