"""
SPARK contributor social network.
Nodes = contributors with 20+ SPARK comments.
Edges = co-commented on same SPARK ticket (weight = # shared tickets).
Layout = spring with weight so heavy collaborators pull together.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re
from collections import Counter, defaultdict
from itertools import combinations
from pymongo import MongoClient
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import json

KEY_RE = re.compile(r'author_key\|([a-f0-9\-]+)\|')
MIN_COMMENTS = 20    # min SPARK comments to include as a node
MIN_EDGE_W   = 2     # min co-commented tickets to draw an edge

# trajectory colours
TRAJ_COLOR = {
    'Growing':   '#10b981',   # green
    'Coasting':  '#f59e0b',   # amber
    'No change': '#444455',   # dark grey-blue
    'Unknown':   '#222233',   # very dim (no trajectory data)
}

# load trajectory labels from learning signals cache
def load_trajectories():
    with open('.cache/apache_learning_signals.json') as f:
        data = json.load(f)
    traj = {}
    for eng in data.get('SPARK', {}).get('engineers', []):
        rho = eng.get('rho', 0)
        if rho > 0.2:
            label = 'Growing'
        elif rho < -0.2:
            label = 'Coasting'
        else:
            label = 'No change'
        traj[eng['uid']] = label
    return traj

def extract_key(f):
    if not isinstance(f, dict): return None
    m = KEY_RE.search(str(f.get('key', '')))
    return m.group(1) if m else None

# ── 0. load trajectories ────────────────────────────────────────────
trajectories = load_trajectories()
print(f"Loaded {len(trajectories)} SPARK trajectory labels")

# ── 1. find qualifying nodes ─────────────────────────────────────────
print("Pass 1: finding SPARK contributors …")
client = MongoClient('mongodb://localhost:27017/')
col    = client['jiradump']['Apache']

spark_counts = Counter()
for doc in col.find(
        {'key': {'$regex': '^SPARK-'},
         'fields.comments': {'$exists': True, '$not': {'$size': 0}}},
        {'fields.comments': 1}, batch_size=500):
    for c in (doc['fields'].get('comments') or []):
        ak = extract_key(c.get('author'))
        if ak:
            spark_counts[ak] += 1

node_set = {k for k, v in spark_counts.items() if v >= MIN_COMMENTS}
print(f"  {len(node_set)} nodes (>= {MIN_COMMENTS} SPARK comments)")

# ── 2. build co-comment edges ────────────────────────────────────────
print("Pass 2: building edges …")
edge_weight = Counter()
ticket_count = 0

for doc in col.find(
        {'key': {'$regex': '^SPARK-'},
         'fields.comments': {'$exists': True, '$not': {'$size': 0}}},
        {'fields.comments': 1}, batch_size=500):
    commenters = set()
    for c in (doc['fields'].get('comments') or []):
        ak = extract_key(c.get('author'))
        if ak and ak in node_set:
            commenters.add(ak)
    if len(commenters) >= 2:
        for a, b in combinations(sorted(commenters), 2):
            edge_weight[(a, b)] += 1
    ticket_count += 1

print(f"  scanned {ticket_count:,} SPARK tickets")
print(f"  raw edges: {len(edge_weight):,}")

# ── 3. build graph ───────────────────────────────────────────────────
G = nx.Graph()
for node in node_set:
    G.add_node(node, comments=spark_counts[node])

for (a, b), w in edge_weight.items():
    if w >= MIN_EDGE_W:
        G.add_edge(a, b, weight=w)

# drop isolates
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)
print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges after filtering")
print(f"  Density: {nx.density(G):.4f}")

# keep LCC
lcc_nodes = max(nx.connected_components(G), key=len)
G = G.subgraph(lcc_nodes).copy()
print(f"  LCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 4. metrics ───────────────────────────────────────────────────────
deg_cent = nx.degree_centrality(G)
btw_cent = nx.betweenness_centrality(G, weight='weight', normalized=True)

print("\nTop 10 by betweenness (bridge nodes):")
for nid, val in sorted(btw_cent.items(), key=lambda x: -x[1])[:10]:
    print(f"  {nid[:8]}...  btw={val:.4f}  comments={spark_counts[nid]:,}")

# ── 5. layout ────────────────────────────────────────────────────────
print("\nComputing spring layout …")
# invert weight so higher weight = shorter spring = closer nodes
for u, v, d in G.edges(data=True):
    d['inv_weight'] = 1.0 / (d['weight'] ** 0.6)

pos = nx.spring_layout(
    G,
    weight='inv_weight',
    k=1.8 / np.sqrt(G.number_of_nodes()),
    seed=42,
    iterations=120
)

# ── 6. plot ──────────────────────────────────────────────────────────
print("Plotting …")
fig, ax = plt.subplots(figsize=(20, 18))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')

node_list = list(G.nodes())
n = len(node_list)

# node sizes: log(comment count)
sizes_raw  = np.array([spark_counts[nd] for nd in node_list], dtype=float)
sizes_log  = np.log1p(sizes_raw)
sizes_norm = 20 + 800 * (sizes_log - sizes_log.min()) / (sizes_log.max() - sizes_log.min() + 1e-9)

# node colour: trajectory (Growing=green / Coasting=amber / No change=grey / Unknown=dim)
node_colors = [TRAJ_COLOR.get(trajectories.get(nd, 'Unknown')) for nd in node_list]

# count coverage
traj_counts = {'Growing': 0, 'Coasting': 0, 'No change': 0, 'Unknown': 0}
for nd in node_list:
    traj_counts[trajectories.get(nd, 'Unknown')] += 1
print(f"Node trajectory coverage: {traj_counts}")

# edges: colour and width by weight
edges_list = list(G.edges(data=True))
ew_raw     = np.array([d['weight'] for _, _, d in edges_list], dtype=float)
ew_log     = np.log1p(ew_raw)
ew_norm    = 0.15 + 2.5 * (ew_log - ew_log.min()) / (ew_log.max() - ew_log.min() + 1e-9)
# colour edges by weight: faint grey → bright teal
ec_norm    = mcolors.Normalize(vmin=ew_raw.min(), vmax=np.percentile(ew_raw, 95))
edge_cmap  = plt.cm.YlGnBu
edge_colors = [edge_cmap(ec_norm(d['weight'])) for _, _, d in edges_list]
edge_pairs  = [(u, v) for u, v, _ in edges_list]

nx.draw_networkx_edges(G, pos,
    edgelist=edge_pairs,
    width=ew_norm,
    edge_color=edge_colors,
    alpha=0.45,
    ax=ax)

# draw nodes sorted small → large
order = np.argsort(sizes_norm)
sorted_nodes  = [node_list[i] for i in order]
sorted_sizes  = sizes_norm[order]
sorted_colors = [node_colors[i] for i in order]

nx.draw_networkx_nodes(G, pos,
    nodelist=sorted_nodes,
    node_size=sorted_sizes,
    node_color=sorted_colors,
    alpha=0.90,
    linewidths=0.3,
    edgecolors='#111111',
    ax=ax)

# label top-15 by betweenness with rank numbers
top15_ranked = sorted(btw_cent.items(), key=lambda x: -x[1])[:15]
labels = {nd: f'#{i+1}' for i, (nd, _) in enumerate(top15_ranked)}
nx.draw_networkx_labels(G, pos,
    labels=labels,
    font_size=7,
    font_color='#ffffff',
    font_weight='bold',
    ax=ax)

# legend: trajectory colours
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=TRAJ_COLOR['Growing'],
           markersize=12, label=f"Growing ({traj_counts['Growing']})", linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=TRAJ_COLOR['Coasting'],
           markersize=12, label=f"Coasting ({traj_counts['Coasting']})", linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=TRAJ_COLOR['No change'],
           markersize=12, label=f"No change ({traj_counts['No change']})", linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=TRAJ_COLOR['Unknown'],
           markersize=12, label=f"No data ({traj_counts['Unknown']})", linestyle='None'),
]
ax.legend(handles=legend_els, loc='lower left',
          framealpha=0.25, facecolor='#111', edgecolor='#333',
          labelcolor='white', fontsize=11,
          title='Career trajectory', title_fontsize=11)

ax.set_title(
    f'Apache Spark  —  Contributor Communication Network\n'
    f'{G.number_of_nodes()} people  ·  {G.number_of_edges()} connections  '
    f'·  density {nx.density(G):.3f}\n'
    f'Proximity = communication volume  ·  colour = career trajectory  ·  size = comment volume',
    color='#ddd', fontsize=12, pad=16)
ax.axis('off')

plt.tight_layout()
out = '.cache/spark_social_network.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
