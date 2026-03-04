"""
Apache commenting network — community-level co-comment graph.
Step 1: build co-comment graph of top N contributors
Step 2: detect communities (greedy modularity)
Step 3: collapse to community graph, plot
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import re, json
from collections import Counter, defaultdict
from itertools import combinations
from pymongo import MongoClient
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── config ───────────────────────────────────────────────────────────
TOP_N      = 200    # top commenters to include
MIN_WEIGHT = 5      # min co-commented tickets to keep an edge
PROJECT_RE = re.compile(r'^(CAMEL|SPARK|HADOOP)-')
KEY_RE     = re.compile(r'author_key\|([a-f0-9\-]+)\|')

def extract_key(field):
    if not isinstance(field, dict): return None
    m = KEY_RE.search(str(field.get('key', '')))
    return m.group(1) if m else None

# ── 1. find top-N commenters ─────────────────────────────────────────
print("Pass 1: counting commenters …")
client = MongoClient('mongodb://localhost:27017/')
col    = client['jiradump']['Apache']

commenter_counts = Counter()
commenter_projects = defaultdict(Counter)   # node → project → comment count

for doc in col.find(
        {'fields.comments': {'$exists': True, '$not': {'$size': 0}}},
        {'key': 1, 'fields.comments': 1}, batch_size=500):
    key  = doc.get('key', '')
    proj = key.split('-')[0] if PROJECT_RE.match(key) else 'OTHER'
    for c in (doc['fields'].get('comments') or []):
        ak = extract_key(c.get('author'))
        if ak:
            commenter_counts[ak] += 1
            commenter_projects[ak][proj] += 1

top_ids = {k for k, _ in commenter_counts.most_common(TOP_N)}
print(f"  {len(commenter_counts):,} total; keeping top {len(top_ids)}")

# ── 2. co-comment edge weights ───────────────────────────────────────
print("Pass 2: building co-comment edges …")
edge_weight = Counter()
scanned = 0

for doc in col.find(
        {'fields.comments': {'$exists': True, '$not': {'$size': 0}}},
        {'fields.comments': 1}, batch_size=500):
    top_on_ticket = set()
    for c in (doc['fields'].get('comments') or []):
        ak = extract_key(c.get('author'))
        if ak and ak in top_ids:
            top_on_ticket.add(ak)
    if len(top_on_ticket) >= 2:
        for a, b in combinations(sorted(top_on_ticket), 2):
            edge_weight[(a, b)] += 1
    scanned += 1
    if scanned % 200_000 == 0:
        print(f"  … {scanned:,} tickets")

print(f"  raw edges: {len(edge_weight):,}")

# ── 3. build individual graph ────────────────────────────────────────
G = nx.Graph()
for node in top_ids:
    G.add_node(node, comment_count=commenter_counts[node])
for (a, b), w in edge_weight.items():
    if w >= MIN_WEIGHT:
        G.add_edge(a, b, weight=w)

# keep only LCC
lcc_nodes = max(nx.connected_components(G), key=len)
G = G.subgraph(lcc_nodes).copy()
print(f"  LCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  Density: {nx.density(G):.4f}")

# ── 4. community detection ───────────────────────────────────────────
print("Detecting communities …")
communities = nx_comm.greedy_modularity_communities(G, weight='weight')
communities = sorted(communities, key=len, reverse=True)
print(f"  {len(communities)} communities found")

node_to_comm = {}
for i, comm in enumerate(communities):
    for node in comm:
        node_to_comm[node] = i

# dominant project per community
PROJ_COLORS = {'CAMEL': '#f97316', 'SPARK': '#7b68ee', 'HADOOP': '#10b981', 'OTHER': '#666'}
comm_project = {}
comm_comment_total = Counter()
comm_proj_breakdown = defaultdict(Counter)

for i, comm in enumerate(communities):
    proj_votes = Counter()
    for node in comm:
        for proj, cnt in commenter_projects[node].items():
            proj_votes[proj] += cnt
            comm_proj_breakdown[i][proj] += cnt
        comm_comment_total[i] += commenter_counts[node]
    comm_project[i] = proj_votes.most_common(1)[0][0] if proj_votes else 'OTHER'

# ── 5. build community graph ─────────────────────────────────────────
CG = nx.Graph()
for i, comm in enumerate(communities):
    CG.add_node(i,
                size=len(comm),
                dominant_proj=comm_project[i],
                total_comments=comm_comment_total[i],
                proj_breakdown=dict(comm_proj_breakdown[i]))

inter_edges = Counter()
for u, v, data in G.edges(data=True):
    cu, cv = node_to_comm[u], node_to_comm[v]
    if cu != cv:
        key = (min(cu, cv), max(cu, cv))
        inter_edges[key] += data.get('weight', 1)

for (cu, cv), w in inter_edges.items():
    CG.add_edge(cu, cv, weight=w)

# ── 6. print node + edge tables ──────────────────────────────────────
print("\n── Community node table ──")
print(f"{'ID':>3}  {'Members':>7}  {'Comments':>10}  {'Dom.proj':<8}  {'CAMEL%':>7}  {'SPARK%':>7}  {'HADOOP%':>7}")
print("─" * 60)
for i in range(len(communities)):
    total = comm_comment_total[i] or 1
    breakdown = comm_proj_breakdown[i]
    c_pct = 100 * breakdown.get('CAMEL',  0) / total
    s_pct = 100 * breakdown.get('SPARK',  0) / total
    h_pct = 100 * breakdown.get('HADOOP', 0) / total
    print(f"{i:>3}  {len(communities[i]):>7}  {comm_comment_total[i]:>10,}  "
          f"{comm_project[i]:<8}  {c_pct:>6.0f}%  {s_pct:>6.0f}%  {h_pct:>6.0f}%")

print("\n── Top inter-community edges ──")
print(f"{'Comm A':>6}  {'Comm B':>6}  {'Weight':>8}")
print("─" * 28)
for (cu, cv), w in sorted(inter_edges.items(), key=lambda x: -x[1])[:20]:
    print(f"{cu:>6}  {cv:>6}  {w:>8,}")

# modularity
mod = nx_comm.modularity(G, communities, weight='weight')
print(f"\nModularity Q = {mod:.4f}")

# ── 7. plot community graph ──────────────────────────────────────────
print("\nDrawing community graph …")

fig, ax = plt.subplots(figsize=(16, 13))
fig.patch.set_facecolor('#0d0d0d')
ax.set_facecolor('#0d0d0d')

comm_nodes = list(CG.nodes())

# layout
pos = nx.spring_layout(CG, weight='weight',
                        k=3.5 / np.sqrt(len(comm_nodes)),
                        seed=42, iterations=120)

# node sizes: log(total_comments)
sizes_raw  = np.array([CG.nodes[n]['total_comments'] for n in comm_nodes], dtype=float)
sizes_log  = np.log1p(sizes_raw)
sizes_norm = 300 + 6000 * (sizes_log - sizes_log.min()) / (sizes_log.max() - sizes_log.min() + 1e-9)

# node colours: dominant project
node_colors = [PROJ_COLORS.get(CG.nodes[n]['dominant_proj'], '#666') for n in comm_nodes]

# edge widths
edges_cg    = list(CG.edges(data='weight'))
edge_nodes  = [(u, v) for u, v, _ in edges_cg]
ew          = np.array([w for _, _, w in edges_cg], dtype=float)
ew_norm     = 0.3 + 4.0 * (np.log1p(ew) - np.log1p(ew).min()) / (np.log1p(ew).max() - np.log1p(ew).min() + 1e-9)

nx.draw_networkx_edges(CG, pos,
    edgelist=edge_nodes,
    width=ew_norm,
    edge_color='#555555',
    alpha=0.6, ax=ax)

nx.draw_networkx_nodes(CG, pos,
    nodelist=comm_nodes,
    node_size=sizes_norm,
    node_color=node_colors,
    alpha=0.88,
    linewidths=1.0,
    edgecolors='#222222',
    ax=ax)

# labels: community id + member count
labels = {n: f"C{n}\n{CG.nodes[n]['size']} members" for n in comm_nodes}
nx.draw_networkx_labels(CG, pos,
    labels=labels,
    font_size=7.5,
    font_color='#ffffff',
    font_weight='bold',
    ax=ax)

# legend
patches = [mpatches.Patch(color=v, label=k) for k, v in PROJ_COLORS.items() if k != 'OTHER']
ax.legend(handles=patches, loc='lower left',
          framealpha=0.2, facecolor='#111', edgecolor='#333',
          labelcolor='white', fontsize=11, title='Dominant project',
          title_fontsize=10)

ax.set_title(
    f'Apache Comment Network — Community Graph\n'
    f'{len(communities)} communities · {G.number_of_nodes()} contributors · '
    f'modularity Q={mod:.3f}\n'
    f'Node size = comment volume · colour = dominant project',
    color='#eee', fontsize=12, pad=14)
ax.axis('off')

plt.tight_layout()
out = '.cache/apache_comment_network.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
