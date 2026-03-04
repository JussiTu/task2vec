"""
Two side-by-side SPARK network plots:
  Left:  node size = comments MADE (how much this person talks)
  Right: node size = comments RECEIVED on their assigned tickets
         (how much their work is being discussed)
Same layout, same edges, same trajectory colours.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re, json
from collections import Counter, defaultdict
from itertools import combinations
from pymongo import MongoClient
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

KEY_RE = re.compile(r'author_key\|([a-f0-9\-]+)\|')
MIN_COMMENTS = 20
MIN_EDGE_W   = 2

TRAJ_COLOR = {
    'Growing':   '#10b981',
    'Coasting':  '#f59e0b',
    'No change': '#444455',
    'Unknown':   '#1e1e2e',
}

def extract_key(f):
    if not isinstance(f, dict): return None
    m = KEY_RE.search(str(f.get('key', '')))
    return m.group(1) if m else None

# ── 0. trajectories ──────────────────────────────────────────────────
with open('.cache/apache_learning_signals.json') as f:
    data = json.load(f)
trajectories = {}
for eng in data.get('SPARK', {}).get('engineers', []):
    rho = eng.get('rho', 0)
    trajectories[eng['uid']] = ('Growing' if rho > 0.2
                                 else 'Coasting' if rho < -0.2
                                 else 'No change')

# ── 1. scan: comments made + comments received ───────────────────────
print("Scanning SPARK tickets …")
client  = MongoClient('mongodb://localhost:27017/')
col     = client['jiradump']['Apache']

comments_made     = Counter()   # uid → comments left by this person
comments_received = Counter()   # uid → comments left on their assigned tickets
edge_weight       = Counter()

for doc in col.find(
        {'key': {'$regex': '^SPARK-'},
         'fields.comments': {'$exists': True, '$not': {'$size': 0}}},
        {'fields.comments': 1, 'fields.assignee': 1}, batch_size=500):

    cmts     = doc['fields'].get('comments') or []
    assignee = extract_key(doc['fields'].get('assignee'))

    commenters_on_ticket = set()
    for c in cmts:
        ak = extract_key(c.get('author'))
        if ak:
            comments_made[ak] += 1
            commenters_on_ticket.add(ak)
            # count as received by assignee (if different person)
            if assignee and ak != assignee:
                comments_received[assignee] += 1

    # co-comment edges
    top_on_ticket = commenters_on_ticket & {k for k, v in comments_made.items() if v >= MIN_COMMENTS}
    if len(top_on_ticket) >= 2:
        for a, b in combinations(sorted(top_on_ticket), 2):
            edge_weight[(a, b)] += 1

node_set = {k for k, v in comments_made.items() if v >= MIN_COMMENTS}
print(f"  {len(node_set)} qualifying nodes")

# ── 2. build graph ───────────────────────────────────────────────────
G = nx.Graph()
for nd in node_set:
    G.add_node(nd)
for (a, b), w in edge_weight.items():
    if w >= MIN_EDGE_W:
        G.add_edge(a, b, weight=w)
G.remove_nodes_from(list(nx.isolates(G)))
lcc = max(nx.connected_components(G), key=len)
G   = G.subgraph(lcc).copy()

node_list = list(G.nodes())
print(f"  LCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 3. layout (once, shared) ─────────────────────────────────────────
print("Computing layout …")
for u, v, d in G.edges(data=True):
    d['inv_weight'] = 1.0 / (d['weight'] ** 0.6)
pos = nx.spring_layout(G, weight='inv_weight',
                        k=1.8/np.sqrt(G.number_of_nodes()),
                        seed=42, iterations=120)

# ── 4. centrality for labels ─────────────────────────────────────────
btw_cent  = nx.betweenness_centrality(G, weight='weight', normalized=True)
top15     = sorted(btw_cent.items(), key=lambda x: -x[1])[:15]
rank_labels = {nd: f'#{i+1}' for i, (nd, _) in enumerate(top15)}

node_colors = [TRAJ_COLOR.get(trajectories.get(nd, 'Unknown')) for nd in node_list]

edges_list  = list(G.edges(data=True))
edge_pairs  = [(u, v) for u, v, _ in edges_list]
ew_raw      = np.array([d['weight'] for _, _, d in edges_list], dtype=float)
ew_log      = np.log1p(ew_raw)
ew_norm     = 0.1 + 1.8 * (ew_log - ew_log.min()) / (ew_log.max() - ew_log.min() + 1e-9)
ec_norm     = plt.Normalize(vmin=ew_raw.min(), vmax=np.percentile(ew_raw, 95))
edge_colors = [plt.cm.YlGnBu(ec_norm(d['weight'])) for _, _, d in edges_list]

# ── 5. helper: draw one panel ────────────────────────────────────────
def draw_panel(ax, title, metric_counter, metric_label):
    ax.set_facecolor('#0a0a0f')

    raw  = np.array([metric_counter.get(nd, 0) for nd in node_list], dtype=float)
    lraw = np.log1p(raw)
    sz   = 15 + 900 * (lraw - lraw.min()) / (lraw.max() - lraw.min() + 1e-9)

    # draw small nodes first
    order = np.argsort(sz)
    nx.draw_networkx_edges(G, pos,
        edgelist=edge_pairs, width=ew_norm,
        edge_color=edge_colors, alpha=0.35, ax=ax)

    nx.draw_networkx_nodes(G, pos,
        nodelist=[node_list[i] for i in order],
        node_size=sz[order],
        node_color=[node_colors[i] for i in order],
        alpha=0.90, linewidths=0.3, edgecolors='#111',
        ax=ax)

    nx.draw_networkx_labels(G, pos,
        labels={nd: lbl for nd, lbl in rank_labels.items() if nd in G.nodes()},
        font_size=6.5, font_color='#fff', font_weight='bold', ax=ax)

    # stats per trajectory
    stats = {}
    for grp in ['Growing', 'Coasting', 'No change']:
        nodes = [nd for nd in node_list if trajectories.get(nd, 'Unknown') == grp]
        vals  = [metric_counter.get(nd, 0) for nd in nodes]
        stats[grp] = int(np.mean(vals)) if vals else 0

    # inset text
    info = '\n'.join([f"{g}: avg {v:,}" for g, v in stats.items()])
    ax.text(0.02, 0.02, info, transform=ax.transAxes,
            fontsize=7.5, color='#aaa', verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#111', alpha=0.7))

    ax.set_title(f'{title}\n(size = {metric_label})',
                 color='#ddd', fontsize=11, pad=10)
    ax.axis('off')

# ── 6. figure ────────────────────────────────────────────────────────
print("Plotting …")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 16))
fig.patch.set_facecolor('#0a0a0f')

draw_panel(ax1, 'Comments Made', comments_made,     'comments left by person')
draw_panel(ax2, 'Comments Received', comments_received, 'comments on their assigned tickets')

# shared legend
legend_els = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=TRAJ_COLOR['Growing'],
           markersize=11, label='Growing', linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=TRAJ_COLOR['Coasting'],
           markersize=11, label='Coasting', linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=TRAJ_COLOR['No change'],
           markersize=11, label='No change', linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=TRAJ_COLOR['Unknown'],
           markersize=11, label='No trajectory data', linestyle='None'),
]
fig.legend(handles=legend_els, loc='lower center', ncol=4,
           framealpha=0.2, facecolor='#111', edgecolor='#333',
           labelcolor='white', fontsize=11, title='Career trajectory',
           title_fontsize=11, bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    'Apache Spark — Communication Network\n'
    '#1–#15 = top bridge nodes by betweenness centrality  ·  '
    'same layout, same edges, different node sizing',
    color='#eee', fontsize=13, y=0.98)

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
out = '.cache/spark_made_vs_received.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
