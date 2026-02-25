"""
build_search_index.py

Precomputes the search index for the ticket advisor API.
Run once (takes ~30 s for 69k tickets):
  python build_search_index.py
"""
import os, json, sqlite3
import numpy as np
from sklearn.preprocessing import normalize
from dateutil import parser as dateparser

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, '.cache')

# ── Load embeddings ────────────────────────────────────────────────
print('Loading embeddings ...')
data = np.load(os.path.join(CACHE, 'embeddings.npz'), allow_pickle=True)
all_keys = data['keys']
all_vecs = data['vectors']

# Spring = everything except Zookeeper
mask = np.array([not k.startswith('ZOOKEEPER-') for k in all_keys])
keys = all_keys[mask]
vecs = all_vecs[mask].astype(np.float32)
print(f'  Spring tickets: {len(keys):,}')

# L2-normalise so cosine similarity = dot product
vecs_norm = normalize(vecs, norm='l2')

# ── UMAP positions from umap_app_data ─────────────────────────────
print('Building UMAP position lookup ...')
with open(os.path.join(CACHE, 'umap_app_data.json'), encoding='utf-8') as f:
    app_data = json.load(f)

key_pos = {}
for trace in app_data['bg_traces']:
    for i, txt in enumerate(trace['text']):
        k = txt.split(' ')[0]
        key_pos[k] = [round(trace['x'][i], 3), round(trace['y'][i], 3)]
for adata in app_data['assignees'].values():
    for t in adata['tickets']:
        key_pos[t['key']] = [t['x'], t['y']]
print(f'  Keys with UMAP position: {len(key_pos):,}')

# ── Cluster per key ────────────────────────────────────────────────
key_cluster = {}
for adata in app_data['assignees'].values():
    for t in adata['tickets']:
        key_cluster[t['key']] = t['cluster']

# ── Ticket metadata from SQLite ────────────────────────────────────
print('Loading ticket metadata ...')
conn = sqlite3.connect(os.path.join(CACHE, 'tickets.db'))
cur = conn.cursor()
cur.execute('SELECT key, summary, assignee_id, created FROM tickets')
db_meta = {row[0]: {'summary': row[1] or '', 'assignee': row[2] or '', 'created': row[3] or ''}
           for row in cur.fetchall()}
conn.close()

# ── Build aligned metadata list ────────────────────────────────────
print('Assembling metadata ...')
meta = []
for k in keys:
    m = db_meta.get(k, {})
    pos = key_pos.get(k, [None, None])
    year = None
    if m.get('created'):
        try:
            year = dateparser.parse(m['created']).year
        except Exception:
            pass
    meta.append({
        'key':      k,
        'summary':  m.get('summary', '')[:200],
        'assignee': m.get('assignee', ''),
        'year':     year,
        'x':        pos[0],
        'y':        pos[1],
        'cluster':  key_cluster.get(k),
    })

# ── Save ───────────────────────────────────────────────────────────
print('Saving ...')
np.save(os.path.join(CACHE, 'search_index.npy'),  vecs_norm)
np.save(os.path.join(CACHE, 'search_keys.npy'),   np.array(keys))
with open(os.path.join(CACHE, 'search_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, separators=(',', ':'))

print(f'Done.  Index: {vecs_norm.shape}  |  Meta: {len(meta):,} entries')
print(f'  search_index.npy  {os.path.getsize(os.path.join(CACHE,"search_index.npy"))/1e6:.0f} MB')
