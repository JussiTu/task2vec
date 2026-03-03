import json, re
from collections import Counter
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
col = client['jiradump']['Apache']
KEY_EXTRACT = re.compile(r'author_key\|([a-f0-9\-]+)\|')

def uid(field):
    if not isinstance(field, dict): return None
    m = KEY_EXTRACT.search(str(field.get('key','')))
    return m.group(1) if m else None

# ---- comments: is text preserved? ----
print('=== COMMENT STRUCTURE ===')
doc = col.find_one({'key': {'$regex': '^SPARK-'}, 'fields.comments': {'$not': {'$size': 0}, '$exists': True}})
if doc:
    cmts = doc['fields'].get('comments', [])
    print(f'  comments count: {len(cmts)}')
    if cmts:
        c = cmts[0]
        print(f'  comment keys: {list(c.keys())}')
        print(f'  body preview: {str(c.get("body",""))[:120]}')
        print(f'  author field: {str(c.get("author",""))[:80]}')

# ---- issuelinks ----
print('\n=== ISSUELINKS ===')
doc2 = col.find_one({'key': {'$regex': '^SPARK-'}, 'fields.issuelinks': {'$not': {'$size': 0}, '$ne': None}})
if doc2:
    links = doc2['fields'].get('issuelinks', []) or []
    print(f'  links count: {len(links)}')
    if links:
        lk = links[0]
        print(f'  link keys: {list(lk.keys())}')
        print(f'  link type: {lk.get("type",{}).get("name","")}')
        print(f'  full link: {str(lk)[:200]}')

# ---- watches / votes ----
print('\n=== WATCHES & VOTES ===')
doc3 = col.find_one({'key': {'$regex': '^SPARK-'}, 'fields.resolutiondate': {'$ne': None}})
if doc3:
    watches = doc3['fields'].get('watches', {})
    votes   = doc3['fields'].get('votes', {})
    print(f'  watches: {watches}')
    print(f'  votes:   {votes}')
    print(f'  priority: {doc3["fields"].get("priority",{}).get("name","")}')
    print(f'  components: {[c.get("name","") for c in (doc3["fields"].get("components") or [])]}')

# ---- reporter vs assignee rate ----
print('\n=== REPORTER != ASSIGNEE RATE (sample 5000 SPARK) ===')
diff_count = 0
same_count = 0
no_reporter = 0
for doc in col.find({'key': {'$regex': '^SPARK-'}, 'fields.resolutiondate': {'$ne': None}},
                    {'fields.reporter': 1, 'fields.assignee': 1}).limit(5000):
    r = uid(doc['fields'].get('reporter'))
    a = uid(doc['fields'].get('assignee'))
    if r is None or a is None:
        no_reporter += 1
    elif r != a:
        diff_count += 1
    else:
        same_count += 1
total = diff_count + same_count
print(f'  reporter != assignee: {diff_count}/{total} ({100*diff_count/total:.0f}%)')
print(f'  reporter == assignee: {same_count}/{total} ({100*same_count/total:.0f}%)')
print(f'  missing:              {no_reporter}')

# ---- comment count distribution ----
print('\n=== COMMENT COUNT DISTRIBUTION (sample 2000 SPARK) ===')
comment_counts = Counter()
for doc in col.find({'key': {'$regex': '^SPARK-'}, 'fields.resolutiondate': {'$ne': None}},
                    {'fields.comments': 1}).limit(2000):
    n = len(doc['fields'].get('comments') or [])
    bucket = 0 if n==0 else (1 if n<=2 else (3 if n<=5 else (6 if n<=10 else 11)))
    comment_counts[bucket] += 1
for b, label in [(0,'0'),(1,'1-2'),(3,'3-5'),(6,'6-10'),(11,'11+')]:
    print(f'  {label:>4} comments: {comment_counts[b]:>5} ({100*comment_counts[b]/2000:.0f}%)')
