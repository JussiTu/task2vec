import json
from collections import Counter

path = r'C:\Users\jussi\ai-driving-license\.cache\apache_git_index.json'
with open(path, encoding='utf-8') as f:
    idx = json.load(f)

for prefix in ['CAMEL', 'SPARK', 'HADOOP']:
    print(f'\n=== {prefix} module prefixes (top 25) ===')
    files_all = []
    for key, rec in idx.items():
        if key.startswith(prefix + '-'):
            files_all.extend(rec.get('files', []))
    prefixes = Counter()
    for f in files_all:
        parts = f.replace('\\', '/').split('/')
        p = '/'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        prefixes[p] += 1
    for p, c in prefixes.most_common(25):
        print(f'  {c:>6}  {p}')
    print(f'  Total files: {len(files_all):,}  Distinct modules: {len(prefixes):,}')

    # Also show a few full example paths
    print(f'\n  Sample full paths:')
    shown = 0
    for key, rec in idx.items():
        if key.startswith(prefix + '-'):
            for f in rec.get('files', [])[:2]:
                print(f'    {f}')
                shown += 1
            if shown >= 6:
                break
