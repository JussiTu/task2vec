from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)

# List databases
dbs = client.list_database_names()
print("\nDatabases:")
for d in dbs:
    print(" -", d)

# Try likely DB names (based on README and your restore)
candidates = ["jiradump", "JiraReposAnon", "JiraRepos", "JiraReposAnonMay2021", "admin", "local", "config"]
print("\nChecking candidate DBs for collections + counts...\n")

for dbname in candidates:
    if dbname not in dbs:
        continue
    db = client[dbname]
    cols = db.list_collection_names()
    if not cols:
        continue
    print(f"== {dbname} ==")
    # show counts for each collection
    for c in cols:
        try:
            n = db[c].estimated_document_count()
        except Exception:
            n = -1
        print(f"  {c:40s} {n}")
    print()

# If jiradump exists, show one sample doc from the largest collection
target_db = None
for name in ["jiradump", "JiraReposAnon", "JiraRepos"]:
    if name in dbs:
        target_db = name
        break

if not target_db:
    print("No expected DB found. Check your mongorestore nsTo/nsFrom.")
    raise SystemExit(1)

db = client[target_db]
cols = db.list_collection_names()
if not cols:
    print(f"DB {target_db} has no collections.")
    raise SystemExit(1)

# pick largest collection by estimated count
largest = max(cols, key=lambda c: db[c].estimated_document_count())
print(f"\nLargest collection in {target_db}: {largest} (approx {db[largest].estimated_document_count()} docs)")

doc = db[largest].find_one()
print("\nSample document keys:")
if isinstance(doc, dict):
    for k in list(doc.keys())[:60]:
        print(" -", k)
else:
    print(doc)
