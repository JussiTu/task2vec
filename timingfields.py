count = 0
with open("jira_100.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        tt = (doc.get("fields", {}) or {}).get("timetracking")
        if tt and tt.get("timeSpentSeconds"):
            count += 1

print("Issues with time spent:", count)
