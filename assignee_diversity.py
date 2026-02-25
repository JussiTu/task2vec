import json
from collections import Counter

assignees = Counter()

with open("jira_1000.jsonl","r",encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        a = d.get("fields",{}).get("assignee")
        if a and a.get("accountId"):
            assignees[a["accountId"]] += 1

print("Unique assignees:", len(assignees))
print("Top 10:", assignees.most_common(10))
