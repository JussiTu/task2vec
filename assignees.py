import json

with open("jira_1000.jsonl","r",encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        print(d.get("fields", {}).get("assignee"))
        break
