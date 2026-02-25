# show_ticket.py
import json, sys

key = sys.argv[1]
with open("jira_1000.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        if doc.get("key") == key:
            fields = doc.get("fields", {})
            print("KEY:", key)
            print("SUMMARY:", fields.get("summary"))
            print("STATUS:", fields.get("status", {}).get("name"))
            print("ISSUETYPE:", fields.get("issuetype", {}).get("name"))
            print("PRIORITY:", fields.get("priority", {}).get("name"))
            print("CREATED:", fields.get("created"))
            print("\nDESCRIPTION:\n", fields.get("description"))
            break
