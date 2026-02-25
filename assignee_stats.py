import argparse, json
from collections import Counter

def get_nested(d, path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def iter_jsonl(path, limit=None):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit and n >= limit:
                break

def extract_assignee_id(doc):
    a = get_nested(doc, ["fields", "assignee"], None)
    if not isinstance(a, dict):
        return ""
    for k in ["accountId", "key", "name", "id"]:
        v = a.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = a.get("displayName")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return ""

def has_assignee_history(doc):
    histories = get_nested(doc, ["changelog", "histories"], [])
    if not isinstance(histories, list):
        return False
    for h in histories:
        items = h.get("items")
        if not isinstance(items, list):
            continue
        for it in items:
            field = it.get("field")
            # Jira is usually exactly "assignee" but be tolerant
            if isinstance(field, str) and field.lower() == "assignee":
                return True
    return False

def count_assignee_transfers(doc):
    """Counts how many 'assignee' change events are recorded in changelog."""
    histories = get_nested(doc, ["changelog", "histories"], [])
    if not isinstance(histories, list):
        return 0
    c = 0
    for h in histories:
        items = h.get("items")
        if not isinstance(items, list):
            continue
        for it in items:
            field = it.get("field")
            if isinstance(field, str) and field.lower() == "assignee":
                c += 1
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl exported issues file")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    total = 0
    with_assignee = 0
    with_history = 0
    transfer_event_total = 0
    transfer_events_per_issue = Counter()

    for doc in iter_jsonl(args.input, limit=(args.limit or None)):
        total += 1

        aid = extract_assignee_id(doc)
        if aid:
            with_assignee += 1

        tcount = count_assignee_transfers(doc)
        if tcount > 0:
            with_history += 1
            transfer_event_total += tcount
            transfer_events_per_issue[tcount] += 1

    print(f"Scanned issues: {total}")
    print(f"Issues with current assignee (fields.assignee): {with_assignee} ({with_assignee/total:.1%})")
    print(f"Issues with assignee history in changelog:      {with_history} ({with_history/total:.1%})")
    print(f"Total assignee change events found:             {transfer_event_total}")

    if with_history:
        print("\nAssignee-change events per issue (count -> #issues):")
        for k in sorted(transfer_events_per_issue):
            print(f"  {k}: {transfer_events_per_issue[k]}")

if __name__ == "__main__":
    main()
