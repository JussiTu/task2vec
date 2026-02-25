import json, csv, argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional

# --------- helpers ----------
def iter_jsonl(path: str, limit: int):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                n += 1
                if limit and n >= limit:
                    return
            except Exception:
                continue

def short(x: Any, max_len: int = 120) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        s = x.replace("\n"," ").replace("\r"," ").strip()
        return s[:max_len] + ("…" if len(s) > max_len else "")
    if isinstance(x, (int,float,bool)):
        return str(x)
    try:
        s = json.dumps(x, ensure_ascii=False)
        s = s.replace("\n"," ").replace("\r"," ")
        return s[:max_len] + ("…" if len(s) > max_len else "")
    except Exception:
        s = str(x)
        return s[:max_len] + ("…" if len(s) > max_len else "")

def tname(x: Any) -> str:
    if x is None: return "null"
    if isinstance(x, bool): return "bool"
    if isinstance(x, int): return "int"
    if isinstance(x, float): return "float"
    if isinstance(x, str): return "str"
    if isinstance(x, list): return "list"
    if isinstance(x, dict): return "dict"
    return type(x).__name__

def get(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

# --------- schema fields we actually care about ----------
# These are "canonical" fields: stable & interpretable.
CANON = [
    ("Core", "key", ["key"]),
    ("Core", "id", ["id"]),
    ("Core", "self", ["self"]),
    ("Core", "created", ["fields","created"]),
    ("Core", "updated", ["fields","updated"]),
    ("Core", "project.key", ["fields","project","key"]),
    ("Core", "project.name", ["fields","project","name"]),

    ("Content", "summary", ["fields","summary"]),
    ("Content", "description", ["fields","description"]),

    ("Classification", "issuetype.name", ["fields","issuetype","name"]),
    ("Classification", "priority.name", ["fields","priority","name"]),
    ("Workflow", "status.name", ["fields","status","name"]),
    ("Workflow", "resolution.name", ["fields","resolution","name"]),

    ("People", "assignee.id", ["fields","assignee","accountId"]),
    ("People", "creator.id", ["fields","creator","accountId"]),
    ("People", "reporter.id", ["fields","reporter","accountId"]),

    ("Classification", "labels", ["fields","labels"]),          # list[str]
    ("Classification", "components", ["fields","components"]),  # list[dict]
    ("Classification", "fixVersions", ["fields","fixVersions"]),# list[dict]
    ("Classification", "versions", ["fields","versions"]),      # list[dict]

    ("Planning", "duedate", ["fields","duedate"]),
    ("Planning", "timeoriginalestimate", ["fields","timeoriginalestimate"]),
    ("Planning", "timeestimate", ["fields","timeestimate"]),
    ("Planning", "timespent", ["fields","timespent"]),

    ("Links", "issuelinks", ["fields","issuelinks"]),
    ("Links", "subtasks", ["fields","subtasks"]),

    ("Social", "comment.count", ["fields","comment","total"]),
    ("Social", "comment.comments", ["fields","comment","comments"]),  # list[dict]

    ("Change", "changelog.histories", ["changelog","histories"]),     # list[dict]
]

# For list-of-objects fields, we’ll extract common scalar subfields.
LIST_OBJECT_SUBFIELDS = {
    "components": ["name", "id"],
    "versions": ["name", "id"],
    "fixVersions": ["name", "id"],
    "issuelinks": ["type", "inwardIssue", "outwardIssue"],
    "subtasks": ["key", "id"],
    "comment.comments": ["id", "created"],  # body is huge
    "changelog.histories": ["created"],     # items summarized separately
}

def summarize_list_of_objects(val: Any, want: List[str]) -> Dict[str, Any]:
    if not isinstance(val, list):
        return {}
    out: Dict[str, Any] = {}
    # For each wanted subfield, collect up to 3 samples
    for sub in want:
        samples = []
        for it in val[:10]:
            if not isinstance(it, dict):
                continue
            v = it.get(sub)
            if v is None:
                continue
            samples.append(v)
            if len(samples) >= 3:
                break
        if samples:
            out[sub] = samples
    out["_len"] = len(val)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default="jira_schema_summary.csv")
    ap.add_argument("--out_groups", default="jira_schema_groups.csv")
    args = ap.parse_args()

    present = Counter()
    types = Counter()
    samples = defaultdict(list)  # field -> list[str]
    doc_count = 0

    # Extra summaries for changelog "what changes"
    changed_fields = Counter()

    for doc in iter_jsonl(args.input, args.limit):
        doc_count += 1

        # changelog items summary (common changed fields)
        histories = get(doc, ["changelog","histories"])
        if isinstance(histories, list):
            for h in histories[:50]:
                items = h.get("items") if isinstance(h, dict) else None
                if isinstance(items, list):
                    for it in items[:50]:
                        if isinstance(it, dict) and it.get("field"):
                            changed_fields[it["field"]] += 1

        for group, name, path in CANON:
            v = get(doc, path)
            field = name  # canonical name

            if v is None:
                continue

            present[field] += 1

            # Determine a human-friendly type
            if name in ("labels",):
                ty = "list[str]" if isinstance(v, list) else tname(v)
                types[(field, ty)] += 1
                if isinstance(v, list) and v:
                    samples[field].append(short(v[:5]))
                else:
                    samples[field].append(short(v))
                continue

            # list-of-objects summarization
            if name in LIST_OBJECT_SUBFIELDS:
                want = LIST_OBJECT_SUBFIELDS[name]
                summ = summarize_list_of_objects(v, want)
                ty = f"list[object] (len≈{summ.get('_len',0)})"
                types[(field, ty)] += 1
                samples[field].append(short(summ))
                continue

            # normal scalar / object
            ty = tname(v)
            types[(field, ty)] += 1
            samples[field].append(short(v))

    # Write schema CSV
    rows = []
    for group, name, path in CANON:
        field = name
        pres = present.get(field, 0)
        pct = 100.0 * pres / max(1, doc_count)

        # dominant type
        ty_counts = [(ty, c) for (f, ty), c in types.items() if f == field]
        ty_counts.sort(key=lambda x: -x[1])
        dom_ty = ty_counts[0][0] if ty_counts else ""

        # unique-ish sample
        s = ""
        if samples.get(field):
            # pick first non-empty
            for cand in samples[field]:
                if cand:
                    s = cand
                    break

        rows.append((group, field, dom_ty, pres, f"{pct:.1f}", s))

    # Sort by group then presence
    rows.sort(key=lambda r: (r[0], -r[3], r[1]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "field", "type", "present_in_docs", "present_pct", "example"])
        for r in rows:
            w.writerow(r)

    # Write groups summary
    grp_counts = Counter()
    for group, field, *_ in rows:
        grp_counts[group] += 1

    with open(args.out_groups, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "fields_in_group", "docs_scanned"])
        for g, c in grp_counts.most_common():
            w.writerow([g, c, doc_count])

    # Also dump top changed fields (very useful)
    top_changed = changed_fields.most_common(30)
    print(f"Scanned {doc_count} docs")
    print(f"Wrote -> {args.out}")
    print(f"Wrote -> {args.out_groups}")
    if top_changed:
        print("\nTop changed Jira fields from changelog (sample):")
        for k, c in top_changed[:15]:
            print(f"  {k}: {c}")

if __name__ == "__main__":
    main()
