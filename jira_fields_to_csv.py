import json
import csv
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

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

def is_primitive(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))

def short_val(x: Any, max_len: int = 120) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        s = x.replace("\n", " ").replace("\r", " ").strip()
        return s[:max_len] + ("…" if len(s) > max_len else "")
    if isinstance(x, (int, float, bool)):
        return str(x)
    try:
        s = json.dumps(x, ensure_ascii=False)
        s = s.replace("\n", " ").replace("\r", " ")
        return s[:max_len] + ("…" if len(s) > max_len else "")
    except Exception:
        s = str(x)
        return s[:max_len] + ("…" if len(s) > max_len else "")

def type_name(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "str"
    if isinstance(x, list):
        return "list"
    if isinstance(x, dict):
        return "dict"
    return type(x).__name__

def flatten(
    obj: Any,
    prefix: str,
    out: Dict[str, List[Any]],
    list_mode: str = "first",  # "first" or "all"
    max_list_items: int = 3
):
    """
    Collect values per field path.
    - For dict: recurse into keys
    - For list: either take first element, or take up to max_list_items items
    - For primitive: store the value
    """
    if is_primitive(obj):
        out[prefix].append(obj)
        return

    if isinstance(obj, dict):
        if prefix:  # store that this path is a dict too (optional)
            out[prefix].append({"_type": "dict"})
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            flatten(v, p, out, list_mode=list_mode, max_list_items=max_list_items)
        return

    if isinstance(obj, list):
        out[prefix].append({"_type": "list", "len": len(obj)})
        if len(obj) == 0:
            return
        if list_mode == "first":
            flatten(obj[0], f"{prefix}[0]", out, list_mode=list_mode, max_list_items=max_list_items)
        else:
            for i, item in enumerate(obj[:max_list_items]):
                flatten(item, f"{prefix}[{i}]", out, list_mode=list_mode, max_list_items=max_list_items)
        return

    # fallback
    out[prefix].append(str(obj))

def summarize_values(values: List[Any], sample_n: int = 3) -> Tuple[str, str, str, int]:
    # Determine dominant type among non-marker entries
    real = [v for v in values if not (isinstance(v, dict) and v.get("_type") in ("dict", "list"))]
    markers = [v for v in values if isinstance(v, dict) and v.get("_type") in ("dict", "list")]

    types = defaultdict(int)
    for v in real:
        types[type_name(v)] += 1
    dominant = sorted(types.items(), key=lambda x: -x[1])[0][0] if types else "mixed/struct"

    # presence count: number of documents where field appeared (approx from collected list length is not exact)
    # We'll compute presence separately; here just samples
    uniq_samples: List[str] = []
    seen: Set[str] = set()

    # Prefer primitive/string-like samples
    for v in real:
        s = short_val(v)
        if s and s not in seen:
            uniq_samples.append(s)
            seen.add(s)
        if len(uniq_samples) >= sample_n:
            break

    # if none, use marker samples
    if len(uniq_samples) < sample_n:
        for v in markers:
            s = short_val(v)
            if s and s not in seen:
                uniq_samples.append(s)
                seen.add(s)
            if len(uniq_samples) >= sample_n:
                break

    # pad
    while len(uniq_samples) < sample_n:
        uniq_samples.append("")

    return dominant, uniq_samples[0], uniq_samples[1], uniq_samples[2], len(values)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jira_1000.jsonl (or similar)")
    ap.add_argument("--limit", type=int, default=1000, help="How many issues to scan")
    ap.add_argument("--out", default="jira_fields_samples.csv")
    ap.add_argument("--list_mode", choices=["first", "all"], default="first",
                    help="How to handle arrays: first element only, or first few elements")
    ap.add_argument("--max_list_items", type=int, default=3)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    field_values: Dict[str, List[Any]] = defaultdict(list)
    field_presence: Dict[str, int] = defaultdict(int)

    doc_count = 0
    for doc in iter_jsonl(args.input, args.limit):
        doc_count += 1
        per_doc: Dict[str, List[Any]] = defaultdict(list)
        flatten(doc, "", per_doc, list_mode=args.list_mode, max_list_items=args.max_list_items)

        # Count presence per document (field appears if we saw it at least once in that doc)
        for path, vals in per_doc.items():
            if path == "":
                continue
            field_presence[path] += 1
            field_values[path].extend(vals)

    # Write CSV sorted by presence desc
    rows = []
    for path, pres in field_presence.items():
        dominant, s1, s2, s3, total_vals = summarize_values(field_values[path], sample_n=args.samples)
        rows.append((pres, path, dominant, s1, s2, s3, total_vals))

    rows.sort(key=lambda x: (-x[0], x[1]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["field_path", "present_in_docs", "present_pct", "dominant_type",
                    "sample_1", "sample_2", "sample_3", "total_observations"])
        for pres, path, dominant, s1, s2, s3, total_vals in rows:
            pct = (pres / max(1, doc_count)) * 100.0
            w.writerow([path, pres, f"{pct:.1f}", dominant, s1, s2, s3, total_vals])

    print(f"Scanned {doc_count} docs from {args.input}")
    print(f"Found {len(rows)} unique field paths")
    print(f"Wrote CSV -> {args.out}")

if __name__ == "__main__":
    main()
