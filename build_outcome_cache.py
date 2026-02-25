"""
build_outcome_cache.py
======================
Reads resolved Spring tickets from MongoDB `jiradump.Spring`, computes outcome
signals per ticket, and writes `.cache/outcome_signals.json`.

Calibration thresholds (computed from data):
  p33_days  = 1.9   (fast  = below this)
  p67_days  = 29.5  (slow  = above this)
  p75_assignee_count = 136 tickets

Label each ticket:
  score = (days ≤ p33 ? 1 : 0) + (watches ≤ 2 ? 1 : 0) + (assignee_count < p75 ? 1 : 0)
  Automate if score == 3, Escalate if score ≤ 1, Assist otherwise

Only tickets present in search_keys.npy are included.

Usage:
    python build_outcome_cache.py
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymongo import MongoClient

# ── Config ──────────────────────────────────────────────────────────────────
MONGO_URI  = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME    = "jiradump"
COLLECTION = "Spring"

BASE       = Path(__file__).parent
CACHE      = BASE / ".cache"
OUT_FILE   = CACHE / "outcome_signals.json"
KEYS_FILE  = CACHE / "search_keys.npy"

# Calibration thresholds (derived from Spring data analysis)
P33_DAYS         = 1.9
P67_DAYS         = 29.5
P75_ASSIGNEE_CNT = 136

# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_dt(val):
    """Return aware datetime or None."""
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    if isinstance(val, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(val, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
    return None


def _label(days, watches, assignee_count):
    score = (
        (1 if days <= P33_DAYS else 0) +
        (1 if watches <= 2     else 0) +
        (1 if assignee_count < P75_ASSIGNEE_CNT else 0)
    )
    if score == 3:
        return "Automate"
    if score <= 1:
        return "Escalate"
    return "Assist"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    CACHE.mkdir(exist_ok=True)

    # Load the set of keys in the search index
    print(f"Loading search keys from {KEYS_FILE} …", flush=True)
    if not KEYS_FILE.exists():
        raise FileNotFoundError(f"Missing {KEYS_FILE} — run the embedding pipeline first.")
    indexed_keys = set(np.load(str(KEYS_FILE)).tolist())
    print(f"  {len(indexed_keys):,} keys in search index.", flush=True)

    # Connect to MongoDB
    print(f"Connecting to MongoDB {MONGO_URI} …", flush=True)
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    coll   = client[DB_NAME][COLLECTION]
    count  = coll.count_documents({})
    print(f"  {count:,} documents in {DB_NAME}.{COLLECTION}.", flush=True)

    # --- First pass: build assignee ticket counts (denominator for exp proxy)
    print("First pass: counting tickets per assignee …", flush=True)
    assignee_counts: dict[str, int] = {}
    for doc in coll.find({}, {"fields.assignee.displayName": 1}):
        name = (doc.get("fields") or {}).get("assignee") or {}
        if isinstance(name, dict):
            name = name.get("displayName", "")
        if name:
            assignee_counts[name] = assignee_counts.get(name, 0) + 1

    max_count = max(assignee_counts.values()) if assignee_counts else 1
    print(f"  {len(assignee_counts):,} unique assignees; max tickets = {max_count}.", flush=True)

    # --- Second pass: compute signals for indexed tickets
    print("Second pass: computing outcome signals …", flush=True)
    signals   = {}
    skipped   = 0
    processed = 0

    cursor = coll.find(
        {},
        {
            "key": 1,
            "fields.created": 1,
            "fields.resolutiondate": 1,
            "fields.watches.watchCount": 1,
            "fields.assignee.displayName": 1,
        }
    )

    for doc in cursor:
        key = doc.get("key", "")
        if key not in indexed_keys:
            skipped += 1
            continue

        fields = doc.get("fields") or {}

        # Resolution time
        created_raw    = fields.get("created")
        resolved_raw   = fields.get("resolutiondate")
        created_dt     = _parse_dt(created_raw)
        resolved_dt    = _parse_dt(resolved_raw)

        if created_dt is None or resolved_dt is None:
            skipped += 1
            continue

        days = max(0.0, (resolved_dt - created_dt).total_seconds() / 86400.0)

        # Watch count
        watches_obj = fields.get("watches") or {}
        if isinstance(watches_obj, dict):
            watches = int(watches_obj.get("watchCount", 0) or 0)
        else:
            watches = int(watches_obj or 0)

        # Assignee experience
        assignee_info = fields.get("assignee") or {}
        if isinstance(assignee_info, dict):
            assignee_name = assignee_info.get("displayName", "")
        else:
            assignee_name = str(assignee_info)

        assignee_count = assignee_counts.get(assignee_name, 0)
        assignee_exp   = round(assignee_count / max_count, 4) if max_count > 0 else 0.0

        label = _label(days, watches, assignee_count)

        signals[key] = {
            "days":         round(days, 2),
            "watches":      watches,
            "assignee_exp": assignee_exp,
            "label":        label,
        }
        processed += 1

        if processed % 5000 == 0:
            print(f"  … {processed:,} processed", flush=True)

    print(f"Done: {processed:,} signals written, {skipped:,} skipped.", flush=True)

    # Label distribution
    from collections import Counter
    dist = Counter(s["label"] for s in signals.values())
    total = sum(dist.values()) or 1
    print("Label distribution:")
    for lbl, cnt in sorted(dist.items()):
        print(f"  {lbl}: {cnt:,}  ({cnt/total*100:.1f}%)")

    # Write output
    output = {
        "calibration": {
            "p33_days":     P33_DAYS,
            "p67_days":     P67_DAYS,
            "p75_assignee": P75_ASSIGNEE_CNT,
        },
        "signals": signals,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = OUT_FILE.stat().st_size / 1_048_576
    print(f"\nWrote {OUT_FILE}  ({size_mb:.1f} MB, {len(signals):,} tickets)", flush=True)
    client.close()


if __name__ == "__main__":
    main()
