#!/usr/bin/env python3
"""
ETL entry point.

Run:
    python run_etl.py
    python run_etl.py --input jira_1000.jsonl --limit 500
    python run_etl.py --input jira_1000.jsonl --no-comments

After the first run, embeddings are cached in .cache/embeddings.npz.
Subsequent runs are instant for already-seen tickets.
"""
import argparse
import sys

from ticketing_intel.config import cfg
from ticketing_intel.etl.pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="Jira → embeddings ETL pipeline")
    ap.add_argument("--input", help="JSONL path (overrides JIRA_DUMP_PATH in .env)")
    ap.add_argument("--limit", type=int, default=None, help="Max tickets to process")
    ap.add_argument("--no-comments", action="store_true", help="Exclude comments from embed text")
    ap.add_argument("--batch-size", type=int, default=128, help="Embedding API batch size")
    args = ap.parse_args()

    if args.input:
        cfg.jira_dump_path = args.input

    try:
        keys, vectors, store = run_pipeline(
            cfg=cfg,
            limit=args.limit,
            include_comments=not args.no_comments,
            batch_size=args.batch_size,
        )
    except (ValueError, RuntimeError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    # Quick summary
    assignees = store.assignees(min_tickets=3)
    print(f"Top assignees by ticket count:")
    for row in assignees[:10]:
        print(f"  {row['assignee_id'][:40]:<42} {row['ticket_count']} tickets")

    store.close()
    print("\nRun complete. Import ticketing_intel in your analysis scripts to use the cache.")


if __name__ == "__main__":
    main()
