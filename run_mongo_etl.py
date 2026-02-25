#!/usr/bin/env python3
"""
MongoDB ETL entry point.

Examples:
  # List all collections
  python run_mongo_etl.py --list-collections

  # List projects in a collection
  python run_mongo_etl.py --collection Apache --list-projects

  # Embed all tickets in one project
  python run_mongo_etl.py --collection Apache --projects ZOOKEEPER

  # Embed multiple projects
  python run_mongo_etl.py --collection Apache --projects KAFKA HADOOP --limit 5000

  # Embed a whole collection (e.g. Spring, ~69k tickets)
  python run_mongo_etl.py --collection Spring

  # Filter by date
  python run_mongo_etl.py --collection MongoDB --projects SERVER --since 2020-01-01

Embeddings are cached in .cache/embeddings.npz — already-embedded tickets
are never re-sent to the API, even across different runs and sources.
"""
import argparse
import sys

from ticketing_intel.config import cfg
from ticketing_intel.etl.mongo_loader import MongoLoader
from ticketing_intel.etl.pipeline import run_pipeline_from_tickets


def main():
    ap = argparse.ArgumentParser(description="MongoDB -> embeddings ETL pipeline")
    ap.add_argument("--collection",       help="MongoDB collection name (e.g. Apache, Spring)")
    ap.add_argument("--projects",         nargs="+", help="Project key(s) to filter (e.g. ZOOKEEPER KAFKA)")
    ap.add_argument("--since",            help="Only tickets created on/after this date (YYYY-MM-DD)")
    ap.add_argument("--limit",            type=int, default=None, help="Max tickets to process")
    ap.add_argument("--no-comments",      action="store_true", help="Exclude comments from embed text")
    ap.add_argument("--batch-size",       type=int, default=128)
    ap.add_argument("--list-collections", action="store_true", help="List available collections and exit")
    ap.add_argument("--list-projects",    action="store_true", help="List projects in --collection and exit")
    ap.add_argument("--min-issues",       type=int, default=10, help="Min issues to show in --list-projects")
    args = ap.parse_args()

    try:
        loader = MongoLoader(uri=cfg.mongo_uri, db=cfg.mongo_db)
    except Exception as e:
        print(f"Could not connect to MongoDB at {cfg.mongo_uri}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- list collections ---
    if args.list_collections:
        print(f"\nAvailable collections in '{cfg.mongo_db}':\n")
        for name in loader.collections():
            n = loader._db[name].estimated_document_count()
            print(f"  {name:<20} {n:>10,} issues")
        loader.close()
        return

    # --- list projects ---
    if args.list_projects:
        if not args.collection:
            print("--list-projects requires --collection", file=sys.stderr)
            sys.exit(1)
        projects = loader.list_projects(args.collection, min_issues=args.min_issues)
        print(f"\nProjects in '{args.collection}' (min {args.min_issues} issues):\n")
        print(f"  {'Project':<30} {'Issues':>8}")
        print(f"  {'-'*38}")
        for p in projects:
            print(f"  {p['project_key']:<30} {p['issue_count']:>8,}")
        print(f"\n  Total: {len(projects)} projects")
        loader.close()
        return

    # --- run ETL ---
    if not args.collection:
        ap.print_help()
        sys.exit(1)

    try:
        cfg.validate()
    except ValueError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    tickets = loader.load_tickets(
        collection=args.collection,
        projects=args.projects or None,
        since=args.since,
        limit=args.limit,
        include_comments=not args.no_comments,
    )
    loader.close()

    if not tickets:
        print("No tickets loaded. Check --collection and --projects.", file=sys.stderr)
        sys.exit(1)

    keys, vectors, store = run_pipeline_from_tickets(
        tickets=tickets,
        cfg=cfg,
        batch_size=args.batch_size,
    )

    # Summary
    assignees = store.assignees(min_tickets=3)
    print(f"Top assignees by ticket count:")
    for row in assignees[:10]:
        print(f"  {row['assignee_id'][:50]:<52} {row['ticket_count']} tickets")

    store.close()
    print("\nDone. Use the cache in your analysis scripts.")


if __name__ == "__main__":
    main()
