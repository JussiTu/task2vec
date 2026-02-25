"""
MongoDB loader for the ticketing intelligence pipeline.

Streams documents from the jiradump MongoDB database and converts
them to TicketRecord objects using the same parser as the JSONL loader.

Usage:
    from ticketing_intel.etl.mongo_loader import MongoLoader

    loader = MongoLoader(uri="mongodb://localhost:27017", db="jiradump")

    # List available projects in a collection
    loader.list_projects("Apache")

    # Load tickets
    tickets = loader.load_tickets(
        collection="Apache",
        projects=["ZOOKEEPER", "KAFKA"],
        limit=5000,
    )
"""
from typing import Any, Dict, Generator, List, Optional

from ticketing_intel.etl.loader import parse_ticket, build_embed_text, TicketRecord


class MongoLoader:

    def __init__(self, uri: str = "mongodb://localhost:27017", db: str = "jiradump"):
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("pymongo is required: pip install pymongo")
        self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self._db = self._client[db]
        # Verify connection
        self._client.admin.command("ping")

    def collections(self) -> List[str]:
        """List all available collections (one per Jira repo)."""
        return sorted(self._db.list_collection_names())

    def list_projects(self, collection: str, min_issues: int = 1) -> List[Dict[str, Any]]:
        """
        Return projects in a collection sorted by issue count descending.
        Uses aggregation — fast with the project.key index.
        """
        coll = self._db[collection]
        pipeline = [
            {"$group": {"_id": "$fields.project.key", "count": {"$sum": 1}}},
            {"$match": {"_id": {"$ne": None}, "count": {"$gte": min_issues}}},
            {"$sort": {"count": -1}},
        ]
        return [{"project_key": r["_id"], "issue_count": r["count"]}
                for r in coll.aggregate(pipeline)]

    def count(
        self,
        collection: str,
        projects: Optional[List[str]] = None,
        since: Optional[str] = None,
    ) -> int:
        query = self._build_query(projects, since)
        return self._db[collection].count_documents(query)

    def stream(
        self,
        collection: str,
        projects: Optional[List[str]] = None,
        since: Optional[str] = None,
        limit: Optional[int] = None,
        sort_by_date: bool = True,
    ) -> Generator[Dict, None, None]:
        """
        Stream raw documents from MongoDB, yielding plain dicts
        (MongoDB _id removed).
        """
        coll = self._db[collection]
        query = self._build_query(projects, since)

        cursor = coll.find(query, {"_id": 0})
        if sort_by_date:
            cursor = cursor.sort("fields.created", 1)
        if limit:
            cursor = cursor.limit(limit)

        for doc in cursor:
            yield doc

    def load_tickets(
        self,
        collection: str,
        projects: Optional[List[str]] = None,
        since: Optional[str] = None,
        limit: Optional[int] = None,
        include_comments: bool = True,
    ) -> List[TicketRecord]:
        """
        Load and parse tickets from MongoDB into TicketRecord objects.
        Ready to pass directly to run_pipeline_from_tickets().
        """
        total = self.count(collection, projects, since)
        limit_str = f" (limit {limit})" if limit else ""
        proj_str = f" projects={projects}" if projects else ""
        print(f"[mongo] {collection}{proj_str}: {total:,} matching issues{limit_str}")

        tickets: List[TicketRecord] = []
        skipped = 0

        for doc in self.stream(collection, projects, since, limit):
            ticket = parse_ticket(doc)
            if ticket is None:
                skipped += 1
                continue
            if include_comments:
                ticket.embed_text = build_embed_text(ticket, include_comments=True)
            tickets.append(ticket)

        print(f"[mongo] Parsed {len(tickets):,} tickets ({skipped} skipped).")
        return tickets

    def close(self):
        self._client.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_query(
        self,
        projects: Optional[List[str]],
        since: Optional[str],
    ) -> Dict:
        query: Dict = {}
        if projects:
            query["fields.project.key"] = (
                {"$in": projects} if len(projects) > 1 else projects[0]
            )
        if since:
            query["fields.created"] = {"$gte": since}
        return query
