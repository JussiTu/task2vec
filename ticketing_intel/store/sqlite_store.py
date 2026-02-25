"""
SQLite metadata store for ticket records.

Stores structured ticket fields (not vectors — those live in the .npz cache).
Provides simple queries needed by the analysis layer:
  - lookup by key
  - filter by assignee / project / date range
  - list all assignees with ticket counts
"""
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ticketing_intel.etl.loader import TicketRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    key         TEXT PRIMARY KEY,
    issue_id    TEXT,
    summary     TEXT,
    description TEXT,
    assignee_id TEXT,
    creator_id  TEXT,
    reporter_id TEXT,
    created     TEXT,
    updated     TEXT,
    status      TEXT,
    issuetype   TEXT,
    project_key TEXT,
    labels      TEXT,   -- JSON array
    comment_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_assignee ON tickets(assignee_id);
CREATE INDEX IF NOT EXISTS idx_project  ON tickets(project_key);
CREATE INDEX IF NOT EXISTS idx_created  ON tickets(created);
"""


class TicketStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def upsert(self, tickets: List[TicketRecord]):
        rows = [
            (
                t.key,
                t.issue_id,
                t.summary,
                t.description,
                t.assignee_id,
                t.creator_id,
                t.reporter_id,
                t.created,
                t.updated,
                t.status,
                t.issuetype,
                t.project_key,
                json.dumps(t.labels),
                len(t.comments),
            )
            for t in tickets
        ]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO tickets
              (key, issue_id, summary, description, assignee_id, creator_id,
               reporter_id, created, updated, status, issuetype, project_key,
               labels, comment_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self.conn.commit()
        print(f"[store] Upserted {len(rows)} tickets into SQLite.")

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]

    def assignees(self, min_tickets: int = 1) -> List[Dict[str, Any]]:
        """Return assignees sorted by ticket count descending."""
        rows = self.conn.execute(
            """
            SELECT assignee_id, COUNT(*) as ticket_count
            FROM tickets
            WHERE assignee_id != ''
            GROUP BY assignee_id
            HAVING ticket_count >= ?
            ORDER BY ticket_count DESC
            """,
            (min_tickets,),
        ).fetchall()
        return [dict(r) for r in rows]

    def tickets_for_assignee(self, assignee_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM tickets WHERE assignee_id = ? ORDER BY created ASC",
            (assignee_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM tickets WHERE key = ?", (key,)
        ).fetchone()
        return dict(row) if row else None

    def all_keys(self) -> List[str]:
        rows = self.conn.execute("SELECT key FROM tickets ORDER BY created ASC").fetchall()
        return [r[0] for r in rows]

    def all_tickets(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM tickets ORDER BY created ASC").fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
