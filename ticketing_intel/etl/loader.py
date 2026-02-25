"""
Loads Jira JSONL dumps and parses tickets into structured records.

Handles:
- JSONL (one JSON object per line) — the format used in this project
- Atlassian Document Format (ADF) descriptions — stringified
- Both new Jira (accountId) and old Jira (name/key) for people fields
- Comment bodies (up to N comments per ticket)
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


@dataclass
class TicketRecord:
    key: str
    issue_id: str
    summary: str
    description: str
    assignee_id: str
    creator_id: str
    reporter_id: str
    created: str            # ISO 8601 string
    updated: str
    status: str
    issuetype: str
    project_key: str
    labels: List[str]
    comments: List[str]     # raw comment bodies (text)
    embed_text: str = ""    # built after parsing; used for embedding


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _get(d: Dict, *path, default=None) -> Any:
    cur: Any = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _person_id(person: Any) -> str:
    """Extract stable ID from a Jira person object."""
    if not isinstance(person, dict):
        return ""
    for k in ("accountId", "key", "name", "id"):
        v = person.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Fall back to displayName (may not be unique but better than nothing)
    v = person.get("displayName")
    return v.strip() if isinstance(v, str) and v.strip() else ""


def _extract_comments(doc: Dict, max_comments: int = 10) -> List[str]:
    comments_raw = _get(doc, "fields", "comment", "comments", default=[]) or []
    bodies: List[str] = []
    for c in comments_raw[:max_comments]:
        body = _safe_str(c.get("body", "")).strip()
        if body:
            bodies.append(body)
    return bodies


# ---------------------------------------------------------------------------
# Canonical embed text
# ---------------------------------------------------------------------------

def build_embed_text(
    ticket: "TicketRecord",
    include_comments: bool = True,
    max_chars: int = 6000,
    max_comment_chars: int = 600,
) -> str:
    """
    Builds the text we send to the embedding model.
    Structured sections help the model weight fields appropriately.
    Comments are included but truncated individually.
    """
    parts: List[str] = []

    if ticket.key:
        parts.append(f"[KEY]\n{ticket.key}")
    if ticket.issuetype:
        parts.append(f"[TYPE]\n{ticket.issuetype}")
    if ticket.summary:
        parts.append(f"[SUMMARY]\n{ticket.summary}")
    if ticket.description:
        parts.append(f"[DESCRIPTION]\n{ticket.description}")

    if include_comments and ticket.comments:
        truncated = [c[:max_comment_chars] for c in ticket.comments]
        parts.append("[COMMENTS]\n" + "\n\n---\n\n".join(truncated))

    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]"
    return text


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_ticket(doc: Dict[str, Any]) -> Optional[TicketRecord]:
    """Parse one raw Jira document into a TicketRecord. Returns None if invalid."""
    key = _safe_str(doc.get("key") or doc.get("id") or "").strip()
    if not key:
        return None

    fields = doc.get("fields") or {}

    ticket = TicketRecord(
        key=key,
        issue_id=_safe_str(doc.get("id", "")),
        summary=_safe_str(fields.get("summary", "")),
        description=_safe_str(fields.get("description", "")),
        assignee_id=_person_id(fields.get("assignee")),
        creator_id=_person_id(fields.get("creator")),
        reporter_id=_person_id(fields.get("reporter")),
        created=_safe_str(fields.get("created", "")),
        updated=_safe_str(fields.get("updated", "")),
        status=_safe_str(_get(fields, "status", "name", default="")),
        issuetype=_safe_str(_get(fields, "issuetype", "name", default="")),
        project_key=_safe_str(_get(fields, "project", "key", default="")),
        labels=fields.get("labels") if isinstance(fields.get("labels"), list) else [],
        comments=_extract_comments(doc),
    )
    ticket.embed_text = build_embed_text(ticket)
    return ticket


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

def stream_jsonl(path: str, limit: Optional[int] = None) -> Generator[Dict, None, None]:
    """Stream raw dicts from a JSONL file without loading all into memory."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                count += 1
            except json.JSONDecodeError:
                continue


def load_tickets(
    path: str,
    limit: Optional[int] = None,
    include_comments: bool = True,
) -> List[TicketRecord]:
    """Load and parse all tickets from a JSONL file."""
    tickets: List[TicketRecord] = []
    for doc in stream_jsonl(path, limit=limit):
        ticket = parse_ticket(doc)
        if ticket is None:
            continue
        if include_comments:
            ticket.embed_text = build_embed_text(ticket, include_comments=True)
        tickets.append(ticket)
    return tickets
