from typing import Any, Dict, Optional, Tuple, List
from collections import Counter

def extract_user_id(user_obj: Any) -> str:
    if not isinstance(user_obj, dict):
        if isinstance(user_obj, str) and user_obj.strip():
            return user_obj.strip()
        return ""
    for k in ["accountId", "key", "name", "id"]:
        v = user_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = user_obj.get("displayName")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return ""

def extract_actor_id(doc: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (id, source_label).
    Prefers assignee, then creator, reporter, then changelog author, then comment author.
    """
    fields = doc.get("fields") or {}

    # 1) assignee
    a = fields.get("assignee")
    aid = extract_user_id(a)
    if aid:
        return aid, "assignee"

    # 2) creator / reporter (often present even when assignee is wiped)
    cid = extract_user_id(fields.get("creator"))
    if cid:
        return cid, "creator"

    rid = extract_user_id(fields.get("reporter"))
    if rid:
        return rid, "reporter"

    # 3) changelog author
    histories = (doc.get("changelog") or {}).get("histories") or []
    if isinstance(histories, list) and histories:
        h0 = histories[0] if isinstance(histories[0], dict) else None
        if isinstance(h0, dict):
            hid = extract_user_id(h0.get("author"))
            if hid:
                return hid, "changelog_author"

    # 4) comment author
    comments = ((fields.get("comment") or {}).get("comments")) or []
    if isinstance(comments, list) and comments:
        c0 = comments[0] if isinstance(comments[0], dict) else None
        if isinstance(c0, dict):
            coid = extract_user_id(c0.get("author"))
            if coid:
                return coid, "comment_author"

    return "", "none"
