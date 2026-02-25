import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx


# -----------------------------
# JSONL utils
# -----------------------------
def iter_jsonl(path: str, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj
            n += 1
            if limit is not None and n >= limit:
                break


def get_nested(d: Any, path: List[Any], default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and isinstance(p, int) and 0 <= p < len(cur):
            cur = cur[p]
        else:
            return default
    return cur


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


# -----------------------------
# Actor extraction
# -----------------------------
def extract_actor_id(person_obj: Any) -> str:
    """
    Canonicalize Jira person objects (pseudonymized in your dump).
    Prefer stable ids; fall back to displayName if that's all we have.
    """
    if not isinstance(person_obj, dict):
        return ""

    for k in ["accountId", "key", "name", "id"]:
        v = person_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    v = person_obj.get("displayName")
    if isinstance(v, str) and v.strip():
        return v.strip()

    return ""


def actor_label(person_obj: Any) -> str:
    """Human-readable label (still pseudonymized in your dump)."""
    if not isinstance(person_obj, dict):
        return ""
    dn = person_obj.get("displayName")
    if isinstance(dn, str) and dn.strip():
        return dn.strip()
    # fall back to whatever id we used
    return extract_actor_id(person_obj)


# -----------------------------
# Pull people fields from a ticket
# -----------------------------
def ticket_key(doc: Dict[str, Any]) -> str:
    return safe_str(doc.get("key") or doc.get("id") or "").strip()


def pull_core_people(doc: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """
    Returns {role: (actor_id, label)}
    roles: creator, reporter, assignee
    """
    out = {}
    for role in ["creator", "reporter", "assignee"]:
        obj = get_nested(doc, ["fields", role], default=None)
        aid = extract_actor_id(obj)
        if aid:
            out[role] = (aid, actor_label(obj))
    return out


def pull_comment_authors(doc: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Returns list of (actor_id, label) for comment authors
    Path in your dump: fields.comments is a LIST of comment dicts
      fields.comments[].author
    """
    authors = []
    comments = get_nested(doc, ["fields", "comments"], default=[])
    if not isinstance(comments, list):
        return authors

    for c in comments:
        aobj = c.get("author") if isinstance(c, dict) else None
        aid = extract_actor_id(aobj)
        if aid:
            authors.append((aid, actor_label(aobj)))
    return authors


def pull_changelog_authors(doc: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    changelog.histories[].author
    """
    authors = []
    histories = get_nested(doc, ["changelog", "histories"], default=[])
    if not isinstance(histories, list):
        return authors
    for h in histories:
        aobj = h.get("author") if isinstance(h, dict) else None
        aid = extract_actor_id(aobj)
        if aid:
            authors.append((aid, actor_label(aobj)))
    return authors


def pull_assignee_transfers(doc: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Directed transfer events A -> B when changelog item.field == "assignee"
    Uses from/to ids where available, else fromString/toString
    """
    transfers = []
    histories = get_nested(doc, ["changelog", "histories"], default=[])
    if not isinstance(histories, list):
        return transfers

    for h in histories:
        items = h.get("items") if isinstance(h, dict) else None
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            if safe_str(it.get("field")).lower() != "assignee":
                continue

            frm = safe_str(it.get("from") or it.get("fromString")).strip()
            to = safe_str(it.get("to") or it.get("toString")).strip()
            if frm and to and frm != to:
                transfers.append((frm, to))
    return transfers


# -----------------------------
# Graph building (multi-layer)
# -----------------------------
def add_edge(G: nx.Graph, u: str, v: str, layer: str, w: float = 1.0):
    if not u or not v or u == v:
        return
    if G.has_edge(u, v):
        # keep per-layer weights + total
        G[u][v]["weight"] = float(G[u][v].get("weight", 0.0) + w)
        key = f"w_{layer}"
        G[u][v][key] = float(G[u][v].get(key, 0.0) + w)
    else:
        G.add_edge(u, v, weight=float(w), **{f"w_{layer}": float(w)})


def add_diedge(G: nx.DiGraph, u: str, v: str, layer: str, w: float = 1.0):
    if not u or not v or u == v:
        return
    if G.has_edge(u, v):
        G[u][v]["weight"] = float(G[u][v].get("weight", 0.0) + w)
        key = f"w_{layer}"
        G[u][v][key] = float(G[u][v].get(key, 0.0) + w)
    else:
        G.add_edge(u, v, weight=float(w), **{f"w_{layer}": float(w)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out_nodes", default="people_nodes.csv")
    ap.add_argument("--out_edges", default="people_edges.csv")
    ap.add_argument("--out_graphml", default="people_graph.graphml")
    ap.add_argument("--directed", action="store_true",
                    help="If set, produces a directed graph (useful if you care about transfer direction).")
    args = ap.parse_args()

    G = nx.DiGraph() if args.directed else nx.Graph()

    # node stats
    node_roles = defaultdict(Counter)   # actor_id -> Counter(role)
    node_label = {}                     # actor_id -> label
    tickets_touched = defaultdict(set)  # actor_id -> set(ticket_key)

    # edge provenance counts (for CSV)
    edge_layers = defaultdict(Counter)  # (u,v) -> Counter(layer)

    scanned = 0
    transfer_events = 0

    for doc in iter_jsonl(args.input, limit=args.limit):
        scanned += 1
        tkey = ticket_key(doc)

        core = pull_core_people(doc)  # creator/reporter/assignee
        comment_authors = pull_comment_authors(doc)
        changelog_authors = pull_changelog_authors(doc)
        transfers = pull_assignee_transfers(doc)

        # register nodes
        for role, (aid, lbl) in core.items():
            node_roles[aid][role] += 1
            node_label.setdefault(aid, lbl or aid)
            if tkey:
                tickets_touched[aid].add(tkey)

        for (aid, lbl) in comment_authors:
            node_roles[aid]["comment_author"] += 1
            node_label.setdefault(aid, lbl or aid)
            if tkey:
                tickets_touched[aid].add(tkey)

        for (aid, lbl) in changelog_authors:
            node_roles[aid]["changelog_author"] += 1
            node_label.setdefault(aid, lbl or aid)
            if tkey:
                tickets_touched[aid].add(tkey)

        # --- Layer A: core role ties (undirected, symmetric meaning)
        creator = core.get("creator", ("", ""))[0]
        reporter = core.get("reporter", ("", ""))[0]
        assignee = core.get("assignee", ("", ""))[0]

        for (u, v, layer) in [
            (creator, assignee, "role_creator_assignee"),
            (reporter, assignee, "role_reporter_assignee"),
            (creator, reporter, "role_creator_reporter"),
        ]:
            if args.directed:
                add_diedge(G, u, v, layer, 1.0)
                add_diedge(G, v, u, layer, 1.0)  # keep symmetric meaning in directed mode
            else:
                add_edge(G, u, v, layer, 1.0)

            if u and v and u != v:
                edge_layers[(u, v)][layer] += 1
                if not args.directed:
                    edge_layers[(v, u)][layer] += 1  # for easier lookups later

        # --- Layer B: comment collaboration
        # Connect comment authors pairwise + to assignee (if present)
        ca_ids = [aid for (aid, _) in comment_authors]
        # pairwise among authors (unique pairs)
        seen = set()
        for i in range(len(ca_ids)):
            for j in range(i + 1, len(ca_ids)):
                u, v = ca_ids[i], ca_ids[j]
                if not u or not v or u == v:
                    continue
                key = tuple(sorted((u, v))) if not args.directed else (u, v)
                if key in seen and not args.directed:
                    continue
                seen.add(key)
                if args.directed:
                    add_diedge(G, u, v, "comment_coauthor", 1.0)
                    add_diedge(G, v, u, "comment_coauthor", 1.0)
                else:
                    add_edge(G, u, v, "comment_coauthor", 1.0)

        if assignee:
            for u in set(ca_ids):
                if args.directed:
                    add_diedge(G, u, assignee, "comment_to_assignee", 1.0)
                    add_diedge(G, assignee, u, "comment_to_assignee", 1.0)
                else:
                    add_edge(G, u, assignee, "comment_to_assignee", 1.0)

        # --- Layer C: changelog interaction
        ch_ids = [aid for (aid, _) in changelog_authors]
        for u in set(ch_ids):
            if assignee:
                if args.directed:
                    add_diedge(G, u, assignee, "changelog_to_assignee", 1.0)
                    add_diedge(G, assignee, u, "changelog_to_assignee", 1.0)
                else:
                    add_edge(G, u, assignee, "changelog_to_assignee", 1.0)
            if creator:
                if args.directed:
                    add_diedge(G, u, creator, "changelog_to_creator", 1.0)
                    add_diedge(G, creator, u, "changelog_to_creator", 1.0)
                else:
                    add_edge(G, u, creator, "changelog_to_creator", 1.0)

        # --- Layer D: transfers (meaningfully directed)
        for frm, to in transfers:
            transfer_events += 1
            if args.directed:
                add_diedge(G, frm, to, "assignee_transfer", 1.0)
            else:
                # if undirected, still link but loses direction
                add_edge(G, frm, to, "assignee_transfer", 1.0)

    # finalize nodes
    for aid in set(list(node_roles.keys()) + list(node_label.keys())):
        if not aid:
            continue
        if aid not in G:
            G.add_node(aid)
        G.nodes[aid]["label"] = safe_str(node_label.get(aid, aid))
        G.nodes[aid]["tickets_touched"] = int(len(tickets_touched.get(aid, set())))

        # role counts
        for role, c in node_roles[aid].items():
            G.nodes[aid][f"role_{role}"] = int(c)

        # a simple "is_assignee_present" indicator (helps in Gephi filtering)
        G.nodes[aid]["ever_assignee"] = int(node_roles[aid].get("assignee", 0) > 0)

    # write nodes CSV
    node_fields = ["id", "label", "tickets_touched", "ever_assignee"] + sorted(
        {f"role_{r}" for aid in node_roles for r in node_roles[aid]}
    )
    with open(args.out_nodes, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=node_fields)
        w.writeheader()
        for n, data in G.nodes(data=True):
            row = {k: "" for k in node_fields}
            row["id"] = n
            for k in node_fields:
                if k in data:
                    row[k] = data[k]
            w.writerow(row)

    # write edges CSV
    # include per-layer weights if present
    layer_cols = sorted({k for _, _, d in G.edges(data=True) for k in d.keys() if k.startswith("w_")})
    edge_fields = ["source", "target", "weight"] + layer_cols
    with open(args.out_edges, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=edge_fields)
        w.writeheader()
        for u, v, d in G.edges(data=True):
            row = {"source": u, "target": v, "weight": float(d.get("weight", 0.0))}
            for c in layer_cols:
                row[c] = float(d.get(c, 0.0))
            w.writerow(row)

    # write GraphML (safe with NumPy 2.x; Gephi reads it)
    nx.write_graphml(G, args.out_graphml)

    # quick summary
    print(f"Scanned issues: {scanned}")
    print(f"Transfer events found: {transfer_events}")
    print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    try:
        # density for undirected; for directed GraphML we still show nx.density
        print(f"Density: {nx.density(G):.6f}")
    except Exception:
        pass
    print(f"Wrote nodes -> {args.out_nodes}")
    print(f"Wrote edges -> {args.out_edges}")
    print(f"Wrote graph -> {args.out_graphml}")


if __name__ == "__main__":
    main()
