import json
import csv
import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional

import networkx as nx


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


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def get_nested(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def extract_issue_key(doc: Dict[str, Any]) -> str:
    return safe_str(doc.get("key") or doc.get("id") or "")


def normalize_user_id(x: Any) -> str:
    """
    The dataset is pseudonymized; IDs may be in from/to strings or nested objects.
    We normalize to a stable string.
    """
    s = safe_str(x)
    if not s:
        return ""
    return s


def extract_assignee_transfer_events(doc: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Returns list of (from_assignee, to_assignee) transitions.
    Pulls from changelog.histories[].items[] where field == 'assignee'.

    Handles cases where from/to might be in:
      - item.from / item.to
      - item.fromString / item.toString
    """
    out: List[Tuple[str, str]] = []
    histories = get_nested(doc, ["changelog", "histories"], default=[])
    if not isinstance(histories, list):
        return out

    for h in histories:
        items = h.get("items") if isinstance(h, dict) else None
        if not isinstance(items, list):
            continue

        for it in items:
            if not isinstance(it, dict):
                continue
            if safe_str(it.get("field")) != "assignee":
                continue

            frm = it.get("fromString")
            to = it.get("toString")
            if frm is None and "from" in it:
                frm = it.get("from")
            if to is None and "to" in it:
                to = it.get("to")

            frm_id = normalize_user_id(frm)
            to_id = normalize_user_id(to)

            # Sometimes "from" is empty when assignee is first set.
            # We can still include it, but it creates a blank node; skip blank endpoints.
            if not frm_id or not to_id:
                continue

            if frm_id == to_id:
                continue

            out.append((frm_id, to_id))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jira_1000.jsonl (or larger)")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--min_edge_weight", type=int, default=1)
    ap.add_argument("--out_edges", default="transfer_edges.csv")
    ap.add_argument("--out_nodes", default="transfer_nodes.csv")
    ap.add_argument("--out_gexf", default="transfer_graph.gexf")
    args = ap.parse_args()

    # Build weighted directed graph
    edge_weights = Counter()
    issue_count = 0
    issues_with_transfers = 0
    total_transfers = 0

    for doc in iter_jsonl(args.input, args.limit):
        issue_count += 1
        events = extract_assignee_transfer_events(doc)
        if events:
            issues_with_transfers += 1
        for frm, to in events:
            edge_weights[(frm, to)] += 1
            total_transfers += 1

    # Create graph
    G = nx.DiGraph()
    for (frm, to), w in edge_weights.items():
        if w >= args.min_edge_weight:
            G.add_edge(frm, to, weight=int(w))

    # Basic stats
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0

    print(f"Scanned issues: {issue_count}")
    print(f"Issues with assignee transfers: {issues_with_transfers}")
    print(f"Total transfer events found: {total_transfers}")
    print(f"Graph nodes: {n}, edges: {m}, density: {density:.6f}")

    if n == 0:
        print("No assignee transitions found in this sample. Try a larger export or verify changelog is present.")
        return

    # Centralities (on directed graph)
    # - Degree centrality uses underlying simple definition; NetworkX provides separate in/out degrees too.
    deg_in = dict(G.in_degree())
    deg_out = dict(G.out_degree())
    wdeg_in = dict(G.in_degree(weight="weight"))
    wdeg_out = dict(G.out_degree(weight="weight"))

    # Betweenness can be slow on huge graphs; fine for ~1000 issues
    betw = nx.betweenness_centrality(G, normalized=True, weight=None)

    # Eigenvector centrality for directed graphs can fail if not strongly connected; handle safely
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {node: 0.0 for node in G.nodes()}

    # PageRank is often more stable for directed graphs
    pr = nx.pagerank(G, weight="weight")

    # Write edges CSV
    with open(args.out_edges, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from_assignee", "to_assignee", "weight"])
        for (frm, to), wt in edge_weights.most_common():
            if wt >= args.min_edge_weight:
                w.writerow([frm, to, int(wt)])

    print(f"Wrote edges -> {args.out_edges}")

    # Write nodes CSV
    with open(args.out_nodes, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "assignee_id",
            "in_degree", "out_degree",
            "in_weight", "out_weight",
            "betweenness",
            "pagerank",
            "eigenvector"
        ])
        # sort by pagerank
        for node, _ in sorted(pr.items(), key=lambda x: -x[1]):
            w.writerow([
                node,
                deg_in.get(node, 0), deg_out.get(node, 0),
                int(wdeg_in.get(node, 0)), int(wdeg_out.get(node, 0)),
                float(betw.get(node, 0.0)),
                float(pr.get(node, 0.0)),
                float(eig.get(node, 0.0)),
            ])

    print(f"Wrote nodes -> {args.out_nodes}")

    # Write GEXF for Gephi
    nx.write_gexf(G, args.out_gexf)
    print(f"Wrote graph -> {args.out_gexf}")
    print("Tip: open .gexf in Gephi, size nodes by PageRank, color by modularity community.")


if __name__ == "__main__":
    main()
