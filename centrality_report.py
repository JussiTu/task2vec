import argparse
import pandas as pd
import networkx as nx
import numpy as np

LAYER_COLS = [
    "w_assignee_transfer",
    "w_changelog_to_assignee",
    "w_changelog_to_creator",
    "w_comment_coauthor",
    "w_comment_to_assignee",
    "w_role_creator_assignee",
    "w_role_creator_reporter",
    "w_role_reporter_assignee",
]

def load_graph(edges_csv: str, weight_col: str = "weight") -> nx.Graph:
    df = pd.read_csv(edges_csv)
    # keep only positive edges
    df = df[df[weight_col].fillna(0) > 0].copy()

    G = nx.Graph()
    for _, r in df.iterrows():
        u, v = r["source"], r["target"]
        w = float(r[weight_col])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G

def add_strength(G: nx.Graph) -> dict:
    # weighted degree
    return dict(G.degree(weight="weight"))

def safe_giant_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    gcc = max(comps, key=len)
    return G.subgraph(gcc).copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="transfer_nodes.csv")
    ap.add_argument("--edges", required=True, help="transfer_edges.csv")
    ap.add_argument("--out", default="centrality_report.csv")
    ap.add_argument("--layer", default="weight", help="weight or one of w_* columns")
    ap.add_argument("--topn", type=int, default=50)
    args = ap.parse_args()

    nodes = pd.read_csv(args.nodes)

    # Build graph using chosen weight column
    G = load_graph(args.edges, weight_col=args.layer)

    # Compute metrics on full graph + on giant component (for closeness)
    strength = add_strength(G)

    # Betweenness & PageRank use distance interpretation.
    # If higher weight = stronger tie, we convert to distance = 1/weight.
    for u, v, d in G.edges(data=True):
        w = max(float(d.get("weight", 0.0)), 1e-12)
        d["distance"] = 1.0 / w

    # Betweenness (weighted via distance)
    betw = nx.betweenness_centrality(G, weight="distance", normalized=True)

    # PageRank (weighted by tie strength)
    pr = nx.pagerank(G, weight="weight")

    # Closeness is best on giant component only
    Gg = safe_giant_component(G)
    clos = {}
    if Gg.number_of_nodes() > 1:
        for u, v, d in Gg.edges(data=True):
            w = max(float(d.get("weight", 0.0)), 1e-12)
            d["distance"] = 1.0 / w
        clos = nx.closeness_centrality(Gg, distance="distance")
    else:
        clos = {n: 0.0 for n in G.nodes()}

    # Degree (unweighted) also useful
    deg = dict(G.degree())

    # Merge into dataframe
    out = nodes.copy()
    out["degree"] = out["id"].map(deg).fillna(0).astype(int)
    out["strength"] = out["id"].map(strength).fillna(0.0)
    out["betweenness"] = out["id"].map(betw).fillna(0.0)
    out["closeness"] = out["id"].map(clos).fillna(0.0)
    out["pagerank"] = out["id"].map(pr).fillna(0.0)

    # A simple “human vs bot-ish” heuristic (optional)
    # High changelog author count but never assignee tends to be system-like.
    if "role_changelog_author" in out.columns and "ever_assignee" in out.columns:
        out["bot_score_hint"] = (
            out["role_changelog_author"].fillna(0).astype(float) /
            (out["tickets_touched"].replace(0, np.nan).fillna(1.0))
        )
    else:
        out["bot_score_hint"] = 0.0

    # Sort by pagerank by default
    out = out.sort_values(["pagerank", "strength", "betweenness"], ascending=False)

    out.to_csv(args.out, index=False)

    # Print topn
    cols_show = ["label", "tickets_touched", "degree", "strength", "pagerank", "betweenness", "closeness"]
    cols_show = [c for c in cols_show if c in out.columns]
    print(f"\nTop {args.topn} by pagerank ({args.layer}):")
    print(out[cols_show].head(args.topn).to_string(index=False))

if __name__ == "__main__":
    main()
