"""
ETL pipeline: load tickets → embed (cached) → SQLite.

Two entry points:
  run_pipeline(cfg, ...)                  - loads from JSONL
  run_pipeline_from_tickets(tickets, cfg) - accepts pre-loaded tickets (e.g. from MongoDB)

Both return (keys, vectors, store) where vectors is L2-normalised.
"""
from typing import List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import normalize

from ticketing_intel.config import Config
from ticketing_intel.etl.loader import load_tickets, TicketRecord
from ticketing_intel.etl.embedder import EmbeddingCache, embed_tickets
from ticketing_intel.store.sqlite_store import TicketStore


def run_pipeline_from_tickets(
    tickets: List[TicketRecord],
    cfg: Config,
    batch_size: int = 128,
) -> Tuple[List[str], np.ndarray, TicketStore]:
    """
    Core pipeline: takes pre-loaded TicketRecord list, stores metadata,
    embeds (with cache), and returns aligned (keys, vectors, store).
    """
    if not tickets:
        raise RuntimeError("No tickets provided to pipeline.")

    store = TicketStore(cfg.db_path)
    store.upsert(tickets)
    print(f"[pipeline] SQLite now has {store.count()} tickets total.")

    cache = EmbeddingCache(cfg.embeddings_path)
    keys, vectors = embed_tickets(
        tickets=tickets,
        cache=cache,
        provider=cfg.embedding_provider,
        model=cfg.embedding_model,
        openai_api_key=cfg.openai_api_key,
        voyage_api_key=cfg.voyage_api_key,
        batch_size=batch_size,
    )

    vectors = normalize(vectors, norm="l2")

    print(f"\n[pipeline] Complete. {len(keys)} tickets ready.")
    print(f"           Embedding shape: {vectors.shape}")
    print(f"           Cache:           {cfg.embeddings_path}")
    print(f"           DB:              {cfg.db_path}\n")

    return keys, vectors, store


def run_pipeline(
    cfg: Config,
    limit: Optional[int] = None,
    include_comments: bool = True,
    batch_size: int = 128,
) -> Tuple[List[str], np.ndarray, TicketStore]:
    """
    Run the full ETL pipeline from a JSONL file.

    Returns:
        keys:    List of ticket keys, aligned with vectors
        vectors: np.ndarray shape (N, embedding_dim), L2-normalised
        store:   TicketStore for metadata queries
    """
    cfg.validate()

    print(f"\n[pipeline] Loading tickets from {cfg.jira_dump_path} ...")
    tickets = load_tickets(
        cfg.jira_dump_path,
        limit=limit,
        include_comments=include_comments,
    )
    print(f"[pipeline] Loaded {len(tickets)} tickets.")

    if not tickets:
        raise RuntimeError(f"No tickets found in {cfg.jira_dump_path}")

    return run_pipeline_from_tickets(tickets, cfg, batch_size=batch_size)
