"""
Embedding layer with persistent disk cache.

The cache stores a .npz file with:
  - keys:    string array of ticket keys (e.g. "PROJ-123")
  - vectors: float32 matrix, shape (N, embedding_dim)

On each ETL run:
  1. Load existing cache (if any)
  2. Find tickets NOT yet in cache
  3. Embed only those new tickets (API call)
  4. Merge + save updated cache

This means you only pay for API calls once per ticket.
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ticketing_intel.etl.loader import TicketRecord


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """
    Thin wrapper around a .npz file that maps ticket_key → embedding vector.
    """

    def __init__(self, path: Path):
        self.path = path
        self._keys: List[str] = []
        self._vectors: Optional[np.ndarray] = None  # shape (N, dim)
        self._index: Dict[str, int] = {}            # key → row index
        self._load()

    def _load(self):
        if self.path.exists():
            data = np.load(self.path, allow_pickle=True)
            self._keys = list(data["keys"])
            self._vectors = data["vectors"].astype(np.float32)
            self._index = {k: i for i, k in enumerate(self._keys)}
            print(f"[cache] Loaded {len(self._keys)} cached embeddings from {self.path}")
        else:
            print(f"[cache] No cache found at {self.path} - will create fresh.")

    def save(self):
        if not self._keys:
            return
        np.savez(
            self.path,
            keys=np.array(self._keys),
            vectors=self._vectors,
        )
        print(f"[cache] Saved {len(self._keys)} embeddings -> {self.path}")

    def missing_keys(self, keys: List[str]) -> List[str]:
        return [k for k in keys if k not in self._index]

    def add(self, keys: List[str], vectors: np.ndarray):
        """Append new key-vector pairs to the cache."""
        if not keys:
            return
        vectors = vectors.astype(np.float32)
        if self._vectors is None:
            self._vectors = vectors
            self._keys = list(keys)
        else:
            self._vectors = np.vstack([self._vectors, vectors])
            self._keys.extend(keys)
        # Rebuild index
        self._index = {k: i for i, k in enumerate(self._keys)}

    def get(self, keys: List[str]) -> Tuple[List[str], np.ndarray]:
        """Return (found_keys, vectors) for the requested keys that exist in cache."""
        found_keys = [k for k in keys if k in self._index]
        if not found_keys:
            return [], np.empty((0, 0), dtype=np.float32)
        idxs = [self._index[k] for k in found_keys]
        return found_keys, self._vectors[idxs]

    def get_all(self) -> Tuple[List[str], np.ndarray]:
        """Return all cached (keys, vectors)."""
        if self._vectors is None:
            return [], np.empty((0, 0), dtype=np.float32)
        return list(self._keys), self._vectors

    def __len__(self):
        return len(self._keys)


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

def _embed_openai(
    texts: List[str],
    model: str,
    api_key: str,
    batch_size: int = 128,
    checkpoint_every: int = 1000,
    cache: "EmbeddingCache" = None,
    keys_for_checkpoint: List[str] = None,
) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    vecs: List[List[float]] = []
    total = len(texts)
    last_ckpt = 0
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        if i + batch_size < total:
            time.sleep(0.2)
        # Incremental checkpoint: only save the delta since last checkpoint
        if cache is not None and keys_for_checkpoint is not None:
            done = len(vecs)
            if done - last_ckpt >= checkpoint_every:
                print(f"[embed] {done:,}/{total:,} ... saving checkpoint")
                delta_keys = keys_for_checkpoint[last_ckpt:done]
                delta_vecs = np.array(vecs[last_ckpt:done], dtype=np.float32)
                cache.add(delta_keys, delta_vecs)
                cache.save()
                last_ckpt = done
    return np.array(vecs, dtype=np.float32)


def _embed_voyage(
    texts: List[str],
    model: str,
    api_key: str,
    batch_size: int = 64,
    checkpoint_every: int = 1000,
    cache: "EmbeddingCache" = None,
    keys_for_checkpoint: List[str] = None,
) -> np.ndarray:
    import voyageai
    client = voyageai.Client(api_key=api_key)
    vecs: List[List[float]] = []
    total = len(texts)
    last_ckpt = 0
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        result = client.embed(batch, model=model, input_type="document")
        vecs.extend(result.embeddings)
        if i + batch_size < total:
            time.sleep(0.2)
        # Incremental checkpoint: only save the delta since last checkpoint
        if cache is not None and keys_for_checkpoint is not None:
            done = len(vecs)
            if done - last_ckpt >= checkpoint_every:
                print(f"[embed] {done:,}/{total:,} ... saving checkpoint")
                delta_keys = keys_for_checkpoint[last_ckpt:done]
                delta_vecs = np.array(vecs[last_ckpt:done], dtype=np.float32)
                cache.add(delta_keys, delta_vecs)
                cache.save()
                last_ckpt = done
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def embed_tickets(
    tickets: List[TicketRecord],
    cache: EmbeddingCache,
    provider: str,
    model: str,
    openai_api_key: str = "",
    voyage_api_key: str = "",
    batch_size: int = 128,
    checkpoint_every: int = 1000,
) -> Tuple[List[str], np.ndarray]:
    """
    Embed all tickets, using cache for already-seen tickets.
    Saves incremental checkpoints every `checkpoint_every` tickets
    so progress is not lost if the process is interrupted.
    Returns (all_keys, all_vectors) covering every ticket in the input list.
    """
    all_keys = [t.key for t in tickets]
    missing = cache.missing_keys(all_keys)

    if missing:
        print(f"[embed] {len(missing)} new tickets to embed (provider={provider}, model={model})")
        key_to_text = {t.key: t.embed_text for t in tickets}
        texts = [key_to_text[k] for k in missing]

        if provider == "openai":
            new_vecs = _embed_openai(
                texts, model, openai_api_key, batch_size,
                checkpoint_every=checkpoint_every,
                cache=cache, keys_for_checkpoint=missing,
            )
        elif provider == "voyage":
            new_vecs = _embed_voyage(
                texts, model, voyage_api_key, batch_size,
                checkpoint_every=checkpoint_every,
                cache=cache, keys_for_checkpoint=missing,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider!r}. Use 'openai' or 'voyage'.")

        # Add any vectors not yet saved by a checkpoint
        still_missing = cache.missing_keys(missing)
        if still_missing:
            missing_map = {k: i for i, k in enumerate(missing)}
            idxs = [missing_map[k] for k in still_missing]
            cache.add(still_missing, new_vecs[idxs])
            cache.save()
        print(f"[embed] Done. Cache now has {len(cache)} embeddings.")
    else:
        print(f"[embed] All {len(all_keys)} tickets already cached - no API calls needed.")

    _, vectors = cache.get(all_keys)
    return all_keys, vectors
