"""
core/cache.py — Thread-safe LRU cache for embedding vectors.

Key design decisions:
  - Keyed by SHA-256 hash of the raw text string (collision-resistant, fast).
  - Backed by collections.OrderedDict for O(1) LRU eviction without extra deps.
  - Thread-safe via threading.RLock (supports nested calls from the same thread).
  - Tracks hits / misses for monitoring (exposed via CacheStats schema).
  - Serialisable to / from disk (numpy .npz + JSON metadata) for warm-start on
    server restart.

Environment variables (read once at import time, overrideable via constructor):
  CACHE_SIZE   — maximum number of cached embeddings (default: 10000)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_SIZE = int(os.getenv("CACHE_SIZE", "10000"))
_DEFAULT_PERSIST_PATH = os.getenv("CACHE_PERSIST_PATH", "./index/cache.npz")


def _hash_text(text: str) -> str:
    """Return a compact hex SHA-256 digest of *text*."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class EmbeddingCache:
    """LRU cache mapping text → 1-D numpy embedding vector."""

    def __init__(
        self,
        max_size: int = _DEFAULT_CACHE_SIZE,
        persist_path: Optional[str] = _DEFAULT_PERSIST_PATH,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")

        self._max_size = max_size
        self._persist_path = Path(persist_path) if persist_path else None
        self._lock = threading.RLock()
        # OrderedDict: key = hex hash, value = (original_text_key, np.ndarray)
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> Optional[np.ndarray]:
        """Return the cached embedding for *text*, or None on a cache miss.

        A hit moves the entry to the MRU (most-recently-used) end.
        """
        key = _hash_text(text)
        with self._lock:
            if key in self._store:
                # Move to MRU end
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Insert or update the embedding for *text*.

        If the cache is full the LRU entry is evicted first.
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.asarray(embedding, dtype=np.float32)
        else:
            embedding = embedding.astype(np.float32, copy=False)

        key = _hash_text(text)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            else:
                if len(self._store) >= self._max_size:
                    evicted_key, _ = self._store.popitem(last=False)
                    logger.debug("Cache evicted key %s (LRU)", evicted_key[:8])
            self._store[key] = embedding

    def get_batch(
        self, texts: List[str]
    ) -> Tuple[List[Optional[np.ndarray]], List[int], List[int]]:
        """Batch get for a list of texts.

        Returns:
            embeddings  — list aligned with *texts*; None where cache missed.
            hit_indices  — indices into *texts* that were cache hits.
            miss_indices — indices into *texts* that were cache misses.
        """
        results: List[Optional[np.ndarray]] = []
        hit_indices: List[int] = []
        miss_indices: List[int] = []

        for i, text in enumerate(texts):
            vec = self.get(text)
            results.append(vec)
            if vec is not None:
                hit_indices.append(i)
            else:
                miss_indices.append(i)

        return results, hit_indices, miss_indices

    def put_batch(self, texts: List[str], embeddings: List[np.ndarray]) -> None:
        """Insert a batch of (text, embedding) pairs."""
        if len(texts) != len(embeddings):
            raise ValueError(
                f"texts ({len(texts)}) and embeddings ({len(embeddings)}) must have the same length"
            )
        for text, emb in zip(texts, embeddings):
            self.put(text, emb)

    def invalidate(self, text: str) -> bool:
        """Remove *text* from the cache. Returns True if it was present."""
        key = _hash_text(text)
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Evict all entries and reset statistics."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
        logger.info("Cache cleared.")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def hit_rate(self) -> float:
        """Fraction of gets that were hits (0.0 – 1.0)."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
            }

    # ------------------------------------------------------------------
    # Persistence (warm-start)
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist the cache to a .npz file for warm-start on restart."""
        target = Path(path) if path else self._persist_path
        if target is None:
            logger.warning("No persist path configured; skipping cache save.")
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            keys = list(self._store.keys())
            arrays = [self._store[k] for k in keys]

        if not keys:
            logger.info("Cache is empty; nothing to save.")
            return

        stacked = np.stack(arrays, axis=0)  # shape: (N, dims)
        meta_path = target.with_suffix(".cache_meta.json")

        np.savez_compressed(str(target), embeddings=stacked)
        meta_path.write_text(json.dumps(keys), encoding="utf-8")
        logger.info("Cache saved: %d entries → %s", len(keys), target)

    def load(self, path: Optional[str] = None) -> int:
        """Restore the cache from a .npz file.  Returns the number of entries loaded."""
        source = Path(path) if path else self._persist_path
        if source is None or not source.exists():
            logger.info("No cache file at %s; starting cold.", source)
            return 0

        meta_path = source.with_suffix(".cache_meta.json")
        if not meta_path.exists():
            logger.warning("Cache metadata missing at %s; skipping load.", meta_path)
            return 0

        try:
            data = np.load(str(source))
            embeddings: np.ndarray = data["embeddings"]  # (N, dims)
            keys: List[str] = json.loads(meta_path.read_text(encoding="utf-8"))

            if len(keys) != embeddings.shape[0]:
                logger.error(
                    "Cache file mismatch: %d keys vs %d embeddings; skipping.",
                    len(keys),
                    embeddings.shape[0],
                )
                return 0

            with self._lock:
                self._store.clear()
                for key, vec in zip(keys, embeddings):
                    if len(self._store) < self._max_size:
                        self._store[key] = vec
                    else:
                        break  # Respect max_size even during warm-start

            loaded = len(self._store)
            logger.info("Cache warm-started: %d entries loaded from %s", loaded, source)
            return loaded

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load cache from %s: %s", source, exc)
            return 0
