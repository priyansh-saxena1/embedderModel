"""
core/embedder.py — Sentence-transformer wrapper with batched inference.

Responsibilities:
  - Lazy-loads the model on first use so server startup stays fast.
  - Normalises embeddings to unit vectors (enables cosine similarity via
    inner-product search in FAISS).
  - Integrates transparently with the LRU cache: cache hits are returned
    immediately; only misses reach the model.
  - Runs inference in configurable batches (BATCH_SIZE env var, default 64).
  - thread-safe (single global lock around model.encode calls — sentence-
    transformers is not multi-thread-safe by default).

Environment variables:
  MODEL_NAME   — HuggingFace model id (default: all-MiniLM-L6-v2)
  BATCH_SIZE   — sentences per inference call (default: 64)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from core.cache import EmbeddingCache
from core.schemas import EmbedMetadata, EmbedResponse

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
_DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))


class EmbeddingEngine:
    """Thread-safe sentence-transformer wrapper with cache integration."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        cache: Optional[EmbeddingCache] = None,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize
        self._cache = cache

        # Model is loaded lazily on first call to avoid blocking startup
        self._model = None
        self._model_lock = threading.Lock()

        # Introspected once after first load
        self._dimensions: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Embedding dimensionality (inferred from model on first use)."""
        if self._dimensions is None:
            self._ensure_model_loaded()
        return self._dimensions  # type: ignore[return-value]

    def embed(self, texts: List[str]) -> EmbedResponse:
        """Embed a list of texts, using the cache where available.

        Returns an :class:`EmbedResponse` with all embeddings, dimension info,
        and latency / cache metadata.
        """
        t0 = time.perf_counter()

        embeddings, cached_count = self._embed_with_cache(texts)

        latency_ms = (time.perf_counter() - t0) * 1000

        return EmbedResponse(
            embeddings=[vec.tolist() for vec in embeddings],
            dimensions=self._dimensions or len(embeddings[0]),
            metadata=EmbedMetadata(
                model=self._model_name,
                cached_count=cached_count,
                latency_ms=round(latency_ms, 2),
            ),
        )

    def embed_raw(self, texts: List[str]) -> Tuple[np.ndarray, int]:
        """Lower-level variant — returns a (N, D) float32 ndarray and cached_count.

        Useful internally when callers need numpy arrays directly (e.g. indexer).
        """
        return self._embed_with_cache(texts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Load the sentence-transformer model if not already loaded."""
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:  # double-checked locking
                return
            logger.info("Loading embedding model '%s' …", self._model_name)
            t0 = time.perf_counter()
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

            model = SentenceTransformer(self._model_name)
            # Use FP16 when a CUDA GPU is available for memory savings
            # (on CPU the model stays in FP32)
            self._model = model
            # Probe dimensions with a dummy sentence
            dummy = model.encode(["probe"], convert_to_numpy=True, show_progress_bar=False)
            self._dimensions = int(dummy.shape[1])
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "Model loaded: %s — dims=%d, load_time=%.0f ms",
                self._model_name,
                self._dimensions,
                elapsed,
            )

    def _embed_with_cache(
        self, texts: List[str]
    ) -> Tuple[np.ndarray, int]:
        """Return (N, D) embedding matrix; fills misses via model inference."""
        self._ensure_model_loaded()

        if self._cache is not None:
            cached_vecs, hit_indices, miss_indices = self._cache.get_batch(texts)
        else:
            cached_vecs = [None] * len(texts)
            hit_indices = []
            miss_indices = list(range(len(texts)))

        cached_count = len(hit_indices)

        # Embed only the cache-miss texts
        miss_embeddings: Optional[np.ndarray] = None
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            miss_embeddings = self._batch_encode(miss_texts)
            if self._cache is not None:
                self._cache.put_batch(miss_texts, list(miss_embeddings))

        # Reconstruct full aligned array
        result: List[np.ndarray] = []
        miss_cursor = 0
        for i in range(len(texts)):
            if cached_vecs[i] is not None:
                result.append(cached_vecs[i])  # type: ignore[arg-type]
            else:
                result.append(miss_embeddings[miss_cursor])  # type: ignore[index]
                miss_cursor += 1

        matrix = np.stack(result, axis=0).astype(np.float32)
        return matrix, cached_count

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Run model.encode in batches; optionally L2-normalise the output."""
        results: List[np.ndarray] = []
        total = len(texts)

        with self._model_lock:
            for start in range(0, total, self._batch_size):
                batch = texts[start: start + self._batch_size]
                vecs: np.ndarray = self._model.encode(  # type: ignore[union-attr]
                    batch,
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=self._normalize,
                )
                results.append(vecs)
                logger.debug(
                    "Encoded batch [%d:%d] / %d",
                    start,
                    start + len(batch),
                    total,
                )

        return np.vstack(results).astype(np.float32)
