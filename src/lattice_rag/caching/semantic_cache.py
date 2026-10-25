from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from typesafe_sdk import AsyncTypeSafeClient, Noul

from lattice_rag.security import sanitize_error_detail

logger = structlog.get_logger(__name__)


class CacheVerificationError(Exception):
    """A near-duplicate was found but its Noul equivalence check could not run.

    The lookup is a miss, and the answer computed instead is an Unverified Answer.
    """


@dataclass
class CacheEntry:
    query_text: str
    query_embedding: np.ndarray
    response: dict[str, Any]
    created_at: float


class SemanticCache:
    """Tier 1 Semantic Vector Cache.

    Stores prior query embeddings and responses in-memory with O(1) exact match,
    vectorized BLAS dot product similarity, and bounded LRU eviction.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        ts_client: AsyncTypeSafeClient | None = None,
        max_size: int = 1000,
        vector_dimensions: int = 384,
    ) -> None:
        """Initialize empty cache storage."""
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.vector_dimensions = vector_dimensions
        self._cache: list[CacheEntry] = []
        self._exact_map: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # Normalized query -> its row in ``_cache`` and ``_embeddings_matrix``, so writes
        # find an existing entry in O(1) instead of scanning.
        self._slots: dict[str, int] = {}
        self._embeddings_matrix: np.ndarray | None = None
        self._hits = 0
        self._misses = 0
        self._exact_hits = 0
        self._fuzzy_hits = 0
        self._evictions = 0
        self._ts_client: AsyncTypeSafeClient | None = None

        if ts_client is not None:
            self._ts_client = ts_client
        elif os.environ.get("TYPESAFE_API_KEY"):
            self._ts_client = AsyncTypeSafeClient()
        else:
            self._ts_client = None

    def __len__(self) -> int:
        """Return the number of cached items."""
        return len(self._cache)

    def get_exact(self, query_text: str) -> dict[str, Any] | None:
        """O(1) exact-match shortcut bypassing embedding and Jev Noul verification."""
        norm_key = query_text.strip().lower()
        if norm_key in self._exact_map:
            self._exact_map.move_to_end(norm_key)
            self._hits += 1
            self._exact_hits += 1
            logger.info("cache_hit_exact")
            return self._exact_map[norm_key]
        return None

    async def get(self, query_text: str, query_embedding: np.ndarray) -> dict[str, Any] | None:
        """Search cache for semantic match using vectorized BLAS dot product and Jev Noul.

        Raises :class:`CacheVerificationError` when a near-duplicate exists but Noul could
        not verify it (no client, or the call failed); callers treat that as a miss whose
        answer is unverified.
        """
        if not self._cache or self._embeddings_matrix is None:
            self._misses += 1
            return None

        # 2. Vectorized BLAS Cosine Similarity
        norm_q = float(np.linalg.norm(query_embedding))
        if norm_q == 0.0:
            self._misses += 1
            return None

        q_unit = (query_embedding / norm_q).astype(np.float32)
        n_entries = len(self._cache)
        # Single C-level BLAS dot product across all cached embeddings: O(N * D)
        sims = np.dot(self._embeddings_matrix[:n_entries], q_unit)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_entry = self._cache[best_idx]

        if best_score >= self.similarity_threshold:
            if self._ts_client is None:
                logger.warning("ts_client_not_configured_for_semantic_verification")
                self._misses += 1
                raise CacheVerificationError("no Jev client configured for Noul verification")

            try:
                # Use TypeSafe AI Jev Noul to verify semantic equivalence
                response = await self._ts_client.system_one(
                    state={
                        "Query A": best_entry.query_text,
                        "Query B": query_text,
                    },
                    questions={
                        "equivalent": Noul(
                            instructions="Is Query B asking for the exact same information as Query A?"
                        ),
                    },
                )
                noul_result = response.nouls.get("equivalent")
                if noul_result is not None and noul_result.noul >= 0.70:
                    self._hits += 1
                    self._fuzzy_hits += 1
                    best_norm = best_entry.query_text.strip().lower()
                    if best_norm in self._exact_map:
                        self._exact_map.move_to_end(best_norm)
                    logger.info(
                        "cache_hit_fuzzy",
                        score=best_score,
                        noul_prob=noul_result.noul,
                    )
                    return best_entry.response
                else:
                    noul_prob = noul_result.noul if noul_result else 0.0
                    logger.info(
                        "cache_noul_rejected",
                        score=best_score,
                        noul_prob=noul_prob,
                    )
            except Exception as e:
                logger.error("noul_verification_failed", error=sanitize_error_detail(str(e)))
                self._misses += 1
                raise CacheVerificationError("Noul verification could not run") from e

        self._misses += 1
        return None

    def _remove_at(self, idx: int) -> None:
        """Drop entry idx in O(1) by moving the last entry (and matrix row) into its slot."""
        assert self._embeddings_matrix is not None
        last = len(self._cache) - 1
        removed = self._cache[idx]
        if idx < last:
            moved = self._cache[last]
            self._cache[idx] = moved
            self._embeddings_matrix[idx] = self._embeddings_matrix[last]
            self._slots[moved.query_text.strip().lower()] = idx
        self._cache.pop()
        self._embeddings_matrix[last] = 0.0
        self._slots.pop(removed.query_text.strip().lower(), None)

    def put(self, query_text: str, query_embedding: np.ndarray, response: dict[str, Any]) -> None:
        """Store an entry with a normalized vector and bounded LRU eviction.

        Invariant: exactly one ``_cache`` entry (and matrix row) per normalized query, and
        ``_exact_map`` holds the same keys in LRU order. Re-putting a query replaces it.
        """
        norm_key = query_text.strip().lower()

        # Compute normalized unit vector for fast BLAS dot product
        norm_val = float(np.linalg.norm(query_embedding))
        norm_vec = (
            (query_embedding / norm_val).astype(np.float32)
            if norm_val > 0.0
            else query_embedding.astype(np.float32)
        )

        # Initialize or adjust embeddings matrix to match vector dimension
        vec_dim = len(norm_vec)
        if self._embeddings_matrix is None or self._embeddings_matrix.shape[1] != vec_dim:
            self._embeddings_matrix = np.zeros((self.max_size, vec_dim), dtype=np.float32)
            for i, c_entry in enumerate(self._cache):
                if len(c_entry.query_embedding) == vec_dim:
                    self._embeddings_matrix[i] = c_entry.query_embedding

        entry = CacheEntry(
            query_text=query_text,
            query_embedding=norm_vec,
            response=response,
            created_at=time.time(),
        )

        existing_idx = self._slots.get(norm_key)
        if existing_idx is not None:
            self._cache[existing_idx] = entry
            self._embeddings_matrix[existing_idx] = norm_vec
        else:
            if len(self._cache) >= self.max_size:
                self._evictions += 1
                oldest_key, _ = self._exact_map.popitem(last=False)
                oldest_idx = self._slots.get(oldest_key)
                if oldest_idx is not None:
                    self._remove_at(oldest_idx)
            self._slots[norm_key] = len(self._cache)
            self._embeddings_matrix[len(self._cache)] = norm_vec
            self._cache.append(entry)

        self._exact_map[norm_key] = response
        self._exact_map.move_to_end(norm_key)
        logger.debug("cache_put", size=len(self._cache))

    @property
    def stats(self) -> tuple[int, int]:
        """Returns (hits, misses) counts."""
        return self._hits, self._misses

    @property
    def detailed_stats(self) -> dict[str, Any]:
        """Returns comprehensive cache diagnostics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "exact_hits": self._exact_hits,
            "fuzzy_hits": self._fuzzy_hits,
            "evictions": self._evictions,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_ratio": (self._hits / total) if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clears all entries."""
        self._cache.clear()
        self._exact_map.clear()
        self._slots.clear()
        if self._embeddings_matrix is not None:
            self._embeddings_matrix.fill(0.0)
        self._hits = 0
        self._misses = 0
        self._exact_hits = 0
        self._fuzzy_hits = 0
        self._evictions = 0
        logger.info("cache_cleared")

    async def aclose(self) -> None:
        """Closes the underlying TypeSafe AI client."""
        if self._ts_client is not None:
            await self._ts_client.aclose()
            logger.debug("SemanticCache client closed")

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()
