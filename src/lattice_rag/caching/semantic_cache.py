from __future__ import annotations

import os
import time
import numpy as np
import structlog
from dataclasses import dataclass
from typesafe_sdk import AsyncTypeSafeClient, Noul

logger = structlog.get_logger(__name__)

@dataclass
class CacheEntry:
    query_text: str
    query_embedding: np.ndarray
    response: dict
    created_at: float


class SemanticCache:
    """Tier 1 Semantic Vector Cache.
    
    Stores prior query embeddings and responses in-memory.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        ts_client: AsyncTypeSafeClient | None = None,
    ) -> None:
        """Initialize empty cache storage."""
        self.similarity_threshold = similarity_threshold
        self._cache: list[CacheEntry] = []
        self._hits = 0
        self._misses = 0
        if ts_client is not None:
            self._ts_client = ts_client
        elif os.environ.get("TYPESAFE_API_KEY"):
            self._ts_client = AsyncTypeSafeClient()
        else:
            self._ts_client = None

    async def get(self, query_text: str, query_embedding: np.ndarray) -> dict | None:
        """Search cache for semantic match."""
        if not self._cache:
            self._misses += 1
            return None

        best_score = -1.0
        best_entry = None

        norm_q = np.linalg.norm(query_embedding)
        if norm_q == 0:
            self._misses += 1
            return None

        for entry in self._cache:
            norm_e = np.linalg.norm(entry.query_embedding)
            if norm_e == 0:
                continue
            
            sim = np.dot(query_embedding, entry.query_embedding) / (norm_q * norm_e)
            if sim > best_score:
                best_score = float(sim)
                best_entry = entry

        if best_score >= self.similarity_threshold and best_entry is not None:
            if self._ts_client is None:
                logger.warning("ts_client_not_configured_for_semantic_verification")
                self._misses += 1
                return None

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
                    logger.info(
                        "cache_hit",
                        query=query_text,
                        score=best_score,
                        noul_prob=noul_result.noul,
                    )
                    return best_entry.response
                else:
                    noul_prob = noul_result.noul if noul_result else 0.0
                    logger.info(
                        "cache_noul_rejected",
                        query=query_text,
                        score=best_score,
                        noul_prob=noul_prob,
                    )
            except Exception as e:
                logger.error("noul_verification_failed", error=str(e))
        
        self._misses += 1
        return None

    def put(self, query_text: str, query_embedding: np.ndarray, response: dict) -> None:
        """Store a new entry."""
        entry = CacheEntry(
            query_text=query_text,
            query_embedding=query_embedding,
            response=response,
            created_at=time.time()
        )
        self._cache.append(entry)
        logger.debug("cache_put", query=query_text)

    @property
    def stats(self) -> tuple[int, int]:
        """Returns (hits, misses) counts."""
        return self._hits, self._misses

    def clear(self) -> None:
        """Clears all entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("cache_cleared")

    async def aclose(self) -> None:
        """Closes the underlying TypeSafe AI client."""
        if self._ts_client is not None:
            await self._ts_client.aclose()
            logger.debug("SemanticCache client closed")

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()

