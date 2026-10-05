from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from litestar import Controller, get
from litestar.di import NamedDependency
from litestar.params import SkipValidation

from lattice_rag.api.dtos import CacheStatsResponse
from lattice_rag.caching.fallback_cache import FallbackCache
from lattice_rag.caching.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


class CacheController(Controller):
    """Litestar controller for cache statistics and monitoring."""

    path = "/api/v1/cache"

    @get("/stats")
    async def get_stats(
        self,
        semantic_cache: NamedDependency[SkipValidation[SemanticCache]],
        fallback_cache: NamedDependency[SkipValidation[FallbackCache]],
    ) -> CacheStatsResponse:
        """Return real-time cache statistics for Tier 1 and Tier 2."""
        logger.info("fetching_cache_stats")

        tier1_hits, tier1_misses = semantic_cache.stats
        tier1_count = len(semantic_cache._cache)

        circuit_status = fallback_cache.circuit_status
        tier2_count = await fallback_cache.cached_query_count()

        return CacheStatsResponse(
            tier1_hits=tier1_hits,
            tier1_misses=tier1_misses,
            tier1_cached_queries=tier1_count,
            tier2_circuit_status=circuit_status,
            tier2_cached_queries=tier2_count,
        )
