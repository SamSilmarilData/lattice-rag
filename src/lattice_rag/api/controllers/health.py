from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from litestar import Controller, get
from litestar.di import NamedDependency
from litestar.params import SkipValidation

from lattice_rag.api.dtos import HealthResponse
from lattice_rag.caching.fallback_cache import FallbackCache
from lattice_rag.config import AppConfig
from lattice_rag.storage.db import LatticeStore

logger = logging.getLogger(__name__)


class HealthController(Controller):
    """Litestar controller for health and connectivity telemetry."""

    path = ""

    @get("/health")
    async def health_check(
        self,
        store: NamedDependency[SkipValidation[LatticeStore]],
        fallback_cache: NamedDependency[SkipValidation[FallbackCache]],
        config: NamedDependency[SkipValidation[AppConfig]],
    ) -> HealthResponse:
        """Check the health status of LatticeDB, Redis, and API key configurations."""
        logger.info("running_health_check")

        latticedb_ok = bool(store and hasattr(store, "db") and store.db.is_open)
        redis_ok = bool(fallback_cache and fallback_cache.redis is not None)

        typesafe_ok = bool(config.typesafe_api_key and config.typesafe_api_key.get_secret_value())
        groq_ok = bool(config.groq_api_key and config.groq_api_key.get_secret_value())
        gemini_ok = bool(config.gemini_api_key and config.gemini_api_key.get_secret_value())

        overall_status = "healthy" if (latticedb_ok and redis_ok and typesafe_ok) else "degraded"

        return HealthResponse(
            status=overall_status,
            latticedb_connected=latticedb_ok,
            redis_connected=redis_ok,
            typesafe_configured=typesafe_ok,
            groq_configured=groq_ok,
            gemini_configured=gemini_ok,
        )
