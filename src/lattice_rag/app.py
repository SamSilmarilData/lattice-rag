from __future__ import annotations

import contextlib
from pathlib import Path
from typing import AsyncGenerator

from litestar import Litestar, MediaType, Request, Response
from litestar.config.app import AppConfig
from litestar.config.cors import CORSConfig
from litestar.datastructures import ImmutableState
from litestar.di import Provide
from litestar.logging import LoggingConfig
from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import ScalarRenderPlugin
from litestar.plugins import InitPluginProtocol
from litestar.plugins.problem_details import ProblemDetailsConfig, ProblemDetailsPlugin

from lattice_rag.api.controllers import (
    CacheController,
    HealthController,
    IngestController,
    QueryController,
)
from lattice_rag.caching.fallback_cache import FallbackCache
from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.config import get_config
from lattice_rag.config import AppConfig as LatticeAppConfig
from lattice_rag.generation.chitchat import ChitchatHandler
from lattice_rag.generation.gemini_fallback import GeminiFallback
from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.orchestration.pool import shutdown_pool
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.pipeline import RetrievalPipeline
from lattice_rag.routing.guardrail import ContextGuardrail
from lattice_rag.routing.router import QueryRouter
from lattice_rag.security import sanitize_error_detail
from lattice_rag.storage.db import LatticeStore
from lattice_rag.storage.extract import EntityExtractor


@contextlib.asynccontextmanager
async def lifespan(app: Litestar) -> AsyncGenerator[None, None]:
    """Lifespan context manager for initializing and closing dependencies."""
    config = get_config()

    # 1. Embedded storage engine
    store = LatticeStore(config.latticedb_path)

    # 2. Local embeddings and entity extraction
    embedding_service = EmbeddingService(
        embed_model=config.embed_model,
        reranker_model=config.reranker_model,
    )
    extractor = EntityExtractor()

    # 3. Decision routing & context guardrail
    router = QueryRouter()
    guardrail = ContextGuardrail()

    # 4. Multi-tier caching
    semantic_cache = SemanticCache()
    fallback_cache = FallbackCache(redis_url=config.redis_url)
    await fallback_cache.connect()

    # Seed top 50 fallback queries if file exists
    fallback_file = Path("fallback_queries.json")
    if fallback_file.exists():
        await fallback_cache.seed_from_file(fallback_file)

    # 5. Generative synthesizers & chitchat
    groq = GroqSynthesizer(
        api_key=config.groq_api_key.get_secret_value() if config.groq_api_key else None,
        model=config.groq_model,
    )
    gemini = GeminiFallback(
        api_key=config.gemini_api_key.get_secret_value() if config.gemini_api_key else None,
        model=config.gemini_model,
    )
    chitchat = ChitchatHandler()

    # 6. 3-Stage Hybrid Retrieval Pipeline
    retrieval_pipeline = RetrievalPipeline(
        store=store,
        embedding_service=embedding_service,
        router_guardrail=guardrail,
    )

    # 7. LangGraph RAG Agentic Orchestrator
    orchestrator = RAGOrchestrator(
        router=router,
        semantic_cache=semantic_cache,
        fallback_cache=fallback_cache,
        retrieval_pipeline=retrieval_pipeline,
        groq=groq,
        gemini=gemini,
        chitchat=chitchat,
        embedding_service=embedding_service,
    )

    # Store state on application for DI resolution
    app.state.config = config
    app.state.store = store
    app.state.embedding_service = embedding_service
    app.state.extractor = extractor
    app.state.semantic_cache = semantic_cache
    app.state.fallback_cache = fallback_cache
    app.state.orchestrator = orchestrator

    try:
        yield
    finally:
        # Graceful shutdown of connections and worker pool
        await fallback_cache.close()
        await semantic_cache.close()
        await router.close()
        await guardrail.close()
        store.close()
        shutdown_pool()


# ── Dependency Providers ──────────────────────────────────────────────

def provide_store(state: ImmutableState) -> LatticeStore:
    return state.store


def provide_embedding_service(state: ImmutableState) -> EmbeddingService:
    return state.embedding_service


def provide_extractor(state: ImmutableState) -> EntityExtractor:
    return state.extractor


def provide_semantic_cache(state: ImmutableState) -> SemanticCache:
    return state.semantic_cache


def provide_fallback_cache(state: ImmutableState) -> FallbackCache:
    return state.fallback_cache


def provide_orchestrator(state: ImmutableState) -> RAGOrchestrator:
    return state.orchestrator


def provide_config(state: ImmutableState) -> LatticeAppConfig:
    return state.config


# ── RFC 9457 Secret-Sanitizing Error Handler ──────────────────────────

def secret_sanitizing_exception_handler(request: Request, exc: Exception) -> Response:
    """Strip all auth credentials, tokens, and sensitive headers from error details."""
    sanitized_detail = sanitize_error_detail(str(exc))
    return Response(
        content={
            "type": "https://api.lattice-rag.io/errors/internal",
            "title": "Internal Server Error",
            "status": 500,
            "detail": sanitized_detail,
        },
        status_code=500,
        media_type="application/problem+json",
    )


class ApplicationCore(InitPluginProtocol):
    """Litestar application factory plugin using ApplicationCore pattern."""

    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        """Register controllers, configure CORS, OpenAPI, DI, and error handling."""
        # Register controllers
        app_config.route_handlers.extend([
            QueryController,
            IngestController,
            CacheController,
            HealthController,
        ])

        # Register DI providers
        app_config.dependencies.update({
            "store": Provide(provide_store, sync_to_thread=False),
            "embedding_service": Provide(provide_embedding_service, sync_to_thread=False),
            "extractor": Provide(provide_extractor, sync_to_thread=False),
            "semantic_cache": Provide(provide_semantic_cache, sync_to_thread=False),
            "fallback_cache": Provide(provide_fallback_cache, sync_to_thread=False),
            "orchestrator": Provide(provide_orchestrator, sync_to_thread=False),
            "config": Provide(provide_config, sync_to_thread=False),
        })

        # Register RFC 9457 error handler
        app_config.exception_handlers[Exception] = secret_sanitizing_exception_handler

        # Configure CORS
        app_config.cors_config = CORSConfig(allow_origins=["*"])

        # Configure OpenAPI with Scalar render plugin
        app_config.openapi_config = OpenAPIConfig(
            title="lattice-rag API",
            version="0.4.0",
            render_plugins=[ScalarRenderPlugin()],
        )

        # Configure logging
        app_config.logging_config = LoggingConfig(
            log_exceptions="always",
        )

        # Configure lifespan
        app_config.lifespan.append(lifespan)

        return app_config


def create_app() -> Litestar:
    """Create and configure the Litestar application."""
    return Litestar(
        plugins=[
            ApplicationCore(),
            ProblemDetailsPlugin(ProblemDetailsConfig(enable_for_all_http_exceptions=True)),
        ]
    )
