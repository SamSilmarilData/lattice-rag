from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import uuid
from collections.abc import AsyncGenerator
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

from litestar import Litestar, MediaType, Request, Response, Router, get
from litestar.config.app import AppConfig
from litestar.config.cors import CORSConfig
from litestar.datastructures import State
from litestar.di import Provide
from litestar.exceptions import HTTPException
from litestar.logging import LoggingConfig
from litestar.middleware.rate_limit import RateLimitConfig
from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import ScalarRenderPlugin
from litestar.plugins import InitPluginProtocol
from litestar.plugins.problem_details import ProblemDetailsConfig, ProblemDetailsPlugin
from typesafe_sdk import AsyncTypeSafeClient

from lattice_rag import __version__
from lattice_rag.api.client_identity import client_address
from lattice_rag.api.controllers import (
    BenchmarkController,
    CacheController,
    DocumentController,
    EvalController,
    GraphController,
    HealthController,
    IngestController,
    QueryController,
)
from lattice_rag.caching.fallback_cache import FallbackCache
from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.config import AppConfig as LatticeAppConfig
from lattice_rag.config import get_config, load_config
from lattice_rag.eval import EvalRunner, JevEvaluator
from lattice_rag.eval.jobs import EvalJobManager
from lattice_rag.generation.chitchat import ChitchatHandler
from lattice_rag.generation.gemini_fallback import GeminiFallback
from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
from lattice_rag.ingestion import DocumentIngester
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.pipeline import RetrievalPipeline
from lattice_rag.routing.guardrail import ContextGuardrail
from lattice_rag.routing.router import QueryRouter
from lattice_rag.security import SecretStr, configure_structlog, sanitize_error_detail
from lattice_rag.storage.db import LatticeStore
from lattice_rag.storage.extract import EntityExtractor

logger = logging.getLogger(__name__)

MAX_REQUEST_BODY_BYTES = 1_000_000


def _resolve_config(config: LatticeAppConfig | None) -> LatticeAppConfig:
    """Use the given config, else the env config, else hermetic mock keys (System 1 fallback)."""
    if config is not None:
        return config
    try:
        return get_config()
    except SystemExit:
        # Keep every other env setting (DB path, admin key, CORS); only the key is mocked
        logger.warning(
            "TYPESAFE_API_KEY is not set: starting in degraded mode. Jev is not called: "
            "routing falls back to hybrid (pure small talk still goes to chitchat), the "
            "context guardrail and semantic-cache verification are off, answers are flagged "
            "unverified and are not cached."
        )
        return dataclasses.replace(
            load_config(require_typesafe=False),
            typesafe_api_key=SecretStr("mock-typesafe-key-hermetic"),
            using_mock_keys=True,
        )


@contextlib.asynccontextmanager
async def lifespan(app: Litestar) -> AsyncGenerator[None, None]:
    """Lifespan context manager for initializing and closing dependencies."""
    config = _resolve_config(getattr(app.state, "config", None))

    def typesafe_client() -> AsyncTypeSafeClient:
        # Built from config rather than os.environ, so the mock key never leaks into the env
        return AsyncTypeSafeClient(api_key=config.typesafe_api_key.get_secret_value())

    # 1. Embedded storage engine
    store = LatticeStore(config.latticedb_path)

    # 2. Local embeddings and entity extraction
    embedding_service = EmbeddingService(
        embed_model=config.embed_model,
        reranker_model=config.reranker_model,
        onnx_reranker_repo=config.reranker_onnx_repo,
        onnx_reranker_revision=config.reranker_onnx_revision,
    )
    extractor = EntityExtractor()

    # 3. Decision routing & context guardrail. In mock-key mode neither is called, so no
    #    query pays for failing Jev calls; the resulting answers are flagged unverified.
    router = QueryRouter.offline() if config.using_mock_keys else QueryRouter(typesafe_client())
    guardrail = None if config.using_mock_keys else ContextGuardrail(ts_client=typesafe_client())

    # 4. Multi-tier caching (fuzzy hits need Noul verification, so none in mock mode)
    semantic_cache = SemanticCache(ts_client=None if config.using_mock_keys else typesafe_client())
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
        max_tokens=config.llm_max_tokens,
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

    # 8. Document Ingester (deep module)
    ingester = DocumentIngester(
        store=store,
        embedding_service=embedding_service,
        extractor=extractor,
        cache=semantic_cache,
    )

    # 9. Jev Evaluator & CI/CD Eval Runner
    evaluator = JevEvaluator(api_key=config.typesafe_api_key)
    eval_runner = EvalRunner(
        orchestrator=orchestrator,
        evaluator=evaluator,
        embedding_service=embedding_service,
        extractor=extractor,
    )
    # HTTP eval runs go through the job manager: one at a time, polled by handle
    eval_jobs = EvalJobManager(eval_runner)

    # Store state on application for DI resolution
    app.state.config = config
    app.state.store = store
    app.state.embedding_service = embedding_service
    app.state.extractor = extractor
    app.state.ingester = ingester
    app.state.semantic_cache = semantic_cache
    app.state.fallback_cache = fallback_cache
    app.state.orchestrator = orchestrator
    app.state.evaluator = evaluator
    app.state.eval_runner = eval_runner
    app.state.eval_jobs = eval_jobs

    try:
        yield
    finally:
        # Graceful shutdown of connections
        await eval_jobs.close()
        await evaluator.close()
        await fallback_cache.close()
        await semantic_cache.close()
        await router.close()
        if guardrail is not None:
            await guardrail.close()
        store.close()


# ── Dependency Providers ──────────────────────────────────────────────


def provide_store(state: State) -> LatticeStore:
    return cast(LatticeStore, state.store)


def provide_embedding_service(state: State) -> EmbeddingService:
    return cast(EmbeddingService, state.embedding_service)


def provide_extractor(state: State) -> EntityExtractor:
    return cast(EntityExtractor, state.extractor)


def provide_semantic_cache(state: State) -> SemanticCache:
    return cast(SemanticCache, state.semantic_cache)


def provide_fallback_cache(state: State) -> FallbackCache:
    return cast(FallbackCache, state.fallback_cache)


def provide_orchestrator(state: State) -> RAGOrchestrator:
    return cast(RAGOrchestrator, state.orchestrator)


def provide_config(state: State) -> LatticeAppConfig:
    return cast(LatticeAppConfig, state.config)


def provide_ingester(state: State) -> DocumentIngester:
    return cast(DocumentIngester, state.ingester)


def provide_eval_runner(state: State) -> EvalRunner:
    return cast(EvalRunner, state.eval_runner)


def provide_eval_jobs(state: State) -> EvalJobManager:
    return cast(EvalJobManager, state.eval_jobs)


# ── RFC 9457 Secret-Sanitizing Error Handler ──────────────────────────


def secret_sanitizing_exception_handler(
    request: Request[Any, Any, Any], exc: Exception
) -> Response[Any]:
    """RFC 9457 handler for unexpected errors.

    HTTP exceptions keep their status and (sanitized) detail. Anything else is logged
    server-side with a request id, and the client only receives that id, never the
    exception text, which can carry file paths, queries, or upstream payloads.
    """
    if isinstance(exc, HTTPException):
        return Response(
            content={
                "type": "about:blank",
                "title": HTTPStatus(exc.status_code).phrase,
                "status": exc.status_code,
                "detail": sanitize_error_detail(str(exc.detail)),
            },
            status_code=exc.status_code,
            media_type="application/problem+json",
        )

    request_id = uuid.uuid4().hex
    logger.error(
        "unhandled_exception request_id=%s path=%s error=%s",
        request_id,
        request.url.path,
        sanitize_error_detail(f"{type(exc).__name__}: {exc}"),
    )
    return Response(
        content={
            "type": "https://api.lattice-rag.io/errors/internal",
            "title": "Internal Server Error",
            "status": 500,
            "detail": f"An internal error occurred. Reference: {request_id}",
            "requestId": request_id,
        },
        status_code=500,
        media_type="application/problem+json",
    )


@get("/", media_type=MediaType.HTML, include_in_schema=False)
async def fallback_landing() -> Response[Any]:
    """Graceful fallback landing page when frontend/dist is not yet built."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>lattice-rag Engine</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0B0F19; color: #F3F4F6; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
    .card { background: #111827; border: 1px solid #1F2937; padding: 2.5rem; border-radius: 0.75rem; max-width: 520px; text-align: center; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); }
    h1 { margin-top: 0; color: #06B6D4; font-size: 1.6rem; letter-spacing: -0.025em; }
    p { color: #9CA3AF; line-height: 1.6; font-size: 0.95rem; }
    a { display: inline-block; margin-top: 1.25rem; padding: 0.65rem 1.4rem; background: #06B6D4; color: #0B0F19; text-decoration: none; border-radius: 0.375rem; font-weight: 600; font-size: 0.9rem; transition: background 0.15s ease; }
    a:hover { background: #22D3EE; }
    code { background: #1F2937; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-family: monospace; font-size: 0.85rem; color: #E5E7EB; }
  </style>
</head>
<body>
  <div class="card">
    <h1>lattice-rag Engine</h1>
    <p>Zero-cloud-cost Hybrid GraphRAG ASGI server is operational.</p>
    <p>Frontend UI is not yet built. Run <code>npm run build</code> in <code>frontend/</code> to activate the Visual Playground.</p>
    <a href="/schema/scalar">Open API Scalar Docs</a>
  </div>
</body>
</html>"""
    return Response(content=html, media_type=MediaType.HTML)


class ApplicationCore(InitPluginProtocol):
    """Litestar application factory plugin using ApplicationCore pattern."""

    def __init__(self, config: LatticeAppConfig) -> None:
        self.config = config

    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        """Register controllers, configure CORS, OpenAPI, DI, and error handling."""
        # Register controllers
        app_config.route_handlers.extend(
            [
                BenchmarkController,
                # Public query endpoints carry a per-client rate limit
                Router(
                    path="/",
                    route_handlers=[QueryController],
                    middleware=[
                        RateLimitConfig(
                            rate_limit=("minute", self.config.query_rate_limit_per_minute),
                            identifier_for_request=functools.partial(
                                client_address,
                                trusted_proxy_hops=self.config.trusted_proxy_hops,
                            ),
                        ).middleware
                    ],
                ),
                IngestController,
                DocumentController,
                CacheController,
                HealthController,
                EvalController,
                GraphController,
            ]
        )

        # Mount frontend static files or fallback landing route
        dist_path = Path("frontend/dist")
        if dist_path.is_dir() and (dist_path / "index.html").exists():
            from litestar.static_files import create_static_files_router

            app_config.route_handlers.append(
                create_static_files_router(
                    path="/",
                    directories=[dist_path],
                    html_mode=True,
                    name="frontend",
                )
            )
        else:
            app_config.route_handlers.append(fallback_landing)

        # Register DI providers
        app_config.dependencies.update(
            {
                "store": Provide(provide_store, sync_to_thread=False),
                "embedding_service": Provide(provide_embedding_service, sync_to_thread=False),
                "extractor": Provide(provide_extractor, sync_to_thread=False),
                "semantic_cache": Provide(provide_semantic_cache, sync_to_thread=False),
                "fallback_cache": Provide(provide_fallback_cache, sync_to_thread=False),
                "ingester": Provide(provide_ingester, sync_to_thread=False),
                "orchestrator": Provide(provide_orchestrator, sync_to_thread=False),
                "config": Provide(provide_config, sync_to_thread=False),
                "eval_runner": Provide(provide_eval_runner, sync_to_thread=False),
                "eval_jobs": Provide(provide_eval_jobs, sync_to_thread=False),
            }
        )

        # Register RFC 9457 error handler
        app_config.exception_handlers[Exception] = secret_sanitizing_exception_handler

        # CORS: same-origin only unless CORS_ALLOW_ORIGINS lists trusted origins
        if self.config.cors_allow_origins:
            app_config.cors_config = CORSConfig(
                allow_origins=list(self.config.cors_allow_origins),
                allow_headers=["Content-Type", "X-Admin-Key"],
            )

        app_config.request_max_body_size = MAX_REQUEST_BODY_BYTES

        # Configure OpenAPI with Scalar render plugin
        app_config.openapi_config = OpenAPIConfig(
            title="lattice-rag API",
            version=__version__,
            render_plugins=[ScalarRenderPlugin()],
        )

        # Configure logging: every stdlib record (and structlog, routed through stdlib)
        # is rendered by the redacting formatter, tracebacks included.
        configure_structlog()
        app_config.logging_config = LoggingConfig(
            log_exceptions="always",
            formatters={"standard": {"()": "lattice_rag.security.RedactingFormatter"}},
        )

        # Configure lifespan
        app_config.lifespan.append(lifespan)

        return app_config


def create_app(config: LatticeAppConfig | None = None) -> Litestar:
    """Create and configure the Litestar application."""
    resolved = _resolve_config(config)
    return Litestar(
        plugins=[
            ApplicationCore(resolved),
            ProblemDetailsPlugin(ProblemDetailsConfig(enable_for_all_http_exceptions=True)),
        ],
        state=State({"config": resolved}),
    )
