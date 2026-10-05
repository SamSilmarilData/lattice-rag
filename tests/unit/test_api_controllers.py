"""Unit tests for Litestar API controllers using AsyncTestClient."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.testing import AsyncTestClient

from lattice_rag.api.controllers import (
    CacheController,
    HealthController,
    IngestController,
    QueryController,
)
from lattice_rag.orchestration.state import PipelineState
from lattice_rag.storage.db import IngestStats


@pytest.fixture
def mock_controllers_app():
    # Mock Orchestrator
    mock_orchestrator = MagicMock()
    mock_state = PipelineState(
        query="What is LatticeDB?",
        route="hybrid",
        cache_hit=False,
        answer="LatticeDB is an embedded database.",
        total_latency_ms=45.2,
        filtered_chunks=[{"node_id": 1, "text": "LatticeDB text", "score": 0.88, "doc_id": "doc1", "position": 0}],
        graph_context={"nodes": [{"node_id": 1, "name": "LatticeDB"}], "edges": []},
        timings=[{"stage": "retrieval", "duration_ms": 20.0}],
    )
    mock_orchestrator.run = AsyncMock(return_value=mock_state)

    async def mock_stream_gen(q):
        yield {"event": "stage", "data": {"stage": "triage", "status": "started"}}
        yield {"event": "token", "data": {"delta": "LatticeDB"}}
        yield {"event": "done", "data": {"answer": "LatticeDB", "route": "hybrid", "latency_ms": 10.0}}

    mock_orchestrator.stream_query = mock_stream_gen

    # Mock Store
    mock_store = MagicMock()
    mock_store.db.is_open = True
    mock_store.ingest_document.return_value = IngestStats(chunk_count=1, entity_count=0, relation_count=0, chunk_ids=[101])
    mock_store.ingest_entities.return_value = IngestStats(chunk_count=0, entity_count=1, relation_count=0)

    # Mock EmbeddingService
    mock_embed_svc = MagicMock()
    mock_embed_svc.embed_texts.return_value = [np.zeros(384, dtype=np.float32)]

    # Mock Extractor
    mock_extractor = MagicMock()
    mock_extractor.extract_entities.return_value = []
    mock_extractor.extract_triples.return_value = []

    # Mock Caches
    mock_sem_cache = MagicMock()
    mock_sem_cache.stats = (5, 2)
    mock_sem_cache._cache = [MagicMock()]
    mock_sem_cache.clear = MagicMock()

    mock_fb_cache = MagicMock()
    mock_fb_cache.redis = MagicMock()
    mock_fb_cache.circuit_status = "closed"
    mock_fb_cache.cached_query_count = AsyncMock(return_value=50)

    # Mock Config
    mock_config = MagicMock()
    mock_config.typesafe_api_key.get_secret_value.return_value = "typesafe_key"
    mock_config.groq_api_key.get_secret_value.return_value = "groq_key"
    mock_config.gemini_api_key.get_secret_value.return_value = "gemini_key"

    app = Litestar(
        route_handlers=[
            QueryController,
            IngestController,
            CacheController,
            HealthController,
        ],
        dependencies={
            "orchestrator": Provide(lambda: mock_orchestrator, sync_to_thread=False),
            "store": Provide(lambda: mock_store, sync_to_thread=False),
            "embedding_service": Provide(lambda: mock_embed_svc, sync_to_thread=False),
            "extractor": Provide(lambda: mock_extractor, sync_to_thread=False),
            "semantic_cache": Provide(lambda: mock_sem_cache, sync_to_thread=False),
            "fallback_cache": Provide(lambda: mock_fb_cache, sync_to_thread=False),
            "config": Provide(lambda: mock_config, sync_to_thread=False),
        },
    )
    return app, {
        "orchestrator": mock_orchestrator,
        "store": mock_store,
        "semantic_cache": mock_sem_cache,
    }


@pytest.mark.asyncio
async def test_api_query_endpoint(mock_controllers_app):
    app, mocks = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        resp = await client.post("/api/v1/query", json={"query": "What is LatticeDB?"})
        assert resp.status_code == 201
        data = resp.json()

        assert data["answer"] == "LatticeDB is an embedded database."
        assert data["route"] == "hybrid"
        assert data["cached"] is False
        assert len(data["sources"]) == 1
        assert data["sources"][0]["documentId"] == "doc1"
        assert len(data["timings"]) == 1
        assert data["graphPath"]["nodes"][0]["name"] == "LatticeDB"


@pytest.mark.asyncio
async def test_api_query_stream_sse(mock_controllers_app):
    app, mocks = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        resp = await client.post("/api/v1/query/stream", json={"query": "What is LatticeDB?"})
        assert resp.status_code == 201
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text
        assert "event: stage" in text
        assert "event: token" in text
        assert "event: done" in text


@pytest.mark.asyncio
async def test_api_ingest_endpoint(mock_controllers_app):
    app, mocks = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        payload = {
            "documentId": "doc_arch",
            "title": "Architecture Overview",
            "text": "LatticeDB operates as an in-process property-graph database. It supports HNSW vectors.",
        }
        resp = await client.post("/api/v1/ingest", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["documentId"] == "doc_arch"
        assert data["chunkCount"] >= 1
        mocks["semantic_cache"].clear.assert_called_once()


@pytest.mark.asyncio
async def test_api_cache_stats_endpoint(mock_controllers_app):
    app, mocks = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        resp = await client.get("/api/v1/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier1Hits"] == 5
        assert data["tier1Misses"] == 2
        assert data["tier1CachedQueries"] == 1
        assert data["tier2CircuitStatus"] == "closed"
        assert data["tier2CachedQueries"] == 50


@pytest.mark.asyncio
async def test_api_health_endpoint(mock_controllers_app):
    app, mocks = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["latticedbConnected"] is True
        assert data["redisConnected"] is True
        assert data["typesafeConfigured"] is True
        assert data["groqConfigured"] is True
        assert data["geminiConfigured"] is True


@pytest.mark.asyncio
async def test_api_rfc9457_error_secret_sanitization(mock_controllers_app):
    app, mocks = mock_controllers_app
    mocks["orchestrator"].run.side_effect = RuntimeError(
        "Fatal API error with secret sk-1234567890abcdef and token=secret_token_123"
    )

    from lattice_rag.app import secret_sanitizing_exception_handler
    app.exception_handlers[Exception] = secret_sanitizing_exception_handler

    async with AsyncTestClient(app=app) as client:
        resp = await client.post("/api/v1/query", json={"query": "Fail query"})
        assert resp.status_code == 500
        assert "application/problem+json" in resp.headers["content-type"]
        data = resp.json()
        assert "sk-1234567890abcdef" not in data["detail"]
        assert "secret_token_123" not in data["detail"]
        assert data["status"] == 500
        assert data["title"] == "Internal Server Error"
