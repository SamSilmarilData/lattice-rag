"""Unit tests for Litestar API controllers using AsyncTestClient."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from litestar.testing import AsyncTestClient

from lattice_rag.api.controllers import (
    CacheController,
    DocumentController,
    HealthController,
    IngestController,
    QueryController,
)
from lattice_rag.api.guards import ADMIN_KEY_HEADER
from lattice_rag.config import AppConfig
from lattice_rag.ingestion.ingester import DocumentIngester
from lattice_rag.orchestration.state import PipelineState
from lattice_rag.security import SecretStr
from lattice_rag.storage.db import (
    EmbeddingDimensionError,
    IngestStats,
    UnresolvedChunkReferenceError,
)


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
        filtered_chunks=[
            {"node_id": 1, "text": "LatticeDB text", "score": 0.88, "doc_id": "doc1", "position": 0}
        ],
        graph_context={"nodes": [{"node_id": 1, "name": "LatticeDB"}], "edges": []},
        timings=[{"stage": "retrieval", "duration_ms": 20.0}],
    )
    mock_orchestrator.run = AsyncMock(return_value=mock_state)
    mock_orchestrator.synthesizer.circuit_status = "closed"

    async def mock_stream_gen(q):
        yield {"event": "stage", "data": {"stage": "triage", "status": "started"}}
        yield {"event": "token", "data": {"delta": "LatticeDB"}}
        yield {
            "event": "done",
            "data": {
                "answer": "LatticeDB",
                "route": "hybrid",
                "cached": False,
                "latency_ms": 10.0,
                "sources": [
                    {
                        "node_id": 7,
                        "text": "t",
                        "doc_id": "doc1",
                        "position": 3,
                        "rerank_score": 0.5,
                    }
                ],
                "graph_path": {"nodes": [{"id": 1, "name": "LatticeDB"}], "edges": []},
                "timings": [{"stage": "retrieval", "duration_ms": 2.0}],
                "degraded": False,
                "low_grounding": False,
            },
        }

    mock_orchestrator.stream_query = mock_stream_gen

    # Mock Store
    mock_store = MagicMock()
    mock_store.db.is_open = True
    mock_store.has_document.return_value = False
    mock_store.ingest_document_bundle.return_value = IngestStats(
        chunk_count=1, entity_count=0, relation_count=0, chunk_ids=[101]
    )

    # Mock EmbeddingService
    mock_embed_svc = MagicMock()
    mock_embed_svc.embed_texts.return_value = [np.zeros(384, dtype=np.float32)]

    # Mock Extractor
    mock_extractor = MagicMock()
    mock_extractor.extract_entities_batch.return_value = [[]]
    mock_extractor.extract_triples.return_value = []

    # Mock Caches
    mock_sem_cache = MagicMock()
    mock_sem_cache.stats = (5, 2)
    mock_sem_cache.__len__.return_value = 1
    mock_sem_cache.clear = MagicMock()

    mock_fb_cache = MagicMock()
    mock_fb_cache.redis = MagicMock()
    mock_fb_cache.cached_query_count = AsyncMock(return_value=50)

    # Mock Config
    mock_config = MagicMock()
    mock_config.typesafe_api_key.get_secret_value.return_value = "typesafe_key"
    mock_config.groq_api_key.get_secret_value.return_value = "groq_key"
    mock_config.gemini_api_key.get_secret_value.return_value = "gemini_key"
    mock_config.using_mock_keys = False

    app = Litestar(
        route_handlers=[
            QueryController,
            IngestController,
            DocumentController,
            CacheController,
            HealthController,
        ],
        dependencies={
            "orchestrator": Provide(lambda: mock_orchestrator, sync_to_thread=False),
            "store": Provide(lambda: mock_store, sync_to_thread=False),
            "ingester": Provide(
                lambda: DocumentIngester(
                    store=mock_store,
                    embedding_service=mock_embed_svc,
                    extractor=mock_extractor,
                    cache=mock_sem_cache,
                ),
                sync_to_thread=False,
            ),
            "semantic_cache": Provide(lambda: mock_sem_cache, sync_to_thread=False),
            "fallback_cache": Provide(lambda: mock_fb_cache, sync_to_thread=False),
            "config": Provide(lambda: mock_config, sync_to_thread=False),
        },
        state=State(
            {
                "config": AppConfig(
                    typesafe_api_key=SecretStr("ts"),
                    groq_api_key=SecretStr(""),
                    gemini_api_key=SecretStr(""),
                    admin_api_key=SecretStr("admin-secret"),
                )
            }
        ),
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
        assert data["unverified"] is False


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

        done_line = text.split("event: done", 1)[1].split("data: ", 1)[1].splitlines()[0]
        done = json.loads(done_line)
        assert done["latencyMs"] == 10.0
        assert done["lowGrounding"] is False
        assert done["sources"][0] == {
            "chunkId": 7,
            "text": "t",
            "score": 0.5,
            "documentId": "doc1",
            "position": 3,
        }
        assert done["graphPath"]["nodes"][0]["nodeId"] == 1
        assert done["timings"][0] == {"stage": "retrieval", "durationMs": 2.0}
        assert "latency_ms" not in done_line


@pytest.mark.asyncio
async def test_api_query_reports_unverified_answer(mock_controllers_app):
    app, mocks = mock_controllers_app
    mocks["orchestrator"].run = AsyncMock(
        return_value=PipelineState(
            query="q", route="hybrid", answer="Unchecked answer.", unverified=True
        )
    )
    async with AsyncTestClient(app=app) as client:
        resp = await client.post("/api/v1/query", json={"query": "What is LatticeDB?"})
        assert resp.json()["unverified"] is True


@pytest.mark.asyncio
async def test_api_query_stream_done_reports_unverified_answer(mock_controllers_app):
    app, mocks = mock_controllers_app

    async def unverified_stream(q):
        yield {
            "event": "done",
            "data": {
                "answer": "Unchecked answer.",
                "route": "hybrid",
                "cached": False,
                "latency_ms": 1.0,
                "unverified": True,
            },
        }

    mocks["orchestrator"].stream_query = unverified_stream
    async with AsyncTestClient(app=app) as client:
        resp = await client.post("/api/v1/query/stream", json={"query": "What is LatticeDB?"})
        done_line = resp.text.split("event: done", 1)[1].split("data: ", 1)[1].splitlines()[0]
        assert json.loads(done_line)["unverified"] is True


@pytest.mark.asyncio
async def test_api_query_stream_reports_failures_in_band(mock_controllers_app):
    """A mid-stream exception becomes an SSE error event with no internal details."""
    app, mocks = mock_controllers_app

    async def failing_stream(q):
        yield {"event": "stage", "data": {"stage": "triage", "status": "started"}}
        raise RuntimeError("/srv/secret/path exploded")

    mocks["orchestrator"].stream_query = failing_stream
    async with AsyncTestClient(app=app) as client:
        resp = await client.post("/api/v1/query/stream", json={"query": "q"})
        assert "event: error" in resp.text
        assert "Reference:" in resp.text
        assert "/srv/secret" not in resp.text


@pytest.mark.asyncio
async def test_api_ingest_endpoint(mock_controllers_app):
    app, mocks = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        payload = {
            "documentId": "doc_arch",
            "title": "Architecture Overview",
            "text": "LatticeDB operates as an in-process property-graph database. It supports HNSW vectors.",
        }
        resp = await client.post(
            "/api/v1/ingest", json=payload, headers={ADMIN_KEY_HEADER: "admin-secret"}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["documentId"] == "doc_arch"
        assert data["chunkCount"] >= 1
        mocks["semantic_cache"].clear.assert_called_once()


@pytest.mark.asyncio
async def test_api_cache_stats_endpoint(mock_controllers_app):
    app, mocks = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        resp = await client.get("/api/v1/cache/stats", headers={ADMIN_KEY_HEADER: "admin-secret"})
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


@pytest.mark.asyncio
async def test_api_ingest_requires_admin_key(mock_controllers_app):
    app, mocks = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        payload = {"documentId": "doc_x", "title": "T", "text": "Some text."}
        resp = await client.post("/api/v1/ingest", json=payload)
        assert resp.status_code == 401
        mocks["store"].ingest_document_bundle.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"documentId": "../etc/passwd", "title": "T", "text": "x"},
        {"documentId": "doc", "title": "T", "text": ""},
        {"documentId": "doc", "title": "T", "text": "x" * 200_001},
    ],
)
async def test_api_ingest_rejects_invalid_payloads(mock_controllers_app, payload):
    app, _ = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        resp = await client.post(
            "/api/v1/ingest", json=payload, headers={ADMIN_KEY_HEADER: "admin-secret"}
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "x" * 2_001])
async def test_api_query_rejects_empty_or_oversized_query(mock_controllers_app, query):
    app, mocks = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        resp = await client.post("/api/v1/query", json={"query": query})
        assert resp.status_code == 400
        mocks["orchestrator"].run.assert_not_called()


@pytest.mark.asyncio
async def test_api_ingest_duplicate_document_returns_409(mock_controllers_app):
    app, mocks = mock_controllers_app
    mocks["store"].has_document.return_value = True
    async with AsyncTestClient(app=app) as client:
        payload = {"documentId": "doc_arch", "title": "T", "text": "Some text."}
        resp = await client.post(
            "/api/v1/ingest", json=payload, headers={ADMIN_KEY_HEADER: "admin-secret"}
        )
        assert resp.status_code == 409
        mocks["store"].ingest_document_bundle.assert_not_called()


@pytest.mark.asyncio
async def test_api_ingest_with_replace_replaces_an_existing_document(mock_controllers_app):
    app, mocks = mock_controllers_app
    mocks["store"].has_document.return_value = True
    async with AsyncTestClient(app=app) as client:
        payload = {"documentId": "doc_arch", "title": "T", "text": "New text.", "replace": True}
        resp = await client.post(
            "/api/v1/ingest", json=payload, headers={ADMIN_KEY_HEADER: "admin-secret"}
        )
        assert resp.status_code == 201
        assert mocks["store"].ingest_document_bundle.call_args.kwargs["replace"] is True
        mocks["semantic_cache"].clear.assert_called_once()


@pytest.mark.asyncio
async def test_api_ingest_response_states_runtime_documents_are_ephemeral(mock_controllers_app):
    app, _ = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        payload = {"documentId": "doc_eph", "title": "T", "text": "Some text."}
        resp = await client.post(
            "/api/v1/ingest", json=payload, headers={ADMIN_KEY_HEADER: "admin-secret"}
        )
        assert resp.status_code == 201
        assert "ephemeral" in resp.json()["persistence"].lower()


@pytest.mark.asyncio
async def test_api_delete_document_clears_tier1(mock_controllers_app):
    app, mocks = mock_controllers_app
    mocks["store"].delete_document.return_value = True
    async with AsyncTestClient(app=app) as client:
        resp = await client.delete(
            "/api/v1/documents/doc_arch", headers={ADMIN_KEY_HEADER: "admin-secret"}
        )
        assert resp.status_code == 204
        mocks["store"].delete_document.assert_called_once_with("doc_arch")
        mocks["semantic_cache"].clear.assert_called_once()


@pytest.mark.asyncio
async def test_api_delete_unknown_document_returns_404(mock_controllers_app):
    app, mocks = mock_controllers_app
    mocks["store"].delete_document.return_value = False
    async with AsyncTestClient(app=app) as client:
        resp = await client.delete(
            "/api/v1/documents/missing", headers={ADMIN_KEY_HEADER: "admin-secret"}
        )
        assert resp.status_code == 404
        mocks["semantic_cache"].clear.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {ADMIN_KEY_HEADER: "wrong"}])
async def test_api_delete_document_requires_admin_key(mock_controllers_app, headers):
    app, mocks = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        resp = await client.delete("/api/v1/documents/doc_arch", headers=headers)
        assert resp.status_code == 401
        mocks["store"].delete_document.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [EmbeddingDimensionError("chunk 0", 192, 384), UnresolvedChunkReferenceError(5, 1)],
)
async def test_api_ingest_validation_error_returns_422(mock_controllers_app, error):
    app, mocks = mock_controllers_app
    mocks["store"].ingest_document_bundle.side_effect = error
    async with AsyncTestClient(app=app) as client:
        payload = {"documentId": "doc_bad", "title": "T", "text": "Some text."}
        resp = await client.post(
            "/api/v1/ingest", json=payload, headers={ADMIN_KEY_HEADER: "admin-secret"}
        )
        assert resp.status_code == 422
        assert resp.json()["detail"] == str(error)


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {ADMIN_KEY_HEADER: "wrong-key"}])
async def test_api_cache_stats_requires_admin_key(mock_controllers_app, headers):
    app, _ = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        resp = await client.get("/api/v1/cache/stats", headers=headers)
        assert resp.status_code == 401
        assert "tier1Hits" not in resp.text


@pytest.mark.asyncio
async def test_public_endpoints_need_no_admin_key(mock_controllers_app):
    """Health, query and streaming query stay reachable without X-Admin-Key."""
    app, _ = mock_controllers_app
    async with AsyncTestClient(app=app) as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.post("/api/v1/query", json={"query": "q"})).status_code == 201
        stream = await client.post("/api/v1/query/stream", json={"query": "q"})
        assert stream.status_code == 201
        assert "event: done" in stream.text
