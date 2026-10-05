"""Live End-to-End Integration Tests for Phase 4 (Litestar + LangGraph + Groq + LatticeDB).

Runs full ingestion and query flow through Litestar AsyncTestClient using real in-process
LatticeDB, real FastEmbed embeddings, and live GroqCloud qwen/qwen3.8-27b generation.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.params import SkipValidation
from litestar.testing import AsyncTestClient

from lattice_rag.api.controllers import (
    CacheController,
    HealthController,
    IngestController,
    QueryController,
)
from lattice_rag.caching.fallback_cache import FallbackCache
from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.config import AppConfig, get_config
from lattice_rag.generation.chitchat import ChitchatHandler
from lattice_rag.generation.gemini_fallback import GeminiFallback
from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.pipeline import RetrievalPipeline
from lattice_rag.routing.guardrail import ContextGuardrail
from lattice_rag.routing.router import QueryRouter
from lattice_rag.storage.db import LatticeStore
from lattice_rag.storage.extract import EntityExtractor


@pytest.fixture
async def live_app(tmp_path: Path):
    config = get_config()
    db_path = tmp_path / "live_e2e.db"
    store = LatticeStore(db_path)

    embed_svc = EmbeddingService(embed_model="BAAI/bge-small-en-v1.5", reranker_model="BAAI/bge-reranker-v2-m3")
    extractor = EntityExtractor()
    router = QueryRouter()
    guardrail = ContextGuardrail(threshold=0.50)
    sem_cache = SemanticCache()
    fb_cache = FallbackCache(redis_url="embedded")
    await fb_cache.connect()

    groq = GroqSynthesizer(
        api_key=config.groq_api_key.get_secret_value(),
        model=config.groq_model,
    )
    gemini = GeminiFallback(
        api_key=config.gemini_api_key.get_secret_value(),
        model=config.gemini_model,
    )
    chitchat = ChitchatHandler()

    ret_pipeline = RetrievalPipeline(
        store=store,
        embedding_service=embed_svc,
        router_guardrail=guardrail,
    )

    orchestrator = RAGOrchestrator(
        router=router,
        semantic_cache=sem_cache,
        fallback_cache=fb_cache,
        retrieval_pipeline=ret_pipeline,
        groq=groq,
        gemini=gemini,
        chitchat=chitchat,
        embedding_service=embed_svc,
    )

    app = Litestar(
        route_handlers=[
            QueryController,
            IngestController,
            CacheController,
            HealthController,
        ],
        debug=True,
        dependencies={
            "store": Provide(lambda: store, sync_to_thread=False),
            "embedding_service": Provide(lambda: embed_svc, sync_to_thread=False),
            "extractor": Provide(lambda: extractor, sync_to_thread=False),
            "semantic_cache": Provide(lambda: sem_cache, sync_to_thread=False),
            "fallback_cache": Provide(lambda: fb_cache, sync_to_thread=False),
            "orchestrator": Provide(lambda: orchestrator, sync_to_thread=False),
            "config": Provide(lambda: config, sync_to_thread=False),
        },
    )

    yield app

    try:
        await fb_cache.close()
    except Exception:
        pass
    try:
        await sem_cache.close()
    except Exception:
        pass
    try:
        await router.close()
    except Exception:
        pass
    try:
        await guardrail.close()
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_live_phase4_end_to_end_flow(live_app):
    """Verify live E2E: Ingest -> Health -> Query with Groq -> Cache Hit -> Chitchat."""
    async with AsyncTestClient(app=live_app) as client:
        # 1. Health Check
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["latticedbConnected"] is True
        assert health_data["groqConfigured"] is True

        # 2. Ingest Technical Knowledge
        ingest_payload = {
            "documentId": "doc_e2e_whitepaper",
            "title": "LatticeDB Technical Specifications",
            "text": (
                "LatticeDB is an embedded, in-process property-graph database. "
                "It uniquely unites Cypher multi-hop graph queries with native HNSW vector similarity search "
                "and BM25 inverted index text searches in a single single-binary local storage engine."
            ),
        }
        ingest_resp = await client.post("/api/v1/ingest", json=ingest_payload)
        assert ingest_resp.status_code == 201
        ingest_data = ingest_resp.json()
        assert ingest_data["documentId"] == "doc_e2e_whitepaper"
        assert ingest_data["chunkCount"] >= 1

        # 3. Live Query with Groq Synthesis
        query_payload = {"query": "What search capabilities does LatticeDB unite in a single binary?"}
        query_resp = await client.post("/api/v1/query", json=query_payload)
        assert query_resp.status_code == 201, f"Query failed with {query_resp.status_code}: {query_resp.text}"
        query_data = query_resp.json()

        print(f"\n[Live Groq Generation Result]:\nRoute: {query_data['route']}\nAnswer: {query_data['answer']}")
        assert len(query_data["answer"]) > 10
        assert query_data["cached"] is False
        assert query_data["degraded"] is False
        assert len(query_data["sources"]) >= 1

        # 4. Instant Tier 1 Semantic Cache Hit
        cached_query_resp = await client.post("/api/v1/query", json=query_payload)
        assert cached_query_resp.status_code == 201
        cached_data = cached_query_resp.json()
        assert cached_data["cached"] is True
        assert cached_data["answer"] == query_data["answer"]

        # 5. Zero-Cost Chitchat
        chitchat_resp = await client.post("/api/v1/query", json={"query": "Hello engine!"})
        assert chitchat_resp.status_code == 201
        chitchat_data = chitchat_resp.json()
        assert chitchat_data["route"] == "chitchat"
        assert "Hello!" in chitchat_data["answer"]

        # 6. Verify Cache Stats
        stats_resp = await client.get("/api/v1/cache/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()
        assert stats_data["tier1Hits"] >= 1

        # 7. Live SSE Streaming Query
        stream_payload = {"query": "Tell me about LatticeDB graph and vector capabilities."}
        stream_resp = await client.post("/api/v1/query/stream", json=stream_payload)
        assert stream_resp.status_code == 201
        assert "text/event-stream" in stream_resp.headers.get("content-type", "")
        stream_text = stream_resp.text
        assert "event: stage" in stream_text
        assert "event: done" in stream_text
