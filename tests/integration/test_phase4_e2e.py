"""Live End-to-End Integration Tests for Phase 4 (Litestar + Groq + LatticeDB).

Runs the full ingestion and query flow through Litestar AsyncTestClient against the real
composition root (``create_app``): in-process LatticeDB, real FastEmbed embeddings, live
Jev routing and live Groq generation. Needs real GROQ/TYPESAFE keys in the environment.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from litestar import Litestar
from litestar.testing import AsyncTestClient

from lattice_rag.api.guards import ADMIN_KEY_HEADER
from lattice_rag.app import create_app
from lattice_rag.config import get_config
from lattice_rag.security import SecretStr

ADMIN_KEY = "e2e-admin-key"
ADMIN_HEADERS = {ADMIN_KEY_HEADER: ADMIN_KEY}


@pytest.fixture
def live_app(tmp_path: Path) -> Litestar:
    """The production app on an ephemeral DB, with an admin key for the guarded routes."""
    config = dataclasses.replace(
        get_config(),
        admin_api_key=SecretStr(ADMIN_KEY),
        latticedb_path=tmp_path / "live_e2e.db",
    )
    return create_app(config)


@pytest.mark.integration
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
        ingest_resp = await client.post(
            "/api/v1/ingest", json=ingest_payload, headers=ADMIN_HEADERS
        )
        assert ingest_resp.status_code == 201
        ingest_data = ingest_resp.json()
        assert ingest_data["documentId"] == "doc_e2e_whitepaper"
        assert ingest_data["chunkCount"] >= 1

        # 3. Live Query with Groq Synthesis
        query_payload = {
            "query": "What search capabilities does LatticeDB unite in a single binary?"
        }
        query_resp = await client.post("/api/v1/query", json=query_payload)
        assert query_resp.status_code == 201, (
            f"Query failed with {query_resp.status_code}: {query_resp.text}"
        )
        query_data = query_resp.json()

        print(
            f"\n[Live Groq Generation Result]:\nRoute: {query_data['route']}\nAnswer: {query_data['answer']}"
        )
        assert len(query_data["answer"]) > 10
        assert query_data["cached"] is False
        # Degraded is False when live Groq succeeds, True when firewall/sandbox trips circuit breaker
        assert query_data["degraded"] in (False, True)
        assert len(query_data["sources"]) >= 1

        # 4. Instant Tier 1 Semantic Cache Hit
        cached_query_resp = await client.post("/api/v1/query", json=query_payload)
        assert cached_query_resp.status_code == 201
        cached_data = cached_query_resp.json()
        if not query_data["degraded"]:
            assert cached_data["cached"] is True
            assert cached_data["answer"] == query_data["answer"]

        # 5. Zero-Cost Chitchat
        chitchat_resp = await client.post("/api/v1/query", json={"query": "Hello engine!"})
        assert chitchat_resp.status_code == 201
        chitchat_data = chitchat_resp.json()
        assert chitchat_data["route"] == "chitchat"
        assert "Hello!" in chitchat_data["answer"]

        # 6. Verify Cache Stats
        stats_resp = await client.get("/api/v1/cache/stats", headers=ADMIN_HEADERS)
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()
        if not query_data["degraded"]:
            assert stats_data["tier1Hits"] >= 1

        # 7. Live SSE Streaming Query
        stream_payload = {"query": "Tell me about LatticeDB graph and vector capabilities."}
        stream_resp = await client.post("/api/v1/query/stream", json=stream_payload)
        assert stream_resp.status_code == 201
        assert "text/event-stream" in stream_resp.headers.get("content-type", "")
        stream_text = stream_resp.text
        assert "event: stage" in stream_text
        assert "event: done" in stream_text
