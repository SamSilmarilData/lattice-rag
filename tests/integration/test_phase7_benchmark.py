"""Integration tests for Phase 7 Benchmark Engine and SLA validation."""
from __future__ import annotations

from pathlib import Path
import pytest
from litestar.testing import AsyncTestClient

from lattice_rag.app import create_app
from lattice_rag.benchmark import BenchmarkEngine
from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.config import get_config
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.storage.db import LatticeStore


@pytest.mark.asyncio
async def test_benchmark_engine_component_slas():
    """Verify that all components meet their latency SLA targets on the seeded database."""
    config = get_config()
    store = LatticeStore(config.latticedb_path)
    embed_svc = EmbeddingService(config.embed_model, config.reranker_model)
    sem_cache = SemanticCache()

    engine = BenchmarkEngine(
        store=store,
        embedding_service=embed_svc,
        semantic_cache=sem_cache,
    )

    queries = [
        "How does LatticeDB combine HNSW vector similarity with BM25 text search in a single binary?",
        "Explain the multi-hop Cypher traversal process from entity anchors.",
        "What is the role of FastEmbed and what dense embedding model does it use by default?",
    ]

    try:
        report = await engine.run_full_suite(queries=queries, iterations=4, warmup=1, live_llm=False)

        # 1. Dual SLA validation
        assert report.tier1_cache_sla_met is True, f"Tier 1 cache SLA failed: {report.stages['tier1_cache'].p95_ms}ms"
        assert report.sub_second_sla_met is True, f"Sub-second SLA failed: {report.stages['e2e_pipeline'].p95_ms}ms"

        # 2. Stage-by-stage latency budget checks
        cache_stats = report.stages["tier1_cache"]
        assert cache_stats.p95_ms < 25.0
        assert cache_stats.p50_ms < 20.0

        stage1_stats = report.stages["stage1_hybrid"]
        assert stage1_stats.p95_ms < 200.0

        stage2_stats = report.stages["stage2_traversal"]
        assert stage2_stats.p95_ms < 60.0

        stage3_stats = report.stages["stage3_rerank"]
        assert stage3_stats.p95_ms < 650.0

        e2e_stats = report.stages["e2e_pipeline"]
        assert e2e_stats.p95_ms < 1000.0

        # 3. Clean string formatting check
        table_output = BenchmarkEngine.format_ascii_table(report)
        assert "Stage 1: Hybrid Retrieval" in table_output
        assert "[PASSED]" in table_output
    finally:
        store.close()
        await sem_cache.close()


@pytest.mark.asyncio
async def test_benchmark_api_endpoint():
    """Verify POST /api/v1/benchmark/run returns formatted BenchmarkRunResponse DTO."""
    app = create_app()
    async with AsyncTestClient(app) as client:
        response = await client.post("/api/v1/benchmark/run")
        assert response.status_code == 201 or response.status_code == 200
        data = response.json()

        assert "stages" in data
        assert "subSecondSlaMet" in data
        assert "tier1CacheSlaMet" in data
        assert data["subSecondSlaMet"] is True
        assert data["tier1CacheSlaMet"] is True
        assert len(data["stages"]) >= 4
        stage_names = [s["stage"] for s in data["stages"]]
        assert any("Stage 1" in s for s in stage_names)
        assert any("Stage 3" in s for s in stage_names)
