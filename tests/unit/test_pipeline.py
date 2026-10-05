"""Unit tests for RetrievalPipeline orchestrator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.retrieval.embeddings import EmbeddingService, RerankResult
from lattice_rag.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from lattice_rag.storage.db import SearchResult, SubgraphResult


@pytest.mark.asyncio
async def test_pipeline_empty_stage1_fast_exit():
    """Verify empty Stage 1 returns empty result with low_grounding=True immediately."""
    mock_store = MagicMock()
    mock_store.vector_search.return_value = []
    mock_store.bm25_search.return_value = []

    mock_embed = MagicMock(spec=EmbeddingService)
    mock_embed.embed_query.return_value = [0.0] * 384

    pipeline = RetrievalPipeline(mock_store, mock_embed)
    result = await pipeline.execute("query with no matches", route="hybrid")

    assert isinstance(result, RetrievalResult)
    assert result.final_chunks == []
    assert result.graph_context.nodes == []
    assert result.graph_context.edges == []
    assert result.low_grounding is True
    assert "stage1_search" in result.timings


@pytest.mark.asyncio
async def test_pipeline_full_execution_flow():
    """Verify complete execution across Stage 1, Stage 2, Stage 3, and Guardrail."""
    mock_store = MagicMock()
    # Stage 1: Vector search returns node 1, BM25 returns node 1
    hit = SearchResult(node_id=1, score=0.9, text="Chunk 1 content", metadata={})
    mock_store.vector_search.return_value = [hit]
    mock_store.bm25_search.return_value = [hit]

    # Stage 2: Chunk 1 links to Entity 10
    mock_store.get_entities_for_chunk.return_value = [10]
    expected_subgraph = SubgraphResult(
        nodes=[{"id": 10, "name": "LatticeDB", "label": "Database"}],
        edges=[],
    )
    mock_store.traverse_from_entities.return_value = expected_subgraph

    # Stage 3: Embedding service returns reranked result
    mock_embed = MagicMock(spec=EmbeddingService)
    mock_embed.embed_query.return_value = [0.0] * 384
    mock_embed.rerank.return_value = [
        RerankResult(index=0, score=9.2, text="Chunk 1 content")
    ]

    # Guardrail: Passes chunk
    mock_guardrail = AsyncMock()
    passed_chunk = {"node_id": 1, "text": "Chunk 1 content", "rerank_score": 9.2, "grounding_score": 0.85}
    mock_guardrail.filter_chunks.return_value = [passed_chunk]
    mock_guardrail.low_grounding_flag = False

    pipeline = RetrievalPipeline(mock_store, mock_embed, router_guardrail=mock_guardrail)
    result = await pipeline.execute("Explain LatticeDB", route="hybrid")

    assert len(result.final_chunks) == 1
    assert result.final_chunks[0]["node_id"] == 1
    assert result.graph_context == expected_subgraph
    assert result.low_grounding is False

    # Check timings are present and non-negative
    for key in ["stage1_search", "stage2_traverse", "stage3_rerank", "guardrail_filter"]:
        assert key in result.timings
        assert result.timings[key] >= 0.0


@pytest.mark.asyncio
async def test_pipeline_guardrail_insufficient_evidence_propagates_flag():
    """Verify low_grounding is flagged when guardrail filters all chunks."""
    mock_store = MagicMock()
    hit = SearchResult(node_id=5, score=0.5, text="Tangential chunk", metadata={})
    mock_store.vector_search.return_value = [hit]
    mock_store.bm25_search.return_value = []
    mock_store.get_entities_for_chunk.return_value = []
    mock_store.traverse_from_entities.return_value = SubgraphResult(nodes=[], edges=[])

    mock_embed = MagicMock(spec=EmbeddingService)
    mock_embed.embed_query.return_value = [0.0] * 384
    mock_embed.rerank.return_value = [RerankResult(index=0, score=1.0, text="Tangential chunk")]

    # Guardrail rejects all chunks
    mock_guardrail = AsyncMock()
    mock_guardrail.filter_chunks.return_value = [
        {"node_id": -1, "text": "INSUFFICIENT_EVIDENCE", "low_grounding_flag": True}
    ]
    mock_guardrail.low_grounding_flag = True

    pipeline = RetrievalPipeline(mock_store, mock_embed, router_guardrail=mock_guardrail)
    result = await pipeline.execute("Difficult question", route="hybrid")

    assert result.low_grounding is True
    assert result.final_chunks[0]["low_grounding_flag"] is True
