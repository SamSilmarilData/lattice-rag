"""Unit tests for Stage 3 PrecisionReranker with graph context linearization."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lattice_rag.retrieval.embeddings import EmbeddingService, RerankResult
from lattice_rag.retrieval.reranker import PrecisionReranker
from lattice_rag.storage.db import SubgraphResult


@pytest.mark.asyncio
async def test_reranker_empty_chunks():
    """Verify empty chunks returns empty list immediately."""
    mock_embed = MagicMock(spec=EmbeddingService)
    reranker = PrecisionReranker(mock_embed)

    result = await reranker.rerank("query", [], SubgraphResult(nodes=[], edges=[]))
    assert result == []
    mock_embed.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_reranker_graph_linearization_and_scoring():
    """Verify SubgraphResult nodes and edges are linearized into triples and combined with chunk text."""
    mock_embed = MagicMock(spec=EmbeddingService)
    # Mock rerank return values
    mock_embed.rerank.return_value = [
        RerankResult(index=0, score=8.5, text="doc 0"),
        RerankResult(index=1, score=1.2, text="doc 1"),
    ]

    reranker = PrecisionReranker(mock_embed)

    subgraph = SubgraphResult(
        nodes=[
            {"id": 1, "name": "LatticeDB", "label": "Database"},
            {"id": 2, "name": "HNSW", "label": "Index"},
        ],
        edges=[
            {"source_id": 1, "target_id": 2, "relation_type": "INDEXED_BY"},
        ],
    )

    chunks = [
        {"node_id": 10, "text": "LatticeDB provides vector search.", "score": 0.8},
        {"node_id": 20, "text": "Unrelated paragraph.", "score": 0.3},
    ]

    results = await reranker.rerank("How does LatticeDB index vectors?", chunks, subgraph, top_k=2)

    assert len(results) == 2
    assert results[0]["node_id"] == 10
    assert results[0]["rerank_score"] == 8.5
    assert results[1]["node_id"] == 20
    assert results[1]["rerank_score"] == 1.2

    # Verify that mock_embed.rerank was called with combined text containing linearized triples
    call_args = mock_embed.rerank.call_args
    passed_query = call_args[0][0]
    passed_docs = call_args[0][1]

    assert passed_query == "How does LatticeDB index vectors?"
    assert len(passed_docs) == 2
    assert "LatticeDB provides vector search." in passed_docs[0]
    assert "LatticeDB --[INDEXED_BY]--> HNSW" in passed_docs[0]


@pytest.mark.asyncio
async def test_reranker_triples_capping():
    """Verify graph linearization caps at 15 triples."""
    mock_embed = MagicMock(spec=EmbeddingService)
    mock_embed.rerank.return_value = [RerankResult(index=0, score=5.0, text="doc 0")]

    reranker = PrecisionReranker(mock_embed)

    # Subgraph with 25 edges
    nodes = [{"id": i, "name": f"Entity_{i}"} for i in range(30)]
    edges = [{"source_id": i, "target_id": i + 1, "relation_type": f"REL_{i}"} for i in range(25)]
    subgraph = SubgraphResult(nodes=nodes, edges=edges)

    chunks = [{"node_id": 100, "text": "Test chunk"}]
    await reranker.rerank("query", chunks, subgraph, top_k=1)

    call_docs = mock_embed.rerank.call_args[0][1]
    combined_doc = call_docs[0]

    # Should contain REL_0 through REL_14 (15 triples), but not REL_15
    assert "REL_0" in combined_doc
    assert "REL_14" in combined_doc
    assert "REL_15" not in combined_doc
