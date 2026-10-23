"""Unit tests for Stage 3 PrecisionReranker."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lattice_rag.retrieval.embeddings import EmbeddingService, RerankResult
from lattice_rag.retrieval.reranker import PrecisionReranker


@pytest.mark.asyncio
async def test_reranker_empty_chunks():
    """Verify empty chunks returns empty list immediately."""
    mock_embed = MagicMock(spec=EmbeddingService)
    reranker = PrecisionReranker(mock_embed)

    result = await reranker.rerank("query", [])
    assert result == []
    mock_embed.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_reranker_scores_chunk_text_and_orders_results():
    """Chunks are scored on their own text and returned in cross-encoder order."""
    mock_embed = MagicMock(spec=EmbeddingService)
    mock_embed.rerank.return_value = [
        RerankResult(index=1, score=8.5, text="doc 1"),
        RerankResult(index=0, score=1.2, text="doc 0"),
    ]
    reranker = PrecisionReranker(mock_embed)
    chunks = [
        {"node_id": 10, "text": "Unrelated paragraph.", "score": 0.8, "doc_id": "a"},
        {"node_id": 20, "text": "LatticeDB provides vector search.", "score": 0.3, "doc_id": "b"},
    ]

    results = await reranker.rerank("How does LatticeDB index vectors?", chunks, top_k=2)

    assert [r["node_id"] for r in results] == [20, 10]
    assert results[0]["rerank_score"] == 8.5
    assert results[0]["doc_id"] == "b"
    passed_query, passed_docs = mock_embed.rerank.call_args[0][:2]
    assert passed_query == "How does LatticeDB index vectors?"
    assert passed_docs == ["Unrelated paragraph.", "LatticeDB provides vector search."]
    assert mock_embed.rerank.call_args.kwargs["top_k"] == 2
