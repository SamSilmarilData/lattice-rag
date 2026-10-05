"""Unit tests for Stage 1 HybridSearcher combining vector and BM25 search via RRF."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.vector_search import HybridSearcher
from lattice_rag.storage.db import SearchResult


@pytest.mark.asyncio
async def test_hybrid_search_empty_query():
    """Verify empty query returns empty list without querying store."""
    mock_store = MagicMock()
    mock_embed = MagicMock(spec=EmbeddingService)
    searcher = HybridSearcher(mock_store, mock_embed)

    results = await searcher.search("")
    assert results == []
    results_ws = await searcher.search("   ")
    assert results_ws == []
    mock_store.vector_search.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_search_fused_scoring_and_deduplication():
    """Verify Reciprocal Rank Fusion merges overlapping hits and ranks dual-matched nodes highest."""
    mock_store = MagicMock()
    mock_embed = MagicMock(spec=EmbeddingService)
    mock_embed.embed_query.return_value = np.zeros(384, dtype=np.float32)

    # Vector search returns: Node 1 (rank 0), Node 2 (rank 1)
    res_v1 = SearchResult(node_id=1, score=0.9, text="Chunk 1", metadata={"doc_id": "d1"})
    res_v2 = SearchResult(node_id=2, score=0.8, text="Chunk 2", metadata={"doc_id": "d1"})
    mock_store.vector_search.return_value = [res_v1, res_v2]

    # BM25 search returns: Node 2 (rank 0), Node 3 (rank 1)
    res_b2 = SearchResult(node_id=2, score=12.5, text="Chunk 2", metadata={"doc_id": "d1"})
    res_b3 = SearchResult(node_id=3, score=8.1, text="Chunk 3", metadata={"doc_id": "d2"})
    mock_store.bm25_search.return_value = [res_b2, res_b3]

    searcher = HybridSearcher(mock_store, mock_embed)
    results = await searcher.search("search query", top_k=5)

    assert len(results) == 3
    # Node 2 appears in both: score = 1/(60+1) + 1/(60+0) = 1/61 + 1/60 ~ 0.03306
    # Node 1 appears only in vector: 1/(60+0) = 1/60 ~ 0.01667
    # Node 3 appears only in BM25: 1/(60+1) = 1/61 ~ 0.01639
    assert results[0].node_id == 2
    assert results[1].node_id == 1
    assert results[2].node_id == 3

    # Check that score was updated to RRF score
    assert pytest.approx(results[0].score, rel=1e-3) == (1.0 / 61 + 1.0 / 60)
    assert pytest.approx(results[1].score, rel=1e-3) == (1.0 / 60)
    assert pytest.approx(results[2].score, rel=1e-3) == (1.0 / 61)


@pytest.mark.asyncio
async def test_hybrid_search_partial_results():
    """Verify hybrid search functions when one search type returns no results."""
    mock_store = MagicMock()
    mock_embed = MagicMock(spec=EmbeddingService)
    mock_embed.embed_query.return_value = np.zeros(384, dtype=np.float32)

    # Vector search has hits, BM25 returns empty
    res_v = SearchResult(node_id=10, score=0.95, text="Chunk 10", metadata={})
    mock_store.vector_search.return_value = [res_v]
    mock_store.bm25_search.return_value = []

    searcher = HybridSearcher(mock_store, mock_embed)
    results = await searcher.search("query", top_k=2)

    assert len(results) == 1
    assert results[0].node_id == 10
    assert pytest.approx(results[0].score, rel=1e-3) == (1.0 / 60)
