"""Unit tests for EmbeddingService and ONNXReranker."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from lattice_rag.retrieval.embeddings import EmbeddingService, ONNXReranker, RerankResult


def test_embedding_service_with_injected_mocks():
    """Verify embed_texts, embed_query, and rerank with mocked models."""
    mock_embedder = MagicMock()
    v1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    v2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    mock_embedder.embed.return_value = [v1, v2]

    mock_reranker = MagicMock()
    # Mock reranker returning scores for 3 documents
    mock_reranker.rerank.return_value = [0.25, 0.89, 0.12]

    service = EmbeddingService(
        embed_model="mock-embed",
        reranker_model="mock-rerank",
        embedder=mock_embedder,
        reranker=mock_reranker,
    )

    # Test embed_texts
    texts = ["doc1", "doc2"]
    embs = service.embed_texts(texts)
    assert len(embs) == 2
    np.testing.assert_array_equal(embs[0], v1)

    # Test embed_query
    mock_embedder.embed.return_value = [v1]
    query_emb = service.embed_query("search query")
    np.testing.assert_array_equal(query_emb, v1)

    # Test rerank (should sort descending: doc 1 with 0.89 first, doc 0 with 0.25 second)
    docs = ["doc 0", "doc 1", "doc 2"]
    reranked = service.rerank("query", docs, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].index == 1
    assert reranked[0].score == 0.89
    assert reranked[0].text == "doc 1"

    assert reranked[1].index == 0
    assert reranked[1].score == 0.25
    assert reranked[1].text == "doc 0"


def test_embedding_service_empty_inputs():
    """Verify empty texts/documents return empty results without error."""
    service = EmbeddingService(embedder=MagicMock(), reranker=MagicMock())
    assert service.embed_texts([]) == []
    assert service.rerank("query", []) == []


def test_onnx_reranker_real_inference():
    """Verify real INT8 ONNX bge-reranker-v2-m3 reranks query-document pairs."""
    reranker = ONNXReranker()
    query = "What is LatticeDB?"
    docs = [
        "LatticeDB is an embedded property-graph database with native vector search.",
        "Bananas are rich in dietary potassium and vitamins.",
    ]
    scores = reranker.rerank(query, docs)
    assert len(scores) == 2
    # Relevant technical doc must score higher than unrelated fruit text
    assert scores[0] > scores[1]
    assert scores[0] > 0.0  # High positive logit for relevant match
    assert scores[1] < 0.0  # Negative logit for irrelevant match
