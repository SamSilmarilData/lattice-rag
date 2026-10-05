"""Unit tests for the deep HybridRetriever module (TDD RED -> GREEN)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

from lattice_rag.retrieval.retriever import HybridRetriever
from lattice_rag.storage.db import SearchResult, SubgraphResult


class FakeStore:
    def __init__(self):
        self.vector_hits = [
            SearchResult(node_id=1, score=0.9, text="Chunk 1 text about LatticeDB", metadata={"text": "Chunk 1 text about LatticeDB"}),
            SearchResult(node_id=2, score=0.8, text="Chunk 2 text about HNSW", metadata={"text": "Chunk 2 text about HNSW"}),
        ]
        self.bm25_hits = [
            SearchResult(node_id=1, score=5.2, text="Chunk 1 text about LatticeDB", metadata={"text": "Chunk 1 text about LatticeDB"}),
            SearchResult(node_id=3, score=4.1, text="Chunk 3 text about Cypher", metadata={"text": "Chunk 3 text about Cypher"}),
        ]

    def vector_search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        return self.vector_hits

    def bm25_search(self, query_text: str, top_k: int = 10) -> list[SearchResult]:
        return self.bm25_hits

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        # Return entity nodes connected to chunk anchors
        return [{"entity_id": 10}, {"entity_id": 11}]

    def get_entities_for_chunk(self, chunk_id: int) -> list[int]:
        return [10, 11]

    def traverse_from_entities(self, entity_ids: list[int], max_hops: int = 1, budget: int = 25) -> SubgraphResult:
        return SubgraphResult(
            nodes=[{"id": 10, "name": "LatticeDB"}, {"id": 11, "name": "HNSW"}],
            edges=[{"source": 10, "target": 11, "type": "IMPLEMENTS"}],
        )


class FakeEmbeddingService:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_query(self, query: str) -> np.ndarray:
        return np.ones(self.dim, dtype=np.float32)

    def rerank(self, query: str, documents: list[str], top_k: int = 5):
        # Return sorted RerankResults with scores
        class FakeRerankResult:
            def __init__(self, index, score, text):
                self.index = index
                self.score = score
                self.text = text

        return [FakeRerankResult(i, 0.95 - (i * 0.1), doc) for i, doc in enumerate(documents[:top_k])]


@pytest.mark.asyncio
async def test_hybrid_retriever_empty_query_fast_exit():
    store = FakeStore()
    store.vector_hits = []
    store.bm25_hits = []
    embed_svc = FakeEmbeddingService()

    retriever = HybridRetriever(store=store, embedding_service=embed_svc)
    result = await retriever.retrieve(query="nonexistent topic", route="hybrid")

    assert result.final_chunks == []
    assert result.graph_context.nodes == []
    assert result.low_grounding is True
    assert "stage1_search" in result.timings


@pytest.mark.asyncio
async def test_hybrid_retriever_full_execution_flow():
    store = FakeStore()
    embed_svc = FakeEmbeddingService()

    retriever = HybridRetriever(store=store, embedding_service=embed_svc)
    result = await retriever.retrieve(
        query="Explain LatticeDB and HNSW",
        route="hybrid",
        top_k=2,
        max_triples=3,
    )

    assert len(result.final_chunks) == 2
    assert result.low_grounding is False
    assert len(result.graph_context.nodes) == 2
    assert len(result.graph_context.edges) == 1
    assert "stage1_search" in result.timings
    assert "stage2_traverse" in result.timings
    assert "stage3_rerank" in result.timings
    assert result.final_chunks[0]["rerank_score"] >= result.final_chunks[1]["rerank_score"]


@pytest.mark.asyncio
async def test_hybrid_retriever_guardrail_filtering():
    store = FakeStore()
    embed_svc = FakeEmbeddingService()

    mock_guardrail = MagicMock()
    # Guardrail filters out all but the first chunk
    async def _filter(query, chunks):
        return [chunks[0]]

    mock_guardrail.filter_chunks = AsyncMock(side_effect=_filter)

    retriever = HybridRetriever(
        store=store,
        embedding_service=embed_svc,
        guardrail=mock_guardrail,
    )

    result = await retriever.retrieve(query="LatticeDB details", route="hybrid", apply_guardrail=True)
    assert len(result.final_chunks) == 1
    assert "guardrail_filter" in result.timings
    mock_guardrail.filter_chunks.assert_awaited_once()


@pytest.mark.asyncio
async def test_hybrid_retriever_guardrail_all_rejected_activates_low_grounding():
    store = FakeStore()
    embed_svc = FakeEmbeddingService()

    mock_guardrail = MagicMock()
    # Guardrail rejects all chunks
    mock_guardrail.filter_chunks = AsyncMock(return_value=[])

    retriever = HybridRetriever(
        store=store,
        embedding_service=embed_svc,
        guardrail=mock_guardrail,
    )

    result = await retriever.retrieve(query="Irrelevant prompt", route="hybrid", apply_guardrail=True)
    assert result.final_chunks == []
    assert result.low_grounding is True
