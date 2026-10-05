from __future__ import annotations

import asyncio
from typing import Any, List

import structlog

from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.storage.db import SearchResult

logger = structlog.get_logger(__name__)


class HybridSearcher:
    """Stage 1: HNSW vector search + BM25 keyword recall merged via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, store: Any, embedding_service: EmbeddingService):
        """Initialize HybridSearcher.

        Args:
            store: The LatticeDB store instance.
            embedding_service: Service to generate embeddings.
        """
        self.store = store
        self.embedding_service = embedding_service

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Perform a hybrid search combining vector search and BM25 using RRF.

        Args:
            query: The search query.
            top_k: Number of final fused results to return.

        Returns:
            List of SearchResult deduplicated by node_id.
        """
        if not query or not query.strip():
            return []

        # Concurrently execute CPU query embedding and BM25 full-text search
        query_embedding, bm25_results = await asyncio.gather(
            asyncio.to_thread(self.embedding_service.embed_query, query),
            asyncio.to_thread(self.store.bm25_search, query, top_k=top_k * 2),
        )

        # Run vector search with ready embedding
        vector_results = await asyncio.to_thread(
            self.store.vector_search, query_embedding, top_k=top_k * 2
        )

        # Merge using Reciprocal Rank Fusion (RRF)
        # score = sum(1 / (k + rank)) for each document across both result lists, k=60
        k = 60
        fused_scores: dict[int, float] = {}
        node_map: dict[int, SearchResult] = {}

        for rank, res in enumerate(vector_results):
            node_id = res.node_id
            if node_id not in fused_scores:
                fused_scores[node_id] = 0.0
                node_map[node_id] = res
            fused_scores[node_id] += 1.0 / (k + rank)

        for rank, res in enumerate(bm25_results):
            node_id = res.node_id
            if node_id not in fused_scores:
                fused_scores[node_id] = 0.0
                node_map[node_id] = res
            fused_scores[node_id] += 1.0 / (k + rank)

        # Select top_k by fused score using O(M log K) heap
        import heapq

        top_nodes = heapq.nlargest(top_k, fused_scores.items(), key=lambda x: x[1])

        # Build final top_k deduplicated results
        final_results: list[SearchResult] = []
        for node_id, score in top_nodes:
            res = node_map[node_id]
            res.score = score
            final_results.append(res)

        logger.debug(
            "hybrid_search_completed",
            query=query,
            vector_hits=len(vector_results),
            bm25_hits=len(bm25_results),
            fused_hits=len(final_results),
        )
        return final_results
