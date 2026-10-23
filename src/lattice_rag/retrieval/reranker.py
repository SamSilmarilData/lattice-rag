from __future__ import annotations

import asyncio
from typing import Any

import structlog

from lattice_rag.retrieval.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


class PrecisionReranker:
    """Stage 3: Cross-encoder precision reranking of candidate chunks.

    Chunks are scored on their own text. The graph context from Stage 2 is shared by every
    candidate, so appending it would not change their relative order; it is passed to the
    LLM prompt instead, where it adds information.
    """

    def __init__(self, embedding_service: EmbeddingService):
        """Initialize PrecisionReranker.

        Args:
            embedding_service: Service to perform cross-encoder reranking.
        """
        self.embedding_service = embedding_service

    async def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank chunks against the query with the cross-encoder.

        Args:
            query: The original search query.
            chunks: Candidate chunk dictionaries (all Stage 1 candidates).
            top_k: Number of top results to return.

        Returns:
            Up to top_k chunk dictionaries ordered by ``rerank_score``.
        """
        if not chunks:
            return []

        texts = [chunk.get("text", "") for chunk in chunks]
        rerank_results = await asyncio.to_thread(
            self.embedding_service.rerank, query, texts, top_k=top_k
        )

        final_results: list[dict[str, Any]] = []
        for res in rerank_results:
            chunk_data = dict(chunks[res.index])
            chunk_data["rerank_score"] = res.score
            final_results.append(chunk_data)

        logger.debug(
            "precision_rerank_completed",
            input_chunks=len(chunks),
            output_chunks=len(final_results),
        )
        return final_results
