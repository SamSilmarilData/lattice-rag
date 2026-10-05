from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import structlog

from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.storage.db import SubgraphResult

logger = structlog.get_logger(__name__)


class PrecisionReranker:
    """Stage 3: Cross-encoder precision reranking with linearized graph context."""

    def __init__(self, embedding_service: EmbeddingService):
        """Initialize PrecisionReranker.

        Args:
            embedding_service: Service to perform cross-encoder reranking.
        """
        self.embedding_service = embedding_service

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        graph_context: SubgraphResult,
        top_k: int = 5,
        max_triples: int = 15,
    ) -> list[dict]:
        """Rerank chunks based on query and linearized graph context.

        Args:
            query: The original search query.
            chunks: List of chunk dictionaries to rerank.
            graph_context: Subgraph result containing relationships.
            top_k: Number of top results to return.
            max_triples: Maximum number of graph triples to linearize.

        Returns:
            List of top_k dictionaries with original chunk data and rerank_score.
        """
        if not chunks:
            return []

        # Linearize graph triples from SubgraphResult (nodes and edges)
        node_name_map = {
            n["id"]: n["name"]
            for n in getattr(graph_context, "nodes", [])
            if isinstance(n, dict) and "id" in n and "name" in n
        }

        triples: list[str] = []
        edges = getattr(graph_context, "edges", [])
        for e in edges[:max_triples]:  # Cap at max_triples to balance graph context and inference speed
            if isinstance(e, dict):
                s_id = e.get("source_id")
                t_id = e.get("target_id")
                rel = e.get("relation_type", "RELATION")
                s_name = node_name_map.get(s_id, f"Node_{s_id}")
                t_name = node_name_map.get(t_id, f"Node_{t_id}")
                triples.append(f"{s_name} --[{rel}]--> {t_name}")

        graph_text = "\n".join(triples)[:1000]  # Cap at 1000 characters

        # Combine chunk texts with linearized graph triples
        combined_texts: list[str] = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if graph_text:
                combined_text = f"{chunk_text}\n\nRelevant Knowledge Graph Context:\n{graph_text}"
            else:
                combined_text = chunk_text
            combined_texts.append(combined_text)

        # Run cross-encoder scoring in thread pool
        rerank_results = await asyncio.to_thread(
            self.embedding_service.rerank,
            query,
            combined_texts,
            top_k=top_k,
        )

        # Map scores back to original chunk dictionaries
        final_results: list[dict] = []
        for res in rerank_results:
            chunk_data = dict(chunks[res.index])
            chunk_data["rerank_score"] = res.score
            final_results.append(chunk_data)

        logger.debug(
            "precision_rerank_completed",
            query=query,
            input_chunks=len(chunks),
            linearized_triples=len(triples),
            output_chunks=len(final_results),
        )
        return final_results
