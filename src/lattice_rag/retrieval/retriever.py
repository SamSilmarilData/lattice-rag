"""Deep HybridRetriever module unifying vector search, graph traversal, and reranking."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.graph_traversal import GraphTraverser
from lattice_rag.retrieval.pipeline import RetrievalResult
from lattice_rag.retrieval.reranker import PrecisionReranker
from lattice_rag.retrieval.vector_search import HybridSearcher
from lattice_rag.storage.db import SubgraphResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Deep retrieval module consolidating 3-stage hybrid search and guardrail filtering."""

    def __init__(
        self,
        store: Any,
        embedding_service: EmbeddingService,
        guardrail: Any = None,
    ) -> None:
        self.store = store
        self.embedding_service = embedding_service
        self.guardrail = guardrail

        # Encapsulated internal stage executors
        self.searcher = HybridSearcher(self.store, self.embedding_service)
        self.traverser = GraphTraverser(self.store)
        self.reranker = PrecisionReranker(self.embedding_service)

    async def retrieve(
        self,
        query: str,
        route: str = "hybrid",
        apply_guardrail: bool = True,
        top_k: int = 5,
        max_triples: int = 5,
        budget: int = 25,
    ) -> RetrievalResult:
        """Execute the 3-stage retrieval pipeline with internal telemetry.

        Stage 1: HNSW vector search + BM25 keyword recall with RRF fusion.
        Stage 2: Dynamic 1-to-2 hop Cypher graph traversal from entity anchors.
        Stage 3: Cross-encoder precision reranking with linearized triple capping.
        Optional: TypeSafe Jev Noul context guardrail pruning.
        """
        timings: dict[str, float] = {}

        # ── Stage 1: Vector Search + BM25 ────────────────────────────────────
        t0 = time.perf_counter()
        stage1_results = await self.searcher.search(query, top_k=max(top_k * 2, 10))
        timings["stage1_search"] = (time.perf_counter() - t0) * 1000.0

        if not stage1_results:
            logger.info("stage1_empty_results", extra={"query": query})
            return RetrievalResult(
                final_chunks=[],
                graph_context=SubgraphResult(nodes=[], edges=[]),
                timings=timings,
                low_grounding=True,
            )

        anchor_ids = [res.node_id for res in stage1_results]
        stage1_chunks: list[dict[str, Any]] = []
        for res in stage1_results:
            chunk_dict: dict[str, Any] = {"node_id": res.node_id, "score": res.score}
            if hasattr(res, "text") and res.text:
                chunk_dict["text"] = res.text
            elif hasattr(res, "metadata") and isinstance(res.metadata, dict):
                chunk_dict["text"] = res.metadata.get("text", "")
            stage1_chunks.append(chunk_dict)

        # ── Stage 2: Dynamic Cypher Graph Traversal ──────────────────────────
        t0 = time.perf_counter()
        graph_context = await self.traverser.traverse(anchor_ids, route, budget=budget)
        timings["stage2_traverse"] = (time.perf_counter() - t0) * 1000.0

        # ── Stage 3: Cross-Encoder Precision Reranking ───────────────────────
        t0 = time.perf_counter()
        candidate_chunks = stage1_chunks[:top_k]
        stage3_results = await self.reranker.rerank(
            query=query,
            chunks=candidate_chunks,
            graph_context=graph_context,
            top_k=top_k,
            max_triples=max_triples,
        )
        timings["stage3_rerank"] = (time.perf_counter() - t0) * 1000.0

        final_chunks = stage3_results
        low_grounding = False

        # ── Guardrail: Context Relevance Filtering ───────────────────────────
        if apply_guardrail and self.guardrail is not None:
            t0 = time.perf_counter()
            filtered = self.guardrail.filter_chunks(query, stage3_results)
            if hasattr(filtered, "__await__"):
                filtered_chunks = await filtered
            else:
                filtered_chunks = filtered
            duration_ms = (time.perf_counter() - t0) * 1000.0
            timings["guardrail_filter"] = duration_ms

            has_flag = getattr(self.guardrail, "low_grounding_flag", False)
            if not filtered_chunks or has_flag is True or any(c.get("low_grounding_flag") for c in filtered_chunks if isinstance(c, dict)):
                low_grounding = True
            final_chunks = filtered_chunks

        return RetrievalResult(
            final_chunks=final_chunks,
            graph_context=graph_context,
            timings=timings,
            low_grounding=low_grounding,
        )
