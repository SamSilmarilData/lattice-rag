from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.graph_traversal import GraphTraverser
from lattice_rag.retrieval.reranker import PrecisionReranker
from lattice_rag.retrieval.vector_search import HybridSearcher
from lattice_rag.storage.db import SubgraphResult

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalResult:
    """Final result of the 3-stage hybrid retrieval pipeline + guardrail."""

    final_chunks: list[dict]
    graph_context: SubgraphResult
    timings: dict[str, float] = field(default_factory=dict)
    low_grounding: bool = False


class RetrievalPipeline:
    """Orchestrator combining all 3 stages + guardrail."""

    def __init__(
        self,
        store: Any,
        embedding_service: EmbeddingService,
        router_guardrail: Any = None,
    ):
        """Initialize the RetrievalPipeline.

        Args:
            store: The LatticeDB store instance.
            embedding_service: Service to generate embeddings and rerank.
            router_guardrail: Optional guardrail to filter chunks (ContextGuardrail).
        """
        self.store = store
        self.embedding_service = embedding_service
        self.router_guardrail = router_guardrail

        self.hybrid_searcher = HybridSearcher(self.store, self.embedding_service)
        self.graph_traverser = GraphTraverser(self.store)
        self.precision_reranker = PrecisionReranker(self.embedding_service)

    async def execute(
        self,
        query: str,
        route: str,
        apply_guardrail: bool = True,
    ) -> RetrievalResult:
        """Execute the full retrieval pipeline:
        Stage 1: Vector + BM25 Hybrid Search
        Stage 2: Graph Traversal
        Stage 3: Precision Reranking
        Guardrail: Context Relevance Filtering

        Args:
            query: The search query.
            route: The chosen route strategy (e.g., 'hybrid', 'graph_relational', 'vector_exact').
            apply_guardrail: Whether to run TypeSafe Jev Noul guardrail filtering.

        Returns:
            RetrievalResult containing chunks, graph context, stage timings, and grounding status.
        """
        timings: dict[str, float] = {}

        # Stage 1: Vector Search + BM25
        t0 = time.perf_counter()
        stage1_results = await self.hybrid_searcher.search(query, top_k=10)
        timings["stage1_search"] = (time.perf_counter() - t0) * 1000.0

        # Fast-exit if no candidate chunks found
        if not stage1_results:
            logger.info("stage1_empty_results", query=query)
            return RetrievalResult(
                final_chunks=[],
                graph_context=SubgraphResult(nodes=[], edges=[]),
                timings=timings,
                low_grounding=True,
            )

        # Extract anchor IDs and prepare chunk dicts
        anchor_ids = [res.node_id for res in stage1_results]
        stage1_chunks: list[dict] = []
        for res in stage1_results:
            chunk_dict = {"node_id": res.node_id, "score": res.score}
            if hasattr(res, "text") and res.text:
                chunk_dict["text"] = res.text
            elif hasattr(res, "metadata") and isinstance(res.metadata, dict):
                chunk_dict["text"] = res.metadata.get("text", "")
            stage1_chunks.append(chunk_dict)

        # Stage 2: Graph Traversal
        t0 = time.perf_counter()
        graph_context = await self.graph_traverser.traverse(anchor_ids, route)
        timings["stage2_traverse"] = (time.perf_counter() - t0) * 1000.0

        # Stage 3: Precision Reranking (rerank top 5 candidates)
        t0 = time.perf_counter()
        stage3_results = await self.precision_reranker.rerank(
            query, stage1_chunks[:5], graph_context, top_k=5, max_triples=5
        )
        timings["stage3_rerank"] = (time.perf_counter() - t0) * 1000.0

        # Guardrail Filtering
        t0 = time.perf_counter()
        final_chunks = stage3_results
        low_grounding = False

        if apply_guardrail and self.router_guardrail is not None:
            filtered = self.router_guardrail.filter_chunks(query, stage3_results)
            if hasattr(filtered, "__await__"):
                final_chunks = await filtered
            else:
                final_chunks = filtered

            low_grounding = getattr(self.router_guardrail, "low_grounding_flag", False)
            if any(c.get("low_grounding_flag") for c in final_chunks if isinstance(c, dict)):
                low_grounding = True

        timings["guardrail_filter"] = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "retrieval_pipeline_completed",
            query=query,
            route=route,
            final_chunks=len(final_chunks),
            low_grounding=low_grounding,
            total_time_ms=sum(timings.values()),
        )

        return RetrievalResult(
            final_chunks=final_chunks,
            graph_context=graph_context,
            timings=timings,
            low_grounding=low_grounding,
        )
