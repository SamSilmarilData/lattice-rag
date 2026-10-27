from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.graph_traversal import GraphTraverser
from lattice_rag.retrieval.reranker import PrecisionReranker
from lattice_rag.retrieval.vector_search import HybridSearcher
from lattice_rag.routing.guardrail import RouteGuardrails
from lattice_rag.security import query_fingerprint, sanitize_error_detail
from lattice_rag.storage.db import SubgraphResult
from lattice_rag.vocabulary import Route

logger = logging.getLogger(__name__)

# Extra Stage 1 candidates the cross-encoder may promote into the top_k. Every candidate is
# reranked, and on CPU each costs roughly 100 ms, so this trades recall against latency.
RERANK_HEADROOM = 3


@dataclass
class RetrievalResult:
    """Final result of the 3-stage hybrid retrieval pipeline + guardrail."""

    final_chunks: list[dict[str, Any]]
    graph_context: SubgraphResult
    timings: dict[str, float] = field(default_factory=dict)
    low_grounding: bool = False
    # True when the guardrail could not confirm grounding (not configured, or Jev failed)
    unverified: bool = False


class RetrievalPipeline:
    """Unified 3-stage hybrid retrieval pipeline with guardrail filtering and telemetry."""

    def __init__(
        self,
        store: Any,
        embedding_service: EmbeddingService,
        router_guardrail: Any = None,
        *,
        forced_hops: int | None = None,
    ) -> None:
        """Initialize the unified retrieval pipeline.

        Args:
            store: The LatticeDB store instance.
            embedding_service: Service to generate embeddings and rerank.
            router_guardrail: Optional guardrail to filter chunks (ContextGuardrail), or a
                :class:`RouteGuardrails` choosing one per Route.
            forced_hops: Eval-only Route ablation (#46): the three retrieval Routes walk this
                many hops; ``massive_context`` keeps its own count.
        """
        self.store = store
        self.embedding_service = embedding_service
        self.guardrails = (
            router_guardrail
            if isinstance(router_guardrail, RouteGuardrails)
            else RouteGuardrails(router_guardrail)
        )
        self.forced_hops = forced_hops

        self.searcher = HybridSearcher(self.store, self.embedding_service)
        self.traverser = GraphTraverser(self.store, forced_hops=forced_hops)
        self.reranker = PrecisionReranker(self.embedding_service)

    def with_store(self, store: Any) -> RetrievalPipeline:
        """The same pipeline (per-Route guardrails, forced hops) over another store, e.g. an eval temp DB."""
        return RetrievalPipeline(
            store,
            self.embedding_service,
            router_guardrail=self.guardrails,
            forced_hops=self.forced_hops,
        )

    async def retrieve(
        self,
        query: str,
        route: Route = "hybrid",
        apply_guardrail: bool = True,
        top_k: int = 5,
        budget: int = 25,
        query_embedding: Any | None = None,
    ) -> RetrievalResult:
        """Execute the 3-stage retrieval pipeline with internal telemetry.

        Steps run in this order, and ``timings`` records them in the same order:

        1. ``search``: Stage 1, HNSW vector search + BM25 keyword recall with RRF
           fusion (top_k + RERANK_HEADROOM candidates).
        2. ``rerank``: Stage 3, cross-encoder reranking of every candidate down to top_k.
        3. ``guardrail_filter``: optional TypeSafe Jev Noul context guardrail pruning.
        4. ``graph_traverse``: Stage 2, 1-to-2 hop graph traversal from the entities of
           the final chunks only.

        Stage numbers are the spec's names; graph traversal runs last so it only expands
        from evidence that survived reranking and the guardrail.

        ``query_embedding`` lets callers that already embedded the query skip re-embedding.
        """
        timings: dict[str, float] = {}

        # ── Step 1 (Stage 1): Vector Search + BM25 ───────────────────────────
        t0 = time.perf_counter()
        stage1_results = await self.searcher.search(
            query, top_k=top_k + RERANK_HEADROOM, query_embedding=query_embedding
        )
        timings["search"] = (time.perf_counter() - t0) * 1000.0

        if not stage1_results:
            logger.info("stage1_empty_results", extra=query_fingerprint(query))
            return RetrievalResult(
                final_chunks=[],
                graph_context=SubgraphResult(nodes=[], edges=[]),
                timings=timings,
                low_grounding=True,
            )

        stage1_chunks: list[dict[str, Any]] = []
        for res in stage1_results:
            # Source metadata (doc_id, position) travels with the chunk for citations
            chunk_dict: dict[str, Any] = {**(res.metadata or {})}
            chunk_dict.update(node_id=res.node_id, score=res.score, text=res.text)
            stage1_chunks.append(chunk_dict)

        # ── Step 2 (Stage 3): Cross-Encoder Precision Reranking ──────────────
        t0 = time.perf_counter()
        # Rerank every candidate: truncating first would leave the cross-encoder
        # only able to reorder what RRF already chose.
        reranked = await self.reranker.rerank(query=query, chunks=stage1_chunks, top_k=top_k)
        timings["rerank"] = (time.perf_counter() - t0) * 1000.0

        final_chunks = reranked
        low_grounding = False
        # Grounding counts as verified only when the guardrail actually ran
        unverified = True

        # ── Step 3: Guardrail Context Relevance Filtering ────────────────────
        guardrail = self.guardrails.for_route(route) if apply_guardrail else None
        if guardrail is not None:
            t0 = time.perf_counter()
            try:
                verdict = await guardrail.filter_chunks(query, reranked)
                timings["guardrail_filter"] = (time.perf_counter() - t0) * 1000.0
                final_chunks = verdict.chunks
                low_grounding = verdict.low_grounding or not verdict.chunks
                unverified = not verdict.verified
            except Exception as e:
                logger.warning(
                    "guardrail_filter_unexpected_error_fallback: error=%s",
                    sanitize_error_detail(str(e)),
                )
                final_chunks = reranked

        # ── Step 4 (Stage 2): Graph Traversal ────────────────────────────────
        # Runs last so the graph context only holds entities of the chunks that
        # survived reranking and the guardrail; the LLM never sees relations
        # reachable only from rejected candidates.
        t0 = time.perf_counter()
        anchor_ids = [chunk["node_id"] for chunk in final_chunks if "node_id" in chunk]
        graph_context = await self.traverser.traverse(anchor_ids, route, budget=budget)
        timings["graph_traverse"] = (time.perf_counter() - t0) * 1000.0

        return RetrievalResult(
            final_chunks=final_chunks,
            graph_context=graph_context,
            timings=timings,
            low_grounding=low_grounding,
            unverified=unverified,
        )

    async def execute(
        self,
        query: str,
        route: Route = "hybrid",
        apply_guardrail: bool = True,
        top_k: int = 5,
        budget: int = 25,
        query_embedding: Any | None = None,
    ) -> RetrievalResult:
        """Execute the full retrieval pipeline (alias for retrieve)."""
        return await self.retrieve(
            query=query,
            route=route,
            apply_guardrail=apply_guardrail,
            top_k=top_k,
            budget=budget,
            query_embedding=query_embedding,
        )


# Alias for backward compatibility
HybridRetriever = RetrievalPipeline
