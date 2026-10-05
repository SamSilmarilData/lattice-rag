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
        from lattice_rag.retrieval.retriever import HybridRetriever
        self._retriever = HybridRetriever(self.store, self.embedding_service, self.router_guardrail)

    async def execute(
        self,
        query: str,
        route: str,
        apply_guardrail: bool = True,
    ) -> RetrievalResult:
        """Execute the full retrieval pipeline via HybridRetriever."""
        return await self._retriever.retrieve(
            query=query,
            route=route,
            apply_guardrail=apply_guardrail,
            top_k=5,
            max_triples=5,
        )
