"""Deep HybridRetriever module unifying vector search, graph traversal, and reranking.

Consolidated into lattice_rag.retrieval.pipeline.RetrievalPipeline for architectural simplicity.
Re-exported here for 100% backward compatibility.
"""
from __future__ import annotations

from lattice_rag.retrieval.pipeline import (
    HybridRetriever,
    RetrievalPipeline,
    RetrievalResult,
)

__all__ = [
    "HybridRetriever",
    "RetrievalPipeline",
    "RetrievalResult",
]
