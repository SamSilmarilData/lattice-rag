"""LangGraph state definition for the RAG agent loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """Typed state flowing through the LangGraph StateGraph.

    Each node in the graph reads and writes specific fields;
    the state accumulates results across pipeline stages.
    """

    # --- Input ---
    query: str = ""
    stream: bool = False
    filter_tags: list[str] = field(default_factory=list)

    # --- Routing ---
    route: str = ""
    route_confidence: float = 0.0

    # --- Caching ---
    cache_hit: bool = False
    cached_response: dict[str, Any] | None = None

    # --- Retrieval Stage 1 (Vector + BM25) ---
    query_embedding: Any = None  # np.ndarray at runtime
    stage1_chunks: list[dict[str, Any]] = field(default_factory=list)

    # --- Retrieval Stage 2 (Graph Traversal) ---
    graph_context: dict[str, Any] | None = None  # SubgraphResult as dict

    # --- Retrieval Stage 3 (Reranking) ---
    reranked_chunks: list[dict[str, Any]] = field(default_factory=list)

    # --- Guardrail (Jev Noul) ---
    filtered_chunks: list[dict[str, Any]] = field(default_factory=list)

    # --- Generation ---
    answer: str = ""
    degraded: bool = False

    # --- Telemetry ---
    timings: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
