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
    use_cache: bool = True  # False: skip Tier 1 lookup and store (eval runs)

    # --- Routing ---
    route: str = ""

    # --- Caching ---
    cache_hit: bool = False
    cached_response: dict[str, Any] | None = None

    # --- Retrieval (query embedding is computed once in triage and reused) ---
    query_embedding: Any = None  # np.ndarray at runtime
    graph_context: dict[str, Any] | None = None  # SubgraphResult as dict

    # --- Reranked chunks that passed the guardrail (Jev Noul) ---
    filtered_chunks: list[dict[str, Any]] = field(default_factory=list)
    low_grounding: bool = False

    # --- Generation ---
    answer: str = ""
    degraded: bool = False

    # --- Telemetry ---
    timings: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
