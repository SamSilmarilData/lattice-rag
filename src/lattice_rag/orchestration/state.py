"""Buffered result of one pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lattice_rag.api.dtos import GenerationProvider


@dataclass
class PipelineState:
    """What ``RAGOrchestrator.run()`` returns: the same fields as a stream ``done`` event."""

    # --- Input ---
    query: str = ""
    use_cache: bool = True  # False: skip Tier 1 lookup and store (eval runs)

    # --- Routing ---
    route: str = ""
    # False when the route is the System 1 fallback because Jev could not choose one
    route_verified: bool = True

    # --- Caching ---
    cache_hit: bool = False

    # --- Retrieval ---
    graph_context: dict[str, Any] | None = None  # SubgraphResult as dict
    # The part of graph_context the generator was given (GRAPH_CONTEXT_IN_PROMPT); the
    # Live Tier judge sees exactly this, so it never judges evidence the answer lacked
    prompt_graph_context: dict[str, Any] | None = None

    # --- Reranked chunks that passed the guardrail (Jev Noul) ---
    filtered_chunks: list[dict[str, Any]] = field(default_factory=list)
    low_grounding: bool = False
    # Unverified Answer: a Jev check (routing or guardrail) could not run for this query
    unverified: bool = False

    # --- Generation ---
    answer: str = ""
    degraded: bool = False
    # Provider that generated the answer; None when nothing was generated (chitchat, cache hit)
    provider: GenerationProvider | None = None

    # --- Telemetry ---
    timings: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
