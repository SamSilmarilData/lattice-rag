"""Buffered result of one pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """What ``RAGOrchestrator.run()`` returns: the same fields as a stream ``done`` event."""

    # --- Input ---
    query: str = ""
    use_cache: bool = True  # False: skip Tier 1 lookup and store (eval runs)

    # --- Routing ---
    route: str = ""

    # --- Caching ---
    cache_hit: bool = False

    # --- Retrieval ---
    graph_context: dict[str, Any] | None = None  # SubgraphResult as dict

    # --- Reranked chunks that passed the guardrail (Jev Noul) ---
    filtered_chunks: list[dict[str, Any]] = field(default_factory=list)
    low_grounding: bool = False
    # Unverified Answer: a Jev check (routing or guardrail) could not run for this query
    unverified: bool = False

    # --- Generation ---
    answer: str = ""
    degraded: bool = False
    provider: str = ""  # who answered: groq, gemini, redis_faq, offline; "" without generation

    # --- Telemetry ---
    timings: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
