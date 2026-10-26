from __future__ import annotations

from datetime import datetime  # noqa: TC003  (msgspec resolves annotations at runtime)
from enum import StrEnum
from typing import Annotated, Any

import msgspec

from lattice_rag.vocabulary import (  # noqa: TC001  (msgspec resolves annotations at runtime)
    Route,
    Slice,
)

# Request size bounds: keep user-supplied text small enough that embedding,
# GLiNER extraction, and LLM prompts stay bounded per request.
MAX_QUERY_CHARS = 2_000
MAX_DOCUMENT_CHARS = 200_000

QueryText = Annotated[str, msgspec.Meta(min_length=1, max_length=MAX_QUERY_CHARS)]
DOCUMENT_ID_PATTERN = r"^[\w.-]{1,128}$"
DocumentId = Annotated[str, msgspec.Meta(pattern=DOCUMENT_ID_PATTERN)]

# ADR-0001: the deployed instance serves a Seeded Database from ephemeral disk
RUNTIME_DOCUMENT_PERSISTENCE = (
    "ephemeral: on the deployed instance, documents ingested at runtime do not survive "
    "a restart or redeploy; durable content is added with `lattice-rag seed` at build time"
)
Title = Annotated[str, msgspec.Meta(min_length=1, max_length=512)]
DocumentText = Annotated[str, msgspec.Meta(min_length=1, max_length=MAX_DOCUMENT_CHARS)]
Tag = Annotated[str, msgspec.Meta(min_length=1, max_length=64)]


class QueryRequest(msgspec.Struct, rename="camel"):
    """Request DTO for queries."""

    query: QueryText


class SourceChunk(msgspec.Struct, rename="camel"):
    """DTO representing a chunk of text from a source document."""

    chunk_id: int
    text: str
    score: float
    document_id: str
    position: int

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_index: int = 0) -> SourceChunk:
        return cls(
            chunk_id=int(data.get("node_id") or data.get("chunk_id") or default_index),
            text=str(data.get("text", "")),
            score=float(data.get("rerank_score") or data.get("score") or 0.0),
            document_id=str(data.get("doc_id") or data.get("document_id") or "unknown"),
            position=int(data.get("position", default_index)),
        )


class GraphNode(msgspec.Struct, rename="camel"):
    """DTO representing a graph node."""

    node_id: int
    label: str
    name: str
    properties: dict[str, Any] = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_id: int = 0) -> GraphNode:
        raw_props = data.get("properties")
        return cls(
            node_id=int(data.get("node_id") or data.get("id") or default_id),
            label=str(data.get("label", "Entity")),
            name=str(data.get("name") or f"node_{default_id}"),
            properties=dict(raw_props) if isinstance(raw_props, dict) else {},
        )


class GraphEdge(msgspec.Struct, rename="camel"):
    """DTO representing an edge in a graph."""

    source_id: int
    target_id: int
    relation_type: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphEdge:
        return cls(
            source_id=int(data.get("source_id") or data.get("source") or 0),
            target_id=int(data.get("target_id") or data.get("target") or 0),
            relation_type=str(data.get("relation_type") or data.get("type", "RELATION")),
        )


class SubgraphDTO(msgspec.Struct, rename="camel"):
    """DTO representing a subgraph."""

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    @classmethod
    def from_subgraph(cls, subgraph: object) -> SubgraphDTO:
        if not subgraph:
            return cls(nodes=[], edges=[])
        raw_nodes = (
            subgraph.get("nodes", [])
            if isinstance(subgraph, dict)
            else getattr(subgraph, "nodes", [])
        )
        raw_edges = (
            subgraph.get("edges", [])
            if isinstance(subgraph, dict)
            else getattr(subgraph, "edges", [])
        )
        nodes = [
            GraphNode.from_dict(n, default_id=i)
            if isinstance(n, dict)
            else GraphNode(node_id=i, label="Entity", name=str(n))
            for i, n in enumerate(raw_nodes)
        ]
        edges = [
            GraphEdge.from_dict(e)
            if isinstance(e, dict)
            else GraphEdge(source_id=0, target_id=0, relation_type="RELATION")
            for e in raw_edges
        ]
        return cls(nodes=nodes, edges=edges)


class StageTiming(msgspec.Struct, rename="camel"):
    """Timing information for a pipeline stage."""

    stage: str
    duration_ms: float


class QueryResponse(msgspec.Struct, rename="camel"):
    """Response DTO for queries."""

    answer: str
    route: str
    cached: bool
    latency_ms: float
    sources: list[SourceChunk] = []
    graph_path: SubgraphDTO | None = None
    timings: list[StageTiming] = []
    degraded: bool = False
    low_grounding: bool = False
    # Unverified Answer: a Jev check (routing or guardrail) could not run; never cached
    unverified: bool = False


# ── Server-Sent Event payloads for POST /api/v1/query/stream ─────────────────


class StreamStage(msgspec.Struct, rename="camel", omit_defaults=True):
    """`stage` event: a pipeline stage started (triage, cache_hit, chitchat, retrieval, generation)."""

    stage: str
    status: str | None = None
    route: str | None = None
    chunks_count: int | None = None


class StreamToken(msgspec.Struct, rename="camel"):
    """`token` event: the next piece of answer text."""

    delta: str


class StreamDone(msgspec.Struct, rename="camel"):
    """`done` event: the final answer with the same fields as QueryResponse."""

    answer: str
    route: str
    cached: bool
    latency_ms: float
    sources: list[SourceChunk] = []
    graph_path: SubgraphDTO | None = None
    timings: list[StageTiming] = []
    degraded: bool = False
    low_grounding: bool = False
    # Unverified Answer: a Jev check (routing or guardrail) could not run; never cached
    unverified: bool = False


class StreamError(msgspec.Struct, rename="camel"):
    """`error` event: generation stopped early; the stream still ends with `done`, or ends here."""

    detail: str


class IngestRequest(msgspec.Struct, rename="camel"):
    """Request DTO for document ingestion."""

    document_id: DocumentId
    title: Title
    text: DocumentText
    tags: Annotated[list[Tag], msgspec.Meta(max_length=32)] = []
    # Atomically replace an existing document with this id instead of returning 409
    replace: bool = False


class IngestResponse(msgspec.Struct, rename="camel"):
    """Response DTO for document ingestion."""

    document_id: str
    chunk_count: int
    entity_count: int
    relation_count: int
    persistence: str = RUNTIME_DOCUMENT_PERSISTENCE


class CacheStatsResponse(msgspec.Struct, rename="camel"):
    """Response DTO for cache statistics."""

    tier1_hits: int
    tier1_misses: int
    tier1_cached_queries: int
    tier2_circuit_status: str
    tier2_cached_queries: int


class GenerationProvider(StrEnum):
    """Who generated an answer: Groq, Gemini, the Tier 2 FAQ, or the offline message."""

    GROQ = "groq"
    GEMINI = "gemini"
    REDIS_FAQ = "redis_faq"
    OFFLINE = "offline"


class EvalQueryError(StrEnum):
    """Why a Gold Set query failed; a fixed code, never exception text."""

    PIPELINE_ERROR = "pipeline_error"
    JUDGE_ERROR = "judge_error"
    PROVIDER_DEGRADED = "provider_degraded"


class EvalQueryResult(msgspec.Struct, rename="camel"):
    """Result of one Gold Set query in the Live Tier.

    A metric is ``None`` when it was not judged: either it does not apply to the
    query's Slice, or the query failed (``error`` is set) and was never judged.
    """

    query: str
    passed: bool
    faithfulness: float | None = None
    context_precision: float | None = None
    answer_relevance: float | None = None
    answer_correctness: float | None = None
    error: EvalQueryError | None = None
    # The Route the pipeline chose; None when the pipeline failed before answering
    chosen_route: Route | None = None
    # False when chosen_route is the System 1 fallback because Jev could not route
    route_verified: bool = True

    # Provider that generated the answer; None when nothing was generated
    # (pipeline error, chitchat, cache hit)
    provider: GenerationProvider | None = None


class Misroute(msgspec.Struct, rename="camel"):
    """A Gold Set query whose chosen Route differs from its Expected Route."""

    query_id: str
    slice: Slice
    chosen_route: Route
    expected_route: Route


class EvalRunResponse(msgspec.Struct, rename="camel"):
    """Response DTO for an evaluation run.

    Means are over the queries where the metric was judged (``None`` if none were).
    Route Accuracy is the share of scored queries whose chosen Route equals the Expected
    Route (``None`` if none were scored). Scored means: outside the chitchat Slice (routed
    deterministically), the pipeline ran, and Jev chose the Route. Unverified fallback
    routes and pipeline errors are counted, not scored. Its floor is advisory only.
    """

    total_queries: int
    passed_gate: bool
    regression_delta: float
    mean_faithfulness: float | None = None
    mean_context_precision: float | None = None
    mean_answer_relevance: float | None = None
    mean_answer_correctness: float | None = None
    failed_queries: int = 0
    results: list[EvalQueryResult] = []
    route_accuracy: float | None = None
    route_accuracy_by_slice: dict[str, float] = {}
    route_accuracy_floor: float | None = None
    route_accuracy_scored: int = 0
    route_accuracy_unverified: int = 0
    route_accuracy_errored: int = 0
    misroutes: list[Misroute] = []


class EvalJobStatus(StrEnum):
    """Lifecycle of an Eval Job."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EvalJobResponse(msgspec.Struct, rename="camel"):
    """Handle and state of an Eval Job; ``report`` is set once it has succeeded."""

    job_id: str
    status: EvalJobStatus
    started_at: datetime
    finished_at: datetime | None = None
    report: EvalRunResponse | None = None
    error: str | None = None


class HealthResponse(msgspec.Struct, rename="camel"):
    """Response DTO for health checks."""

    status: str
    latticedb_connected: bool
    redis_connected: bool
    typesafe_configured: bool
    groq_configured: bool
    gemini_configured: bool


class StageBenchmarkDTO(msgspec.Struct, rename="camel"):
    """DTO representing latency metrics for a benchmarked stage."""

    stage: str
    samples_count: int
    mean_ms: float
    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    qps: float


class BenchmarkRunResponse(msgspec.Struct, rename="camel"):
    """Response DTO for an end-to-end performance benchmark run."""

    total_duration_sec: float
    sub_second_sla_met: bool
    tier1_cache_sla_met: bool
    stages: list[StageBenchmarkDTO] = []
