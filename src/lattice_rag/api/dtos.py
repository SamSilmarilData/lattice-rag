from __future__ import annotations

import msgspec

class QueryRequest(msgspec.Struct, rename='camel'):
    """Request DTO for queries."""
    query: str
    stream: bool = False
    filter_tags: list[str] = []

class SourceChunk(msgspec.Struct, rename='camel'):
    """DTO representing a chunk of text from a source document."""
    chunk_id: int
    text: str
    score: float
    document_id: str
    position: int

class GraphNode(msgspec.Struct, rename='camel'):
    """DTO representing a graph node."""
    node_id: int
    label: str
    name: str
    properties: dict[str, object] = {}

class GraphEdge(msgspec.Struct, rename='camel'):
    """DTO representing an edge in a graph."""
    source_id: int
    target_id: int
    relation_type: str

class SubgraphDTO(msgspec.Struct, rename='camel'):
    """DTO representing a subgraph."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

class StageTiming(msgspec.Struct, rename='camel'):
    """Timing information for a pipeline stage."""
    stage: str
    duration_ms: float

class QueryResponse(msgspec.Struct, rename='camel'):
    """Response DTO for queries."""
    answer: str
    route: str
    cached: bool
    latency_ms: float
    sources: list[SourceChunk] = []
    graph_path: SubgraphDTO | None = None
    timings: list[StageTiming] = []
    degraded: bool = False

class IngestRequest(msgspec.Struct, rename='camel'):
    """Request DTO for document ingestion."""
    document_id: str
    title: str
    text: str
    tags: list[str] = []

class IngestResponse(msgspec.Struct, rename='camel'):
    """Response DTO for document ingestion."""
    document_id: str
    chunk_count: int
    entity_count: int
    relation_count: int

class CacheStatsResponse(msgspec.Struct, rename='camel'):
    """Response DTO for cache statistics."""
    tier1_hits: int
    tier1_misses: int
    tier1_cached_queries: int
    tier2_circuit_status: str
    tier2_cached_queries: int

class EvalQueryResult(msgspec.Struct, rename='camel'):
    """Result of an evaluation query."""
    query: str
    faithfulness: float
    context_precision: float
    answer_relevance: float
    passed: bool

class EvalRunResponse(msgspec.Struct, rename='camel'):
    """Response DTO for an evaluation run."""
    total_queries: int
    mean_faithfulness: float
    mean_context_precision: float
    mean_answer_relevance: float
    passed_gate: bool
    regression_delta: float
    results: list[EvalQueryResult] = []

class HealthResponse(msgspec.Struct, rename='camel'):
    """Response DTO for health checks."""
    status: str
    latticedb_connected: bool
    redis_connected: bool
    typesafe_configured: bool
    groq_configured: bool
    gemini_configured: bool

class StageBenchmarkDTO(msgspec.Struct, rename='camel'):
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

class BenchmarkRunResponse(msgspec.Struct, rename='camel'):
    """Response DTO for an end-to-end performance benchmark run."""
    total_duration_sec: float
    sub_second_sla_met: bool
    tier1_cache_sla_met: bool
    stages: list[StageBenchmarkDTO] = []

