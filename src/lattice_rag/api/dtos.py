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

    @classmethod
    def from_dict(cls, data: dict[str, object], default_index: int = 0) -> SourceChunk:
        return cls(
            chunk_id=int(data.get("node_id") or data.get("chunk_id") or default_index),
            text=str(data.get("text", "")),
            score=float(data.get("rerank_score") or data.get("score") or 0.0),
            document_id=str(data.get("doc_id") or data.get("document_id") or "unknown"),
            position=int(data.get("position", default_index)),
        )

class GraphNode(msgspec.Struct, rename='camel'):
    """DTO representing a graph node."""
    node_id: int
    label: str
    name: str
    properties: dict[str, object] = {}

    @classmethod
    def from_dict(cls, data: dict[str, object], default_id: int = 0) -> GraphNode:
        return cls(
            node_id=int(data.get("node_id") or data.get("id") or default_id),
            label=str(data.get("label", "Entity")),
            name=str(data.get("name") or f"node_{default_id}"),
            properties=dict(data.get("properties", {})) if isinstance(data.get("properties"), dict) else {},
        )

class GraphEdge(msgspec.Struct, rename='camel'):
    """DTO representing an edge in a graph."""
    source_id: int
    target_id: int
    relation_type: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GraphEdge:
        return cls(
            source_id=int(data.get("source_id") or data.get("source") or 0),
            target_id=int(data.get("target_id") or data.get("target") or 0),
            relation_type=str(data.get("relation_type") or data.get("type", "RELATION")),
        )

class SubgraphDTO(msgspec.Struct, rename='camel'):
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
            GraphNode.from_dict(n, default_id=i) if isinstance(n, dict) else GraphNode(node_id=i, label="Entity", name=str(n))
            for i, n in enumerate(raw_nodes)
        ]
        edges = [
            GraphEdge.from_dict(e) if isinstance(e, dict) else GraphEdge(source_id=0, target_id=0, relation_type="RELATION")
            for e in raw_edges
        ]
        return cls(nodes=nodes, edges=edges)


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

