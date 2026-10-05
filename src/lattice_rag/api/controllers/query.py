from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncGenerator

import msgspec
from litestar import Controller, post
from litestar.di import NamedDependency
from litestar.params import JSONBody, SkipValidation
from litestar.response import ServerSentEvent, ServerSentEventMessage

from lattice_rag.api.dtos import (
    GraphEdge,
    GraphNode,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StageTiming,
    SubgraphDTO,
)
from lattice_rag.orchestration.graph import RAGOrchestrator

logger = logging.getLogger(__name__)


class QueryController(Controller):
    """Litestar controller for synchronous and streaming query endpoints."""

    path = "/api/v1"

    @post("/query")
    async def query(
        self,
        data: JSONBody[QueryRequest],
        orchestrator: NamedDependency[SkipValidation[RAGOrchestrator]],
    ) -> QueryResponse:
        """Run the full GraphRAG pipeline and return a structured QueryResponse."""
        logger.info("received_query_request", extra={"query": data.query})

        state = await orchestrator.run(data.query, stream=False)

        # Build SourceChunk DTOs
        sources: list[SourceChunk] = []
        for i, c in enumerate(state.filtered_chunks):
            sources.append(
                SourceChunk(
                    chunk_id=c.get("node_id", c.get("chunk_id", i)),
                    text=c.get("text", ""),
                    score=float(c.get("rerank_score", c.get("score", 0.0))),
                    document_id=str(c.get("doc_id", c.get("document_id", "unknown"))),
                    position=int(c.get("position", i)),
                )
            )

        # Build SubgraphDTO if graph context exists
        graph_path: SubgraphDTO | None = None
        if state.graph_context:
            nodes = [
                GraphNode(
                    node_id=n.get("node_id", n.get("id", i)),
                    label=n.get("label", "Entity"),
                    name=n.get("name", ""),
                    properties=n.get("properties", {}),
                )
                for i, n in enumerate(state.graph_context.get("nodes", []))
            ]
            edges = [
                GraphEdge(
                    source_id=e.get("source_id", 0),
                    target_id=e.get("target_id", 0),
                    relation_type=e.get("relation_type", "RELATION"),
                )
                for e in state.graph_context.get("edges", [])
            ]
            graph_path = SubgraphDTO(nodes=nodes, edges=edges)

        # Build StageTiming DTOs
        timings = [
            StageTiming(stage=t.get("stage", "stage"), duration_ms=float(t.get("duration_ms", 0.0)))
            for t in state.timings
        ]

        return QueryResponse(
            answer=state.answer,
            route=state.route,
            cached=state.cache_hit,
            latency_ms=state.total_latency_ms,
            sources=sources,
            graph_path=graph_path,
            timings=timings,
            degraded=state.degraded,
        )

    @post("/query/stream")
    async def query_stream(
        self,
        data: JSONBody[QueryRequest],
        orchestrator: NamedDependency[SkipValidation[RAGOrchestrator]],
    ) -> ServerSentEvent:
        """Run the GraphRAG pipeline and stream typed SSE events."""
        logger.info("received_streaming_query_request", extra={"query": data.query})

        async def sse_generator() -> AsyncGenerator[ServerSentEventMessage, None]:
            async for event in orchestrator.stream_query(data.query):
                payload = msgspec.json.encode(event["data"]).decode("utf-8")
                yield ServerSentEventMessage(
                    event=event["event"],
                    data=payload,
                )

        return ServerSentEvent(sse_generator())
