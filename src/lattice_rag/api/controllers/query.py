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
        sources = [
            SourceChunk.from_dict(c, default_index=i)
            for i, c in enumerate(state.filtered_chunks)
        ]

        # Build SubgraphDTO if graph context exists
        graph_path = (
            SubgraphDTO.from_subgraph(state.graph_context)
            if state.graph_context
            else None
        )

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
