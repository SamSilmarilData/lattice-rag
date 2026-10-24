from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import msgspec
from litestar import Controller, post
from litestar.di import NamedDependency
from litestar.params import JSONBody, SkipValidation
from litestar.response import ServerSentEvent, ServerSentEventMessage

from lattice_rag.api.dtos import (
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StageTiming,
    StreamDone,
    StreamError,
    StreamStage,
    StreamToken,
    SubgraphDTO,
)
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.security import sanitize_error_detail

logger = logging.getLogger(__name__)


def _sources(chunks: list[dict[str, Any]]) -> list[SourceChunk]:
    return [SourceChunk.from_dict(c, default_index=i) for i, c in enumerate(chunks)]


def _timings(raw: list[dict[str, Any]]) -> list[StageTiming]:
    return [
        StageTiming(stage=t.get("stage", "stage"), duration_ms=float(t.get("duration_ms", 0.0)))
        for t in raw
    ]


def stream_event_payload(event: dict[str, Any]) -> msgspec.Struct:
    """Convert an orchestrator stream event into its camelCase wire DTO."""
    kind, data = event["event"], event["data"]
    if kind == "token":
        return StreamToken(delta=data["delta"])
    if kind == "error":
        return StreamError(detail=data["detail"])
    if kind == "done":
        graph = data.get("graph_path")
        return StreamDone(
            answer=data["answer"],
            route=data["route"],
            cached=data["cached"],
            latency_ms=data["latency_ms"],
            sources=_sources(data.get("sources", [])),
            graph_path=SubgraphDTO.from_subgraph(graph) if graph else None,
            timings=_timings(data.get("timings", [])),
            degraded=data.get("degraded", False),
            low_grounding=data.get("low_grounding", False),
            unverified=data.get("unverified", False),
        )
    return StreamStage(
        stage=data["stage"],
        status=data.get("status"),
        route=data.get("route"),
        chunks_count=data.get("chunks_count"),
    )


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
        logger.debug("received_query_request")

        state = await orchestrator.run(data.query)
        graph_path = SubgraphDTO.from_subgraph(state.graph_context) if state.graph_context else None

        return QueryResponse(
            answer=state.answer,
            route=state.route,
            cached=state.cache_hit,
            latency_ms=state.total_latency_ms,
            sources=_sources(state.filtered_chunks),
            graph_path=graph_path,
            timings=_timings(state.timings),
            degraded=state.degraded,
            low_grounding=state.low_grounding,
            unverified=state.unverified,
        )

    @post("/query/stream")
    async def query_stream(
        self,
        data: JSONBody[QueryRequest],
        orchestrator: NamedDependency[SkipValidation[RAGOrchestrator]],
    ) -> ServerSentEvent:
        """Run the GraphRAG pipeline and stream typed SSE events (stage, token, error, done)."""
        logger.debug("received_streaming_query_request")

        async def sse_generator() -> AsyncGenerator[ServerSentEventMessage, None]:
            try:
                async for event in orchestrator.stream_query(data.query):
                    payload = msgspec.json.encode(stream_event_payload(event)).decode("utf-8")
                    yield ServerSentEventMessage(event=event["event"], data=payload)
            except Exception as e:
                # Headers are already sent, so report failures in-band instead of
                # silently dropping the connection
                request_id = uuid.uuid4().hex
                logger.error(
                    "stream_query_failed request_id=%s error=%s",
                    request_id,
                    sanitize_error_detail(f"{type(e).__name__}: {e}"),
                )
                error = StreamError(detail=f"The query failed. Reference: {request_id}")
                yield ServerSentEventMessage(
                    event="error", data=msgspec.json.encode(error).decode("utf-8")
                )

        return ServerSentEvent(sse_generator())
