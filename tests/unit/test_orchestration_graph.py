"""Unit tests for LangGraph RAGOrchestrator and failover cascades."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.orchestration.state import PipelineState
from lattice_rag.routing.router import RouteDecision


@pytest.fixture
def mock_dependencies():
    router = MagicMock()
    router.route_query = AsyncMock(return_value=RouteDecision(route="hybrid", confidence=0.95))

    semantic_cache = MagicMock()
    semantic_cache.get = AsyncMock(return_value=None)
    semantic_cache.put = MagicMock()

    fallback_cache = MagicMock()
    fallback_cache.circuit_status = "closed"
    fallback_cache.call_with_breaker = AsyncMock()
    fallback_cache.find_closest_fallback = AsyncMock(return_value="Cached FAQ fallback answer.")

    retrieval_pipeline = MagicMock()
    mock_ret_result = MagicMock()
    mock_ret_result.final_chunks = [{"doc_id": "doc1", "text": "Relevant chunk text"}]
    mock_ret_result.graph_context = MagicMock()
    mock_ret_result.graph_context.nodes = [{"node_id": 1, "name": "NodeA"}]
    mock_ret_result.graph_context.edges = []
    retrieval_pipeline.execute = AsyncMock(return_value=mock_ret_result)

    groq = MagicMock()
    groq.synthesize = AsyncMock(return_value="Groq synthesized answer.")

    gemini = MagicMock()
    gemini.synthesize = AsyncMock(return_value="Gemini fallback answer.")

    chitchat = MagicMock()
    chitchat.handle = AsyncMock(return_value="Hello! I am lattice-rag.")

    embedding_service = MagicMock()
    embedding_service.embed_query = MagicMock(return_value=np.zeros(384, dtype=np.float32))

    return {
        "router": router,
        "semantic_cache": semantic_cache,
        "fallback_cache": fallback_cache,
        "retrieval_pipeline": retrieval_pipeline,
        "groq": groq,
        "gemini": gemini,
        "chitchat": chitchat,
        "embedding_service": embedding_service,
    }


@pytest.mark.asyncio
async def test_orchestrator_cache_hit_short_circuit(mock_dependencies):
    deps = mock_dependencies
    deps["semantic_cache"].get.return_value = {
        "answer": "Instant cached answer.",
        "route": "hybrid",
        "sources": [{"doc_id": "doc_cached"}],
    }

    orchestrator = RAGOrchestrator(**deps)
    result = await orchestrator.run("What is LatticeDB?")

    assert result.cache_hit is True
    assert result.answer == "Instant cached answer."
    deps["retrieval_pipeline"].execute.assert_not_called()
    deps["groq"].synthesize.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_chitchat_route(mock_dependencies):
    deps = mock_dependencies
    deps["router"].route_query.return_value = RouteDecision(route="chitchat", confidence=1.0)

    orchestrator = RAGOrchestrator(**deps)
    result = await orchestrator.run("hello there")

    assert result.route == "chitchat"
    assert result.answer == "Hello! I am lattice-rag."
    deps["chitchat"].handle.assert_awaited_once_with("hello there")
    deps["retrieval_pipeline"].execute.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_standard_groq_generation(mock_dependencies):
    deps = mock_dependencies
    deps["fallback_cache"].call_with_breaker.return_value = "Groq synthesized answer."

    orchestrator = RAGOrchestrator(**deps)
    result = await orchestrator.run("Explain LatticeDB HNSW search")

    assert result.route == "hybrid"
    assert result.answer == "Groq synthesized answer."
    assert result.degraded is False
    deps["retrieval_pipeline"].execute.assert_awaited_once()
    deps["fallback_cache"].call_with_breaker.assert_awaited_once()
    deps["semantic_cache"].put.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_failover_to_gemini(mock_dependencies):
    deps = mock_dependencies
    # Groq fails under breaker
    deps["fallback_cache"].call_with_breaker.side_effect = RuntimeError("Groq 429 Rate Limit")
    deps["gemini"].synthesize.return_value = "Gemini fallback answer."

    orchestrator = RAGOrchestrator(**deps)
    result = await orchestrator.run("Explain LatticeDB")

    assert result.answer == "Gemini fallback answer."
    assert result.degraded is True
    deps["gemini"].synthesize.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_double_failover_to_redis_faq(mock_dependencies):
    deps = mock_dependencies
    # Groq and Gemini both fail
    deps["fallback_cache"].call_with_breaker.side_effect = RuntimeError("Groq Outage")
    deps["gemini"].synthesize.side_effect = RuntimeError("Gemini Outage")
    deps["fallback_cache"].find_closest_fallback.return_value = "Redis FAQ Answer"

    orchestrator = RAGOrchestrator(**deps)
    result = await orchestrator.run("What is LatticeDB?")

    assert result.answer == "Redis FAQ Answer"
    assert result.degraded is True
    deps["fallback_cache"].find_closest_fallback.assert_awaited_once_with("What is LatticeDB?")


@pytest.mark.asyncio
async def test_orchestrator_stream_query_events(mock_dependencies):
    deps = mock_dependencies

    async def mock_groq_stream(*args, **kwargs):
        for tok in ["Streaming ", "token ", "response."]:
            yield tok

    deps["groq"].stream = mock_groq_stream

    orchestrator = RAGOrchestrator(**deps)
    events = []
    async for event in orchestrator.stream_query("Stream test"):
        events.append(event)

    event_types = [e["event"] for e in events]
    assert "stage" in event_types
    assert "token" in event_types
    assert "done" in event_types

    done_event = [e for e in events if e["event"] == "done"][0]
    assert done_event["data"]["answer"] == "Streaming token response."
    assert done_event["data"]["route"] == "hybrid"
