"""Unit tests verifying unified behavior between run() and stream_query() in RAGOrchestrator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

from lattice_rag.generation.resilient_synthesizer import ResilientSynthesizer
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.routing.router import RouteDecision
from lattice_rag.storage.db import SubgraphResult


@pytest.fixture
def mock_orchestrator_components():
    mock_router = MagicMock()
    mock_router.route_query = AsyncMock(return_value=RouteDecision(route="hybrid", confidence=0.9))

    mock_sem_cache = MagicMock()
    mock_sem_cache.get = AsyncMock(return_value=None)
    mock_sem_cache.put = MagicMock()

    mock_fb_cache = MagicMock()
    mock_fb_cache.find_closest_fallback = AsyncMock(return_value="Cached FAQ")
    mock_fb_cache.circuit_status = "closed"
    async def _call_breaker(fn, *args, **kwargs):
        res = fn(*args, **kwargs)
        if hasattr(res, "__await__"):
            return await res
        return res
    mock_fb_cache.call_with_breaker = AsyncMock(side_effect=_call_breaker)

    mock_pipeline = MagicMock()
    mock_pipeline.execute = AsyncMock(return_value=MagicMock(
        final_chunks=[{"node_id": 1, "text": "LatticeDB is fast", "score": 0.9}],
        graph_context=SubgraphResult(nodes=[{"id": 1, "name": "LatticeDB"}], edges=[]),
        timings={"stage1_search": 1.0},
        low_grounding=False,
    ))

    mock_groq = MagicMock()
    mock_groq.synthesize = AsyncMock(return_value="Grounded answer from Groq.")
    async def _groq_stream(q, c, g):
        for tok in ["Grounded", " ", "answer", " ", "from", " ", "Groq."]:
            yield tok
    mock_groq.stream = _groq_stream

    mock_gemini = MagicMock()
    mock_gemini.synthesize = AsyncMock(return_value="Gemini answer.")

    mock_chitchat = MagicMock()
    mock_chitchat.handle = AsyncMock(return_value="Hello! I am lattice-rag.")

    mock_embed_svc = MagicMock()
    mock_embed_svc.embed_query = MagicMock(return_value=np.ones(384, dtype=np.float32))

    return {
        "router": mock_router,
        "semantic_cache": mock_sem_cache,
        "fallback_cache": mock_fb_cache,
        "pipeline": mock_pipeline,
        "groq": mock_groq,
        "gemini": mock_gemini,
        "chitchat": mock_chitchat,
        "embed_svc": mock_embed_svc,
    }


@pytest.mark.asyncio
async def test_unified_orchestrator_chitchat_parity(mock_orchestrator_components):
    c = mock_orchestrator_components
    c["router"].route_query = AsyncMock(return_value=RouteDecision(route="chitchat", confidence=0.99))

    orchestrator = RAGOrchestrator(
        router=c["router"],
        semantic_cache=c["semantic_cache"],
        fallback_cache=c["fallback_cache"],
        retrieval_pipeline=c["pipeline"],
        groq=c["groq"],
        gemini=c["gemini"],
        chitchat=c["chitchat"],
        embedding_service=c["embed_svc"],
    )

    # 1. Run via batch
    state = await orchestrator.run("hello")
    assert state.answer == "Hello! I am lattice-rag."
    assert state.route == "chitchat"
    assert state.cache_hit is False

    # 2. Run via stream
    stream_events = [evt async for evt in orchestrator.stream_query("hello")]
    done_evt = [e for e in stream_events if e["event"] == "done"][0]
    token_str = "".join(e["data"]["delta"] for e in stream_events if e["event"] == "token")

    assert token_str == state.answer
    assert done_evt["data"]["route"] == state.route
    assert done_evt["data"]["answer"] == state.answer


@pytest.mark.asyncio
async def test_unified_orchestrator_cache_hit_parity(mock_orchestrator_components):
    c = mock_orchestrator_components
    cached_payload = {
        "answer": "Instant cached answer.",
        "route": "hybrid",
        "sources": [{"node_id": 1, "text": "cached source"}],
        "graph_path": {"nodes": [], "edges": []},
    }
    c["semantic_cache"].get = AsyncMock(return_value=cached_payload)

    orchestrator = RAGOrchestrator(
        router=c["router"],
        semantic_cache=c["semantic_cache"],
        fallback_cache=c["fallback_cache"],
        retrieval_pipeline=c["pipeline"],
        groq=c["groq"],
        gemini=c["gemini"],
        chitchat=c["chitchat"],
        embedding_service=c["embed_svc"],
    )

    state = await orchestrator.run("cached query")
    assert state.answer == "Instant cached answer."
    assert state.cache_hit is True

    stream_events = [evt async for evt in orchestrator.stream_query("cached query")]
    done_evt = [e for e in stream_events if e["event"] == "done"][0]
    assert done_evt["data"]["cached"] is True
    assert done_evt["data"]["answer"] == state.answer


@pytest.mark.asyncio
async def test_unified_orchestrator_standard_query_parity(mock_orchestrator_components):
    c = mock_orchestrator_components

    orchestrator = RAGOrchestrator(
        router=c["router"],
        semantic_cache=c["semantic_cache"],
        fallback_cache=c["fallback_cache"],
        retrieval_pipeline=c["pipeline"],
        groq=c["groq"],
        gemini=c["gemini"],
        chitchat=c["chitchat"],
        embedding_service=c["embed_svc"],
    )

    state = await orchestrator.run("Explain LatticeDB")
    assert state.answer == "Grounded answer from Groq."
    assert state.route == "hybrid"
    assert len(state.filtered_chunks) == 1

    stream_events = [evt async for evt in orchestrator.stream_query("Explain LatticeDB")]
    done_evt = [e for e in stream_events if e["event"] == "done"][0]
    tokens = "".join(e["data"]["delta"] for e in stream_events if e["event"] == "token")

    assert tokens == state.answer
    assert done_evt["data"]["answer"] == state.answer
    assert done_evt["data"]["route"] == state.route
