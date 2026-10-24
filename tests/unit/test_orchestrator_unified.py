"""Unit tests verifying unified behavior between run() and stream_query() in RAGOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

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

    mock_pipeline = MagicMock()
    mock_pipeline.execute = AsyncMock(
        return_value=MagicMock(
            final_chunks=[{"node_id": 1, "text": "LatticeDB is fast", "score": 0.9}],
            graph_context=SubgraphResult(nodes=[{"id": 1, "name": "LatticeDB"}], edges=[]),
            timings={"stage1_search": 1.0},
            low_grounding=False,
            unverified=False,
        )
    )

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
    c["router"].route_query = AsyncMock(
        return_value=RouteDecision(route="chitchat", confidence=0.99)
    )

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


@pytest.mark.asyncio
async def test_unified_orchestrator_degraded_parity_and_no_caching(mock_orchestrator_components):
    """When Groq is down, both paths report degraded=True and neither caches the answer."""
    c = mock_orchestrator_components
    c["groq"].synthesize = AsyncMock(side_effect=RuntimeError("Groq 503"))

    async def _groq_stream_fail(q, ch, g):
        raise RuntimeError("Groq 503")
        yield "never"  # pragma: no cover

    async def _gemini_stream(q, ch, g):
        yield "Gemini answer."

    c["groq"].stream = _groq_stream_fail
    c["gemini"].stream = _gemini_stream

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
    stream_events = [evt async for evt in orchestrator.stream_query("Explain LatticeDB")]
    done_evt = [e for e in stream_events if e["event"] == "done"][0]

    assert state.degraded is True
    assert done_evt["data"]["degraded"] is True
    assert done_evt["data"]["answer"] == state.answer == "Gemini answer."
    c["semantic_cache"].put.assert_not_called()


@pytest.mark.asyncio
async def test_unified_orchestrator_low_grounding_is_reported_and_not_cached(
    mock_orchestrator_components,
):
    """Answers built on evidence the guardrail rejected are flagged and never cached."""
    c = mock_orchestrator_components
    c["pipeline"].execute = AsyncMock(
        return_value=MagicMock(
            final_chunks=[],
            graph_context=SubgraphResult(nodes=[], edges=[]),
            timings={},
            low_grounding=True,
            unverified=False,
        )
    )

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

    state = await orchestrator.run("Unanswerable question")
    stream_events = [evt async for evt in orchestrator.stream_query("Unanswerable question")]
    done_evt = [e for e in stream_events if e["event"] == "done"][0]

    assert state.low_grounding is True
    assert done_evt["data"]["low_grounding"] is True
    c["semantic_cache"].put.assert_not_called()


@pytest.mark.asyncio
async def test_massive_context_retrieves_more_evidence_in_both_paths(mock_orchestrator_components):
    from lattice_rag.orchestration.graph import MASSIVE_CONTEXT_TOP_K

    c = mock_orchestrator_components
    c["router"].route_query = AsyncMock(
        return_value=RouteDecision(route="massive_context", confidence=0.9)
    )
    c["gemini"].synthesize = AsyncMock(return_value="Summary.")

    async def _gemini_stream(q, ch, g):
        yield "Summary."

    c["gemini"].stream = _gemini_stream
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

    await orchestrator.run("Summarize everything about caching")
    _ = [e async for e in orchestrator.stream_query("Summarize everything about caching")]

    for call in c["pipeline"].execute.await_args_list:
        assert call.kwargs["top_k"] == MASSIVE_CONTEXT_TOP_K


def _orchestrator(c, semantic_cache=None):
    return RAGOrchestrator(
        router=c["router"],
        semantic_cache=semantic_cache if semantic_cache is not None else c["semantic_cache"],
        fallback_cache=c["fallback_cache"],
        retrieval_pipeline=c["pipeline"],
        groq=c["groq"],
        gemini=c["gemini"],
        chitchat=c["chitchat"],
        embedding_service=c["embed_svc"],
    )


async def _ask(orchestrator: RAGOrchestrator, entry_point: str, query: str) -> dict:
    """Run one query through either entry point and return the flags the caller sees."""
    if entry_point == "run":
        state = await orchestrator.run(query)
        return {"unverified": state.unverified, "cached": state.cache_hit}
    events = [e async for e in orchestrator.stream_query(query)]
    done = events[-1]
    assert done["event"] == "done"
    return {"unverified": done["data"]["unverified"], "cached": done["data"]["cached"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["run", "stream_query"])
async def test_guardrail_unavailable_answer_is_unverified_and_recomputed(
    mock_orchestrator_components, entry_point
):
    """An answer whose guardrail could not run is flagged and never served from Tier 1."""
    from lattice_rag.caching.semantic_cache import SemanticCache
    from lattice_rag.retrieval.pipeline import RetrievalResult

    c = mock_orchestrator_components
    c["pipeline"].execute = AsyncMock(
        return_value=RetrievalResult(
            final_chunks=[{"node_id": 1, "text": "LatticeDB is fast", "score": 0.9}],
            graph_context=SubgraphResult(nodes=[], edges=[]),
            unverified=True,
        )
    )
    orchestrator = _orchestrator(c, semantic_cache=SemanticCache(ts_client=None))

    first = await _ask(orchestrator, entry_point, "Explain LatticeDB")
    second = await _ask(orchestrator, entry_point, "Explain LatticeDB")

    assert first == {"unverified": True, "cached": False}
    assert second == {"unverified": True, "cached": False}
    assert c["pipeline"].execute.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["run", "stream_query"])
async def test_routing_fallback_answer_is_unverified_and_recomputed(
    mock_orchestrator_components, entry_point
):
    """When Jev routing could not run, the answer is flagged and never served from Tier 1."""
    from lattice_rag.caching.semantic_cache import SemanticCache

    c = mock_orchestrator_components
    c["router"].route_query = AsyncMock(
        return_value=RouteDecision(route="hybrid", confidence=0.5, verified=False)
    )
    orchestrator = _orchestrator(c, semantic_cache=SemanticCache(ts_client=None))

    first = await _ask(orchestrator, entry_point, "Explain LatticeDB")
    second = await _ask(orchestrator, entry_point, "Explain LatticeDB")

    assert first == {"unverified": True, "cached": False}
    assert second == {"unverified": True, "cached": False}
    assert c["pipeline"].execute.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["run", "stream_query"])
async def test_verified_answer_is_cached_and_served_on_repeat(
    mock_orchestrator_components, entry_point
):
    """Control case: a fully verified answer is stored and the repeat is a cache hit."""
    from lattice_rag.caching.semantic_cache import SemanticCache

    c = mock_orchestrator_components
    orchestrator = _orchestrator(c, semantic_cache=SemanticCache(ts_client=None))

    first = await _ask(orchestrator, entry_point, "Explain LatticeDB")
    second = await _ask(orchestrator, entry_point, "Explain LatticeDB")

    assert first == {"unverified": False, "cached": False}
    assert second == {"unverified": False, "cached": True}
    assert c["pipeline"].execute.await_count == 1
