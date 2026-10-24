"""Unit tests for QueryRouter with TypeSafe AI Jev Choice routing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lattice_rag.routing.router import QueryRouter, RouteDecision


@pytest.mark.asyncio
async def test_router_missing_key_raises_runtime_error(monkeypatch: pytest.MonkeyPatch):
    """Verify QueryRouter raises RuntimeError if TYPESAFE_API_KEY is missing."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY environment variable is not set"):
        QueryRouter()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route_name,confidence",
    [
        ("vector_exact", 0.96),
        ("graph_relational", 0.91),
        ("hybrid", 0.88),
        ("chitchat", 0.99),
        ("massive_context", 0.85),
    ],
)
async def test_router_all_five_routes(route_name: str, confidence: float):
    """Verify QueryRouter correctly parses all 5 supported routing categories."""
    mock_client = AsyncMock()
    mock_response = SimpleNamespace(
        choices={"route": SimpleNamespace(choice=route_name, confidence=confidence)}
    )
    mock_client.system_one.return_value = mock_response

    router = QueryRouter(ts_client=mock_client)
    decision = await router.route_query("Test query for routing")

    assert isinstance(decision, RouteDecision)
    assert decision.route == route_name
    assert decision.confidence == confidence
    mock_client.system_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_router_aclose_and_close():
    """Verify router cleanly invokes client.aclose() on shutdown."""
    mock_client = AsyncMock()
    router = QueryRouter(ts_client=mock_client)

    await router.aclose()
    mock_client.aclose.assert_awaited_once()

    # close() is an alias for aclose()
    await router.close()
    assert mock_client.aclose.await_count == 2


@pytest.mark.asyncio
async def test_router_system_one_error_falls_back_to_hybrid():
    """Verify router catches system_one exceptions and gracefully returns hybrid decision."""
    mock_client = AsyncMock()
    mock_client.system_one.side_effect = RuntimeError(
        "403 Forbidden: Request not allowed by policy"
    )

    router = QueryRouter(ts_client=mock_client)
    decision = await router.route_query("Explain vector indexing")

    assert isinstance(decision, RouteDecision)
    assert decision.route == "hybrid"
    assert decision.confidence == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "How does HNSW help recall?",
        "Can you help me configure Redis eviction?",
        "Thanks to the WAL, are writes durable?",
        "Hi, what is LatticeDB's vector index?",
        "Which tools assist entity extraction?",
        "Summarize the morning standup notes",
    ],
)
async def test_router_real_questions_with_chitchat_words_reach_jev(query: str):
    """Questions that merely contain a chitchat keyword must not be short-circuited."""
    mock_client = AsyncMock()
    mock_client.system_one.return_value = SimpleNamespace(
        choices={"route": SimpleNamespace(choice="hybrid", confidence=0.9)}
    )
    router = QueryRouter(ts_client=mock_client)

    decision = await router.route_query(query)

    assert decision.route == "hybrid"
    mock_client.system_one.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["hi!", "Hello there", "thanks a lot", "bye for now", "what can you do?", "Good morning!"],
)
async def test_router_pure_chitchat_uses_fast_path(query: str):
    mock_client = AsyncMock()
    router = QueryRouter(ts_client=mock_client)

    decision = await router.route_query(query)

    assert decision.route == "chitchat"
    assert decision.verified is True
    mock_client.system_one.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "jev_outcome",
    [
        RuntimeError("503 Service Unavailable"),
        SimpleNamespace(choices={}),
        SimpleNamespace(choices={"route": SimpleNamespace(choice="poetry", confidence=0.9)}),
    ],
    ids=["jev_error", "missing_choice", "unknown_route"],
)
async def test_router_fallbacks_are_unverified(jev_outcome):
    """Every System 1 fallback to hybrid is reported as unverified."""
    mock_client = AsyncMock()
    if isinstance(jev_outcome, Exception):
        mock_client.system_one.side_effect = jev_outcome
    else:
        mock_client.system_one.return_value = jev_outcome
    router = QueryRouter(ts_client=mock_client)

    decision = await router.route_query("Explain vector indexing")

    assert decision.route == "hybrid"
    assert decision.verified is False


@pytest.mark.asyncio
async def test_router_successful_jev_route_is_verified():
    mock_client = AsyncMock()
    mock_client.system_one.return_value = SimpleNamespace(
        choices={"route": SimpleNamespace(choice="graph_relational", confidence=0.9)}
    )
    router = QueryRouter(ts_client=mock_client)

    decision = await router.route_query("How does X relate to Y?")

    assert decision.verified is True


@pytest.mark.asyncio
async def test_offline_router_makes_no_jev_calls(monkeypatch: pytest.MonkeyPatch):
    """Mock-key mode: the router never contacts Jev and reports its fallback as unverified."""
    from typesafe_sdk import AsyncTypeSafeClient

    calls = AsyncMock()
    monkeypatch.setattr(AsyncTypeSafeClient, "system_one", calls)
    router = QueryRouter.offline()

    decision = await router.route_query("Explain vector indexing")
    greeting = await router.route_query("hello there")
    await router.aclose()

    assert (decision.route, decision.verified) == ("hybrid", False)
    assert (greeting.route, greeting.verified) == ("chitchat", True)
    calls.assert_not_awaited()
