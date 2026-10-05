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
