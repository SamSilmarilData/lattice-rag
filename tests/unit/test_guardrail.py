"""Unit tests for ContextGuardrail using TypeSafe AI Jev Noul."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lattice_rag.routing.guardrail import ContextGuardrail, GuardrailResult


@pytest.mark.asyncio
async def test_guardrail_missing_key_raises_runtime_error(monkeypatch: pytest.MonkeyPatch):
    """Verify ContextGuardrail raises RuntimeError if TYPESAFE_API_KEY is not set."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY environment variable is not set"):
        ContextGuardrail()


@pytest.mark.asyncio
async def test_guardrail_empty_chunks():
    """Verify empty input returns empty list without calling system_one."""
    mock_client = AsyncMock()
    guardrail = ContextGuardrail(ts_client=mock_client)

    result = await guardrail.filter_chunks("What is LatticeDB?", [])
    assert result == GuardrailResult(chunks=[], low_grounding=False)
    mock_client.system_one.assert_not_called()


@pytest.mark.asyncio
async def test_guardrail_filtering_above_and_below_threshold():
    """Verify batched Noul filtering prunes chunks below threshold=0.50 and preserves those above."""
    mock_client = AsyncMock()
    # 3 chunks: chunk 0 passes (0.85), chunk 1 fails (0.30), chunk 2 passes (0.60)
    mock_response = SimpleNamespace(
        nouls={
            "relevant_0": SimpleNamespace(noul=0.85),
            "relevant_1": SimpleNamespace(noul=0.30),
            "relevant_2": SimpleNamespace(noul=0.60),
        }
    )
    mock_client.system_one.return_value = mock_response

    guardrail = ContextGuardrail(threshold=0.50, ts_client=mock_client)
    chunks = [
        {"node_id": 1, "text": "LatticeDB has native HNSW vector search.", "score": 0.95},
        {"node_id": 2, "text": "Bananas are rich in potassium.", "score": 0.40},
        {"node_id": 3, "text": "BM25 index matches text via @@ operator.", "score": 0.88},
    ]

    result = await guardrail.filter_chunks("How does LatticeDB search work?", chunks)
    filtered = result.chunks

    assert len(filtered) == 2
    assert filtered[0]["node_id"] == 1
    assert filtered[0]["grounding_score"] == 0.85

    assert filtered[1]["node_id"] == 3
    assert filtered[1]["grounding_score"] == 0.60

    assert result.low_grounding is False
    mock_client.system_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_guardrail_all_rejected_activates_low_grounding_flag():
    """When every chunk fails the threshold, no chunks pass and low_grounding is reported.

    No placeholder chunk is injected: it would reach the LLM as evidence and users as a source.
    """
    mock_client = AsyncMock()
    mock_response = SimpleNamespace(
        nouls={
            "relevant_0": SimpleNamespace(noul=0.15),
            "relevant_1": SimpleNamespace(noul=0.22),
        }
    )
    mock_client.system_one.return_value = mock_response

    guardrail = ContextGuardrail(threshold=0.50, ts_client=mock_client)
    chunks = [
        {"node_id": 10, "text": "Completely unrelated paragraph A.", "score": 0.3},
        {"node_id": 11, "text": "Completely unrelated paragraph B.", "score": 0.2},
    ]

    result = await guardrail.filter_chunks("Specific technical question", chunks)

    assert result.chunks == []
    assert result.low_grounding is True


@pytest.mark.asyncio
async def test_guardrail_aclose_and_close():
    """Verify ContextGuardrail cleanly invokes client.aclose() on shutdown."""
    mock_client = AsyncMock()
    guardrail = ContextGuardrail(ts_client=mock_client)

    await guardrail.aclose()
    mock_client.aclose.assert_awaited_once()

    await guardrail.close()
    assert mock_client.aclose.await_count == 2


@pytest.mark.asyncio
async def test_guardrail_system_one_error_graceful_fallback():
    """Verify guardrail catches system_one exceptions and gracefully preserves chunks."""
    mock_client = AsyncMock()
    mock_client.system_one.side_effect = RuntimeError(
        "403 Forbidden: Request not allowed by policy"
    )

    guardrail = ContextGuardrail(threshold=0.50, ts_client=mock_client)
    chunks = [
        {"node_id": 1, "text": "Chunk text 1", "rerank_score": 0.88},
        {"node_id": 2, "text": "Chunk text 2", "score": 0.75},
    ]

    result = await guardrail.filter_chunks("Any query", chunks)
    filtered = result.chunks

    assert len(filtered) == 2
    assert filtered[0]["node_id"] == 1
    assert filtered[0]["grounding_score"] == 0.88
    assert filtered[1]["node_id"] == 2
    assert filtered[1]["grounding_score"] == 0.75
    assert result.low_grounding is False


@pytest.mark.asyncio
async def test_guardrail_concurrent_calls_do_not_share_grounding_state():
    """One request's verdict must not leak into a concurrent request on the shared instance."""
    import asyncio

    async def _system_one(state, questions):
        await asyncio.sleep(0)
        prob = 0.9 if state["query"] == "good" else 0.1
        return SimpleNamespace(nouls={k: SimpleNamespace(noul=prob) for k in questions})

    mock_client = AsyncMock()
    mock_client.system_one.side_effect = _system_one
    guardrail = ContextGuardrail(threshold=0.5, ts_client=mock_client)
    chunks = [{"node_id": 1, "text": "x"}]

    good, bad = await asyncio.gather(
        guardrail.filter_chunks("good", chunks), guardrail.filter_chunks("bad", chunks)
    )
    assert good.low_grounding is False
    assert bad.low_grounding is True
