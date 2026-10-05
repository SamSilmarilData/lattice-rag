"""Unit tests for ContextGuardrail using TypeSafe AI Jev Noul."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lattice_rag.routing.guardrail import ContextGuardrail


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
    assert result == []
    assert guardrail.low_grounding_flag is False
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

    filtered = await guardrail.filter_chunks("How does LatticeDB search work?", chunks)

    assert len(filtered) == 2
    assert filtered[0]["node_id"] == 1
    assert filtered[0]["grounding_score"] == 0.85
    assert filtered[0]["low_grounding_flag"] is False

    assert filtered[1]["node_id"] == 3
    assert filtered[1]["grounding_score"] == 0.60
    assert filtered[1]["low_grounding_flag"] is False

    assert guardrail.low_grounding_flag is False
    mock_client.system_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_guardrail_all_rejected_activates_low_grounding_flag():
    """Verify when all candidate chunks fail threshold, low_grounding_flag is set and fallback injected."""
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

    filtered = await guardrail.filter_chunks("Specific technical question", chunks)

    assert len(filtered) == 1
    assert filtered[0]["low_grounding_flag"] is True
    assert filtered[0]["node_id"] == -1
    assert "INSUFFICIENT_EVIDENCE" in filtered[0]["text"]
    assert guardrail.low_grounding_flag is True


@pytest.mark.asyncio
async def test_guardrail_aclose_and_close():
    """Verify ContextGuardrail cleanly invokes client.aclose() on shutdown."""
    mock_client = AsyncMock()
    guardrail = ContextGuardrail(ts_client=mock_client)

    await guardrail.aclose()
    mock_client.aclose.assert_awaited_once()

    await guardrail.close()
    assert mock_client.aclose.await_count == 2
