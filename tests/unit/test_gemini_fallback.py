"""Unit tests for GeminiFallback with mocked client flows."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.generation.gemini_fallback import GeminiFallback


@pytest.fixture
def mock_gemini_client():
    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock()
    mock_client.aio.models.generate_content_stream = MagicMock()
    return mock_client


@pytest.mark.asyncio
async def test_gemini_fallback_synthesize_success(mock_gemini_client):
    synthesizer = GeminiFallback(client=mock_gemini_client, model="gemini-3.8-flash")

    mock_resp = MagicMock()
    mock_resp.text = "Synthesized fallback answer via Gemini 3.8 Flash."
    mock_gemini_client.aio.models.generate_content.return_value = mock_resp

    answer = await synthesizer.synthesize("Question", [{"doc_id": "d1", "text": "evidence"}])

    assert answer == "Synthesized fallback answer via Gemini 3.8 Flash."
    mock_gemini_client.aio.models.generate_content.assert_awaited_once()
    call_kwargs = mock_gemini_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    assert "untrusted" in call_kwargs["config"]["system_instruction"]
    assert "<evidence>" in call_kwargs["contents"]


@pytest.mark.asyncio
async def test_gemini_fallback_stream_success(mock_gemini_client):
    synthesizer = GeminiFallback(client=mock_gemini_client, model="gemini-3.8-flash")

    async def mock_stream_gen():
        for token in ["Gemini", " fallback", " stream."]:
            chunk = MagicMock()
            chunk.text = token
            yield chunk

    mock_gemini_client.aio.models.generate_content_stream.return_value = mock_stream_gen()

    tokens = []
    async for t in synthesizer.stream("Query", []):
        tokens.append(t)

    assert "".join(tokens) == "Gemini fallback stream."


@pytest.mark.asyncio
async def test_gemini_fallback_error_sanitization(mock_gemini_client):
    synthesizer = GeminiFallback(client=mock_gemini_client)
    mock_gemini_client.aio.models.generate_content.side_effect = Exception(
        "Authorization: Bearer secret_api_key_fail"
    )

    with pytest.raises(RuntimeError) as exc_info:
        await synthesizer.synthesize("Query", [])

    err_msg = str(exc_info.value)
    assert "secret_api_key_fail" not in err_msg
    assert "Gemini fallback synthesis failed" in err_msg


@pytest.mark.asyncio
async def test_gemini_without_key_constructs_and_raises_on_use(monkeypatch):
    """A missing GEMINI_API_KEY must not crash construction; calls raise so the cascade degrades."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    synthesizer = GeminiFallback(api_key=None)
    assert synthesizer.client is None
    with pytest.raises(RuntimeError, match="not configured"):
        await synthesizer.synthesize("q", [])


async def _hang(*_args, **_kwargs):
    await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_gemini_synthesize_times_out_and_raises(mock_gemini_client):
    """A hung generate_content must fail like any provider error so the cascade moves on."""
    synthesizer = GeminiFallback(client=mock_gemini_client, timeout_seconds=0.05)
    mock_gemini_client.aio.models.generate_content.side_effect = _hang

    with pytest.raises(RuntimeError, match="Gemini fallback synthesis failed: timed out"):
        await asyncio.wait_for(synthesizer.synthesize("Query", []), timeout=2)


@pytest.mark.asyncio
async def test_gemini_stream_times_out_waiting_for_first_chunk(mock_gemini_client):
    synthesizer = GeminiFallback(client=mock_gemini_client, timeout_seconds=0.05)

    async def stalled_stream():
        await asyncio.sleep(3600)
        yield MagicMock(text="never")

    mock_gemini_client.aio.models.generate_content_stream.return_value = stalled_stream()

    async def drain() -> list[str]:
        return [t async for t in synthesizer.stream("Query", [])]

    with pytest.raises(RuntimeError, match="Gemini fallback stream synthesis failed: timed out"):
        await asyncio.wait_for(drain(), timeout=2)


@pytest.mark.asyncio
async def test_gemini_stream_times_out_when_a_later_chunk_stalls(mock_gemini_client):
    synthesizer = GeminiFallback(client=mock_gemini_client, timeout_seconds=0.05)

    async def stalls_after_first():
        yield MagicMock(text="partial")
        await asyncio.sleep(3600)
        yield MagicMock(text="never")

    mock_gemini_client.aio.models.generate_content_stream.return_value = stalls_after_first()
    tokens: list[str] = []

    async def drain() -> None:
        async for t in synthesizer.stream("Query", []):
            tokens.append(t)

    with pytest.raises(RuntimeError, match="timed out"):
        await asyncio.wait_for(drain(), timeout=2)
    assert tokens == ["partial"]


def test_gemini_client_built_from_key_carries_the_http_timeout(monkeypatch):
    synthesizer = GeminiFallback(api_key="test-key", timeout_seconds=12.5)
    assert synthesizer.client is not None
    assert synthesizer.client._api_client._http_options.timeout == 12500
