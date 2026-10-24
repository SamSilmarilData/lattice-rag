"""Unit tests for GroqSynthesizer with mocked and structured evidence flows."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.generation.groq_synthesizer import GroqSynthesizer


@pytest.fixture
def mock_groq_client():
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client


@pytest.mark.asyncio
async def test_groq_synthesizer_synthesize_success(mock_groq_client):
    synthesizer = GroqSynthesizer(client=mock_groq_client, model="qwen/qwen3.8-27b")

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "LatticeDB uses native HNSW indexing [Chunk 0]."
    mock_resp.choices = [mock_choice]
    mock_groq_client.chat.completions.create.return_value = mock_resp

    answer = await synthesizer.synthesize(
        "What is LatticeDB?", [{"doc_id": "doc1", "text": "LatticeDB text"}]
    )

    assert answer == "LatticeDB uses native HNSW indexing [Chunk 0]."
    mock_groq_client.chat.completions.create.assert_awaited_once()
    call_kwargs = mock_groq_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "qwen/qwen3.8-27b"
    assert call_kwargs["stream"] is False
    roles = [m["role"] for m in call_kwargs["messages"]]
    assert roles == ["system", "user"]
    assert '<chunk index="0" doc="doc1">' in call_kwargs["messages"][1]["content"]


@pytest.mark.asyncio
async def test_groq_synthesizer_stream_success(mock_groq_client):
    synthesizer = GroqSynthesizer(client=mock_groq_client, model="qwen/qwen3.8-27b")

    async def mock_stream_gen():
        for token in ["Lattice", "DB", " is", " fast."]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            yield chunk

    mock_groq_client.chat.completions.create.return_value = mock_stream_gen()

    tokens = []
    async for t in synthesizer.stream("Query", []):
        tokens.append(t)

    assert "".join(tokens) == "LatticeDB is fast."


@pytest.mark.asyncio
async def test_groq_synthesizer_error_sanitization(mock_groq_client):
    synthesizer = GroqSynthesizer(client=mock_groq_client)
    mock_groq_client.chat.completions.create.side_effect = Exception(
        "Bearer gsk_1234567890abcdef Secret Key error"
    )

    with pytest.raises(RuntimeError) as exc_info:
        await synthesizer.synthesize("Query", [])

    err_msg = str(exc_info.value)
    assert "gsk_1234567890abcdef" not in err_msg
    assert "Groq synthesis failed" in err_msg


@pytest.mark.asyncio
async def test_groq_without_key_constructs_and_raises_on_use(monkeypatch):
    """A missing GROQ_API_KEY must not crash construction; calls raise so the cascade degrades."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    synthesizer = GroqSynthesizer(api_key=None)
    assert synthesizer.client is None
    with pytest.raises(RuntimeError, match="not configured"):
        await synthesizer.synthesize("q", [])


@pytest.mark.asyncio
async def test_groq_stream_does_not_replay_tokens_after_midstream_429(
    mock_groq_client, monkeypatch
):
    """A 429 raised after tokens were emitted must not restart the answer from the top."""
    from types import SimpleNamespace

    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    def _chunk(text):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    async def _partial_then_429():
        yield _chunk("First ")
        raise RuntimeError("Error code: 429 rate_limit_exceeded")

    mock_groq_client.chat.completions.create.side_effect = lambda **_: _partial_then_429()
    synthesizer = GroqSynthesizer(client=mock_groq_client)

    tokens: list[str] = []
    with pytest.raises(RuntimeError):
        async for tok in synthesizer.stream("q", []):
            tokens.append(tok)

    assert tokens == ["First "]
    assert mock_groq_client.chat.completions.create.call_count == 1
