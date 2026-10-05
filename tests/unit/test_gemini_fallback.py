"""Unit tests for GeminiFallback with mocked client flows."""
from __future__ import annotations

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
async def test_gemini_fallback_prompt_construction(mock_gemini_client):
    synthesizer = GeminiFallback(client=mock_gemini_client, model="gemini-3.8-flash")

    chunks = [{"doc_id": "doc1", "text": "Large cross-document context paragraph."}]
    graph_context = {
        "nodes": [{"node_id": 10, "name": "Gemini"}],
        "edges": [],
    }

    prompt = synthesizer._build_prompt("Summarize cross-doc context", chunks, graph_context)

    assert "Large cross-document context paragraph" in prompt
    assert "acting as a secondary synthesizer" in prompt
    assert "User Question: Summarize cross-doc context" in prompt


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
    mock_gemini_client.aio.models.generate_content.side_effect = Exception("Authorization: Bearer secret_api_key_fail")

    with pytest.raises(RuntimeError) as exc_info:
        await synthesizer.synthesize("Query", [])

    err_msg = str(exc_info.value)
    assert "secret_api_key_fail" not in err_msg
    assert "Gemini fallback synthesis failed" in err_msg
