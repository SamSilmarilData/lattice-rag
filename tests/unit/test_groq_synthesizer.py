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
async def test_groq_synthesizer_prompt_construction(mock_groq_client):
    synthesizer = GroqSynthesizer(client=mock_groq_client, model="qwen/qwen3.8-27b")

    chunks = [
        {"doc_id": "doc1", "text": "LatticeDB supports HNSW indexing."},
        {"doc_id": "doc2", "text": "FastEmbed generates quantized embeddings."},
    ]
    graph_context = {
        "nodes": [{"node_id": 1, "name": "LatticeDB"}, {"node_id": 2, "name": "HNSW"}],
        "edges": [{"source_id": 1, "target_id": 2, "relation_type": "USES_INDEX"}],
    }

    prompt = synthesizer._build_prompt("How does LatticeDB index?", chunks, graph_context)

    assert "LatticeDB supports HNSW indexing" in prompt
    assert "[Chunk 0 (Doc: doc1)]" in prompt
    assert "LatticeDB --[USES_INDEX]--> HNSW" in prompt
    assert "User Question: How does LatticeDB index?" in prompt


@pytest.mark.asyncio
async def test_groq_synthesizer_synthesize_success(mock_groq_client):
    synthesizer = GroqSynthesizer(client=mock_groq_client, model="qwen/qwen3.8-27b")

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "LatticeDB uses native HNSW indexing [Chunk 0]."
    mock_resp.choices = [mock_choice]
    mock_groq_client.chat.completions.create.return_value = mock_resp

    answer = await synthesizer.synthesize("What is LatticeDB?", [{"doc_id": "doc1", "text": "LatticeDB text"}])

    assert answer == "LatticeDB uses native HNSW indexing [Chunk 0]."
    mock_groq_client.chat.completions.create.assert_awaited_once()
    call_kwargs = mock_groq_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "qwen/qwen3.8-27b"
    assert call_kwargs["stream"] is False


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
    mock_groq_client.chat.completions.create.side_effect = Exception("Bearer gsk_1234567890abcdef Secret Key error")

    with pytest.raises(RuntimeError) as exc_info:
        await synthesizer.synthesize("Query", [])

    err_msg = str(exc_info.value)
    assert "gsk_1234567890abcdef" not in err_msg
    assert "Groq synthesis failed" in err_msg
