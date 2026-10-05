"""Unit tests for the deep ResilientSynthesizer module (TDD RED -> GREEN)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from lattice_rag.generation.resilient_synthesizer import ResilientSynthesizer, SynthesisResult


@pytest.mark.asyncio
async def test_resilient_synthesizer_primary_groq_success():
    mock_groq = MagicMock()
    mock_groq.synthesize = AsyncMock(return_value="Answer from Groq.")
    mock_gemini = MagicMock()
    mock_fb_cache = MagicMock()

    synthesizer = ResilientSynthesizer(
        groq=mock_groq,
        gemini=mock_gemini,
        fallback_cache=mock_fb_cache,
    )

    res = await synthesizer.synthesize("What is LatticeDB?", context_chunks=[{"text": "LatticeDB"}])
    assert isinstance(res, SynthesisResult)
    assert res.answer == "Answer from Groq."
    assert res.degraded is False
    assert res.provider == "groq"
    mock_groq.synthesize.assert_awaited_once()
    mock_gemini.synthesize.assert_not_called()


@pytest.mark.asyncio
async def test_resilient_synthesizer_groq_failure_cascades_to_gemini():
    mock_groq = MagicMock()
    mock_groq.synthesize = AsyncMock(side_effect=RuntimeError("Groq 503 Service Unavailable"))
    mock_gemini = MagicMock()
    mock_gemini.synthesize = AsyncMock(return_value="Fallback answer from Gemini.")
    mock_fb_cache = MagicMock()

    synthesizer = ResilientSynthesizer(
        groq=mock_groq,
        gemini=mock_gemini,
        fallback_cache=mock_fb_cache,
    )

    res = await synthesizer.synthesize("What is LatticeDB?", context_chunks=[{"text": "LatticeDB"}])
    assert res.answer == "Fallback answer from Gemini."
    assert res.degraded is True
    assert res.provider == "gemini"
    mock_groq.synthesize.assert_awaited_once()
    mock_gemini.synthesize.assert_awaited_once()


@pytest.mark.asyncio
async def test_resilient_synthesizer_double_failure_cascades_to_redis_faq():
    mock_groq = MagicMock()
    mock_groq.synthesize = AsyncMock(side_effect=RuntimeError("Groq rate limit"))
    mock_gemini = MagicMock()
    mock_gemini.synthesize = AsyncMock(side_effect=RuntimeError("Gemini timeout"))
    mock_fb_cache = MagicMock()
    mock_fb_cache.find_closest_fallback = AsyncMock(return_value="Pre-computed FAQ response.")

    synthesizer = ResilientSynthesizer(
        groq=mock_groq,
        gemini=mock_gemini,
        fallback_cache=mock_fb_cache,
    )

    res = await synthesizer.synthesize("How does LatticeDB work?", context_chunks=[])
    assert res.answer == "Pre-computed FAQ response."
    assert res.degraded is True
    assert res.provider == "redis_faq"
    mock_fb_cache.find_closest_fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_resilient_synthesizer_massive_context_routes_to_gemini():
    mock_groq = MagicMock()
    mock_gemini = MagicMock()
    mock_gemini.synthesize = AsyncMock(return_value="Massive context synthesis.")
    mock_fb_cache = MagicMock()

    synthesizer = ResilientSynthesizer(
        groq=mock_groq,
        gemini=mock_gemini,
        fallback_cache=mock_fb_cache,
    )

    res = await synthesizer.synthesize("Summarize everything", context_chunks=[], route="massive_context")
    assert res.answer == "Massive context synthesis."
    assert res.provider == "gemini"
    mock_groq.synthesize.assert_not_called()
    mock_gemini.synthesize.assert_awaited_once()


@pytest.mark.asyncio
async def test_resilient_synthesizer_streaming_primary_and_fallback():
    # 1. Successful stream
    mock_groq = MagicMock()
    async def _stream_groq(q, c, g):
        for token in ["Hello", " ", "from", " ", "Groq"]:
            yield token

    mock_groq.stream = _stream_groq
    mock_gemini = MagicMock()
    mock_fb_cache = MagicMock()

    synthesizer = ResilientSynthesizer(groq=mock_groq, gemini=mock_gemini, fallback_cache=mock_fb_cache)
    tokens = [t async for t in synthesizer.stream("Hi", context_chunks=[])]
    assert "".join(tokens) == "Hello from Groq"

    # 2. Failed stream cascades to Gemini
    mock_groq_fail = MagicMock()
    async def _stream_fail(q, c, g):
        raise RuntimeError("Stream disconnect")
        yield "never"

    async def _stream_gemini(q, c, g):
        for token in ["Hello", " ", "from", " ", "Gemini"]:
            yield token

    mock_groq_fail.stream = _stream_fail
    mock_gemini.stream = _stream_gemini
    synthesizer_failover = ResilientSynthesizer(groq=mock_groq_fail, gemini=mock_gemini, fallback_cache=mock_fb_cache)
    fallback_tokens = [t async for t in synthesizer_failover.stream("Hi", context_chunks=[])]
    assert "".join(fallback_tokens) == "Hello from Gemini"
