"""Unit tests for the deep ResilientSynthesizer module (TDD RED -> GREEN)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.generation.resilient_synthesizer import (
    ResilientSynthesizer,
    StreamOutcome,
    SynthesisResult,
)


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

    res = await synthesizer.synthesize(
        "Summarize everything", context_chunks=[], route="massive_context"
    )
    assert res.answer == "Massive context synthesis."
    assert res.provider == "gemini"
    mock_groq.synthesize.assert_not_called()
    mock_gemini.synthesize.assert_awaited_once()


@pytest.mark.asyncio
async def test_resilient_synthesizer_streaming_primary_and_fallback():
    # 1. Successful stream
    mock_groq = MagicMock()

    async def _stream_groq(q, c, g, **_kw):
        for token in ["Hello", " ", "from", " ", "Groq"]:
            yield token

    mock_groq.stream = _stream_groq
    mock_gemini = MagicMock()
    mock_fb_cache = MagicMock()

    synthesizer = ResilientSynthesizer(
        groq=mock_groq, gemini=mock_gemini, fallback_cache=mock_fb_cache
    )
    tokens = [t async for t in synthesizer.stream("Hi", context_chunks=[])]
    assert "".join(tokens) == "Hello from Groq"

    # 2. Failed stream cascades to Gemini
    mock_groq_fail = MagicMock()

    async def _stream_fail(q, c, g, **_kw):
        raise RuntimeError("Stream disconnect")
        yield "never"

    async def _stream_gemini(q, c, g, **_kw):
        for token in ["Hello", " ", "from", " ", "Gemini"]:
            yield token

    mock_groq_fail.stream = _stream_fail
    mock_gemini.stream = _stream_gemini
    synthesizer_failover = ResilientSynthesizer(
        groq=mock_groq_fail, gemini=mock_gemini, fallback_cache=mock_fb_cache
    )
    fallback_tokens = [t async for t in synthesizer_failover.stream("Hi", context_chunks=[])]
    assert "".join(fallback_tokens) == "Hello from Gemini"


def _failing_stream(exc: Exception):
    async def _stream(q, c, g, **_kw):
        raise exc
        yield "never"  # pragma: no cover

    return _stream


def _token_stream(tokens: list[str]):
    async def _stream(q, c, g, **_kw):
        for tok in tokens:
            yield tok

    return _stream


@pytest.mark.asyncio
async def test_breaker_trips_after_repeated_groq_failures_and_skips_groq():
    groq = MagicMock()
    groq.synthesize = AsyncMock(side_effect=RuntimeError("Groq 503"))
    gemini = MagicMock()
    gemini.synthesize = AsyncMock(return_value="Gemini answer")
    synthesizer = ResilientSynthesizer(groq=groq, gemini=gemini, fail_max=3)

    for _ in range(3):
        await synthesizer.synthesize("q", context_chunks=[])
    assert synthesizer.circuit_status == "open"

    groq.synthesize.reset_mock()
    res = await synthesizer.synthesize("q", context_chunks=[])
    groq.synthesize.assert_not_called()
    assert res.provider == "gemini"
    assert res.degraded is True


@pytest.mark.asyncio
async def test_stream_failures_feed_the_same_breaker():
    groq = MagicMock()
    groq.stream = _failing_stream(RuntimeError("Groq stream 503"))
    gemini = MagicMock()
    gemini.stream = _token_stream(["ok"])
    synthesizer = ResilientSynthesizer(groq=groq, gemini=gemini, fail_max=2)

    for _ in range(2):
        _ = [t async for t in synthesizer.stream("q", context_chunks=[])]
    assert synthesizer.circuit_status == "open"


@pytest.mark.asyncio
async def test_stream_outcome_reports_gemini_fallback_as_degraded():
    groq = MagicMock()
    groq.stream = _failing_stream(RuntimeError("Groq down"))
    gemini = MagicMock()
    gemini.stream = _token_stream(["Gemini", " answer"])
    synthesizer = ResilientSynthesizer(groq=groq, gemini=gemini)

    outcome = StreamOutcome()
    tokens = [t async for t in synthesizer.stream("q", context_chunks=[], outcome=outcome)]

    assert "".join(tokens) == "Gemini answer"
    assert outcome.provider == "gemini"
    assert outcome.degraded is True
    assert outcome.error is None


@pytest.mark.asyncio
async def test_stream_failure_after_first_token_does_not_append_second_answer():
    async def _partial(q, c, g, **_kw):
        yield "Partial "
        raise RuntimeError("connection reset")

    groq = MagicMock()
    groq.stream = _partial
    gemini = MagicMock()
    gemini.stream = _token_stream(["A complete different answer"])
    synthesizer = ResilientSynthesizer(groq=groq, gemini=gemini)

    outcome = StreamOutcome()
    tokens = [t async for t in synthesizer.stream("q", context_chunks=[], outcome=outcome)]

    assert tokens == ["Partial "]
    assert outcome.degraded is True
    assert outcome.error


@pytest.mark.asyncio
async def test_cancellation_is_not_counted_as_groq_failure():
    import asyncio

    groq = MagicMock()
    groq.synthesize = AsyncMock(side_effect=asyncio.CancelledError())
    synthesizer = ResilientSynthesizer(groq=groq, gemini=None, fail_max=1)

    with pytest.raises(asyncio.CancelledError):
        await synthesizer.synthesize("q", context_chunks=[])
    assert synthesizer.circuit_status == "closed"


def _hung_groq_and_answering_gemini(timeout: float):
    """Real provider adapters over fake clients: Groq never answers, Gemini does."""
    import asyncio

    from lattice_rag.generation.gemini_fallback import GeminiFallback
    from lattice_rag.generation.groq_synthesizer import GroqSynthesizer

    async def hang(*_a, **_kw):
        await asyncio.sleep(3600)

    groq_client = MagicMock()
    groq_client.chat.completions.create = AsyncMock(side_effect=hang)

    async def gemini_chunks():
        yield MagicMock(text="Gemini answer.", candidates=[])

    gemini_client = MagicMock()
    gemini_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text="Gemini answer.", candidates=[])
    )
    gemini_client.aio.models.generate_content_stream = MagicMock(
        side_effect=lambda **_kw: gemini_chunks()
    )
    return (
        GroqSynthesizer(client=groq_client, timeout_seconds=timeout),
        GeminiFallback(client=gemini_client, timeout_seconds=timeout),
    )


@pytest.mark.asyncio
async def test_a_hung_primary_times_out_and_cascades_on_both_paths():
    import asyncio

    groq, gemini = _hung_groq_and_answering_gemini(timeout=0.05)
    synthesizer = ResilientSynthesizer(groq=groq, gemini=gemini, fallback_cache=MagicMock())

    res = await asyncio.wait_for(synthesizer.synthesize("q", context_chunks=[]), timeout=2)
    assert (res.answer, res.provider, res.degraded) == ("Gemini answer.", "gemini", True)

    outcome = StreamOutcome()

    async def drain() -> list[str]:
        return [t async for t in synthesizer.stream("q", context_chunks=[], outcome=outcome)]

    tokens = await asyncio.wait_for(drain(), timeout=2)
    assert "".join(tokens) == "Gemini answer."
    assert (outcome.provider, outcome.degraded, outcome.error) == ("gemini", True, None)
