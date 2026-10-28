"""The Resilient Synthesizer over a Model Pool: fall through, attribute, never degrade.

Every test drives :class:`ResilientSynthesizer` through its public ``synthesize`` and
``stream`` methods with fake pool members, buffered and streaming alike.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.generation.resilient_synthesizer import (
    PoolMember,
    ResilientSynthesizer,
    StreamOutcome,
)
from lattice_rag.vocabulary import GenerationProvider

GROQ = GenerationProvider.GROQ
GEMINI = GenerationProvider.GEMINI


class RateLimitedError(Exception):
    """Stands in for a provider 429 / quota exhaustion."""


class FakeModel:
    """A pool member's adapter: answers, or raises a rate limit, buffered and streaming."""

    def __init__(self, answer: str = "", *, throttled: bool = False) -> None:
        self.answer = answer
        self.throttled = throttled
        self.calls = 0
        self.max_tokens: list[int | None] = []

    async def synthesize(self, query, chunks, graph=None, *, max_tokens=None, outcome=None):
        self.calls += 1
        self.max_tokens.append(max_tokens)
        if self.throttled:
            raise RateLimitedError("429 rate_limit_exceeded")
        return self.answer

    async def stream(
        self, query, chunks, graph=None, *, max_tokens=None, outcome=None
    ) -> AsyncGenerator[str, None]:
        self.calls += 1
        self.max_tokens.append(max_tokens)
        if self.throttled:
            raise RateLimitedError("429 rate_limit_exceeded")
        for word in self.answer.split(" "):
            yield word + " "


def _pool(*members: tuple[GenerationProvider, str, Any]) -> list[PoolMember]:
    return [PoolMember(provider=p, model=m, client=c) for p, m, c in members]


async def _drain(synth: ResilientSynthesizer, outcome: StreamOutcome, route: str = "hybrid"):
    return "".join([t async for t in synth.stream("q", [], route=route, outcome=outcome)])


async def test_throttled_first_member_falls_through_to_the_next_buffered():
    first = FakeModel(throttled=True)
    second = FakeModel("Answer from gpt-oss.")
    synth = ResilientSynthesizer(
        pool=_pool((GROQ, "qwen/qwen3.8-27b", first), (GROQ, "openai/gpt-oss-120b", second))
    )

    res = await synth.synthesize("q", [])

    assert res.answer == "Answer from gpt-oss."
    assert (res.provider, res.model) == (GROQ, "openai/gpt-oss-120b")
    assert res.degraded is False
    assert (first.calls, second.calls) == (1, 1)


async def test_throttled_first_member_falls_through_to_the_next_streaming():
    first = FakeModel(throttled=True)
    second = FakeModel("Gemini answer.")
    synth = ResilientSynthesizer(
        pool=_pool((GROQ, "qwen/qwen3.8-27b", first), (GEMINI, "gemini-3.5-flash-lite", second))
    )

    outcome = StreamOutcome()
    text = await _drain(synth, outcome)

    assert text.strip() == "Gemini answer."
    assert (outcome.provider, outcome.model) == (GEMINI, "gemini-3.5-flash-lite")
    assert outcome.degraded is False
    assert outcome.error is None


async def test_first_member_answers_when_healthy_and_later_members_are_not_called():
    first, second = FakeModel("First."), FakeModel("Second.")
    synth = ResilientSynthesizer(pool=_pool((GROQ, "a", first), (GROQ, "b", second)))

    res = await synth.synthesize("q", [])

    assert (res.answer, res.model, res.degraded) == ("First.", "a", False)
    assert second.calls == 0


async def test_massive_context_uses_the_same_order_with_its_larger_budget():
    first, second = FakeModel("Groq wide answer."), FakeModel("Gemini.")
    synth = ResilientSynthesizer(
        pool=_pool((GROQ, "a", first), (GEMINI, "g", second)), default_max_tokens=768
    )

    res = await synth.synthesize("q", [], route="massive_context")

    assert (res.provider, res.model) == (GROQ, "a")
    assert first.max_tokens == [1536]
    assert second.calls == 0


async def test_each_member_has_its_own_breaker():
    first, second = FakeModel(throttled=True), FakeModel("Second.")
    synth = ResilientSynthesizer(pool=_pool((GROQ, "a", first), (GROQ, "b", second)), fail_max=2)

    for _ in range(2):
        await synth.synthesize("q", [])
    assert synth.circuit_states == {"groq:a": "open", "groq:b": "closed"}
    assert synth.circuit_status == "partially_open"

    res = await synth.synthesize("q", [])
    assert first.calls == 2  # skipped once its breaker opened
    assert res.model == "b"


async def test_stream_failures_feed_the_same_member_breaker():
    first, second = FakeModel(throttled=True), FakeModel("ok")
    synth = ResilientSynthesizer(pool=_pool((GROQ, "a", first), (GEMINI, "g", second)), fail_max=2)

    for _ in range(2):
        await _drain(synth, StreamOutcome())

    assert synth.circuit_states["groq:a"] == "open"


async def test_circuit_status_is_closed_when_every_member_is_closed_and_open_when_all_are():
    synth = ResilientSynthesizer(
        pool=_pool(
            (GROQ, "a", FakeModel(throttled=True)), (GEMINI, "g", FakeModel(throttled=True))
        ),
        fail_max=1,
    )
    assert synth.circuit_status == "closed"

    await synth.synthesize("q", [])

    assert synth.circuit_status == "open"


async def test_no_member_answering_serves_a_degraded_faq_answer_on_both_paths():
    faq = MagicMock()
    faq.find_closest_fallback = AsyncMock(return_value="Pre-computed FAQ answer.")
    synth = ResilientSynthesizer(
        pool=_pool(
            (GROQ, "a", FakeModel(throttled=True)), (GEMINI, "g", FakeModel(throttled=True))
        ),
        fallback_cache=faq,
    )

    res = await synth.synthesize("q", [])
    assert (res.answer, res.provider, res.model, res.degraded) == (
        "Pre-computed FAQ answer.",
        GenerationProvider.REDIS_FAQ,
        None,
        True,
    )

    outcome = StreamOutcome()
    assert await _drain(synth, outcome) == "Pre-computed FAQ answer."
    assert (outcome.provider, outcome.model, outcome.degraded) == (
        GenerationProvider.REDIS_FAQ,
        None,
        True,
    )


async def test_a_stream_failing_after_its_first_token_does_not_cascade():
    async def partial(q, c, g=None, **_kw):
        yield "Partial "
        raise RuntimeError("connection reset")

    first = MagicMock()
    first.stream = partial
    second = FakeModel("A different full answer")
    synth = ResilientSynthesizer(pool=_pool((GROQ, "a", first), (GROQ, "b", second)))

    outcome = StreamOutcome()
    assert await _drain(synth, outcome) == "Partial "
    assert second.calls == 0
    assert (outcome.provider, outcome.model) == (GROQ, "a")
    assert outcome.error
    # An interrupted answer is not a complete pool answer: degraded, so never cached
    assert outcome.degraded is True


async def test_cancellation_is_not_counted_as_a_member_failure():
    client = MagicMock()
    client.synthesize = AsyncMock(side_effect=asyncio.CancelledError())
    synth = ResilientSynthesizer(pool=_pool((GROQ, "a", client)), fail_max=1)

    with pytest.raises(asyncio.CancelledError):
        await synth.synthesize("q", [])
    assert synth.circuit_status == "closed"


async def test_a_hung_first_member_times_out_and_falls_through_on_both_paths():
    """Real provider adapters over fake SDK clients: Groq never answers, Gemini does."""
    from lattice_rag.generation.gemini_synthesizer import GeminiSynthesizer
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
    groq = GroqSynthesizer(client=groq_client, model="a", timeout_seconds=0.05)
    gemini = GeminiSynthesizer(client=gemini_client, model="g", timeout_seconds=0.05)
    synth = ResilientSynthesizer(pool=_pool((GROQ, "a", groq), (GEMINI, "g", gemini)))

    res = await asyncio.wait_for(synth.synthesize("q", []), timeout=2)
    assert (res.answer, res.model, res.degraded) == ("Gemini answer.", "g", False)

    outcome = StreamOutcome()
    text = await asyncio.wait_for(_drain(synth, outcome), timeout=2)
    assert text == "Gemini answer."
    assert (outcome.model, outcome.degraded, outcome.error) == ("g", False, None)
