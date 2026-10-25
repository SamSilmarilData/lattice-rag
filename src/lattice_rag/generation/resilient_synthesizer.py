"""Deep ResilientSynthesizer module encapsulating LLM circuit breaking and failover cascades."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import pybreaker

from lattice_rag.security import sanitize_error_detail

logger = logging.getLogger(__name__)

OFFLINE_MESSAGE = (
    "I'm sorry, all generation services are currently unavailable. Please try again in a moment."
)


@dataclass(frozen=True)
class SynthesisResult:
    """Represents the output of the generation subsystem."""

    answer: str
    degraded: bool
    provider: str


@dataclass
class StreamOutcome:
    """Per-call result of :meth:`ResilientSynthesizer.stream`, filled in as the stream ends.

    Passed in by the caller so concurrent streams never share state.
    """

    provider: str = ""
    degraded: bool = False
    error: str | None = None


class ResilientSynthesizer:
    """Single generation path: Groq under a circuit breaker -> Gemini -> Redis FAQ.

    ``massive_context`` reverses the first two: Gemini -> Groq (breaker) -> Redis FAQ.

    Both the buffered and streaming paths record Groq outcomes on the same breaker, so
    repeated Groq failures in either path open it and later requests skip Groq.
    """

    def __init__(
        self,
        groq: Any = None,
        gemini: Any = None,
        fallback_cache: Any = None,
        fail_max: int = 3,
        reset_timeout: int = 30,
    ) -> None:
        self.groq = groq
        self.gemini = gemini
        self.fallback_cache = fallback_cache
        # Client disconnects cancel the task / close the stream; those are not Groq failures.
        self.breaker = pybreaker.CircuitBreaker(
            fail_max=fail_max,
            reset_timeout=reset_timeout,
            exclude=[asyncio.CancelledError, GeneratorExit],
            name="SynthesisBreaker",
        )

    @property
    def circuit_status(self) -> str:
        """Current state of the synthesis circuit breaker."""
        return str(self.breaker.current_state)

    @staticmethod
    def _primary(route: str) -> str:
        return "gemini" if route == "massive_context" else "groq"

    def _providers(self, route: str) -> list[tuple[str, Any, bool]]:
        """Cascade order for a route; only Groq runs under the circuit breaker."""
        groq = ("groq", self.groq, True)
        gemini = ("gemini", self.gemini, False)
        return [gemini, groq] if route == "massive_context" else [groq, gemini]

    async def synthesize(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
        route: str = "hybrid",
    ) -> SynthesisResult:
        """Synthesize answer with automatic cascading failover: Groq -> Gemini -> Redis FAQ."""
        for name, client, use_breaker in self._providers(route):
            if client is None:
                continue
            try:
                guard = self.breaker.calling() if use_breaker else contextlib.nullcontext()
                with guard:
                    answer = str(await client.synthesize(query, context_chunks, graph_context))
                return SynthesisResult(
                    answer=answer,
                    degraded=name != self._primary(route),
                    provider=name,
                )
            except Exception as e:
                logger.warning(
                    "%s synthesis failed, cascading: %s", name, sanitize_error_detail(str(e))
                )

        return await self._offline_result(query)

    async def stream(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
        route: str = "hybrid",
        outcome: StreamOutcome | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens, cascading only while nothing has been emitted yet.

        Once a provider has yielded a token, a later failure ends the stream (recorded in
        ``outcome.error``) instead of appending a second provider's full answer.
        """
        outcome = outcome if outcome is not None else StreamOutcome()

        primary = self._primary(route)

        for name, client, use_breaker in self._providers(route):
            if client is None:
                continue
            emitted = False
            try:
                guard = self.breaker.calling() if use_breaker else contextlib.nullcontext()
                with guard:
                    async for tok in client.stream(query, context_chunks, graph_context):
                        emitted = True
                        yield tok
                outcome.provider = name
                outcome.degraded = name != primary
                return
            except Exception as e:
                detail = sanitize_error_detail(str(e))
                if emitted:
                    logger.warning("%s stream failed mid-answer: %s", name, detail)
                    outcome.provider = name
                    outcome.degraded = True
                    outcome.error = "Generation was interrupted before the answer completed."
                    return
                logger.warning("%s stream failed before first token, cascading: %s", name, detail)

        fallback = await self._offline_result(query)
        outcome.provider = fallback.provider
        outcome.degraded = True
        yield fallback.answer

    async def _offline_result(self, query: str) -> SynthesisResult:
        answer = await self._get_offline_fallback(query)
        return SynthesisResult(
            answer=answer,
            degraded=True,
            provider="redis_faq" if self.fallback_cache else "offline",
        )

    async def _get_offline_fallback(self, query: str) -> str:
        """Lookup pre-computed FAQ fallback or return clean offline message."""
        if self.fallback_cache is not None:
            try:
                cached = await self.fallback_cache.find_closest_fallback(query)
                if cached:
                    return str(cached)
            except Exception as e:
                logger.error("Failed to query fallback cache: %s", sanitize_error_detail(str(e)))

        return OFFLINE_MESSAGE
