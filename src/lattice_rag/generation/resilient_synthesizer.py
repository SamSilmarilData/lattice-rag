"""Deep ResilientSynthesizer module encapsulating LLM circuit breaking and failover cascades."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import pybreaker

from lattice_rag.api.dtos import GenerationProvider
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
    provider: GenerationProvider


@dataclass
class StreamOutcome:
    """Per-call result of :meth:`ResilientSynthesizer.stream`, filled in as the stream ends.

    Passed in by the caller so concurrent streams never share state.
    """

    provider: GenerationProvider | None = None
    degraded: bool = False
    error: str | None = None


class ResilientSynthesizer:
    """Single generation path: Groq under a circuit breaker -> Gemini -> Redis FAQ.

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

    async def synthesize(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
        route: str = "hybrid",
    ) -> SynthesisResult:
        """Synthesize answer with automatic cascading failover: Groq -> Gemini -> Redis FAQ."""
        # Massive context queries route directly to large-context Gemini
        if route == "massive_context" and self.gemini is not None:
            try:
                answer = await self.gemini.synthesize(query, context_chunks, graph_context)
                return SynthesisResult(
                    answer=answer, degraded=False, provider=GenerationProvider.GEMINI
                )
            except Exception as e:
                logger.warning(
                    "Gemini massive context synthesis failed: %s", sanitize_error_detail(str(e))
                )
                return await self._offline_result(query)

        if self.groq is not None:
            try:
                with self.breaker.calling():
                    answer = str(await self.groq.synthesize(query, context_chunks, graph_context))
                return SynthesisResult(
                    answer=answer, degraded=False, provider=GenerationProvider.GROQ
                )
            except Exception as e:
                logger.warning(
                    "Primary Groq synthesis failed, cascading to Gemini: %s",
                    sanitize_error_detail(str(e)),
                )

        if self.gemini is not None:
            try:
                answer = await self.gemini.synthesize(query, context_chunks, graph_context)
                return SynthesisResult(
                    answer=answer, degraded=True, provider=GenerationProvider.GEMINI
                )
            except Exception as e2:
                logger.warning(
                    "Gemini fallback synthesis failed, cascading to Redis FAQ: %s",
                    sanitize_error_detail(str(e2)),
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

        if route == "massive_context":
            providers = [(GenerationProvider.GEMINI, self.gemini, False)]
            primary = GenerationProvider.GEMINI
        else:
            providers = [
                (GenerationProvider.GROQ, self.groq, True),
                (GenerationProvider.GEMINI, self.gemini, False),
            ]
            primary = GenerationProvider.GROQ

        for name, client, use_breaker in providers:
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
            provider=(
                GenerationProvider.REDIS_FAQ if self.fallback_cache else GenerationProvider.OFFLINE
            ),
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
