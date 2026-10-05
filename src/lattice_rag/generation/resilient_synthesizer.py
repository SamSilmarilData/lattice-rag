"""Deep ResilientSynthesizer module encapsulating LLM circuit breaking and failover cascades."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import pybreaker

from lattice_rag.security import sanitize_error_detail

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesisResult:
    """Represents the output of the generation subsystem."""
    answer: str
    degraded: bool
    provider: str


class ResilientSynthesizer:
    """Deep module managing primary Groq generation, circuit breaker, and Gemini/Redis failovers."""

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
        self.breaker = pybreaker.CircuitBreaker(
            fail_max=fail_max,
            reset_timeout=reset_timeout,
            name="SynthesisBreaker",
        )

    @property
    def circuit_status(self) -> str:
        """Current state of the synthesis circuit breaker."""
        return self.breaker.current_state

    async def synthesize(
        self,
        query: str,
        context_chunks: list[dict],
        graph_context: dict | None = None,
        route: str = "hybrid",
    ) -> SynthesisResult:
        """Synthesize answer with automatic cascading failover: Groq -> Gemini -> Redis FAQ."""
        # Massive context queries route directly to large-context Gemini
        if route == "massive_context" and self.gemini is not None:
            try:
                answer = await self.gemini.synthesize(query, context_chunks, graph_context)
                return SynthesisResult(answer=answer, degraded=False, provider="gemini")
            except Exception as e:
                logger.warning("Gemini massive context synthesis failed: %s", sanitize_error_detail(str(e)))
                fallback = await self._get_offline_fallback(query)
                return SynthesisResult(answer=fallback, degraded=True, provider="redis_faq" if self.fallback_cache else "offline")

        # Standard routes: Primary Groq under circuit breaker
        if self.groq is not None:
            try:
                # Execute Groq synthesis through circuit breaker
                async def _call_groq():
                    return await self.groq.synthesize(query, context_chunks, graph_context)

                if self.breaker.current_state == "open":
                    raise pybreaker.CircuitBreakerError("Synthesis circuit breaker is OPEN")

                answer = await _call_groq()
                return SynthesisResult(answer=answer, degraded=False, provider="groq")
            except Exception as e:
                logger.warning("Primary Groq synthesis failed, cascading to Gemini: %s", sanitize_error_detail(str(e)))

        # Cascade 1: Gemini Fallback
        if self.gemini is not None:
            try:
                answer = await self.gemini.synthesize(query, context_chunks, graph_context)
                return SynthesisResult(answer=answer, degraded=True, provider="gemini")
            except Exception as e2:
                logger.warning("Gemini fallback synthesis failed, cascading to Redis FAQ: %s", sanitize_error_detail(str(e2)))

        # Cascade 2: Offline Redis FAQ Fallback
        fallback = await self._get_offline_fallback(query)
        return SynthesisResult(
            answer=fallback,
            degraded=True,
            provider="redis_faq" if self.fallback_cache else "offline",
        )

    async def stream(
        self,
        query: str,
        context_chunks: list[dict],
        graph_context: dict | None = None,
        route: str = "hybrid",
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens with transparent failover cascading."""
        # Massive context queries stream directly from Gemini
        if route == "massive_context" and self.gemini is not None:
            try:
                async for tok in self.gemini.stream(query, context_chunks, graph_context):
                    yield tok
                return
            except Exception as e:
                logger.warning("Gemini massive context stream failed: %s", sanitize_error_detail(str(e)))
                yield await self._get_offline_fallback(query)
                return

        # Standard routes: Primary Groq streaming
        groq_failed = False
        if self.groq is not None and self.breaker.current_state != "open":
            try:
                async for tok in self.groq.stream(query, context_chunks, graph_context):
                    yield tok
                return
            except Exception as e:
                logger.warning("Groq stream failed, falling back to Gemini: %s", sanitize_error_detail(str(e)))
                groq_failed = True

        # Cascade 1: Gemini streaming
        if self.gemini is not None:
            try:
                async for tok in self.gemini.stream(query, context_chunks, graph_context):
                    yield tok
                return
            except Exception as e2:
                logger.warning("Gemini stream failed, falling back to Redis FAQ: %s", sanitize_error_detail(str(e2)))

        # Cascade 2: Offline Redis FAQ
        yield await self._get_offline_fallback(query)

    async def _get_offline_fallback(self, query: str) -> str:
        """Lookup pre-computed FAQ fallback or return clean offline message."""
        if self.fallback_cache is not None and hasattr(self.fallback_cache, "find_closest_fallback"):
            try:
                cached = await self.fallback_cache.find_closest_fallback(query)
                if cached:
                    return cached
            except Exception as e:
                logger.error("Failed to query fallback cache: %s", sanitize_error_detail(str(e)))

        return "I'm sorry, all generation services are currently unavailable. Please try again in a moment."
