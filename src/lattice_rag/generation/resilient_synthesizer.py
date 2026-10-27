"""Deep ResilientSynthesizer module: the Model Pool cascade, its breakers and the FAQ."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any

import pybreaker

from lattice_rag.api.dtos import GenerationProvider
from lattice_rag.config import DEFAULT_LLM_MAX_TOKENS
from lattice_rag.security import sanitize_error_detail

logger = logging.getLogger(__name__)

# Broad cross-document answers need more room: massive_context gets the larger of this
# floor and the configured default, so raising LLM_MAX_TOKENS never leaves it with less
# room than the narrow routes.
MASSIVE_CONTEXT_MIN_MAX_TOKENS = 1536

OFFLINE_MESSAGE = (
    "I'm sorry, all generation services are currently unavailable. Please try again in a moment."
)


@dataclass(frozen=True)
class SynthesisResult:
    """Represents the output of the generation subsystem.

    ``model`` names the Model Pool member that answered; ``None`` for the FAQ or offline
    message. ``degraded`` is True only when no pool member answered.
    """

    answer: str
    degraded: bool
    provider: GenerationProvider
    model: str | None = None
    truncated: bool = False


@dataclass
class ProviderOutcome:
    """Per-call facts a provider adapter records; adapters record, the synthesizer logs."""

    truncated: bool = False


@dataclass
class StreamOutcome:
    """Per-call result of :meth:`ResilientSynthesizer.stream`, filled in as the stream ends.

    Passed in by the caller so concurrent streams never share state.
    """

    provider: GenerationProvider | None = None
    model: str | None = None
    degraded: bool = False
    error: str | None = None
    truncated: bool = False


@dataclass(frozen=True)
class PoolMember:
    """One Model Pool member: a provider, a model, and the adapter that calls it.

    ``client`` has the provider adapter shape: async ``synthesize(...)`` and async-generator
    ``stream(...)``, both taking ``max_tokens`` and ``outcome`` keywords.
    """

    provider: GenerationProvider
    model: str
    client: Any

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


class ResilientSynthesizer:
    """Single generation path: the Model Pool in order, each member under its own circuit
    breaker, then the Redis FAQ.

    Every Route uses the same order (ADR-0004). ``massive_context`` only gets a larger
    budget, ``max(MASSIVE_CONTEXT_MIN_MAX_TOKENS, default_max_tokens)``; other routes keep
    each adapter's configured default. Any member's answer is a normal answer, attributed
    by provider and model; only the FAQ/offline fallback (or a stream cut off mid-answer)
    is degraded. Adapters record a length stop on a :class:`ProviderOutcome`; this class
    emits the single ``generation_truncated`` log.

    The buffered and streaming paths record a member's outcomes on that member's breaker,
    so repeated failures in either path open it and later requests skip that member.
    """

    def __init__(
        self,
        pool: Sequence[PoolMember] = (),
        fallback_cache: Any = None,
        fail_max: int = 3,
        reset_timeout: int = 30,
        default_max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ) -> None:
        self.pool = tuple(pool)
        self.fallback_cache = fallback_cache
        self._massive_context_max_tokens = max(MASSIVE_CONTEXT_MIN_MAX_TOKENS, default_max_tokens)
        # Client disconnects cancel the task / close the stream; those are not member failures
        self._breakers = {
            member.label: pybreaker.CircuitBreaker(
                fail_max=fail_max,
                reset_timeout=reset_timeout,
                exclude=[asyncio.CancelledError, GeneratorExit],
                name=f"SynthesisBreaker[{member.label}]",
            )
            for member in self.pool
        }

    @property
    def circuit_states(self) -> dict[str, str]:
        """Breaker state per pool member, keyed ``provider:model``."""
        return {label: str(b.current_state) for label, b in self._breakers.items()}

    @property
    def circuit_status(self) -> str:
        """``closed`` or ``open`` when every member's breaker agrees, else ``partially_open``."""
        states = set(self.circuit_states.values())
        if states <= {"closed"}:
            return "closed"
        if states == {"open"}:
            return "open"
        return "partially_open"

    def _max_tokens(self, route: str) -> int | None:
        return self._massive_context_max_tokens if route == "massive_context" else None

    @staticmethod
    def _log_truncated(member: PoolMember, route: str) -> None:
        logger.warning(
            "generation_truncated",
            extra={"provider": str(member.provider), "model": member.model, "route": route},
        )

    async def synthesize(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
        route: str = "hybrid",
    ) -> SynthesisResult:
        """Answer from the first pool member that succeeds, else the Redis FAQ."""
        max_tokens = self._max_tokens(route)
        for member in self.pool:
            attempt = ProviderOutcome()
            try:
                with self._breakers[member.label].calling():
                    answer = str(
                        await member.client.synthesize(
                            query,
                            context_chunks,
                            graph_context,
                            max_tokens=max_tokens,
                            outcome=attempt,
                        )
                    )
                if attempt.truncated:
                    self._log_truncated(member, route)
                return SynthesisResult(
                    answer=answer,
                    degraded=False,
                    provider=member.provider,
                    model=member.model,
                    truncated=attempt.truncated,
                )
            except Exception as e:
                logger.warning(
                    "%s synthesis failed, trying the next pool member: %s",
                    member.label,
                    sanitize_error_detail(str(e)),
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
        """Stream response tokens, falling through only while nothing has been emitted yet.

        Once a member has yielded a token, a later failure ends the stream (recorded in
        ``outcome.error``; the partial answer is degraded) instead of appending a second
        member's full answer.
        """
        outcome = outcome if outcome is not None else StreamOutcome()
        max_tokens = self._max_tokens(route)

        for member in self.pool:
            emitted = False
            attempt = ProviderOutcome()
            try:
                with self._breakers[member.label].calling():
                    async for tok in member.client.stream(
                        query,
                        context_chunks,
                        graph_context,
                        max_tokens=max_tokens,
                        outcome=attempt,
                    ):
                        emitted = True
                        yield tok
                if attempt.truncated:
                    self._log_truncated(member, route)
                outcome.provider = member.provider
                outcome.model = member.model
                outcome.degraded = False
                outcome.truncated = attempt.truncated
                return
            except Exception as e:
                detail = sanitize_error_detail(str(e))
                if emitted:
                    logger.warning("%s stream failed mid-answer: %s", member.label, detail)
                    outcome.provider = member.provider
                    outcome.model = member.model
                    outcome.degraded = True
                    outcome.error = "Generation was interrupted before the answer completed."
                    return
                logger.warning(
                    "%s stream failed before first token, trying the next pool member: %s",
                    member.label,
                    detail,
                )

        fallback = await self._offline_result(query)
        outcome.provider = fallback.provider
        outcome.model = None
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
