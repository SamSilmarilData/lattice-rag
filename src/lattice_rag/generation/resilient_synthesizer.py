"""Deep ResilientSynthesizer module encapsulating LLM circuit breaking and failover cascades."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, NamedTuple

import pybreaker

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
    """Represents the output of the generation subsystem."""

    answer: str
    degraded: bool
    provider: str
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

    provider: str = ""
    degraded: bool = False
    error: str | None = None
    truncated: bool = False


class _Provider(NamedTuple):
    name: str
    client: Any
    use_breaker: bool


@dataclass(frozen=True)
class _RoutePolicy:
    """How one route generates: provider cascade order and answer budget.

    ``max_tokens=None`` lets each provider use its configured default. The first provider in
    ``order`` is the primary; any other provider's answer is degraded.
    """

    order: tuple[str, ...]
    max_tokens: int | None

    @property
    def primary(self) -> str:
        return self.order[0]


class ResilientSynthesizer:
    """Single generation path: Groq under a circuit breaker -> Gemini -> Redis FAQ.

    ``massive_context`` reverses the first two: Gemini -> Groq (breaker) -> Redis FAQ, and
    gets a budget of ``max(MASSIVE_CONTEXT_MIN_MAX_TOKENS, default_max_tokens)``. Other
    routes keep each provider's configured default. Adapters record a length stop on a
    :class:`ProviderOutcome`; this class emits the single ``generation_truncated`` log.

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
        default_max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ) -> None:
        self.groq = groq
        self.gemini = gemini
        self.fallback_cache = fallback_cache
        self._default_policy = _RoutePolicy(order=("groq", "gemini"), max_tokens=None)
        self._massive_context_policy = _RoutePolicy(
            order=("gemini", "groq"),
            max_tokens=max(MASSIVE_CONTEXT_MIN_MAX_TOKENS, default_max_tokens),
        )
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

    def _policy(self, route: str) -> _RoutePolicy:
        return self._massive_context_policy if route == "massive_context" else self._default_policy

    def _providers(self, policy: _RoutePolicy) -> list[_Provider]:
        """Cascade order for a policy; only Groq runs under the circuit breaker."""
        clients = {
            "groq": _Provider("groq", self.groq, True),
            "gemini": _Provider("gemini", self.gemini, False),
        }
        return [clients[name] for name in policy.order]

    @staticmethod
    def _log_truncated(provider: str, route: str) -> None:
        logger.warning("generation_truncated", extra={"provider": provider, "route": route})

    async def synthesize(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
        route: str = "hybrid",
    ) -> SynthesisResult:
        """Synthesize answer with automatic cascading failover: Groq -> Gemini -> Redis FAQ."""
        policy = self._policy(route)
        for name, client, use_breaker in self._providers(policy):
            if client is None:
                continue
            attempt = ProviderOutcome()
            try:
                guard = self.breaker.calling() if use_breaker else contextlib.nullcontext()
                with guard:
                    answer = str(
                        await client.synthesize(
                            query,
                            context_chunks,
                            graph_context,
                            max_tokens=policy.max_tokens,
                            outcome=attempt,
                        )
                    )
                if attempt.truncated:
                    self._log_truncated(name, route)
                return SynthesisResult(
                    answer=answer,
                    degraded=name != policy.primary,
                    provider=name,
                    truncated=attempt.truncated,
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

        policy = self._policy(route)

        for name, client, use_breaker in self._providers(policy):
            if client is None:
                continue
            emitted = False
            attempt = ProviderOutcome()
            try:
                guard = self.breaker.calling() if use_breaker else contextlib.nullcontext()
                with guard:
                    async for tok in client.stream(
                        query,
                        context_chunks,
                        graph_context,
                        max_tokens=policy.max_tokens,
                        outcome=attempt,
                    ):
                        emitted = True
                        yield tok
                if attempt.truncated:
                    self._log_truncated(name, route)
                outcome.provider = name
                outcome.degraded = name != policy.primary
                outcome.truncated = attempt.truncated
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
