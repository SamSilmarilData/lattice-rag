from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam

from lattice_rag.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TIMEOUT_SECONDS,
)
from lattice_rag.generation.prompt import build_messages
from lattice_rag.generation.resilient_synthesizer import ProviderOutcome
from lattice_rag.security import sanitize_error_detail

logger = structlog.get_logger(__name__)


class GroqSynthesizer:
    """GroqCloud provider adapter for one Model Pool member.

    It never retries: a rate limit, quota exhaustion, 5xx or timeout raises at once so the
    Resilient Synthesizer moves to the next pool member (ADR-0004).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GROQ_MODEL,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        client: AsyncGroq | None = None,
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        """Initializes the AsyncGroq client with configurable model and API key.

        Without a key (argument or GROQ_API_KEY) the client stays unset and every call
        raises, so the pool falls through to its next member instead of failing boot.

        ``timeout_seconds`` bounds each call: the whole buffered request, and each chunk
        (the first included) of a stream. A timeout raises like any provider failure.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client: AsyncGroq | None
        if client is not None:
            self.client = client
        elif resolved_key:
            # No SDK retries either: waiting out a 429 here would stall the whole pool
            self.client = AsyncGroq(api_key=resolved_key, max_retries=0, timeout=timeout_seconds)
        else:
            self.client = None

    def _describe_failure(self, error: Exception) -> str:
        if isinstance(error, TimeoutError):
            return f"timed out after {self.timeout_seconds:g}s"
        return sanitize_error_detail(str(error))

    def _require_client(self) -> AsyncGroq:
        if self.client is None:
            raise RuntimeError("Groq synthesis unavailable: GROQ_API_KEY is not configured")
        return self.client

    def _messages(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None,
    ) -> list[ChatCompletionMessageParam]:
        prompt = build_messages(query, context_chunks, graph_context)
        return [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

    async def synthesize(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        outcome: ProviderOutcome | None = None,
    ) -> str:
        """Non-streaming synthesis of a grounded answer.

        A length stop is recorded on ``outcome``; the caller decides how to report it.
        """
        logger.info("groq_synthesizing_answer", model=self.model)
        client = self._require_client()
        messages = self._messages(query, context_chunks, graph_context)

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                    stream=False,
                ),
                timeout=self.timeout_seconds,
            )
        except Exception as e:
            sanitized_error = self._describe_failure(e)
            logger.error("groq_synthesis_failed", model=self.model, error=sanitized_error)
            raise RuntimeError(f"Groq synthesis failed: {sanitized_error}") from e
        if response.choices[0].finish_reason == "length" and outcome is not None:
            outcome.truncated = True
        return response.choices[0].message.content or ""

    async def stream(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        outcome: ProviderOutcome | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming synthesis of a grounded answer yielding delta text tokens.

        A length stop is recorded on ``outcome``; the caller decides how to report it.
        """
        logger.info("groq_streaming_synthesis", model=self.model)
        client = self._require_client()
        messages = self._messages(query, context_chunks, graph_context)

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                    stream=True,
                ),
                timeout=self.timeout_seconds,
            )
            chunks = aiter(response)
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(chunks), timeout=self.timeout_seconds)
                except StopAsyncIteration:
                    break
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                if (
                    chunk.choices
                    and chunk.choices[0].finish_reason == "length"
                    and outcome is not None
                ):
                    outcome.truncated = True
        except Exception as e:
            sanitized_error = self._describe_failure(e)
            logger.error("groq_stream_synthesis_failed", model=self.model, error=sanitized_error)
            raise RuntimeError(f"Groq stream synthesis failed: {sanitized_error}") from e
