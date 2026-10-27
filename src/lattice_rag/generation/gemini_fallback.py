from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import structlog
from google import genai
from google.genai import types

from lattice_rag.config import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TIMEOUT_SECONDS
from lattice_rag.generation.prompt import build_messages
from lattice_rag.generation.resilient_synthesizer import ProviderOutcome
from lattice_rag.security import sanitize_error_detail

logger = structlog.get_logger(__name__)


class GeminiFallback:
    """Gemini Flash fallback for massive cross-document context and circuit breaker cascade."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.8-flash",
        max_output_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        client: genai.Client | None = None,
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        """Initializes the Gemini genai client with configurable model and API key.

        Without a key (argument, GEMINI_API_KEY or GOOGLE_API_KEY) the client stays unset
        and every call raises, so the cascade degrades instead of failing boot.

        ``timeout_seconds`` bounds each call: the whole buffered request, and each chunk
        (the first included) of a stream. A timeout raises like any provider failure.
        """
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        resolved_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.client: genai.Client | None
        if client is not None:
            self.client = client
        elif resolved_key:
            self.client = genai.Client(
                api_key=resolved_key,
                http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
            )
        else:
            self.client = None

    def _require_client(self) -> genai.Client:
        if self.client is None:
            raise RuntimeError("Gemini synthesis unavailable: GEMINI_API_KEY is not configured")
        return self.client

    def _describe_failure(self, error: Exception) -> str:
        if isinstance(error, TimeoutError):
            return f"timed out after {self.timeout_seconds:g}s"
        return sanitize_error_detail(str(error))

    @staticmethod
    def _hit_token_cap(response: Any) -> bool:
        for cand in getattr(response, "candidates", None) or []:
            reason = getattr(cand, "finish_reason", None)
            if getattr(reason, "name", str(reason)) == "MAX_TOKENS":
                return True
        return False

    async def synthesize(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        outcome: ProviderOutcome | None = None,
    ) -> str:
        """Non-streaming synthesis using Gemini."""
        logger.info("gemini_synthesizing_answer", model=self.model)
        client = self._require_client()
        prompt = build_messages(query, context_chunks, graph_context)

        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt.user,
                    config={
                        "system_instruction": prompt.system,
                        "max_output_tokens": (
                            self.max_output_tokens if max_tokens is None else max_tokens
                        ),
                    },
                ),
                timeout=self.timeout_seconds,
            )
            if outcome is not None and self._hit_token_cap(response):
                outcome.truncated = True
            return response.text or ""
        except Exception as e:
            sanitized_error = self._describe_failure(e)
            logger.error("gemini_fallback_synthesis_failed", error=sanitized_error)
            raise RuntimeError(f"Gemini fallback synthesis failed: {sanitized_error}") from e

    async def stream(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        outcome: ProviderOutcome | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming synthesis using Gemini."""
        logger.info("gemini_streaming_synthesis", model=self.model)
        client = self._require_client()
        prompt = build_messages(query, context_chunks, graph_context)

        try:
            # The SDK returns a coroutine that resolves to the chunk iterator
            stream_result: Any = client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt.user,
                config={
                    "system_instruction": prompt.system,
                    "max_output_tokens": (
                        self.max_output_tokens if max_tokens is None else max_tokens
                    ),
                },
            )
            if inspect.isawaitable(stream_result):
                stream_result = await asyncio.wait_for(stream_result, timeout=self.timeout_seconds)
            chunks: AsyncIterator[types.GenerateContentResponse] = aiter(stream_result)
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(chunks), timeout=self.timeout_seconds)
                except StopAsyncIteration:
                    break
                if chunk.text:
                    yield chunk.text
                if outcome is not None and self._hit_token_cap(chunk):
                    outcome.truncated = True
        except Exception as e:
            sanitized_error = self._describe_failure(e)
            logger.error("gemini_fallback_stream_failed", error=sanitized_error)
            raise RuntimeError(f"Gemini fallback stream synthesis failed: {sanitized_error}") from e
