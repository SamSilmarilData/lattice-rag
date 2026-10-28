from __future__ import annotations

import inspect
import os
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from google import genai

from lattice_rag.config import DEFAULT_LLM_MAX_TOKENS
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
    ) -> None:
        """Initializes the Gemini genai client with configurable model and API key.

        Without a key (argument, GEMINI_API_KEY or GOOGLE_API_KEY) the client stays unset
        and every call raises, so the cascade degrades instead of failing boot.
        """
        self.model = model
        self.max_output_tokens = max_output_tokens
        resolved_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.client: genai.Client | None
        if client is not None:
            self.client = client
        elif resolved_key:
            self.client = genai.Client(api_key=resolved_key)
        else:
            self.client = None

    def _require_client(self) -> genai.Client:
        if self.client is None:
            raise RuntimeError("Gemini synthesis unavailable: GEMINI_API_KEY is not configured")
        return self.client

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
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=prompt.user,
                config={
                    "system_instruction": prompt.system,
                    "max_output_tokens": (
                        self.max_output_tokens if max_tokens is None else max_tokens
                    ),
                },
            )
            if outcome is not None and self._hit_token_cap(response):
                outcome.truncated = True
            return response.text or ""
        except Exception as e:
            sanitized_error = sanitize_error_detail(str(e))
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
            stream_result = client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt.user,
                config={
                    "system_instruction": prompt.system,
                    "max_output_tokens": (
                        self.max_output_tokens if max_tokens is None else max_tokens
                    ),
                },
            )
            response = await stream_result if inspect.isawaitable(stream_result) else stream_result
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                if outcome is not None and self._hit_token_cap(chunk):
                    outcome.truncated = True
        except Exception as e:
            sanitized_error = sanitize_error_detail(str(e))
            logger.error("gemini_fallback_stream_failed", error=sanitized_error)
            raise RuntimeError(f"Gemini fallback stream synthesis failed: {sanitized_error}") from e
