from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from groq import AsyncGroq

from lattice_rag.generation.prompt import build_messages
from lattice_rag.security import sanitize_error_detail

logger = structlog.get_logger(__name__)


class GroqSynthesizer:
    """Primary synthesizer using GroqCloud API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen/qwen3.8-27b",
        max_tokens: int = 256,
        client: AsyncGroq | None = None,
    ) -> None:
        """Initializes the AsyncGroq client with configurable model and API key.

        Without a key (argument or GROQ_API_KEY) the client stays unset and every call
        raises, so the resilient cascade degrades to Gemini/FAQ instead of failing boot.
        """
        self.model = model
        self.max_tokens = max_tokens
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client: AsyncGroq | None
        if client is not None:
            self.client = client
        elif resolved_key:
            self.client = AsyncGroq(api_key=resolved_key, max_retries=2)
        else:
            self.client = None

    def _require_client(self) -> AsyncGroq:
        if self.client is None:
            raise RuntimeError("Groq synthesis unavailable: GROQ_API_KEY is not configured")
        return self.client

    async def synthesize(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
    ) -> str:
        """Non-streaming synthesis of a grounded answer."""
        logger.info("groq_synthesizing_answer", model=self.model)
        client = self._require_client()
        prompt = build_messages(query, context_chunks, graph_context)

        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user", "content": prompt.user},
                    ],
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e)
                if (
                    "429" in err_str or "rate_limit_exceeded" in err_str
                ) and attempt < max_attempts - 1:
                    wait_sec = 2.0 * (attempt + 1) + 0.5
                    logger.warning("groq_rate_limit_retry", attempt=attempt + 1, wait_sec=wait_sec)
                    await asyncio.sleep(wait_sec)
                    continue
                sanitized_error = sanitize_error_detail(err_str)
                logger.error("groq_synthesis_failed", error=sanitized_error)
                raise RuntimeError(f"Groq synthesis failed: {sanitized_error}") from e
        return ""

    async def stream(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming synthesis of a grounded answer yielding delta text tokens."""
        logger.info("groq_streaming_synthesis", model=self.model)
        client = self._require_client()
        prompt = build_messages(query, context_chunks, graph_context)

        max_attempts = 4
        emitted = False
        for attempt in range(max_attempts):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user", "content": prompt.user},
                    ],
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        emitted = True
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                err_str = str(e)
                # Retrying after tokens went out would replay the answer from the start
                if (
                    not emitted
                    and ("429" in err_str or "rate_limit_exceeded" in err_str)
                    and attempt < max_attempts - 1
                ):
                    wait_sec = 2.0 * (attempt + 1) + 0.5
                    logger.warning(
                        "groq_stream_rate_limit_retry",
                        attempt=attempt + 1,
                        wait_sec=wait_sec,
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                sanitized_error = sanitize_error_detail(err_str)
                logger.error("groq_stream_synthesis_failed", error=sanitized_error)
                raise RuntimeError(f"Groq stream synthesis failed: {sanitized_error}") from e
