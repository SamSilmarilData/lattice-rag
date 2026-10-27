from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from groq import AsyncGroq

from lattice_rag.config import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TIMEOUT_SECONDS
from lattice_rag.generation.prompt import build_messages
from lattice_rag.generation.resilient_synthesizer import ProviderOutcome
from lattice_rag.security import sanitize_error_detail

logger = structlog.get_logger(__name__)


class GroqSynthesizer:
    """Primary synthesizer using GroqCloud API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen/qwen3.8-27b",
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        client: AsyncGroq | None = None,
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        """Initializes the AsyncGroq client with configurable model and API key.

        Without a key (argument or GROQ_API_KEY) the client stays unset and every call
        raises, so the resilient cascade degrades to Gemini/FAQ instead of failing boot.

        ``timeout_seconds`` bounds each attempt: the whole buffered request, and each chunk
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
            self.client = AsyncGroq(api_key=resolved_key, max_retries=2, timeout=timeout_seconds)
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
        prompt = build_messages(query, context_chunks, graph_context)

        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": prompt.system},
                            {"role": "user", "content": prompt.user},
                        ],
                        max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                        stream=False,
                    ),
                    timeout=self.timeout_seconds,
                )
                if response.choices[0].finish_reason == "length" and outcome is not None:
                    outcome.truncated = True
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
                sanitized_error = self._describe_failure(e)
                logger.error("groq_synthesis_failed", error=sanitized_error)
                raise RuntimeError(f"Groq synthesis failed: {sanitized_error}") from e
        return ""

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
        prompt = build_messages(query, context_chunks, graph_context)

        max_attempts = 4
        emitted = False
        for attempt in range(max_attempts):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": prompt.system},
                            {"role": "user", "content": prompt.user},
                        ],
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
                        emitted = True
                        yield chunk.choices[0].delta.content
                    if (
                        chunk.choices
                        and chunk.choices[0].finish_reason == "length"
                        and outcome is not None
                    ):
                        outcome.truncated = True
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
                sanitized_error = self._describe_failure(e)
                logger.error("groq_stream_synthesis_failed", error=sanitized_error)
                raise RuntimeError(f"Groq stream synthesis failed: {sanitized_error}") from e
