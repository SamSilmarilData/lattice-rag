from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator

import structlog
from groq import AsyncGroq

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
        """Initializes the AsyncGroq client with configurable model and API key."""
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self.client = client
        elif api_key:
            self.client = AsyncGroq(api_key=api_key, max_retries=2)
        else:
            self.client = AsyncGroq(max_retries=2)

    def _build_prompt(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
    ) -> str:
        """Builds the structured system prompt with verified evidence and citation anchors."""
        parts: list[str] = [
            "You are a helpful and precise technical assistant answering questions based on provided evidence.",
            "Instructions:",
            "- Ground every factual claim directly in the provided evidence.",
            "- Explicitly mention exact technical terms, operators, and lifecycle identifiers as stated in the evidence (such as operators <=> and @@, or event types stage, token, done).",
            "- Cite source chunks by their position or ID (e.g. [Chunk 0], [Doc: xyz]) when making statements.",
            "- If relational graph triples are provided, use them to explain connections between entities.",
            "- Acknowledge when the evidence is insufficient to answer the question.",
            "- Be concise, direct, and authoritative.",
            "",
            "Text Chunks:",
        ]

        if not context_chunks:
            parts.append("(No text chunks retrieved)")
        else:
            for i, chunk in enumerate(context_chunks):
                doc_id = chunk.get("doc_id") or chunk.get("document_id") or "unknown"
                chunk_text = chunk.get("text", str(chunk))
                parts.append(f"[Chunk {i} (Doc: {doc_id})]: {chunk_text}")

        if graph_context:
            parts.append("")
            parts.append("Knowledge Graph Context:")
            edges = graph_context.get("edges", [])
            nodes = {n.get("node_id", n.get("id")): n.get("name", "") for n in graph_context.get("nodes", [])}
            if edges:
                for edge in edges:
                    src = nodes.get(edge.get("source_id"), edge.get("source_name", "Unknown"))
                    tgt = nodes.get(edge.get("target_id"), edge.get("target_name", "Unknown"))
                    rel = edge.get("relation_type", "RELATION")
                    parts.append(f"- {src} --[{rel}]--> {tgt}")
            else:
                parts.append(json.dumps(graph_context))

        parts.append("")
        parts.append(f"User Question: {query}")
        return "\n".join(parts)

    async def synthesize(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
    ) -> str:
        """Non-streaming synthesis of a grounded answer."""
        logger.info("groq_synthesizing_answer", query=query, model=self.model)
        prompt = self._build_prompt(query, context_chunks, graph_context)

        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate_limit_exceeded" in err_str) and attempt < max_attempts - 1:
                    wait_sec = 2.0 * (attempt + 1) + 0.5
                    logger.warning("groq_rate_limit_retry", attempt=attempt + 1, wait_sec=wait_sec, query=query)
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
        logger.info("groq_streaming_synthesis", query=query, model=self.model)
        prompt = self._build_prompt(query, context_chunks, graph_context)

        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        yield text
                return
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate_limit_exceeded" in err_str) and attempt < max_attempts - 1:
                    wait_sec = 2.0 * (attempt + 1) + 0.5
                    logger.warning("groq_stream_rate_limit_retry", attempt=attempt + 1, wait_sec=wait_sec, query=query)
                    await asyncio.sleep(wait_sec)
                    continue
                sanitized_error = sanitize_error_detail(err_str)
                logger.error("groq_stream_synthesis_failed", error=sanitized_error)
                raise RuntimeError(f"Groq stream synthesis failed: {sanitized_error}") from e
