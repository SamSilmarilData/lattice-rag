from __future__ import annotations

import inspect
import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from google import genai

from lattice_rag.security import sanitize_error_detail

logger = structlog.get_logger(__name__)


class GeminiFallback:
    """Gemini Flash fallback for massive cross-document context and circuit breaker cascade."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.8-flash",
        client: genai.Client | None = None,
    ) -> None:
        """Initializes the Gemini genai client with configurable model and API key.

        Without a key (argument, GEMINI_API_KEY or GOOGLE_API_KEY) the client stays unset
        and every call raises, so the cascade degrades instead of failing boot.
        """
        self.model = model
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

    def _build_prompt(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
    ) -> str:
        """Builds the system prompt with evidence for fallback/massive context synthesis."""
        parts: list[str] = [
            "You are a helpful knowledge assistant answering questions based on provided context.",
            "Note: You are acting as a secondary synthesizer for high-context or failover requests.",
            "Instructions:",
            "- Ground every claim directly in the provided evidence.",
            "- Cite source chunks by their position or ID when possible.",
            "- If relational graph triples are provided, use them to explain connections between entities.",
            "- Acknowledge when evidence is insufficient to answer the question.",
            "- Be concise and precise.",
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
            nodes = {
                n.get("node_id", n.get("id")): n.get("name", "")
                for n in graph_context.get("nodes", [])
            }
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
        """Non-streaming synthesis using Gemini."""
        logger.info("gemini_synthesizing_answer", model=self.model)
        client = self._require_client()
        prompt = self._build_prompt(query, context_chunks, graph_context)

        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
            )
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
    ) -> AsyncGenerator[str, None]:
        """Streaming synthesis using Gemini."""
        logger.info("gemini_streaming_synthesis", model=self.model)
        client = self._require_client()
        prompt = self._build_prompt(query, context_chunks, graph_context)

        try:
            stream_result = client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt,
            )
            response = await stream_result if inspect.isawaitable(stream_result) else stream_result
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            sanitized_error = sanitize_error_detail(str(e))
            logger.error("gemini_fallback_stream_failed", error=sanitized_error)
            raise RuntimeError(f"Gemini fallback stream synthesis failed: {sanitized_error}") from e
