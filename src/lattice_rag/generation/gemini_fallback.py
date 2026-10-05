from __future__ import annotations

import inspect
import json
from typing import Any, AsyncGenerator

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
        """Initializes the Gemini genai client with configurable model and API key."""
        self.model = model
        if client is not None:
            self.client = client
        elif api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

    def _build_prompt(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
    ) -> str:
        """Builds the system prompt with evidence for fallback/massive context synthesis."""
        prompt = (
            "You are a helpful knowledge assistant answering questions based on provided context.\n"
            "Note: You are acting as a secondary synthesizer for high-context or failover requests.\n"
            "Instructions:\n"
            "- Ground every claim directly in the provided evidence.\n"
            "- Cite source chunks by their position or ID when possible.\n"
            "- If relational graph triples are provided, use them to explain connections between entities.\n"
            "- Acknowledge when evidence is insufficient to answer the question.\n"
            "- Be concise and precise.\n\n"
        )
        prompt += "Text Chunks:\n"
        if not context_chunks:
            prompt += "(No text chunks retrieved)\n"
        for i, chunk in enumerate(context_chunks):
            doc_id = chunk.get("doc_id") or chunk.get("document_id") or "unknown"
            chunk_text = chunk.get("text", str(chunk))
            prompt += f"[Chunk {i} (Doc: {doc_id})]: {chunk_text}\n"

        if graph_context:
            prompt += "\nKnowledge Graph Context:\n"
            edges = graph_context.get("edges", [])
            nodes = {n.get("node_id", n.get("id")): n.get("name", "") for n in graph_context.get("nodes", [])}
            if edges:
                for edge in edges:
                    src = nodes.get(edge.get("source_id"), edge.get("source_name", "Unknown"))
                    tgt = nodes.get(edge.get("target_id"), edge.get("target_name", "Unknown"))
                    rel = edge.get("relation_type", "RELATION")
                    prompt += f"- {src} --[{rel}]--> {tgt}\n"
            else:
                prompt += f"{json.dumps(graph_context)}\n"

        prompt += f"\nUser Question: {query}"
        return prompt

    async def synthesize(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
    ) -> str:
        """Non-streaming synthesis using Gemini."""
        logger.info("gemini_synthesizing_answer", query=query, model=self.model)
        prompt = self._build_prompt(query, context_chunks, graph_context)

        try:
            response = await self.client.aio.models.generate_content(
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
        logger.info("gemini_streaming_synthesis", query=query, model=self.model)
        prompt = self._build_prompt(query, context_chunks, graph_context)

        try:
            stream_result = self.client.aio.models.generate_content_stream(
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
