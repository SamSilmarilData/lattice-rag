from __future__ import annotations

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
        client: AsyncGroq | None = None,
    ) -> None:
        """Initializes the AsyncGroq client with configurable model and API key."""
        self.model = model
        if client is not None:
            self.client = client
        elif api_key:
            self.client = AsyncGroq(api_key=api_key)
        else:
            self.client = AsyncGroq()

    def _build_prompt(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
    ) -> str:
        """Builds the structured system prompt with verified evidence and citation anchors."""
        prompt = (
            "You are a helpful and precise technical assistant answering questions based on provided evidence.\n"
            "Instructions:\n"
            "- Ground every factual claim directly in the provided evidence.\n"
            "- Cite source chunks by their position or ID (e.g. [Chunk 0], [Doc: xyz]) when making statements.\n"
            "- If relational graph triples are provided, use them to explain connections between entities.\n"
            "- Acknowledge when the evidence is insufficient to answer the question.\n"
            "- Be concise, direct, and authoritative.\n\n"
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
        """Non-streaming synthesis of a grounded answer."""
        logger.info("groq_synthesizing_answer", query=query, model=self.model)
        prompt = self._build_prompt(query, context_chunks, graph_context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            sanitized_error = sanitize_error_detail(str(e))
            logger.error("groq_synthesis_failed", error=sanitized_error)
            raise RuntimeError(f"Groq synthesis failed: {sanitized_error}") from e

    async def stream(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming synthesis of a grounded answer yielding delta text tokens."""
        logger.info("groq_streaming_synthesis", query=query, model=self.model)
        prompt = self._build_prompt(query, context_chunks, graph_context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    yield text
        except Exception as e:
            sanitized_error = sanitize_error_detail(str(e))
            logger.error("groq_stream_synthesis_failed", error=sanitized_error)
            raise RuntimeError(f"Groq stream synthesis failed: {sanitized_error}") from e
