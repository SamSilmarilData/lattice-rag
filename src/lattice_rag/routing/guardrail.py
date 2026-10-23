from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import structlog
from typesafe_sdk import AsyncTypeSafeClient, Noul

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class GuardrailResult:
    """Per-call guardrail verdict: the chunks that passed and whether grounding is too weak."""

    chunks: list[dict[str, Any]]
    low_grounding: bool


class ContextGuardrail:
    """Filters retrieved chunks using TypeSafe AI Jev Noul to ensure contextual relevance."""

    def __init__(
        self,
        threshold: float = 0.5,
        ts_client: AsyncTypeSafeClient | None = None,
    ) -> None:
        """Initializes the ContextGuardrail.

        Args:
            threshold: The minimum noul probability required to keep a chunk (default: 0.5).
            ts_client: Optional injected AsyncTypeSafeClient (useful for testing).

        Raises:
            RuntimeError: If TYPESAFE_API_KEY environment variable is not set and ts_client is None.
        """
        if ts_client is not None:
            self._client = ts_client
        else:
            if not os.environ.get("TYPESAFE_API_KEY"):
                raise RuntimeError(
                    "TYPESAFE_API_KEY environment variable is not set. "
                    "Please set it to use the ContextGuardrail."
                )
            self._client = AsyncTypeSafeClient()
        self.threshold = threshold
        logger.debug("ContextGuardrail initialized", threshold=threshold)

    async def filter_chunks(self, query: str, chunks: list[dict[str, Any]]) -> GuardrailResult:
        """Filters chunks based on their relevance to the query.

        Args:
            query: The user's query.
            chunks: A list of dictionaries, each containing 'text', 'node_id', and 'score'.

        Returns:
            The chunks that meet the relevance threshold (each with a ``grounding_score``).
            ``low_grounding`` is True when candidates existed but none passed. The result is
            returned per call; the guardrail keeps no request state, so it is safe to share.
        """
        if not chunks:
            return GuardrailResult(chunks=[], low_grounding=False)

        logger.info("Filtering chunks", chunk_count=len(chunks))

        # Build a single state object mapping chunk indices to their text
        state = {"query": query}
        for i, chunk in enumerate(chunks):
            state[f"chunk_{i}"] = chunk.get("text", "")

        # Build a Noul question for each chunk
        questions = {}
        for i in range(len(chunks)):
            questions[f"relevant_{i}"] = Noul(
                instructions=f"Does chunk_{i} provide relevant factual grounding or context to answer the user's query?"
            )

        try:
            response = await self._client.system_one(
                state=state,
                questions=questions,
            )

            filtered_chunks = []
            for i, chunk in enumerate(chunks):
                noul_result = response.nouls.get(f"relevant_{i}")
                if noul_result is None:
                    continue
                prob = float(getattr(noul_result, "noul", 0.0))
                if prob >= self.threshold:
                    logger.debug("Chunk passed", index=i, probability=prob)
                    chunk_copy = dict(chunk)
                    chunk_copy["grounding_score"] = prob
                    filtered_chunks.append(chunk_copy)
                else:
                    logger.debug("Chunk filtered out", index=i, probability=prob)

            if not filtered_chunks:
                logger.warning("All candidate chunks failed grounding threshold")
                return GuardrailResult(chunks=[], low_grounding=True)

            logger.info(
                "Filtering complete",
                original_count=len(chunks),
                filtered_count=len(filtered_chunks),
            )
            return GuardrailResult(chunks=filtered_chunks, low_grounding=False)
        except Exception as e:
            logger.warning(
                "context_guardrail_system_one_failed_fallback_to_candidate_chunks",
                error=str(e),
            )
            fallback_chunks: list[dict[str, Any]] = []
            for c in chunks:
                c_copy = dict(c)
                c_copy["grounding_score"] = float(c.get("rerank_score", c.get("score", 1.0)))
                fallback_chunks.append(c_copy)
            return GuardrailResult(chunks=fallback_chunks, low_grounding=False)

    async def aclose(self) -> None:
        """Closes the underlying TypeSafe AI client."""
        await self._client.aclose()
        logger.debug("ContextGuardrail closed")

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()
