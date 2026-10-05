from __future__ import annotations

import os

import structlog
from typesafe_sdk import AsyncTypeSafeClient, Noul

logger = structlog.get_logger(__name__)


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
        self.low_grounding_flag: bool = False
        logger.debug("ContextGuardrail initialized", threshold=threshold)


    async def filter_chunks(self, query: str, chunks: list[dict]) -> list[dict]:
        """Filters chunks based on their relevance to the query.

        Args:
            query: The user's query.
            chunks: A list of dictionaries, each containing 'text', 'node_id', and 'score'.

        Returns:
            A filtered list of chunk dictionaries that meet the relevance threshold.
            If all chunks fail the threshold, injects a fallback item with low_grounding_flag=True.
        """
        if not chunks:
            self.low_grounding_flag = False
            return []

        logger.info("Filtering chunks", chunk_count=len(chunks))

        # Build a single state object mapping chunk indices to their text
        state = {"query": query}
        for i, chunk in enumerate(chunks):
            state[f"chunk_{i}"] = chunk.get("text", "")

        # Build a Noul question for each chunk
        questions = {}
        for i in range(len(chunks)):
            questions[f"relevant_{i}"] = Noul(
                instructions=f"Does chunk_{i} provide necessary and directly relevant factual grounding to answer the user's query?"
            )

        response = await self._client.system_one(
            state=state,
            questions=questions,
        )

        filtered_chunks = []
        for i, chunk in enumerate(chunks):
            noul_result = response.nouls[f"relevant_{i}"]
            prob = noul_result.noul
            if prob >= self.threshold:
                logger.debug("Chunk passed", index=i, probability=prob)
                chunk_copy = dict(chunk)
                chunk_copy["low_grounding_flag"] = False
                chunk_copy["grounding_score"] = prob
                filtered_chunks.append(chunk_copy)
            else:
                logger.debug("Chunk filtered out", index=i, probability=prob)

        if not filtered_chunks:
            self.low_grounding_flag = True
            logger.warning("All candidate chunks failed grounding threshold, low_grounding_flag activated")
            return [
                {
                    "node_id": -1,
                    "text": "INSUFFICIENT_EVIDENCE: All candidate chunks failed relevance grounding.",
                    "score": 0.0,
                    "grounding_score": 0.0,
                    "low_grounding_flag": True,
                }
            ]

        self.low_grounding_flag = False
        logger.info("Filtering complete", original_count=len(chunks), filtered_count=len(filtered_chunks))
        return filtered_chunks

    async def aclose(self) -> None:
        """Closes the underlying TypeSafe AI client."""
        await self._client.aclose()
        logger.debug("ContextGuardrail closed")

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()

