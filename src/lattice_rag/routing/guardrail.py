from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import structlog
from typesafe_sdk import AsyncTypeSafeClient, Noul

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class GuardrailResult:
    """Per-call guardrail verdict: the chunks that passed and whether grounding is too weak."""

    chunks: list[dict[str, Any]]
    low_grounding: bool
    # False when Jev Noul could not run and the candidates were passed through unchecked
    verified: bool = True


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
            return GuardrailResult(chunks=fallback_chunks, low_grounding=False, verified=False)

    async def aclose(self) -> None:
        """Closes the underlying TypeSafe AI client."""
        await self._client.aclose()
        logger.debug("ContextGuardrail closed")

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()


class ChunkGuardrail(Protocol):
    """What the retrieval pipeline needs from a guardrail."""

    async def filter_chunks(self, query: str, chunks: list[dict[str, Any]]) -> GuardrailResult: ...

    async def aclose(self) -> None: ...


class RerankScoreGate:
    """Skips the Noul guardrail for chunks the cross-encoder already rates clearly relevant.

    Chunks with ``rerank_score >= min_score`` are kept without a Noul question; the rest go
    to the wrapped guardrail. Output keeps the reranker's order. Off by default
    (``GUARDRAIL_SKIP_RERANK_SCORE`` unset); enabling it requires Live Tier results showing
    no quality loss (docs/production-readiness.md).
    """

    def __init__(self, guardrail: ChunkGuardrail, min_score: float) -> None:
        self.guardrail = guardrail
        self.min_score = min_score

    def _clears(self, chunk: dict[str, Any]) -> bool:
        score = chunk.get("rerank_score")
        return score is not None and float(score) >= self.min_score

    async def filter_chunks(self, query: str, chunks: list[dict[str, Any]]) -> GuardrailResult:
        to_check = [c for c in chunks if not self._clears(c)]
        skipped = len(chunks) - len(to_check)
        logger.debug("rerank_score_gate", skipped=skipped, checked=len(to_check))
        if not to_check:
            return GuardrailResult(chunks=list(chunks), low_grounding=False)

        verdict = await self.guardrail.filter_chunks(query, to_check)
        # The guardrail returns its survivors as copies, in input order: merge them back
        # between the skipped chunks so the reranker's order holds.
        survivors = verdict.chunks
        kept: list[dict[str, Any]] = []
        j = 0
        for chunk in chunks:
            if self._clears(chunk):
                kept.append(chunk)
            elif j < len(survivors) and survivors[j].get("node_id") == chunk.get("node_id"):
                kept.append(survivors[j])
                j += 1
        return GuardrailResult(
            chunks=kept,
            low_grounding=verdict.low_grounding and skipped == 0,
            verified=verdict.verified,
        )

    async def aclose(self) -> None:
        await self.guardrail.aclose()

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()


def with_rerank_skip(
    guardrail: ContextGuardrail | None, min_score: float | None
) -> ContextGuardrail | RerankScoreGate | None:
    """Wrap ``guardrail`` in a :class:`RerankScoreGate` when a skip score is configured."""
    if guardrail is None or min_score is None:
        return guardrail
    return RerankScoreGate(guardrail, min_score)
