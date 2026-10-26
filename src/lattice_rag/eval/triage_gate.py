from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import msgspec
import structlog
from typesafe_sdk import AsyncTypeSafeClient, Noul, Score

from lattice_rag.api.dtos import EvalQueryResult
from lattice_rag.eval.gate import (
    DEFAULT_PASS_THRESHOLDS,
    METRICS,
    PassThresholds,
    query_passes,
)
from lattice_rag.generation.prompt import graph_triples
from lattice_rag.security import SecretStr, query_fingerprint

logger = structlog.get_logger(__name__)

FAITHFULNESS_CRITERIA = [
    "1: Completely unfaithful; contains significant hallucinations or contradicts `context` entirely.",
    "2: Mostly unfaithful; makes multiple unsupported factual claims with minimal `context` grounding.",
    "3: Moderately faithful; core facts are grounded in `context` but contains minor unsupported extraneous details.",
    "4: Mostly faithful; nearly all claims are supported by `context` with only trivial semantic extrapolations.",
    "5: Completely faithful; every assertion and fact in the `answer` is directly and fully grounded in the `context`.",
]

CONTEXT_PRECISION_CRITERIA = [
    "1: Completely irrelevant; `context` contains no useful facts or entities to answer the `query`.",
    "2: Low precision; `context` contains isolated keyword mentions but misses core relational facts.",
    "3: Moderate precision; contains partial relevant facts buried among substantial tangential noise.",
    "4: High precision; contains almost all necessary facts and relationships with minimal extraneous `context`.",
    "5: Ideal precision; compact, exact factual grounding directly addressing all `query` requirements.",
]

ANSWER_RELEVANCE_CRITERIA = [
    "1: Completely non-responsive; fails to address the user's `query`.",
    "2: Poor relevance; addresses adjacent topics while ignoring the specific `query` asked.",
    "3: Partially relevant; partially answers the `query` but leaves key parts unresolved or wanders.",
    "4: Highly relevant; directly and clearly answers the `query` with minor gaps.",
    "5: Perfectly relevant; complete, direct, concise, and precisely tailored to the user's `query`.",
]

ANSWER_CORRECTNESS_CRITERIA = [
    "1: Incorrect; the `answer` contradicts the `reference_answer` or answers a different question.",
    "2: Mostly incorrect; the `answer` gets the central fact of the `reference_answer` wrong but shares minor details.",
    "3: Partially correct; the `answer` agrees with part of the `reference_answer` but misses or misstates a key fact.",
    "4: Mostly correct; the `answer` agrees with every key fact of the `reference_answer`, with minor omissions.",
    "5: Fully correct; the `answer` states the same facts and conclusion as the `reference_answer`.",
]

CONTRADICTION_INSTRUCTIONS = (
    "Does the generated `answer` make any claim, fact, number, or assertion that "
    "directly contradicts the facts stated in the source `context`?"
)

_METRIC_QUESTIONS: dict[str, Score] = {
    "faithfulness": Score(
        instructions="Rate the factual faithfulness of the `answer` based strictly on the provided `context`.",
        criteria=FAITHFULNESS_CRITERIA,
    ),
    "context_precision": Score(
        instructions="Rate how precisely the retrieved `context` contains the exact facts necessary to answer the `query` according to the `reference_answer`.",
        criteria=CONTEXT_PRECISION_CRITERIA,
    ),
    "answer_relevance": Score(
        instructions="Rate how directly, completely, and appropriately the `answer` addresses the specific question asked in the `query`.",
        criteria=ANSWER_RELEVANCE_CRITERIA,
    ),
    "answer_correctness": Score(
        instructions=(
            "Rate how well the facts and conclusion of the `answer` agree with the "
            "`reference_answer` for the `query`, regardless of what the `context` says."
        ),
        criteria=ANSWER_CORRECTNESS_CRITERIA,
    ),
}

# Vetoes faithfulness, so it is asked only when faithfulness is judged
_CONTRADICTION_QUESTION = Noul(instructions=CONTRADICTION_INSTRUCTIONS)
CONTRADICTION_VETO = 0.40


class JevEvaluator:
    """TypeSafe AI Jev System One evaluation engine.

    Judges faithfulness, context precision, answer relevance and Answer Correctness
    (agreement with the reference answer) with calibrated expected-value Score
    primitives, and applies a Noul contradiction veto to faithfulness.
    """

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        client: AsyncTypeSafeClient | None = None,
    ) -> None:
        """Initialize JevEvaluator with client injection or environment key."""
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            raw_key: str | None = None
            if isinstance(api_key, SecretStr):
                raw_key = api_key.get_secret_value()
            elif isinstance(api_key, str):
                raw_key = api_key
            else:
                raw_key = os.environ.get("TYPESAFE_API_KEY")

            if not raw_key:
                raise RuntimeError(
                    "TYPESAFE_API_KEY is required to initialize JevEvaluator. "
                    "Please configure it in .env or provide it directly."
                )

            self._client = AsyncTypeSafeClient(api_key=raw_key)
            self._owns_client = True

    def _normalize_score(self, score_val: float, num_levels: int = 5) -> float:
        """Normalize discrete level expected value to continuous [0.0, 1.0] range."""
        max_idx = float(num_levels - 1)
        normalized = score_val / max_idx
        return max(0.0, min(1.0, normalized))

    async def evaluate_query(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]] | list[str],
        generated_answer: str,
        ground_truth: str,
        metrics: Sequence[str] = METRICS,
        thresholds: PassThresholds = DEFAULT_PASS_THRESHOLDS,
        graph_context: dict[str, Any] | None = None,
    ) -> EvalQueryResult:
        """Judge one answer on ``metrics``; ``ground_truth`` is the Gold Set reference answer.

        ``graph_context`` is the subgraph the generator was given; its triples are
        judged as part of the context.

        Metrics not in ``metrics`` are left ``None``. Pass/fail applies ``thresholds``
        to every judged metric.
        """
        unknown = set(metrics) - set(METRICS)
        if unknown or not metrics:
            raise ValueError(f"Unknown or empty metrics: {sorted(unknown)}")

        # Format context chunks
        context_parts: list[str] = []
        for i, c in enumerate(retrieved_chunks):
            text = c.get("text", "") if isinstance(c, dict) else str(c)
            if text:
                context_parts.append(f"[{i + 1}] {text}")

        if graph_context:
            context_parts.append("Graph triples:\n" + "\n".join(graph_triples(graph_context)))

        context_text = "\n\n".join(context_parts) if context_parts else "NO_CONTEXT_RETRIEVED"

        state = {
            "query": query,
            "context": context_text,
            "answer": generated_answer,
            "reference_answer": ground_truth,
        }
        questions: dict[str, Any] = {m: _METRIC_QUESTIONS[m] for m in METRICS if m in metrics}
        if "faithfulness" in questions:
            questions["contradiction"] = _CONTRADICTION_QUESTION

        response = await self._client.system_one(state=state, questions=questions)

        scores: dict[str, float | None] = dict.fromkeys(METRICS)
        for m in metrics:
            scores[m] = self._normalize_score(response.scores[m].score)

        # Hard Zero Contradiction Veto: a contradiction of the context zeroes faithfulness
        contradiction_prob: float | None = None
        if "contradiction" in questions:
            contradiction_prob = response.nouls["contradiction"].noul
            if contradiction_prob >= CONTRADICTION_VETO:
                logger.warning(
                    "factual_contradiction_veto_triggered",
                    **query_fingerprint(query),
                    contradiction_prob=contradiction_prob,
                    previous_faithfulness=scores["faithfulness"],
                )
                scores["faithfulness"] = 0.0

        rounded = {m: (round(v, 4) if v is not None else None) for m, v in scores.items()}
        result = EvalQueryResult(
            query=query,
            passed=False,
            faithfulness=rounded["faithfulness"],
            context_precision=rounded["context_precision"],
            answer_relevance=rounded["answer_relevance"],
            answer_correctness=rounded["answer_correctness"],
        )
        result = msgspec.structs.replace(result, passed=query_passes(result, thresholds))

        logger.info(
            "jev_evaluation_completed",
            **query_fingerprint(query),
            **rounded,
            contradiction_prob=contradiction_prob,
            passed=result.passed,
        )
        return result

    async def aclose(self) -> None:
        """Close underlying TypeSafe client if owned."""
        if self._owns_client and hasattr(self._client, "aclose"):
            await self._client.aclose()
            logger.debug("JevEvaluator client closed")

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()
