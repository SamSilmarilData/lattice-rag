from __future__ import annotations

import os
from typing import Any
import structlog
from typesafe_sdk import AsyncTypeSafeClient, Noul, Score

from lattice_rag.api.dtos import EvalQueryResult
from lattice_rag.security import SecretStr

logger = structlog.get_logger(__name__)

FAITHFULNESS_CRITERIA = [
    "1: Completely unfaithful; contains significant hallucinations or contradicts the context entirely.",
    "2: Mostly unfaithful; makes multiple unsupported factual claims with minimal context grounding.",
    "3: Moderately faithful; core facts are grounded in context but contains minor unsupported extraneous details.",
    "4: Mostly faithful; nearly all claims are supported by context with only trivial semantic extrapolations.",
    "5: Completely faithful; every assertion and fact in the answer is directly and fully grounded in the context.",
]

CONTEXT_PRECISION_CRITERIA = [
    "1: Completely irrelevant; context contains no useful facts or entities to answer the query.",
    "2: Low precision; context contains isolated keyword mentions but misses core relational facts.",
    "3: Moderate precision; contains partial relevant facts buried among substantial tangential noise.",
    "4: High precision; contains almost all necessary facts and relationships with minimal extraneous context.",
    "5: Ideal precision; compact, exact factual grounding directly addressing all query requirements.",
]

ANSWER_RELEVANCE_CRITERIA = [
    "1: Completely non-responsive; fails to address the user's question.",
    "2: Poor relevance; addresses adjacent topics while ignoring the specific question asked.",
    "3: Partially relevant; partially answers the query but leaves key parts unresolved or wanders.",
    "4: Highly relevant; directly and clearly answers the question with minor gaps.",
    "5: Perfectly relevant; complete, direct, concise, and precisely tailored to the user's inquiry.",
]

CONTRADICTION_INSTRUCTIONS = (
    "Does the generated answer make any claim, fact, number, or assertion that "
    "directly contradicts the facts stated in the source context?"
)


class JevEvaluator:
    """TypeSafe AI Jev System One evaluation engine.

    Evaluates RAG Triad dimensions (Faithfulness, Context Precision, Answer Relevance)
    using calibrated expected-value Score primitives and applies a boolean Noul contradiction veto.
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
    ) -> EvalQueryResult:
        """Evaluate a single query-answer-context tuple against ground truth."""
        # Format context chunks
        context_parts: list[str] = []
        for i, c in enumerate(retrieved_chunks):
            if isinstance(c, dict):
                text = c.get("text", "")
            else:
                text = str(c)
            if text:
                context_parts.append(f"[{i + 1}] {text}")

        context_text = "\n\n".join(context_parts) if context_parts else "NO_CONTEXT_RETRIEVED"

        state = {
            "query": query,
            "context": context_text,
            "answer": generated_answer,
            "ground_truth": ground_truth,
        }

        questions = {
            "faithfulness": Score(
                instructions="Rate the factual faithfulness of the answer based strictly on the provided context.",
                criteria=FAITHFULNESS_CRITERIA,
            ),
            "context_precision": Score(
                instructions="Rate how precisely the retrieved context contains the exact facts necessary to answer the query according to the ground truth.",
                criteria=CONTEXT_PRECISION_CRITERIA,
            ),
            "answer_relevance": Score(
                instructions="Rate how directly, completely, and appropriately the answer addresses the specific question asked in the query.",
                criteria=ANSWER_RELEVANCE_CRITERIA,
            ),
            "contradiction": Noul(
                instructions=CONTRADICTION_INSTRUCTIONS,
            ),
        }

        response = await self._client.system_one(state=state, questions=questions)

        raw_faith = response.scores["faithfulness"].score
        raw_prec = response.scores["context_precision"].score
        raw_rel = response.scores["answer_relevance"].score
        contradiction_prob = response.nouls["contradiction"].noul

        faith_score = self._normalize_score(raw_faith)
        prec_score = self._normalize_score(raw_prec)
        rel_score = self._normalize_score(raw_rel)

        # Hard Zero Contradiction Veto:
        # If contradiction probability >= 0.40, force faithfulness to 0.0
        vetoed = False
        if contradiction_prob >= 0.40:
            logger.warning(
                "factual_contradiction_veto_triggered",
                query=query,
                contradiction_prob=contradiction_prob,
                previous_faithfulness=faith_score,
            )
            faith_score = 0.0
            vetoed = True

        passed = (faith_score >= 0.70) and (prec_score >= 0.65) and (rel_score >= 0.70) and not vetoed

        logger.info(
            "jev_evaluation_completed",
            query=query,
            faithfulness=round(faith_score, 4),
            context_precision=round(prec_score, 4),
            answer_relevance=round(rel_score, 4),
            contradiction_prob=round(contradiction_prob, 4),
            passed=passed,
        )

        return EvalQueryResult(
            query=query,
            faithfulness=round(faith_score, 4),
            context_precision=round(prec_score, 4),
            answer_relevance=round(rel_score, 4),
            passed=passed,
        )

    async def aclose(self) -> None:
        """Close underlying TypeSafe client if owned."""
        if self._owns_client and hasattr(self._client, "aclose"):
            await self._client.aclose()
            logger.debug("JevEvaluator client closed")

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()
