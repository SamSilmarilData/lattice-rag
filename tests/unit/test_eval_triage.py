"""Hermetic unit tests for JevEvaluator and RAG Triad scoring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.api.dtos import EvalQueryResult
from lattice_rag.eval.triage_gate import JevEvaluator


@pytest.mark.asyncio
async def test_jev_evaluator_missing_key_raises():
    """Verify JevEvaluator raises RuntimeError when no API key is provided."""
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY is required"):
        JevEvaluator(api_key="", client=None)


@pytest.mark.asyncio
async def test_jev_evaluator_score_normalization():
    """Verify 5-level Score normalization to [0.0, 1.0]."""
    mock_client = AsyncMock()
    evaluator = JevEvaluator(client=mock_client)

    # 0.0 -> 0.0
    assert evaluator._normalize_score(0.0) == 0.0
    # 4.0 -> 1.0
    assert evaluator._normalize_score(4.0) == 1.0
    # 2.0 -> 0.5
    assert evaluator._normalize_score(2.0) == 0.5
    # 3.41 -> 0.8525
    assert round(evaluator._normalize_score(3.41), 4) == 0.8525


@pytest.mark.asyncio
async def test_jev_evaluator_evaluate_query_success():
    """Verify evaluate_query successfully scores Faithfulness, Precision, and Relevance."""
    mock_client = AsyncMock()

    mock_resp = MagicMock()
    mock_resp.scores = {
        "faithfulness": MagicMock(score=3.6, confidence=0.88),
        "context_precision": MagicMock(score=3.4, confidence=0.82),
        "answer_relevance": MagicMock(score=3.8, confidence=0.91),
        "answer_correctness": MagicMock(score=3.6, confidence=0.9),
    }
    mock_resp.nouls = {
        "contradiction": MagicMock(noul=0.05),
    }
    mock_client.system_one.return_value = mock_resp

    evaluator = JevEvaluator(client=mock_client)

    result = await evaluator.evaluate_query(
        query="What is LatticeDB?",
        retrieved_chunks=[{"text": "LatticeDB is an embedded graph database."}],
        generated_answer="LatticeDB is an embedded property-graph database.",
        ground_truth="LatticeDB is an embedded graph database.",
    )

    assert isinstance(result, EvalQueryResult)
    assert result.faithfulness == 0.9  # 3.6 / 4.0
    assert result.context_precision == 0.85  # 3.4 / 4.0
    assert result.answer_relevance == 0.95  # 3.8 / 4.0
    assert result.answer_correctness == 0.9  # 3.6 / 4.0
    assert result.passed is True


@pytest.mark.asyncio
async def test_jev_evaluator_contradiction_veto():
    """Verify contradiction Noul veto forces faithfulness to 0.0 when P(contradiction) >= 0.40."""
    mock_client = AsyncMock()

    mock_resp = MagicMock()
    # High score from Score primitive, but high contradiction probability from Noul
    mock_resp.scores = {
        "faithfulness": MagicMock(score=3.8, confidence=0.9),
        "context_precision": MagicMock(score=3.5, confidence=0.8),
        "answer_relevance": MagicMock(score=3.9, confidence=0.95),
        "answer_correctness": MagicMock(score=3.9, confidence=0.9),
    }
    mock_resp.nouls = {
        "contradiction": MagicMock(noul=0.75),  # Contradiction flagged!
    }
    mock_client.system_one.return_value = mock_resp

    evaluator = JevEvaluator(client=mock_client)

    result = await evaluator.evaluate_query(
        query="What search does LatticeDB support?",
        retrieved_chunks=[{"text": "LatticeDB supports HNSW vector and BM25 searches."}],
        generated_answer="LatticeDB does not support vector searches, only relational Cypher.",
        ground_truth="LatticeDB supports HNSW and BM25.",
    )

    # Faithfulness should be forced to 0.0 by the veto
    assert result.faithfulness == 0.0
    assert result.passed is False


def _judge_response(scores: dict[str, float], contradiction: float = 0.05) -> MagicMock:
    resp = MagicMock()
    resp.scores = {k: MagicMock(score=v, confidence=0.9) for k, v in scores.items()}
    resp.nouls = {"contradiction": MagicMock(noul=contradiction)}
    return resp


@pytest.mark.asyncio
async def test_grounded_but_wrong_answer_scores_low_on_correctness():
    """An answer faithful to off-target context but disagreeing with the reference fails."""
    mock_client = AsyncMock()
    mock_client.system_one.return_value = _judge_response(
        {
            "faithfulness": 4.0,
            "context_precision": 3.6,
            "answer_relevance": 3.8,
            "answer_correctness": 0.4,
        }
    )
    evaluator = JevEvaluator(client=mock_client)

    result = await evaluator.evaluate_query(
        query="What is the default reranker headroom?",
        retrieved_chunks=[{"text": "The default top_k is 5."}],
        generated_answer="The default is 5.",
        ground_truth="The reranker headroom defaults to 10 extra candidates.",
    )

    assert result.faithfulness == 1.0
    assert result.answer_correctness == 0.1
    assert result.passed is False

    call = mock_client.system_one.await_args
    assert call.kwargs["state"]["reference_answer"] == (
        "The reranker headroom defaults to 10 extra candidates."
    )
    assert "answer_correctness" in call.kwargs["questions"]


@pytest.mark.asyncio
async def test_only_requested_metrics_are_judged():
    """A chitchat query is judged on relevance and correctness only; no context metrics."""
    mock_client = AsyncMock()
    mock_client.system_one.return_value = _judge_response(
        {"answer_relevance": 4.0, "answer_correctness": 4.0}
    )
    evaluator = JevEvaluator(client=mock_client)

    result = await evaluator.evaluate_query(
        query="hello",
        retrieved_chunks=[],
        generated_answer="Hi there!",
        ground_truth="Hi there!",
        metrics=("answer_relevance", "answer_correctness"),
    )

    questions = mock_client.system_one.await_args.kwargs["questions"]
    assert set(questions) == {"answer_relevance", "answer_correctness"}
    assert result.faithfulness is None
    assert result.context_precision is None
    assert result.answer_correctness == 1.0
    assert result.passed is True


@pytest.mark.asyncio
async def test_pass_thresholds_are_supplied_by_the_caller():
    """Per-query pass thresholds come from the caller (the baseline file), not the judge."""
    from lattice_rag.eval.gate import PassThresholds

    mock_client = AsyncMock()
    mock_client.system_one.return_value = _judge_response(
        {
            "faithfulness": 3.2,
            "context_precision": 3.2,
            "answer_relevance": 3.2,
            "answer_correctness": 3.2,
        }
    )
    evaluator = JevEvaluator(client=mock_client)
    strict = PassThresholds(
        faithfulness=0.9, context_precision=0.9, answer_relevance=0.9, answer_correctness=0.9
    )

    result = await evaluator.evaluate_query(
        query="q", retrieved_chunks=["c"], generated_answer="a", ground_truth="r", thresholds=strict
    )

    assert result.answer_correctness == 0.8
    assert result.passed is False


@pytest.mark.asyncio
async def test_contradiction_question_allows_following_one_of_disagreeing_sources():
    """Gold Set 3.0.0's conflicting_sources and ambiguity Cases put passages that disagree
    into one context. An answer that follows one of them, or reports the disagreement, must
    not count as contradicting the context, or every correct resolution would be vetoed."""
    mock_client = AsyncMock()
    mock_client.system_one.return_value = _judge_response(
        dict.fromkeys(
            ("faithfulness", "context_precision", "answer_relevance", "answer_correctness"), 4.0
        )
    )
    evaluator = JevEvaluator(client=mock_client)

    await evaluator.evaluate_query(
        query="Does a container need 1 GB or 2 GB?",
        retrieved_chunks=[{"text": "Target: 1 GB."}, {"text": "Requires at least 2 GB."}],
        generated_answer="At least 2 GB; one document's 1 GB target is out of date.",
        ground_truth="At least 2 GB.",
    )

    question = mock_client.system_one.call_args.kwargs["questions"]["contradiction"]
    instructions = question.instructions.lower()
    assert "disagree" in instructions
    assert "not a contradiction" in instructions
