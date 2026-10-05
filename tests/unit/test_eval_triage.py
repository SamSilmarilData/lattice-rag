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
