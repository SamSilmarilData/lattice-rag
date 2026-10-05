"""Hermetic unit tests for EvalRunner regression gating and dataset processing."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from lattice_rag.api.dtos import EvalQueryResult, EvalRunResponse
from lattice_rag.eval.runner import EvalRunner


@pytest.fixture
def mock_dataset_and_baseline(tmp_path: Path):
    """Create sample dataset and baseline files."""
    dataset = {
        "version": "1.0.0",
        "corpus": [],
        "queries": [
            {
                "id": "q1",
                "query": "Test query 1",
                "ground_truth": "Expected 1",
            },
            {
                "id": "q2",
                "query": "Test query 2",
                "ground_truth": "Expected 2",
            },
        ],
    }
    baseline = {
        "version": "0.4.0",
        "metrics": {
            "faithfulness": 0.85,
            "context_precision": 0.80,
            "answer_relevance": 0.88,
        },
        "gate_thresholds": {
            "max_regression_delta": 0.03,
            "min_query_floor": 0.50,
        },
    }

    ds_path = tmp_path / "test_ds.json"
    base_path = tmp_path / "test_base.json"

    ds_path.write_text(json.dumps(dataset), encoding="utf-8")
    base_path.write_text(json.dumps(baseline), encoding="utf-8")

    return ds_path, base_path


@pytest.mark.asyncio
async def test_eval_runner_passes_gate(mock_dataset_and_baseline):
    """Verify EvalRunner marks passed_gate=True when metrics are at or above baseline."""
    ds_path, base_path = mock_dataset_and_baseline

    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="Valid answer",
        filtered_chunks=[{"text": "Valid context"}],
    )
    mock_orch.semantic_cache = None

    mock_evaluator = AsyncMock()
    # High scores above baseline
    mock_evaluator.evaluate_query.return_value = EvalQueryResult(
        query="q",
        faithfulness=0.90,
        context_precision=0.85,
        answer_relevance=0.92,
        passed=True,
    )

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    resp = await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path, use_temp_db=False)

    assert isinstance(resp, EvalRunResponse)
    assert resp.total_queries == 2
    assert resp.passed_gate is True
    assert resp.mean_faithfulness == 0.90
    assert resp.regression_delta >= 0.0


@pytest.mark.asyncio
async def test_eval_runner_fails_on_regression(mock_dataset_and_baseline):
    """Verify EvalRunner marks passed_gate=False when metric drop exceeds 0.03."""
    ds_path, base_path = mock_dataset_and_baseline

    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="Regressed answer",
        filtered_chunks=[{"text": "Context"}],
    )
    mock_orch.semantic_cache = None

    mock_evaluator = AsyncMock()
    # Faithfulness 0.70 is a drop of 0.15 compared to baseline 0.85 (> 0.03 drop)
    mock_evaluator.evaluate_query.return_value = EvalQueryResult(
        query="q",
        faithfulness=0.70,
        context_precision=0.82,
        answer_relevance=0.88,
        passed=False,
    )

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    resp = await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path, use_temp_db=False)

    assert resp.passed_gate is False
    assert resp.regression_delta < -0.03


@pytest.mark.asyncio
async def test_eval_runner_fails_on_floor_violation(mock_dataset_and_baseline):
    """Verify EvalRunner fails when an individual query drops below min_query_floor (0.50)."""
    ds_path, base_path = mock_dataset_and_baseline

    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(answer="Ans", filtered_chunks=[])
    mock_orch.semantic_cache = None

    mock_evaluator = AsyncMock()
    # First query 0.95, but second query 0.40 (below 0.50 floor)
    mock_evaluator.evaluate_query.side_effect = [
        EvalQueryResult(query="q1", faithfulness=0.95, context_precision=0.90, answer_relevance=0.95, passed=True),
        EvalQueryResult(query="q2", faithfulness=0.40, context_precision=0.90, answer_relevance=0.95, passed=False),
    ]

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    resp = await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path, use_temp_db=False)

    # Even though mean faithfulness is (0.95+0.40)/2 = 0.675, floor was violated
    assert resp.passed_gate is False


@pytest.mark.asyncio
async def test_eval_runner_missing_dataset_raises():
    """Verify FileNotFoundError when dataset file is missing."""
    runner = EvalRunner(orchestrator=AsyncMock(), evaluator=AsyncMock())
    with pytest.raises(FileNotFoundError):
        await runner.run_evaluation(dataset_path="nonexistent.json", baseline_path="eval_baseline.json")
