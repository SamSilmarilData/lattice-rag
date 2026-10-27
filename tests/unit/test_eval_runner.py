"""Hermetic unit tests for EvalRunner regression gating and dataset processing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.api.dtos import EvalQueryResult, EvalRunResponse
from lattice_rag.eval.runner import EvalRunner
from tests.unit.pool_helpers import pool_of

pytestmark = pytest.mark.usefixtures("small_gold_sets")

PASS_THRESHOLDS = {
    "faithfulness": 0.70,
    "context_precision": 0.65,
    "answer_relevance": 0.70,
    "answer_correctness": 0.70,
}


def _baseline(**metrics: float) -> dict:
    return {
        "version": "3.0.0",
        "metrics": metrics
        or {"faithfulness": 0.85, "context_precision": 0.80, "answer_relevance": 0.88},
        "gate_thresholds": {
            "max_regression_delta": 0.03,
            "min_query_floor": 0.50,
            "pass_thresholds": PASS_THRESHOLDS,
        },
    }


def _result(query: str = "q", score: float = 0.9, **overrides: float) -> EvalQueryResult:
    scores = dict.fromkeys(
        ("faithfulness", "context_precision", "answer_relevance", "answer_correctness"), score
    )
    scores.update(overrides)
    return EvalQueryResult(query=query, passed=True, **scores)


@pytest.fixture
def mock_dataset_and_baseline(tmp_path: Path):
    """Create sample dataset and baseline files."""
    dataset = {
        "version": "3.0.0",
        "domain": "test",
        "corpus": [],
        "queries": [
            {
                "id": "q1",
                "query": "Test query 1",
                "slice": "unanswerable",
                "case": "direct",
                "expected_route": "hybrid",
                "reference_answer": "Expected 1",
                "gold_passages": [],
            },
            {
                "id": "q2",
                "query": "Test query 2",
                "slice": "unanswerable",
                "case": "direct",
                "expected_route": "hybrid",
                "reference_answer": "Expected 2",
                "gold_passages": [],
            },
        ],
    }
    baseline = _baseline()

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
        degraded=False,
        provider="groq",
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
    resp = await runner.run_evaluation(
        dataset_path=ds_path, baseline_path=base_path, use_temp_db=False
    )

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
        degraded=False,
        provider="groq",
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
    resp = await runner.run_evaluation(
        dataset_path=ds_path, baseline_path=base_path, use_temp_db=False
    )

    assert resp.passed_gate is False
    assert resp.regression_delta < -0.03


@pytest.mark.asyncio
async def test_eval_runner_fails_on_floor_violation(mock_dataset_and_baseline):
    """Verify EvalRunner fails when an individual query drops below min_query_floor (0.50)."""
    ds_path, base_path = mock_dataset_and_baseline

    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="Ans", filtered_chunks=[], degraded=False, provider="groq"
    )
    mock_orch.semantic_cache = None

    mock_evaluator = AsyncMock()
    # First query 0.95, but second query 0.40 (below 0.50 floor)
    mock_evaluator.evaluate_query.side_effect = [
        EvalQueryResult(
            query="q1",
            faithfulness=0.95,
            context_precision=0.90,
            answer_relevance=0.95,
            passed=True,
        ),
        EvalQueryResult(
            query="q2",
            faithfulness=0.40,
            context_precision=0.90,
            answer_relevance=0.95,
            passed=False,
        ),
    ]

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    resp = await runner.run_evaluation(
        dataset_path=ds_path, baseline_path=base_path, use_temp_db=False
    )

    # Even though mean faithfulness is (0.95+0.40)/2 = 0.675, floor was violated
    assert resp.passed_gate is False


@pytest.mark.asyncio
async def test_eval_runner_missing_dataset_raises():
    """Verify FileNotFoundError when dataset file is missing."""
    runner = EvalRunner(orchestrator=AsyncMock(), evaluator=AsyncMock())
    with pytest.raises(FileNotFoundError):
        await runner.run_evaluation(
            dataset_path="nonexistent.json", baseline_path="eval_baseline.json"
        )


@pytest.mark.asyncio
async def test_eval_runner_bypasses_and_preserves_live_semantic_cache(mock_dataset_and_baseline):
    """Eval must neither clear the live Tier 1 cache nor read/write it."""
    ds_path, base_path = mock_dataset_and_baseline

    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="a", filtered_chunks=[], degraded=False, provider="groq"
    )
    mock_orch.semantic_cache = MagicMock()

    mock_evaluator = AsyncMock()
    mock_evaluator.evaluate_query.return_value = EvalQueryResult(
        query="q", faithfulness=0.9, context_precision=0.9, answer_relevance=0.9, passed=True
    )

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path, use_temp_db=False)

    mock_orch.semantic_cache.clear.assert_not_called()
    for call in mock_orch.run.await_args_list:
        assert call.kwargs.get("use_cache") is False


@pytest.mark.asyncio
async def test_orchestrator_run_without_cache_skips_lookup_and_store():
    """run(use_cache=False) neither consults nor populates Tier 1."""
    import numpy as np

    from lattice_rag.orchestration.graph import RAGOrchestrator
    from lattice_rag.routing.router import RouteDecision
    from lattice_rag.storage.db import SubgraphResult

    sem_cache = MagicMock()
    sem_cache.get_exact = MagicMock(return_value={"answer": "stale"})
    sem_cache.get = AsyncMock(return_value={"answer": "stale"})
    router = MagicMock()
    router.route_query = AsyncMock(return_value=RouteDecision(route="hybrid", confidence=0.9))
    pipeline = MagicMock()
    pipeline.execute = AsyncMock(
        return_value=MagicMock(
            final_chunks=[{"text": "ctx"}],
            graph_context=SubgraphResult(nodes=[], edges=[]),
            low_grounding=False,
        )
    )
    groq = MagicMock()
    groq.synthesize = AsyncMock(return_value="fresh")
    embed = MagicMock()
    embed.embed_query = MagicMock(return_value=np.ones(8, dtype=np.float32))

    orch = RAGOrchestrator(
        router=router,
        semantic_cache=sem_cache,
        fallback_cache=MagicMock(),
        retrieval_pipeline=pipeline,
        pool=pool_of(groq=groq, gemini=MagicMock()),
        chitchat=MagicMock(),
        embedding_service=embed,
    )
    state = await orch.run("q", use_cache=False)

    assert state.answer == "fresh"
    assert state.cache_hit is False
    sem_cache.get_exact.assert_not_called()
    sem_cache.get.assert_not_called()
    sem_cache.put.assert_not_called()


@pytest.mark.asyncio
async def test_eval_runner_records_evaluator_failure_instead_of_aborting(mock_dataset_and_baseline):
    """A Jev timeout on one query fails that query, not the whole run."""
    ds_path, base_path = mock_dataset_and_baseline

    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="a", filtered_chunks=[], degraded=False, provider="groq"
    )
    mock_orch.semantic_cache = None

    good = EvalQueryResult(
        query="Test query 2",
        faithfulness=0.9,
        context_precision=0.9,
        answer_relevance=0.9,
        passed=True,
    )
    mock_evaluator = AsyncMock()
    mock_evaluator.evaluate_query.side_effect = [TimeoutError("Jev timed out"), good]

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    resp = await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path)

    assert resp.total_queries == 2
    failed = resp.results[0]
    assert failed.query == "Test query 1"
    assert failed.passed is False
    assert failed.error == "judge_error"
    assert failed.faithfulness is None  # not judged, so not scored
    assert resp.failed_queries == 1
    assert resp.passed_gate is False


@pytest.mark.asyncio
async def test_eval_runner_judges_against_the_reference_answer(mock_dataset_and_baseline):
    """The Gold Set's reference answer is what the judge compares the answer to."""
    ds_path, base_path = mock_dataset_and_baseline

    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="a", filtered_chunks=[], degraded=False, provider="groq"
    )
    mock_evaluator = AsyncMock()
    mock_evaluator.evaluate_query.return_value = EvalQueryResult(
        query="q", faithfulness=0.9, context_precision=0.9, answer_relevance=0.9, passed=True
    )

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path)

    refs = {c.kwargs["ground_truth"] for c in mock_evaluator.evaluate_query.await_args_list}
    assert refs == {"Expected 1", "Expected 2"}


@pytest.mark.asyncio
async def test_eval_runner_rejects_invalid_gold_set_before_running(tmp_path: Path):
    """A malformed Gold Set fails fast with the missing field named; no query is run."""
    from lattice_rag.eval.gold_set import GoldSetError

    ds_path = tmp_path / "bad.json"
    ds_path.write_text(
        json.dumps(
            {
                "version": "3.0.0",
                "domain": "test",
                "corpus": [],
                "queries": [{"id": "q1", "query": "Q", "ground_truth": "old v1 field"}],
            }
        ),
        encoding="utf-8",
    )
    mock_orch = AsyncMock()

    runner = EvalRunner(orchestrator=mock_orch, evaluator=AsyncMock())
    with pytest.raises(GoldSetError, match=r"queries\[0\]"):
        await runner.run_evaluation(dataset_path=ds_path, baseline_path="eval_baseline.json")
    mock_orch.run.assert_not_called()


def _gold_query(qid: str, query: str, slice_name: str, **extra: object) -> dict:
    return {
        "id": qid,
        "query": query,
        "slice": slice_name,
        "case": "direct",
        "expected_route": "hybrid",
        "reference_answer": "R",
        "gold_passages": [],
        **extra,
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_pipeline_error_fails_query_without_judging(mock_dataset_and_baseline):
    """A query whose pipeline raises is reported failed and is never sent to the judge."""
    ds_path, base_path = mock_dataset_and_baseline

    mock_orch = AsyncMock()
    mock_orch.run.side_effect = [
        RuntimeError("groq exploded"),
        MagicMock(answer="fine", filtered_chunks=[], degraded=False, provider="groq"),
    ]
    mock_evaluator = AsyncMock()
    mock_evaluator.evaluate_query.return_value = _result("Test query 2")

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    resp = await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path)

    assert mock_evaluator.evaluate_query.await_count == 1
    assert mock_evaluator.evaluate_query.await_args.kwargs["query"] == "Test query 2"

    failed = resp.results[0]
    assert failed.query == "Test query 1"
    assert failed.passed is False
    assert failed.error == "pipeline_error"
    assert failed.faithfulness is None
    assert failed.answer_correctness is None
    assert resp.failed_queries == 1
    assert resp.passed_gate is False
    # The failed query is not scored, so it does not drag the means
    assert resp.mean_answer_correctness == 0.9


@pytest.mark.asyncio
async def test_floor_applies_to_answer_correctness(mock_dataset_and_baseline):
    """The per-query floor from the baseline covers every metric, not just faithfulness."""
    ds_path, base_path = mock_dataset_and_baseline
    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="a", filtered_chunks=[], degraded=False, provider="groq"
    )
    mock_evaluator = AsyncMock()
    mock_evaluator.evaluate_query.side_effect = [
        _result("Test query 1", 0.95),
        _result("Test query 2", 0.95, answer_correctness=0.40),
    ]

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    resp = await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path)

    # The fixture baseline has no correctness mean to regress against; only the floor trips
    assert resp.mean_answer_correctness == 0.675
    assert resp.passed_gate is False


@pytest.mark.asyncio
async def test_floor_and_pass_thresholds_come_from_the_baseline_file(tmp_path: Path):
    """Raising the floor in the baseline file fails a run that the default floor passes."""
    ds_path = _write(
        tmp_path / "ds.json",
        {
            "version": "3.0.0",
            "domain": "t",
            "corpus": [],
            "queries": [_gold_query("q1", "Q", "unanswerable")],
        },
    )
    baseline = _baseline()
    baseline["gate_thresholds"]["min_query_floor"] = 0.95
    baseline["gate_thresholds"]["pass_thresholds"] = dict.fromkeys(PASS_THRESHOLDS, 0.85)
    base_path = _write(tmp_path / "base.json", baseline)

    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="a", filtered_chunks=[], degraded=False, provider="groq"
    )
    mock_evaluator = AsyncMock()
    mock_evaluator.evaluate_query.return_value = _result("Q", 0.9)

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    resp = await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path)

    assert resp.passed_gate is False
    thresholds = mock_evaluator.evaluate_query.await_args.kwargs["thresholds"]
    assert thresholds.answer_correctness == 0.85


@pytest.mark.asyncio
async def test_correctness_regression_against_baseline_fails_gate(
    tmp_path: Path, mock_dataset_and_baseline
):
    ds_path, _ = mock_dataset_and_baseline
    base_path = _write(
        tmp_path / "with_correctness.json",
        _baseline(
            faithfulness=0.85,
            context_precision=0.80,
            answer_relevance=0.88,
            answer_correctness=0.90,
        ),
    )
    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="a", filtered_chunks=[], degraded=False, provider="groq"
    )
    mock_evaluator = AsyncMock()
    mock_evaluator.evaluate_query.return_value = _result(score=0.9, answer_correctness=0.75)

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    resp = await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path)

    assert resp.regression_delta == -0.15
    assert resp.passed_gate is False


@pytest.mark.asyncio
async def test_baseline_without_pass_thresholds_is_rejected(
    tmp_path: Path, mock_dataset_and_baseline
):
    from lattice_rag.eval.gate import BaselineError

    ds_path, _ = mock_dataset_and_baseline
    base_path = _write(
        tmp_path / "old.json",
        {
            "version": "1.0.0",
            "metrics": {"faithfulness": 0.88},
            "gate_thresholds": {"max_regression_delta": 0.03, "min_query_floor": 0.5},
        },
    )
    mock_orch = AsyncMock()
    runner = EvalRunner(orchestrator=mock_orch, evaluator=AsyncMock())

    with pytest.raises(BaselineError, match="pass_thresholds"):
        await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path)
    mock_orch.run.assert_not_called()


@pytest.mark.asyncio
async def test_slice_decides_which_metrics_are_judged(tmp_path: Path, mock_dataset_and_baseline):
    """Chitchat is judged on the reply only; unanswerable on faithfulness and correctness."""
    _, base_path = mock_dataset_and_baseline
    doc_text = "LatticeDB is embedded."
    ds_path = _write(
        tmp_path / "slices.json",
        {
            "version": "3.0.0",
            "domain": "t",
            "corpus": [{"document_id": "d", "title": "D", "text": doc_text}],
            "queries": [
                _gold_query("c", "hello", "chitchat"),
                _gold_query("u", "unknowable", "unanswerable"),
                _gold_query(
                    "f",
                    "what is it",
                    "factoid",
                    gold_passages=[{"document_id": "d", "text": doc_text}],
                ),
            ],
        },
    )
    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="a", filtered_chunks=[], degraded=False, provider="groq"
    )
    mock_evaluator = AsyncMock()
    mock_evaluator.evaluate_query.return_value = _result()

    runner = EvalRunner(orchestrator=mock_orch, evaluator=mock_evaluator)
    await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path)

    by_query = {
        c.kwargs["query"]: set(c.kwargs["metrics"])
        for c in mock_evaluator.evaluate_query.await_args_list
    }
    assert by_query == {
        "hello": {"answer_relevance", "answer_correctness"},
        "unknowable": {"faithfulness", "answer_correctness"},
        "what is it": {
            "faithfulness",
            "context_precision",
            "answer_relevance",
            "answer_correctness",
        },
    }


class _MetricHonouringEvaluator:
    """Fake evaluator: scores only the requested metrics; a refusal is irrelevant to the query."""

    SCORES = {
        "faithfulness": 0.95,
        "context_precision": 0.9,
        "answer_relevance": 0.1,
        "answer_correctness": 0.9,
    }

    async def evaluate_query(self, *, query, metrics, thresholds, **_: object) -> EvalQueryResult:
        scores = {m: self.SCORES[m] for m in metrics}
        passed = all(v >= getattr(thresholds, m) for m, v in scores.items())
        return EvalQueryResult(query=query, passed=passed, **scores)


@pytest.mark.asyncio
async def test_correct_refusal_on_unanswerable_passes_without_answer_relevance(tmp_path: Path):
    ds_path = _write(
        tmp_path / "ds.json",
        {
            "version": "3.0.0",
            "domain": "t",
            "corpus": [],
            "queries": [_gold_query("u", "unknowable", "unanswerable")],
        },
    )
    base_path = _write(tmp_path / "base.json", _baseline(faithfulness=0.9))
    mock_orch = AsyncMock()
    mock_orch.run.return_value = MagicMock(
        answer="The knowledge base does not contain this.",
        filtered_chunks=[],
        degraded=False,
        provider="groq",
    )

    runner = EvalRunner(orchestrator=mock_orch, evaluator=_MetricHonouringEvaluator())  # type: ignore[arg-type]
    resp = await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path)

    (result,) = resp.results
    assert result.passed is True
    assert result.answer_relevance is None
    assert resp.mean_answer_relevance is None
    assert resp.passed_gate is True
    report = runner.format_markdown_report(resp)
    row = next(line for line in report.splitlines() if "unknowable" in line)
    assert row.split("|")[4].strip() == "n/a"  # the Relevance column
