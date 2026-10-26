"""Live Tier: a degraded answer is retried once, then fails the query as provider_degraded."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lattice_rag.api.dtos import EvalQueryResult
from lattice_rag.eval.runner import EvalRunner

PASS = {
    "faithfulness": 0.7,
    "context_precision": 0.65,
    "answer_relevance": 0.7,
    "answer_correctness": 0.7,
}


def _state(*, degraded: bool, provider: str) -> SimpleNamespace:
    return SimpleNamespace(
        answer="an answer", filtered_chunks=[], degraded=degraded, provider=provider
    )


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    query = {
        "id": "q1",
        "query": "Q one",
        "slice": "unanswerable",
        "expected_route": "hybrid",
        "reference_answer": "R",
        "gold_passages": [],
    }
    ds = tmp_path / "ds.json"
    ds.write_text(
        json.dumps({"version": "2.0.0", "domain": "t", "corpus": [], "queries": [query]}),
        encoding="utf-8",
    )
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "version": "2.0.0",
                "metrics": {"faithfulness": 0.85},
                "gate_thresholds": {
                    "max_regression_delta": 0.06,
                    "min_query_floor": 0.5,
                    "pass_thresholds": PASS,
                },
            }
        ),
        encoding="utf-8",
    )
    return ds, base


def _evaluator() -> AsyncMock:
    ev = AsyncMock()
    ev.evaluate_query.return_value = EvalQueryResult(
        query="Q one", passed=True, faithfulness=0.9, answer_correctness=0.9
    )
    return ev


async def test_degraded_twice_fails_query_without_judging(files):
    orch = AsyncMock()
    orch.run.return_value = _state(degraded=True, provider="gemini")
    ev = _evaluator()
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    runner = EvalRunner(orch, ev, degraded_retry_delay=7.0, sleep=fake_sleep)
    resp = await runner.run_evaluation(*files)

    assert orch.run.await_count == 2
    assert sleeps == [7.0]
    ev.evaluate_query.assert_not_awaited()
    res = resp.results[0]
    assert res.error == "provider_degraded"
    assert res.passed is False
    assert res.provider == "gemini"
    assert resp.passed_gate is False
    assert resp.failed_queries == 1


async def test_healthy_retry_is_judged_normally(files):
    orch = AsyncMock()
    orch.run.side_effect = [
        _state(degraded=True, provider="gemini"),
        _state(degraded=False, provider="groq"),
    ]
    ev = _evaluator()
    runner = EvalRunner(orch, ev, degraded_retry_delay=0, sleep=AsyncMock())
    resp = await runner.run_evaluation(*files)

    assert orch.run.await_count == 2
    ev.evaluate_query.assert_awaited_once()
    res = resp.results[0]
    assert res.error is None
    assert res.provider == "groq"


async def test_healthy_first_answer_is_not_retried_and_records_provider(files):
    orch = AsyncMock()
    orch.run.return_value = _state(degraded=False, provider="groq")
    runner = EvalRunner(orch, _evaluator(), sleep=AsyncMock())
    resp = await runner.run_evaluation(*files)

    assert orch.run.await_count == 1
    assert resp.results[0].provider == "groq"


async def test_report_json_lists_provider_per_query(files, tmp_path):
    orch = AsyncMock()
    orch.run.return_value = _state(degraded=False, provider="groq")
    runner = EvalRunner(orch, _evaluator(), sleep=AsyncMock())
    resp = await runner.run_evaluation(*files)

    out = EvalRunner.export_report_json(resp, tmp_path / "r.json")
    assert json.loads(out.read_bytes())["results"][0]["provider"] == "groq"
