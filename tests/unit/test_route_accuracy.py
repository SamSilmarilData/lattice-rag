"""Route Accuracy in the Live Tier report, tested through ``EvalRunner.run_evaluation``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.api.dtos import EvalQueryResult
from lattice_rag.eval.runner import EvalRunner

pytestmark = pytest.mark.usefixtures("small_gold_sets")

ROWS = [
    ("r1", "factoid", "hybrid"),
    ("r2", "factoid", "hybrid"),
    ("r3", "multihop", "massive_context"),
    ("r4", "multihop", "massive_context"),
    ("c1", "chitchat", "chitchat"),
]
ALL_CORRECT = {
    "query r1": "hybrid",
    "query r2": "hybrid",
    "query r3": "massive_context",
    "query r4": "massive_context",
    "query c1": "chitchat",
}


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    dataset = {
        "version": "3.0.0",
        "domain": "test",
        "corpus": [{"document_id": "d1", "title": "T", "text": "alpha beta gamma"}],
        "queries": [
            {
                "id": qid,
                "query": f"query {qid}",
                "slice": slice_name,
                "case": "direct",
                "expected_route": route,
                "reference_answer": "ref",
                "gold_passages": []
                if slice_name == "chitchat"
                else [{"document_id": "d1", "text": "alpha beta"}],
            }
            for qid, slice_name, route in ROWS
        ],
    }
    baseline = {
        "version": "3.0.0",
        "metrics": {"faithfulness": 0.85},
        "gate_thresholds": {
            "max_regression_delta": 0.03,
            "min_query_floor": 0.50,
            "route_accuracy_floor": 0.80,
            "pass_thresholds": {
                "faithfulness": 0.70,
                "context_precision": 0.65,
                "answer_relevance": 0.70,
                "answer_correctness": 0.70,
            },
        },
    }
    ds, base = tmp_path / "ds.json", tmp_path / "base.json"
    ds.write_text(json.dumps(dataset), encoding="utf-8")
    base.write_text(json.dumps(baseline), encoding="utf-8")
    return ds, base


class FakeOrchestrator:
    """Routes each query to a scripted Route through the ``run`` seam.

    "boom" raises; "fallback" returns the unverified System 1 fallback (``hybrid``).
    """

    semantic_cache = None

    def __init__(self, routes: dict[str, str]) -> None:
        self.routes = routes

    async def run(self, query: str, use_cache: bool = True) -> MagicMock:
        route = self.routes[query]
        if route == "boom":
            raise RuntimeError("pipeline down")
        verified = route != "fallback"
        return MagicMock(
            answer="a",
            filtered_chunks=[{"text": "c"}],
            route="hybrid" if route == "fallback" else route,
            route_verified=verified,
            degraded=False,
            provider="groq",
        )


async def _run(files: tuple[Path, Path], routes: dict[str, str]):
    evaluator = AsyncMock()
    evaluator.evaluate_query.return_value = EvalQueryResult(
        query="q",
        passed=True,
        faithfulness=0.9,
        context_precision=0.9,
        answer_relevance=0.9,
        answer_correctness=0.9,
    )
    runner = EvalRunner(orchestrator=FakeOrchestrator(routes), evaluator=evaluator)  # type: ignore[arg-type]
    return await runner.run_evaluation(dataset_path=files[0], baseline_path=files[1])


async def test_reports_overall_and_per_slice_accuracy_and_misroutes(files):
    resp = await _run(files, {**ALL_CORRECT, "query r2": "massive_context", "query r4": "hybrid"})

    assert resp.route_accuracy == 0.5
    assert resp.route_accuracy_by_slice == {"factoid": 0.5, "multihop": 0.5}
    assert [(m.query_id, m.chosen_route, m.expected_route) for m in resp.misroutes] == [
        ("r2", "massive_context", "hybrid"),
        ("r4", "hybrid", "massive_context"),
    ]
    assert resp.route_accuracy_floor == 0.80


async def test_chitchat_slice_is_outside_the_denominator(files):
    """Chitchat is routed deterministically, so it is not a router decision to score."""
    resp = await _run(files, {**ALL_CORRECT, "query c1": "hybrid", "query r1": "massive_context"})

    assert resp.route_accuracy_scored == 4
    assert resp.route_accuracy == 0.75
    assert "chitchat" not in resp.route_accuracy_by_slice
    assert [m.query_id for m in resp.misroutes] == ["r1"]


async def test_each_result_records_its_chosen_route(files):
    resp = await _run(files, {**ALL_CORRECT, "query r2": "massive_context", "query r3": "boom"})

    assert [r.chosen_route for r in resp.results] == [
        "hybrid",
        "massive_context",
        None,
        "massive_context",
        "chitchat",
    ]


async def test_missed_floor_is_advisory_and_never_fails_the_gate(files):
    resp = await _run(files, dict.fromkeys(ALL_CORRECT, "chitchat"))

    assert resp.route_accuracy == 0.0
    assert resp.passed_gate is True


async def test_unverified_fallback_route_is_counted_not_scored(files):
    """A Jev outage must not score the fallback ``hybrid`` as a router decision."""
    resp = await _run(files, {**ALL_CORRECT, "query r1": "fallback", "query r3": "fallback"})

    assert resp.route_accuracy_unverified == 2
    assert resp.route_accuracy_scored == 2
    assert resp.route_accuracy == 1.0
    assert resp.misroutes == []


async def test_jev_outage_reports_no_accuracy_rather_than_a_score(files):
    routed = {q: "fallback" for q in ALL_CORRECT if q != "query c1"}
    resp = await _run(files, {**ALL_CORRECT, **routed})

    assert resp.route_accuracy is None
    assert resp.route_accuracy_unverified == 4
    assert resp.route_accuracy_by_slice == {}


async def test_pipeline_failure_is_counted_and_not_scored(files):
    resp = await _run(files, {**ALL_CORRECT, "query r2": "boom"})

    assert resp.route_accuracy_errored == 1
    assert resp.route_accuracy_scored == 3
    assert resp.route_accuracy == 1.0
    assert resp.misroutes == []
    assert resp.route_accuracy_by_slice == {"factoid": 1.0, "multihop": 1.0}


async def test_every_pipeline_failing_reports_no_accuracy(files):
    resp = await _run(files, dict.fromkeys(ALL_CORRECT, "boom"))

    assert resp.route_accuracy is None
    assert resp.route_accuracy_errored == 4
    assert resp.route_accuracy_scored == 0


async def test_markdown_report_states_the_denominator_and_exclusions(files):
    resp = await _run(files, {**ALL_CORRECT, "query r1": "fallback", "query r2": "boom"})

    text = EvalRunner.format_markdown_report(resp)

    assert "Route Accuracy**: 1.0000 over 2 routed queries" in text
    assert "1 unverified fallback route" in text
    assert "1 pipeline error" in text
