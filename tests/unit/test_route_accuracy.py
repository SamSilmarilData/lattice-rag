"""Route Accuracy in the Live Tier report, tested through ``EvalRunner.run_evaluation``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.api.dtos import EvalQueryResult
from lattice_rag.eval.runner import EvalRunner

ROWS = [
    ("r1", "factoid", "vector_exact"),
    ("r2", "factoid", "vector_exact"),
    ("r3", "multihop", "graph_relational"),
    ("r4", "multihop", "graph_relational"),
]


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    dataset = {
        "version": "2.0.0",
        "domain": "test",
        "corpus": [{"document_id": "d1", "title": "T", "text": "alpha beta gamma"}],
        "queries": [
            {
                "id": qid,
                "query": f"query {qid}",
                "slice": slice_name,
                "expected_route": route,
                "reference_answer": "ref",
                "gold_passages": [{"document_id": "d1", "text": "alpha beta"}],
            }
            for qid, slice_name, route in ROWS
        ],
    }
    baseline = {
        "version": "2.0.0",
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
    """Routes each query to a scripted Route ("boom" raises) through the ``run`` seam."""

    semantic_cache = None

    def __init__(self, routes: dict[str, str]) -> None:
        self.routes = routes

    async def run(self, query: str, use_cache: bool = True) -> MagicMock:
        if self.routes[query] == "boom":
            raise RuntimeError("pipeline down")
        return MagicMock(answer="a", filtered_chunks=[{"text": "c"}], route=self.routes[query])


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
    resp = await _run(
        files,
        {
            "query r1": "vector_exact",
            "query r2": "hybrid",
            "query r3": "graph_relational",
            "query r4": "massive_context",
        },
    )

    assert resp.route_accuracy == 0.5
    assert resp.route_accuracy_by_slice == {"factoid": 0.5, "multihop": 0.5}
    assert [(m.query_id, m.chosen_route, m.expected_route) for m in resp.misroutes] == [
        ("r2", "hybrid", "vector_exact"),
        ("r4", "massive_context", "graph_relational"),
    ]
    assert resp.route_accuracy_floor == 0.80


async def test_missed_floor_is_advisory_and_never_fails_the_gate(files):
    resp = await _run(files, {f"query r{i}": "chitchat" for i in range(1, 5)})

    assert resp.route_accuracy == 0.0
    assert resp.passed_gate is True


async def test_pipeline_failure_is_not_scored_for_routing(files):
    resp = await _run(
        files,
        {
            "query r1": "vector_exact",
            "query r2": "boom",
            "query r3": "graph_relational",
            "query r4": "graph_relational",
        },
    )

    assert resp.route_accuracy == 1.0
    assert resp.misroutes == []
    assert resp.route_accuracy_by_slice == {"factoid": 1.0, "multihop": 1.0}
