"""The Live Tier judge sees the graph triples the generator was given."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.api.dtos import EvalQueryResult
from lattice_rag.eval.runner import EvalRunner
from lattice_rag.eval.triage_gate import JevEvaluator

METRICS = ("faithfulness", "context_precision", "answer_relevance", "answer_correctness")

GRAPH = {
    "nodes": [{"node_id": 1, "name": "GLiNER"}, {"node_id": 2, "name": "Entity"}],
    "edges": [{"source_id": 1, "target_id": 2, "relation_type": "EXTRACTS"}],
}


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    dataset = {
        "version": "3.0.0",
        "domain": "test",
        "corpus": [],
        "queries": [
            {
                "id": "q1",
                "query": "Q one",
                "slice": "unanswerable",
                "case": "direct",
                "expected_route": "hybrid",
                "reference_answer": "R",
                "gold_passages": [],
            }
        ],
    }
    baseline = {
        "version": "3.0.0",
        "metrics": {"faithfulness": 0.5},
        "gate_thresholds": {
            "max_regression_delta": 0.06,
            "min_query_floor": 0.5,
            "pass_thresholds": dict.fromkeys(METRICS, 0.7),
        },
    }
    ds, base = tmp_path / "ds.json", tmp_path / "base.json"
    ds.write_text(json.dumps(dataset), encoding="utf-8")
    base.write_text(json.dumps(baseline), encoding="utf-8")
    return ds, base


async def _evaluator_calls(files: tuple[Path, Path], graph_context: dict | None) -> list[dict]:
    seen: list[dict] = []

    class FakeEvaluator:
        async def evaluate_query(self, **kwargs):
            seen.append(kwargs)
            return EvalQueryResult(
                query=kwargs["query"], passed=True, **dict.fromkeys(METRICS, 0.9)
            )

    orch = AsyncMock()
    orch.run.return_value = MagicMock(
        answer="a",
        filtered_chunks=[{"text": "chunk"}],
        graph_context=graph_context,
        degraded=False,
        provider="groq",
    )
    runner = EvalRunner(orchestrator=orch, evaluator=FakeEvaluator())  # type: ignore[arg-type]
    await runner.run_evaluation(dataset_path=files[0], baseline_path=files[1])
    return seen


async def test_runner_hands_generator_graph_to_evaluator(files):
    calls = await _evaluator_calls(files, GRAPH)
    assert [c["graph_context"] for c in calls] == [GRAPH]


async def test_runner_passes_no_graph_when_none_retrieved(files):
    calls = await _evaluator_calls(files, None)
    assert [c["graph_context"] for c in calls] == [None]


def _judge() -> tuple[JevEvaluator, AsyncMock]:
    resp = MagicMock()
    resp.scores = {m: MagicMock(score=4.0) for m in METRICS}
    resp.nouls = {"contradiction": MagicMock(noul=0.0)}
    client = AsyncMock()
    client.system_one.return_value = resp
    return JevEvaluator(client=client), client


_BASE = {
    "query": "q",
    "retrieved_chunks": [{"text": "chunk"}],
    "generated_answer": "a",
    "ground_truth": "r",
}


async def test_judge_context_unchanged_without_graph():
    evaluator, client = _judge()
    await evaluator.evaluate_query(**_BASE)
    assert client.system_one.await_args.kwargs["state"]["context"] == "[Chunk 1] chunk"


async def test_judge_context_includes_triples_with_graph():
    evaluator, client = _judge()
    await evaluator.evaluate_query(**_BASE, graph_context=GRAPH)
    context = client.system_one.await_args.kwargs["state"]["context"]
    assert context.startswith("[Chunk 1] chunk")
    assert "GLiNER --[EXTRACTS]--> Entity" in context


async def test_judge_context_omits_triples_section_when_graph_has_no_edges():
    evaluator, client = _judge()
    await evaluator.evaluate_query(
        **_BASE, graph_context={"nodes": [{"node_id": 1, "name": "GLiNER"}], "edges": []}
    )
    context = client.system_one.await_args.kwargs["state"]["context"]
    assert context == "[Chunk 1] chunk"
