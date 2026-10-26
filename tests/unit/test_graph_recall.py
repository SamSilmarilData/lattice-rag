"""Graph Recall in the Live Tier report, tested through ``EvalRunner.run_evaluation``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lattice_rag.api.dtos import EvalQueryResult
from lattice_rag.eval.runner import EvalRunner
from lattice_rag.orchestration.state import PipelineState

pytestmark = pytest.mark.usefixtures("small_gold_sets")

CORPUS_TEXT = "LatticeDB stores an HNSW index. GLiNER extracts entities. Groq answers."


def _query(qid: str, slice_name: str, entities: list[str] | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": qid,
        "query": f"query {qid}",
        "slice": slice_name,
        "case": "direct",
        "expected_route": "hybrid",
        "reference_answer": "ref",
        "gold_passages": [{"document_id": "d1", "text": "LatticeDB stores an HNSW index."}],
    }
    if entities is not None:
        row["expected_entities"] = entities
    return row


def _write_files(tmp_path: Path, queries: list[dict[str, Any]]) -> tuple[Path, Path]:
    dataset = {
        "version": "3.0.0",
        "domain": "test",
        "corpus": [{"document_id": "d1", "title": "T", "text": CORPUS_TEXT}],
        "queries": queries,
    }
    baseline = {
        "version": "3.0.0",
        "metrics": {"faithfulness": 0.85},
        "gate_thresholds": {
            "max_regression_delta": 0.03,
            "min_query_floor": 0.50,
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
    """Answers every query and surfaces a scripted set of graph node names per query.

    A query mapped to ``None`` raises, as a failed pipeline would.
    """

    semantic_cache = None

    def __init__(self, nodes: dict[str, list[str] | None]) -> None:
        self.nodes = nodes

    async def run(self, query: str, use_cache: bool = True) -> PipelineState:
        names = self.nodes[query]
        if names is None:
            raise RuntimeError("pipeline down")
        return PipelineState(
            query=query,
            route="hybrid",
            answer="a",
            filtered_chunks=[{"text": "c"}],
            graph_context={
                "nodes": [{"id": i, "name": n, "label": "Entity"} for i, n in enumerate(names)],
                "edges": [],
            },
            provider="groq",
        )


class PassingEvaluator:
    async def evaluate_query(self, *, query: str, **_: object) -> EvalQueryResult:
        return EvalQueryResult(query=query, passed=True, faithfulness=0.9)


async def _run(tmp_path: Path, queries: list[dict[str, Any]], nodes: dict[str, list[str] | None]):
    ds, base = _write_files(tmp_path, queries)
    runner = EvalRunner(orchestrator=FakeOrchestrator(nodes), evaluator=PassingEvaluator())  # type: ignore[arg-type]
    return await runner.run_evaluation(dataset_path=ds, baseline_path=base)


@pytest.mark.asyncio
async def test_graph_recall_is_the_share_of_expected_entities_the_graph_surfaced(
    tmp_path: Path,
) -> None:
    report = await _run(
        tmp_path,
        [
            _query("m1", "multihop", ["LatticeDB", "HNSW", "GLiNER", "Groq"]),
            _query("m2", "multihop", ["GLiNER", "Groq"]),
            _query("f1", "factoid", ["latticedb"]),
        ],
        {
            # 2 of 4: matching ignores case and collapses whitespace
            "query m1": ["latticedb", " HNSW ", "Gemini"],
            "query m2": ["GLiNER", "Groq"],
            "query f1": [],
        },
    )

    assert [r.graph_recall for r in report.results] == [0.5, 1.0, 0.0]
    assert report.graph_recall_by_slice == {"multihop": 0.75, "factoid": 0.0}
    assert report.graph_recall == 0.5
    assert report.graph_recall_scored == 3


@pytest.mark.asyncio
async def test_zero_graph_recall_is_advisory_and_never_fails_the_gate(tmp_path: Path) -> None:
    report = await _run(
        tmp_path,
        [_query("m1", "multihop", ["LatticeDB", "GLiNER"])],
        {"query m1": []},
    )

    assert report.graph_recall == 0.0
    assert report.passed_gate is True


@pytest.mark.asyncio
async def test_queries_without_expected_entities_or_a_pipeline_are_not_scored(
    tmp_path: Path,
) -> None:
    report = await _run(
        tmp_path,
        [
            _query("m1", "multihop", ["LatticeDB"]),
            _query("n1", "factoid", None),
            _query("e1", "factoid", ["Groq"]),
        ],
        {"query m1": ["LatticeDB"], "query n1": ["LatticeDB"], "query e1": None},
    )

    assert [r.graph_recall for r in report.results] == [1.0, None, None]
    assert report.graph_recall_by_slice == {"multihop": 1.0}
    assert report.graph_recall_scored == 1


@pytest.mark.asyncio
async def test_markdown_report_shows_graph_recall_per_slice(tmp_path: Path) -> None:
    report = await _run(
        tmp_path,
        [_query("m1", "multihop", ["LatticeDB", "GLiNER"])],
        {"query m1": ["LatticeDB"]},
    )

    md = EvalRunner.format_markdown_report(report)

    assert "Graph Recall" in md
    assert "multihop: 0.5000" in md
