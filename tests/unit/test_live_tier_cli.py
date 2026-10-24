"""Live Tier CLI tests: report export, missing-key failures and baseline measurement.

No real keys, models or network: the harness and the Jev judge are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest
from click.testing import CliRunner

from lattice_rag import cli
from lattice_rag.api.dtos import EvalQueryResult, EvalRunResponse


def _report(passed: bool = True) -> EvalRunResponse:
    return EvalRunResponse(
        total_queries=1,
        passed_gate=passed,
        regression_delta=0.0,
        mean_faithfulness=0.9,
        mean_context_precision=0.8,
        mean_answer_relevance=0.95,
        mean_answer_correctness=0.85,
        results=[EvalQueryResult(query="q", passed=passed, faithfulness=0.9)],
    )


@pytest.fixture
def mocked_live_eval(monkeypatch):
    """Replace the harness, judge and runner so `lattice-rag eval` runs offline."""
    config = MagicMock()
    harness = {"config": config, "orchestrator": MagicMock(), "embed_svc": MagicMock()}
    monkeypatch.setattr(cli, "_build_harness", AsyncMock(return_value=harness))
    monkeypatch.setattr(cli, "_close_harness", AsyncMock())
    monkeypatch.setattr("lattice_rag.eval.JevEvaluator", lambda **_: AsyncMock())
    monkeypatch.setattr("lattice_rag.storage.extract.EntityExtractor", MagicMock())
    runner = MagicMock()
    runner.run_evaluation = AsyncMock(return_value=_report())
    monkeypatch.setattr("lattice_rag.eval.EvalRunner", lambda **_: runner)
    monkeypatch.setenv("TYPESAFE_API_KEY", "mock-typesafe")
    monkeypatch.setenv("GROQ_API_KEY", "mock-groq")
    return runner


def test_eval_writes_the_report_to_output(mocked_live_eval, tmp_path: Path):
    out = tmp_path / "run.json"

    result = CliRunner().invoke(cli.main, ["eval", "--output", str(out)])

    assert result.exit_code == 0, result.output
    written = msgspec.json.decode(out.read_bytes(), type=EvalRunResponse)
    assert written.mean_answer_correctness == 0.85
    assert written.results[0].query == "q"


def test_eval_writes_the_report_even_when_the_gate_fails(mocked_live_eval, tmp_path: Path):
    """A failing run is still a measurement; the report must survive the non-zero exit."""
    mocked_live_eval.run_evaluation.return_value = _report(passed=False)
    out = tmp_path / "run.json"

    result = CliRunner().invoke(cli.main, ["eval", "--output", str(out)])

    assert result.exit_code == 1
    assert json.loads(out.read_text())["passedGate"] is False


@pytest.mark.parametrize("missing", ["TYPESAFE_API_KEY", "GROQ_API_KEY"])
def test_eval_without_a_live_key_fails_and_names_it(mocked_live_eval, monkeypatch, missing):
    """A missing secret is a visible failure, never a silent skip or a degraded run."""
    monkeypatch.setenv(missing, "")

    result = CliRunner().invoke(cli.main, ["eval"])

    assert result.exit_code != 0
    assert missing in result.output
    mocked_live_eval.run_evaluation.assert_not_called()
