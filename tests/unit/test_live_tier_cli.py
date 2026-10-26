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


# --- Baseline measurement from repeated Live Tier runs ---------------------------------

_BASELINE = {
    "version": "3.0.0",
    "updated_at": "2026-01-01T00:00:00Z",
    "total_queries": 3,
    "metrics": {"faithfulness": 0.5},
    "gate_thresholds": {
        "max_regression_delta": 0.03,
        "min_query_floor": 0.5,
        "pass_thresholds": {
            "faithfulness": 0.7,
            "context_precision": 0.65,
            "answer_relevance": 0.7,
            "answer_correctness": 0.7,
        },
    },
}


def _run_file(tmp_path: Path, name: str, faithfulness: float, precision: float) -> str:
    report = EvalRunResponse(
        total_queries=3,
        passed_gate=False,
        regression_delta=0.0,
        mean_faithfulness=faithfulness,
        mean_context_precision=precision,
        mean_answer_relevance=0.9,
        mean_answer_correctness=None,  # never judged in any run
    )
    path = tmp_path / name
    path.write_bytes(msgspec.json.encode(report))
    return str(path)


def test_measure_baseline_sets_means_and_two_sigma_tolerance(tmp_path: Path):
    """Faithfulness 0.80/0.84/0.88 has mean 0.84 and sample sd 0.04, so 2 sd = 0.08,
    the widest of the metrics; precision is constant (sd 0)."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_BASELINE))
    runs = [
        _run_file(tmp_path, "r1.json", 0.80, 0.70),
        _run_file(tmp_path, "r2.json", 0.84, 0.70),
        _run_file(tmp_path, "r3.json", 0.88, 0.70),
    ]

    result = CliRunner().invoke(cli.main, ["measure-baseline", *runs, "--baseline", str(baseline)])

    assert result.exit_code == 0, result.output
    from lattice_rag.eval.gate import load_baseline

    loaded = load_baseline(baseline)
    assert loaded.metrics == pytest.approx(
        {"faithfulness": 0.84, "context_precision": 0.70, "answer_relevance": 0.9}
    )
    assert loaded.gate_thresholds.max_regression_delta == pytest.approx(0.08)
    # Floor and per-query thresholds are policy, not measurements: untouched
    assert loaded.gate_thresholds.min_query_floor == 0.5
    assert loaded.gate_thresholds.pass_thresholds.faithfulness == 0.7
    assert loaded.measurement is not None
    assert [r["faithfulness"] for r in loaded.measurement.runs] == [0.80, 0.84, 0.88]
    assert loaded.measurement.stdev["faithfulness"] == pytest.approx(0.04)
    assert "2" in loaded.measurement.tolerance_rule


def test_measure_baseline_needs_at_least_two_runs(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_BASELINE))
    run = _run_file(tmp_path, "r1.json", 0.8, 0.7)

    result = CliRunner().invoke(cli.main, ["measure-baseline", run, "--baseline", str(baseline)])

    assert result.exit_code != 0
    assert "at least two" in result.output
    assert json.loads(baseline.read_text()) == _BASELINE


def test_eval_passes_configured_degraded_retry_delay(monkeypatch, mocked_live_eval):
    captured: dict = {}

    def _runner(**kwargs):
        captured.update(kwargs)
        return mocked_live_eval

    monkeypatch.setattr("lattice_rag.eval.EvalRunner", _runner)
    cli._build_harness.return_value["config"].eval_degraded_retry_delay = 9.0

    result = CliRunner().invoke(cli.main, ["eval"])

    assert result.exit_code == 0, result.output
    assert captured["degraded_retry_delay"] == 9.0
