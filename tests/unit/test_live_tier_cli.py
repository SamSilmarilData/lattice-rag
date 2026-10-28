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


def test_measure_baseline_notes_record_the_model_mix(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_BASELINE))
    mixes = [
        {"groq:qwen/qwen3.8-27b": 2, "groq:openai/gpt-oss-120b": 1},
        {"groq:qwen/qwen3.8-27b": 3},
    ]
    runs = []
    for i, mix in enumerate(mixes):
        path = tmp_path / f"r{i}.json"
        path.write_bytes(
            msgspec.json.encode(
                EvalRunResponse(
                    total_queries=3,
                    passed_gate=True,
                    regression_delta=0.0,
                    mean_faithfulness=0.8 + i / 100,
                    answers_by_model=mix,
                )
            )
        )
        runs.append(str(path))

    result = CliRunner().invoke(cli.main, ["measure-baseline", *runs, "--baseline", str(baseline)])

    assert result.exit_code == 0, result.output
    from lattice_rag.eval.gate import load_baseline

    measurement = load_baseline(baseline).measurement
    assert measurement is not None
    assert measurement.notes == (
        "Model mix over 2 runs: groq:qwen/qwen3.8-27b 5, groq:openai/gpt-oss-120b 1"
    )


def test_measure_baseline_records_route_accuracy_per_run_and_mean(tmp_path: Path):
    """Route Accuracy is advisory, so it is recorded next to the means but sets no tolerance."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_BASELINE))
    runs = []
    for i, accuracy in enumerate([0.975, 0.9625, None]):
        path = tmp_path / f"r{i}.json"
        path.write_bytes(
            msgspec.json.encode(
                EvalRunResponse(
                    total_queries=3,
                    passed_gate=True,
                    regression_delta=0.0,
                    mean_faithfulness=0.8,
                    route_accuracy=accuracy,
                )
            )
        )
        runs.append(str(path))

    result = CliRunner().invoke(cli.main, ["measure-baseline", *runs, "--baseline", str(baseline)])

    assert result.exit_code == 0, result.output
    from lattice_rag.eval.gate import load_baseline

    measurement = load_baseline(baseline).measurement
    assert measurement is not None
    assert measurement.route_accuracy_runs == [0.975, 0.9625]  # a run with none scored is skipped
    assert measurement.route_accuracy == pytest.approx(0.9688)
    assert measurement.tolerance == pytest.approx(0.01)  # Route Accuracy never widens it


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


# --- Experiment Arms (#46 forced hops, #47 graph context) ----------------------


def test_eval_runs_the_configured_system_by_default(mocked_live_eval):
    result = CliRunner().invoke(cli.main, ["eval"])

    assert result.exit_code == 0, result.output
    cli._build_harness.assert_awaited_once_with(force_hops=None, graph_context=None)


def test_eval_passes_the_experiment_arm_to_the_harness(mocked_live_eval):
    from lattice_rag.vocabulary import GraphContextMode

    result = CliRunner().invoke(
        cli.main, ["eval", "--force-hops", "2", "--graph-context", "relational"]
    )

    assert result.exit_code == 0, result.output
    cli._build_harness.assert_awaited_once_with(
        force_hops=2, graph_context=GraphContextMode.RELATIONAL
    )


@pytest.mark.parametrize("args", [["--force-hops", "3"], ["--graph-context", "sometimes"]])
def test_eval_rejects_an_unknown_arm(mocked_live_eval, args):
    result = CliRunner().invoke(cli.main, ["eval", *args])

    assert result.exit_code == 2
    mocked_live_eval.run_evaluation.assert_not_called()


@pytest.fixture
def offline_services(monkeypatch):
    """Stub every service `_build_harness` constructs, so the real wiring runs offline."""
    made_guardrails: list[float] = []

    def guardrail(threshold: float = 0.5, **_kw):
        made_guardrails.append(threshold)
        return MagicMock()

    for target in (
        "lattice_rag.storage.db.LatticeStore",
        "lattice_rag.retrieval.embeddings.EmbeddingService",
        "lattice_rag.routing.router.QueryRouter",
        "lattice_rag.caching.semantic_cache.SemanticCache",
        "lattice_rag.generation.groq_synthesizer.GroqSynthesizer",
        "lattice_rag.generation.gemini_synthesizer.GeminiSynthesizer",
    ):
        monkeypatch.setattr(target, MagicMock())
    fallback = MagicMock()
    fallback.return_value.connect = AsyncMock()
    monkeypatch.setattr("lattice_rag.caching.fallback_cache.FallbackCache", fallback)
    monkeypatch.setattr("lattice_rag.routing.guardrail.ContextGuardrail", guardrail)
    return made_guardrails


async def test_harness_applies_the_arm_and_the_phase5_config(offline_services, monkeypatch):
    from lattice_rag.vocabulary import GraphContextMode

    monkeypatch.setenv("GRAPH_CONTEXT_IN_PROMPT", "on")
    monkeypatch.setenv("GUARDRAIL_MASSIVE_CONTEXT_THRESHOLD", "0.35")
    monkeypatch.setenv("GUARDRAIL_SKIP_MASSIVE_CONTEXT", "false")

    h = await cli._build_harness(force_hops=1, graph_context=GraphContextMode.OFF)

    assert h["pipeline"].forced_hops == 1
    assert h["orchestrator"].graph_context_in_prompt is GraphContextMode.OFF
    assert offline_services == [0.5, 0.35]
    guardrails = h["pipeline"].guardrails
    assert guardrails.for_route("massive_context") is not guardrails.for_route("hybrid")


async def test_harness_builds_the_configured_pool_with_the_llm_timeout(
    offline_services, monkeypatch
):
    from lattice_rag.generation import gemini_synthesizer, groq_synthesizer

    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "11")
    monkeypatch.setenv("LLM_POOL", "groq:a,gemini:g@low")

    h = await cli._build_harness()

    assert [m.label for m in h["orchestrator"].pool] == ["groq:a", "gemini:g"]
    groq_kwargs = groq_synthesizer.GroqSynthesizer.call_args.kwargs
    gemini_kwargs = gemini_synthesizer.GeminiSynthesizer.call_args.kwargs
    assert (groq_kwargs["model"], groq_kwargs["timeout_seconds"]) == ("a", 11.0)
    assert (gemini_kwargs["model"], gemini_kwargs["timeout_seconds"]) == ("g", 11.0)
    assert gemini_kwargs["thinking"] == "low"


async def test_harness_without_an_arm_follows_the_environment(offline_services, monkeypatch):
    from lattice_rag.vocabulary import GraphContextMode

    monkeypatch.setenv("GRAPH_CONTEXT_IN_PROMPT", "relational")
    monkeypatch.setenv("GUARDRAIL_SKIP_MASSIVE_CONTEXT", "true")

    h = await cli._build_harness()

    assert h["pipeline"].forced_hops is None
    assert h["orchestrator"].graph_context_in_prompt is GraphContextMode.RELATIONAL
    assert h["pipeline"].guardrails.for_route("massive_context") is None
