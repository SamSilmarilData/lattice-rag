"""`lattice-rag hermetic`: the CI entry point of the Hermetic Tier, run with no API keys."""

from __future__ import annotations

import json
from pathlib import Path

import msgspec
import pytest
from click.testing import CliRunner

from lattice_rag import cli
from lattice_rag.eval.gold_set import PASSAGE_FREE_SLICES, load_gold_set
from lattice_rag.eval.hermetic import load_thresholds
from tests.unit.test_hermetic_tier import fake_embedding_service, gold_set

pytestmark = pytest.mark.usefixtures("small_gold_sets")

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("TYPESAFE_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(
        "lattice_rag.retrieval.embeddings.EmbeddingService",
        lambda *a, **k: fake_embedding_service(),
    )


def _write_inputs(tmp_path: Path, floor: float) -> tuple[Path, Path]:
    dataset = tmp_path / "gold.json"
    dataset.write_bytes(msgspec.json.encode(gold_set()))
    thresholds = tmp_path / "thresholds.json"
    gate = {"recall": floor, "mrr": floor}
    thresholds.write_text(
        json.dumps(
            {
                "version": "1",
                "overall": gate,
                "slices": {"factoid": gate, "multihop": gate},
            }
        )
    )
    return dataset, thresholds


def _invoke(dataset: Path, thresholds: Path, *extra: str) -> object:
    return CliRunner().invoke(
        cli.main,
        ["hermetic", "--dataset", str(dataset), "--thresholds", str(thresholds), *extra],
    )


def test_hermetic_passes_without_api_keys_and_exports_the_report(
    keyless: None, tmp_path: Path
) -> None:
    dataset, thresholds = _write_inputs(tmp_path, floor=0.9)
    out = tmp_path / "report.json"

    result = _invoke(dataset, thresholds, "--output", str(out))

    assert result.exit_code == 0, result.output
    assert "HERMETIC TIER PASSED" in result.output
    assert "factoid" in result.output
    report = json.loads(out.read_text())
    assert report["passed"] is True
    assert report["overall"]["recall"] == 1.0


def test_hermetic_prints_and_exports_the_per_case_breakdown(keyless: None, tmp_path: Path) -> None:
    dataset, thresholds = _write_inputs(tmp_path, floor=0.9)
    out = tmp_path / "report.json"

    result = _invoke(dataset, thresholds, "--output", str(out))

    assert result.exit_code == 0, result.output
    assert "Case" in result.output
    case_rows = [line.split() for line in result.output.splitlines() if line.startswith("direct")]
    assert case_rows == [["direct", "4", "1.000", "1.000"]]
    report = json.loads(out.read_text())
    assert report["cases"] == [{"case": "direct", "queries": 4, "recall": 1.0, "mrr": 1.0}]


def test_hermetic_exits_nonzero_below_the_floors(keyless: None, tmp_path: Path) -> None:
    dataset, thresholds = _write_inputs(tmp_path, floor=1.01)

    result = _invoke(dataset, thresholds)

    assert result.exit_code == 1
    assert "HERMETIC TIER FAILED" in result.output
    assert "below floor" in result.output


def test_hermetic_refuses_an_invalid_thresholds_file(keyless: None, tmp_path: Path) -> None:
    dataset, thresholds = _write_inputs(tmp_path, floor=0.5)
    thresholds.write_text(json.dumps({"version": "1", "overall": {"recall": 0.5}}))

    result = _invoke(dataset, thresholds)

    assert result.exit_code != 0
    assert "Invalid hermetic thresholds" in result.output


def test_shipped_thresholds_gate_every_scored_slice_of_the_shipped_gold_set() -> None:
    gold = load_gold_set(REPO_ROOT / "eval_dataset.json")
    gate = load_thresholds(REPO_ROOT / "hermetic_thresholds.json")

    scored = {q.slice for q in gold.queries if q.slice not in PASSAGE_FREE_SLICES}
    assert scored <= set(gate.slices)
