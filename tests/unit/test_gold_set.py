"""Gold Set v2: loading, validation, and the shipped dataset itself."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lattice_rag.eval.gold_set import GoldSetError, load_gold_set

SHIPPED = Path(__file__).resolve().parents[2] / "eval_dataset.json"


def _raw() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(SHIPPED.read_text(encoding="utf-8"))
    return data


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_shipped_gold_set_is_valid() -> None:
    gold = load_gold_set(SHIPPED)

    assert gold.version == "2.0.0"
    assert len(gold.queries) >= 30


def test_altered_passage_is_rejected(tmp_path: Path) -> None:
    data = _raw()
    query = data["queries"][0]
    query["gold_passages"][0]["text"] += " (altered)"

    with pytest.raises(GoldSetError) as exc:
        load_gold_set(_write(tmp_path, data))

    message = str(exc.value)
    assert query["id"] in message
    assert "not verbatim" in message


def test_missing_required_field_is_rejected_with_its_name(tmp_path: Path) -> None:
    data = _raw()
    del data["queries"][3]["slice"]

    with pytest.raises(GoldSetError) as exc:
        load_gold_set(_write(tmp_path, data))

    message = str(exc.value)
    assert "slice" in message
    assert "queries[3]" in message


def test_shipped_gold_set_covers_every_slice() -> None:
    gold = load_gold_set(SHIPPED)

    slices = {q.slice for q in gold.queries}
    assert slices == {"factoid", "multihop", "unanswerable", "chitchat", "long_context"}


def test_shipped_chitchat_queries_take_the_deterministic_path() -> None:
    from lattice_rag.generation.chitchat import is_pure_chitchat

    gold = load_gold_set(SHIPPED)

    chitchat = [q for q in gold.queries if q.slice == "chitchat"]
    assert chitchat
    assert all(q.expected_route == "chitchat" for q in chitchat)
    assert all(is_pure_chitchat(q.query) for q in chitchat)
    others = [q for q in gold.queries if q.slice != "chitchat"]
    assert not any(is_pure_chitchat(q.query) for q in others)


def test_passage_citing_unknown_document_is_rejected(tmp_path: Path) -> None:
    data = _raw()
    data["queries"][0]["gold_passages"][0]["document_id"] = "doc_missing"

    with pytest.raises(GoldSetError, match="unknown document 'doc_missing'"):
        load_gold_set(_write(tmp_path, data))


def test_unanswerable_query_with_passages_is_rejected(tmp_path: Path) -> None:
    data = _raw()
    unanswerable = next(q for q in data["queries"] if q["slice"] == "unanswerable")
    unanswerable["gold_passages"] = data["queries"][0]["gold_passages"]

    with pytest.raises(GoldSetError, match=f"{unanswerable['id']}: slice 'unanswerable'"):
        load_gold_set(_write(tmp_path, data))


def test_unknown_slice_is_rejected(tmp_path: Path) -> None:
    data = _raw()
    data["queries"][0]["slice"] = "trivia"

    with pytest.raises(GoldSetError, match=r"queries\[0\]\.slice"):
        load_gold_set(_write(tmp_path, data))


def test_v1_dataset_is_rejected(tmp_path: Path) -> None:
    data = _raw()
    data["version"] = "1.0.0"

    with pytest.raises(GoldSetError, match="expected '2.0.0'"):
        load_gold_set(_write(tmp_path, data))
