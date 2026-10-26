"""Gold Set v3: loading, validation, and the shipped dataset itself."""

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

    assert gold.version == "3.0.0"
    assert len(gold.queries) == 83


def test_shipped_near_miss_queries_are_unanswerable_with_a_refusal() -> None:
    gold = load_gold_set(SHIPPED)

    near_misses = [q for q in gold.queries if q.case == "unanswerable_near_miss"]
    assert len(near_misses) >= 10
    for q in near_misses:
        assert q.slice == "unanswerable", q.id
        assert not q.gold_passages, q.id
        assert q.reference_answer.startswith(
            "The knowledge base does not contain this information."
        ), q.id


def test_shipped_ambiguity_and_conflicting_source_queries_carry_evidence() -> None:
    gold = load_gold_set(SHIPPED)

    hard = [q for q in gold.queries if q.case in ("ambiguity", "conflicting_sources")]
    assert sum(q.case == "ambiguity" for q in hard) >= 8
    assert sum(q.case == "conflicting_sources" for q in hard) >= 8
    for q in hard:
        assert len(q.gold_passages) >= 2, q.id


def test_shipped_corpus_is_the_five_documents_plus_the_long_form_docs() -> None:
    from lattice_rag.ingestion.ingester import chunk_text

    gold = load_gold_set(SHIPPED)

    ids = {d.document_id for d in gold.corpus}
    assert {
        "doc_latticedb_internals",
        "doc_fastembed_reranking",
        "doc_gliner_extraction",
        "doc_typesafe_system_one",
        "doc_litestar_granian_stack",
    } <= ids
    for stem in ("architecture", "storage_and_schema", "api_reference", "deployment", "security"):
        assert any(i.startswith(f"doc_{stem}_") for i in ids), stem
    assert any(i.startswith("doc_performance_benchmark_") for i in ids)
    chunks = sum(len(chunk_text(d.text)) for d in gold.corpus)
    assert 120 <= chunks <= 220


def test_corpus_sends_graph_triples_to_the_generation_prompt_not_the_reranker() -> None:
    gold = load_gold_set(SHIPPED)

    text = " ".join(d.text for d in gold.corpus)
    assert "appended to candidate text chunks" not in text
    assert "jointly score" not in text
    assert "graph triples are not appended to candidates" in text
    assert "generation prompt" in text
    by_id = {q.id: q for q in gold.queries}
    for qid in ("eval_04", "eval_20", "eval_21", "eval_30"):
        answer = by_id[qid].reference_answer.lower()
        assert "reranker scores" not in answer
        assert "linearized graph triples" not in answer
    assert "prompt" in by_id["eval_04"].reference_answer.lower()
    assert "prompt" in by_id["eval_21"].reference_answer.lower()


def test_no_neo4j_or_pinecone_comparison_without_corpus_backing() -> None:
    gold = load_gold_set(SHIPPED)

    query = next(q for q in gold.queries if q.id == "eval_19")
    corpus = " ".join(d.text for d in gold.corpus)
    for name in ("Neo4j", "Pinecone"):
        assert name not in query.query + query.reference_answer or name in corpus


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

    with pytest.raises(GoldSetError, match="expected '3.0.0'"):
        load_gold_set(_write(tmp_path, data))


def test_unknown_case_is_rejected(tmp_path: Path) -> None:
    data = _raw()
    data["queries"][0]["case"] = "trick_question"

    with pytest.raises(GoldSetError, match=r"queries\[0\]\.case"):
        load_gold_set(_write(tmp_path, data))


def test_shipped_gold_set_has_paraphrase_and_negation_queries_with_evidence() -> None:
    gold = load_gold_set(SHIPPED)

    reworded = [q for q in gold.queries if q.case in ("paraphrase", "negation")]
    assert sum(q.case == "paraphrase" for q in reworded) >= 8
    assert sum(q.case == "negation" for q in reworded) >= 8
    assert len(reworded) >= 20
    for q in reworded:
        assert q.gold_passages, q.id
        assert q.slice not in ("unanswerable", "chitchat"), q.id


def _content_words(text: str) -> set[str]:
    import re

    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 4}


def test_paraphrase_queries_mostly_avoid_their_passages_wording() -> None:
    gold = load_gold_set(SHIPPED)

    for q in gold.queries:
        if q.case != "paraphrase":
            continue
        asked = _content_words(q.query)
        shared = asked & _content_words(" ".join(p.text for p in q.gold_passages))
        # A paraphrase may keep a product name or two, not the passage's phrasing
        assert len(shared) <= 2, (q.id, sorted(shared))


def test_missing_case_is_rejected_with_its_name(tmp_path: Path) -> None:
    data = _raw()
    del data["queries"][2]["case"]

    with pytest.raises(GoldSetError) as exc:
        load_gold_set(_write(tmp_path, data))

    message = str(exc.value)
    assert "case" in message
    assert "queries[2]" in message


def test_a_non_chitchat_slice_with_fewer_than_ten_queries_is_rejected(tmp_path: Path) -> None:
    data = _raw()
    factoid = [q for q in data["queries"] if q["slice"] == "factoid"]
    data["queries"] = [q for q in data["queries"] if q["slice"] != "factoid"] + factoid[:9]

    with pytest.raises(GoldSetError, match="slice 'factoid' has 9 queries, needs at least 10"):
        load_gold_set(_write(tmp_path, data))


def test_chitchat_needs_no_minimum_number_of_queries(tmp_path: Path) -> None:
    data = _raw()
    chitchat = [q for q in data["queries"] if q["slice"] == "chitchat"]
    data["queries"] = [q for q in data["queries"] if q["slice"] != "chitchat"] + chitchat[:1]

    load_gold_set(_write(tmp_path, data))


def test_shipped_gold_set_has_ten_queries_in_every_non_chitchat_slice() -> None:
    from collections import Counter

    gold = load_gold_set(SHIPPED)

    counts = Counter(q.slice for q in gold.queries if q.slice != "chitchat")
    assert min(counts.values()) >= 10, counts


def test_expected_entity_absent_from_the_corpus_is_rejected(tmp_path: Path) -> None:
    data = _raw()
    data["queries"][0]["expected_entities"] = ["Zyxquorp Engine"]

    with pytest.raises(
        GoldSetError, match="expected entity 'Zyxquorp Engine' is not in the corpus"
    ):
        load_gold_set(_write(tmp_path, data))


def test_expected_entity_matches_the_corpus_ignoring_case(tmp_path: Path) -> None:
    data = _raw()
    word = data["corpus"][0]["text"].split()[0]
    data["queries"][0]["expected_entities"] = [word.upper()]

    load_gold_set(_write(tmp_path, data))
