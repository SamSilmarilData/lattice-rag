"""Unit tests for GLiNER entity and relation triple extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

from lattice_rag.storage.extract import EntityExtractor


def test_entity_extraction_with_mock_model():
    """Verify EntityExtractor parses entities and derives relation triples correctly."""
    extractor = EntityExtractor(model_name="test-model")
    mock_gliner = MagicMock()
    mock_gliner.batch_predict_entities.return_value = [
        [
            {"text": "LatticeDB", "label": "database", "score": 0.95},
            {"text": "FastEmbed", "label": "technology", "score": 0.92},
            {"text": "LatticeDB", "label": "database", "score": 0.91},  # duplicate to verify dedup
        ]
    ]
    extractor._model = mock_gliner

    text = "LatticeDB integrates with FastEmbed for vector similarity search."
    entities = extractor.extract_entities(text)

    # Verify extraction and deduplication
    assert len(entities) == 2
    assert entities[0].name == "LatticeDB"
    assert entities[0].entity_type == "database"
    assert entities[1].name == "FastEmbed"
    assert entities[1].entity_type == "technology"

    # Verify triple extraction via sentence co-occurrence
    triples = extractor.extract_triples(text, entities)
    assert len(triples) == 1
    assert triples[0].source_name == "LatticeDB"
    assert triples[0].target_name == "FastEmbed"
    assert "DATABASE_TECHNOLOGY" in triples[0].relation_type


def test_extract_triples_ignores_substring_matches():
    """An entity named "DB" must not co-occur with a sentence that only mentions "LatticeDB"."""
    from lattice_rag.storage.db import EntityData

    extractor = EntityExtractor()
    entities = [
        EntityData(name="LatticeDB", entity_type="database"),
        EntityData(name="DB", entity_type="concept"),
        EntityData(name="HNSW", entity_type="algorithm"),
    ]
    triples = extractor.extract_triples("LatticeDB uses HNSW.", entities)

    pairs = {(t.source_name, t.target_name) for t in triples}
    assert pairs == {("LatticeDB", "HNSW")}


def test_extract_triples_matches_names_with_symbols():
    """Whole-token matching still works for names like C++ that end in non-word characters."""
    from lattice_rag.storage.db import EntityData

    extractor = EntityExtractor()
    entities = [
        EntityData(name="C++", entity_type="technology"),
        EntityData(name="ONNX", entity_type="technology"),
    ]
    triples = extractor.extract_triples("ONNX runtime is written in C++.", entities)
    assert {(t.source_name, t.target_name) for t in triples} == {("C++", "ONNX")}
