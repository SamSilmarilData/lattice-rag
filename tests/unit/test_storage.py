"""Unit tests for LatticeStore in-process storage engine."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from lattice_rag.storage.db import (
    ChunkData,
    EntityData,
    LatticeStore,
    RelationData,
)


@pytest.fixture
def temp_store(tmp_path: Path):
    """Provide a fresh LatticeStore instance in a temporary directory."""
    db_file = tmp_path / "test_lattice.db"
    store = LatticeStore(db_path=db_file, vector_dimensions=384)
    yield store
    store.close()


def test_lattice_store_initialization(temp_store: LatticeStore):
    """Verify database file is created, opened, and FTS indices are ensured."""
    assert temp_store.db.is_open
    assert temp_store.db_path.exists()
    assert temp_store.db.has_node_fts_index("Chunk", "text")


def test_document_and_chunk_ingestion(temp_store: LatticeStore):
    """Verify document and chunk creation with HNSW vector setting."""
    vec1 = np.ones(384, dtype=np.float32)
    vec1 = vec1 / np.linalg.norm(vec1)

    chunks = [
        ChunkData(text="LatticeDB is an embedded graph database.", embedding=vec1, position=0),
        ChunkData(text="It combines Cypher, BM25, and HNSW vector search.", embedding=vec1, position=1),
    ]

    stats = temp_store.ingest_document(doc_id="doc_1", title="LatticeDB Overview", chunks=chunks)
    assert stats.chunk_count == 2

    # Verify vector search recalls the chunks
    results = temp_store.vector_search(vec1, top_k=2)
    assert len(results) >= 1
    assert any("LatticeDB" in r.text for r in results)
    assert results[0].score > 0.5


def test_bm25_full_text_search(temp_store: LatticeStore):
    """Verify native BM25 inverted index search."""
    vec = np.zeros(384, dtype=np.float32)
    chunks = [
        ChunkData(text="The quick brown fox jumps over the lazy dog.", embedding=vec, position=0),
        ChunkData(text="Quantum computing utilizes qubits for superposition.", embedding=vec, position=1),
    ]
    temp_store.ingest_document(doc_id="doc_2", title="Sample Text", chunks=chunks)

    # Search for quantum
    results = temp_store.bm25_search("quantum", top_k=2)
    assert len(results) >= 1
    assert "Quantum" in results[0].text


def test_entity_ingestion_and_multi_hop_traversal(temp_store: LatticeStore):
    """Verify entity node creation, CONTAINS edge, RELATION edge, and multi-hop traversal."""
    vec = np.ones(384, dtype=np.float32)
    chunks = [
        ChunkData(text="FastEmbed runs ONNX models on CPU.", embedding=vec, position=0),
    ]
    temp_store.ingest_document(doc_id="doc_3", title="FastEmbed Info", chunks=chunks)

    # Find the chunk node ID via vector search
    chunk_res = temp_store.vector_search(vec, top_k=1)
    assert len(chunk_res) >= 1
    chunk_id = chunk_res[0].node_id

    # Ingest entities
    entities = [
        EntityData(name="FastEmbed", entity_type="TECHNOLOGY", embedding=vec),
        EntityData(name="ONNX Runtime", entity_type="ENGINE", embedding=vec),
        EntityData(name="CPU", entity_type="HARDWARE", embedding=vec),
    ]
    relations = [
        RelationData(source_name="FastEmbed", target_name="ONNX Runtime", relation_type="USES_RUNTIME"),
        RelationData(source_name="ONNX Runtime", target_name="CPU", relation_type="TARGETS_HARDWARE"),
    ]

    ent_stats = temp_store.ingest_entities(chunk_id=chunk_id, entities=entities, relations=relations)
    assert ent_stats.entity_count == 3
    assert ent_stats.relation_count == 2

    # Check entities linked to chunk
    linked_entities = temp_store.get_entities_for_chunk(chunk_id)
    assert len(linked_entities) == 3

    # Traverse 2 hops from the first entity (FastEmbed -> ONNX Runtime -> CPU)
    subgraph = temp_store.traverse_from_entities(entity_ids=[linked_entities[0]], max_hops=2, budget=10)
    assert len(subgraph.nodes) >= 2
    assert len(subgraph.edges) >= 1


def test_canonical_entity_deduplication(temp_store: LatticeStore):
    """Verify that identical entities across different chunks reuse the canonical node ID."""
    vec = np.zeros(384, dtype=np.float32)
    chunks = [
        ChunkData(text="Chunk 1 talks about LatticeDB.", embedding=vec, position=0),
        ChunkData(text="Chunk 2 also mentions LatticeDB.", embedding=vec, position=1),
    ]
    stats = temp_store.ingest_document(doc_id="doc_canon", title="Canonical Test", chunks=chunks)
    c1, c2 = stats.chunk_ids[0], stats.chunk_ids[1]

    # Ingest LatticeDB entity for chunk 1
    temp_store.ingest_entities(
        chunk_id=c1,
        entities=[EntityData(name="LatticeDB", entity_type="database", embedding=vec)],
        relations=[],
    )
    ent_id_1 = temp_store.get_canonical_entity_id("LatticeDB")
    assert ent_id_1 is not None

    # Ingest LatticeDB entity again for chunk 2 (case-insensitive)
    temp_store.ingest_entities(
        chunk_id=c2,
        entities=[EntityData(name="latticedb", entity_type="database", embedding=vec)],
        relations=[],
    )
    ent_id_2 = temp_store.get_canonical_entity_id("latticedb")
    # Must reuse the same node ID!
    assert ent_id_1 == ent_id_2

    # Both chunks must be connected to this single canonical entity
    c1_ents = temp_store.get_entities_for_chunk(c1)
    c2_ents = temp_store.get_entities_for_chunk(c2)
    assert ent_id_1 in c1_ents
    assert ent_id_1 in c2_ents
