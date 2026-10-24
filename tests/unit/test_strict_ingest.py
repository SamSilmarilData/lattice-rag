"""Strict ingest: embedding dimension, chunk references and entity identity.

Behavior is observed through ingest results and retrieval on a real LatticeStore.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from lattice_rag.ingestion.ingester import DocumentIngester
from lattice_rag.storage.db import (
    ChunkData,
    EmbeddingDimensionError,
    EntityData,
    LatticeStore,
    RelationData,
    UnresolvedChunkReferenceError,
)

DIM = 384


@pytest.fixture
def store(tmp_path: Path) -> Iterator[LatticeStore]:
    s = LatticeStore(db_path=tmp_path / "strict.db", vector_dimensions=DIM)
    yield s
    s.close()


class FixedDimEmbedder:
    """Embeds every text as a unit vector of the configured dimension."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        vec = np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)
        return [vec.copy() for _ in texts]


class TableExtractor:
    """Returns a fixed entity list for every chunk and links consecutive entities."""

    def __init__(self, entities: list[EntityData]) -> None:
        self.entities = entities

    def extract_entities_batch(self, texts: list[str]) -> list[list[EntityData]]:
        return [list(self.entities) for _ in texts]

    def extract_triples(self, text: str, entities: list[EntityData]) -> list[RelationData]:
        return []


def _vec(dim: int = DIM) -> np.ndarray:
    return np.ones(dim, dtype=np.float32) / np.sqrt(dim)


def _graph_entities(store: LatticeStore) -> list[tuple[str, str]]:
    return sorted((n["name"], n["label"]) for n in store.get_subgraph_snapshot(limit=100).nodes)


async def test_wrong_chunk_embedding_dimension_raises_and_writes_nothing(store: LatticeStore):
    ingester = DocumentIngester(
        store=store,
        embedding_service=FixedDimEmbedder(dim=DIM // 2),
        extractor=TableExtractor([EntityData(name="LatticeDB", entity_type="database")]),
    )

    with pytest.raises(EmbeddingDimensionError, match=str(DIM)):
        await ingester.ingest(doc_id="doc_dim", title="T", text="LatticeDB stores vectors.")

    assert store.has_document("doc_dim") is False
    assert store.bm25_search("LatticeDB", top_k=5) == []
    assert _graph_entities(store) == []


def test_wrong_entity_embedding_dimension_raises_and_writes_nothing(store: LatticeStore):
    with pytest.raises(EmbeddingDimensionError):
        store.ingest_document_bundle(
            "doc_ent_dim",
            "T",
            [ChunkData(text="LatticeDB stores vectors.", embedding=_vec(), position=0)],
            [(0, [EntityData(name="LatticeDB", entity_type="database", embedding=_vec(8))], [])],
        )

    assert store.has_document("doc_ent_dim") is False
    assert store.vector_search(_vec(), top_k=5) == []
    assert _graph_entities(store) == []


@pytest.mark.parametrize("bad_ref", [1, -1, 999])
def test_unresolved_chunk_reference_raises_and_writes_nothing(store: LatticeStore, bad_ref: int):
    with pytest.raises(UnresolvedChunkReferenceError, match=str(bad_ref)):
        store.ingest_document_bundle(
            "doc_ref",
            "T",
            [ChunkData(text="LatticeDB stores vectors.", embedding=_vec(), position=0)],
            [(bad_ref, [EntityData(name="LatticeDB", entity_type="database")], [])],
        )

    assert store.has_document("doc_ref") is False
    assert store.bm25_search("LatticeDB", top_k=5) == []
    assert _graph_entities(store) == []


def test_chunk_reference_resolves_to_its_own_chunk(store: LatticeStore):
    stats = store.ingest_document_bundle(
        "doc_two",
        "T",
        [
            ChunkData(text="First chunk.", embedding=_vec(), position=0),
            ChunkData(text="Second chunk mentions HNSW.", embedding=_vec(), position=1),
        ],
        [(1, [EntityData(name="HNSW", entity_type="algorithm")], [])],
    )

    first, second = stats.chunk_ids
    assert store.get_entities_for_chunks([first]) == []
    assert len(store.get_entities_for_chunks([second])) == 1


async def test_same_name_different_type_stay_distinct_across_ingests(store: LatticeStore):
    embedder = FixedDimEmbedder(DIM)
    first = DocumentIngester(
        store=store,
        embedding_service=embedder,
        extractor=TableExtractor([EntityData(name="Mercury", entity_type="planet")]),
    )
    second = DocumentIngester(
        store=store,
        embedding_service=embedder,
        extractor=TableExtractor([EntityData(name="Mercury", entity_type="element")]),
    )

    s1 = await first.ingest(doc_id="astro", title="A", text="Mercury orbits the sun.")
    s2 = await second.ingest(doc_id="chem", title="C", text="Mercury is a liquid metal.")

    assert (s1.entity_count, s2.entity_count) == (1, 1)
    assert _graph_entities(store) == [("Mercury", "element"), ("Mercury", "planet")]
    assert store.get_entities_for_chunks(s1.chunk_ids) != store.get_entities_for_chunks(
        s2.chunk_ids
    )


def test_same_name_different_type_stay_distinct_within_one_bundle(store: LatticeStore):
    stats = store.ingest_document_bundle(
        "doc_mixed",
        "T",
        [
            ChunkData(text="Apple released a phone.", embedding=_vec(), position=0),
            ChunkData(text="An apple fell from the tree.", embedding=_vec(), position=1),
        ],
        [
            (0, [EntityData(name="Apple", entity_type="organization")], []),
            (1, [EntityData(name="Apple", entity_type="fruit")], []),
        ],
    )

    assert stats.entity_count == 2
    assert _graph_entities(store) == [("Apple", "fruit"), ("Apple", "organization")]


async def test_same_name_and_type_differing_in_case_and_whitespace_merge(store: LatticeStore):
    embedder = FixedDimEmbedder(DIM)
    first = DocumentIngester(
        store=store,
        embedding_service=embedder,
        extractor=TableExtractor([EntityData(name="LatticeDB", entity_type="database")]),
    )
    second = DocumentIngester(
        store=store,
        embedding_service=embedder,
        extractor=TableExtractor([EntityData(name="  latticedb\t", entity_type=" Database")]),
    )
    # Same name and type spelled differently inside one document merge too
    third = DocumentIngester(
        store=store,
        embedding_service=embedder,
        extractor=TableExtractor(
            [
                EntityData(name="LATTICEDB", entity_type="DATABASE"),
                EntityData(name="latticedb", entity_type="database"),
            ]
        ),
    )

    s1 = await first.ingest(doc_id="d1", title="T", text="LatticeDB is embedded.")
    s2 = await second.ingest(doc_id="d2", title="T", text="It is a lattice DB.")
    s3 = await third.ingest(doc_id="d3", title="T", text="LATTICEDB again.")

    assert (s1.entity_count, s2.entity_count, s3.entity_count) == (1, 0, 0)
    assert _graph_entities(store) == [("LatticeDB", "database")]
    shared = store.get_entities_for_chunks(s1.chunk_ids)
    assert store.get_entities_for_chunks(s2.chunk_ids) == shared
    assert store.get_entities_for_chunks(s3.chunk_ids) == shared


def test_entity_identity_survives_reopening_the_store(tmp_path: Path):
    path = tmp_path / "reopen.db"
    first = LatticeStore(db_path=path, vector_dimensions=DIM)
    first.ingest_document_bundle(
        "d1",
        "T",
        [ChunkData(text="Mercury orbits.", embedding=_vec(), position=0)],
        [(0, [EntityData(name="Mercury", entity_type="planet")], [])],
    )
    first.close()

    reopened = LatticeStore(db_path=path, vector_dimensions=DIM)
    try:
        stats = reopened.ingest_document_bundle(
            "d2",
            "T",
            [ChunkData(text="Mercury is a metal; mercury orbits.", embedding=_vec(), position=0)],
            [
                (
                    0,
                    [
                        EntityData(name="Mercury", entity_type="element"),
                        EntityData(name=" mercury", entity_type="Planet"),
                    ],
                    [],
                )
            ],
        )
        assert stats.entity_count == 1
        assert _graph_entities(reopened) == [("Mercury", "element"), ("Mercury", "planet")]
    finally:
        reopened.close()


def test_ingest_entities_for_unknown_chunk_raises_and_writes_nothing(store: LatticeStore):
    stats = store.ingest_document(
        "doc_legacy", "T", [ChunkData(text="Some chunk.", embedding=_vec(), position=0)]
    )
    not_a_chunk = max(stats.chunk_ids) + 1000

    with pytest.raises(UnresolvedChunkReferenceError):
        store.ingest_entities(
            not_a_chunk, [EntityData(name="LatticeDB", entity_type="database")], []
        )

    assert _graph_entities(store) == []
