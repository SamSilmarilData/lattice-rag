"""Document lookup and delete, observed through a real LatticeStore (retrieval and lookup)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lattice_rag.ingestion.ingester import (
    DocumentExistsError,
    DocumentIngester,
    DocumentNotFoundError,
)
from lattice_rag.storage.db import ChunkData, EntityData, LatticeStore, RelationData

DIM = 384


@pytest.fixture
def store(tmp_path: Path):
    s = LatticeStore(db_path=tmp_path / "delete.db", vector_dimensions=DIM)
    yield s
    s.close()


def _vec(axis: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    return v


def _chunk(text: str, axis: int, position: int = 0) -> ChunkData:
    return ChunkData(text=text, embedding=_vec(axis), position=position)


def _entity_names(store: LatticeStore) -> set[str]:
    return {n["name"] for n in store.get_subgraph_snapshot(limit=500).nodes}


def test_deleted_document_is_unretrievable(store: LatticeStore):
    store.ingest_document_bundle("doc_a", "A", [_chunk("Zebras graze on savanna grass.", 0)], [])
    store.ingest_document_bundle("doc_b", "B", [_chunk("Penguins swim in icy water.", 1)], [])

    assert store.delete_document("doc_a") is True

    assert store.has_document("doc_a") is False
    assert store.bm25_search("zebras", top_k=5) == []
    assert all(r.metadata.get("doc_id") != "doc_a" for r in store.vector_search(_vec(0), top_k=5))
    # The other document is untouched
    assert store.has_document("doc_b") is True
    assert [r.metadata["doc_id"] for r in store.bm25_search("penguins", top_k=5)] == ["doc_b"]


def test_delete_removes_orphaned_entities_and_keeps_shared_ones(store: LatticeStore):
    store.ingest_document_bundle(
        "doc_a",
        "A",
        [_chunk("LatticeDB uses HNSW.", 0), _chunk("LatticeDB also uses BM25.", 1, 1)],
        [
            (
                0,
                [EntityData("LatticeDB", "tech"), EntityData("HNSW", "algorithm")],
                [RelationData("LatticeDB", "HNSW", "USES")],
            ),
            (1, [EntityData("LatticeDB", "tech"), EntityData("BM25", "algorithm")], []),
        ],
    )
    store.ingest_document_bundle(
        "doc_b",
        "B",
        [_chunk("LatticeDB is embedded.", 2)],
        [(0, [EntityData("LatticeDB", "tech")], [])],
    )

    store.delete_document("doc_a")

    assert _entity_names(store) == {"LatticeDB"}
    assert store.get_subgraph_snapshot(limit=500).edges == []
    # The surviving entity is still reachable from doc_b's chunk
    (hit,) = store.bm25_search("embedded", top_k=5)
    assert store.get_entities_for_chunks([hit.node_id]) != []


def test_reingest_after_delete_recreates_orphaned_entities(store: LatticeStore):
    bundle = [(0, [EntityData("HNSW", "algorithm")], [])]
    store.ingest_document_bundle("doc_a", "A", [_chunk("HNSW graphs.", 0)], bundle)
    store.delete_document("doc_a")

    store.ingest_document_bundle("doc_a", "A", [_chunk("HNSW graphs.", 0)], bundle)

    assert _entity_names(store) == {"HNSW"}
    (hit,) = store.bm25_search("graphs", top_k=5)
    assert len(store.get_entities_for_chunks([hit.node_id])) == 1


def test_delete_evicts_orphan_from_entity_cache_and_reingest_creates_new_entity(
    store: LatticeStore,
):
    bundle = [(0, [EntityData("HNSW", "algorithm"), EntityData("LatticeDB", "tech")], [])]
    store.ingest_document_bundle("doc_a", "A", [_chunk("HNSW graphs.", 0)], bundle)
    store.ingest_document_bundle(
        "doc_b",
        "B",
        [_chunk("LatticeDB is embedded.", 1)],
        [(0, [EntityData("LatticeDB", "tech")], [])],
    )
    old_hnsw = store.get_canonical_entity_id("HNSW", "algorithm")
    shared = store.get_canonical_entity_id("LatticeDB", "tech")
    assert old_hnsw is not None and shared is not None

    store.delete_document("doc_a")

    # The orphan is evicted under its (name, type) key; the shared entity stays cached
    assert store.get_canonical_entity_id("HNSW", "algorithm") is None
    assert store.get_canonical_entity_id("LatticeDB", "tech") == shared

    store.ingest_document_bundle("doc_a", "A", [_chunk("HNSW graphs.", 0)], bundle)
    new_hnsw = store.get_canonical_entity_id("  hnsw ", "ALGORITHM")
    assert new_hnsw is not None and new_hnsw != old_hnsw
    assert store.get_canonical_entity_id("LatticeDB", "tech") == shared
    (hit,) = store.bm25_search("graphs", top_k=5)
    assert set(store.get_entities_for_chunks([hit.node_id])) == {new_hnsw, shared}


class _Embedder:
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [_vec(i % DIM) for i, _ in enumerate(texts)]


class _Extractor:
    def extract_entities_batch(self, texts: list[str]) -> list[list[EntityData]]:
        return [[EntityData("LatticeDB", "tech")] for _ in texts]

    def extract_triples(self, text: str, entities: list[EntityData]) -> list[RelationData]:
        return []


class _Cache:
    def __init__(self) -> None:
        self.clears = 0

    def clear(self) -> None:
        self.clears += 1


async def test_ingester_conflict_delete_and_reingest_on_real_store(store: LatticeStore):
    cache = _Cache()
    ingester = DocumentIngester(
        store=store, embedding_service=_Embedder(), extractor=_Extractor(), cache=cache
    )
    await ingester.ingest(doc_id="doc_a", title="A", text="LatticeDB stores graphs.")

    with pytest.raises(DocumentExistsError):
        await ingester.ingest(doc_id="doc_a", title="A", text="LatticeDB stores graphs.")

    clears_before = cache.clears
    await ingester.delete("doc_a")
    assert cache.clears == clears_before + 1
    assert store.bm25_search("graphs", top_k=5) == []
    assert _entity_names(store) == set()

    with pytest.raises(DocumentNotFoundError):
        await ingester.delete("doc_a")

    await ingester.ingest(doc_id="doc_a", title="A", text="LatticeDB stores graphs.")
    assert store.has_document("doc_a") is True


def test_delete_unknown_document_returns_false(store: LatticeStore):
    store.ingest_document_bundle("doc_a", "A", [_chunk("Alpha.", 0)], [])
    assert store.delete_document("missing") is False
    assert store.has_document("doc_a") is True
