"""Atomic replace-on-ingest, observed through a real LatticeStore's retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lattice_rag.ingestion.ingester import DocumentExistsError, DocumentIngester
from lattice_rag.storage.db import EntityData, LatticeStore, RelationData

DIM = 384


@pytest.fixture
def store(tmp_path: Path):
    s = LatticeStore(db_path=tmp_path / "replace.db", vector_dimensions=DIM)
    yield s
    s.close()


def _vec(axis: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    return v


class _Embedder:
    """Embeds each text on an axis chosen by its first word, so old and new text differ."""

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [_vec(sum(map(ord, t.split()[0])) % DIM) for t in texts]


class _Extractor:
    """Extracts every capitalized word as a ``concept`` entity."""

    def extract_entities_batch(self, texts: list[str]) -> list[list[EntityData]]:
        return [
            [EntityData(w.strip(".,"), "concept") for w in t.split() if w[:1].isupper()]
            for t in texts
        ]

    def extract_triples(self, text: str, entities: list[EntityData]) -> list[RelationData]:
        if len(entities) < 2:
            return []
        return [RelationData(entities[0].name, entities[1].name, "RELATED_TO")]


class _Cache:
    def __init__(self) -> None:
        self.clears = 0

    def clear(self) -> None:
        self.clears += 1


@pytest.fixture
def cache() -> _Cache:
    return _Cache()


@pytest.fixture
def ingester(store: LatticeStore, cache: _Cache) -> DocumentIngester:
    return DocumentIngester(
        store=store, embedding_service=_Embedder(), extractor=_Extractor(), cache=cache
    )


OLD_TEXT = "Zebras graze on Savanna grass."
NEW_TEXT = "Penguins swim in Antarctic water."


def _entity_names(store: LatticeStore) -> set[str]:
    return {n["name"] for n in store.get_subgraph_snapshot(limit=500).nodes}


def _doc_hits(store: LatticeStore, query: str) -> list[str]:
    return [r.metadata.get("doc_id", "") for r in store.bm25_search(query, top_k=5)]


async def test_replace_serves_new_content_and_none_of_the_old(
    store: LatticeStore, ingester: DocumentIngester
):
    await ingester.ingest(doc_id="doc_a", title="Old", text=OLD_TEXT)

    stats = await ingester.ingest(doc_id="doc_a", title="New", text=NEW_TEXT, replace=True)

    assert stats.chunk_count == 1
    assert _doc_hits(store, "penguins") == ["doc_a"]
    assert _doc_hits(store, "zebras") == []
    old_axis = _Embedder().embed_texts([OLD_TEXT])[0]
    assert all("Zebras" not in r.text for r in store.vector_search(old_axis, top_k=5))
    assert _entity_names(store) == {"Penguins", "Antarctic"}


async def test_without_replace_an_existing_id_still_conflicts(
    store: LatticeStore, ingester: DocumentIngester
):
    await ingester.ingest(doc_id="doc_a", title="Old", text=OLD_TEXT)

    with pytest.raises(DocumentExistsError):
        await ingester.ingest(doc_id="doc_a", title="New", text=NEW_TEXT)

    assert _doc_hits(store, "zebras") == ["doc_a"]
    assert _doc_hits(store, "penguins") == []


async def test_replace_clears_tier1(ingester: DocumentIngester, cache: _Cache):
    await ingester.ingest(doc_id="doc_a", title="Old", text=OLD_TEXT)
    clears_before = cache.clears

    await ingester.ingest(doc_id="doc_a", title="New", text=NEW_TEXT, replace=True)

    assert cache.clears == clears_before + 1


async def test_replace_of_an_unknown_id_inserts_it(store: LatticeStore, ingester: DocumentIngester):
    await ingester.ingest(doc_id="doc_new", title="New", text=NEW_TEXT, replace=True)

    assert _doc_hits(store, "penguins") == ["doc_new"]


async def test_failed_replace_leaves_the_original_fully_retrievable(
    store: LatticeStore, ingester: DocumentIngester, cache: _Cache, monkeypatch
):
    await ingester.ingest(doc_id="doc_a", title="Old", text=OLD_TEXT)
    old_entity = store.get_canonical_entity_id("Zebras", "concept")
    clears_before = cache.clears

    def fail_mid_write(*args, **kwargs):
        raise RuntimeError("disk full")

    # Fails in-transaction, after the old document was deleted and the new chunks created
    monkeypatch.setattr(store, "_write_entity_group", fail_mid_write)
    with pytest.raises(RuntimeError):
        await ingester.ingest(doc_id="doc_a", title="New", text=NEW_TEXT, replace=True)
    monkeypatch.undo()

    assert store.has_document("doc_a") is True
    assert _doc_hits(store, "zebras") == ["doc_a"]
    assert _doc_hits(store, "penguins") == []
    (hit,) = store.bm25_search("zebras", top_k=5)
    assert old_entity in store.get_entities_for_chunks([hit.node_id])
    assert store.get_canonical_entity_id("Zebras", "concept") == old_entity
    assert _entity_names(store) == {"Zebras", "Savanna"}
    assert cache.clears == clears_before


async def test_entity_kept_across_replace_is_linked_to_a_live_node(
    store: LatticeStore, ingester: DocumentIngester
):
    await ingester.ingest(doc_id="doc_a", title="Old", text=OLD_TEXT)

    await ingester.ingest(
        doc_id="doc_a", title="New", text="Lions hunt on Savanna plains.", replace=True
    )

    savanna = store.get_canonical_entity_id("Savanna", "concept")
    assert savanna is not None
    (hit,) = store.bm25_search("lions", top_k=5)
    assert savanna in store.get_entities_for_chunks([hit.node_id])
    assert _entity_names(store) == {"Lions", "Savanna"}
    assert [e["relation_type"] for e in store.get_subgraph_snapshot(limit=500).edges] == [
        "RELATED_TO"
    ]
