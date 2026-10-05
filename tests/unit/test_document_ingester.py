"""Unit tests for the deep DocumentIngester module (TDD RED -> GREEN)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

from lattice_rag.ingestion.ingester import DocumentIngester
from lattice_rag.storage.db import ChunkData, EntityData, IngestStats, RelationData


class FakeStore:
    """Mock store tracking document and entity insertions."""

    def __init__(self):
        self.documents = {}
        self.entities = {}

    def ingest_document(self, doc_id: str, title: str, chunks: list[ChunkData]) -> IngestStats:
        self.documents[doc_id] = {"title": title, "chunks": chunks}
        chunk_ids = list(range(100, 100 + len(chunks)))
        return IngestStats(chunk_count=len(chunks), entity_count=0, relation_count=0, chunk_ids=chunk_ids)

    def ingest_entities(self, chunk_id: int, entities: list[EntityData], relations: list[RelationData]) -> IngestStats:
        self.entities[chunk_id] = {"entities": entities, "relations": relations}
        return IngestStats(chunk_count=0, entity_count=len(entities), relation_count=len(relations))


class FakeEmbeddingService:
    """Mock embedding service producing deterministic vectors."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.batch_calls = []

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        self.batch_calls.append(texts)
        return [np.ones(self.dim, dtype=np.float32) * (i + 1) for i, _ in enumerate(texts)]


class FakeExtractor:
    """Mock entity and relation extractor."""

    def extract_entities(self, text: str) -> list[EntityData]:
        if "LatticeDB" in text:
            return [
                EntityData(name="LatticeDB", entity_type="database"),
                EntityData(name="HNSW", entity_type="algorithm"),
            ]
        return []

    def extract_triples(self, text: str, entities: list[EntityData]) -> list[RelationData]:
        if len(entities) >= 2:
            return [RelationData(source_name="LatticeDB", target_name="HNSW", relation_type="USES")]
        return []


class FakeCache:
    """Mock semantic cache verifying invalidation."""

    def __init__(self):
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True


@pytest.mark.asyncio
async def test_document_ingester_empty_text():
    store = FakeStore()
    embed_svc = FakeEmbeddingService()
    extractor = FakeExtractor()
    cache = FakeCache()

    ingester = DocumentIngester(
        store=store,
        embedding_service=embed_svc,
        extractor=extractor,
        cache=cache,
    )

    stats = await ingester.ingest(doc_id="doc_empty", title="Empty Doc", text="")
    assert stats.chunk_count == 0
    assert stats.entity_count == 0
    assert stats.relation_count == 0
    assert len(store.documents) == 0


@pytest.mark.asyncio
async def test_document_ingester_single_chunk():
    store = FakeStore()
    embed_svc = FakeEmbeddingService()
    extractor = FakeExtractor()
    cache = FakeCache()

    ingester = DocumentIngester(
        store=store,
        embedding_service=embed_svc,
        extractor=extractor,
        cache=cache,
    )

    text = "LatticeDB is an embedded database that implements HNSW vector indexing."
    stats = await ingester.ingest(doc_id="doc_single", title="LatticeDB Doc", text=text)

    # Ingest stats validation
    assert stats.chunk_count == 1
    assert stats.entity_count == 2
    assert stats.relation_count == 1

    # Store verification
    assert "doc_single" in store.documents
    assert len(store.documents["doc_single"]["chunks"]) == 1
    assert store.documents["doc_single"]["chunks"][0].text == text

    # Entity verification
    assert 100 in store.entities
    assert len(store.entities[100]["entities"]) == 2
    assert store.entities[100]["entities"][0].name == "LatticeDB"
    assert store.entities[100]["entities"][1].name == "HNSW"
    assert len(store.entities[100]["relations"]) == 1

    # Cache invalidation check
    assert cache.cleared is True


@pytest.mark.asyncio
async def test_document_ingester_batch_entity_embedding_efficiency():
    store = FakeStore()
    embed_svc = FakeEmbeddingService()
    extractor = FakeExtractor()

    ingester = DocumentIngester(
        store=store,
        embedding_service=embed_svc,
        extractor=extractor,
    )

    # Text that generates 2 chunks
    p1 = "LatticeDB combines HNSW vector similarity with BM25 full-text indexing in a single file."
    p2 = "LatticeDB also provides Cypher property-graph traversals on HNSW indices."
    text = f"{p1}\n\n{p2}"

    stats = await ingester.ingest(
        doc_id="doc_multi",
        title="Multi Chunk Doc",
        text=text,
        target_chunk_size=90,
        chunk_overlap=20,
    )

    assert stats.chunk_count >= 2
    # Verify entity embeddings were batched across the entire document
    # Calls: 1 for chunks batch, 1 for all unique entity names batch
    assert len(embed_svc.batch_calls) == 2
    chunk_call = embed_svc.batch_calls[0]
    entity_call = embed_svc.batch_calls[1]
    assert len(chunk_call) == stats.chunk_count
    # Unique entities: LatticeDB and HNSW
    assert "LatticeDB" in entity_call
    assert "HNSW" in entity_call


@pytest.mark.asyncio
async def test_document_ingester_ingest_file(tmp_path: Path):
    store = FakeStore()
    embed_svc = FakeEmbeddingService()
    extractor = FakeExtractor()

    ingester = DocumentIngester(
        store=store,
        embedding_service=embed_svc,
        extractor=extractor,
    )

    test_file = tmp_path / "sample_architecture.txt"
    test_file.write_text("LatticeDB uses HNSW vector index.", encoding="utf-8")

    stats = await ingester.ingest_file(test_file)
    assert stats.chunk_count == 1
    assert "sample_architecture" in store.documents
    assert store.documents["sample_architecture"]["title"] == "sample_architecture.txt"
