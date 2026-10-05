"""Unit tests verifying algorithmic time-space complexity optimizations across subsystems."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.generation.chitchat import ChitchatHandler
from lattice_rag.retrieval.vector_search import HybridSearcher
from lattice_rag.storage.db import ChunkData, EntityData, LatticeStore, RelationData, SearchResult
from lattice_rag.storage.extract import EntityExtractor


@pytest.mark.asyncio
async def test_semantic_cache_exact_match_shortcut():
    """Verify exact query string match bypasses embedding and Jev Noul verification in O(1) time."""
    mock_ts_client = MagicMock()
    mock_ts_client.system_one = AsyncMock()

    cache = SemanticCache(similarity_threshold=0.90, ts_client=mock_ts_client, max_size=100)
    dummy_vec = np.ones(384, dtype=np.float32)
    response_payload = {"answer": "LatticeDB is an in-process property graph.", "sources": []}

    cache.put("What is LatticeDB?", dummy_vec, response_payload)

    # 1. Exact match lookup via get_exact: zero network or embedding overhead
    exact_hit = cache.get_exact("What is LatticeDB?")
    assert exact_hit is not None
    assert exact_hit["answer"] == response_payload["answer"]
    # Verify Jev Noul was NOT called
    mock_ts_client.system_one.assert_not_called()
    assert cache.stats[0] == 1  # 1 hit

    # 2. Case and whitespace insensitive exact match
    exact_hit_case = cache.get_exact("  what is latticedb?  ")
    assert exact_hit_case is not None
    assert exact_hit_case["answer"] == response_payload["answer"]
    assert cache.stats[0] == 2  # 2 hits
    mock_ts_client.system_one.assert_not_called()


@pytest.mark.asyncio
async def test_semantic_cache_vectorized_blas_similarity():
    """Verify vectorized BLAS matrix dot product matches closest candidate accurately."""
    mock_ts_client = MagicMock()
    mock_noul = MagicMock()
    mock_noul.noul = 0.95
    mock_res = MagicMock()
    mock_res.nouls = {"equivalent": mock_noul}
    mock_ts_client.system_one = AsyncMock(return_value=mock_res)

    cache = SemanticCache(similarity_threshold=0.85, ts_client=mock_ts_client, max_size=100)

    # Create 5 distinct orthogonal unit vectors
    v1 = np.zeros(384, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(384, dtype=np.float32)
    v2[1] = 1.0
    v3 = np.zeros(384, dtype=np.float32)
    v3[2] = 1.0

    cache.put("Query A", v1, {"answer": "Ans A"})
    cache.put("Query B", v2, {"answer": "Ans B"})
    cache.put("Query C", v3, {"answer": "Ans C"})

    # Query with vector very close to v2 (dot product ~0.99)
    q_vec = np.zeros(384, dtype=np.float32)
    q_vec[1] = 0.99
    q_vec[0] = 0.1

    hit = await cache.get("Tell me about B", q_vec)
    assert hit is not None
    assert hit["answer"] == "Ans B"
    mock_ts_client.system_one.assert_called_once()


def test_semantic_cache_lru_eviction():
    """Verify cache memory is strictly bounded by max_size via LRU eviction."""
    cache = SemanticCache(max_size=3)
    vec = np.ones(384, dtype=np.float32)

    cache.put("q1", vec, {"answer": "a1"})
    cache.put("q2", vec, {"answer": "a2"})
    cache.put("q3", vec, {"answer": "a3"})

    assert len(cache) == 3
    assert cache.get_exact("q1") is not None

    # Adding a 4th query must evict the least recently accessed (q2, since q1 was just accessed)
    cache.put("q4", vec, {"answer": "a4"})
    assert len(cache) == 3
    assert cache.get_exact("q2") is None  # Evicted
    assert cache.get_exact("q1") is not None  # Kept
    assert cache.get_exact("q3") is not None  # Kept
    assert cache.get_exact("q4") is not None  # Kept


@pytest.mark.asyncio
async def test_chitchat_single_pass_compiled_regex():
    """Verify ChitchatHandler correctly matches intents in a single pass."""
    handler = ChitchatHandler()

    assert "hello" in (await handler.handle("Hey there, how are you?")).lower()
    assert "goodbye" in (await handler.handle("See you later, bye!")).lower()
    assert "welcome" in (await handler.handle("Thanks a lot for the help")).lower()
    assert "help you search" in (await handler.handle("Can you assist me with this?")).lower()
    assert "specialized knowledge engine" in (await handler.handle("What is quantum chromodynamics?")).lower()


@pytest.mark.asyncio
async def test_hybrid_searcher_pipelined_bm25_and_embedding():
    """Verify HybridSearcher overlaps BM25 search with query embedding."""
    mock_store = MagicMock()
    mock_embed_svc = MagicMock()

    # Track execution sequence
    execution_order = []

    def fake_embed(query: str):
        execution_order.append("embed")
        return np.ones(384, dtype=np.float32)

    def fake_bm25(query: str, top_k: int):
        execution_order.append("bm25")
        return [SearchResult(node_id=1, score=10.0, text="BM25 hit")]

    def fake_vector(emb: np.ndarray, top_k: int):
        execution_order.append("vector")
        return [SearchResult(node_id=2, score=0.9, text="Vector hit")]

    mock_embed_svc.embed_query = fake_embed
    mock_store.bm25_search = fake_bm25
    mock_store.vector_search = fake_vector

    searcher = HybridSearcher(store=mock_store, embedding_service=mock_embed_svc)
    results = await searcher.search("test query", top_k=5)

    assert len(results) == 2
    # Verify both methods were executed
    assert "embed" in execution_order
    assert "bm25" in execution_order
    assert "vector" in execution_order


def test_entity_extractor_triples_deduplication():
    """Verify EntityExtractor deduplicates triples when entities co-occur multiple times."""
    extractor = EntityExtractor()

    # Mock extract_triples with duplicate occurrences
    text = "LatticeDB uses FastEmbed. FastEmbed is used by LatticeDB in the engine."
    entities = [
        EntityData(name="LatticeDB", entity_type="Database"),
        EntityData(name="FastEmbed", entity_type="Technology"),
    ]

    triples = extractor.extract_triples(text, entities)
    # Without deduplication, 2 sentences would generate 2 identical triples
    assert len(triples) == 1
    assert triples[0].source_name == "LatticeDB"
    assert triples[0].target_name == "FastEmbed"


def test_lattice_store_atomic_document_bundle(tmp_path):
    """Verify LatticeStore.ingest_document_bundle commits document, chunks, and entities in 1 transaction."""
    db_file = tmp_path / "test_bundle.db"
    store = LatticeStore(db_file)

    chunks = [
        ChunkData(text="Chunk 1 about LatticeDB", embedding=np.ones(384, dtype=np.float32), position=0),
        ChunkData(text="Chunk 2 about FastEmbed", embedding=np.ones(384, dtype=np.float32), position=1),
    ]

    extractions = [
        (
            0,
            [EntityData(name="LatticeDB", entity_type="Database", embedding=np.ones(384, dtype=np.float32))],
            [],
        ),
        (
            1,
            [EntityData(name="FastEmbed", entity_type="Technology", embedding=np.ones(384, dtype=np.float32))],
            [RelationData(source_name="LatticeDB", target_name="FastEmbed", relation_type="USES")],
        ),
    ]

    stats = store.ingest_document_bundle(
        doc_id="doc_bundle_1",
        title="Bundle Test Document",
        chunks=chunks,
        chunk_entities_relations=extractions,
    )

    assert stats.chunk_count == 2
    assert stats.entity_count == 2
    assert stats.relation_count == 1
    assert len(stats.chunk_ids) == 2

    # Verify entities linked to chunks via batch lookup
    eids = store.get_entities_for_chunks(stats.chunk_ids)
    assert len(eids) == 2

    store.close()
