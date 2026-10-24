"""Unit tests for Tier 1 Semantic Cache and Tier 2 Fallback Cache with Circuit Breaker."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lattice_rag.caching.fallback_cache import FallbackCache
from lattice_rag.caching.semantic_cache import SemanticCache

# =========================================================================
# Tier 1: SemanticCache Unit Tests
# =========================================================================


@pytest.mark.asyncio
async def test_semantic_cache_empty():
    """Verify empty semantic cache returns None and increments misses."""
    cache = SemanticCache()
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    result = await cache.get("What is LatticeDB?", emb)
    assert result is None
    assert cache.stats == (0, 1)


@pytest.mark.asyncio
async def test_semantic_cache_hit_with_noul_verification():
    """Verify semantic cache returns response when cosine >= 0.90 and Noul >= 0.70."""
    mock_client = AsyncMock()
    mock_response = SimpleNamespace(nouls={"equivalent": SimpleNamespace(noul=0.95)})
    mock_client.system_one.return_value = mock_response

    cache = SemanticCache(similarity_threshold=0.90, ts_client=mock_client)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    stored_response = {
        "answer": "LatticeDB is an embedded graph database.",
        "route": "vector_exact",
    }

    cache.put("What is LatticeDB?", emb, stored_response)

    # Identical vector (cosine = 1.0)
    hit = await cache.get("What is LatticeDB?", emb)
    assert hit == stored_response
    assert cache.stats == (1, 0)
    mock_client.system_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_semantic_cache_miss_low_cosine():
    """Verify cosine < 0.90 returns None without calling TypeSafe Noul."""
    mock_client = AsyncMock()
    cache = SemanticCache(similarity_threshold=0.90, ts_client=mock_client)

    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Orthogonal vector (cosine = 0.0)
    emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    cache.put("What is LatticeDB?", emb1, {"answer": "..."})

    result = await cache.get("Tell me about apples", emb2)
    assert result is None
    assert cache.stats == (0, 1)
    mock_client.system_one.assert_not_called()


@pytest.mark.asyncio
async def test_semantic_cache_miss_noul_rejected():
    """Verify cosine >= 0.90 but Noul < 0.70 rejects false positive and records miss."""
    mock_client = AsyncMock()
    # High cosine but Noul detects semantic difference
    mock_response = SimpleNamespace(nouls={"equivalent": SimpleNamespace(noul=0.45)})
    mock_client.system_one.return_value = mock_response

    cache = SemanticCache(similarity_threshold=0.90, ts_client=mock_client)

    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.99, 0.01, 0.0], dtype=np.float32)

    cache.put("How to install LatticeDB?", emb1, {"answer": "pip install latticedb"})

    result = await cache.get("How to uninstall LatticeDB?", emb2)
    assert result is None
    assert cache.stats == (0, 1)
    mock_client.system_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_semantic_cache_clear_and_aclose():
    """Verify clear empties the cache and resets stats; aclose cleans up client."""
    mock_client = AsyncMock()
    cache = SemanticCache(ts_client=mock_client)

    cache.put("q1", np.array([1.0, 0.0]), {"ans": "1"})
    cache.clear()
    assert cache.stats == (0, 0)
    assert len(cache._cache) == 0

    await cache.aclose()
    mock_client.aclose.assert_awaited_once()


# =========================================================================
# Tier 2: FallbackCache Unit Tests
# =========================================================================


@pytest.mark.asyncio
async def test_fallback_cache_embedded_connect_and_seed():
    """Verify FallbackCache boots embedded fakeredis without external Redis or API keys."""
    fb = FallbackCache(redis_url="embedded")
    await fb.connect()

    queries = {
        "What is Litestar?": "Litestar is an ASGI framework.",
        "What is FastEmbed?": "FastEmbed is an ONNX text embedding library.",
    }
    await fb.seed_fallback(queries)

    # Retrieval by raw query
    ans1 = await fb.get_fallback("What is Litestar?")
    assert ans1 == "Litestar is an ASGI framework."

    # Retrieval case-insensitive
    ans2 = await fb.get_fallback("what is fastembed?")
    assert ans2 == "FastEmbed is an ONNX text embedding library."

    # Non-existent query returns None
    assert await fb.get_fallback("Non existent query") is None

    count = await fb.cached_query_count()
    assert count == 2

    await fb.close()


@pytest.mark.asyncio
async def test_fallback_cache_seed_from_file_and_fuzzy_match():
    """Verify seed_from_file loads fallback_queries.json and fuzzy matches queries."""
    fb = FallbackCache(redis_url="embedded")
    await fb.connect()

    json_path = Path("fallback_queries.json")
    assert json_path.exists(), "fallback_queries.json must exist in project root"

    seeded_count = await fb.seed_from_file(json_path)
    assert seeded_count == 50

    # Direct query retrieval
    ans = await fb.get_fallback("What is LatticeDB?")
    assert ans is not None
    assert "property-graph" in ans

    # Fuzzy match with slight variations/typos
    fuzzy_ans = await fb.find_closest_fallback("what is latticedb database")
    assert fuzzy_ans is not None
    assert "property-graph" in fuzzy_ans

    await fb.close()


def _unit(i: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def test_semantic_cache_duplicate_put_replaces_entry():
    """Two concurrent misses on the same query both put; that must not create two entries."""
    cache = SemanticCache(ts_client=AsyncMock(), max_size=3, vector_dimensions=8)
    cache.put("What is X?", _unit(0), {"answer": "first"})
    cache.put("what is x?  ", _unit(0), {"answer": "second"})

    assert len(cache) == 1
    assert cache.get_exact("What is X?") == {"answer": "second"}


def test_semantic_cache_eviction_stays_consistent_with_duplicates():
    """Eviction never raises and evicted queries disappear from both indexes."""
    cache = SemanticCache(ts_client=AsyncMock(), max_size=2, vector_dimensions=8)
    cache.put("a", _unit(0), {"answer": "a"})
    cache.put("a", _unit(0), {"answer": "a2"})
    cache.put("b", _unit(1), {"answer": "b"})
    cache.put("c", _unit(2), {"answer": "c"})  # evicts "a" (least recently used)
    cache.put("d", _unit(3), {"answer": "d"})  # evicts "b"

    assert len(cache) == 2
    assert cache.get_exact("a") is None
    assert cache.get_exact("b") is None
    assert {e.query_text for e in cache._cache} == {"c", "d"}
    assert set(cache._exact_map) == {"c", "d"}
    # Matrix rows line up with the surviving entries
    for i, entry in enumerate(cache._cache):
        assert np.allclose(cache._embeddings_matrix[i], entry.query_embedding)


def _rand_unit(rng: np.random.Generator, dim: int = 16) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.asyncio
async def test_semantic_cache_lookups_stay_correct_under_eviction_churn():
    """After many replacements and evictions every surviving query still finds its own answer."""
    ts_client = AsyncMock()
    ts_client.system_one.return_value = SimpleNamespace(
        nouls={"equivalent": SimpleNamespace(noul=0.99)}
    )
    cache = SemanticCache(ts_client=ts_client, max_size=5, vector_dimensions=16)
    rng = np.random.default_rng(7)
    vectors: dict[str, np.ndarray] = {}
    for step in range(40):
        key = f"q{rng.integers(0, 12)}"
        vectors[key] = _rand_unit(rng)
        cache.put(key, vectors[key], {"answer": key, "step": step})
        if step % 3 == 0:
            cache.get_exact(f"q{rng.integers(0, 12)}")  # refresh LRU order

    assert len(cache) == 5
    survivors = {e.query_text for e in cache._cache}
    assert len(survivors) == 5
    for key in survivors:
        hit = await cache.get(f"paraphrase of {key}", vectors[key])
        assert hit is not None and hit["answer"] == key


def test_semantic_cache_put_cost_does_not_grow_with_cache_size():
    """Writing to a full cache (eviction, then replacement) must not scan every entry."""

    def put_cost(size: int) -> float:
        cache = SemanticCache(ts_client=AsyncMock(), max_size=size, vector_dimensions=8)
        for i in range(size):
            cache.put(f"query number {i}", _unit(i), {"answer": i})
        best = float("inf")
        for i in range(size, size + 50):
            start = time.perf_counter()
            cache.put(f"query number {i}", _unit(i), {"answer": i})  # evicts the LRU entry
            cache.put(f"query number {i}", _unit(i + 1), {"answer": -i})  # replaces it
            best = min(best, time.perf_counter() - start)
        return best

    small, large = put_cost(10), put_cost(50_000)
    assert large < small * 20, f"put on 50k entries took {large / small:.0f}x a 10-entry put"
