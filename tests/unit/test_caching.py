"""Unit tests for Tier 1 Semantic Cache and Tier 2 Fallback Cache with Circuit Breaker."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pybreaker
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
    mock_response = SimpleNamespace(
        nouls={"equivalent": SimpleNamespace(noul=0.95)}
    )
    mock_client.system_one.return_value = mock_response

    cache = SemanticCache(similarity_threshold=0.90, ts_client=mock_client)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    stored_response = {"answer": "LatticeDB is an embedded graph database.", "route": "vector_exact"}

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
    mock_response = SimpleNamespace(
        nouls={"equivalent": SimpleNamespace(noul=0.45)}
    )
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
# Tier 2: FallbackCache & Circuit Breaker Unit Tests
# =========================================================================

@pytest.mark.asyncio
async def test_fallback_cache_embedded_connect_and_seed():
    """Verify FallbackCache boots embedded fakeredis without external Redis or API keys."""
    fb = FallbackCache(redis_url="embedded")
    await fb.connect()

    assert fb.circuit_status == "closed"

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


@pytest.mark.asyncio
async def test_fallback_circuit_breaker_transitions():
    """Verify circuit breaker transitions: closed -> fail 3 times -> open -> blocks calls."""
    fb = FallbackCache(redis_url="embedded")

    # Define mock failing and succeeding coroutines
    fail_count = 0

    async def flaky_llm():
        nonlocal fail_count
        fail_count += 1
        raise RuntimeError(f"Simulated Groq 503 error #{fail_count}")

    async def healthy_llm():
        return "LLM response successfully generated"

    # Initially closed
    assert fb.circuit_status == "closed"

    # Successful call passes through breaker
    healthy_res = await fb.call_with_breaker(healthy_llm)
    assert healthy_res == "LLM response successfully generated"
    assert fb.circuit_status == "closed"

    # Trip breaker by calling failing coroutine 3 times
    for i in range(3):
        with pytest.raises((RuntimeError, pybreaker.CircuitBreakerError)):
            await fb.call_with_breaker(flaky_llm)

    # Breaker is now OPEN
    assert fb.circuit_status == "open"

    # Next call should be blocked immediately with CircuitBreakerError without executing flaky_llm
    calls_before = fail_count
    with pytest.raises(pybreaker.CircuitBreakerError):
        await fb.call_with_breaker(flaky_llm)
    assert fail_count == calls_before  # flaky_llm was NOT called

    await fb.close()
