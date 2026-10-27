"""App-level hardening: hermetic boot without LLM keys, rate limiting, CORS defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from lattice_rag.app import create_app
from lattice_rag.config import AppConfig
from lattice_rag.security import SecretStr


def _config(tmp_path: Path, **overrides: object) -> AppConfig:
    values: dict[str, object] = {
        "typesafe_api_key": SecretStr("mock-typesafe"),
        "groq_api_key": SecretStr(""),
        "gemini_api_key": SecretStr(""),
        "latticedb_path": tmp_path / "test.db",
        "query_rate_limit_per_minute": 2,
    }
    values.update(overrides)
    return AppConfig(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_app_boots_without_llm_keys(monkeypatch, tmp_path):
    """Missing GROQ/GEMINI keys degrade generation instead of crashing the lifespan."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    app = create_app(_config(tmp_path))
    async with AsyncTestClient(app) as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["groqConfigured"] is False


@pytest.mark.asyncio
async def test_lifespan_gives_both_providers_the_llm_timeout(tmp_path):
    app = create_app(_config(tmp_path, llm_timeout_seconds=9.0))
    async with AsyncTestClient(app):
        orchestrator = app.state.orchestrator
        assert orchestrator.groq.timeout_seconds == 9.0
        assert orchestrator.gemini.timeout_seconds == 9.0


@pytest.mark.asyncio
async def test_query_endpoints_are_rate_limited(tmp_path):
    app = create_app(_config(tmp_path))
    async with AsyncTestClient(app) as client:
        statuses = [(await client.post("/api/v1/query", json={})).status_code for _ in range(3)]
        assert statuses[:2] == [400, 400]
        assert statuses[2] == 429
        # Health is not rate limited
        assert (await client.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_cors_is_same_origin_by_default(tmp_path):
    app = create_app(_config(tmp_path))
    async with AsyncTestClient(app) as client:
        resp = await client.get("/health", headers={"Origin": "https://evil.example.com"})
        assert "access-control-allow-origin" not in resp.headers


@pytest.mark.asyncio
async def test_cors_allows_configured_origin(tmp_path):
    app = create_app(_config(tmp_path, cors_allow_origins=("http://localhost:5173",)))
    async with AsyncTestClient(app) as client:
        resp = await client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_hermetic_fallback_keeps_env_settings(monkeypatch, tmp_path):
    """Without TYPESAFE_API_KEY only the key is mocked; DB path and admin key still apply."""
    from lattice_rag.app import _resolve_config

    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.setenv("ADMIN_API_KEY", "env-admin")
    monkeypatch.setenv("LATTICEDB_PATH", str(tmp_path / "env.db"))

    cfg = _resolve_config(None)
    assert cfg.admin_api_key.get_secret_value() == "env-admin"
    assert cfg.latticedb_path == tmp_path / "env.db"
    assert cfg.typesafe_api_key.get_secret_value() == "mock-typesafe-key-hermetic"


@pytest.mark.asyncio
async def test_mock_key_mode_is_reported_and_not_written_to_env(monkeypatch, tmp_path):
    """Booting without TYPESAFE_API_KEY reports degraded health and leaves os.environ alone."""
    import os

    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.setenv("LATTICEDB_PATH", str(tmp_path / "mock.db"))

    app = create_app()
    async with AsyncTestClient(app) as client:
        data = (await client.get("/health")).json()
        assert data["typesafeConfigured"] is False
        assert data["status"] == "degraded"
        assert os.environ.get("TYPESAFE_API_KEY") == ""


@pytest.mark.asyncio
async def test_mock_key_mode_makes_no_jev_calls_per_query(monkeypatch, tmp_path):
    """Degraded mode skips routing and the guardrail instead of paying for failing Jev calls,
    and the answer it produces is flagged unverified."""
    from unittest.mock import AsyncMock, MagicMock

    import numpy as np
    from typesafe_sdk import AsyncTypeSafeClient

    import lattice_rag.app as app_module
    from lattice_rag.retrieval.embeddings import RerankResult
    from lattice_rag.storage.db import SearchResult, SubgraphResult

    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.setenv("LATTICEDB_PATH", str(tmp_path / "mock.db"))

    store = MagicMock()
    store.vector_search.return_value = [
        SearchResult(node_id=1, score=0.9, text="LatticeDB uses HNSW.", metadata={})
    ]
    store.bm25_search.return_value = []
    store.get_entities_for_chunks.return_value = []
    store.traverse_from_entities.return_value = SubgraphResult(nodes=[], edges=[])
    embedder = MagicMock()
    embedder.embed_query.return_value = np.ones(384, dtype=np.float32)
    embedder.rerank.return_value = [RerankResult(index=0, score=5.0, text="LatticeDB uses HNSW.")]
    monkeypatch.setattr(app_module, "LatticeStore", lambda *a, **k: store)
    monkeypatch.setattr(app_module, "EmbeddingService", lambda *a, **k: embedder)
    monkeypatch.setattr(app_module, "EntityExtractor", lambda *a, **k: MagicMock())

    jev_calls = AsyncMock(side_effect=RuntimeError("network call in mock-key mode"))
    monkeypatch.setattr(AsyncTypeSafeClient, "system_one", jev_calls)

    app = create_app()
    async with AsyncTestClient(app) as client:
        resp = await client.post("/api/v1/query", json={"query": "How does vector search work?"})

    assert resp.status_code == 201
    assert resp.json()["unverified"] is True
    jev_calls.assert_not_awaited()


def test_mock_key_startup_message_matches_behavior(monkeypatch, tmp_path, caplog):
    from lattice_rag.app import _resolve_config

    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.setenv("LATTICEDB_PATH", str(tmp_path / "env.db"))

    with caplog.at_level("WARNING"):
        _resolve_config(None)

    message = caplog.text.lower()
    assert "not called" in message
    assert "unverified" in message


async def _query_statuses(client: AsyncTestClient, forwarded_for: list[str]) -> list[int]:
    statuses = []
    for header in forwarded_for:
        resp = await client.post("/api/v1/query", json={}, headers={"X-Forwarded-For": header})
        statuses.append(resp.status_code)
    return statuses


@pytest.mark.asyncio
async def test_forged_forwarded_for_does_not_change_rate_limit_identity(tmp_path):
    """With no trusted proxies, rotating X-Forwarded-For cannot dodge the limit."""
    app = create_app(_config(tmp_path))
    async with AsyncTestClient(app) as client:
        statuses = await _query_statuses(client, ["1.1.1.1", "2.2.2.2", "3.3.3.3"])
        assert statuses == [400, 400, 429]


@pytest.mark.asyncio
async def test_trusted_proxy_hop_identifies_client_and_ignores_spoofed_entries(tmp_path):
    app = create_app(_config(tmp_path, trusted_proxy_hops=1))
    async with AsyncTestClient(app) as client:
        # Same client behind the proxy, different spoofed prefixes: one identity
        statuses = await _query_statuses(
            client, ["198.51.100.4", "6.6.6.6, 198.51.100.4", "7.7.7.7, 198.51.100.4"]
        )
        assert statuses == [400, 400, 429]
        # A different real client keeps its own budget
        assert await _query_statuses(client, ["198.51.100.5"]) == [400]


def test_trusted_proxy_hops_read_from_env(monkeypatch):
    from lattice_rag.config import load_config

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    assert load_config(require_typesafe=False).trusted_proxy_hops == 2
    monkeypatch.delenv("TRUSTED_PROXY_HOPS")
    assert load_config(require_typesafe=False).trusted_proxy_hops == 0


def test_negative_trusted_proxy_hops_rejected(tmp_path):
    with pytest.raises(ValueError):
        _config(tmp_path, trusted_proxy_hops=-1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("skip_score", "expected_jev_calls"),
    [(None, 2), (4.0, 1)],  # off: route + Noul guardrail; on: the 5.0-scored chunk skips Noul
)
async def test_rerank_score_noul_skip_follows_the_setting(
    monkeypatch, tmp_path, skip_score, expected_jev_calls
):
    """GUARDRAIL_SKIP_RERANK_SCORE reaches the served pipeline; unset, every chunk goes to Noul."""
    from unittest.mock import AsyncMock, MagicMock

    import numpy as np
    from typesafe_sdk import AsyncTypeSafeClient

    import lattice_rag.app as app_module
    from lattice_rag.retrieval.embeddings import RerankResult
    from lattice_rag.storage.db import SearchResult, SubgraphResult

    store = MagicMock()
    store.vector_search.return_value = [
        SearchResult(node_id=1, score=0.9, text="LatticeDB uses HNSW.", metadata={})
    ]
    store.bm25_search.return_value = []
    store.get_entities_for_chunks.return_value = []
    store.traverse_from_entities.return_value = SubgraphResult(nodes=[], edges=[])
    embedder = MagicMock()
    embedder.embed_query.return_value = np.ones(384, dtype=np.float32)
    embedder.rerank.return_value = [RerankResult(index=0, score=5.0, text="LatticeDB uses HNSW.")]
    monkeypatch.setattr(app_module, "LatticeStore", lambda *a, **k: store)
    monkeypatch.setattr(app_module, "EmbeddingService", lambda *a, **k: embedder)
    monkeypatch.setattr(app_module, "EntityExtractor", lambda *a, **k: MagicMock())
    # Every Jev call fails fast (no network): routing degrades to hybrid, Noul to unverified
    jev_calls = AsyncMock(side_effect=RuntimeError("no network in unit tests"))
    monkeypatch.setattr(AsyncTypeSafeClient, "system_one", jev_calls)

    app = create_app(_config(tmp_path, guardrail_skip_rerank_score=skip_score))
    async with AsyncTestClient(app) as client:
        resp = await client.post("/api/v1/query", json={"query": "How does vector search work?"})

    assert resp.status_code == 201
    assert jev_calls.await_count == expected_jev_calls


@pytest.mark.asyncio
async def test_eval_runner_uses_configured_degraded_retry_delay(tmp_path):
    app = create_app(_config(tmp_path, eval_degraded_retry_delay=9.0))
    async with AsyncTestClient(app):
        assert app.state.eval_runner.degraded_retry_delay == 9.0
