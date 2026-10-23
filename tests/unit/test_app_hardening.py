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
