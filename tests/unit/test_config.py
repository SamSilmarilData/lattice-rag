"""Unit tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from lattice_rag.config import AppConfig, load_config
from lattice_rag.security import SecretStr


def test_load_config_success(monkeypatch: pytest.MonkeyPatch):
    """Verify load_config populates all fields with types and masked secrets."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test-typesafe-1234")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-groq-5678")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-gemini-9012")
    monkeypatch.setenv("PORT", "8888")

    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.typesafe_api_key, SecretStr)
    assert cfg.typesafe_api_key.get_secret_value() == "sk-test-typesafe-1234"
    assert "sk-****" in str(cfg.typesafe_api_key)
    assert cfg.port == 8888


def test_load_config_missing_typesafe_key_raises_exit(monkeypatch: pytest.MonkeyPatch):
    """Verify load_config fails fast if TYPESAFE_API_KEY is not set."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        load_config()
    assert "TYPESAFE_API_KEY" in str(exc_info.value)


def test_load_config_without_typesafe_requirement(monkeypatch: pytest.MonkeyPatch):
    """Storage-only commands (seed/ingest) can load config without a TypeSafe key."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    cfg = load_config(require_typesafe=False)
    assert not cfg.typesafe_api_key


def test_load_config_admin_key_and_cors(monkeypatch: pytest.MonkeyPatch):
    """Admin key and CORS origins are read from the environment."""
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://localhost:5173, https://studio.example.com")
    monkeypatch.setenv("QUERY_RATE_LIMIT_PER_MINUTE", "30")

    cfg = load_config()
    assert cfg.admin_api_key.get_secret_value() == "admin-secret"
    assert cfg.cors_allow_origins == ("http://localhost:5173", "https://studio.example.com")
    assert cfg.query_rate_limit_per_minute == 30


def test_load_config_defaults_are_locked_down(monkeypatch: pytest.MonkeyPatch):
    """Without explicit settings there is no admin key and no cross-origin access."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)

    cfg = load_config()
    assert not cfg.admin_api_key
    assert cfg.cors_allow_origins == ()
