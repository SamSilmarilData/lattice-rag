"""Unit tests for configuration loading and validation."""
from __future__ import annotations

import os
from unittest.mock import patch

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
