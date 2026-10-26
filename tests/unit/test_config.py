"""Unit tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from lattice_rag.config import DEFAULT_RERANKER_ONNX_REVISION, AppConfig, load_config
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


def test_guardrail_skip_rerank_score_is_off_by_default(monkeypatch: pytest.MonkeyPatch):
    """The rerank-score Noul skip ships disabled until the Live Tier shows no quality loss."""
    monkeypatch.delenv("GUARDRAIL_SKIP_RERANK_SCORE", raising=False)

    assert load_config().guardrail_skip_rerank_score is None


def test_guardrail_skip_rerank_score_reads_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GUARDRAIL_SKIP_RERANK_SCORE", "4.5")

    assert load_config().guardrail_skip_rerank_score == 4.5


def test_guardrail_skip_rerank_score_empty_means_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GUARDRAIL_SKIP_RERANK_SCORE", "")

    assert load_config().guardrail_skip_rerank_score is None


def test_eval_degraded_retry_delay_defaults_to_five_seconds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_DEGRADED_RETRY_DELAY", raising=False)

    assert load_config().eval_degraded_retry_delay == 5.0


def test_eval_degraded_retry_delay_reads_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVAL_DEGRADED_RETRY_DELAY", "12.5")

    assert load_config().eval_degraded_retry_delay == 12.5


def test_eval_degraded_retry_delay_rejects_negative(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVAL_DEGRADED_RETRY_DELAY", "-1")

    with pytest.raises(ValueError, match="EVAL_DEGRADED_RETRY_DELAY"):
        load_config()


def test_reranker_revision_is_pinned_unless_overridden(monkeypatch: pytest.MonkeyPatch):
    """An unset or empty RERANKER_ONNX_REVISION uses the pinned commit, never 'latest'."""
    monkeypatch.setenv("RERANKER_ONNX_REVISION", "")
    assert load_config(require_typesafe=False).reranker_onnx_revision == (
        DEFAULT_RERANKER_ONNX_REVISION
    )
    monkeypatch.setenv("RERANKER_ONNX_REVISION", "abc123")
    assert load_config(require_typesafe=False).reranker_onnx_revision == "abc123"


# --- Phase 5 experiment switches (#47, #48) --------------------------------------------

_PHASE5_VARS = (
    "GRAPH_CONTEXT_IN_PROMPT",
    "GUARDRAIL_MASSIVE_CONTEXT_THRESHOLD",
    "GUARDRAIL_SKIP_MASSIVE_CONTEXT",
)


def test_phase5_switches_default_to_current_behaviour(monkeypatch: pytest.MonkeyPatch):
    for var in _PHASE5_VARS:
        monkeypatch.delenv(var, raising=False)

    cfg = load_config(require_typesafe=False)

    assert cfg.graph_context_in_prompt == "on"
    assert cfg.guardrail_massive_context_threshold == 0.5
    assert cfg.guardrail_skip_massive_context is False


def test_phase5_switches_read_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GRAPH_CONTEXT_IN_PROMPT", "Relational")
    monkeypatch.setenv("GUARDRAIL_MASSIVE_CONTEXT_THRESHOLD", "0.35")
    monkeypatch.setenv("GUARDRAIL_SKIP_MASSIVE_CONTEXT", "true")

    cfg = load_config(require_typesafe=False)

    assert cfg.graph_context_in_prompt == "relational"
    assert cfg.guardrail_massive_context_threshold == 0.35
    assert cfg.guardrail_skip_massive_context is True


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("GRAPH_CONTEXT_IN_PROMPT", "sometimes"),
        ("GUARDRAIL_MASSIVE_CONTEXT_THRESHOLD", "1.5"),
        ("GUARDRAIL_SKIP_MASSIVE_CONTEXT", "maybe"),
    ],
)
def test_phase5_switches_reject_invalid_values(monkeypatch: pytest.MonkeyPatch, var, value):
    monkeypatch.setenv(var, value)

    with pytest.raises(ValueError, match=var):
        load_config(require_typesafe=False)
