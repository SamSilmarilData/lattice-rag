"""LLM_POOL configuration: the Model Pool order, members and Gemini thinking settings."""

from __future__ import annotations

import pytest

from lattice_rag.config import ModelSpec, load_config
from lattice_rag.vocabulary import GenerationProvider

_POOL_ENV = ("LLM_POOL", "GROQ_MODEL", "GEMINI_MODEL", "GEMINI_THINKING")


@pytest.fixture(autouse=True)
def _clean_pool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _POOL_ENV:
        monkeypatch.delenv(name, raising=False)


def test_default_pool_is_two_groq_models_then_gemini_flash_lite():
    cfg = load_config(require_typesafe=False)
    assert cfg.llm_pool == (
        ModelSpec(GenerationProvider.GROQ, "qwen/qwen3.8-27b"),
        ModelSpec(GenerationProvider.GROQ, "openai/gpt-oss-120b"),
        # Flash-Lite rejects thinking_budget=0 (400, probed 2026-10-27): send no config
        ModelSpec(GenerationProvider.GEMINI, "gemini-3.5-flash-lite", thinking=None),
    )


def test_groq_and_gemini_model_still_pick_the_default_pool_members(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    cfg = load_config(require_typesafe=False)
    assert cfg.llm_pool == (
        ModelSpec(GenerationProvider.GROQ, "openai/gpt-oss-20b"),
        ModelSpec(GenerationProvider.GROQ, "openai/gpt-oss-120b"),
        ModelSpec(GenerationProvider.GEMINI, "gemini-3.8-flash", thinking="low"),
    )


def test_default_pool_does_not_list_a_model_twice(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    cfg = load_config(require_typesafe=False)
    assert [m.model for m in cfg.llm_pool] == ["openai/gpt-oss-120b", "gemini-3.5-flash-lite"]


def test_gemini_thinking_env_overrides_the_default_pool_setting(monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING", "low")
    cfg = load_config(require_typesafe=False)
    assert cfg.llm_pool[-1] == ModelSpec(
        GenerationProvider.GEMINI, "gemini-3.5-flash-lite", thinking="low"
    )


def test_llm_pool_sets_order_members_and_thinking(monkeypatch):
    monkeypatch.setenv(
        "LLM_POOL",
        "gemini:gemini-3.8-flash@none, groq:openai/gpt-oss-20b,gemini:gemini-2.5-flash",
    )
    cfg = load_config(require_typesafe=False)
    assert cfg.llm_pool == (
        ModelSpec(GenerationProvider.GEMINI, "gemini-3.8-flash", thinking=None),
        ModelSpec(GenerationProvider.GROQ, "openai/gpt-oss-20b"),
        ModelSpec(GenerationProvider.GEMINI, "gemini-2.5-flash", thinking="0"),
    )


@pytest.mark.parametrize("raw", ["none", "NONE", " none "])
def test_gemini_thinking_none_sends_no_thinking_config(monkeypatch, raw):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_THINKING", raw)
    cfg = load_config(require_typesafe=False)
    assert cfg.llm_pool[-1] == ModelSpec(
        GenerationProvider.GEMINI, "gemini-3.8-flash", thinking=None
    )


def test_flash_lite_thinking_can_still_be_set_explicitly(monkeypatch):
    monkeypatch.setenv("LLM_POOL", "gemini:gemini-3.5-flash-lite@low")
    cfg = load_config(require_typesafe=False)
    assert cfg.llm_pool == (
        ModelSpec(GenerationProvider.GEMINI, "gemini-3.5-flash-lite", thinking="low"),
    )


@pytest.mark.parametrize(
    "raw",
    [
        "openai:gpt-5",  # unknown provider
        "groq",  # no model
        "groq:",  # empty model
        "groq:a,groq:a",  # duplicate member
        "groq:openai/gpt-oss-20b@low",  # thinking is Gemini-only
        "gemini:gemini-3.8-flash@lots",  # not a level, a budget or none
        "gemini:gemini-3.8-flash@default",  # the keyword is none
    ],
)
def test_invalid_llm_pool_is_rejected(monkeypatch, raw):
    monkeypatch.setenv("LLM_POOL", raw)
    with pytest.raises(ValueError, match="LLM_POOL"):
        load_config(require_typesafe=False)
