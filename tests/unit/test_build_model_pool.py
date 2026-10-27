"""build_model_pool: config -> provider adapters, and what each adapter sends its API."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice_rag.api.dtos import GenerationProvider
from lattice_rag.config import AppConfig, ModelSpec
from lattice_rag.generation.gemini_fallback import GeminiFallback
from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
from lattice_rag.generation.model_pool import build_model_pool
from lattice_rag.security import SecretStr


def _config(*pool: ModelSpec, **overrides) -> AppConfig:
    base = AppConfig(
        typesafe_api_key=SecretStr("t"),
        groq_api_key=SecretStr("gsk-test"),
        gemini_api_key=SecretStr("AIza-test"),
        latticedb_path=Path("unused.db"),
        llm_max_tokens=500,
        llm_timeout_seconds=7.0,
    )
    return dataclasses.replace(base, llm_pool=pool or base.llm_pool, **overrides)


def test_pool_members_follow_the_configured_order_with_their_adapters():
    pool = build_model_pool(
        _config(
            ModelSpec("groq", "qwen/qwen3.8-27b"),
            ModelSpec("groq", "openai/gpt-oss-120b"),
            ModelSpec("gemini", "gemini-3.5-flash-lite"),
        )
    )

    assert [(m.provider, m.model) for m in pool] == [
        (GenerationProvider.GROQ, "qwen/qwen3.8-27b"),
        (GenerationProvider.GROQ, "openai/gpt-oss-120b"),
        (GenerationProvider.GEMINI, "gemini-3.5-flash-lite"),
    ]
    assert isinstance(pool[0].client, GroqSynthesizer)
    assert isinstance(pool[2].client, GeminiFallback)
    for member in pool:
        assert member.client.model == member.model
        assert member.client.timeout_seconds == 7.0
    assert pool[0].client.max_tokens == 500
    assert pool[2].client.max_output_tokens == 500


def _gemini_client() -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(text="Answer.", candidates=[])
    )

    async def chunks(**_kw):
        yield SimpleNamespace(text="Answer.", candidates=[])

    client.aio.models.generate_content_stream = MagicMock(side_effect=chunks)
    return client


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        ("0", {"thinking_budget": 0}),
        ("low", {"thinking_level": "LOW"}),
        (None, None),
    ],
)
async def test_gemini_sends_its_thinking_setting_on_both_paths(thinking, expected):
    client = _gemini_client()
    gemini = GeminiFallback(client=client, model="gemini-x", thinking=thinking)

    await gemini.synthesize("q", [])
    _ = [t async for t in gemini.stream("q", [])]

    for fn in (client.aio.models.generate_content, client.aio.models.generate_content_stream):
        assert fn.call_args.kwargs["config"].get("thinking_config") == expected


async def test_groq_rate_limit_raises_at_once_so_the_pool_moves_on(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("Error code: 429 rate_limit_exceeded")
    )
    groq = GroqSynthesizer(client=client)

    with pytest.raises(RuntimeError, match="Groq synthesis failed"):
        await groq.synthesize("q", [])
    with pytest.raises(RuntimeError, match="Groq stream synthesis failed"):
        _ = [t async for t in groq.stream("q", [])]

    assert client.chat.completions.create.call_count == 2
    sleep.assert_not_awaited()


def test_groq_sdk_client_does_not_retry_on_its_own():
    assert GroqSynthesizer(api_key="test-key").client.max_retries == 0
