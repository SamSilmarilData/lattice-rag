"""Route-aware answer length and truncation logging, observed through the orchestrator."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from structlog.testing import capture_logs

from lattice_rag.generation.gemini_fallback import GeminiFallback
from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.routing.router import RouteDecision
from lattice_rag.storage.db import SubgraphResult


def _groq_client(finish_reason: str = "stop"):
    """Fake AsyncGroq: records create() kwargs; returns one answer or a token stream."""
    client = MagicMock()

    async def create(**kwargs):
        if kwargs["stream"]:

            async def gen():
                for tok, fr in (("Part one.", None), (" Part two", finish_reason)):
                    delta = SimpleNamespace(content=tok)
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=fr)])

            return gen()
        msg = SimpleNamespace(content="Answer.")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])

    client.chat.completions.create = AsyncMock(side_effect=create)
    return client


def _gemini_client(finish_reason: str = "STOP"):
    client = MagicMock()
    cand = SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))

    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(text="Answer.", candidates=[cand])
    )

    async def stream(**kwargs):
        yield SimpleNamespace(text="Part one.", candidates=[])
        yield SimpleNamespace(text=" Part two", candidates=[cand])

    client.aio.models.generate_content_stream = MagicMock(side_effect=stream)
    return client


def _orchestrator(route: str, groq: GroqSynthesizer, gemini: GeminiFallback):
    router = MagicMock()
    router.route_query = AsyncMock(return_value=RouteDecision(route=route, confidence=0.9))
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.put = MagicMock()
    pipeline = MagicMock()
    pipeline.execute = AsyncMock(
        return_value=MagicMock(
            final_chunks=[{"node_id": 1, "text": "evidence", "score": 0.9}],
            graph_context=SubgraphResult(nodes=[], edges=[]),
            timings={},
            low_grounding=False,
            unverified=False,
        )
    )
    embed = MagicMock()
    embed.embed_query = MagicMock(return_value=np.ones(384, dtype=np.float32))
    return RAGOrchestrator(
        router=router,
        semantic_cache=cache,
        fallback_cache=MagicMock(),
        retrieval_pipeline=pipeline,
        groq=groq,
        gemini=gemini,
        chitchat=MagicMock(),
        embedding_service=embed,
    )


async def _ask(orch: RAGOrchestrator, entry_point: str) -> None:
    if entry_point == "run":
        await orch.run("q", use_cache=False)
    else:
        _ = [e async for e in orch.stream_query("q")]


def _groq_budget(client: MagicMock) -> int:
    return int(client.chat.completions.create.call_args.kwargs["max_tokens"])


def _gemini_budget(client: MagicMock) -> int:
    fn = client.aio.models.generate_content
    if not fn.called:
        fn = client.aio.models.generate_content_stream
    return int(fn.call_args.kwargs["config"]["max_output_tokens"])


@pytest.mark.parametrize("entry_point", ["run", "stream"])
async def test_massive_context_answer_budget_exceeds_default(entry_point):
    gem = _gemini_client()
    orch = _orchestrator(
        "massive_context", GroqSynthesizer(client=_groq_client()), GeminiFallback(client=gem)
    )
    await _ask(orch, entry_point)
    assert _gemini_budget(gem) == 1536


@pytest.mark.parametrize("entry_point", ["run", "stream"])
async def test_massive_context_groq_fallback_keeps_the_larger_budget(entry_point):
    groq = _groq_client()
    broken = MagicMock()
    broken.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("down"))
    broken.aio.models.generate_content_stream = MagicMock(side_effect=RuntimeError("down"))
    orch = _orchestrator(
        "massive_context", GroqSynthesizer(client=groq), GeminiFallback(client=broken)
    )
    await _ask(orch, entry_point)
    assert _groq_budget(groq) == 1536


@pytest.mark.parametrize("entry_point", ["run", "stream"])
async def test_default_answer_budget_is_768(entry_point):
    groq = _groq_client()
    orch = _orchestrator(
        "hybrid", GroqSynthesizer(client=groq), GeminiFallback(client=_gemini_client())
    )
    await _ask(orch, entry_point)
    assert _groq_budget(groq) == 768


@pytest.mark.parametrize("entry_point", ["run", "stream"])
async def test_explicit_max_tokens_overrides_default(entry_point):
    groq = _groq_client()
    orch = _orchestrator(
        "hybrid",
        GroqSynthesizer(client=groq, max_tokens=300),
        GeminiFallback(client=_gemini_client()),
    )
    await _ask(orch, entry_point)
    assert _groq_budget(groq) == 300


@pytest.mark.parametrize("entry_point", ["run", "stream"])
async def test_gemini_default_cap_is_explicit(entry_point):
    gem = _gemini_client()
    broken = MagicMock()
    broken.chat.completions.create = AsyncMock(side_effect=RuntimeError("down"))
    orch = _orchestrator("hybrid", GroqSynthesizer(client=broken), GeminiFallback(client=gem))
    await _ask(orch, entry_point)
    assert _gemini_budget(gem) == 768


@pytest.mark.parametrize("entry_point", ["run", "stream"])
@pytest.mark.parametrize("provider", ["groq", "gemini"])
async def test_length_stop_is_logged(entry_point, provider):
    if provider == "groq":
        orch = _orchestrator(
            "hybrid",
            GroqSynthesizer(client=_groq_client("length")),
            GeminiFallback(client=_gemini_client()),
        )
    else:
        orch = _orchestrator(
            "massive_context",
            GroqSynthesizer(client=_groq_client()),
            GeminiFallback(client=_gemini_client("MAX_TOKENS")),
        )
    with capture_logs() as logs:
        await _ask(orch, entry_point)
    hits = [e for e in logs if e["event"] == "generation_truncated"]
    assert len(hits) == 1
    assert hits[0]["provider"] == provider


@pytest.mark.parametrize("entry_point", ["run", "stream"])
async def test_normal_stop_is_not_logged(entry_point):
    orch = _orchestrator(
        "hybrid", GroqSynthesizer(client=_groq_client()), GeminiFallback(client=_gemini_client())
    )
    with capture_logs() as logs:
        await _ask(orch, entry_point)
    assert not [e for e in logs if e["event"] == "generation_truncated"]


def test_llm_max_tokens_env_overrides_config_default(monkeypatch):
    from lattice_rag.config import get_config

    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    get_config.cache_clear()
    assert get_config().llm_max_tokens == 768
    monkeypatch.setenv("LLM_MAX_TOKENS", "400")
    get_config.cache_clear()
    assert get_config().llm_max_tokens == 400
