"""RAGOrchestrator.run() reports which provider answered, and the Live Tier acts on it."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lattice_rag.api.dtos import EvalQueryError, EvalQueryResult, GenerationProvider
from lattice_rag.eval.runner import EvalRunner
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.routing.router import RouteDecision


class RateLimitError(Exception):
    """Stands in for a Groq 429."""


class FakeGroq:
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail

    async def synthesize(self, query, chunks, graph, **kwargs):
        if self.fail:
            raise RateLimitError("429 Too Many Requests")
        return "Groq answer."


class FakeGemini:
    model = "gemini-3.5-flash-lite"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def synthesize(self, query, chunks, graph, **kwargs):
        if self.fail:
            raise RateLimitError("503 high demand")
        return "Gemini answer."


def _orchestrator(
    *, groq_fails: bool = False, gemini_fails: bool = False, route: str = "hybrid"
) -> RAGOrchestrator:
    router = MagicMock()
    router.route_query = AsyncMock(return_value=RouteDecision(route=route, confidence=0.9))
    cache = MagicMock()
    cache.get_exact = MagicMock(return_value=None)
    cache.get = AsyncMock(return_value=None)
    retrieval = MagicMock()
    retrieval.execute = AsyncMock(
        return_value=MagicMock(
            final_chunks=[{"node_id": 1, "text": "evidence"}],
            graph_context=None,
            low_grounding=False,
            unverified=False,
        )
    )
    chitchat = MagicMock()
    chitchat.handle = AsyncMock(return_value="Hello!")
    embed = MagicMock()
    embed.embed_query = MagicMock(return_value=np.ones(384, dtype=np.float32))
    faq = MagicMock()
    faq.find_closest_fallback = AsyncMock(return_value="FAQ answer.")
    return RAGOrchestrator(
        router=router,
        semantic_cache=cache,
        fallback_cache=faq,
        retrieval_pipeline=retrieval,
        groq=FakeGroq(fail=groq_fails),
        gemini=FakeGemini(fail=gemini_fails),
        chitchat=chitchat,
        embedding_service=embed,
    )


async def test_healthy_groq_answer_records_groq():
    state = await _orchestrator().run("q", use_cache=False)
    assert state.provider is GenerationProvider.GROQ
    assert state.degraded is False


async def test_groq_rate_limit_falls_through_to_an_attributed_gemini_answer():
    state = await _orchestrator(groq_fails=True).run("q", use_cache=False)
    assert (state.provider, state.model) == (GenerationProvider.GEMINI, "gemini-3.5-flash-lite")
    assert state.degraded is False
    assert state.answer == "Gemini answer."


async def test_chitchat_has_no_provider():
    state = await _orchestrator(route="chitchat").run("hello", use_cache=False)
    assert state.provider is None


async def test_cache_hit_has_no_provider():
    orch = _orchestrator()
    orch.semantic_cache.get_exact = MagicMock(  # type: ignore[method-assign]
        return_value={"answer": "cached", "route": "hybrid", "sources": [], "graph_path": None}
    )
    state = await orch.run("q", use_cache=True)
    assert state.cache_hit is True
    assert state.provider is None


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    query = {
        "id": "q1",
        "query": "Q one",
        "slice": "unanswerable",
        "case": "direct",
        "expected_route": "hybrid",
        "reference_answer": "R",
        "gold_passages": [],
    }
    ds = tmp_path / "ds.json"
    ds.write_text(
        json.dumps({"version": "3.0.0", "domain": "t", "corpus": [], "queries": [query]}),
        encoding="utf-8",
    )
    base = tmp_path / "base.json"
    thresholds = dict.fromkeys(
        ("faithfulness", "context_precision", "answer_relevance", "answer_correctness"), 0.7
    )
    base.write_text(
        json.dumps(
            {
                "version": "3.0.0",
                "metrics": {"faithfulness": 0.85},
                "gate_thresholds": {
                    "max_regression_delta": 0.06,
                    "min_query_floor": 0.5,
                    "pass_thresholds": thresholds,
                },
            }
        ),
        encoding="utf-8",
    )
    return ds, base


@pytest.mark.usefixtures("small_gold_sets")
async def test_live_tier_fails_a_query_no_pool_member_answered_as_provider_degraded(files):
    evaluator = AsyncMock()
    evaluator.evaluate_query.return_value = EvalQueryResult(query="Q one", passed=True)
    runner = EvalRunner(
        _orchestrator(groq_fails=True, gemini_fails=True), evaluator, sleep=AsyncMock()
    )

    resp = await runner.run_evaluation(*files)

    res = resp.results[0]
    assert res.error is EvalQueryError.PROVIDER_DEGRADED
    assert (res.provider, res.model) == (GenerationProvider.REDIS_FAQ, None)
    assert res.passed is False
    evaluator.evaluate_query.assert_not_awaited()


@pytest.mark.usefixtures("small_gold_sets")
async def test_live_tier_judges_a_later_pool_members_answer_and_counts_its_model(files):
    evaluator = AsyncMock()
    evaluator.evaluate_query.return_value = EvalQueryResult(query="Q one", passed=True)
    runner = EvalRunner(_orchestrator(groq_fails=True), evaluator, sleep=AsyncMock())

    resp = await runner.run_evaluation(*files)

    res = resp.results[0]
    assert res.error is None
    assert (res.provider, res.model) == (GenerationProvider.GEMINI, "gemini-3.5-flash-lite")
    evaluator.evaluate_query.assert_awaited_once()
    assert resp.answers_by_model == {"gemini:gemini-3.5-flash-lite": 1}
    assert "gemini:gemini-3.5-flash-lite" in EvalRunner.format_markdown_report(resp)
