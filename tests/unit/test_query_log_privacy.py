"""User query text never appears in log records above DEBUG.

Queries go through the real HTTP controllers, ``RAGOrchestrator``, ``QueryRouter``,
``RetrievalPipeline``, ``EvalRunner`` and ``JevEvaluator``; only the external
boundaries (Jev client, LatticeDB store, embeddings, LLM providers) are faked.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from litestar.testing import AsyncTestClient

from lattice_rag.api.controllers import QueryController
from lattice_rag.api.controllers.eval import EvalController
from lattice_rag.api.guards import ADMIN_KEY_HEADER
from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.config import AppConfig
from lattice_rag.eval.runner import EvalRunner
from lattice_rag.eval.triage_gate import JevEvaluator
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.retrieval.pipeline import RetrievalPipeline
from lattice_rag.routing.router import QueryRouter
from lattice_rag.security import SecretStr, configure_structlog

SECRET_QUERY = "Explain the zebra-quantum-7731 indexing scheme in detail"
ADMIN_HEADERS = {ADMIN_KEY_HEADER: "admin-secret"}


def _route_response(route: str) -> Any:
    return SimpleNamespace(choices={"route": SimpleNamespace(choice=route, confidence=0.9)})


def _eval_response(contradiction: float) -> Any:
    return SimpleNamespace(
        scores={
            "faithfulness": SimpleNamespace(score=3.6),
            "context_precision": SimpleNamespace(score=3.4),
            "answer_relevance": SimpleNamespace(score=3.8),
        },
        nouls={"contradiction": SimpleNamespace(noul=contradiction)},
    )


def _build_app(
    *,
    router_client: Any,
    eval_client: Any | None = None,
    store_error: Exception | None = None,
) -> Litestar:
    store = MagicMock()
    if store_error is not None:
        store.bm25_search.side_effect = store_error
    else:
        store.bm25_search.return_value = []
    store.vector_search.return_value = []

    embed_svc = MagicMock()
    embed_svc.embed_query.return_value = np.ones(384, dtype=np.float32)

    fallback_cache = MagicMock()
    fallback_cache.find_closest_fallback = AsyncMock(return_value=None)

    groq = MagicMock()
    groq.synthesize = AsyncMock(return_value="An answer.")

    async def _stream(*_args: Any) -> Any:
        yield "An answer."

    groq.stream = _stream

    orchestrator = RAGOrchestrator(
        router=QueryRouter(ts_client=router_client),
        semantic_cache=SemanticCache(),
        fallback_cache=fallback_cache,
        retrieval_pipeline=RetrievalPipeline(store, embed_svc),
        groq=groq,
        gemini=MagicMock(),
        chitchat=MagicMock(),
        embedding_service=embed_svc,
    )
    runner = EvalRunner(
        orchestrator=orchestrator,
        evaluator=JevEvaluator(client=eval_client or AsyncMock()),
    )
    return Litestar(
        route_handlers=[QueryController, EvalController],
        # Keep Litestar from replacing the root handlers that caplog installs
        logging_config=None,
        dependencies={
            "orchestrator": Provide(lambda: orchestrator, sync_to_thread=False),
            "eval_runner": Provide(lambda: runner, sync_to_thread=False),
        },
        state=State(
            {
                "config": AppConfig(
                    typesafe_api_key=SecretStr("ts"),
                    groq_api_key=SecretStr(""),
                    gemini_api_key=SecretStr(""),
                    admin_api_key=SecretStr("admin-secret"),
                )
            }
        ),
    )


@pytest.fixture
def capture(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    configure_structlog()
    caplog.set_level(logging.DEBUG)
    return caplog


def _assert_query_absent_above_debug(caplog: pytest.LogCaptureFixture) -> None:
    leaks = [
        rec
        for rec in caplog.records
        if rec.levelno > logging.DEBUG
        and (SECRET_QUERY in rec.getMessage() or SECRET_QUERY in repr(vars(rec)))
    ]
    assert not leaks, [f"{r.name}:{r.levelname}:{r.getMessage()}" for r in leaks]


def _write_dataset(tmp_path: Path) -> None:
    (tmp_path / "ds.json").write_text(
        json.dumps({"queries": [{"query": SECRET_QUERY, "ground_truth": "gt"}]})
    )
    (tmp_path / "base.json").write_text(json.dumps({"metrics": {}}))


async def _post_eval(app: Litestar) -> Any:
    async with AsyncTestClient(app=app) as client:
        return await client.post(
            "/api/v1/eval/run",
            params={"dataset": "ds.json", "baseline": "base.json", "use_temp_db": "false"},
            headers=ADMIN_HEADERS,
        )


@pytest.mark.parametrize("endpoint", ["/api/v1/query", "/api/v1/query/stream"])
async def test_empty_retrieval_does_not_log_query(capture, endpoint):
    app = _build_app(
        router_client=AsyncMock(system_one=AsyncMock(return_value=_route_response("hybrid")))
    )
    async with AsyncTestClient(app=app) as client:
        resp = await client.post(endpoint, json={"query": SECRET_QUERY})
    assert resp.status_code == 201
    assert any(r.levelno > logging.DEBUG for r in capture.records)
    _assert_query_absent_above_debug(capture)


@pytest.mark.parametrize(
    "router_client",
    [
        AsyncMock(system_one=AsyncMock(side_effect=RuntimeError("jev down"))),
        AsyncMock(system_one=AsyncMock(return_value=SimpleNamespace(choices={}))),
    ],
    ids=["router-raises", "missing-choice"],
)
async def test_routing_failure_does_not_log_query(capture, router_client):
    app = _build_app(router_client=router_client)
    async with AsyncTestClient(app=app) as client:
        resp = await client.post("/api/v1/query", json={"query": SECRET_QUERY})
    assert resp.status_code == 201
    assert resp.json()["route"] == "hybrid"
    assert any(r.levelno >= logging.WARNING for r in capture.records)
    _assert_query_absent_above_debug(capture)


async def test_eval_execution_failure_does_not_log_query(capture, tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _build_app(
        router_client=AsyncMock(system_one=AsyncMock(return_value=_route_response("hybrid"))),
        eval_client=AsyncMock(system_one=AsyncMock(return_value=_eval_response(0.05))),
        store_error=RuntimeError("store unavailable"),
    )
    resp = await _post_eval(app)
    assert resp.status_code == 201
    assert any(r.levelno >= logging.ERROR for r in capture.records)
    _assert_query_absent_above_debug(capture)


async def test_eval_scoring_failure_does_not_log_query(capture, tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _build_app(
        router_client=AsyncMock(system_one=AsyncMock(return_value=_route_response("hybrid"))),
        eval_client=AsyncMock(system_one=AsyncMock(side_effect=RuntimeError("jev down"))),
    )
    resp = await _post_eval(app)
    assert resp.status_code == 201
    assert resp.json()["results"][0]["passed"] is False
    assert any(r.levelno >= logging.ERROR for r in capture.records)
    _assert_query_absent_above_debug(capture)


async def test_eval_contradiction_veto_does_not_log_query(capture, tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _build_app(
        router_client=AsyncMock(system_one=AsyncMock(return_value=_route_response("hybrid"))),
        eval_client=AsyncMock(system_one=AsyncMock(return_value=_eval_response(0.9))),
    )
    resp = await _post_eval(app)
    assert resp.status_code == 201
    assert any(r.levelno >= logging.WARNING for r in capture.records)
    _assert_query_absent_above_debug(capture)
