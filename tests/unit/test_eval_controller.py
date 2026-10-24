"""Unit tests for EvalController endpoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from litestar.testing import AsyncTestClient

from lattice_rag.api.controllers.eval import EvalController
from lattice_rag.api.dtos import EvalQueryResult, EvalRunResponse
from lattice_rag.api.guards import ADMIN_KEY_HEADER
from lattice_rag.config import AppConfig
from lattice_rag.security import SecretStr

ADMIN_HEADERS = {ADMIN_KEY_HEADER: "admin-secret"}


@pytest.fixture
def mock_eval_runner():
    runner = AsyncMock()
    runner.run_evaluation.return_value = EvalRunResponse(
        total_queries=20,
        mean_faithfulness=0.89,
        mean_context_precision=0.86,
        mean_answer_relevance=0.91,
        passed_gate=True,
        regression_delta=0.01,
        results=[
            EvalQueryResult(
                query="Sample query",
                faithfulness=0.90,
                context_precision=0.88,
                answer_relevance=0.92,
                passed=True,
            )
        ],
    )
    return runner


@pytest.fixture
def test_app(mock_eval_runner):
    return Litestar(
        route_handlers=[EvalController],
        dependencies={
            "eval_runner": Provide(lambda: mock_eval_runner, sync_to_thread=False),
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


@pytest.mark.asyncio
async def test_api_eval_run_endpoint(test_app, mock_eval_runner):
    """Verify POST /api/v1/eval/run calls eval runner and returns EvalRunResponse."""
    async with AsyncTestClient(app=test_app) as client:
        resp = await client.post("/api/v1/eval/run", headers=ADMIN_HEADERS)
        assert resp.status_code == 201

        data = resp.json()
        assert data["totalQueries"] == 20
        assert data["meanFaithfulness"] == 0.89
        assert data["passedGate"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["query"] == "Sample query"

        mock_eval_runner.run_evaluation.assert_awaited_once_with(
            dataset_path=Path("eval_dataset.json").resolve(),
            baseline_path=Path("eval_baseline.json").resolve(),
            use_temp_db=True,
            limit=None,
        )


@pytest.mark.asyncio
async def test_api_eval_run_endpoint_with_limit(test_app, mock_eval_runner):
    """Verify POST /api/v1/eval/run?limit=5 passes limit parameter to eval runner."""
    mock_eval_runner.run_evaluation.reset_mock()
    async with AsyncTestClient(app=test_app) as client:
        resp = await client.post("/api/v1/eval/run?limit=5", headers=ADMIN_HEADERS)
        assert resp.status_code == 201

        mock_eval_runner.run_evaluation.assert_awaited_once_with(
            dataset_path=Path("eval_dataset.json").resolve(),
            baseline_path=Path("eval_baseline.json").resolve(),
            use_temp_db=True,
            limit=5,
        )


@pytest.mark.asyncio
async def test_api_eval_requires_admin_key(test_app, mock_eval_runner):
    mock_eval_runner.run_evaluation.reset_mock()
    async with AsyncTestClient(app=test_app) as client:
        resp = await client.post("/api/v1/eval/run")
        assert resp.status_code == 401
        mock_eval_runner.run_evaluation.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dataset", ["/etc/hosts", "../eval_dataset.json", "sub/eval.json", ".env", "pyproject.toml"]
)
async def test_api_eval_rejects_paths_outside_project_root(test_app, mock_eval_runner, dataset):
    mock_eval_runner.run_evaluation.reset_mock()
    async with AsyncTestClient(app=test_app) as client:
        resp = await client.post(
            "/api/v1/eval/run", params={"dataset": dataset}, headers=ADMIN_HEADERS
        )
        assert resp.status_code == 400
        mock_eval_runner.run_evaluation.assert_not_called()
