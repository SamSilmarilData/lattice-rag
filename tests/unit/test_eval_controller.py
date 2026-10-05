"""Unit tests for EvalController endpoint."""
from __future__ import annotations

from unittest.mock import AsyncMock
import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.testing import AsyncTestClient

from lattice_rag.api.controllers.eval import EvalController
from lattice_rag.api.dtos import EvalQueryResult, EvalRunResponse


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
    )


@pytest.mark.asyncio
async def test_api_eval_run_endpoint(test_app, mock_eval_runner):
    """Verify POST /api/v1/eval/run calls eval runner and returns EvalRunResponse."""
    async with AsyncTestClient(app=test_app) as client:
        resp = await client.post("/api/v1/eval/run")
        assert resp.status_code == 201

        data = resp.json()
        assert data["totalQueries"] == 20
        assert data["meanFaithfulness"] == 0.89
        assert data["passedGate"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["query"] == "Sample query"

        mock_eval_runner.run_evaluation.assert_awaited_once_with(
            dataset_path="eval_dataset.json",
            baseline_path="eval_baseline.json",
        )
