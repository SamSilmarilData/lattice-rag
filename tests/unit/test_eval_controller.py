"""HTTP-seam tests for the eval job API (serialized runs behind a handle)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from litestar.testing import AsyncTestClient

from lattice_rag.api.controllers.eval import EvalController
from lattice_rag.api.dtos import EvalQueryResult, EvalRunResponse
from lattice_rag.api.guards import ADMIN_KEY_HEADER
from lattice_rag.config import AppConfig
from lattice_rag.eval.jobs import EvalJobManager
from lattice_rag.security import SecretStr

ADMIN_HEADERS = {ADMIN_KEY_HEADER: "admin-secret"}
JOBS = "/api/v1/eval/jobs"

REPORT = EvalRunResponse(
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


class BlockingRunner:
    """Fake EvalRunner whose runs finish only when the test releases them."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    async def run_evaluation(self, **kwargs: Any) -> EvalRunResponse:
        self.calls.append(kwargs)
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return REPORT


@pytest.fixture
def runner() -> BlockingRunner:
    return BlockingRunner()


@pytest.fixture
async def client(runner: BlockingRunner) -> AsyncIterator[AsyncTestClient[Litestar]]:
    jobs = EvalJobManager(runner)  # type: ignore[arg-type]
    app = Litestar(
        route_handlers=[EvalController],
        dependencies={"eval_jobs": Provide(lambda: jobs, sync_to_thread=False)},
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
    async with AsyncTestClient(app=app) as c:
        yield c
    await jobs.close()


async def _wait_for_status(client: AsyncTestClient[Litestar], job_id: str, status: str) -> Any:
    for _ in range(200):
        body = (await client.get(f"{JOBS}/{job_id}", headers=ADMIN_HEADERS)).json()
        if body["status"] == status:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {status!r}: {body}")


async def test_start_returns_handle_while_run_is_still_in_progress(client, runner):
    resp = await client.post(JOBS, headers=ADMIN_HEADERS)

    assert resp.status_code == 202
    body = resp.json()
    assert body["jobId"]
    assert body["status"] == "running"
    assert body["report"] is None
    assert not runner.release.is_set()


async def test_second_start_while_running_is_rejected_with_conflict(client, runner):
    first = (await client.post(JOBS, headers=ADMIN_HEADERS)).json()

    resp = await client.post(JOBS, headers=ADMIN_HEADERS)

    assert resp.status_code == 409
    assert first["jobId"] in resp.json()["detail"]
    assert len(runner.calls) == 1


async def test_report_is_fetchable_by_handle_once_the_run_finishes(client, runner):
    job_id = (await client.post(JOBS, params={"limit": 5}, headers=ADMIN_HEADERS)).json()["jobId"]

    running = (await client.get(f"{JOBS}/{job_id}", headers=ADMIN_HEADERS)).json()
    assert running["status"] == "running"
    assert running["finishedAt"] is None

    runner.release.set()
    done = await _wait_for_status(client, job_id, "succeeded")

    assert done["finishedAt"] is not None
    assert done["error"] is None
    assert done["report"]["totalQueries"] == 20
    assert done["report"]["passedGate"] is True
    assert done["report"]["results"][0]["query"] == "Sample query"
    assert runner.calls == [
        {
            "dataset_path": Path("eval_dataset.json").resolve(),
            "baseline_path": Path("eval_baseline.json").resolve(),
            "use_temp_db": True,
            "limit": 5,
        }
    ]


async def test_a_new_run_can_start_after_the_previous_one_finished(client, runner):
    first = (await client.post(JOBS, headers=ADMIN_HEADERS)).json()["jobId"]
    runner.release.set()
    await _wait_for_status(client, first, "succeeded")

    resp = await client.post(JOBS, headers=ADMIN_HEADERS)

    assert resp.status_code == 202
    assert resp.json()["jobId"] != first


async def test_failed_run_reports_failure_without_exception_text(client, runner):
    runner.error = RuntimeError("boom at /secret/path")
    job_id = (await client.post(JOBS, headers=ADMIN_HEADERS)).json()["jobId"]
    runner.release.set()

    failed = await _wait_for_status(client, job_id, "failed")

    assert failed["report"] is None
    assert job_id in failed["error"]
    assert "boom" not in failed["error"]
    assert (await client.post(JOBS, headers=ADMIN_HEADERS)).status_code == 202


async def test_unknown_handle_is_not_found(client):
    resp = await client.get(f"{JOBS}/does-not-exist", headers=ADMIN_HEADERS)
    assert resp.status_code == 404


async def test_eval_job_endpoints_require_admin_key(client, runner):
    assert (await client.post(JOBS)).status_code == 401
    assert (await client.post(JOBS, headers={ADMIN_KEY_HEADER: "wrong"})).status_code == 401
    job_id = (await client.post(JOBS, headers=ADMIN_HEADERS)).json()["jobId"]
    assert (await client.get(f"{JOBS}/{job_id}")).status_code == 401
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "dataset", ["/etc/hosts", "../eval_dataset.json", "sub/eval.json", ".env", "pyproject.toml"]
)
async def test_start_rejects_paths_outside_project_root(client, runner, dataset):
    resp = await client.post(JOBS, params={"dataset": dataset}, headers=ADMIN_HEADERS)
    assert resp.status_code == 400
    assert runner.calls == []
