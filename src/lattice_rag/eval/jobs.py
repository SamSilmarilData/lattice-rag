"""Serialized, handle-based eval runs for the HTTP API.

An Eval Job wraps one ``EvalRunner.run_evaluation`` call in a background task. At most
one job runs at a time: starting another while one is running raises
``EvalJobConflictError`` (HTTP 409) instead of queueing, so parallel requests cannot
multiply judge and LLM API cost. Finished jobs are kept in memory (bounded) so their
status and report can be fetched by id; they do not survive a restart.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

from lattice_rag.api.dtos import EvalJobResponse, EvalJobStatus
from lattice_rag.eval.runner import EvalRunner
from lattice_rag.security import sanitize_error_detail

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_LIMIT = 20


class EvalJobConflictError(Exception):
    """Raised when an Eval Job is started while another one is still running."""

    def __init__(self, running_job_id: str) -> None:
        super().__init__(f"An evaluation is already running (job {running_job_id})")
        self.running_job_id = running_job_id


class EvalJobManager:
    """Runs at most one Eval Job at a time and remembers recent jobs by id."""

    def __init__(self, runner: EvalRunner, history_limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        self._runner = runner
        self._history_limit = history_limit
        self._jobs: OrderedDict[str, EvalJobResponse] = OrderedDict()
        self._running: tuple[str, asyncio.Task[None]] | None = None

    def start(
        self,
        *,
        dataset_path: Path,
        baseline_path: Path,
        use_temp_db: bool,
        limit: int | None,
    ) -> EvalJobResponse:
        """Start a job and return its handle immediately.

        The check-and-claim below has no ``await`` in it, so two concurrent requests on
        the event loop cannot both start a run.
        """
        if self._running is not None:
            raise EvalJobConflictError(self._running[0])

        job_id = uuid.uuid4().hex
        job = EvalJobResponse(
            job_id=job_id, status=EvalJobStatus.RUNNING, started_at=datetime.now(UTC)
        )
        self._remember(job)
        task = asyncio.create_task(
            self._execute(
                job_id,
                dataset_path=dataset_path,
                baseline_path=baseline_path,
                use_temp_db=use_temp_db,
                limit=limit,
            ),
            name=f"eval-job-{job_id}",
        )
        self._running = (job_id, task)
        return job

    def get(self, job_id: str) -> EvalJobResponse | None:
        return self._jobs.get(job_id)

    async def close(self) -> None:
        """Cancel a running job (application shutdown)."""
        if self._running is None:
            return
        _, task = self._running
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _execute(
        self,
        job_id: str,
        *,
        dataset_path: Path,
        baseline_path: Path,
        use_temp_db: bool,
        limit: int | None,
    ) -> None:
        job = self._jobs[job_id]
        try:
            job.report = await self._runner.run_evaluation(
                dataset_path=dataset_path,
                baseline_path=baseline_path,
                use_temp_db=use_temp_db,
                limit=limit,
            )
            job.status = EvalJobStatus.SUCCEEDED
        except asyncio.CancelledError:
            job.status = EvalJobStatus.FAILED
            job.error = "Evaluation was cancelled."
            raise
        except Exception as exc:
            # Exception text can carry paths or upstream payloads: log it, never return it
            logger.error(
                "eval_job_failed job_id=%s error=%s",
                job_id,
                sanitize_error_detail(f"{type(exc).__name__}: {exc}"),
            )
            job.status = EvalJobStatus.FAILED
            job.error = f"Evaluation failed. Reference: {job_id}"
        finally:
            job.finished_at = datetime.now(UTC)
            self._running = None

    def _remember(self, job: EvalJobResponse) -> None:
        self._jobs[job.job_id] = job
        while len(self._jobs) > self._history_limit:
            self._jobs.popitem(last=False)
