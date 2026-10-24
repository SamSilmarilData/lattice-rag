from __future__ import annotations

import logging
from pathlib import Path

from litestar import Controller, get, post
from litestar.di import NamedDependency
from litestar.exceptions import ClientException, NotFoundException, ValidationException
from litestar.params import FromPath, FromQuery, SkipValidation
from litestar.status_codes import HTTP_202_ACCEPTED, HTTP_409_CONFLICT

from lattice_rag.api.dtos import EvalJobResponse
from lattice_rag.api.guards import require_admin_key
from lattice_rag.eval.jobs import EvalJobConflictError, EvalJobManager

logger = logging.getLogger(__name__)


def resolve_eval_file(name: str, root: Path | None = None) -> Path:
    """Resolve a dataset/baseline name to a JSON file directly inside the project root.

    Only bare ``*.json`` file names are accepted, so API callers cannot point the
    eval runner at arbitrary paths on the server.
    """
    base = (root or Path.cwd()).resolve()
    candidate = (base / name).resolve()
    if Path(name).name != name or candidate.suffix != ".json" or candidate.parent != base:
        raise ValidationException(detail=f"Invalid eval file name: {name!r}")
    return candidate


class EvalController(Controller):
    """Eval Jobs: start a serialized evaluation run, then poll it by handle."""

    path = "/api/v1/eval"
    guards = [require_admin_key]

    @post("/jobs", status_code=HTTP_202_ACCEPTED)
    async def start_eval_job(
        self,
        eval_jobs: NamedDependency[SkipValidation[EvalJobManager]],
        dataset: FromQuery[str] = "eval_dataset.json",
        baseline: FromQuery[str] = "eval_baseline.json",
        limit: FromQuery[int | None] = None,
        use_temp_db: FromQuery[bool] = True,
    ) -> EvalJobResponse:
        """Start an evaluation run and return its handle without waiting for it.

        Only one run executes at a time; starting another while one is running
        returns 409 Conflict (runs are not queued).
        """
        dataset_path = resolve_eval_file(dataset)
        baseline_path = resolve_eval_file(baseline)
        logger.info(
            "received_eval_job_request",
            extra={
                "dataset": dataset,
                "baseline": baseline,
                "limit": limit,
                "use_temp_db": use_temp_db,
            },
        )
        try:
            return eval_jobs.start(
                dataset_path=dataset_path,
                baseline_path=baseline_path,
                use_temp_db=use_temp_db,
                limit=limit,
            )
        except EvalJobConflictError as exc:
            raise ClientException(status_code=HTTP_409_CONFLICT, detail=str(exc)) from exc

    @get("/jobs/{job_id:str}")
    async def get_eval_job(
        self,
        job_id: FromPath[str],
        eval_jobs: NamedDependency[SkipValidation[EvalJobManager]],
    ) -> EvalJobResponse:
        """Fetch an Eval Job's status and, once it has succeeded, its report."""
        job = eval_jobs.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"No eval job {job_id!r}")
        return job
