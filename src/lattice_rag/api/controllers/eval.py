from __future__ import annotations

import logging
from pathlib import Path

from litestar import Controller, post
from litestar.di import NamedDependency
from litestar.exceptions import ValidationException
from litestar.params import FromQuery, SkipValidation

from lattice_rag.api.dtos import EvalRunResponse
from lattice_rag.api.guards import require_admin_key
from lattice_rag.eval.runner import EvalRunner

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
    """Litestar controller for CI/CD evaluation and regression gating."""

    path = "/api/v1/eval"
    guards = [require_admin_key]

    @post("/run")
    async def run_eval(
        self,
        eval_runner: NamedDependency[SkipValidation[EvalRunner]],
        dataset: FromQuery[str] = "eval_dataset.json",
        baseline: FromQuery[str] = "eval_baseline.json",
        limit: FromQuery[int | None] = None,
        use_temp_db: FromQuery[bool] = True,
    ) -> EvalRunResponse:
        """Trigger an evaluation run against the golden benchmark dataset and compare with baseline."""
        dataset_path = resolve_eval_file(dataset)
        baseline_path = resolve_eval_file(baseline)
        logger.info(
            "received_eval_run_request",
            extra={
                "dataset": dataset,
                "baseline": baseline,
                "limit": limit,
                "use_temp_db": use_temp_db,
            },
        )
        return await eval_runner.run_evaluation(
            dataset_path=dataset_path,
            baseline_path=baseline_path,
            use_temp_db=use_temp_db,
            limit=limit,
        )
