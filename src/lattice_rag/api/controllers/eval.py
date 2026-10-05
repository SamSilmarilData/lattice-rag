from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from litestar import Controller, post
from litestar.di import NamedDependency
from litestar.params import FromQuery, SkipValidation

from lattice_rag.api.dtos import EvalRunResponse
from lattice_rag.eval.runner import EvalRunner

logger = logging.getLogger(__name__)


class EvalController(Controller):
    """Litestar controller for CI/CD evaluation and regression gating."""

    path = "/api/v1/eval"

    @post("/run")
    async def run_eval(
        self,
        eval_runner: NamedDependency[SkipValidation[EvalRunner]],
        dataset: FromQuery[str] = "eval_dataset.json",
        baseline: FromQuery[str] = "eval_baseline.json",
        limit: FromQuery[int | None] = None,
        use_temp_db: FromQuery[bool] = False,
    ) -> EvalRunResponse:
        """Trigger an evaluation run against the golden benchmark dataset and compare with baseline."""
        logger.info(
            "received_eval_run_request",
            extra={"dataset": dataset, "baseline": baseline, "limit": limit, "use_temp_db": use_temp_db},
        )
        return await eval_runner.run_evaluation(
            dataset_path=dataset,
            baseline_path=baseline,
            use_temp_db=use_temp_db,
            limit=limit,
        )
