"""Live Tier gate rules: the metrics, which Slices they apply to, and the baseline file.

The baseline file (``eval_baseline.json``) holds the measured mean of each metric and
every threshold the gate applies, so the gate's rules are visible in one place:

```
{ "version": "2.0.0", "updated_at": str, "total_queries": int,
  "metrics": {"faithfulness": float, ...},          # measured means; a metric absent here
                                                     # is reported but not regression-checked
  "gate_thresholds": {
     "max_regression_delta": float,                  # allowed drop of any mean vs "metrics"
     "min_query_floor": float,                       # no judged metric of any query below this
     "pass_thresholds": {"faithfulness": float, "context_precision": float,
                         "answer_relevance": float, "answer_correctness": float}}}
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import msgspec

from lattice_rag.api.dtos import EvalQueryResult

METRICS: Final = ("faithfulness", "context_precision", "answer_relevance", "answer_correctness")

# Metrics that compare against retrieved evidence mean nothing when the Gold Set says
# there is no evidence to retrieve. Chitchat never retrieves, so it is judged on the
# reply alone; an unanswerable query has no Gold Passages, so context precision has
# no target, but the answer must still be faithful (no invented facts) and correct
# (a refusal that matches the reference answer).
_SLICE_METRICS: Final[dict[str, tuple[str, ...]]] = {
    "chitchat": ("answer_relevance", "answer_correctness"),
    "unanswerable": ("faithfulness", "answer_relevance", "answer_correctness"),
}


def metrics_for_slice(slice_name: str) -> tuple[str, ...]:
    """Return the metrics the Live Tier judges for a Gold Set Slice."""
    return _SLICE_METRICS.get(slice_name, METRICS)


class BaselineError(ValueError):
    """The baseline file is missing a threshold or is malformed."""


class PassThresholds(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Minimum score per metric for a single query to pass."""

    faithfulness: float
    context_precision: float
    answer_relevance: float
    answer_correctness: float


class GateThresholds(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Run-level gate rules."""

    max_regression_delta: float
    min_query_floor: float
    pass_thresholds: PassThresholds


class Baseline(msgspec.Struct, frozen=True):
    """The measured baseline and the gate's thresholds."""

    version: str
    gate_thresholds: GateThresholds
    metrics: dict[str, float] = {}


def load_baseline(path: str | Path) -> Baseline:
    """Load and validate the baseline file; every threshold is required."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Evaluation baseline not found at: {file}")
    try:
        baseline = msgspec.json.decode(file.read_bytes(), type=Baseline)
    except msgspec.ValidationError as e:
        raise BaselineError(f"Invalid baseline {file}: {e}") from e
    unknown = sorted(set(baseline.metrics) - set(METRICS))
    if unknown:
        raise BaselineError(f"Invalid baseline {file}: unknown metrics {unknown}")
    return baseline


def judged_scores(result: EvalQueryResult) -> dict[str, float]:
    """Return the metrics that were actually judged for a query, by name."""
    scores = {m: getattr(result, m) for m in METRICS}
    return {m: v for m, v in scores.items() if v is not None}


def format_metric(value: float | None, digits: int = 4) -> str:
    """Format a metric for reports, showing ``n/a`` when it was not judged."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def query_passes(result: EvalQueryResult, thresholds: PassThresholds) -> bool:
    """A query passes when it was judged and every judged metric meets its threshold."""
    if result.error is not None:
        return False
    scores = judged_scores(result)
    if not scores:
        return False
    return all(score >= getattr(thresholds, m) for m, score in scores.items())


DEFAULT_PASS_THRESHOLDS: Final = PassThresholds(
    faithfulness=0.70, context_precision=0.65, answer_relevance=0.70, answer_correctness=0.70
)
