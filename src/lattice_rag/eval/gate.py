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
                         "answer_relevance": float, "answer_correctness": float}},
  "measurement": {                                   # optional; written by measure-baseline
     "runs": [{metric: mean}, ...], "means": {...}, "stdev": {...},
     "tolerance": float, "tolerance_rule": str, "measured_at": str, "notes": str}}
```

``metrics`` and ``max_regression_delta`` come from ``measurement`` (``lattice-rag
measure-baseline`` over three ``lattice-rag eval --output`` reports); the floor and the
per-query pass thresholds are policy and are set by hand.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import msgspec

from lattice_rag.api.dtos import EvalQueryResult, EvalRunResponse

METRICS: Final = ("faithfulness", "context_precision", "answer_relevance", "answer_correctness")

# Metrics that compare against retrieved evidence mean nothing when the Gold Set says
# there is no evidence to retrieve. Chitchat never retrieves, so it is judged on the
# reply alone; an unanswerable query has no Gold Passages, so context precision has
# no target, but the answer must still be faithful (no invented facts) and correct
# (a refusal that matches the reference answer). A correct refusal does not address the
# question asked, so answer relevance would fail it; it is reported as not judged (n/a).
_SLICE_METRICS: Final[dict[str, tuple[str, ...]]] = {
    "chitchat": ("answer_relevance", "answer_correctness"),
    "unanswerable": ("faithfulness", "answer_correctness"),
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


TOLERANCE_SIGMAS: Final = 2.0
MIN_TOLERANCE: Final = 0.01
TOLERANCE_RULE: Final = (
    "max_regression_delta = 2 x the largest per-metric sample standard deviation of the "
    "run means, rounded up to 0.01 (minimum 0.01); metrics = mean of the run means"
)


class Measurement(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Repeated Live Tier runs that the baseline means and tolerance were derived from."""

    runs: list[dict[str, float]]  # per-run means, one dict per run, in run order
    means: dict[str, float]
    stdev: dict[str, float]  # sample standard deviation of the run means
    tolerance: float
    tolerance_rule: str = TOLERANCE_RULE
    measured_at: str = ""
    notes: str = ""


class Baseline(msgspec.Struct, frozen=True):
    """The measured baseline and the gate's thresholds."""

    version: str
    gate_thresholds: GateThresholds
    metrics: dict[str, float] = {}
    measurement: Measurement | None = None


def _run_means(report: EvalRunResponse) -> dict[str, float]:
    means = {m: getattr(report, f"mean_{m}") for m in METRICS}
    return {m: v for m, v in means.items() if v is not None}


def measure(reports: Sequence[EvalRunResponse], measured_at: str = "") -> Measurement:
    """Derive baseline means and the regression tolerance from repeated Live Tier runs.

    Only metrics judged in every run are measured. The tolerance is two sample standard
    deviations of the noisiest metric, so a rerun of unchanged code stays inside it.
    """
    if len(reports) < 2:
        raise BaselineError("Measuring judge variance needs at least two Live Tier runs")
    runs = [_run_means(r) for r in reports]
    common = [m for m in METRICS if all(m in run for run in runs)]
    means = {m: round(statistics.fmean(run[m] for run in runs), 4) for m in common}
    stdev = {m: round(statistics.stdev(run[m] for run in runs), 4) for m in common}
    widest = TOLERANCE_SIGMAS * max(stdev.values(), default=0.0)
    tolerance = max(MIN_TOLERANCE, math.ceil(round(widest * 100, 6)) / 100)
    return Measurement(
        runs=runs, means=means, stdev=stdev, tolerance=tolerance, measured_at=measured_at
    )


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


def write_measurement(path: str | Path, measurement: Measurement, total_queries: int) -> None:
    """Record a measurement in the baseline file: means, tolerance and the raw runs.

    The query floor and per-query pass thresholds are policy and stay as they are.
    """
    file = Path(path)
    load_baseline(file)  # refuse to rewrite a baseline that is already invalid
    data = json.loads(file.read_text(encoding="utf-8"))
    data["updated_at"] = measurement.measured_at or data.get("updated_at", "")
    data["total_queries"] = total_queries
    data["metrics"] = measurement.means
    data["gate_thresholds"]["max_regression_delta"] = measurement.tolerance
    data["measurement"] = msgspec.to_builtins(measurement)
    file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


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
