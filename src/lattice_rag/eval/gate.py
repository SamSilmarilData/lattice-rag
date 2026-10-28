"""Live Tier gate rules: the metrics, which Slices they apply to, and the baseline file.

The baseline file (``eval_baseline.json``) holds the measured mean of each metric and
every threshold the gate applies, so the gate's rules are visible in one place:

```
{ "version": "2.0.0", "updated_at": str, "total_queries": int,
  "metrics": {"faithfulness": float, ...},          # measured means; a metric absent here
                                                     # is reported but not regression-checked
  "gate_thresholds": {
     "max_regression_delta": float,                  # allowed drop of any mean vs "metrics"
     "min_query_floor": float,                       # no judged metric of a Known-Good Query
                                                     # below this (of any query, without
                                                     # "known_good"); others are Known-Weak
     "route_accuracy_floor": float,                  # optional; ADVISORY, reported, never gates
     "pass_thresholds": {"faithfulness": float, "context_precision": float,
                         "answer_relevance": float, "answer_correctness": float}},
  "measurement": {                                   # optional; written by measure-baseline
     "runs": [{metric: mean}, ...], "means": {...}, "stdev": {...},
     "tolerance": float, "tolerance_rule": str, "measured_at": str,
     "notes": str,                                   # notes: the runs' Model Pool mix
     "route_accuracy_runs": [float, ...],            # ADVISORY; runs with nothing scored
     "route_accuracy": float | null},                # are skipped; never sets the tolerance
  "known_good": [str, ...]}                          # optional; written by measure-baseline:
                                                     # Gold Set ids that cleared the floor in
                                                     # every run. Absent: any-query floor rule
```

``metrics`` and ``max_regression_delta`` come from ``measurement`` (``lattice-rag
measure-baseline`` over three ``lattice-rag eval --output`` reports); the floor and the
per-query pass thresholds are policy and are set by hand. ``known_good`` comes from the
same reports, so the gate fails on new failures rather than on every hard query.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
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
    # Advisory per ADR-0002: reported next to Route Accuracy, never fails the gate
    route_accuracy_floor: float | None = None


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
    route_accuracy_runs: list[float] = []  # advisory: recorded, never part of the tolerance
    route_accuracy: float | None = None  # mean of route_accuracy_runs


class Baseline(msgspec.Struct, frozen=True):
    """The measured baseline and the gate's thresholds."""

    version: str
    gate_thresholds: GateThresholds
    metrics: dict[str, float] = {}
    measurement: Measurement | None = None
    # Known-Good Queries: Gold Set ids that cleared the floor in every baseline run.
    # None (absent) keeps the legacy rule: any query below the floor fails.
    known_good: list[str] | None = None


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
    route_accuracy_runs = [r.route_accuracy for r in reports if r.route_accuracy is not None]
    return Measurement(
        runs=runs,
        means=means,
        stdev=stdev,
        tolerance=tolerance,
        measured_at=measured_at,
        notes=_model_mix_note(reports),
        route_accuracy_runs=route_accuracy_runs,
        route_accuracy=(
            round(statistics.fmean(route_accuracy_runs), 4) if route_accuracy_runs else None
        ),
    )


def known_good_queries(
    reports: Sequence[EvalRunResponse], min_query_floor: float
) -> list[str] | None:
    """Return the Known-Good Queries: ids that cleared the floor in every run.

    Clearing means the query ran in that run without error and every judged metric is
    at or above ``min_query_floor`` (a Noul veto zeroes faithfulness, so it does not
    clear). Returns ``None`` when any result has no ``query_id`` (a report written
    before ids were recorded), since those runs cannot name the queries.
    """
    results = [r for report in reports for r in report.results]
    if not results or any(r.query_id is None for r in results):
        return None
    cleared_per_run = [
        {r.query_id for r in report.results if _clears_floor(r, min_query_floor)}
        for report in reports
    ]
    cleared = set.intersection(*cleared_per_run)
    order = [r.query_id for r in reports[0].results]
    return [qid for qid in order if qid is not None and qid in cleared]


def _clears_floor(result: EvalQueryResult, min_query_floor: float) -> bool:
    scores = judged_scores(result)
    return (
        result.error is None
        and bool(scores)
        and all(score >= min_query_floor for score in scores.values())
    )


def _model_mix_note(reports: Sequence[EvalRunResponse]) -> str:
    """Which models answered across the runs, so a baseline says what it measured."""
    mix: Counter[str] = Counter()
    for report in reports:
        mix.update(report.answers_by_model)
    if not mix:
        return ""
    counts = ", ".join(f"{who} {n}" for who, n in mix.most_common())
    return f"Model mix over {len(reports)} runs: {counts}"


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


def write_measurement(
    path: str | Path,
    measurement: Measurement,
    total_queries: int,
    known_good: list[str] | None = None,
) -> None:
    """Record a measurement in the baseline file: means, tolerance, raw runs, Known-Good.

    ``known_good`` of ``None`` removes any earlier list, so the gate falls back to the
    any-query floor rather than trusting ids from another measurement. The query floor
    and per-query pass thresholds are policy and stay as they are.
    """
    file = Path(path)
    load_baseline(file)  # refuse to rewrite a baseline that is already invalid
    data = json.loads(file.read_text(encoding="utf-8"))
    data["updated_at"] = measurement.measured_at or data.get("updated_at", "")
    data["total_queries"] = total_queries
    data["metrics"] = measurement.means
    data["gate_thresholds"]["max_regression_delta"] = measurement.tolerance
    data["measurement"] = msgspec.to_builtins(measurement)
    if known_good is None:
        data.pop("known_good", None)
    else:
        data["known_good"] = known_good
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
