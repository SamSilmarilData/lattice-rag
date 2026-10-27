from __future__ import annotations

import asyncio
import contextlib
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import msgspec
import structlog

from lattice_rag.api.dtos import EvalQueryError, EvalQueryResult, EvalRunResponse, Misroute
from lattice_rag.eval.gate import (
    METRICS,
    Baseline,
    judged_scores,
    load_baseline,
    metrics_for_slice,
)
from lattice_rag.eval.gate import format_metric as _fmt
from lattice_rag.eval.gold_set import GoldQuery, load_gold_set
from lattice_rag.eval.triage_gate import JevEvaluator
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.security import query_fingerprint, sanitize_error_detail
from lattice_rag.storage.db import LatticeStore, normalize_entity_name
from lattice_rag.storage.extract import EntityExtractor
from lattice_rag.vocabulary import Route

logger = structlog.get_logger(__name__)

# Fixed failure codes on EvalQueryResult.error; exception text stays in the logs
PIPELINE_ERROR = EvalQueryError.PIPELINE_ERROR
JUDGE_ERROR = EvalQueryError.JUDGE_ERROR


class EvalRunner:
    """Orchestrator for Golden Dataset execution, Jev evaluation, and regression gating."""

    def __init__(
        self,
        orchestrator: RAGOrchestrator,
        evaluator: JevEvaluator,
        embedding_service: EmbeddingService | None = None,
        extractor: EntityExtractor | None = None,
        degraded_retry_delay: float = 5.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize EvalRunner with orchestrator, Jev evaluator, and optional support services."""
        self.orchestrator = orchestrator
        self.evaluator = evaluator
        self.embedding_service = embedding_service
        self.extractor = extractor
        self.degraded_retry_delay = degraded_retry_delay
        self._sleep = sleep

    async def run_evaluation(
        self,
        dataset_path: str | Path = "eval_dataset.json",
        baseline_path: str | Path = "eval_baseline.json",
        use_temp_db: bool = False,
        max_concurrency: int = 2,
        limit: int | None = None,
    ) -> EvalRunResponse:
        """Execute full evaluation run against golden benchmark dataset and compare with baseline."""
        ds_file = Path(dataset_path)
        if not ds_file.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at: {ds_file}")

        gold_set = load_gold_set(ds_file)
        baseline = load_baseline(baseline_path)
        pass_thresholds = baseline.gate_thresholds.pass_thresholds

        queries_data: list[GoldQuery] = list(gold_set.queries)
        corpus_data = gold_set.corpus

        if not queries_data:
            raise ValueError(f"No queries found in dataset: {ds_file}")

        if limit is not None and limit > 0:
            queries_data = queries_data[:limit]

        active_orchestrator = self.orchestrator
        temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
        temp_store: LatticeStore | None = None

        try:
            if use_temp_db and corpus_data:
                logger.info("seeding_ephemeral_eval_database", doc_count=len(corpus_data))
                temp_dir_obj = tempfile.TemporaryDirectory()
                temp_db_path = Path(temp_dir_obj.name) / "eval_ephemeral.db"
                temp_store = LatticeStore(temp_db_path)

                embed_svc = self.embedding_service or self.orchestrator.retrieval.embedding_service
                extractor = self.extractor or EntityExtractor()

                from lattice_rag.ingestion.ingester import DocumentIngester

                ingester = DocumentIngester(
                    store=temp_store,
                    embedding_service=embed_svc,
                    extractor=extractor,
                )

                # Seed documents using batched DocumentIngester
                for doc in corpus_data:
                    await ingester.ingest(doc_id=doc.document_id, title=doc.title, text=doc.text)

                # Same pipeline and orchestrator settings (guardrails, forced hops, graph
                # context in the prompt, answer budget) over the ephemeral store
                active_orchestrator = self.orchestrator.with_retrieval(
                    self.orchestrator.retrieval.with_store(temp_store)
                )

            # For suites with >5 queries, enforce concurrency 1 and inter-query pacing to avoid Groq 1000 OTPM rate limits
            effective_concurrency = 1 if len(queries_data) > 5 else max_concurrency
            semaphore = asyncio.Semaphore(effective_concurrency)

            async def _eval_single(item: GoldQuery, idx: int) -> EvalQueryResult:
                async with semaphore:
                    if idx > 0 and len(queries_data) > 5:
                        await asyncio.sleep(1.8)

                    query = item.query

                    try:
                        # Bypass Tier 1 so every query is freshly evaluated and eval
                        # answers (possibly from the ephemeral corpus) never reach users
                        pipeline_state = await active_orchestrator.run(query, use_cache=False)
                        if pipeline_state.degraded:
                            # One retry after a pause; a still non-primary answer fails the query.
                            # The pause keeps the semaphore slot on purpose: releasing it would
                            # let the next query call Groq inside the same rate-limit window
                            # the pause is waiting out.
                            await self._sleep(self.degraded_retry_delay)
                            pipeline_state = await active_orchestrator.run(query, use_cache=False)
                    except Exception as e:
                        # A pipeline error fails the query; it is never judged as an answer
                        logger.error(
                            "eval_query_execution_failed",
                            **query_fingerprint(query),
                            error=sanitize_error_detail(str(e)),
                        )
                        return EvalQueryResult(query=query, passed=False, error=PIPELINE_ERROR)

                    chosen_route = cast("Route", pipeline_state.route)
                    route_verified = pipeline_state.route_verified
                    provider = pipeline_state.provider
                    graph_recall = graph_recall_of(
                        item.expected_entities, pipeline_state.graph_context
                    )
                    if pipeline_state.degraded:
                        logger.error(
                            "eval_query_provider_degraded",
                            **query_fingerprint(query),
                            provider=provider,
                        )
                        return EvalQueryResult(
                            query=query,
                            passed=False,
                            error=EvalQueryError.PROVIDER_DEGRADED,
                            chosen_route=chosen_route,
                            route_verified=route_verified,
                            provider=provider,
                            graph_recall=graph_recall,
                        )

                    try:
                        judged = await self.evaluator.evaluate_query(
                            query=query,
                            retrieved_chunks=pipeline_state.filtered_chunks,
                            generated_answer=pipeline_state.answer,
                            ground_truth=item.reference_answer,
                            metrics=metrics_for_slice(item.slice),
                            thresholds=pass_thresholds,
                            # Judge the graph the generator received, not the whole traversal
                            graph_context=pipeline_state.prompt_graph_context,
                        )
                        return msgspec.structs.replace(
                            judged,
                            chosen_route=chosen_route,
                            route_verified=route_verified,
                            provider=provider,
                            graph_recall=graph_recall,
                        )
                    except Exception as e:
                        # Fail the query rather than aborting the run
                        logger.error(
                            "eval_query_scoring_failed",
                            **query_fingerprint(query),
                            error=sanitize_error_detail(str(e)),
                        )
                        return EvalQueryResult(
                            query=query,
                            passed=False,
                            error=JUDGE_ERROR,
                            chosen_route=chosen_route,
                            route_verified=route_verified,
                            provider=provider,
                            graph_recall=graph_recall,
                        )

            logger.info("running_eval_suite", total_queries=len(queries_data))
            eval_tasks = [_eval_single(q, idx) for idx, q in enumerate(queries_data)]
            results = await asyncio.gather(*eval_tasks)
        finally:
            # Clean up ephemeral resources
            if temp_store is not None:
                with contextlib.suppress(Exception):
                    temp_store.close()

            if temp_dir_obj is not None:
                with contextlib.suppress(Exception):
                    temp_dir_obj.cleanup()

        return _apply_gate(queries_data, list(results), baseline)

    @staticmethod
    def format_markdown_report(report: EvalRunResponse) -> str:
        """Format an evaluation run response into a clean Markdown audit summary."""
        status_badge = "✅ PASSED" if report.passed_gate else "❌ REGRESSION DETECTED"
        lines = [
            f"# Evaluation Gate Audit Report: {status_badge}",
            "",
            f"- **Total Queries**: {report.total_queries}",
            f"- **Failed Queries** (not judged): {report.failed_queries}",
            f"- **Mean Faithfulness**: {_fmt(report.mean_faithfulness)}",
            f"- **Mean Context Precision**: {_fmt(report.mean_context_precision)}",
            f"- **Mean Answer Relevance**: {_fmt(report.mean_answer_relevance)}",
            f"- **Mean Answer Correctness**: {_fmt(report.mean_answer_correctness)}",
            *_route_accuracy_lines(report),
            *_graph_recall_lines(report),
            f"- **Regression Delta**: {report.regression_delta:+.4f}",
            f"- **Gate Status**: {'PASSED' if report.passed_gate else 'FAILED'}",
            "",
            "## Query Breakdown",
            "",
            "| Query | Provider | Faithfulness | Precision | Relevance | Correctness | Passed |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for res in report.results:
            q_short = (res.query[:50] + "...") if len(res.query) > 50 else res.query
            status = "✅" if res.passed else f"❌ {res.error or ''}".rstrip()
            lines.append(
                f"| {q_short} | {res.provider or '-'} "
                f"| {_fmt(res.faithfulness, 3)} | {_fmt(res.context_precision, 3)} "
                f"| {_fmt(res.answer_relevance, 3)} | {_fmt(res.answer_correctness, 3)} | {status} |"
            )
        return "\n".join(lines)

    @staticmethod
    def export_report_json(
        report: EvalRunResponse, output_path: str | Path = "eval_run_report.json"
    ) -> Path:
        """Export evaluation report to JSON file."""
        import msgspec

        out = Path(output_path)
        payload = msgspec.json.encode(report)
        out.write_bytes(payload)
        logger.info("eval_report_exported", path=str(out))
        return out


def _apply_gate(
    queries: list[GoldQuery], results: list[EvalQueryResult], baseline: Baseline
) -> EvalRunResponse:
    """Apply the baseline file's gate rules to per-query results.

    The gate fails when any query failed (pipeline or judge error), when any judged
    metric of any query is below ``min_query_floor``, or when a mean drops more than
    ``max_regression_delta`` below its baseline. A metric without a baseline mean is
    reported but not regression-checked. Route Accuracy is reported alongside and never
    gates (ADR-0002). ``results[i]`` is the result of ``queries[i]``.
    """
    rules = baseline.gate_thresholds
    judged = [r for r in results if r.error is None]
    failed = len(results) - len(judged)

    means: dict[str, float | None] = {}
    for metric in METRICS:
        values = [v for r in judged if (v := judged_scores(r).get(metric)) is not None]
        means[metric] = round(sum(values) / len(values), 4) if values else None

    deltas = {
        m: round(mean - baseline.metrics[m], 4)
        for m, mean in means.items()
        if mean is not None and m in baseline.metrics
    }
    regression_passed = all(d >= -rules.max_regression_delta for d in deltas.values())
    floor_passed = all(
        score >= rules.min_query_floor for r in judged for score in judged_scores(r).values()
    )
    passed_gate = failed == 0 and regression_passed and floor_passed
    min_delta = min(deltas.values(), default=0.0)

    logger.info(
        "eval_suite_completed",
        total_queries=len(results),
        failed_queries=failed,
        **{f"mean_{m}": v for m, v in means.items()},
        **{f"delta_{m}": d for m, d in deltas.items()},
        passed_gate=passed_gate,
        floor_passed=floor_passed,
    )

    return EvalRunResponse(
        total_queries=len(results),
        passed_gate=passed_gate,
        regression_delta=min_delta,
        mean_faithfulness=means["faithfulness"],
        mean_context_precision=means["context_precision"],
        mean_answer_relevance=means["answer_relevance"],
        mean_answer_correctness=means["answer_correctness"],
        failed_queries=failed,
        results=results,
        route_accuracy_floor=rules.route_accuracy_floor,
        **_route_accuracy(queries, results),
        **_graph_recall(queries, results),
    )


def _route_accuracy(queries: list[GoldQuery], results: list[EvalQueryResult]) -> dict[str, Any]:
    """Route Accuracy fields over the scored queries.

    A query is scored when it is outside the chitchat Slice (routed deterministically, never
    by Jev), its pipeline ran, and Jev chose the Route. Pipeline errors and unverified
    fallback routes are counted instead, so an outage cannot read as accurate routing.
    """
    routed = [(q, r) for q, r in zip(queries, results, strict=True) if q.slice != "chitchat"]
    errored = sum(r.chosen_route is None for _, r in routed)
    ran = [(q, r) for q, r in routed if r.chosen_route is not None]
    scored = [(q, r) for q, r in ran if r.route_verified]

    by_slice: dict[str, list[bool]] = {}
    for q, r in scored:
        by_slice.setdefault(q.slice, []).append(r.chosen_route == q.expected_route)
    hits = sum(ok for oks in by_slice.values() for ok in oks)
    return {
        "route_accuracy": round(hits / len(scored), 4) if scored else None,
        "route_accuracy_by_slice": {s: round(sum(o) / len(o), 4) for s, o in by_slice.items()},
        "route_accuracy_scored": len(scored),
        "route_accuracy_unverified": len(ran) - len(scored),
        "route_accuracy_errored": errored,
        "misroutes": [
            Misroute(
                query_id=q.id,
                slice=q.slice,
                chosen_route=r.chosen_route,
                expected_route=q.expected_route,
            )
            for q, r in scored
            if r.chosen_route is not None and r.chosen_route != q.expected_route
        ],
    }


def graph_recall_of(expected: list[str], graph_context: object) -> float | None:
    """Share of ``expected`` entity names among the graph context's Entity names.

    ``None`` when the query lists no expected entities. A missing or malformed graph
    context surfaced no Entities, so its recall is 0.0.
    """
    wanted = {normalize_entity_name(e) for e in expected}
    if not wanted:
        return None
    nodes = graph_context.get("nodes") if isinstance(graph_context, dict) else None
    surfaced = {
        normalize_entity_name(str(n["name"]))
        for n in nodes or []
        if isinstance(n, dict) and n.get("name")
    }
    return round(len(wanted & surfaced) / len(wanted), 4)


def _graph_recall(queries: list[GoldQuery], results: list[EvalQueryResult]) -> dict[str, Any]:
    """Graph Recall fields: per-query means, overall and per Slice. Advisory, never gates."""
    by_slice: dict[str, list[float]] = {}
    for q, r in zip(queries, results, strict=True):
        if r.graph_recall is not None:
            by_slice.setdefault(q.slice, []).append(r.graph_recall)
    scored = [v for vs in by_slice.values() for v in vs]
    return {
        "graph_recall": round(sum(scored) / len(scored), 4) if scored else None,
        "graph_recall_by_slice": {s: round(sum(v) / len(v), 4) for s, v in by_slice.items()},
        "graph_recall_scored": len(scored),
    }


def _plural(n: int, noun: str, plural: str | None = None) -> str:
    return f"{n} {noun if n == 1 else plural or noun + 's'}"


def _graph_recall_lines(report: EvalRunResponse) -> list[str]:
    scored = report.graph_recall_scored
    lines = [
        f"- **Graph Recall** (advisory, unjudged): {_fmt(report.graph_recall)} over "
        f"{_plural(scored, 'query', 'queries')} with expected entities"
    ]
    lines += [
        f"  - {name}: {_fmt(rec)}" for name, rec in sorted(report.graph_recall_by_slice.items())
    ]
    return lines


def _route_accuracy_lines(report: EvalRunResponse) -> list[str]:
    excluded = [
        _plural(report.route_accuracy_unverified, "unverified fallback route"),
        _plural(report.route_accuracy_errored, "pipeline error"),
    ]
    lines = [
        f"- **Route Accuracy**: {_fmt(report.route_accuracy)} over "
        f"{report.route_accuracy_scored} routed "
        f"{'query' if report.route_accuracy_scored == 1 else 'queries'}"
        f" (not scored: {', '.join(excluded)}; chitchat Slice excluded)"
    ]
    floor = report.route_accuracy_floor
    if floor is not None and report.route_accuracy is not None:
        status = "met" if report.route_accuracy >= floor else "MISSED"
        lines.append(f"  - advisory floor {floor:.2f}: {status}")
    lines += [
        f"  - {name}: {_fmt(acc)}" for name, acc in sorted(report.route_accuracy_by_slice.items())
    ]
    lines += [
        f"  - misroute {m.query_id} ({m.slice}): chose {m.chosen_route}, expected {m.expected_route}"
        for m in report.misroutes
    ]
    return lines
