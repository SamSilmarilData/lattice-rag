from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click


@click.group()
def main() -> None:
    """lattice-rag: In-process Hybrid GraphRAG engine."""
    from lattice_rag.security import configure_logging

    configure_logging()


@main.command()
def setup_keys() -> None:
    """Migrate apikeys.md to secure .env file."""
    from lattice_rag.security import migrate_apikeys_to_dotenv

    project_root = Path.cwd()
    apikeys_path = project_root / "apikeys.md"
    dotenv_path = project_root / ".env"

    if not apikeys_path.exists():
        click.echo("No apikeys.md found. Nothing to migrate.")
        return

    result = migrate_apikeys_to_dotenv(apikeys_path, dotenv_path)
    if result:
        click.echo("✓ API keys migrated to .env (permissions: 600)")
        click.echo("✓ apikeys.md has been securely deleted")
    else:
        click.echo("No keys found in apikeys.md")


@main.command()
@click.option("--host", default=None, help="Server host (default: HOST env or 0.0.0.0)")
@click.option("--port", default=None, type=int, help="Server port (default: PORT env or 8000)")
def serve(host: str | None, port: int | None) -> None:
    """Start the Granian ASGI server."""
    from lattice_rag.server import main as server_main

    server_main(host=host, port=port)


async def _build_harness() -> dict[str, Any]:
    """Build unified CLI service harness with all dependencies wired."""
    from lattice_rag.caching.fallback_cache import FallbackCache
    from lattice_rag.caching.semantic_cache import SemanticCache
    from lattice_rag.config import get_config
    from lattice_rag.generation.chitchat import ChitchatHandler
    from lattice_rag.generation.gemini_fallback import GeminiFallback
    from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
    from lattice_rag.orchestration.graph import RAGOrchestrator
    from lattice_rag.retrieval.embeddings import EmbeddingService
    from lattice_rag.retrieval.pipeline import RetrievalPipeline
    from lattice_rag.routing.guardrail import ContextGuardrail, with_rerank_skip
    from lattice_rag.routing.router import QueryRouter
    from lattice_rag.storage.db import LatticeStore

    config = get_config()
    store = LatticeStore(config.latticedb_path)
    embed_svc = EmbeddingService(
        config.embed_model,
        config.reranker_model,
        onnx_reranker_repo=config.reranker_onnx_repo,
        onnx_reranker_revision=config.reranker_onnx_revision,
    )
    router = QueryRouter()
    guardrail = with_rerank_skip(
        ContextGuardrail(threshold=0.5), config.guardrail_skip_rerank_score
    )
    sem_cache = SemanticCache()
    fb_cache = FallbackCache(config.redis_url)
    await fb_cache.connect()

    groq = GroqSynthesizer(
        api_key=config.groq_api_key.get_secret_value() if config.groq_api_key else None,
        model=config.groq_model,
        max_tokens=config.llm_max_tokens,
    )
    gemini = GeminiFallback(
        api_key=config.gemini_api_key.get_secret_value() if config.gemini_api_key else None,
        model=config.gemini_model,
        max_output_tokens=config.llm_max_tokens,
    )
    cc = ChitchatHandler()
    pipeline = RetrievalPipeline(store, embed_svc, guardrail)
    orchestrator = RAGOrchestrator(
        router=router,
        semantic_cache=sem_cache,
        fallback_cache=fb_cache,
        retrieval_pipeline=pipeline,
        groq=groq,
        gemini=gemini,
        chitchat=cc,
        embedding_service=embed_svc,
    )
    return {
        "config": config,
        "store": store,
        "embed_svc": embed_svc,
        "router": router,
        "guardrail": guardrail,
        "sem_cache": sem_cache,
        "fb_cache": fb_cache,
        "pipeline": pipeline,
        "orchestrator": orchestrator,
    }


async def _close_harness(harness: dict[str, Any]) -> None:
    """Gracefully close harness connections."""
    await harness["fb_cache"].close()
    await harness["sem_cache"].close()
    await harness["router"].close()
    await harness["guardrail"].close()
    harness["store"].close()


@main.command()
@click.argument("text")
@click.option(
    "--stream", is_flag=True, default=False, help="Stream generated tokens via SSE events"
)
def query(text: str, stream: bool) -> None:
    """Run a single query through the GraphRAG pipeline."""
    import asyncio

    async def _run() -> None:
        h = await _build_harness()
        try:
            await _query(h["orchestrator"])
        finally:
            await _close_harness(h)

    async def _query(orchestrator: Any) -> None:
        click.echo(f"Query: {text}\n")
        if stream:
            click.echo("--- Streaming Response ---")
            async for evt in orchestrator.stream_query(text):
                if evt["event"] == "stage":
                    stage_name = evt["data"].get("stage")
                    click.echo(f"[{stage_name.upper()}] ", nl=False)
                elif evt["event"] == "token":
                    click.echo(evt["data"].get("delta", ""), nl=False)
                elif evt["event"] == "done":
                    click.echo("\n--- Done ---")
                    click.echo(f"Total Latency: {evt['data']['latency_ms']:.1f}ms")
        else:
            state = await orchestrator.run(text)
            click.echo(f"Route: {state.route}")
            click.echo(f"Answer: {state.answer}")
            click.echo(
                f"Latency: {state.total_latency_ms:.1f}ms (Cached: {state.cache_hit}, Degraded: {state.degraded})"
            )

    asyncio.run(_run())


@main.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--doc-id", default=None, help="Document ID")
@click.option("--title", default=None, help="Document title")
@click.option(
    "--replace",
    is_flag=True,
    default=False,
    help="Atomically replace an existing document with this doc_id",
)
def ingest(filepath: str, doc_id: str | None, title: str | None, replace: bool) -> None:
    """Ingest a text file into the knowledge graph."""
    import asyncio

    from lattice_rag.config import load_config
    from lattice_rag.ingestion import DocumentIngester
    from lattice_rag.ingestion.ingester import DocumentExistsError
    from lattice_rag.retrieval.embeddings import EmbeddingService
    from lattice_rag.storage.db import IngestValidationError, LatticeStore
    from lattice_rag.storage.extract import EntityExtractor

    path = Path(filepath)
    click.echo(f"Ingesting: {path.name} as doc_id='{doc_id or path.stem}'")

    async def _ingest() -> None:
        config = load_config(require_typesafe=False)
        store = LatticeStore(config.latticedb_path)
        embed_svc = EmbeddingService(
            config.embed_model,
            config.reranker_model,
            onnx_reranker_repo=config.reranker_onnx_repo,
            onnx_reranker_revision=config.reranker_onnx_revision,
        )
        extractor = EntityExtractor()
        ingester = DocumentIngester(store=store, embedding_service=embed_svc, extractor=extractor)

        try:
            stats = await ingester.ingest_file(path, doc_id=doc_id, title=title, replace=replace)
        except DocumentExistsError as e:
            raise click.ClickException(f"{e}. Use a different --doc-id, or --replace.") from e
        except IngestValidationError as e:
            raise click.ClickException(f"Ingest rejected, nothing was written: {e}") from e
        finally:
            store.close()
        click.echo(
            f"✓ Ingestion complete: {stats.chunk_count} chunks, {stats.entity_count} entities, {stats.relation_count} relations."
        )

    asyncio.run(_ingest())


@main.command()
@click.option("--dataset", default="eval_dataset.json", help="Path to dataset containing corpus")
def seed(dataset: str) -> None:
    """Pre-seed LatticeDB with the corpus of a validated Gold Set."""
    import asyncio

    from lattice_rag.config import load_config
    from lattice_rag.eval.gold_set import GoldSetError, load_gold_set
    from lattice_rag.ingestion import DocumentIngester
    from lattice_rag.ingestion.ingester import DocumentExistsError
    from lattice_rag.retrieval.embeddings import EmbeddingService
    from lattice_rag.storage.db import LatticeStore
    from lattice_rag.storage.extract import EntityExtractor

    path = Path(dataset)
    if not path.exists():
        click.echo(f"Error: Dataset {dataset} not found.")
        return

    try:
        corpus = load_gold_set(path).corpus
    except GoldSetError as e:
        raise click.ClickException(str(e)) from e
    if not corpus:
        click.echo("No corpus documents found in dataset.")
        return

    click.echo(f"Seeding LatticeDB with {len(corpus)} corpus documents from {dataset}...")

    async def _seed() -> None:
        config = load_config(require_typesafe=False)
        store = LatticeStore(config.latticedb_path)
        embed_svc = EmbeddingService(
            config.embed_model,
            config.reranker_model,
            onnx_reranker_repo=config.reranker_onnx_repo,
            onnx_reranker_revision=config.reranker_onnx_revision,
        )
        extractor = EntityExtractor()
        ingester = DocumentIngester(store=store, embedding_service=embed_svc, extractor=extractor)

        for doc in corpus:
            d_id, d_title, d_text = doc.document_id, doc.title, doc.text
            try:
                await ingester.ingest(d_id, d_title, d_text)
                click.echo(f"  Ingested '{d_title}' ({len(d_text)} chars)")
            except DocumentExistsError:
                click.echo(f"  Skipped '{d_title}' (already seeded)")

        click.echo("✓ LatticeDB pre-seeding complete!")
        store.close()

    asyncio.run(_seed())


@main.command()
@click.option("--dataset", default="eval_dataset.json", help="Path to the Gold Set")
@click.option(
    "--thresholds",
    default="hermetic_thresholds.json",
    help="Path to the versioned recall/MRR floors",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(writable=True),
    help="Export the full report to a JSON file",
)
def hermetic(dataset: str, thresholds: str, output: str | None) -> None:
    """Run the key-free Hermetic Tier: Gold Passage recall and MRR per Slice (the PR gate)."""
    import asyncio

    import msgspec

    from lattice_rag.config import load_config
    from lattice_rag.eval.gold_set import GoldSetError, load_gold_set
    from lattice_rag.eval.hermetic import HermeticError, load_thresholds, run_hermetic_tier
    from lattice_rag.retrieval.embeddings import EmbeddingService

    try:
        gold_set = load_gold_set(dataset)
        gate = load_thresholds(thresholds)
    except (FileNotFoundError, GoldSetError, HermeticError) as e:
        raise click.ClickException(str(e)) from e

    config = load_config(require_typesafe=False)
    embed_svc = EmbeddingService(
        config.embed_model,
        config.reranker_model,
        onnx_reranker_repo=config.reranker_onnx_repo,
        onnx_reranker_revision=config.reranker_onnx_revision,
    )
    click.echo(f"Hermetic Tier: {len(gold_set.queries)} queries from {dataset}\n")
    try:
        report = asyncio.run(run_hermetic_tier(gold_set, gate, embed_svc))
    except HermeticError as e:
        raise click.ClickException(str(e)) from e

    def fmt(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    click.echo(f"{'Query ID':<12} {'Slice':<14} {'Recall':>7} {'RR':>7} {'Top score':>10}  Status")
    click.echo("-" * 64)
    for q in report.queries:
        status = "FAIL" if q.problems else "ok"
        click.echo(
            f"{q.id:<12} {q.slice:<14} {fmt(q.recall):>7} {fmt(q.reciprocal_rank):>7} "
            f"{fmt(q.top_rerank_score):>10}  {status}"
        )
    click.echo("")
    click.echo(f"{'Group':<14} {'Queries':>7} {'Recall':>7} {'MRR':>7} {'Floor R/MRR':>14}")
    click.echo("-" * 54)
    for m in [report.overall, *report.slices]:
        floor = f"{m.floor.recall:.2f}/{m.floor.mrr:.2f}" if m.floor else "none"
        click.echo(f"{m.slice:<14} {m.queries:>7} {m.recall:>7.3f} {m.mrr:>7.3f} {floor:>14}")

    if output:
        Path(output).write_bytes(msgspec.json.encode(report))
        click.echo(f"\nReport written to {output}")

    if report.passed:
        click.echo("\n✓ HERMETIC TIER PASSED")
        return
    click.echo("\n✗ HERMETIC TIER FAILED")
    for failure in report.failures:
        click.echo(f"  - {failure}")
    sys.exit(1)


@main.command()
@click.option(
    "--dataset",
    default="eval_dataset.json",
    help="Path to evaluation dataset containing benchmark queries",
)
@click.option(
    "--iterations", default=6, type=int, help="Sample iterations per benchmarked component"
)
@click.option("--warmup", default=2, type=int, help="Warmup iterations before sampling")
@click.option(
    "--live",
    is_flag=True,
    default=False,
    help="Execute live Groq LLM generation (default: zero-cost CPU hybrid)",
)
@click.option(
    "--sla-check",
    is_flag=True,
    default=False,
    help="Enforce Dual SLA gating (exit 1 if E2E p95 >= 1000ms or Cache p95 >= 25ms)",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(writable=True),
    help="Export benchmark metrics to JSON file",
)
def benchmark(
    dataset: str,
    iterations: int,
    warmup: int,
    live: bool,
    sla_check: bool,
    output: str | None,
) -> None:
    """Run statistical latency benchmarks and SLA validation across the pipeline."""
    import asyncio
    import json
    from pathlib import Path

    from lattice_rag.benchmark import BenchmarkEngine

    click.echo("================================================================================")
    click.echo("  lattice-rag Statistical Latency Benchmark Engine")
    click.echo(
        f"  Mode: {'Live GroqCloud LLM' if live else 'Zero-Cost CPU Hybrid'} | Iterations: {iterations} | Warmup: {warmup}"
    )
    click.echo("================================================================================\n")

    queries: list[str] = []
    dataset_path = Path(dataset)
    if dataset_path.exists():
        try:
            raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
            queries = [q["query"] for q in raw_data.get("queries", [])]
        except Exception as e:
            click.echo(f"Warning: Failed to load queries from {dataset}: {e}")

    async def _run() -> None:
        h = await _build_harness()
        engine = BenchmarkEngine(
            store=h["store"],
            embedding_service=h["embed_svc"],
            semantic_cache=h["sem_cache"],
            orchestrator=h["orchestrator"],
        )

        click.echo("Profiling pipeline stages (p50, p90, p95, p99, throughput)...")
        report = await engine.run_full_suite(
            queries=queries or None,
            iterations=iterations,
            warmup=warmup,
            live_llm=live,
        )

        table_str = BenchmarkEngine.format_ascii_table(report)
        click.echo(table_str)

        if output:
            out_dict = {
                "totalDurationSec": report.total_duration_sec,
                "subSecondSlaMet": report.sub_second_sla_met,
                "tier1CacheSlaMet": report.tier1_cache_sla_met,
                "stages": {
                    k: {
                        "stage": s.stage,
                        "samples": s.samples_count,
                        "meanMs": s.mean_ms,
                        "p50Ms": s.p50_ms,
                        "p90Ms": s.p90_ms,
                        "p95Ms": s.p95_ms,
                        "p99Ms": s.p99_ms,
                        "maxMs": s.max_ms,
                        "qps": s.qps,
                    }
                    for k, s in report.stages.items()
                },
            }
            Path(output).write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
            click.echo(f"✓ Exported benchmark metrics to {output}")

        await _close_harness(h)

        if sla_check:
            if not report.sub_second_sla_met or not report.tier1_cache_sla_met:
                click.echo(
                    "❌ SLA Violation: One or more latency SLA targets exceeded threshold.",
                    err=True,
                )
                sys.exit(1)
            else:
                click.echo("✓ Dual SLA targets verified successfully.")

    asyncio.run(_run())


@main.command(name="eval")
@click.option("--dataset", default="eval_dataset.json", help="Path to evaluation dataset")
@click.option("--baseline", default="eval_baseline.json", help="Path to baseline scores")
@click.option(
    "--fail-on-regression/--no-fail-on-regression",
    default=True,
    help="Exit with 1 if regression occurs",
)
@click.option(
    "--output",
    default=None,
    help="Write the full run report (JSON) here, whether or not the gate passes",
)
def run_eval(dataset: str, baseline: str, fail_on_regression: bool, output: str | None) -> None:
    """Run the Live Tier: Jev-judged metrics over the Gold Set, compared with the baseline."""
    import asyncio

    from lattice_rag.config import load_config
    from lattice_rag.eval import EvalRunner, JevEvaluator
    from lattice_rag.eval.gate import format_metric
    from lattice_rag.storage.extract import EntityExtractor

    # Without the judge key nothing can be scored; without Groq every answer would come
    # from the fallback, so the run would measure the wrong system. Either is a failure.
    keys = load_config(require_typesafe=False)
    missing = [
        name
        for name, value in (
            ("TYPESAFE_API_KEY", keys.typesafe_api_key),
            ("GROQ_API_KEY", keys.groq_api_key),
        )
        if not value
    ]
    if missing:
        raise click.ClickException(
            f"The Live Tier needs {', '.join(missing)}; set it in the environment or .env "
            "(in CI, as a repository secret). Refusing to run a skipped or degraded evaluation."
        )

    click.echo(f"Evaluating dataset: {dataset} against baseline: {baseline}\n")

    async def _run() -> bool:
        h = await _build_harness()
        evaluator = JevEvaluator(api_key=h["config"].typesafe_api_key)
        runner = EvalRunner(
            orchestrator=h["orchestrator"],
            evaluator=evaluator,
            embedding_service=h["embed_svc"],
            extractor=EntityExtractor(),
        )

        try:
            resp = await runner.run_evaluation(
                dataset_path=dataset, baseline_path=baseline, use_temp_db=True
            )
            if output:
                import msgspec

                Path(output).write_bytes(msgspec.json.encode(resp))
                click.echo(f"Report written to {output}")

            click.echo("\n=================== Evaluation Results ===================")
            click.echo(
                f"{'Query ID':<10} {'Faithfulness':<15} {'Precision':<15} {'Relevance':<15} "
                f"{'Correctness':<15} {'Status':<10}"
            )
            click.echo("-" * 80)
            for idx, r in enumerate(resp.results):
                status_str = "✓ PASS" if r.passed else f"✗ FAIL {r.error or ''}".rstrip()
                q_id = f"eval_{idx + 1:02d}"
                click.echo(
                    f"{q_id:<10} {format_metric(r.faithfulness):<15} "
                    f"{format_metric(r.context_precision):<15} "
                    f"{format_metric(r.answer_relevance):<15} "
                    f"{format_metric(r.answer_correctness):<15} {status_str:<10}"
                )

            click.echo("=" * 80)
            click.echo(f"Total Evaluated Queries: {resp.total_queries}")
            click.echo(f"Failed (not judged):     {resp.failed_queries}")
            click.echo(f"Mean Faithfulness:       {format_metric(resp.mean_faithfulness)}")
            click.echo(f"Mean Context Precision:  {format_metric(resp.mean_context_precision)}")
            click.echo(f"Mean Answer Relevance:   {format_metric(resp.mean_answer_relevance)}")
            click.echo(f"Mean Answer Correctness: {format_metric(resp.mean_answer_correctness)}")
            click.echo(f"Max Regression Delta:    {resp.regression_delta:+.4f}")

            if resp.passed_gate:
                click.echo("\n✓ CI/CD QUALITY GATE PASSED (No regression detected)")
                return True
            else:
                click.echo(
                    "\n✗ CI/CD QUALITY GATE FAILED (failed query, floor violation, or regression; "
                    "thresholds are in the baseline file)"
                )
                return False
        finally:
            await evaluator.close()
            await _close_harness(h)

    passed = asyncio.run(_run())
    if fail_on_regression and not passed:
        sys.exit(1)


@main.command(name="measure-baseline")
@click.argument("reports", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--baseline", default="eval_baseline.json", help="Baseline file to update")
def measure_baseline(reports: tuple[str, ...], baseline: str) -> None:
    """Set baseline means and tolerance from repeated `eval --output` reports.

    Run the Live Tier three times on unchanged code, then pass the three reports here.
    """
    from datetime import UTC, datetime

    import msgspec

    from lattice_rag.api.dtos import EvalRunResponse
    from lattice_rag.eval.gate import BaselineError, format_metric, measure, write_measurement

    runs = [msgspec.json.decode(Path(p).read_bytes(), type=EvalRunResponse) for p in reports]
    sizes = {r.total_queries for r in runs}
    if len(sizes) != 1:
        raise click.ClickException(f"Reports cover different query counts: {sorted(sizes)}")
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        measurement = measure(runs, measured_at=stamp)
        write_measurement(baseline, measurement, total_queries=sizes.pop())
    except BaselineError as e:
        raise click.ClickException(str(e)) from e

    for metric, mean in measurement.means.items():
        click.echo(
            f"{metric:<20} mean {format_metric(mean)}  sd {format_metric(measurement.stdev[metric])}"
        )
    click.echo(f"max_regression_delta -> {measurement.tolerance:.2f} (written to {baseline})")


if __name__ == "__main__":
    main()
