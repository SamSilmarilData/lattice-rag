from __future__ import annotations

import sys
from pathlib import Path
import click


@click.group()
def main() -> None:
    """lattice-rag: In-process Hybrid GraphRAG engine."""
    pass


@main.command()
def setup_keys() -> None:
    """Migrate apikeys.md to secure .env file."""
    from lattice_rag.security import migrate_apikeys_to_dotenv
    project_root = Path.cwd()
    apikeys_path = project_root / 'apikeys.md'
    dotenv_path = project_root / '.env'

    if not apikeys_path.exists():
        click.echo('No apikeys.md found. Nothing to migrate.')
        return

    result = migrate_apikeys_to_dotenv(apikeys_path, dotenv_path)
    if result:
        click.echo('✓ API keys migrated to .env (permissions: 600)')
        click.echo('✓ apikeys.md has been securely deleted')
    else:
        click.echo('No keys found in apikeys.md')


@main.command()
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--port', default=8000, type=int, help='Server port')
def serve(host: str, port: int) -> None:
    """Start the Granian ASGI server."""
    click.echo(f'Starting lattice-rag on {host}:{port}')
    from lattice_rag.server import main as server_main
    server_main()


@main.command()
@click.argument('text')
@click.option('--stream', is_flag=True, default=False, help='Stream generated tokens via SSE events')
def query(text: str, stream: bool) -> None:
    """Run a single query through the GraphRAG pipeline."""
    import asyncio
    from lattice_rag.caching.fallback_cache import FallbackCache
    from lattice_rag.caching.semantic_cache import SemanticCache
    from lattice_rag.config import get_config
    from lattice_rag.generation.chitchat import ChitchatHandler
    from lattice_rag.generation.gemini_fallback import GeminiFallback
    from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
    from lattice_rag.orchestration.graph import RAGOrchestrator
    from lattice_rag.retrieval.embeddings import EmbeddingService
    from lattice_rag.retrieval.pipeline import RetrievalPipeline
    from lattice_rag.routing.guardrail import ContextGuardrail
    from lattice_rag.routing.router import QueryRouter
    from lattice_rag.storage.db import LatticeStore

    async def _run() -> None:
        config = get_config()
        store = LatticeStore(config.latticedb_path)
        embed_svc = EmbeddingService(config.embed_model, config.reranker_model)
        router = QueryRouter()
        guardrail = ContextGuardrail()
        sem_cache = SemanticCache()
        fb_cache = FallbackCache(config.redis_url)
        await fb_cache.connect()

        groq = GroqSynthesizer(
            api_key=config.groq_api_key.get_secret_value() if config.groq_api_key else None,
            model=config.groq_model,
        )
        gemini = GeminiFallback(
            api_key=config.gemini_api_key.get_secret_value() if config.gemini_api_key else None,
            model=config.gemini_model,
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
            click.echo(f"Latency: {state.total_latency_ms:.1f}ms (Cached: {state.cache_hit}, Degraded: {state.degraded})")

        await fb_cache.close()
        await sem_cache.close()
        await router.close()
        await guardrail.close()
        store.close()

    asyncio.run(_run())


@main.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--doc-id', default=None, help='Document ID')
@click.option('--title', default=None, help='Document title')
def ingest(filepath: str, doc_id: str | None, title: str | None) -> None:
    """Ingest a text file into the knowledge graph."""
    import asyncio
    from lattice_rag.api.controllers.ingest import chunk_text
    from lattice_rag.config import get_config
    from lattice_rag.retrieval.embeddings import EmbeddingService
    from lattice_rag.storage.db import ChunkData, EntityData, LatticeStore
    from lattice_rag.storage.extract import EntityExtractor

    path = Path(filepath)
    text = path.read_text(encoding='utf-8')
    d_id = doc_id or path.stem
    d_title = title or path.name

    click.echo(f"Ingesting: {path.name} ({len(text)} chars) as doc_id='{d_id}'")

    async def _ingest() -> None:
        config = get_config()
        store = LatticeStore(config.latticedb_path)
        embed_svc = EmbeddingService(config.embed_model, config.reranker_model)
        extractor = EntityExtractor()

        text_chunks = chunk_text(text, 512, 64)
        click.echo(f"Created {len(text_chunks)} chunks. Generating embeddings...")
        embeddings = embed_svc.embed_texts(text_chunks)
        chunk_data = [ChunkData(text=t, embedding=e, position=i) for i, (t, e) in enumerate(zip(text_chunks, embeddings))]
        stats = store.ingest_document(d_id, d_title, chunk_data)

        total_entities = 0
        total_relations = 0
        click.echo("Extracting entities & relations...")
        for i, chunk_node_id in enumerate(stats.chunk_ids):
            c_text = text_chunks[i]
            entities = extractor.extract_entities(c_text)
            if entities:
                triples = extractor.extract_triples(c_text, entities)
                ent_names = [e.name for e in entities]
                ent_embs = embed_svc.embed_texts(ent_names)
                embedded_entities = [
                    EntityData(name=e.name, entity_type=e.entity_type, embedding=ent_embs[j])
                    for j, e in enumerate(entities)
                ]
                ent_stats = store.ingest_entities(chunk_node_id, embedded_entities, triples)
                total_entities += ent_stats.entity_count
                total_relations += ent_stats.relation_count

        click.echo(f"✓ Ingestion complete: {len(text_chunks)} chunks, {total_entities} entities, {total_relations} relations.")
        store.close()

    asyncio.run(_ingest())


@main.command()
@click.option('--dataset', default='eval_dataset.json', help='Path to dataset containing corpus')
def seed(dataset: str) -> None:
    """Pre-seed LatticeDB with the golden corpus documents."""
    import json
    import asyncio
    from lattice_rag.api.controllers.ingest import chunk_text
    from lattice_rag.config import get_config
    from lattice_rag.retrieval.embeddings import EmbeddingService
    from lattice_rag.storage.db import ChunkData, EntityData, LatticeStore
    from lattice_rag.storage.extract import EntityExtractor

    path = Path(dataset)
    if not path.exists():
        click.echo(f"Error: Dataset {dataset} not found.")
        return

    data = json.loads(path.read_text(encoding='utf-8'))
    corpus = data.get("corpus", [])
    if not corpus:
        click.echo("No corpus documents found in dataset.")
        return

    click.echo(f"Seeding LatticeDB with {len(corpus)} corpus documents from {dataset}...")

    async def _seed() -> None:
        config = get_config()
        store = LatticeStore(config.latticedb_path)
        embed_svc = EmbeddingService(config.embed_model, config.reranker_model)
        extractor = EntityExtractor()

        for doc in corpus:
            d_id = doc.get("document_id", "doc")
            d_title = doc.get("title", d_id)
            d_text = doc.get("text", "")
            click.echo(f"  Ingesting '{d_title}' ({len(d_text)} chars)...")

            text_chunks = chunk_text(d_text, 512, 64)
            embeddings = embed_svc.embed_texts(text_chunks)
            chunk_data = [ChunkData(text=t, embedding=e, position=i) for i, (t, e) in enumerate(zip(text_chunks, embeddings))]
            stats = store.ingest_document(d_id, d_title, chunk_data)

            for i, chunk_node_id in enumerate(stats.chunk_ids):
                c_text = text_chunks[i]
                entities = extractor.extract_entities(c_text)
                if entities:
                    triples = extractor.extract_triples(c_text, entities)
                    ent_names = [e.name for e in entities]
                    ent_embs = embed_svc.embed_texts(ent_names)
                    embedded_entities = [
                        EntityData(name=e.name, entity_type=e.entity_type, embedding=ent_embs[j])
                        for j, e in enumerate(entities)
                    ]
                    store.ingest_entities(chunk_node_id, embedded_entities, triples)

        click.echo("✓ LatticeDB pre-seeding complete!")
        store.close()

    asyncio.run(_seed())



@main.command()
def benchmark() -> None:
    """Run latency benchmarks on the retrieval pipeline."""
    click.echo('Running benchmarks...')
    click.echo('(Benchmark integration pending full wiring)')


@main.command()
@click.option('--dataset', default='eval_dataset.json', help='Path to evaluation dataset')
@click.option('--baseline', default='eval_baseline.json', help='Path to baseline scores')
@click.option('--fail-on-regression/--no-fail-on-regression', default=True, help='Exit with 1 if regression occurs')
def eval(dataset: str, baseline: str, fail_on_regression: bool) -> None:
    """Run the CI/CD evaluation gate against the golden benchmark dataset."""
    import asyncio
    import sys
    from pathlib import Path
    from lattice_rag.caching.fallback_cache import FallbackCache
    from lattice_rag.caching.semantic_cache import SemanticCache
    from lattice_rag.config import get_config
    from lattice_rag.eval import EvalRunner, JevEvaluator
    from lattice_rag.generation.chitchat import ChitchatHandler
    from lattice_rag.generation.gemini_fallback import GeminiFallback
    from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
    from lattice_rag.orchestration.graph import RAGOrchestrator
    from lattice_rag.orchestration.pool import shutdown_pool
    from lattice_rag.retrieval.embeddings import EmbeddingService
    from lattice_rag.retrieval.pipeline import RetrievalPipeline
    from lattice_rag.routing.guardrail import ContextGuardrail
    from lattice_rag.routing.router import QueryRouter
    from lattice_rag.storage.db import LatticeStore
    from lattice_rag.storage.extract import EntityExtractor

    click.echo(f"Evaluating dataset: {dataset} against baseline: {baseline}\n")

    async def _run() -> bool:
        config = get_config()
        store = LatticeStore(config.latticedb_path)
        embed_svc = EmbeddingService(config.embed_model, config.reranker_model)
        extractor = EntityExtractor()
        router = QueryRouter()
        guardrail = ContextGuardrail()
        sem_cache = SemanticCache()
        fb_cache = FallbackCache(config.redis_url)
        await fb_cache.connect()

        groq = GroqSynthesizer(
            api_key=config.groq_api_key.get_secret_value() if config.groq_api_key else None,
            model=config.groq_model,
        )
        gemini = GeminiFallback(
            api_key=config.gemini_api_key.get_secret_value() if config.gemini_api_key else None,
            model=config.gemini_model,
        )
        chitchat = ChitchatHandler()
        pipeline = RetrievalPipeline(store, embed_svc, guardrail)
        orchestrator = RAGOrchestrator(
            router=router,
            semantic_cache=sem_cache,
            fallback_cache=fb_cache,
            retrieval_pipeline=pipeline,
            groq=groq,
            gemini=gemini,
            chitchat=chitchat,
            embedding_service=embed_svc,
        )
        evaluator = JevEvaluator(api_key=config.typesafe_api_key)
        runner = EvalRunner(
            orchestrator=orchestrator,
            evaluator=evaluator,
            embedding_service=embed_svc,
            extractor=extractor,
        )

        try:
            resp = await runner.run_evaluation(dataset_path=dataset, baseline_path=baseline, use_temp_db=True)

            click.echo("\n=================== Evaluation Results ===================")
            click.echo(f"{'Query ID':<10} {'Faithfulness':<15} {'Precision':<15} {'Relevance':<15} {'Status':<10}")
            click.echo("-" * 65)
            for idx, r in enumerate(resp.results):
                status_str = "✓ PASS" if r.passed else "✗ FAIL"
                q_id = f"eval_{idx+1:02d}"
                click.echo(f"{q_id:<10} {r.faithfulness:<15.4f} {r.context_precision:<15.4f} {r.answer_relevance:<15.4f} {status_str:<10}")

            click.echo("=" * 65)
            click.echo(f"Total Evaluated Queries: {resp.total_queries}")
            click.echo(f"Mean Faithfulness:       {resp.mean_faithfulness:.4f}")
            click.echo(f"Mean Context Precision:  {resp.mean_context_precision:.4f}")
            click.echo(f"Mean Answer Relevance:   {resp.mean_answer_relevance:.4f}")
            click.echo(f"Max Regression Delta:    {resp.regression_delta:+.4f}")

            if resp.passed_gate:
                click.echo("\n✓ CI/CD QUALITY GATE PASSED (No regression detected)")
                return True
            else:
                click.echo("\n✗ CI/CD QUALITY GATE FAILED (Regression delta < -0.03 or floor violation)")
                return False
        finally:
            await evaluator.close()
            await fb_cache.close()
            await sem_cache.close()
            await router.close()
            await guardrail.close()
            store.close()
            shutdown_pool()

    passed = asyncio.run(_run())
    if fail_on_regression and not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
