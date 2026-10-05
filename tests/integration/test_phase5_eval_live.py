"""Live integration test for Phase 5 Jev-Native Evaluation Gate.

Verifies end-to-end evaluation flow: ephemeral LatticeDB seeding, query execution,
live TypeSafe AI Jev Score evaluation, and baseline comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from lattice_rag.caching.fallback_cache import FallbackCache
from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.config import get_config
from lattice_rag.eval import EvalRunner, JevEvaluator
from lattice_rag.generation.chitchat import ChitchatHandler
from lattice_rag.generation.gemini_fallback import GeminiFallback
from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.pipeline import RetrievalPipeline
from lattice_rag.routing.guardrail import ContextGuardrail
from lattice_rag.routing.router import QueryRouter
from lattice_rag.storage.db import LatticeStore
from lattice_rag.storage.extract import EntityExtractor


@pytest.mark.asyncio
async def test_live_phase5_eval_runner_flow(tmp_path: Path):
    """Verify live Jev-native evaluation on representative queries with ephemeral database."""
    config = get_config()

    # Load 3 representative queries from eval_dataset.json for fast integration check
    dataset_file = Path("eval_dataset.json")
    assert dataset_file.exists()
    with open(dataset_file, encoding="utf-8") as f:
        full_ds = json.load(f)

    mini_dataset = {
        "version": full_ds.get("version", "1.0.0"),
        "corpus": full_ds.get("corpus", [])[:2],  # First 2 documents
        "queries": full_ds.get("queries", [])[:2],  # First 2 queries
    }
    mini_baseline = {
        "version": "0.4.0",
        "metrics": {
            "faithfulness": 0.80,
            "context_precision": 0.75,
            "answer_relevance": 0.80,
        },
        "gate_thresholds": {
            "max_regression_delta": 0.05,
            "min_query_floor": 0.50,
        },
    }

    mini_ds_path = tmp_path / "mini_eval_dataset.json"
    mini_base_path = tmp_path / "mini_eval_baseline.json"

    mini_ds_path.write_text(json.dumps(mini_dataset), encoding="utf-8")
    mini_base_path.write_text(json.dumps(mini_baseline), encoding="utf-8")

    # Wire real services
    dummy_db = tmp_path / "dummy_template.db"
    store = LatticeStore(dummy_db)
    embed_svc = EmbeddingService(config.embed_model, config.reranker_model)
    extractor = EntityExtractor()
    router = QueryRouter()
    guardrail = ContextGuardrail()
    sem_cache = SemanticCache()
    fb_cache = FallbackCache(redis_url="embedded")
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
        resp = await runner.run_evaluation(
            dataset_path=mini_ds_path,
            baseline_path=mini_base_path,
            use_temp_db=True,
            max_concurrency=2,
        )

        assert resp.total_queries == 2
        assert 0.0 <= resp.mean_faithfulness <= 1.0
        assert 0.0 <= resp.mean_context_precision <= 1.0
        assert 0.0 <= resp.mean_answer_relevance <= 1.0
        assert len(resp.results) == 2

        print(f"\n[Live Phase 5 Eval Results]:")
        print(f"Mean Faithfulness:      {resp.mean_faithfulness:.4f}")
        print(f"Mean Context Precision: {resp.mean_context_precision:.4f}")
        print(f"Mean Answer Relevance:  {resp.mean_answer_relevance:.4f}")
        print(f"Gate Passed:            {resp.passed_gate}")
        print(f"Regression Delta:       {resp.regression_delta:+.4f}")

        for i, r in enumerate(resp.results):
            print(f"Query {i+1}: {r.query[:50]}... | Faith: {r.faithfulness:.2f} | Prec: {r.context_precision:.2f} | Rel: {r.answer_relevance:.2f}")

    finally:
        try:
            await evaluator.close()
        except Exception:
            pass
        try:
            await fb_cache.close()
        except Exception:
            pass
        try:
            await sem_cache.close()
        except Exception:
            pass
        try:
            await router.close()
        except Exception:
            pass
        try:
            await guardrail.close()
        except Exception:
            pass
        try:
            store.close()
        except Exception:
            pass
