from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from typing import Any
import structlog

from lattice_rag.api.dtos import EvalQueryResult, EvalRunResponse
from lattice_rag.eval.triage_gate import JevEvaluator
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.pipeline import RetrievalPipeline
from lattice_rag.storage.db import ChunkData, LatticeStore
from lattice_rag.storage.extract import EntityExtractor

logger = structlog.get_logger(__name__)


class EvalRunner:
    """Orchestrator for Golden Dataset execution, Jev evaluation, and regression gating."""

    def __init__(
        self,
        orchestrator: RAGOrchestrator,
        evaluator: JevEvaluator,
        embedding_service: EmbeddingService | None = None,
        extractor: EntityExtractor | None = None,
    ) -> None:
        """Initialize EvalRunner with orchestrator, Jev evaluator, and optional support services."""
        self.orchestrator = orchestrator
        self.evaluator = evaluator
        self.embedding_service = embedding_service
        self.extractor = extractor

    async def run_evaluation(
        self,
        dataset_path: str | Path = "eval_dataset.json",
        baseline_path: str | Path = "eval_baseline.json",
        use_temp_db: bool = True,
        max_concurrency: int = 5,
    ) -> EvalRunResponse:
        """Execute full evaluation run against golden benchmark dataset and compare with baseline."""
        ds_file = Path(dataset_path)
        if not ds_file.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at: {ds_file}")

        base_file = Path(baseline_path)
        if not base_file.exists():
            raise FileNotFoundError(f"Evaluation baseline not found at: {base_file}")

        with open(ds_file, encoding="utf-8") as f:
            dataset_data = json.load(f)

        with open(base_file, encoding="utf-8") as f:
            baseline_data = json.load(f)

        queries_data: list[dict[str, Any]] = dataset_data.get("queries", [])
        corpus_data: list[dict[str, Any]] = dataset_data.get("corpus", [])

        if not queries_data:
            raise ValueError(f"No queries found in dataset: {ds_file}")

        active_orchestrator = self.orchestrator
        temp_dir_obj: tempfile.TemporaryDirectory | None = None
        temp_store: LatticeStore | None = None

        if use_temp_db and corpus_data:
            logger.info("seeding_ephemeral_eval_database", doc_count=len(corpus_data))
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_db_path = Path(temp_dir_obj.name) / "eval_ephemeral.db"
            temp_store = LatticeStore(temp_db_path)

            embed_svc = self.embedding_service or self.orchestrator.retrieval.embedding_service
            extractor = self.extractor or EntityExtractor()

            # Seed documents
            for doc in corpus_data:
                doc_id = doc.get("document_id", "doc_eval")
                title = doc.get("title", "Evaluation Document")
                text = doc.get("text", "")

                # Paragraph / sentence chunking (~512 chars with 64 overlap)
                step = 512 - 64
                text_chunks: list[str] = []
                for i in range(0, len(text), step):
                    chunk_text = text[i : i + 512].strip()
                    if chunk_text:
                        text_chunks.append(chunk_text)

                if not text_chunks:
                    text_chunks = [text]

                embeddings = embed_svc.embed_texts(text_chunks)
                chunk_nodes = [
                    ChunkData(text=c_text, embedding=c_emb, position=idx)
                    for idx, (c_text, c_emb) in enumerate(zip(text_chunks, embeddings))
                ]
                stats = temp_store.ingest_document(doc_id=doc_id, title=title, chunks=chunk_nodes)

                # Extract and persist entities
                for chunk_id, c_text in zip(stats.chunk_ids, text_chunks):
                    entities = extractor.extract_entities(c_text)
                    if entities:
                        triples = extractor.extract_triples(c_text, entities)
                        temp_store.ingest_entities(chunk_id=chunk_id, entities=entities, relations=triples)

            # Build ephemeral retrieval pipeline and orchestrator
            ephemeral_retrieval = RetrievalPipeline(
                store=temp_store,
                embedding_service=embed_svc,
                router_guardrail=self.orchestrator.retrieval.router_guardrail,
            )

            active_orchestrator = RAGOrchestrator(
                router=self.orchestrator.router,
                semantic_cache=self.orchestrator.semantic_cache,
                fallback_cache=self.orchestrator.fallback_cache,
                retrieval_pipeline=ephemeral_retrieval,
                groq=self.orchestrator.groq,
                gemini=self.orchestrator.gemini,
                chitchat=self.orchestrator.chitchat,
                embedding_service=embed_svc,
            )

        # Clear semantic cache before eval run so every query is freshly evaluated
        if active_orchestrator.semantic_cache is not None:
            active_orchestrator.semantic_cache.clear()

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _eval_single(item: dict[str, Any]) -> EvalQueryResult:
            async with semaphore:
                query = item["query"]
                ground_truth = item.get("ground_truth", "")

                try:
                    pipeline_state = await active_orchestrator.run(query, stream=False)
                    answer = pipeline_state.answer
                    chunks = pipeline_state.filtered_chunks
                except Exception as e:
                    logger.error("eval_query_execution_failed", query=query, error=str(e))
                    answer = f"ERROR: {e}"
                    chunks = []

                return await self.evaluator.evaluate_query(
                    query=query,
                    retrieved_chunks=chunks,
                    generated_answer=answer,
                    ground_truth=ground_truth,
                )

        logger.info("running_eval_suite", total_queries=len(queries_data))
        eval_tasks = [_eval_single(q) for q in queries_data]
        results = await asyncio.gather(*eval_tasks)

        # Clean up ephemeral resources
        if temp_store is not None:
            try:
                temp_store.close()
            except Exception:
                pass

        if temp_dir_obj is not None:
            try:
                temp_dir_obj.cleanup()
            except Exception:
                pass

        # Calculate mean scores
        total = len(results)
        mean_faith = sum(r.faithfulness for r in results) / total
        mean_prec = sum(r.context_precision for r in results) / total
        mean_rel = sum(r.answer_relevance for r in results) / total

        # Baseline comparison
        base_metrics = baseline_data.get("metrics", {})
        base_faith = base_metrics.get("faithfulness", 0.88)
        base_prec = base_metrics.get("context_precision", 0.85)
        base_rel = base_metrics.get("answer_relevance", 0.90)

        delta_faith = mean_faith - base_faith
        delta_prec = mean_prec - base_prec
        delta_rel = mean_rel - base_rel

        gate_thresholds = baseline_data.get("gate_thresholds", {})
        max_allowed_drop = gate_thresholds.get("max_regression_delta", 0.03)
        min_query_floor = gate_thresholds.get("min_query_floor", 0.50)

        # Non-regression: PR score cannot drop by more than max_allowed_drop (e.g. -0.03)
        faith_passed = delta_faith >= -max_allowed_drop
        prec_passed = delta_prec >= -max_allowed_drop
        rel_passed = delta_rel >= -max_allowed_drop

        # Safety floor: no individual query should drop below critical floor
        floor_passed = all(r.faithfulness >= min_query_floor for r in results)

        passed_gate = faith_passed and prec_passed and rel_passed and floor_passed
        min_delta = min(delta_faith, delta_prec, delta_rel)

        logger.info(
            "eval_suite_completed",
            total_queries=total,
            mean_faithfulness=round(mean_faith, 4),
            mean_context_precision=round(mean_prec, 4),
            mean_answer_relevance=round(mean_rel, 4),
            delta_faithfulness=round(delta_faith, 4),
            delta_context_precision=round(delta_prec, 4),
            delta_answer_relevance=round(delta_rel, 4),
            passed_gate=passed_gate,
            floor_passed=floor_passed,
        )

        return EvalRunResponse(
            total_queries=total,
            mean_faithfulness=round(mean_faith, 4),
            mean_context_precision=round(mean_prec, 4),
            mean_answer_relevance=round(mean_rel, 4),
            passed_gate=passed_gate,
            regression_delta=round(min_delta, 4),
            results=list(results),
        )
