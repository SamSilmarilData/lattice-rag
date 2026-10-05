from __future__ import annotations

import logging
from litestar import Controller, post
from litestar.di import NamedDependency
from litestar.params import SkipValidation

from lattice_rag.api.dtos import BenchmarkRunResponse, StageBenchmarkDTO
from lattice_rag.benchmark import BenchmarkEngine
from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.storage.db import LatticeStore

logger = logging.getLogger(__name__)


class BenchmarkController(Controller):
    """Litestar controller for executing automated performance latency benchmarks."""

    path = "/api/v1/benchmark"

    @post("/run")
    async def run_benchmark(
        self,
        store: NamedDependency[SkipValidation[LatticeStore]],
        embedding_service: NamedDependency[SkipValidation[EmbeddingService]],
        semantic_cache: NamedDependency[SkipValidation[SemanticCache]],
        orchestrator: NamedDependency[SkipValidation[RAGOrchestrator]],
    ) -> BenchmarkRunResponse:
        """Trigger an end-to-end latency benchmark across all pipeline stages."""
        logger.info("received_benchmark_run_request")
        engine = BenchmarkEngine(
            store=store,
            embedding_service=embedding_service,
            semantic_cache=semantic_cache,
            orchestrator=orchestrator,
        )

        report = await engine.run_full_suite(iterations=6, warmup=2, live_llm=False)

        stages_dtos = [
            StageBenchmarkDTO(
                stage=s.stage,
                samples_count=s.samples_count,
                mean_ms=s.mean_ms,
                min_ms=s.min_ms,
                p50_ms=s.p50_ms,
                p90_ms=s.p90_ms,
                p95_ms=s.p95_ms,
                p99_ms=s.p99_ms,
                max_ms=s.max_ms,
                qps=s.qps,
            )
            for s in report.stages.values()
        ]

        return BenchmarkRunResponse(
            total_duration_sec=report.total_duration_sec,
            sub_second_sla_met=report.sub_second_sla_met,
            tier1_cache_sla_met=report.tier1_cache_sla_met,
            stages=stages_dtos,
        )
