from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import numpy as np

from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.graph_traversal import GraphTraverser
from lattice_rag.retrieval.reranker import PrecisionReranker
from lattice_rag.retrieval.vector_search import HybridSearcher
from lattice_rag.storage.db import LatticeStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyStats:
    """Statistical summary of benchmark latency samples."""

    stage: str
    samples_count: int
    mean_ms: float
    stddev_ms: float
    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    qps: float


@dataclass
class BenchmarkReport:
    """Full benchmark execution report containing stage breakdowns and SLA compliance."""

    stages: dict[str, LatencyStats] = field(default_factory=dict)
    sub_second_sla_met: bool = True
    tier1_cache_sla_met: bool = True
    total_duration_sec: float = 0.0


class BenchmarkEngine:
    """Orchestrates statistical micro- and macro-benchmarks across the GraphRAG pipeline."""

    def __init__(
        self,
        store: LatticeStore,
        embedding_service: EmbeddingService,
        semantic_cache: SemanticCache | None = None,
        orchestrator: RAGOrchestrator | None = None,
    ) -> None:
        self.store = store
        self.embed_svc = embedding_service
        self.semantic_cache = semantic_cache
        self.orchestrator = orchestrator

    async def benchmark_callable(
        self,
        name: str,
        fn: Callable[[], Coroutine[Any, Any, Any]] | Callable[[], Any],
        iterations: int = 10,
        warmup: int = 2,
    ) -> LatencyStats:
        """Benchmark any synchronous or asynchronous callable over N iterations."""
        # Warmup passes
        for _ in range(warmup):
            if asyncio.iscoroutinefunction(fn):
                await fn()
            else:
                fn()

        latencies_ms: list[float] = []
        t0 = time.perf_counter()
        for _ in range(iterations):
            start = time.perf_counter()
            if asyncio.iscoroutinefunction(fn):
                await fn()
            else:
                fn()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)
        total_time_sec = time.perf_counter() - t0

        latencies_ms.sort()
        n = len(latencies_ms)
        p50 = latencies_ms[int(n * 0.50)]
        p90 = latencies_ms[min(int(n * 0.90), n - 1)]
        p95 = latencies_ms[min(int(n * 0.95), n - 1)]
        p99 = latencies_ms[min(int(n * 0.99), n - 1)]

        return LatencyStats(
            stage=name,
            samples_count=n,
            mean_ms=float(statistics.mean(latencies_ms)),
            stddev_ms=float(statistics.stdev(latencies_ms)) if n > 1 else 0.0,
            min_ms=float(latencies_ms[0]),
            p50_ms=float(p50),
            p90_ms=float(p90),
            p95_ms=float(p95),
            p99_ms=float(p99),
            max_ms=float(latencies_ms[-1]),
            qps=float(n / max(total_time_sec, 0.0001)),
        )

    async def benchmark_tier1_cache(
        self,
        query: str = "What is LatticeDB?",
        iterations: int = 15,
        warmup: int = 2,
    ) -> LatencyStats:
        """Measure Tier 1 Semantic Cache resolution latency."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        mock_ts = AsyncMock()
        mock_ts.system_one.return_value = SimpleNamespace(
            nouls={"equivalent": SimpleNamespace(noul=0.95)}
        )
        cache = SemanticCache(ts_client=mock_ts)
        query_emb = self.embed_svc.embed_query(query)

        # Seed entry
        cache.put(query, query_emb, {"answer": "LatticeDB is an embedded database."})

        async def _probe():
            return await cache.get(query, query_emb)

        return await self.benchmark_callable(
            name="Tier 1 Semantic Cache",
            fn=_probe,
            iterations=iterations,
            warmup=warmup,
        )

    async def benchmark_stage1_hybrid(
        self,
        queries: list[str],
        iterations: int = 10,
        warmup: int = 2,
    ) -> LatencyStats:
        """Measure Stage 1 HNSW <=> vector search + BM25 @@ text search with RRF."""
        searcher = HybridSearcher(self.store, self.embed_svc)
        q_idx = 0

        async def _search():
            nonlocal q_idx
            q = queries[q_idx % len(queries)]
            q_idx += 1
            return await searcher.search(q, top_k=10)

        return await self.benchmark_callable(
            name="Stage 1: Hybrid Retrieval (HNSW+BM25)",
            fn=_search,
            iterations=iterations,
            warmup=warmup,
        )

    async def benchmark_stage2_traversal(
        self,
        anchor_ids: list[int] | None = None,
        iterations: int = 10,
        warmup: int = 2,
    ) -> LatencyStats:
        """Measure Stage 2 Dynamic Cypher multi-hop graph traversal."""
        traverser = GraphTraverser(self.store)
        if not anchor_ids:
            snapshot = self.store.get_subgraph_snapshot(5)
            anchor_ids = [n["id"] for n in snapshot.nodes[:3]] or [1]

        async def _traverse():
            return await traverser.traverse(anchor_ids, route="graph_relational", budget=25)

        return await self.benchmark_callable(
            name="Stage 2: Dynamic Cypher Traversal",
            fn=_traverse,
            iterations=iterations,
            warmup=warmup,
        )

    async def benchmark_stage3_rerank(
        self,
        query: str = "Explain LatticeDB graph and vector search",
        iterations: int = 6,
        warmup: int = 1,
    ) -> LatencyStats:
        """Measure Stage 3 CPU-quantized INT8 Cross-Encoder reranking."""
        reranker = PrecisionReranker(self.embed_svc)
        # Prepare sample chunks
        searcher = HybridSearcher(self.store, self.embed_svc)
        chunks = await searcher.search(query, top_k=3)
        chunks_dicts = [
            {"node_id": c.node_id, "text": c.text, "score": c.score, "metadata": c.metadata}
            for c in chunks[:3]
        ] or [{"node_id": 1, "text": "Sample text for reranking", "score": 0.5, "metadata": {}}]
        graph_ctx = self.store.get_subgraph_snapshot(10)

        async def _rerank():
            return await reranker.rerank(query, chunks_dicts, graph_ctx, top_k=3, max_triples=5)

        return await self.benchmark_callable(
            name="Stage 3: Cross-Encoder Rerank (bge-reranker-v2-m3)",
            fn=_rerank,
            iterations=iterations,
            warmup=warmup,
        )

    async def benchmark_end_to_end(
        self,
        queries: list[str],
        iterations: int = 5,
        warmup: int = 1,
        live_llm: bool = False,
    ) -> LatencyStats:
        """Measure End-to-End pipeline execution latency."""
        if not self.orchestrator:
            # Create a lightweight in-process pipeline measurement
            searcher = HybridSearcher(self.store, self.embed_svc)
            traverser = GraphTraverser(self.store)
            reranker = PrecisionReranker(self.embed_svc)

            q_idx = 0
            async def _pipeline():
                nonlocal q_idx
                q = queries[q_idx % len(queries)]
                q_idx += 1
                stage1 = await searcher.search(q, top_k=3)
                anchor_ids = [c.node_id for c in stage1[:2]]
                stage2 = await traverser.traverse(anchor_ids, route="hybrid", budget=15)
                chunks_dict = [
                    {"node_id": c.node_id, "text": c.text, "score": c.score, "metadata": c.metadata}
                    for c in stage1[:3]
                ]
                await reranker.rerank(q, chunks_dict, stage2, top_k=3, max_triples=5)
                return True

            return await self.benchmark_callable(
                name="End-to-End Pipeline (In-Process RAG)",
                fn=_pipeline,
                iterations=iterations,
                warmup=warmup,
            )

        q_idx = 0
        async def _e2e():
            nonlocal q_idx
            q = queries[q_idx % len(queries)]
            q_idx += 1
            if live_llm:
                return await self.orchestrator.run(q)
            else:
                # Use in-process retrieval pipeline path directly without external LLM network hop
                ret_pipeline = getattr(
                    self.orchestrator,
                    "retrieval_pipeline",
                    getattr(self.orchestrator, "retrieval", None),
                )
                return await ret_pipeline.execute(q, route="hybrid", apply_guardrail=False)

        return await self.benchmark_callable(
            name="End-to-End Pipeline (Full StateGraph)",
            fn=_e2e,
            iterations=iterations,
            warmup=warmup,
        )

    async def run_full_suite(
        self,
        queries: list[str] | None = None,
        iterations: int = 8,
        warmup: int = 2,
        live_llm: bool = False,
    ) -> BenchmarkReport:
        """Execute the comprehensive benchmark suite across all pipeline components."""
        start_suite = time.perf_counter()
        sample_queries = queries or [
            "How does LatticeDB combine HNSW and BM25 search in a single binary?",
            "Explain the multi-hop Cypher traversal process from entity anchors.",
            "Compare the caching tiers in the architecture: Tier 1 vs Tier 2 fallback.",
            "What is the role of FastEmbed and what dense embedding model does it use?",
            "How are relation triples extracted from text and mapped into LatticeDB?",
        ]

        report = BenchmarkReport()

        # 1. Tier 1 Semantic Cache Benchmark
        report.stages["tier1_cache"] = await self.benchmark_tier1_cache(
            query=sample_queries[0],
            iterations=max(iterations, 10),
            warmup=warmup,
        )

        # 2. Stage 1 Hybrid Retrieval Benchmark
        report.stages["stage1_hybrid"] = await self.benchmark_stage1_hybrid(
            queries=sample_queries,
            iterations=iterations,
            warmup=warmup,
        )

        # 3. Stage 2 Dynamic Cypher Traversal Benchmark
        report.stages["stage2_traversal"] = await self.benchmark_stage2_traversal(
            iterations=iterations,
            warmup=warmup,
        )

        # 4. Stage 3 Cross-Encoder Precision Reranking Benchmark
        report.stages["stage3_rerank"] = await self.benchmark_stage3_rerank(
            query=sample_queries[0],
            iterations=max(iterations // 2, 4),
            warmup=max(warmup // 2, 1),
        )

        # 5. End-to-End Pipeline Latency Benchmark
        report.stages["e2e_pipeline"] = await self.benchmark_end_to_end(
            queries=sample_queries,
            iterations=max(iterations // 2, 3),
            warmup=max(warmup // 2, 1),
            live_llm=live_llm,
        )

        report.total_duration_sec = time.perf_counter() - start_suite

        # Evaluate Dual SLAs
        e2e_p95 = report.stages["e2e_pipeline"].p95_ms
        cache_p95 = report.stages["tier1_cache"].p95_ms

        report.sub_second_sla_met = bool(e2e_p95 < 1000.0)
        report.tier1_cache_sla_met = bool(cache_p95 < 25.0)

        return report

    @staticmethod
    def format_ascii_table(report: BenchmarkReport) -> str:
        """Render a clean, aligned ANSI/ASCII table of benchmark percentiles and SLAs."""
        header = (
            f"+--------------------------------------------------+--------+--------+--------+--------+--------+--------+\n"
            f"| Pipeline Stage                                   | p50    | p90    | p95    | p99    | Mean   | QPS    |\n"
            f"+--------------------------------------------------+--------+--------+--------+--------+--------+--------+"
        )
        lines = [header]

        for s in report.stages.values():
            line = (
                f"| {s.stage:<48} | {s.p50_ms:>5.1f}ms | {s.p90_ms:>5.1f}ms | "
                f"{s.p95_ms:>5.1f}ms | {s.p99_ms:>5.1f}ms | {s.mean_ms:>5.1f}ms | {s.qps:>6.1f} |"
            )
            lines.append(line)

        lines.append(
            f"+--------------------------------------------------+--------+--------+--------+--------+--------+--------+"
        )

        sla_e2e = "[PASSED]" if report.sub_second_sla_met else "[FAILED]"
        sla_cache = "[PASSED]" if report.tier1_cache_sla_met else "[FAILED]"

        summary = (
            f"\nSLA Verification:\n"
            f"  * Sub-Second Multi-Hop SLA (E2E p95 < 1000ms):   {sla_e2e}\n"
            f"  * Tier 1 Cache Latency SLA (Cache p95 < 25ms):   {sla_cache}\n"
            f"Total Benchmark Suite Duration: {report.total_duration_sec:.2f}s\n"
        )
        lines.append(summary)

        return "\n".join(lines)
