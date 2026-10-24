"""Canary for the Hermetic Tier gate: it must fail when retrieval is deliberately broken.

Runs the shipped Gold Set and ``hermetic_thresholds.json`` through the real local models,
so it needs no API keys but does need the ONNX models (downloaded on first run, cached in
CI). It is not part of ``tests/unit``; CI runs it in the Hermetic Tier job right after the
gate itself, proving the floors are tight enough to catch a broken retriever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lattice_rag.config import load_config
from lattice_rag.eval.gold_set import load_gold_set
from lattice_rag.eval.hermetic import load_thresholds, run_hermetic_tier
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.pipeline import RetrievalPipeline, RetrievalResult

REPO_ROOT = Path(__file__).resolve().parents[2]


class ReversedRetriever:
    """The real pipeline with its ranking inverted: the least relevant Chunks come first."""

    def __init__(self, store: Any, embedding_service: EmbeddingService) -> None:
        self.pipeline = RetrievalPipeline(store, embedding_service)

    async def retrieve(
        self, query: str, route: str = "hybrid", apply_guardrail: bool = True, top_k: int = 5
    ) -> RetrievalResult:
        everything = await self.pipeline.retrieve(query, route, apply_guardrail=False, top_k=100)
        everything.final_chunks = list(reversed(everything.final_chunks))[:top_k]
        return everything


async def test_gate_fails_on_a_deliberately_broken_retriever() -> None:
    config = load_config(require_typesafe=False)
    embed_svc = EmbeddingService(
        config.embed_model,
        config.reranker_model,
        onnx_reranker_repo=config.reranker_onnx_repo,
        onnx_reranker_revision=config.reranker_onnx_revision,
    )

    report = await run_hermetic_tier(
        load_gold_set(REPO_ROOT / "eval_dataset.json"),
        load_thresholds(REPO_ROOT / "hermetic_thresholds.json"),
        embed_svc,
        retriever_factory=ReversedRetriever,
    )

    assert not report.passed
    assert any(f.startswith("overall:") for f in report.failures), report.failures
