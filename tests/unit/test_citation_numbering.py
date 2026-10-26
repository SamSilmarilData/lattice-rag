"""[Chunk N] numbers from 1 in the generation prompt and in the Live Tier judge context."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lattice_rag.eval.runner import EvalRunner
from lattice_rag.eval.triage_gate import JevEvaluator
from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.routing.router import RouteDecision
from lattice_rag.storage.db import SubgraphResult

pytestmark = pytest.mark.usefixtures("small_gold_sets")

METRIC_NAMES = ("faithfulness", "context_precision", "answer_relevance", "answer_correctness")


def _write_gold_set(tmp_path: Path) -> tuple[Path, Path]:
    dataset = {
        "version": "3.0.0",
        "domain": "test",
        "corpus": [],
        "queries": [
            {
                "id": "q1",
                "query": "What is first?",
                "slice": "unanswerable",
                "case": "direct",
                "expected_route": "hybrid",
                "reference_answer": "The first.",
                "gold_passages": [],
            }
        ],
    }
    baseline = {
        "version": "3.0.0",
        "metrics": {"faithfulness": 0.5, "context_precision": 0.5, "answer_relevance": 0.5},
        "gate_thresholds": {
            "max_regression_delta": 0.06,
            "min_query_floor": 0.5,
            "pass_thresholds": dict.fromkeys(METRIC_NAMES, 0.5),
        },
    }
    ds, base = tmp_path / "ds.json", tmp_path / "base.json"
    ds.write_text(json.dumps(dataset), encoding="utf-8")
    base.write_text(json.dumps(baseline), encoding="utf-8")
    return ds, base


@pytest.mark.asyncio
async def test_prompt_and_judge_number_chunks_from_one_identically(tmp_path: Path):
    ds_path, base_path = _write_gold_set(tmp_path)
    chunks = [
        {"node_id": 1, "text": "FIRST-CHUNK-TEXT", "score": 0.9},
        {"node_id": 2, "text": "SECOND-CHUNK-TEXT", "score": 0.8},
    ]

    router = MagicMock()
    router.route_query = AsyncMock(return_value=RouteDecision(route="hybrid", confidence=0.9))
    pipeline = MagicMock()
    pipeline.execute = AsyncMock(
        return_value=MagicMock(
            final_chunks=chunks,
            graph_context=SubgraphResult(nodes=[], edges=[]),
            timings={},
            low_grounding=False,
            unverified=False,
        )
    )
    embed_svc = MagicMock()
    embed_svc.embed_query = MagicMock(return_value=np.ones(384, dtype=np.float32))

    groq_client = MagicMock()
    groq_resp = MagicMock()
    groq_resp.choices = [MagicMock()]
    groq_resp.choices[0].message.content = "Answer [Chunk 1]."
    groq_client.chat.completions.create = AsyncMock(return_value=groq_resp)

    orchestrator = RAGOrchestrator(
        router=router,
        semantic_cache=MagicMock(get=AsyncMock(return_value=None), put=MagicMock()),
        fallback_cache=MagicMock(find_closest_fallback=AsyncMock(return_value=None)),
        retrieval_pipeline=pipeline,
        groq=GroqSynthesizer(client=groq_client, model="m"),
        gemini=MagicMock(),
        chitchat=MagicMock(),
        embedding_service=embed_svc,
    )

    judge_client = AsyncMock()
    judge_resp = MagicMock()
    judge_resp.scores = {m: MagicMock(score=4.0) for m in METRIC_NAMES}
    judge_resp.nouls = {"contradiction": MagicMock(noul=0.0)}
    judge_client.system_one.return_value = judge_resp

    runner = EvalRunner(orchestrator=orchestrator, evaluator=JevEvaluator(client=judge_client))
    await runner.run_evaluation(dataset_path=ds_path, baseline_path=base_path)

    prompt = groq_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    judge_context = judge_client.system_one.call_args.kwargs["state"]["context"]

    assert '<chunk index="1" doc="unknown">\nFIRST-CHUNK-TEXT' in prompt
    assert '<chunk index="2" doc="unknown">\nSECOND-CHUNK-TEXT' in prompt
    assert 'index="0"' not in prompt
    assert "[Chunk 1] FIRST-CHUNK-TEXT" in judge_context
    assert "[Chunk 2] SECOND-CHUNK-TEXT" in judge_context
