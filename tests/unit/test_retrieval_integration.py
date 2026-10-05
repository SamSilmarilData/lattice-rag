"""Live integration tests for the full 3-Stage Hybrid Retrieval Pipeline + Context Guardrail.

Verifies end-to-end execution on real in-process LatticeDB, real FastEmbed BGE embeddings,
real INT8 ONNX bge-reranker-v2-m3, and live TypeSafe AI Jev Noul guardrails.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pytest


from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.pipeline import RetrievalPipeline
from lattice_rag.routing.guardrail import ContextGuardrail
from lattice_rag.storage.db import (
    ChunkData,
    EntityData,
    LatticeStore,
    RelationData,
)


@pytest.fixture
def temp_store(tmp_path: Path) -> LatticeStore:
    db_file = tmp_path / "integration_test.db"
    store = LatticeStore(db_file)
    yield store
    store.close()


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    # Use real FastEmbed embedder and INT8 ONNX reranker
    svc = EmbeddingService(
        embed_model="BAAI/bge-small-en-v1.5",
        reranker_model="BAAI/bge-reranker-v2-m3",
    )
    # Warmup models so session loading does not skew steady-state query latency
    _ = svc.embed_query("warmup")
    _ = svc.rerank("warmup", ["warmup document"])
    return svc



@pytest.mark.asyncio
async def test_full_pipeline_live_integration(
    temp_store: LatticeStore,
    embedding_service: EmbeddingService,
):
    """Verify live end-to-end retrieval with real models, graph traversal, and Jev guardrails."""
    # 1. Ingest Technical Knowledge into LatticeDB
    c1_text = "LatticeDB is an embedded property-graph database with native HNSW vector search."
    c2_text = "FastEmbed provides local quantized CPU text embeddings without GPU."
    c3_text = "Carrots and potatoes grow well in loose, sandy loam garden soil."

    # Compute real embeddings
    embeddings = embedding_service.embed_texts([c1_text, c2_text, c3_text])

    chunks = [
        ChunkData(text=c1_text, embedding=embeddings[0], position=0),
        ChunkData(text=c2_text, embedding=embeddings[1], position=1),
        ChunkData(text=c3_text, embedding=embeddings[2], position=2),
    ]
    temp_store.ingest_document("doc_tech", "AI Infrastructure", chunks[:2])
    temp_store.ingest_document("doc_farm", "Agriculture", [chunks[2]])

    # Link entities and relationships
    temp_store.ingest_entities(
        chunk_id=2,  # Chunk 0 node_id
        entities=[
            EntityData(name="LatticeDB", entity_type="Database", embedding=embeddings[0]),
            EntityData(name="HNSW", entity_type="Index", embedding=embeddings[0]),
        ],
        relations=[
            RelationData(source_name="LatticeDB", target_name="HNSW", relation_type="USES_INDEX"),
        ],
    )

    # 2. Setup Guardrail (uses live TYPESAFE_API_KEY from .env)
    guardrail = ContextGuardrail(threshold=0.50)

    # 3. Assemble and execute RetrievalPipeline
    pipeline = RetrievalPipeline(
        store=temp_store,
        embedding_service=embedding_service,
        router_guardrail=guardrail,
    )

    query = "How does LatticeDB index vectors?"
    result = await pipeline.execute(query, route="hybrid")

    # 4. Verify Results
    assert len(result.final_chunks) >= 1
    top_chunk = result.final_chunks[0]
    assert "LatticeDB" in top_chunk["text"]
    assert top_chunk.get("rerank_score") is not None
    assert top_chunk.get("rerank_score") > 0.0  # Positive logit from bge-reranker-v2-m3
    assert result.low_grounding is False

    # Verify Graph Traversal Context
    assert len(result.graph_context.nodes) >= 1
    node_names = [n["name"] for n in result.graph_context.nodes]
    assert "LatticeDB" in node_names or "HNSW" in node_names

    # Verify Timings (Total under 1,000ms)
    total_latency_ms = sum(result.timings.values())
    print(f"\nLive Pipeline Execution Timings:")
    for stage, dur in result.timings.items():
        print(f"  - {stage}: {dur:.1f}ms")
    print(f"Total Pipeline Latency: {total_latency_ms:.1f}ms")
    assert total_latency_ms < 1000.0, f"Pipeline exceeded 1s budget: {total_latency_ms}ms"

    # Cleanup guardrail client
    await guardrail.aclose()


@pytest.mark.asyncio
async def test_full_pipeline_irrelevant_query_low_grounding(
    temp_store: LatticeStore,
    embedding_service: EmbeddingService,
):
    """Verify irrelevant query triggers low_grounding_flag through live Jev Noul guardrail."""
    c_text = "LatticeDB uses ACID transactions and writes require an explicit txn.commit() call."
    emb = embedding_service.embed_query(c_text)
    temp_store.ingest_document("doc_acid", "ACID", [ChunkData(text=c_text, embedding=emb, position=0)])

    guardrail = ContextGuardrail(threshold=0.50)
    pipeline = RetrievalPipeline(
        store=temp_store,
        embedding_service=embedding_service,
        router_guardrail=guardrail,
    )

    # Query completely unrelated to database transactions
    irrelevant_query = "What is the capital city of Australia and what is its population?"
    result = await pipeline.execute(irrelevant_query, route="hybrid")

    # Either Stage 1 pruned it or Guardrail marked low grounding
    assert result.low_grounding is True

    await guardrail.aclose()
