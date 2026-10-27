"""Unit tests for Stage 2 GraphTraverser dynamic 1-to-2 hop traversal."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from lattice_rag.retrieval.graph_traversal import GraphTraverser
from lattice_rag.storage.db import (
    ChunkData,
    EntityData,
    LatticeStore,
    RelationData,
    SubgraphResult,
)


@pytest.mark.asyncio
async def test_traverse_empty_anchors():
    """Verify empty anchor node IDs returns empty SubgraphResult."""
    mock_store = MagicMock()
    traverser = GraphTraverser(mock_store)

    subgraph = await traverser.traverse([], route="hybrid")
    assert subgraph.nodes == []
    assert subgraph.edges == []
    mock_store.get_entities_for_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_traverse_hops_by_route():
    """Every Route walks 1 hop; only a forced hop count (Route ablation) walks 2."""
    mock_store = MagicMock()
    # Chunk 1 contains Entity 100
    mock_store.get_entities_for_chunks.return_value = [100]
    expected_subgraph = SubgraphResult(
        nodes=[{"id": 100, "name": "LatticeDB", "label": "Database"}],
        edges=[],
    )
    mock_store.traverse_from_entities.return_value = expected_subgraph

    # 1. hybrid -> max_hops=1
    res = await GraphTraverser(mock_store).traverse([1], route="hybrid")
    mock_store.traverse_from_entities.assert_called_with([100], max_hops=1, budget=25)
    assert res == expected_subgraph

    # 2. massive_context -> max_hops=1
    await GraphTraverser(mock_store).traverse([1], route="massive_context")
    mock_store.traverse_from_entities.assert_called_with([100], max_hops=1, budget=25)

    # 3. hybrid with the ablation's forced 2 hops -> max_hops=2
    await GraphTraverser(mock_store, forced_hops=2).traverse([1], route="hybrid")
    mock_store.traverse_from_entities.assert_called_with([100], max_hops=2, budget=25)


@pytest.mark.asyncio
async def test_traverse_passes_ranked_anchors_and_budget_to_store():
    """Anchor chunks are handed to the store in rank order with the traversal budget."""
    mock_store = MagicMock()
    mock_store.get_entities_for_chunks.return_value = [1, 2, 3]
    mock_store.traverse_from_entities.return_value = SubgraphResult(nodes=[], edges=[])

    traverser = GraphTraverser(mock_store)
    await traverser.traverse([1, 2], route="hybrid", budget=3)

    mock_store.get_entities_for_chunks.assert_called_once_with([1, 2], 3)
    mock_store.traverse_from_entities.assert_called_once_with([1, 2, 3], max_hops=1, budget=3)


# ── Budget is a hard cap on the whole traversal (real store) ──────────────────


def _rel(src: str, dst: str, kind: str = "LINKS") -> RelationData:
    return RelationData(source_name=src, target_name=dst, relation_type=kind)


@pytest.fixture
def fan_out_store(tmp_path: Path):
    """One chunk mentioning A, B, C; A and B fan out, and every neighbour links on."""
    store = LatticeStore(db_path=tmp_path / "budget.db", vector_dimensions=384)
    names = ["A", "B", "C", "X1", "X2", "X3", "Y1", "Y2", "Z1", "Z2", "Z3"]
    relations = [
        _rel("A", "X1"),
        _rel("A", "X2"),
        _rel("A", "X3"),
        _rel("B", "Y1"),
        _rel("B", "Y2"),
        _rel("C", "A"),
        _rel("X1", "Z1"),
        _rel("X2", "Z2"),
        _rel("X3", "Z3"),
        _rel("Y1", "Z1"),
    ]
    vec = np.zeros(384, dtype=np.float32)
    stats = store.ingest_document_bundle(
        "doc_budget",
        "Budget",
        [
            ChunkData(text="A, B and C.", embedding=vec, position=0),
            ChunkData(text="How everything links.", embedding=vec, position=1),
        ],
        [
            (0, [EntityData(name=n, entity_type="x") for n in ["A", "B", "C"]], []),
            (1, [EntityData(name=n, entity_type="x") for n in names], relations),
        ],
    )
    yield store, stats.chunk_ids[0]
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("hops", [1, 2])
@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 7, 9])
async def test_traversal_never_exceeds_budget(fan_out_store, hops: int, budget: int):
    """Neither nodes nor edges exceed the budget at any hop count, and no edge dangles."""
    store, chunk_id = fan_out_store
    traverser = GraphTraverser(store, forced_hops=hops)

    subgraph = await traverser.traverse([chunk_id], route="hybrid", budget=budget)

    assert len(subgraph.nodes) <= budget
    assert len(subgraph.edges) <= budget
    node_ids = {n["id"] for n in subgraph.nodes}
    for edge in subgraph.edges:
        assert edge["source_id"] in node_ids
        assert edge["target_id"] in node_ids


@pytest.mark.asyncio
async def test_traversal_within_budget_returns_full_two_hop_graph(fan_out_store):
    """With a generous budget the 2-hop walk is not truncated."""
    store, chunk_id = fan_out_store
    traverser = GraphTraverser(store, forced_hops=2)

    subgraph = await traverser.traverse([chunk_id], route="hybrid", budget=50)

    assert {n["name"] for n in subgraph.nodes} == {
        "A", "B", "C", "X1", "X2", "X3", "Y1", "Y2", "Z1", "Z2", "Z3"
    }  # fmt: skip
    assert len(subgraph.edges) == 10
