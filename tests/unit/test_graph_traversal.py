"""Unit tests for Stage 2 GraphTraverser dynamic 1-to-2 hop traversal."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lattice_rag.retrieval.graph_traversal import GraphTraverser
from lattice_rag.storage.db import SubgraphResult


@pytest.mark.asyncio
async def test_traverse_empty_anchors():
    """Verify empty anchor node IDs returns empty SubgraphResult."""
    mock_store = MagicMock()
    traverser = GraphTraverser(mock_store)

    subgraph = await traverser.traverse([], route="hybrid")
    assert subgraph.nodes == []
    assert subgraph.edges == []
    mock_store.get_entities_for_chunk.assert_not_called()


@pytest.mark.asyncio
async def test_traverse_hops_by_route():
    """Verify graph_relational expands 2 hops while other routes expand 1 hop."""
    mock_store = MagicMock()
    # Chunk 1 contains Entity 100
    mock_store.get_entities_for_chunk.return_value = [100]
    expected_subgraph = SubgraphResult(
        nodes=[{"id": 100, "name": "LatticeDB", "label": "Database"}],
        edges=[],
    )
    mock_store.traverse_from_entities.return_value = expected_subgraph

    traverser = GraphTraverser(mock_store)

    # 1. graph_relational -> max_hops=2
    res_rel = await traverser.traverse([1], route="graph_relational")
    mock_store.traverse_from_entities.assert_called_with([100], max_hops=2, budget=25)
    assert res_rel == expected_subgraph

    # 2. hybrid -> max_hops=1
    await traverser.traverse([1], route="hybrid")
    mock_store.traverse_from_entities.assert_called_with([100], max_hops=1, budget=25)


@pytest.mark.asyncio
async def test_traverse_rank_ordered_entity_collection_and_budget():
    """Verify entities are gathered chunk-by-chunk in rank order and respect budget cap."""
    mock_store = MagicMock()
    # Chunk 1 (rank 0) -> Entities 1, 2
    # Chunk 2 (rank 1) -> Entities 2, 3, 4
    def fake_get_entities(chunk_id: int) -> list[int]:
        if chunk_id == 1:
            return [1, 2]
        elif chunk_id == 2:
            return [2, 3, 4]
        return []

    mock_store.get_entities_for_chunk.side_effect = fake_get_entities
    mock_store.traverse_from_entities.return_value = SubgraphResult(nodes=[], edges=[])

    traverser = GraphTraverser(mock_store)

    # With budget=3: Chunk 1 gives [1, 2], Chunk 2 adds [3], reaching budget 3
    await traverser.traverse([1, 2], route="hybrid", budget=3)
    mock_store.traverse_from_entities.assert_called_once_with([1, 2, 3], max_hops=1, budget=3)
