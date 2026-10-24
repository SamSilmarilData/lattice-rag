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
    mock_store.get_entities_for_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_traverse_hops_by_route():
    """Verify graph_relational expands 2 hops while other routes expand 1 hop."""
    mock_store = MagicMock()
    # Chunk 1 contains Entity 100
    mock_store.get_entities_for_chunks.return_value = [100]
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
async def test_traverse_passes_ranked_anchors_and_budget_to_store():
    """Anchor chunks are handed to the store in rank order with the traversal budget."""
    mock_store = MagicMock()
    mock_store.get_entities_for_chunks.return_value = [1, 2, 3]
    mock_store.traverse_from_entities.return_value = SubgraphResult(nodes=[], edges=[])

    traverser = GraphTraverser(mock_store)
    await traverser.traverse([1, 2], route="hybrid", budget=3)

    mock_store.get_entities_for_chunks.assert_called_once_with([1, 2], 3)
    mock_store.traverse_from_entities.assert_called_once_with([1, 2, 3], max_hops=1, budget=3)
