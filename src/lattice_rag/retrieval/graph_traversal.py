from __future__ import annotations

import asyncio

import structlog

from lattice_rag.storage.db import LatticeStore, SubgraphResult
from lattice_rag.vocabulary import Route

logger = structlog.get_logger(__name__)

# The Routes the Route ablation (#46, ADR-0003) forces a hop count on
ABLATED_ROUTES: frozenset[Route] = frozenset({"vector_exact", "graph_relational", "hybrid"})


class GraphTraverser:
    """Stage 2: 1-to-2 hop graph traversal over LatticeDB edges (in-process pointer walk)."""

    def __init__(self, store: LatticeStore, forced_hops: int | None = None):
        """Initialize GraphTraverser.

        Args:
            store: The LatticeDB store instance.
            forced_hops: Walk this many hops on the three retrieval Routes (``vector_exact``,
                ``graph_relational``, ``hybrid``) instead of the per-Route count;
                ``massive_context`` keeps its own. Eval-only: the Route ablation (#46,
                ADR-0003) sets it through ``lattice-rag eval --force-hops``; the app never does.
        """
        if forced_hops is not None and forced_hops not in (1, 2):
            raise ValueError(f"forced_hops must be 1 or 2 hops, got {forced_hops}")
        self.store = store
        self.forced_hops = forced_hops

    def _hops_for(self, route: Route) -> int:
        """Per-Route hop count; the Route ablation overrides it except on massive_context."""
        if self.forced_hops is not None and route in ABLATED_ROUTES:
            return self.forced_hops
        return 2 if route == "graph_relational" else 1

    async def traverse(
        self, anchor_node_ids: list[int], route: Route, budget: int = 25
    ) -> SubgraphResult:
        """Traverse the graph starting from entity nodes connected to anchor chunks.

        Args:
            anchor_node_ids: List of chunk node IDs from Stage 1, ordered by rank.
            route: The routing strategy, determines max hops (2 for graph_relational, 1 otherwise).
            budget: Maximum number of nodes/edges to explore.

        Returns:
            A SubgraphResult containing the traversed subgraph nodes and edges.
        """
        if not anchor_node_ids:
            return SubgraphResult(nodes=[], edges=[])

        max_hops = self._hops_for(route)

        # Extract entity IDs from anchor chunks in rank order
        entity_ids = await asyncio.to_thread(
            self.store.get_entities_for_chunks, anchor_node_ids, budget
        )

        if not entity_ids:
            logger.debug("no_entities_found_for_anchors", anchor_ids=anchor_node_ids)
            return SubgraphResult(nodes=[], edges=[])

        # Traverse from entities using the store's built-in traversal method
        subgraph = await asyncio.to_thread(
            self.store.traverse_from_entities,
            entity_ids,
            max_hops=max_hops,
            budget=budget,
        )

        logger.debug(
            "graph_traversal_completed",
            route=route,
            max_hops=max_hops,
            anchor_entities=len(entity_ids),
            traversed_nodes=len(subgraph.nodes),
            traversed_edges=len(subgraph.edges),
        )
        return subgraph
