from __future__ import annotations

import asyncio
from typing import Any, List

import structlog

from lattice_rag.storage.db import SubgraphResult

logger = structlog.get_logger(__name__)


class GraphTraverser:
    """Stage 2: Dynamic 1-to-2 hop Cypher / pointer graph traversal."""

    def __init__(self, store: Any):
        """Initialize GraphTraverser.

        Args:
            store: The LatticeDB store instance.
        """
        self.store = store

    async def traverse(
        self, anchor_node_ids: list[int], route: str, budget: int = 25
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

        # Determine max_hops based on route
        max_hops = 2 if route == "graph_relational" else 1

        # Extract entity IDs from anchor chunks in rank order
        if hasattr(type(self.store), "get_entities_for_chunks"):
            entity_ids = await asyncio.to_thread(
                self.store.get_entities_for_chunks, anchor_node_ids, budget
            )
        else:
            entity_ids = []
            seen_eids: set[int] = set()
            for chunk_id in anchor_node_ids:
                c_entities = await asyncio.to_thread(self.store.get_entities_for_chunk, chunk_id)
                for eid in c_entities:
                    if eid not in seen_eids:
                        seen_eids.add(eid)
                        entity_ids.append(eid)
                    if len(entity_ids) >= budget:
                        break
                if len(entity_ids) >= budget:
                    break

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
