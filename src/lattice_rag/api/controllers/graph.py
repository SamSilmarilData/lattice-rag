from __future__ import annotations

import logging
from litestar import Controller, get
from litestar.di import NamedDependency
from litestar.params import FromQuery, SkipValidation

from lattice_rag.api.dtos import GraphEdge, GraphNode, SubgraphDTO
from lattice_rag.storage.db import LatticeStore

logger = logging.getLogger(__name__)


class GraphController(Controller):
    """Litestar controller for inspecting the embedded LatticeDB knowledge graph."""

    path = "/api/v1/graph"

    @get("/subgraph")
    async def get_subgraph(
        self,
        store: NamedDependency[SkipValidation[LatticeStore]],
        limit: FromQuery[int] = 50,
    ) -> SubgraphDTO:
        """Return a snapshot of the knowledge graph nodes and edges up to limit."""
        logger.info("fetching_graph_subgraph", extra={"limit": limit})
        subgraph = store.get_subgraph_snapshot(limit=limit)
        return SubgraphDTO(
            nodes=[
                GraphNode(
                    node_id=n["id"],
                    label=n.get("label", "Entity"),
                    name=n.get("name", f"node_{n['id']}"),
                )
                for n in subgraph.nodes
            ],
            edges=[
                GraphEdge(
                    source_id=e["source_id"],
                    target_id=e["target_id"],
                    relation_type=e["relation_type"],
                )
                for e in subgraph.edges
            ],
        )
