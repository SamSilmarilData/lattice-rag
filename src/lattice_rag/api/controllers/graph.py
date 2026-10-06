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
        return SubgraphDTO.from_subgraph(subgraph)

