from __future__ import annotations

import logging
from typing import Annotated

from litestar import Controller, get
from litestar.di import NamedDependency
from litestar.params import QueryParameter, SkipValidation

from lattice_rag.api.dtos import SubgraphDTO
from lattice_rag.api.guards import require_admin_key
from lattice_rag.storage.db import LatticeStore

logger = logging.getLogger(__name__)


class GraphController(Controller):
    """Litestar controller for inspecting the embedded LatticeDB knowledge graph (admin-only)."""

    path = "/api/v1/graph"
    guards = [require_admin_key]

    @get("/subgraph")
    async def get_subgraph(
        self,
        store: NamedDependency[SkipValidation[LatticeStore]],
        limit: Annotated[int, QueryParameter(ge=1, le=500)] = 50,
    ) -> SubgraphDTO:
        """Return a snapshot of the knowledge graph nodes and edges up to limit."""
        logger.info("fetching_graph_subgraph", extra={"limit": limit})
        subgraph = store.get_subgraph_snapshot(limit=limit)
        return SubgraphDTO.from_subgraph(subgraph)
