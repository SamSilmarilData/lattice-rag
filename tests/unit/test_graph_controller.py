"""Unit tests for GraphController and LatticeStore.get_subgraph_snapshot."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.testing import AsyncTestClient

from lattice_rag.api.controllers.graph import GraphController
from lattice_rag.storage.db import (
    ChunkData,
    EntityData,
    LatticeStore,
    RelationData,
    SubgraphResult,
)


@pytest.fixture
def mock_graph_app():
    mock_store = MagicMock()
    mock_store.get_subgraph_snapshot.return_value = SubgraphResult(
        nodes=[
            {"id": 1, "name": "LatticeDB", "label": "Technology"},
            {"id": 2, "name": "FastEmbed", "label": "Technology"},
        ],
        edges=[
            {"source_id": 1, "target_id": 2, "relation_type": "INTEGRATES_WITH"},
        ],
    )

    app = Litestar(
        route_handlers=[GraphController],
        dependencies={"store": Provide(lambda: mock_store, sync_to_thread=False)},
    )
    return app, mock_store


@pytest.mark.asyncio
async def test_get_subgraph_endpoint(mock_graph_app):
    app, mock_store = mock_graph_app
    async with AsyncTestClient(app) as client:
        response = await client.get("/api/v1/graph/subgraph?limit=25")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2
        assert data["nodes"][0]["name"] == "LatticeDB"
        assert data["nodes"][0]["label"] == "Technology"
        assert len(data["edges"]) == 1
        assert data["edges"][0]["relationType"] == "INTEGRATES_WITH"
        mock_store.get_subgraph_snapshot.assert_called_once_with(limit=25)


def test_lattice_store_get_subgraph_snapshot():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_snapshot.db"
        store = LatticeStore(db_path)

        # Ingest a document and some entities with relations
        chunks = [
            ChunkData(text="LatticeDB is an embedded database.", embedding=np.zeros(384, dtype=np.float32), position=0)
        ]
        stats = store.ingest_document("doc1", "Doc 1", chunks)
        chunk_id = stats.chunk_ids[0]

        entities = [
            EntityData(name="LatticeDB", entity_type="Database", embedding=np.zeros(384, dtype=np.float32)),
            EntityData(name="FastEmbed", entity_type="Library", embedding=np.zeros(384, dtype=np.float32)),
        ]
        relations = [
            RelationData(source_name="LatticeDB", target_name="FastEmbed", relation_type="USES"),
        ]
        store.ingest_entities(chunk_id, entities, relations)

        # Query snapshot
        snapshot = store.get_subgraph_snapshot(limit=10)
        assert len(snapshot.nodes) >= 2
        node_names = {n["name"] for n in snapshot.nodes}
        assert "LatticeDB" in node_names
        assert "FastEmbed" in node_names
        assert len(snapshot.edges) >= 1
        assert snapshot.edges[0]["relation_type"] == "USES"

        store.close()
