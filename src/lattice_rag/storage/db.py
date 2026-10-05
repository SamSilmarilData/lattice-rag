"""LatticeDB in-process property-graph and vector storage engine."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
from latticedb import Database

logger = logging.getLogger(__name__)


class ChunkData(msgspec.Struct, rename="camel"):
    """Represents a text chunk to be stored in LatticeDB with its embedding."""
    text: str
    embedding: np.ndarray
    position: int
    doc_id: str = ""


class EntityData(msgspec.Struct, rename="camel"):
    """Represents an extracted entity node."""
    name: str
    entity_type: str
    embedding: np.ndarray | None = None


class RelationData(msgspec.Struct, rename="camel"):
    """Represents a directional relationship between two entities."""
    source_name: str
    target_name: str
    relation_type: str


class SearchResult(msgspec.Struct, rename="camel"):
    """Represents a search result from vector or BM25 search."""
    node_id: int
    score: float
    text: str
    metadata: dict[str, Any] = {}


class SubgraphResult(msgspec.Struct, rename="camel"):
    """Represents a subgraph of traversed nodes and edges."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []


class IngestStats(msgspec.Struct, rename="camel"):
    """Statistics returned after document ingestion."""
    chunk_count: int
    entity_count: int
    relation_count: int
    chunk_ids: list[int] = []


class LatticeStore:
    """Embedded property-graph, HNSW vector, and BM25 full-text storage engine."""

    def __init__(
        self,
        db_path: Path | str = "data/lattice_rag.db",
        vector_dimensions: int = 384,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_dimensions = vector_dimensions

        # Open embedded database with vector support and WAL enabled
        self.db = Database(
            str(self.db_path),
            create=True,
            enable_vectors=True,
            vector_dimensions=self.vector_dimensions,
        )
        self.db.open()
        self._ensure_schema()
        logger.info("Initialized LatticeStore at %s (dim=%d)", self.db_path, self.vector_dimensions)

    def _ensure_schema(self) -> None:
        """Create full-text search indices if not already present."""
        try:
            if not self.db.has_node_fts_index("Chunk", "text"):
                self.db.create_node_fts_index("Chunk", "text")
        except Exception as e:
            logger.warning("Could not create Chunk text index: %s", e)

        try:
            if not self.db.has_node_fts_index("Entity", "name"):
                self.db.create_node_fts_index("Entity", "name")
        except Exception as e:
            logger.warning("Could not create Entity name index: %s", e)

    def ingest_document(
        self,
        doc_id: str,
        title: str,
        chunks: list[ChunkData],
    ) -> IngestStats:
        """Ingest a document with its text chunks and vector embeddings."""
        chunk_ids: list[int] = []
        with self.db.write() as txn:
            doc_node = txn.create_node(
                labels=["Document"],
                properties={"doc_id": doc_id, "title": title},
            )

            for chunk in chunks:
                chunk_node = txn.create_node(
                    labels=["Chunk"],
                    properties={
                        "text": chunk.text,
                        "position": chunk.position,
                        "doc_id": doc_id,
                    },
                )
                if chunk.embedding is not None and len(chunk.embedding) == self.vector_dimensions:
                    txn.set_vector(chunk_node.id, "embedding", chunk.embedding)

                txn.create_edge(doc_node.id, chunk_node.id, "HAS_CHUNK")
                chunk_ids.append(chunk_node.id)

            txn.commit()

        return IngestStats(chunk_count=len(chunks), entity_count=0, relation_count=0, chunk_ids=chunk_ids)

    def ingest_entities(
        self,
        chunk_id: int,
        entities: list[EntityData],
        relations: list[RelationData],
    ) -> IngestStats:
        """Ingest extracted entities and relations linked to their source chunk."""
        entity_name_to_id: dict[str, int] = {}

        with self.db.write() as txn:
            for ent in entities:
                ent_node = txn.create_node(
                    labels=["Entity", ent.entity_type],
                    properties={"name": ent.name, "entity_type": ent.entity_type},
                )
                if ent.embedding is not None and len(ent.embedding) == self.vector_dimensions:
                    txn.set_vector(ent_node.id, "embedding", ent.embedding)

                txn.create_edge(chunk_id, ent_node.id, "CONTAINS")
                entity_name_to_id[ent.name] = ent_node.id

            created_relations = 0
            for rel in relations:
                source_id = entity_name_to_id.get(rel.source_name)
                target_id = entity_name_to_id.get(rel.target_name)
                if source_id is not None and target_id is not None:
                    txn.create_edge(
                        source_id,
                        target_id,
                        "RELATION",
                        properties={"type": rel.relation_type},
                    )
                    created_relations += 1

            txn.commit()

        return IngestStats(
            chunk_count=0,
            entity_count=len(entities),
            relation_count=created_relations,
        )

    def vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Perform native HNSW vector similarity search on Chunk embeddings."""
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        elif query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        raw_results = self.db.vector_search(query_embedding, k=top_k)

        search_results: list[SearchResult] = []
        with self.db.read() as txn:
            for r in raw_results:
                text = txn.get_property(r.node_id, "text") or ""
                pos = txn.get_property(r.node_id, "position")
                doc_id = txn.get_property(r.node_id, "doc_id") or ""
                metadata = {}
                if pos is not None:
                    metadata["position"] = pos
                if doc_id:
                    metadata["doc_id"] = doc_id

                # LatticeDB returns distance (lower is closer); score = 1 / (1 + distance)
                similarity_score = 1.0 / (1.0 + float(r.distance))
                search_results.append(
                    SearchResult(
                        node_id=r.node_id,
                        score=similarity_score,
                        text=str(text),
                        metadata=metadata,
                    )
                )

        return search_results

    def bm25_search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Perform native BM25 full-text search on Chunk text."""
        try:
            raw_results = self.db.fts_search("Chunk", "text", query_text, limit=top_k)
        except Exception as e:
            logger.debug("BM25 search error or no index: %s", e)
            return []

        search_results: list[SearchResult] = []
        with self.db.read() as txn:
            for r in raw_results:
                text = txn.get_property(r.node_id, "text") or ""
                pos = txn.get_property(r.node_id, "position")
                doc_id = txn.get_property(r.node_id, "doc_id") or ""
                metadata = {}
                if pos is not None:
                    metadata["position"] = pos
                if doc_id:
                    metadata["doc_id"] = doc_id

                search_results.append(
                    SearchResult(
                        node_id=r.node_id,
                        score=float(r.score),
                        text=str(text),
                        metadata=metadata,
                    )
                )

        return search_results

    def traverse_from_entities(
        self,
        entity_ids: list[int],
        max_hops: int = 1,
        budget: int = 25,
    ) -> SubgraphResult:
        """Traverse connected relations expanding from entity anchor nodes up to max_hops."""
        if not entity_ids:
            return SubgraphResult(nodes=[], edges=[])

        visited_nodes: dict[int, dict[str, Any]] = {}
        visited_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[int, int, str]] = set()

        with self.db.read() as txn:
            frontier = list(entity_ids)
            for hop in range(max_hops):
                if not frontier or len(visited_nodes) >= budget:
                    break

                next_frontier: list[int] = []
                for curr_id in frontier:
                    if curr_id not in visited_nodes:
                        node_name = txn.get_property(curr_id, "name") or f"node_{curr_id}"
                        node_type = txn.get_property(curr_id, "entity_type") or "Entity"
                        visited_nodes[curr_id] = {
                            "id": curr_id,
                            "name": str(node_name),
                            "label": str(node_type),
                        }

                    outgoing = txn.get_outgoing_edges(curr_id)
                    for edge in outgoing:
                        if len(visited_edges) >= budget:
                            break

                        rel_type = str(txn.get_edge_property(edge.id, "type") or edge.edge_type)
                        edge_key = (edge.source_id, edge.target_id, rel_type)
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            visited_edges.append({
                                "source_id": edge.source_id,
                                "target_id": edge.target_id,
                                "relation_type": rel_type,
                            })

                        if edge.target_id not in visited_nodes and len(visited_nodes) < budget:
                            target_name = txn.get_property(edge.target_id, "name") or f"node_{edge.target_id}"
                            target_type = txn.get_property(edge.target_id, "entity_type") or "Entity"
                            visited_nodes[edge.target_id] = {
                                "id": edge.target_id,
                                "name": str(target_name),
                                "label": str(target_type),
                            }
                            next_frontier.append(edge.target_id)

                frontier = next_frontier

        return SubgraphResult(
            nodes=list(visited_nodes.values()),
            edges=visited_edges,
        )

    def get_subgraph_snapshot(self, limit: int = 50) -> SubgraphResult:
        """Return a snapshot of the knowledge graph up to limit entities and their relations."""
        try:
            entity_ids = self.db.get_nodes_by_label("Entity")
        except Exception as e:
            logger.debug("Failed to get Entity nodes by label: %s", e)
            return SubgraphResult(nodes=[], edges=[])

        if not entity_ids:
            return SubgraphResult(nodes=[], edges=[])

        return self.traverse_from_entities(entity_ids[:limit], max_hops=1, budget=limit)

    def get_entities_for_chunk(self, chunk_id: int) -> list[int]:
        """Retrieve all entity IDs linked to a given Chunk via CONTAINS edge."""
        entity_ids: list[int] = []
        with self.db.read() as txn:
            edges = txn.get_outgoing_edges_by_type(chunk_id, "CONTAINS")
            for e in edges:
                entity_ids.append(e.target_id)
        return entity_ids

    def close(self) -> None:
        """Close the database connection safely."""
        if hasattr(self, "db") and self.db is not None and self.db.is_open:
            self.db.close()
