"""LatticeDB in-process property-graph and vector storage engine."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
from latticedb import Database, LatticeAlreadyExistsError
from latticedb.transaction import Transaction

logger = logging.getLogger(__name__)

# Upper bound on Document nodes sharing one doc_id (databases from before the duplicate check)
_MAX_DOCUMENT_NODES = 64
# LatticeDB's default HNSW ef_search; raised to k when over-fetching so k results are reachable
_DEFAULT_EF_SEARCH = 64
# Upper bound on one vector_search fetch while widening past legacy Entity vectors
_MAX_VECTOR_FETCH = 4096


class ChunkData(msgspec.Struct, rename="camel"):
    """Represents a text chunk to be stored in LatticeDB with its embedding."""

    text: str
    embedding: np.ndarray
    position: int
    doc_id: str = ""


class EntityData(msgspec.Struct, rename="camel"):
    """Represents an extracted entity node. Entities carry no vector; only Chunks are embedded."""

    name: str
    entity_type: str


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
        self._entity_cache: dict[str, int] = {}

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
        """Create full-text search indices if not already present and preload entity cache."""
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

        # Equality index for duplicate checks and delete; building it backfills existing nodes
        with contextlib.suppress(LatticeAlreadyExistsError):
            self.db.create_node_property_index("Document", "doc_id")

        self._load_entity_cache()

    def _load_entity_cache(self) -> None:
        """Preload canonical entity name -> node_id mapping for O(1) deduplication."""
        try:
            entity_ids = self.db.get_nodes_by_label("Entity")
            if not entity_ids:
                return
            with self.db.read() as txn:
                for nid in entity_ids:
                    name = txn.get_property(nid, "name")
                    if name:
                        self._entity_cache[str(name).strip().lower()] = nid
        except Exception as e:
            logger.debug("Could not preload entity cache: %s", e)

    def get_canonical_entity_id(self, name: str) -> int | None:
        """Return canonical node_id for an entity name if it exists in the graph."""
        return self._entity_cache.get(name.strip().lower())

    @staticmethod
    def _find_document_nodes(txn: Transaction, doc_id: str) -> list[int]:
        """Return the Document node ids for ``doc_id`` via the equality index."""
        return txn.find_nodes_by_label_property(
            "Document", "doc_id", doc_id, limit=_MAX_DOCUMENT_NODES
        )

    def has_document(self, doc_id: str) -> bool:
        """Return True if a Document node with this doc_id is already stored."""
        with self.db.read() as txn:
            return bool(self._find_document_nodes(txn, doc_id))

    @staticmethod
    def _delete_node_with_edges(txn: Transaction, node_id: int) -> None:
        # LatticeDB's delete_node leaves the node's edges dangling, so remove them first
        for edge in txn.get_outgoing_edges(node_id) + txn.get_incoming_edges(node_id):
            txn.delete_edge(edge.source_id, edge.target_id, edge.edge_type)
        txn.delete_node(node_id)

    def delete_document_in_txn(self, txn: Transaction, doc_id: str) -> set[int] | None:
        """Delete ``doc_id`` inside an open write transaction; the caller commits.

        Removes the Document, its Chunks, their edges, and every Entity no remaining
        Chunk references. Returns the removed entity ids, which the caller must pass to
        ``evict_entities`` after commit, or None (deleting nothing) when the document is
        not stored. Splitting it this way lets a replace run delete and insert in one
        transaction.
        """
        doc_nodes = self._find_document_nodes(txn, doc_id)
        if not doc_nodes:
            return None

        candidate_entities: set[int] = set()
        for doc_node in doc_nodes:
            chunk_ids = [e.target_id for e in txn.get_outgoing_edges_by_type(doc_node, "HAS_CHUNK")]
            for chunk_id in chunk_ids:
                candidate_entities.update(
                    e.target_id for e in txn.get_outgoing_edges_by_type(chunk_id, "CONTAINS")
                )
                self._delete_node_with_edges(txn, chunk_id)
            self._delete_node_with_edges(txn, doc_node)

        orphaned = {
            ent_id
            for ent_id in candidate_entities
            if not txn.get_incoming_edges_by_type(ent_id, "CONTAINS")
        }
        for ent_id in orphaned:
            self._delete_node_with_edges(txn, ent_id)
        return orphaned

    def evict_entities(self, entity_ids: set[int]) -> None:
        """Drop deleted entities from the canonical-entity cache (call after commit)."""
        if entity_ids:
            self._entity_cache = {
                key: nid for key, nid in self._entity_cache.items() if nid not in entity_ids
            }

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document, its chunks and its orphaned entities; False if not stored."""
        with self.db.write() as txn:
            orphaned = self.delete_document_in_txn(txn, doc_id)
            if orphaned is None:
                return False
            txn.commit()
        self.evict_entities(orphaned)
        logger.info("Deleted document %s (%d orphaned entities removed)", doc_id, len(orphaned))
        return True

    def _lookup_entity(self, name: str, pending: dict[str, int]) -> int | None:
        norm_name = name.strip().lower()
        found = self._entity_cache.get(norm_name)
        return found if found is not None else pending.get(norm_name)

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

        return IngestStats(
            chunk_count=len(chunks), entity_count=0, relation_count=0, chunk_ids=chunk_ids
        )

    def ingest_entities(
        self,
        chunk_id: int,
        entities: list[EntityData],
        relations: list[RelationData],
    ) -> IngestStats:
        """Ingest extracted entities and relations linked to their source chunk with canonical deduplication."""
        entity_name_to_id: dict[str, int] = {}
        created_relations = 0
        # New canonical entities are published to the shared cache only after commit
        pending: dict[str, int] = {}

        with self.db.write() as txn:
            for ent in entities:
                norm_name = ent.name.strip().lower()
                existing = self._lookup_entity(ent.name, pending)
                if existing is not None:
                    ent_id = existing
                else:
                    ent_node = txn.create_node(
                        labels=["Entity", ent.entity_type],
                        properties={"name": ent.name, "entity_type": ent.entity_type},
                    )
                    ent_id = ent_node.id
                    pending[norm_name] = ent_id

                txn.create_edge(chunk_id, ent_id, "CONTAINS")
                entity_name_to_id[ent.name] = ent_id

            for rel in relations:
                source_id = entity_name_to_id.get(rel.source_name) or self._lookup_entity(
                    rel.source_name, pending
                )
                target_id = entity_name_to_id.get(rel.target_name) or self._lookup_entity(
                    rel.target_name, pending
                )
                if source_id is not None and target_id is not None:
                    txn.create_edge(
                        source_id,
                        target_id,
                        "RELATION",
                        properties={"type": rel.relation_type},
                    )
                    created_relations += 1

            txn.commit()
        self._entity_cache.update(pending)

        return IngestStats(
            chunk_count=0,
            entity_count=len(entities),
            relation_count=created_relations,
        )

    def ingest_document_bundle(
        self,
        doc_id: str,
        title: str,
        chunks: list[ChunkData],
        chunk_entities_relations: list[tuple[int, list[EntityData], list[RelationData]]],
        tags: list[str] | None = None,
    ) -> IngestStats:
        """Atomically ingest document, chunks, entities, and relations in a single write transaction."""
        chunk_ids: list[int] = []
        entity_name_to_id: dict[str, int] = {}
        total_entities = 0
        total_relations = 0
        # New canonical entities are published to the shared cache only after commit, so a
        # rolled-back transaction cannot leave ids for nodes that were never written.
        pending: dict[str, int] = {}

        doc_properties: dict[str, Any] = {"doc_id": doc_id, "title": title}
        if tags:
            doc_properties["tags"] = list(tags)

        with self.db.write() as txn:
            doc_node = txn.create_node(labels=["Document"], properties=doc_properties)

            chunk_idx_to_id: dict[int, int] = {}
            for idx, chunk in enumerate(chunks):
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
                chunk_idx_to_id[idx] = chunk_node.id

            for c_ref, entities, relations in chunk_entities_relations:
                if isinstance(c_ref, int) and c_ref in chunk_idx_to_id:
                    real_chunk_id = chunk_idx_to_id[c_ref]
                elif isinstance(c_ref, int) and c_ref in chunk_ids:
                    real_chunk_id = c_ref
                elif chunk_ids:
                    real_chunk_id = chunk_ids[0]
                else:
                    real_chunk_id = None

                for ent in entities:
                    norm_name = ent.name.strip().lower()
                    existing = self._lookup_entity(ent.name, pending)
                    if existing is not None:
                        ent_id = existing
                    elif ent.name in entity_name_to_id:
                        ent_id = entity_name_to_id[ent.name]
                    else:
                        ent_node = txn.create_node(
                            labels=["Entity", ent.entity_type],
                            properties={"name": ent.name, "entity_type": ent.entity_type},
                        )
                        ent_id = ent_node.id
                        pending[norm_name] = ent_id
                        total_entities += 1

                    entity_name_to_id[ent.name] = ent_id
                    if real_chunk_id is not None:
                        txn.create_edge(real_chunk_id, ent_id, "CONTAINS")

                for rel in relations:
                    s_id = entity_name_to_id.get(rel.source_name) or self._lookup_entity(
                        rel.source_name, pending
                    )
                    t_id = entity_name_to_id.get(rel.target_name) or self._lookup_entity(
                        rel.target_name, pending
                    )
                    if s_id is not None and t_id is not None:
                        txn.create_edge(
                            s_id,
                            t_id,
                            "RELATION",
                            properties={"type": rel.relation_type},
                        )
                        total_relations += 1

            txn.commit()
        self._entity_cache.update(pending)

        return IngestStats(
            chunk_count=len(chunks),
            entity_count=total_entities,
            relation_count=total_relations,
            chunk_ids=chunk_ids,
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

        if top_k <= 0:
            return []

        # Databases seeded before entity vectors were dropped still hold Entity vectors in the
        # shared HNSW index. Over-fetch, and widen the fetch until top_k Chunks are found or the
        # index is exhausted, so those vectors cannot crowd Chunks out.
        fetch_k = max(top_k * 3, 20)
        while True:
            search_results, exhausted = self._chunk_hits(query_embedding, fetch_k, top_k)
            if len(search_results) >= top_k or exhausted or fetch_k >= _MAX_VECTOR_FETCH:
                return search_results
            fetch_k = min(fetch_k * 4, _MAX_VECTOR_FETCH)

    def _chunk_hits(
        self, query_embedding: np.ndarray, fetch_k: int, top_k: int
    ) -> tuple[list[SearchResult], bool]:
        """Return up to top_k Chunk hits among the fetch_k nearest vectors.

        The flag is True when the index returned fewer than fetch_k vectors, so a wider fetch
        cannot find more.
        """
        raw_results = self.db.vector_search(
            query_embedding, k=fetch_k, ef_search=max(_DEFAULT_EF_SEARCH, fetch_k)
        )
        exhausted = len(raw_results) < fetch_k

        search_results: list[SearchResult] = []
        with self.db.read() as txn:
            for r in raw_results:
                text = txn.get_property(r.node_id, "text")
                # Filter out nodes without text (e.g. Entity nodes)
                if not text or not str(text).strip():
                    continue

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
                if len(search_results) >= top_k:
                    break

        return search_results, exhausted

    def bm25_search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Perform native BM25 full-text search on Chunk text with keyword recall fallback."""
        raw_results = []
        try:
            raw_results = self.db.fts_search("Chunk", "text", query_text, limit=top_k)
        except Exception as e:
            logger.debug("BM25 search error: %s", e)

        # If natural-language question did not match exact conjunction, fallback to keyword search
        if not raw_results and query_text:
            import re

            stopwords = {
                "what",
                "is",
                "the",
                "of",
                "and",
                "are",
                "in",
                "a",
                "to",
                "for",
                "with",
                "from",
                "by",
                "that",
                "this",
                "how",
                "does",
                "do",
                "an",
            }
            words = [
                w
                for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", query_text)
                if w.lower() not in stopwords and len(w) > 2
            ]
            accum: dict[int, float] = {}
            for w in words:
                try:
                    for item in self.db.fts_search("Chunk", "text", w, limit=top_k):
                        accum[item.node_id] = accum.get(item.node_id, 0.0) + item.score
                except Exception:
                    pass
            if accum:
                top_sorted = sorted(accum.items(), key=lambda x: x[1], reverse=True)[:top_k]
                raw_results = [
                    type("FtsItem", (), {"node_id": nid, "score": sc})() for nid, sc in top_sorted
                ]

        search_results: list[SearchResult] = []
        with self.db.read() as txn:
            for r in raw_results:
                text = txn.get_property(r.node_id, "text")
                if not text or not str(text).strip():
                    continue

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
        """Traverse connected relations expanding from entity anchor nodes up to max_hops.

        ``budget`` caps nodes and edges separately. Reaching either cap ends the whole
        traversal, across every hop, and an edge is only kept together with both of its
        endpoints, so the result never holds a dangling edge.
        """
        if not entity_ids:
            return SubgraphResult(nodes=[], edges=[])

        visited_nodes: dict[int, dict[str, Any]] = {}
        visited_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[int, int, str]] = set()

        def exhausted() -> bool:
            return len(visited_nodes) >= budget or len(visited_edges) >= budget

        with self.db.read() as txn:

            def visit(node_id: int) -> None:
                name = txn.get_property(node_id, "name") or f"node_{node_id}"
                node_type = txn.get_property(node_id, "entity_type") or "Entity"
                visited_nodes[node_id] = {
                    "id": node_id,
                    "name": str(name),
                    "label": str(node_type),
                }

            frontier = list(entity_ids)
            for _hop in range(max_hops):
                next_frontier: list[int] = []
                for curr_id in frontier:
                    if exhausted():
                        break
                    if curr_id not in visited_nodes:
                        visit(curr_id)

                    for edge in txn.get_outgoing_edges(curr_id):
                        if exhausted():
                            break
                        rel_type = str(txn.get_edge_property(edge.id, "type") or edge.edge_type)
                        edge_key = (edge.source_id, edge.target_id, rel_type)
                        if edge_key in seen_edges:
                            continue
                        if edge.target_id not in visited_nodes:
                            visit(edge.target_id)
                            next_frontier.append(edge.target_id)
                        seen_edges.add(edge_key)
                        visited_edges.append(
                            {
                                "source_id": edge.source_id,
                                "target_id": edge.target_id,
                                "relation_type": rel_type,
                            }
                        )

                if exhausted() or not next_frontier:
                    break
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

    def get_entities_for_chunks(self, chunk_ids: list[int], budget: int = 25) -> list[int]:
        """Retrieve all entity IDs linked to a batch of Chunk IDs via CONTAINS edge in a single read transaction."""
        if not chunk_ids:
            return []
        entity_ids: list[int] = []
        seen: set[int] = set()
        with self.db.read() as txn:
            for cid in chunk_ids:
                edges = txn.get_outgoing_edges_by_type(cid, "CONTAINS")
                for e in edges:
                    if e.target_id not in seen:
                        seen.add(e.target_id)
                        entity_ids.append(e.target_id)
                        if len(entity_ids) >= budget:
                            return entity_ids
        return entity_ids

    def close(self) -> None:
        """Close the database connection safely."""
        if hasattr(self, "db") and self.db is not None and self.db.is_open:
            self.db.close()
