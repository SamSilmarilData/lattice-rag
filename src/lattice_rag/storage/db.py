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


def _clean(text: str) -> str:
    """Trim and collapse internal whitespace, keeping case."""
    return " ".join(text.split())


def entity_key(name: str, entity_type: str) -> tuple[str, str]:
    """Canonical Entity identity: name plus type, whitespace-collapsed and case-folded."""
    return _clean(name).casefold(), _clean(entity_type).casefold()


class IngestValidationError(ValueError):
    """An ingest payload is malformed; raised before anything is written."""


class EmbeddingDimensionError(IngestValidationError):
    """A chunk or entity embedding does not match the store's vector dimension."""

    def __init__(self, what: str, actual: int, expected: int) -> None:
        super().__init__(
            f"{what} embedding has dimension {actual}, but the store expects {expected}"
        )
        self.actual = actual
        self.expected = expected


class UnresolvedChunkReferenceError(IngestValidationError):
    """Entities or relations reference a chunk that is not part of the ingest payload."""

    def __init__(self, ref: object, chunk_count: int | None) -> None:
        scope = (
            "any stored chunk"
            if chunk_count is None
            else f"any of the {chunk_count} chunk(s) in this document"
        )
        super().__init__(f"chunk reference {ref!r} does not match {scope}")
        self.ref = ref


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
        self._entity_cache: dict[tuple[str, str], int] = {}

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

        self._load_entity_cache()

    def _load_entity_cache(self) -> None:
        """Preload the canonical (name, type) -> node_id mapping for O(1) deduplication."""
        try:
            entity_ids = self.db.get_nodes_by_label("Entity")
            if not entity_ids:
                return
            with self.db.read() as txn:
                for nid in entity_ids:
                    name = txn.get_property(nid, "name")
                    if name:
                        entity_type = txn.get_property(nid, "entity_type") or ""
                        key = entity_key(str(name), str(entity_type))
                        self._entity_cache.setdefault(key, nid)
        except Exception as e:
            logger.debug("Could not preload entity cache: %s", e)

    def get_canonical_entity_id(self, name: str, entity_type: str) -> int | None:
        """Return the canonical node_id for an entity's identity if it exists in the graph."""
        return self._entity_cache.get(entity_key(name, entity_type))

    def has_document(self, doc_id: str) -> bool:
        """Return True if a Document node with this doc_id is already stored."""
        doc_nodes = self.db.get_nodes_by_label("Document")
        if not doc_nodes:
            return False
        with self.db.read() as txn:
            return any(txn.get_property(nid, "doc_id") == doc_id for nid in doc_nodes)

    def _check_dimension(self, what: str, embedding: Any) -> None:
        actual = len(embedding)
        if actual != self.vector_dimensions:
            raise EmbeddingDimensionError(what, actual, self.vector_dimensions)

    def _validate_entities(self, entities: list[EntityData]) -> None:
        for ent in entities:
            if ent.embedding is not None:
                self._check_dimension(f"entity {ent.name!r}", ent.embedding)

    def _write_entity_group(
        self,
        txn: Any,
        chunk_id: int,
        entities: list[EntityData],
        relations: list[RelationData],
        pending: dict[tuple[str, str], int],
        doc_names: dict[str, int],
    ) -> tuple[int, int]:
        """Link one chunk's entities (creating new canonical ones) and their relations.

        New canonical ids go into ``pending``; the caller publishes them to the shared
        cache only after commit, so a rolled-back transaction cannot leave ids for nodes
        that were never written. Returns (entities created, relations created).

        Relations name their endpoints without a type, so they resolve against this
        chunk's entities first, then against entities seen earlier in the same document
        (``doc_names``). They never resolve by bare name against the whole graph, where a
        name alone is ambiguous; an unresolved relation is skipped.
        """
        created_entities = 0
        created_relations = 0
        by_name: dict[str, int] = {}
        linked: set[int] = set()

        for ent in entities:
            key = entity_key(ent.name, ent.entity_type)
            ent_id = self._entity_cache.get(key)
            if ent_id is None:
                ent_id = pending.get(key)
            if ent_id is None:
                name, entity_type = _clean(ent.name), _clean(ent.entity_type)
                ent_node = txn.create_node(
                    labels=["Entity", entity_type],
                    properties={"name": name, "entity_type": entity_type},
                )
                ent_id = ent_node.id
                if ent.embedding is not None:
                    txn.set_vector(ent_id, "embedding", ent.embedding)
                pending[key] = ent_id
                created_entities += 1

            by_name.setdefault(key[0], ent_id)
            if ent_id not in linked:
                linked.add(ent_id)
                txn.create_edge(chunk_id, ent_id, "CONTAINS")

        for name, ent_id in by_name.items():
            doc_names.setdefault(name, ent_id)

        def resolve(name: str) -> int | None:
            norm = entity_key(name, "")[0]
            found = by_name.get(norm)
            return found if found is not None else doc_names.get(norm)

        for rel in relations:
            s_id = resolve(rel.source_name)
            t_id = resolve(rel.target_name)
            if s_id is not None and t_id is not None:
                txn.create_edge(s_id, t_id, "RELATION", properties={"type": rel.relation_type})
                created_relations += 1

        return created_entities, created_relations

    def ingest_document(
        self,
        doc_id: str,
        title: str,
        chunks: list[ChunkData],
    ) -> IngestStats:
        """Ingest a document with its text chunks and vector embeddings (no entities)."""
        return self.ingest_document_bundle(doc_id, title, chunks, [])

    def ingest_entities(
        self,
        chunk_id: int,
        entities: list[EntityData],
        relations: list[RelationData],
    ) -> IngestStats:
        """Ingest entities and relations for an already stored chunk.

        Raises:
            UnresolvedChunkReferenceError: if ``chunk_id`` is not a stored Chunk node.
            EmbeddingDimensionError: if an entity embedding has the wrong dimension.
        """
        self._validate_entities(entities)
        with self.db.read() as txn:
            node = txn.get_node(chunk_id)
        if node is None or "Chunk" not in node.labels:
            raise UnresolvedChunkReferenceError(chunk_id, None)

        pending: dict[tuple[str, str], int] = {}
        with self.db.write() as txn:
            _created, created_relations = self._write_entity_group(
                txn, chunk_id, entities, relations, pending, {}
            )
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
        """Atomically ingest document, chunks, entities, and relations in a single write transaction.

        Each ``chunk_entities_relations`` entry names its chunk by index into ``chunks``.
        The whole payload is validated before anything is written.

        Raises:
            EmbeddingDimensionError: if a chunk or entity embedding has the wrong dimension.
            UnresolvedChunkReferenceError: if an entry's chunk index is out of range.
        """
        for idx, chunk in enumerate(chunks):
            self._check_dimension(f"chunk {idx}", chunk.embedding)
        for c_ref, entities, _relations in chunk_entities_relations:
            # bool is an int subclass; only a real index into ``chunks`` resolves
            if type(c_ref) is not int or not 0 <= c_ref < len(chunks):
                raise UnresolvedChunkReferenceError(c_ref, len(chunks))
            self._validate_entities(entities)

        chunk_ids: list[int] = []
        total_entities = 0
        total_relations = 0
        pending: dict[tuple[str, str], int] = {}
        doc_names: dict[str, int] = {}

        doc_properties: dict[str, Any] = {"doc_id": doc_id, "title": title}
        if tags:
            doc_properties["tags"] = list(tags)

        with self.db.write() as txn:
            doc_node = txn.create_node(labels=["Document"], properties=doc_properties)

            for chunk in chunks:
                chunk_node = txn.create_node(
                    labels=["Chunk"],
                    properties={
                        "text": chunk.text,
                        "position": chunk.position,
                        "doc_id": doc_id,
                    },
                )
                txn.set_vector(chunk_node.id, "embedding", chunk.embedding)
                txn.create_edge(doc_node.id, chunk_node.id, "HAS_CHUNK")
                chunk_ids.append(chunk_node.id)

            for c_ref, entities, relations in chunk_entities_relations:
                created_entities, created_relations = self._write_entity_group(
                    txn, chunk_ids[c_ref], entities, relations, pending, doc_names
                )
                total_entities += created_entities
                total_relations += created_relations

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

        # Retrieve a broader candidate pool to filter out non-chunk nodes (e.g. Entity nodes sharing vector index)
        fetch_k = max(top_k * 3, 20)
        raw_results = self.db.vector_search(query_embedding, k=fetch_k)

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

        return search_results

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
        """Traverse connected relations expanding from entity anchor nodes up to max_hops."""
        if not entity_ids:
            return SubgraphResult(nodes=[], edges=[])

        visited_nodes: dict[int, dict[str, Any]] = {}
        visited_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[int, int, str]] = set()

        with self.db.read() as txn:
            frontier = list(entity_ids)
            for _hop in range(max_hops):
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
                            visited_edges.append(
                                {
                                    "source_id": edge.source_id,
                                    "target_id": edge.target_id,
                                    "relation_type": rel_type,
                                }
                            )

                        if edge.target_id not in visited_nodes and len(visited_nodes) < budget:
                            target_name = (
                                txn.get_property(edge.target_id, "name") or f"node_{edge.target_id}"
                            )
                            target_type = (
                                txn.get_property(edge.target_id, "entity_type") or "Entity"
                            )
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
