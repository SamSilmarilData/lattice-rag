"""Deep DocumentIngester module managing the end-to-end ingestion lifecycle."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import re
from typing import Any, List, Optional

from lattice_rag.storage.db import ChunkData, EntityData, IngestStats, RelationData

logger = logging.getLogger(__name__)


_SENTENCE_SPLITTER = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, target_size: int = 512, overlap: int = 64) -> list[str]:
    """Paragraph and sentence-aware chunking targeting ~512 chars with 64 overlap."""
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    sentences: list[str] = []
    for p in paragraphs:
        p_sentences = _SENTENCE_SPLITTER.split(p)
        for s in p_sentences:
            s = s.strip()
            if s:
                sentences.append(s)

    if not sentences:
        return [text]

    for sentence in sentences:
        s_len = len(sentence)
        if current_len + s_len + 1 <= target_size:
            current_chunk.append(sentence)
            current_len += s_len + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep tail sentences for overlap using slice
                overlap_len = 0
                keep_count = 0
                for prev in reversed(current_chunk):
                    if overlap_len + len(prev) <= overlap:
                        keep_count += 1
                        overlap_len += len(prev) + 1
                    else:
                        break
                current_chunk = current_chunk[-keep_count:] if keep_count > 0 else []
                current_chunk.append(sentence)
                current_len = sum(len(s) + 1 for s in current_chunk)
            else:
                chunks.append(sentence)
                current_chunk = []
                current_len = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class DocumentIngester:
    """Deep module orchestrating document chunking, embedding, extraction, and graph storage."""

    def __init__(
        self,
        store: Any,
        embedding_service: Any,
        extractor: Any,
        cache: Any = None,
        target_chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.store = store
        self.embedding_service = embedding_service
        self.extractor = extractor
        self.cache = cache
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    async def ingest(
        self,
        doc_id: str,
        title: str,
        text: str,
        tags: list[str] | None = None,
        target_chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestStats:
        """Ingest a text document: chunk, embed, extract entities/relations, and commit to LatticeDB."""
        t_size = target_chunk_size or self.target_chunk_size
        t_overlap = chunk_overlap or self.chunk_overlap

        text_chunks = chunk_text(text, target_size=t_size, overlap=t_overlap)
        if not text_chunks:
            logger.info("ingest_empty_text", extra={"doc_id": doc_id, "title": title})
            return IngestStats(chunk_count=0, entity_count=0, relation_count=0, chunk_ids=[])

        # 1. Batch embed all text chunks off the async event loop
        chunk_embeddings = await asyncio.to_thread(self.embedding_service.embed_texts, text_chunks)
        chunk_data_list = [
            ChunkData(text=c_text, embedding=emb, position=i, doc_id=doc_id)
            for i, (c_text, emb) in enumerate(zip(text_chunks, chunk_embeddings))
        ]

        # 2. Extract entities in batch off the event loop
        if hasattr(self.extractor, "extract_entities_batch"):
            chunk_entities_list = await asyncio.to_thread(
                self.extractor.extract_entities_batch, text_chunks
            )
        else:
            chunk_entities_list = await asyncio.to_thread(
                lambda: [self.extractor.extract_entities(t) for t in text_chunks]
            )

        chunk_extractions: list[tuple[int, list[EntityData], list[RelationData]]] = []
        all_entity_names: set[str] = set()

        for i, (c_text, entities) in enumerate(zip(text_chunks, chunk_entities_list)):
            triples = self.extractor.extract_triples(c_text, entities) if entities else []
            chunk_extractions.append((i, entities, triples))
            for e in entities:
                all_entity_names.add(e.name)

        # 3. Batched entity embedding: embed all unique entity names in a single ONNX call
        entity_embedding_map: dict[str, Any] = {}
        if all_entity_names:
            unique_names_list = list(all_entity_names)
            embedded_vectors = await asyncio.to_thread(
                self.embedding_service.embed_texts, unique_names_list
            )
            for name, vec in zip(unique_names_list, embedded_vectors):
                entity_embedding_map[name] = vec

        # Prepare embedded entities
        prepared_chunk_extractions: list[tuple[int, list[EntityData], list[RelationData]]] = []
        for i, entities, triples in chunk_extractions:
            embedded_entities = [
                EntityData(
                    name=e.name,
                    entity_type=e.entity_type,
                    embedding=entity_embedding_map.get(e.name),
                )
                for e in entities
            ]
            prepared_chunk_extractions.append((i, embedded_entities, triples))

        # 4. Atomic single-transaction ingestion if supported
        use_bundle = hasattr(type(self.store), "ingest_document_bundle") or (
            "ingest_document_bundle" in getattr(self.store, "__dict__", {})
        )
        if use_bundle:
            ingest_stats = self.store.ingest_document_bundle(
                doc_id=doc_id,
                title=title,
                chunks=chunk_data_list,
                chunk_entities_relations=prepared_chunk_extractions,
            )
            raw_ent = getattr(ingest_stats, "entity_count", 0)
            raw_rel = getattr(ingest_stats, "relation_count", 0)
            raw_ids = getattr(ingest_stats, "chunk_ids", [])
            total_entities = int(raw_ent) if isinstance(raw_ent, (int, float)) else 0
            total_relations = int(raw_rel) if isinstance(raw_rel, (int, float)) else 0
            chunk_node_ids = (
                [int(cid) for cid in raw_ids if isinstance(cid, (int, float))]
                if isinstance(raw_ids, list)
                else []
            )
        else:
            # Fallback for mock stores
            ingest_stats = self.store.ingest_document(
                doc_id=doc_id,
                title=title,
                chunks=chunk_data_list,
            )
            raw_ids = getattr(ingest_stats, "chunk_ids", [])
            chunk_node_ids = (
                [int(cid) for cid in raw_ids if isinstance(cid, (int, float))]
                if isinstance(raw_ids, list)
                else []
            )
            total_entities = 0
            total_relations = 0

            for i, chunk_node_id in enumerate(chunk_node_ids):
                if i < len(prepared_chunk_extractions):
                    _, embedded_entities, triples = prepared_chunk_extractions[i]
                    if embedded_entities:
                        ent_stats = self.store.ingest_entities(
                            chunk_id=chunk_node_id,
                            entities=embedded_entities,
                            relations=triples,
                        )
                        raw_ent = getattr(ent_stats, "entity_count", 0)
                        raw_rel = getattr(ent_stats, "relation_count", 0)
                        total_entities += int(raw_ent) if isinstance(raw_ent, (int, float)) else 0
                        total_relations += int(raw_rel) if isinstance(raw_rel, (int, float)) else 0

        # 5. Invalidate Tier 1 Semantic Cache to ensure stale answers are evicted
        if self.cache is not None:
            self.cache.clear()
            logger.info("tier1_cache_cleared_on_ingestion", extra={"doc_id": doc_id})

        return IngestStats(
            chunk_count=len(text_chunks),
            entity_count=total_entities,
            relation_count=total_relations,
            chunk_ids=chunk_node_ids,
        )

    async def ingest_file(
        self,
        filepath: Path | str,
        doc_id: str | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> IngestStats:
        """Ingest a text document from a local file path."""
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")
        resolved_doc_id = doc_id or path.stem
        resolved_title = title or path.name

        return await self.ingest(
            doc_id=resolved_doc_id,
            title=resolved_title,
            text=text,
            tags=tags,
        )
