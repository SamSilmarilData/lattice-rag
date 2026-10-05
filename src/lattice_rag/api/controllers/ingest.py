from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from litestar import Controller, post
from litestar.di import NamedDependency
from litestar.params import JSONBody, SkipValidation

from lattice_rag.api.dtos import IngestRequest, IngestResponse
from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.storage.db import ChunkData, EntityData, LatticeStore
from lattice_rag.storage.extract import EntityExtractor

logger = logging.getLogger(__name__)


def chunk_text(text: str, target_size: int = 512, overlap: int = 64) -> list[str]:
    """Paragraph and sentence-aware chunking targeting ~512 chars with 64 overlap."""
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    sentence_splitter = re.compile(r"(?<=[.!?])\s+")

    sentences: list[str] = []
    for p in paragraphs:
        p_sentences = sentence_splitter.split(p)
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
                chunk_str = " ".join(current_chunk)
                chunks.append(chunk_str)
                # Keep tail sentences for overlap
                overlap_sentences: list[str] = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    if overlap_len + len(prev) <= overlap:
                        overlap_sentences.insert(0, prev)
                        overlap_len += len(prev) + 1
                    else:
                        break
                current_chunk = overlap_sentences + [sentence]
                current_len = sum(len(s) + 1 for s in current_chunk)
            else:
                chunks.append(sentence)
                current_chunk = []
                current_len = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class IngestController(Controller):
    """Litestar controller for document ingestion."""

    path = "/api/v1"

    @post("/ingest")
    async def ingest(
        self,
        data: JSONBody[IngestRequest],
        store: NamedDependency[SkipValidation[LatticeStore]],
        embedding_service: NamedDependency[SkipValidation[EmbeddingService]],
        extractor: NamedDependency[SkipValidation[EntityExtractor]],
        semantic_cache: NamedDependency[SkipValidation[SemanticCache]],
    ) -> IngestResponse:
        """Ingest a document, chunk it, embed, extract entities, and store in LatticeDB."""
        logger.info("received_ingest_request", extra={"doc_id": data.document_id, "title": data.title})

        text_chunks = chunk_text(data.text, target_size=512, overlap=64)
        if not text_chunks:
            return IngestResponse(
                document_id=data.document_id,
                chunk_count=0,
                entity_count=0,
                relation_count=0,
            )

        # 1. Embed text chunks
        chunk_embeddings = embedding_service.embed_texts(text_chunks)
        chunk_data_list = [
            ChunkData(text=c_text, embedding=emb, position=i)
            for i, (c_text, emb) in enumerate(zip(text_chunks, chunk_embeddings))
        ]

        # 2. Ingest document and chunks into LatticeDB
        ingest_stats = store.ingest_document(
            doc_id=data.document_id,
            title=data.title,
            chunks=chunk_data_list,
        )

        total_entities = 0
        total_relations = 0

        # 3. Extract entities & triples per chunk and link to chunk in graph
        for i, chunk_node_id in enumerate(ingest_stats.chunk_ids):
            c_text = text_chunks[i]
            entities = extractor.extract_entities(c_text)
            if entities:
                triples = extractor.extract_triples(c_text, entities)
                ent_names = [e.name for e in entities]
                ent_embs = embedding_service.embed_texts(ent_names)
                embedded_entities = [
                    EntityData(name=e.name, entity_type=e.entity_type, embedding=ent_embs[j])
                    for j, e in enumerate(entities)
                ]
                ent_stats = store.ingest_entities(
                    chunk_id=chunk_node_id,
                    entities=embedded_entities,
                    relations=triples,
                )
                total_entities += ent_stats.entity_count
                total_relations += ent_stats.relation_count

        # 4. Invalidate Tier 1 Semantic Cache on new document ingestion
        semantic_cache.clear()
        logger.info("tier1_cache_cleared_on_ingestion", extra={"doc_id": data.document_id})

        return IngestResponse(
            document_id=data.document_id,
            chunk_count=len(text_chunks),
            entity_count=total_entities,
            relation_count=total_relations,
        )
