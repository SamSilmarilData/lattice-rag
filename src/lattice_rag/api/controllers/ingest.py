from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from litestar import Controller, post
from litestar.di import NamedDependency
from litestar.params import JSONBody, SkipValidation

from lattice_rag.api.dtos import IngestRequest, IngestResponse
from lattice_rag.caching.semantic_cache import SemanticCache
from lattice_rag.ingestion.ingester import DocumentIngester, chunk_text
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.storage.db import LatticeStore
from lattice_rag.storage.extract import EntityExtractor

logger = logging.getLogger(__name__)

# Re-export chunk_text for backward-compatibility
__all__ = ["IngestController", "chunk_text"]


class IngestController(Controller):
    """Litestar controller for document ingestion."""

    path = "/api/v1"

    @post("/ingest")
    async def ingest(
        self,
        data: JSONBody[IngestRequest],
        ingester: NamedDependency[SkipValidation[DocumentIngester | None]] = None,
        store: NamedDependency[SkipValidation[LatticeStore | None]] = None,
        embedding_service: NamedDependency[SkipValidation[EmbeddingService | None]] = None,
        extractor: NamedDependency[SkipValidation[EntityExtractor | None]] = None,
        semantic_cache: NamedDependency[SkipValidation[SemanticCache | None]] = None,
    ) -> IngestResponse:
        """Ingest a document, chunk it, embed, extract entities, and store in LatticeDB."""
        logger.info("received_ingest_request", extra={"doc_id": data.document_id, "title": data.title})

        active_ingester = ingester
        if active_ingester is None:
            if store is not None and embedding_service is not None and extractor is not None:
                active_ingester = DocumentIngester(
                    store=store,
                    embedding_service=embedding_service,
                    extractor=extractor,
                    cache=semantic_cache,
                )
            else:
                raise RuntimeError("DocumentIngester or its underlying storage dependencies must be provided.")

        stats = await active_ingester.ingest(
            doc_id=data.document_id,
            title=data.title,
            text=data.text,
            tags=data.tags,
        )

        return IngestResponse(
            document_id=data.document_id,
            chunk_count=stats.chunk_count,
            entity_count=stats.entity_count,
            relation_count=stats.relation_count,
        )
