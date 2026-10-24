from __future__ import annotations

import logging

from litestar import Controller, post
from litestar.di import NamedDependency
from litestar.exceptions import HTTPException
from litestar.params import JSONBody, SkipValidation
from litestar.status_codes import HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_ENTITY

from lattice_rag.api.dtos import IngestRequest, IngestResponse
from lattice_rag.api.guards import require_admin_key
from lattice_rag.ingestion.ingester import DocumentExistsError, DocumentIngester, chunk_text
from lattice_rag.storage.db import IngestValidationError

logger = logging.getLogger(__name__)

# Re-export chunk_text for backward-compatibility
__all__ = ["IngestController", "chunk_text"]


class IngestController(Controller):
    """Litestar controller for document ingestion."""

    path = "/api/v1"
    guards = [require_admin_key]

    @post("/ingest")
    async def ingest(
        self,
        data: JSONBody[IngestRequest],
        ingester: NamedDependency[SkipValidation[DocumentIngester]],
    ) -> IngestResponse:
        """Ingest a document, chunk it, embed, extract entities, and store in LatticeDB."""
        logger.info(
            "received_ingest_request", extra={"doc_id": data.document_id, "title": data.title}
        )

        try:
            stats = await ingester.ingest(
                doc_id=data.document_id,
                title=data.title,
                text=data.text,
                tags=data.tags,
                replace=data.replace,
            )
        except DocumentExistsError as e:
            raise HTTPException(status_code=HTTP_409_CONFLICT, detail=str(e)) from e
        except IngestValidationError as e:
            # Messages carry only dimensions and chunk indexes, never document text
            raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

        return IngestResponse(
            document_id=data.document_id,
            chunk_count=stats.chunk_count,
            entity_count=stats.entity_count,
            relation_count=stats.relation_count,
        )
