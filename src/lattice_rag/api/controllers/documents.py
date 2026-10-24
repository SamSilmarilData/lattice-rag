from __future__ import annotations

import logging
from typing import Annotated

from litestar import Controller, delete
from litestar.di import NamedDependency
from litestar.exceptions import NotFoundException
from litestar.params import Parameter, SkipValidation

from lattice_rag.api.dtos import DOCUMENT_ID_PATTERN
from lattice_rag.api.guards import require_admin_key
from lattice_rag.ingestion.ingester import DocumentIngester, DocumentNotFoundError

logger = logging.getLogger(__name__)


class DocumentController(Controller):
    """Litestar controller for stored documents."""

    path = "/api/v1/documents"
    guards = [require_admin_key]

    @delete("/{document_id:str}")
    async def delete_document(
        self,
        document_id: Annotated[str, Parameter(pattern=DOCUMENT_ID_PATTERN)],
        ingester: NamedDependency[SkipValidation[DocumentIngester]],
    ) -> None:
        """Delete a document, its chunks and its orphaned entities, then clear Tier 1."""
        try:
            await ingester.delete(document_id)
        except DocumentNotFoundError as e:
            raise NotFoundException(detail=str(e)) from e
        logger.info("deleted_document", extra={"doc_id": document_id})
