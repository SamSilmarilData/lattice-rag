"""Ingestion subsystem package."""
from __future__ import annotations

from lattice_rag.ingestion.ingester import DocumentIngester, chunk_text

__all__ = ["DocumentIngester", "chunk_text"]
