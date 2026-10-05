"""Entity and relation triple extraction using GLiNER2.5-Decide on CPU."""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np

from lattice_rag.storage.db import EntityData, RelationData

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Simple sentence boundary regex — avoids requiring nltk as a dependency.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class EntityExtractor:
    """Local entity and relation triple extractor powered by GLiNER2.5-Decide."""

    def __init__(self, model_name: str = "FastinoLabs/gliner-2.5-decide") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self) -> None:
        """Lazy-load the GLiNER model on first use with automatic fallback."""
        if self._model is None:
            logger.info("Loading GLiNER model: %s", self.model_name)
            from gliner import GLiNER  # noqa: PLC0415

            try:
                self._model = GLiNER.from_pretrained(self.model_name)
            except Exception as e:
                fallback = "urchade/gliner_small-v2.1"
                logger.warning(
                    "Primary GLiNER model %s unavailable (%s). Falling back to %s",
                    self.model_name,
                    e,
                    fallback,
                )
                self._model = GLiNER.from_pretrained(fallback)

    def extract_entities(
        self,
        text: str,
        labels: list[str] | None = None,
    ) -> list[EntityData]:
        """Extract named entities from text using GLiNER.

        Embeddings are initialised to zero vectors here — the retrieval
        pipeline will populate them via FastEmbed before ingestion into
        LatticeDB's HNSW index.
        """
        self._ensure_model()
        if labels is None:
            labels = [
                "person",
                "organization",
                "technology",
                "concept",
                "algorithm",
                "database",
                "protocol",
            ]

        predictions = self._model.predict_entities(text, labels=labels)

        entities: list[EntityData] = []
        seen: set[str] = set()
        for p in predictions:
            name = p["text"].strip()
            if not name or name in seen:
                continue
            seen.add(name)
            entities.append(
                EntityData(
                    name=name,
                    entity_type=p["label"],
                    # Placeholder — FastEmbed fills real embeddings before storage.
                    embedding=None,
                )
            )
        return entities

    def extract_triples(
        self,
        text: str,
        entities: list[EntityData],
    ) -> list[RelationData]:
        """Derive relation triples via sentence-level co-occurrence heuristic.

        If two entities appear in the same sentence, they are linked with a
        ``CO_OCCURS`` relation whose type encodes both entity labels.
        Duplicate triples across sentences are pruned in O(1) time.
        """
        sentences = _SENTENCE_SPLIT.split(text) if text else []

        relations: list[RelationData] = []
        seen_triples: set[tuple[str, str, str]] = set()

        for sentence in sentences:
            present = [e for e in entities if e.name in sentence]
            for i, ent1 in enumerate(present):
                for ent2 in present[i + 1 :]:
                    rel_type = (
                        f"CO_OCCURS_IN_"
                        f"{ent1.entity_type.upper()}_"
                        f"{ent2.entity_type.upper()}"
                    )
                    triple_key = (ent1.name, ent2.name, rel_type)
                    if triple_key not in seen_triples:
                        seen_triples.add(triple_key)
                        relations.append(
                            RelationData(
                                source_name=ent1.name,
                                target_name=ent2.name,
                                relation_type=rel_type,
                            )
                        )
        return relations
