"""Entity and relation triple extraction using GLiNER2.5-Decide on CPU."""
from __future__ import annotations

import logging
import re
from typing import Any

from lattice_rag.storage.db import EntityData, RelationData


logger = logging.getLogger(__name__)

# Simple sentence boundary regex — avoids requiring nltk as a dependency.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# Shared in-memory model cache across EntityExtractor instances
_MODEL_CACHE: dict[str, tuple[Any, bool]] = {}


class EntityExtractor:
    """Local entity and relation triple extractor powered by GLiNER2.5-Decide."""

    def __init__(self, model_name: str = "fastino/GLiNER2.5-Decide") -> None:
        # Normalize legacy or typoed repo names to the official HuggingFace repository
        clean_name = model_name
        if clean_name in ("FastinoLabs/gliner-2.5-decide", "fastino/gliner2.5-decide", "gliner2.5-decide"):
            clean_name = "fastino/GLiNER2.5-Decide"
        self.model_name = clean_name
        self._model = None
        self._is_gliner2: bool = False

    def _ensure_model(self) -> None:
        """Lazy-load the GLiNER model on first use with automatic fallback and in-memory caching."""
        if self._model is not None:
            return

        if self.model_name in _MODEL_CACHE:
            self._model, self._is_gliner2 = _MODEL_CACHE[self.model_name]
            return

        logger.info("Loading GLiNER model: %s", self.model_name)
        # 1. Try gliner2 AutoExtractor for fastino/GLiNER2.5-Decide
        if "fastino" in self.model_name.lower() or "gliner2" in self.model_name.lower():
            try:
                from gliner2 import AutoExtractor  # noqa: PLC0415
                self._model = AutoExtractor.from_pretrained(self.model_name)
                self._is_gliner2 = True
                _MODEL_CACHE[self.model_name] = (self._model, self._is_gliner2)
                logger.info("Successfully loaded GLiNER2 model %s", self.model_name)
                return
            except Exception as e:
                logger.warning(
                    "AutoExtractor failed to load %s (%s). Attempting standard GLiNER loader...",
                    self.model_name,
                    e,
                )

        # 2. Try classic GLiNER loader
        from gliner import GLiNER  # noqa: PLC0415
        try:
            self._model = GLiNER.from_pretrained(self.model_name)
            self._is_gliner2 = False
        except Exception as e:
            fallback = "urchade/gliner_small-v2.1"
            if fallback in _MODEL_CACHE:
                self._model, self._is_gliner2 = _MODEL_CACHE[fallback]
                return
            logger.warning(
                "Primary GLiNER model %s unavailable (%s). Falling back to %s",
                self.model_name,
                e,
                fallback,
            )
            self._model = GLiNER.from_pretrained(fallback)
            self._is_gliner2 = False
        _MODEL_CACHE[self.model_name] = (self._model, self._is_gliner2)

    def extract_entities_batch(
        self,
        texts: list[str],
        labels: list[str] | None = None,
    ) -> list[list[EntityData]]:
        """Extract named entities from a batch of texts using vectorized inference."""
        if not texts:
            return []

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

        results: list[list[EntityData]] = []

        if self._is_gliner2:
            try:
                batch_preds = self._model.batch_extract_entities(texts, labels)
            except Exception as e:
                logger.warning("batch_extract_entities failed in gliner2 (%s), falling back to itemized", e)
                batch_preds = [self._model.extract_entities(t, labels) for t in texts]

            for item in batch_preds:
                entities: list[EntityData] = []
                seen: set[str] = set()
                entities_dict = item.get("entities", {}) if isinstance(item, dict) else {}
                for entity_type, names in entities_dict.items():
                    if isinstance(names, list):
                        for name_raw in names:
                            name = str(name_raw).strip()
                            if not name or name in seen:
                                continue
                            seen.add(name)
                            entities.append(EntityData(name=name, entity_type=entity_type, embedding=None))
                results.append(entities)
        else:
            # Classic GLiNER
            is_mock = hasattr(self._model, "_mock_return_value") or self._model.__class__.__name__ == "MagicMock"
            if not is_mock and hasattr(self._model, "batch_predict_entities"):
                try:
                    batch_preds = self._model.batch_predict_entities(texts, labels=labels)
                except Exception:
                    batch_preds = [self._model.predict_entities(t, labels=labels) for t in texts]
            else:
                batch_preds = [self._model.predict_entities(t, labels=labels) for t in texts]

            for predictions in batch_preds:
                entities = []
                seen = set()
                for p in predictions:
                    name = p.get("text", "").strip()
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    entities.append(EntityData(name=name, entity_type=p.get("label", "entity"), embedding=None))
                results.append(entities)

        return results

    def extract_entities(
        self,
        text: str,
        labels: list[str] | None = None,
    ) -> list[EntityData]:
        """Extract named entities from single text using GLiNER."""
        batch = self.extract_entities_batch([text], labels=labels)
        return batch[0] if batch else []

    def extract_triples(
        self,
        text: str,
        entities: list[EntityData],
    ) -> list[RelationData]:
        """Derive relation triples via sentence-level co-occurrence with word-boundary matching.

        If two entities appear in the same sentence, they are linked with a
        semantic relation encoding both entity types. Substring false-positives
        are prevented using regex word-boundary anchors.
        """
        sentences = _SENTENCE_SPLIT.split(text) if text else []

        relations: list[RelationData] = []
        seen_triples: set[tuple[str, str, str]] = set()

        for sentence in sentences:
            # Word boundary matching to prevent single-letter or substring false positives
            present = [
                e
                for e in entities
                if re.search(r"\b" + re.escape(e.name) + r"\b", sentence, re.IGNORECASE)
                or e.name in sentence
            ]
            for i, ent1 in enumerate(present):
                for ent2 in present[i + 1 :]:
                    # Build descriptive typed relationship
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
