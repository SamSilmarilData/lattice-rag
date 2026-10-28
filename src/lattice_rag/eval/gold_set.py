"""Gold Set v3: the evaluation dataset's schema, loader and validator.

The file format (``eval_dataset.json``, version 3.0.0) is::

    {
      "version": "3.0.0",
      "domain": "...",
      "corpus": [{"document_id": "...", "title": "...", "text": "..."}],
      "queries": [{
        "id": "...", "query": "...",
        "slice": "factoid|multihop|unanswerable|chitchat|long_context",
        "case": "direct|paraphrase|negation|ambiguity|conflicting_sources|
                 unanswerable_near_miss",
        "expected_route": "hybrid|massive_context|chitchat",
        "reference_answer": "...",
        "gold_passages": [{"document_id": "...", "text": "<verbatim quote>"}],
        "expected_entities": ["..."]
      }]
    }

``load_gold_set`` rejects a file that is structurally wrong (missing field, unknown
Slice, Case or Route, including the Routes ADR-0003 collapsed into ``hybrid``) or
whose content is inconsistent (a Gold Passage that is not verbatim in its Document, a
passage for an unknown Document, duplicate ids, or a Slice whose passage count makes no
sense), an expected entity the corpus never mentions, or a
non-chitchat Slice with fewer than ``MIN_SLICE_QUERIES`` queries.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Final

import msgspec

from lattice_rag.vocabulary import Case, Route, Slice

GOLD_SET_VERSION: Final = "3.0.0"

# Slices whose queries the corpus cannot (unanswerable) or should not (chitchat) answer.
PASSAGE_FREE_SLICES: Final[frozenset[str]] = frozenset({"unanswerable", "chitchat"})

# Fewest queries a Slice needs before a mean over it means anything. Every Slice but
# chitchat (deterministic, never scored for recall) must reach it, and per-Slice gates
# apply only to Slices that do.
MIN_SLICE_QUERIES = 10
UNSIZED_SLICES: Final[frozenset[str]] = frozenset({"chitchat"})


def meets_slice_minimum(query_count: int) -> bool:
    """Whether a Slice of ``query_count`` queries is large enough to be gated on its own."""
    return query_count >= MIN_SLICE_QUERIES


class GoldSetError(ValueError):
    """The Gold Set file is malformed or inconsistent with its own corpus."""


class GoldDocument(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A corpus Document seeded into the store before the Gold Set is run."""

    document_id: str
    title: str
    text: str


class GoldPassage(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A verbatim quote from one Gold Set Document that correct retrieval must surface."""

    document_id: str
    text: str


class GoldQuery(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One evaluation query with its Slice, Case, expected Route, reference answer and evidence."""

    id: str
    query: str
    slice: Slice
    case: Case
    expected_route: Route
    reference_answer: str
    gold_passages: list[GoldPassage]
    expected_entities: list[str] = msgspec.field(default_factory=list)


class GoldSet(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The curated evaluation dataset: a corpus plus queries over it."""

    version: str
    domain: str
    corpus: list[GoldDocument]
    queries: list[GoldQuery]


def _fold(text: str) -> str:
    """Whitespace-collapsed, case-folded text for case-insensitive containment checks."""
    return " ".join(text.split()).casefold()


def _mentions(folded_text: str, entity: str) -> bool:
    """Whether ``entity`` appears in ``folded_text`` as a whole term, ignoring case.

    Whole term: not inside a longer word, so "Rust" is not found in "trust".
    """
    pattern = rf"(?<!\w){re.escape(_fold(entity))}(?!\w)"
    return re.search(pattern, folded_text) is not None


def find_problems(gold_set: GoldSet) -> list[str]:
    """Return every content problem in the Gold Set; an empty list means it is valid."""
    problems: list[str] = []
    if gold_set.version != GOLD_SET_VERSION:
        problems.append(f"version is {gold_set.version!r}, expected {GOLD_SET_VERSION!r}")

    docs: dict[str, GoldDocument] = {}
    for doc in gold_set.corpus:
        if doc.document_id in docs:
            problems.append(f"duplicate document_id {doc.document_id!r}")
        docs[doc.document_id] = doc

    corpus_text = _fold("\n".join(doc.text for doc in gold_set.corpus))

    seen_ids: set[str] = set()
    for q in gold_set.queries:
        problems.extend(
            f"{q.id}: expected entity {entity!r} is not in the corpus"
            for entity in q.expected_entities
            if not entity.strip() or not _mentions(corpus_text, entity)
        )

        if q.id in seen_ids:
            problems.append(f"duplicate query id {q.id!r}")
        seen_ids.add(q.id)

        if not q.query.strip():
            problems.append(f"{q.id}: query is empty")
        if not q.reference_answer.strip():
            problems.append(f"{q.id}: reference_answer is empty")

        if q.slice in PASSAGE_FREE_SLICES and q.gold_passages:
            problems.append(f"{q.id}: slice {q.slice!r} must not have gold_passages")
        if q.slice not in PASSAGE_FREE_SLICES and not q.gold_passages:
            problems.append(f"{q.id}: slice {q.slice!r} needs at least one gold passage")

        for i, passage in enumerate(q.gold_passages):
            source = docs.get(passage.document_id)
            if source is None:
                problems.append(
                    f"{q.id}: gold_passages[{i}] cites unknown document {passage.document_id!r}"
                )
            elif not passage.text.strip():
                problems.append(f"{q.id}: gold_passages[{i}] is empty")
            elif passage.text not in source.text:
                problems.append(
                    f"{q.id}: gold_passages[{i}] is not verbatim in {passage.document_id!r}: "
                    f"{passage.text[:60]!r}"
                )

    sizes = Counter(q.slice for q in gold_set.queries if q.slice not in UNSIZED_SLICES)
    problems.extend(
        f"slice {name!r} has {count} queries, needs at least {MIN_SLICE_QUERIES}"
        for name, count in sorted(sizes.items())
        if not meets_slice_minimum(count)
    )
    return problems


def load_gold_set(path: str | Path) -> GoldSet:
    """Load and validate a Gold Set file.

    Raises:
        FileNotFoundError: the file does not exist.
        GoldSetError: the file is malformed or fails validation; the message names
            the offending field or query.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Gold Set not found at: {file}")
    try:
        gold_set = msgspec.json.decode(file.read_bytes(), type=GoldSet)
    except msgspec.ValidationError as e:
        raise GoldSetError(f"Invalid Gold Set {file}: {e}") from e
    except msgspec.DecodeError as e:
        raise GoldSetError(f"Gold Set {file} is not valid JSON: {e}") from e

    problems = find_problems(gold_set)
    if problems:
        joined = "\n  - ".join(problems)
        raise GoldSetError(f"Invalid Gold Set {file}:\n  - {joined}")
    return gold_set
