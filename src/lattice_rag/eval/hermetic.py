"""Hermetic Tier: the key-free retrieval gate over the Gold Set (see ADR-0002).

The Gold Set corpus is seeded into an ephemeral LatticeStore with the real local
embedder, every Gold Passage is resolved to the Chunk(s) that contain it, and each query
runs through the retrieval pipeline exactly as the orchestrator would call it (same
``top_k`` per Route) but without the Jev guardrail, so no API key or network is needed.

Scored Slices report recall of Gold Passages and MRR, overall and per Slice, against the
floors in a versioned thresholds file (``hermetic_thresholds.json``). A Slice is gated on
its own floor only when it holds at least ``MIN_SLICE_QUERIES`` queries; a smaller one is
reported but not gated. The same metrics are also reported per Case (direct, paraphrase,
negation, ...), without floors. The passage-free
Slices are not scored for recall; they are checked for their own behavior instead:

- ``chitchat`` queries must take the deterministic chitchat path and get the
  ChitchatHandler reply recorded as their reference answer.
- Every other query (``unanswerable`` included) must *not* be short-circuited to chitchat,
  and must retrieve without error. The top rerank score is reported but never gated: the
  INT8 cross-encoder's raw logits shift by up to ~0.9 between CPU kernels (x86 AVX2 vs
  AVX-512 VNNI), so no fixed logit ceiling separates answerable from unanswerable.
  Abstention is judged in the Live Tier instead.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Protocol

import msgspec
import structlog

from lattice_rag.eval.gold_set import (
    PASSAGE_FREE_SLICES,
    GoldDocument,
    GoldQuery,
    GoldSet,
    meets_slice_minimum,
)
from lattice_rag.generation.chitchat import ChitchatHandler, is_pure_chitchat
from lattice_rag.ingestion.ingester import DocumentIngester, chunk_text
from lattice_rag.orchestration.graph import top_k_for_route
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from lattice_rag.security import sanitize_error_detail
from lattice_rag.storage.db import EntityData, LatticeStore, RelationData

logger = structlog.get_logger(__name__)

THRESHOLDS_VERSION: Final = "1"
OVERALL: Final = "overall"


class HermeticError(RuntimeError):
    """The Hermetic Tier could not run (bad thresholds file, unresolvable Gold Passage)."""


class MetricFloor(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Minimum mean recall and MRR a group of queries must reach."""

    recall: float
    mrr: float


class HermeticThresholds(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The gate's floors: overall and per scored Slice."""

    version: str
    overall: MetricFloor
    slices: dict[str, MetricFloor]


class SliceMetrics(msgspec.Struct, frozen=True):
    """Mean recall and MRR of one Slice (or ``overall``) against its floor.

    ``gated`` is False for a Slice too small to be gated on its own; it then has no floor.
    """

    slice: str
    queries: int
    recall: float
    mrr: float
    floor: MetricFloor | None
    passed: bool
    gated: bool = True


class CaseMetrics(msgspec.Struct, frozen=True):
    """Mean recall and MRR of the scored queries of one Case; reported, never gated."""

    case: str
    queries: int
    recall: float
    mrr: float


class QueryOutcome(msgspec.Struct, frozen=True):
    """One Gold Set query's result. ``recall``/``reciprocal_rank`` are None when unscored."""

    id: str
    slice: str
    case: str
    recall: float | None
    reciprocal_rank: float | None
    top_rerank_score: float | None
    problems: list[str]


class HermeticReport(msgspec.Struct, frozen=True):
    """Outcome of a Hermetic Tier run; ``failures`` lists every reason it did not pass.

    ``cases`` breaks the scored queries down by Case (orthogonal to Slice) for diagnosis;
    the gate's floors apply only to ``overall`` and ``slices``.
    """

    passed: bool
    overall: SliceMetrics
    slices: list[SliceMetrics]
    cases: list[CaseMetrics]
    queries: list[QueryOutcome]
    failures: list[str]


class Retriever(Protocol):
    async def retrieve(
        self,
        query: str,
        route: str = ...,
        apply_guardrail: bool = ...,
        top_k: int = ...,
    ) -> RetrievalResult: ...


RetrieverFactory = Callable[[LatticeStore, EmbeddingService], Retriever]


def _default_retriever(store: LatticeStore, embedding_service: EmbeddingService) -> Retriever:
    return RetrievalPipeline(store, embedding_service, router_guardrail=None)


class _NoEntities:
    """Extractor that finds nothing.

    The Hermetic Tier scores the final Chunks, which Stage 2 graph traversal does not
    change, so GLiNER would only add a model download and minutes of CPU to every run.
    """

    def extract_entities_batch(self, texts: list[str]) -> list[list[EntityData]]:
        return [[] for _ in texts]

    def extract_triples(self, text: str, entities: list[EntityData]) -> list[RelationData]:
        return []


def load_thresholds(path: str | Path) -> HermeticThresholds:
    """Load the versioned thresholds file; raises HermeticError if it is invalid."""
    file = Path(path)
    if not file.exists():
        raise HermeticError(f"Hermetic thresholds not found at: {file}")
    try:
        thresholds = msgspec.json.decode(file.read_bytes(), type=HermeticThresholds)
    except (msgspec.ValidationError, msgspec.DecodeError) as e:
        raise HermeticError(f"Invalid hermetic thresholds {file}: {e}") from e
    if thresholds.version != THRESHOLDS_VERSION:
        raise HermeticError(
            f"Hermetic thresholds {file} has version {thresholds.version!r}, "
            f"expected {THRESHOLDS_VERSION!r}"
        )
    return thresholds


# ── Gold Passage → Chunk resolution ─────────────────────────────────────────────


def _normalize(text: str) -> str:
    # Chunking re-joins sentences with single spaces, so compare whitespace-collapsed text
    return " ".join(text.split())


class _Span(msgspec.Struct, frozen=True, order=True):
    doc_id: str
    start: int
    end: int


class _SeededCorpus:
    """Where each seeded Chunk and each Gold Passage sits in its Document's text."""

    def __init__(self) -> None:
        self.chunk_spans: dict[int, _Span] = {}
        self._doc_text: dict[str, str] = {}

    def add_document(self, doc: GoldDocument, chunk_ids: list[int], chunk_texts: list[str]) -> None:
        if len(chunk_ids) != len(chunk_texts):
            raise HermeticError(
                f"{doc.document_id}: stored {len(chunk_ids)} chunks, expected {len(chunk_texts)}"
            )
        doc_text = _normalize(doc.text)
        self._doc_text[doc.document_id] = doc_text
        cursor = 0
        for chunk_id, text in zip(chunk_ids, chunk_texts, strict=True):
            norm = _normalize(text)
            start = doc_text.find(norm, cursor)
            if start < 0:
                raise HermeticError(f"{doc.document_id}: chunk {chunk_id} not found in its text")
            self.chunk_spans[chunk_id] = _Span(doc.document_id, start, start + len(norm))
            # Overlap can start a Chunk exactly where its predecessor starts
            cursor = start

    def passage_span(self, doc_id: str, text: str) -> _Span:
        norm = _normalize(text)
        start = self._doc_text.get(doc_id, "").find(norm)
        if start < 0:
            raise HermeticError(f"Gold Passage not found in {doc_id!r}: {text[:60]!r}")
        return _Span(doc_id, start, start + len(norm))


def _covered(passage: _Span, retrieved: list[_Span]) -> bool:
    """True when the retrieved Chunks of the passage's Document jointly contain all of it."""
    reach = passage.start
    for span in sorted(s for s in retrieved if s.doc_id == passage.doc_id):
        # Adjacent Chunks without overlap are separated by the single space between sentences
        if span.start > reach + 1:
            break
        reach = max(reach, span.end)
        if reach >= passage.end:
            return True
    return False


def _overlaps(chunk: _Span, passage: _Span) -> bool:
    return (
        chunk.doc_id == passage.doc_id and chunk.start < passage.end and passage.start < chunk.end
    )


# ── Run ─────────────────────────────────────────────────────────────────────────


async def _seed(
    store: LatticeStore, embedding_service: EmbeddingService, gold_set: GoldSet
) -> _SeededCorpus:
    ingester = DocumentIngester(
        store=store, embedding_service=embedding_service, extractor=_NoEntities()
    )
    corpus = _SeededCorpus()
    for doc in gold_set.corpus:
        stats = await ingester.ingest(doc_id=doc.document_id, title=doc.title, text=doc.text)
        texts = chunk_text(
            doc.text, target_size=ingester.target_chunk_size, overlap=ingester.chunk_overlap
        )
        corpus.add_document(doc, stats.chunk_ids, texts)
    return corpus


async def _evaluate_query(
    q: GoldQuery,
    retriever: Retriever,
    corpus: _SeededCorpus,
    chitchat: ChitchatHandler,
) -> QueryOutcome:
    problems: list[str] = []
    chitchat_routed = is_pure_chitchat(q.query)

    if q.slice == "chitchat":
        if not chitchat_routed:
            problems.append("not taken down the deterministic chitchat path")
        elif await chitchat.handle(q.query) != q.reference_answer:
            problems.append("chitchat reply differs from the reference answer")
        return QueryOutcome(q.id, q.slice, q.case, None, None, None, problems)

    if chitchat_routed:
        problems.append(f"expected route {q.expected_route!r} but is short-circuited to chitchat")

    try:
        result = await retriever.retrieve(
            q.query,
            route=q.expected_route,
            apply_guardrail=False,
            top_k=top_k_for_route(q.expected_route),
        )
        chunks: list[dict[str, Any]] = result.final_chunks
    except Exception as e:
        problems.append(f"retrieval failed: {sanitize_error_detail(str(e))}")
        chunks = []

    scores = [float(c["rerank_score"]) for c in chunks if c.get("rerank_score") is not None]
    top_score = max(scores) if scores else None

    if q.slice in PASSAGE_FREE_SLICES:
        return QueryOutcome(q.id, q.slice, q.case, None, None, top_score, problems)

    spans = [corpus.chunk_spans.get(c.get("node_id", -1)) for c in chunks]
    passages = [corpus.passage_span(p.document_id, p.text) for p in q.gold_passages]
    retrieved = [s for s in spans if s is not None]
    recall = sum(_covered(p, retrieved) for p in passages) / len(passages)
    reciprocal_rank = 0.0
    for rank, span in enumerate(spans, start=1):
        if span is not None and any(_overlaps(span, p) for p in passages):
            reciprocal_rank = 1.0 / rank
            break
    return QueryOutcome(q.id, q.slice, q.case, recall, reciprocal_rank, top_score, problems)


def _metrics(
    name: str, outcomes: list[QueryOutcome], floor: MetricFloor | None, gated: bool = True
) -> SliceMetrics:
    n = len(outcomes)
    recall = round(sum(o.recall or 0.0 for o in outcomes) / n, 4) if n else 0.0
    mrr = round(sum(o.reciprocal_rank or 0.0 for o in outcomes) / n, 4) if n else 0.0
    passed = floor is not None and recall >= floor.recall and mrr >= floor.mrr
    return SliceMetrics(name, n, recall, mrr, floor, passed, gated)


async def run_hermetic_tier(
    gold_set: GoldSet,
    thresholds: HermeticThresholds,
    embedding_service: EmbeddingService,
    retriever_factory: RetrieverFactory = _default_retriever,
) -> HermeticReport:
    """Seed an ephemeral store with the Gold Set corpus, run every query, apply the floors."""
    chitchat = ChitchatHandler()
    with tempfile.TemporaryDirectory(prefix="hermetic_") as tmp:
        store = LatticeStore(Path(tmp) / "hermetic.db")
        try:
            corpus = await _seed(store, embedding_service, gold_set)
            retriever = retriever_factory(store, embedding_service)
            outcomes = [
                await _evaluate_query(q, retriever, corpus, chitchat) for q in gold_set.queries
            ]
        finally:
            store.close()

    failures = [f"{o.id}: {p}" for o in outcomes for p in o.problems]

    scored = [o for o in outcomes if o.recall is not None]
    overall = _metrics(OVERALL, scored, thresholds.overall)
    slices: list[SliceMetrics] = []
    for name in dict.fromkeys(o.slice for o in scored):
        members = [o for o in scored if o.slice == name]
        if not meets_slice_minimum(len(members)):
            slices.append(_metrics(name, members, None, gated=False))
            continue
        floor = thresholds.slices.get(name)
        slices.append(_metrics(name, members, floor))
        if floor is None:
            failures.append(f"slice {name!r} has no threshold in the thresholds file")
    for m in [overall, *slices]:
        if m.floor is not None and not m.passed:
            failures.append(
                f"{m.slice}: recall {m.recall:.4f} / MRR {m.mrr:.4f} below floor "
                f"{m.floor.recall:.4f} / {m.floor.mrr:.4f}"
            )

    cases = []
    for name in dict.fromkeys(o.case for o in scored):
        m = _metrics(name, [o for o in scored if o.case == name], None)
        cases.append(CaseMetrics(name, m.queries, m.recall, m.mrr))

    report = HermeticReport(not failures, overall, slices, cases, outcomes, failures)
    logger.info(
        "hermetic_tier_completed",
        passed=report.passed,
        queries=len(outcomes),
        recall=overall.recall,
        mrr=overall.mrr,
        failures=len(failures),
    )
    return report
