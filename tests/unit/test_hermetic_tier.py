"""Seam 3, Hermetic Tier: a Gold Set run through a real LatticeStore with no API keys.

The local ONNX models are replaced by deterministic word-overlap fakes so the suite stays
offline; everything else (seeding, Chunk resolution, the retrieval pipeline, scoring and
the thresholds gate) is the production code path.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pytest

from lattice_rag.eval.gold_set import GoldDocument, GoldPassage, GoldQuery, GoldSet
from lattice_rag.eval.hermetic import (
    HermeticThresholds,
    MetricFloor,
    load_thresholds,
    run_hermetic_tier,
)
from lattice_rag.retrieval.embeddings import EmbeddingService
from lattice_rag.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from lattice_rag.storage.db import SubgraphResult

_WORD = re.compile(r"[a-z0-9]+")
_DIM = 384


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3}


class WordHashEmbedder:
    """Bag-of-words vectors: texts sharing rare words land close together."""

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        out = []
        for text in texts:
            vec = np.zeros(_DIM, dtype=np.float32)
            for word in _tokens(text):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIM] += 1.0
            norm = float(np.linalg.norm(vec)) or 1.0
            out.append(vec / norm)
        return out


class WordOverlapReranker:
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        q = _tokens(query)
        return [float(len(q & _tokens(d))) for d in documents]


def fake_embedding_service() -> EmbeddingService:
    return EmbeddingService(embedder=WordHashEmbedder(), reranker=WordOverlapReranker())


# Each Document chunks into several ~512-char Chunks, so retrieval has to pick the right one.
ASTRONOMY = (
    "Pulsars are rotating neutron stars that emit beams of radiation. "
    "Their rotation periods range from milliseconds to several seconds. "
    "Glitches in pulsar timing reveal superfluid interiors of neutron stars. "
    "Millisecond pulsars are spun up by accreting matter from a binary companion.\n\n"
    "Quasars are extremely luminous galactic nuclei powered by supermassive black holes. "
    "Accretion discs around those black holes radiate across the spectrum. "
    "Relativistic jets launched from quasars can extend far beyond their host galaxies. "
    "Distant quasars backlight intergalactic hydrogen clouds along the line of sight.\n\n"
    "Exoplanets are detected through the transit method, which measures the dimming of a star "
    "when a planet passes in front of it. Radial velocity surveys find planets through stellar "
    "wobble. Direct imaging blocks starlight with a coronagraph to photograph young giant planets. "
    "Gravitational microlensing finds cold planets far from their host stars.\n\n"
    "Cosmic microwave background photons were released when the universe became transparent. "
    "Tiny temperature anisotropies in that background map primordial density fluctuations. "
    "Polarization patterns of the background constrain reionization history."
)
COOKING = (
    "Sourdough bread rises through wild yeast and lactobacilli in a fermented starter. "
    "Bakers feed the starter with flour and water every day. "
    "Long cold proofing develops sour flavour and an open crumb. "
    "Steam during the first minutes of baking lets the loaf expand before the crust sets.\n\n"
    "Risotto is made by toasting arborio rice and adding warm broth one ladle at a time. "
    "Constant stirring releases starch and gives risotto its creamy texture. "
    "Finishing with cold butter and parmesan is called mantecatura. "
    "Carnaroli rice holds its shape better than arborio under long stirring.\n\n"
    "Tempering chocolate means heating and cooling it to precise temperatures so cocoa butter "
    "crystallizes evenly, which gives the chocolate snap and gloss. Seeding melted chocolate "
    "with tempered callets is the simplest tempering technique. Untempered chocolate blooms "
    "with grey streaks of fat.\n\n"
    "Fermented hot sauce relies on lactic bacteria converting chilli sugars into acid. "
    "A brine of two percent salt suppresses spoilage organisms during fermentation. "
    "Blending the mash with vinegar halts fermentation and stabilises the sauce."
)

PULSAR = "Pulsars are rotating neutron stars that emit beams of radiation."
TRANSIT = "Exoplanets are detected through the transit method"
RISOTTO = "Constant stirring releases starch and gives risotto its creamy texture."
TEMPERING = "Tempering chocolate means heating and cooling it to precise temperatures"


def _q(
    qid: str,
    query: str,
    passages: list[tuple[str, str]],
    slice_: Any = "factoid",
    route: Any = "vector_exact",
    reference: str = "reference",
) -> GoldQuery:
    return GoldQuery(
        id=qid,
        query=query,
        slice=slice_,
        expected_route=route,
        reference_answer=reference,
        gold_passages=[GoldPassage(document_id=d, text=t) for d, t in passages],
    )


def gold_set(*extra: GoldQuery) -> GoldSet:
    return GoldSet(
        version="2.0.0",
        domain="test",
        corpus=[
            GoldDocument(document_id="doc_astro", title="Astronomy", text=ASTRONOMY),
            GoldDocument(document_id="doc_cook", title="Cooking", text=COOKING),
        ],
        queries=[
            _q(
                "q_pulsar",
                "What are pulsars and their rotating neutron stars?",
                [("doc_astro", PULSAR)],
            ),
            _q(
                "q_transit",
                "How are exoplanets detected by the transit method?",
                [("doc_astro", TRANSIT)],
            ),
            _q(
                "q_risotto",
                "Why does stirring risotto give creamy texture?",
                [("doc_cook", RISOTTO)],
            ),
            _q(
                "q_both",
                "Compare tempering chocolate temperatures with exoplanets transit detection",
                [("doc_cook", TEMPERING), ("doc_astro", TRANSIT)],
                slice_="multihop",
                route="hybrid",
            ),
            *extra,
        ],
    )


def thresholds(recall: float = 0.9, mrr: float = 0.9) -> HermeticThresholds:
    floor = MetricFloor(recall=recall, mrr=mrr)
    return HermeticThresholds(
        version="1", overall=floor, slices={"factoid": floor, "multihop": floor}
    )


async def test_healthy_retriever_passes_with_full_recall_and_mrr() -> None:
    report = await run_hermetic_tier(gold_set(), thresholds(), fake_embedding_service())

    assert report.passed, report.failures
    assert report.overall.recall == 1.0
    assert report.overall.mrr == 1.0
    by_slice = {s.slice: s for s in report.slices}
    assert by_slice["factoid"].queries == 3
    assert by_slice["multihop"].queries == 1


class ReversedRetriever:
    """Deliberately broken: asks the real pipeline for every chunk and returns the worst first."""

    def __init__(self, store: Any, embedding_service: EmbeddingService) -> None:
        self.pipeline = RetrievalPipeline(store, embedding_service)

    async def retrieve(
        self, query: str, route: str = "hybrid", apply_guardrail: bool = True, top_k: int = 5
    ) -> RetrievalResult:
        everything = await self.pipeline.retrieve(query, route, apply_guardrail=False, top_k=100)
        return RetrievalResult(
            final_chunks=list(reversed(everything.final_chunks))[:top_k],
            graph_context=SubgraphResult(nodes=[], edges=[]),
        )


async def test_broken_retriever_fails_the_gate() -> None:
    report = await run_hermetic_tier(
        gold_set(), thresholds(), fake_embedding_service(), retriever_factory=ReversedRetriever
    )

    assert not report.passed
    assert report.overall.mrr < 0.9
    assert any(f.startswith("overall:") for f in report.failures)


class KeywordRetriever:
    """Returns the BM25 hit for each of a fixed list of keywords, in order, whatever the query."""

    def __init__(self, store: Any, keywords: list[str]) -> None:
        self.store = store
        self.keywords = keywords

    async def retrieve(
        self, query: str, route: str = "hybrid", apply_guardrail: bool = True, top_k: int = 5
    ) -> RetrievalResult:
        chunks = []
        for word in self.keywords:
            hit = self.store.bm25_search(word, top_k=1)[0]
            chunks.append({"node_id": hit.node_id, "text": hit.text, "rerank_score": 1.0})
        return RetrievalResult(
            final_chunks=chunks, graph_context=SubgraphResult(nodes=[], edges=[])
        )


def _with_queries(*queries: GoldQuery) -> GoldSet:
    base = gold_set()
    return GoldSet(
        version=base.version, domain=base.domain, corpus=base.corpus, queries=list(queries)
    )


SPANNING = "radiate across the spectrum. Relativistic jets launched"


@pytest.mark.parametrize(
    ("keywords", "recall", "rr"),
    [
        (["accretion"], 0.0, 1.0),  # the first Chunk holds only the start of the passage
        (["sourdough", "relativistic"], 1.0, 0.5),  # the second Chunk carries the overlap sentence
        (["sourdough", "accretion", "relativistic"], 1.0, 0.5),  # together they cover it
    ],
)
async def test_passage_spanning_overlapping_chunks_is_recalled_from_either_side_of_the_overlap(
    keywords: list[str], recall: float, rr: float
) -> None:
    report = await run_hermetic_tier(
        _with_queries(_q("q_span", "quasar jets", [("doc_astro", SPANNING)])),
        thresholds(0.0, 0.0),
        fake_embedding_service(),
        retriever_factory=lambda store, _: KeywordRetriever(store, keywords),
    )

    (outcome,) = report.queries
    assert outcome.recall == recall
    assert outcome.reciprocal_rank == rr


GREETING_REPLY = (
    "Hello! I am lattice-rag, a Hybrid GraphRAG engine. "
    "How can I help you with your knowledge base today?"
)


async def test_passage_free_slices_are_excluded_from_recall_and_pass_their_own_checks() -> None:
    report = await run_hermetic_tier(
        gold_set(
            _q("q_hi", "hello there", [], "chitchat", "chitchat", GREETING_REPLY),
            _q("q_licence", "Who owns licence royalties?", [], "unanswerable"),
        ),
        HermeticThresholds(
            version="1",
            overall=MetricFloor(recall=0.9, mrr=0.9),
            slices={s: MetricFloor(recall=0.9, mrr=0.9) for s in ("factoid", "multihop")},
        ),
        fake_embedding_service(),
    )

    assert report.passed, report.failures
    assert report.overall.queries == 4
    assert {s.slice for s in report.slices} == {"factoid", "multihop"}
    by_id = {o.id: o for o in report.queries}
    assert by_id["q_hi"].recall is None
    assert by_id["q_licence"].recall is None


class ShiftedReranker:
    """The word-overlap reranker with every score moved by a constant, as another CPU would."""

    def __init__(self, shift: float) -> None:
        self.shift = shift

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [s + self.shift for s in WordOverlapReranker().rerank(query, documents)]


# The INT8 reranker's logits differ by up to 0.90 between x86 AVX2 and AVX-512 VNNI
# kernels (measured over the Gold Set on GitHub runners), with identical ranking quality.
CROSS_CPU_RERANK_SPREAD = 0.9


@pytest.mark.parametrize("shift", [-CROSS_CPU_RERANK_SPREAD, CROSS_CPU_RERANK_SPREAD])
async def test_gate_verdict_does_not_depend_on_which_cpu_scored_the_chunks(shift: float) -> None:
    gold = gold_set(_q("q_licence", "Who owns licence royalties?", [], "unanswerable"))
    gate = load_thresholds("hermetic_thresholds.json")

    here = await run_hermetic_tier(gold, gate, fake_embedding_service())
    elsewhere = await run_hermetic_tier(
        gold,
        gate,
        EmbeddingService(embedder=WordHashEmbedder(), reranker=ShiftedReranker(shift)),
    )

    assert here.passed, here.failures
    assert elsewhere.passed == here.passed, elsewhere.failures


@pytest.mark.parametrize(
    ("query", "expected_problem"),
    [
        # A chitchat query whose recorded reply is not what the handler says
        (_q("q_bye", "goodbye", [], "chitchat", "chitchat", GREETING_REPLY), "differs"),
        # A chitchat query that is really a question never reaches the chitchat path
        (
            _q("q_mixed", "hello, what is a pulsar?", [], "chitchat", "chitchat", GREETING_REPLY),
            "not taken down",
        ),
        # A real query that the router would short-circuit to chitchat
        (_q("q_thanks", "thanks", [("doc_astro", PULSAR)]), "short-circuited"),
    ],
)
async def test_a_query_breaking_its_slice_behavior_fails_the_gate(
    query: GoldQuery, expected_problem: str
) -> None:
    report = await run_hermetic_tier(
        _with_queries(query),
        HermeticThresholds(
            version="1",
            overall=MetricFloor(recall=0.0, mrr=0.0),
            slices={"factoid": MetricFloor(recall=0.0, mrr=0.0)},
        ),
        fake_embedding_service(),
    )

    assert not report.passed
    assert any(f.startswith(f"{query.id}:") and expected_problem in f for f in report.failures)


async def test_a_slice_without_a_threshold_fails_the_gate() -> None:
    gate = HermeticThresholds(
        version="1",
        overall=MetricFloor(recall=0.0, mrr=0.0),
        slices={"factoid": MetricFloor(0.0, 0.0)},
    )

    report = await run_hermetic_tier(gold_set(), gate, fake_embedding_service())

    assert not report.passed
    assert any("'multihop' has no threshold" in f for f in report.failures)


class ExplodingRetriever:
    async def retrieve(
        self, query: str, route: str = "hybrid", apply_guardrail: bool = True, top_k: int = 5
    ) -> RetrievalResult:
        raise RuntimeError("index unavailable")


async def test_a_retrieval_error_fails_its_query() -> None:
    report = await run_hermetic_tier(
        gold_set(),
        thresholds(0.0, 0.0),
        fake_embedding_service(),
        retriever_factory=lambda *_: ExplodingRetriever(),
    )

    assert not report.passed
    assert report.overall.recall == 0.0
    assert "q_pulsar: retrieval failed: index unavailable" in report.failures
