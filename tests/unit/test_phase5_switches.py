"""Phase 5 experiment switches (spec #22: #46, #47, #48), tested through the orchestrator.

Each test drives ``RAGOrchestrator.run()`` and ``stream_query()`` over a real
``RetrievalPipeline`` whose store, embeddings, router, guardrails and providers are fakes,
so what is asserted is what the pipeline did, not how it is wired inside.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lattice_rag.generation.prompt import GraphContextMode
from lattice_rag.orchestration.graph import RAGOrchestrator
from lattice_rag.orchestration.state import PipelineState
from lattice_rag.retrieval.embeddings import EmbeddingService, RerankResult
from lattice_rag.retrieval.pipeline import RetrievalPipeline
from lattice_rag.routing.guardrail import GuardrailResult, build_route_guardrails
from lattice_rag.routing.router import RouteDecision
from lattice_rag.storage.db import SearchResult, SubgraphResult

ROUTES = ["vector_exact", "hybrid", "graph_relational", "massive_context"]


def _subgraph(edge_count: int) -> SubgraphResult:
    nodes = [{"id": i, "name": f"E{i}"} for i in range(edge_count + 1)]
    edges = [
        {"source_id": i, "target_id": i + 1, "relation_type": "LINKS"} for i in range(edge_count)
    ]
    return SubgraphResult(nodes=nodes, edges=edges)


class _Store:
    """LatticeDB fake: one chunk, linked to one entity, whose walk returns ``subgraph``."""

    def __init__(self, subgraph: SubgraphResult) -> None:
        hit = SearchResult(node_id=1, score=0.9, text="LatticeDB stores graphs.", metadata={})
        self.vector_search = MagicMock(return_value=[hit])
        self.bm25_search = MagicMock(return_value=[hit])
        self.get_entities_for_chunks = MagicMock(return_value=[10])
        self.traverse_from_entities = MagicMock(return_value=subgraph)

    def hops_walked(self) -> list[int]:
        return [c.kwargs["max_hops"] for c in self.traverse_from_entities.call_args_list]


class _Guardrail:
    """Noul fake that keeps every chunk and records that it ran."""

    def __init__(self) -> None:
        self.calls = 0

    async def filter_chunks(self, query: str, chunks: list[dict[str, Any]]) -> GuardrailResult:
        self.calls += 1
        return GuardrailResult(chunks=list(chunks), low_grounding=False)

    async def aclose(self) -> None:
        return None


class _Groq:
    """Primary provider fake that records the graph context it was given."""

    def __init__(self) -> None:
        self.graphs: list[dict[str, Any] | None] = []

    async def synthesize(self, query, chunks, graph_context, **_kw) -> str:
        self.graphs.append(graph_context)
        return "Grounded answer [Chunk 1]."

    async def stream(self, query, chunks, graph_context, **_kw):
        self.graphs.append(graph_context)
        yield "Grounded answer [Chunk 1]."


def _embed() -> MagicMock:
    embed = MagicMock(spec=EmbeddingService)
    embed.embed_query.return_value = np.ones(384, dtype=np.float32)
    embed.rerank.return_value = [RerankResult(index=0, score=9.0, text="LatticeDB stores graphs.")]
    return embed


def _orchestrator(
    route: str,
    *,
    store: _Store | None = None,
    groq: _Groq | None = None,
    guardrail: Any = None,
    pipeline_kwargs: dict[str, Any] | None = None,
    **orchestrator_kwargs: Any,
) -> tuple[RAGOrchestrator, MagicMock]:
    embed = _embed()
    pipeline = RetrievalPipeline(
        store or _Store(_subgraph(1)),
        embed,
        router_guardrail=guardrail if guardrail is not None else _Guardrail(),
        **(pipeline_kwargs or {}),
    )
    router = MagicMock()
    router.route_query = AsyncMock(return_value=RouteDecision(route=route, confidence=0.9))
    cache = MagicMock()
    cache.get_exact = MagicMock(return_value=None)
    cache.get = AsyncMock(return_value=None)
    cache.put = MagicMock()
    orchestrator = RAGOrchestrator(
        router=router,
        semantic_cache=cache,
        fallback_cache=MagicMock(),
        retrieval_pipeline=pipeline,
        groq=groq or _Groq(),
        gemini=groq or _Groq(),
        chitchat=MagicMock(),
        embedding_service=embed,
        **orchestrator_kwargs,
    )
    return orchestrator, cache


async def _stream_done(orchestrator: RAGOrchestrator) -> dict[str, Any]:
    done: dict[str, Any] = {}
    async for event in orchestrator.stream_query("How do the entities connect?"):
        if event["event"] == "done":
            done = event["data"]
    return done


async def _both(orchestrator: RAGOrchestrator) -> tuple[PipelineState, dict[str, Any]]:
    state = await orchestrator.run("How do the entities connect?")
    return state, await _stream_done(orchestrator)


# --- #47 GRAPH_CONTEXT_IN_PROMPT ------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
async def test_graph_context_on_sends_the_whole_subgraph_to_the_generator(route):
    groq = _Groq()
    orchestrator, _ = _orchestrator(route, store=_Store(_subgraph(12)), groq=groq)

    state, done = await _both(orchestrator)

    assert [len(g["edges"]) for g in groq.graphs] == [12, 12]
    assert state.prompt_graph_context == groq.graphs[0]
    assert done["prompt_graph"] == groq.graphs[1]


@pytest.mark.parametrize("route", ROUTES)
async def test_graph_context_off_sends_no_graph_but_still_reports_the_traversal(route):
    groq = _Groq()
    orchestrator, _ = _orchestrator(
        route,
        store=_Store(_subgraph(3)),
        groq=groq,
        graph_context_in_prompt=GraphContextMode.OFF,
    )

    state, done = await _both(orchestrator)

    assert groq.graphs == [None, None]
    assert state.prompt_graph_context is None and done["prompt_graph"] is None
    # The traversal still ran: Graph Recall and the UI graph path are unchanged
    assert len(state.graph_context["edges"]) == 3
    assert len(done["graph_path"]["edges"]) == 3


@pytest.mark.parametrize(
    ("route", "edges_sent"),
    [("graph_relational", 10), ("hybrid", 10), ("vector_exact", None), ("massive_context", None)],
)
async def test_graph_context_relational_sends_at_most_ten_edges_on_relational_routes(
    route, edges_sent
):
    groq = _Groq()
    orchestrator, _ = _orchestrator(
        route,
        store=_Store(_subgraph(12)),
        groq=groq,
        graph_context_in_prompt=GraphContextMode.RELATIONAL,
    )

    state, done = await _both(orchestrator)

    sent = [None if g is None else len(g["edges"]) for g in groq.graphs]
    assert sent == [edges_sent, edges_sent]
    if edges_sent:
        # Only the entities the kept edges connect: E0..E10
        assert [n["name"] for n in groq.graphs[0]["nodes"]] == [f"E{i}" for i in range(11)]
    assert state.prompt_graph_context == groq.graphs[0]
    assert done["prompt_graph"] == groq.graphs[1]


# --- #46 forced hop count (Route ablation) --------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
async def test_without_forced_hops_only_graph_relational_walks_two_hops(route):
    store = _Store(_subgraph(1))
    orchestrator, _ = _orchestrator(route, store=store)

    await _both(orchestrator)

    expected = 2 if route == "graph_relational" else 1
    assert store.hops_walked() == [expected, expected]


@pytest.mark.parametrize("hops", [1, 2])
@pytest.mark.parametrize("route", ROUTES)
async def test_forced_hops_override_every_retrieval_route_and_keep_the_routers_pick(route, hops):
    store = _Store(_subgraph(1))
    orchestrator, _ = _orchestrator(route, store=store, pipeline_kwargs={"forced_hops": hops})

    state, done = await _both(orchestrator)

    assert store.hops_walked() == [hops, hops]
    # Route Accuracy scores the router's actual choice, not the forced behaviour
    assert state.route == route and done["route"] == route
    assert state.route_verified is True


@pytest.mark.parametrize("hops", [0, 3])
def test_forced_hops_outside_one_or_two_are_rejected(hops):
    with pytest.raises(ValueError, match="hops"):
        RetrievalPipeline(_Store(_subgraph(1)), _embed(), forced_hops=hops)


async def test_switches_survive_rebuilding_on_the_ephemeral_eval_store():
    """The Live Tier reruns the pipeline on a temp DB; the arm's switches must come along."""
    groq = _Groq()
    orchestrator, _ = _orchestrator(
        "vector_exact",
        groq=groq,
        pipeline_kwargs={"forced_hops": 2},
        graph_context_in_prompt=GraphContextMode.OFF,
        answer_max_tokens=321,
    )
    eval_store = _Store(_subgraph(2))

    ephemeral = orchestrator.with_retrieval(orchestrator.retrieval.with_store(eval_store))
    state, _ = await _both(ephemeral)

    assert eval_store.hops_walked() == [2, 2]
    assert groq.graphs == [None, None]
    assert len(state.graph_context["edges"]) == 2
    assert ephemeral.answer_max_tokens == 321


# --- #48 Noul on massive_context ------------------------------------------------------


class _ThresholdGuardrails:
    """Factory of Noul fakes: each remembers the threshold it was built with."""

    def __init__(self) -> None:
        self.built: list[float] = []
        self.ran: list[float] = []
        self.closed = 0

    def __call__(self, threshold: float) -> Any:
        self.built.append(threshold)
        factory = self

        class _Noul(_Guardrail):
            async def filter_chunks(self, query, chunks):
                factory.ran.append(threshold)
                return await super().filter_chunks(query, chunks)

            async def aclose(self) -> None:
                factory.closed += 1

        return _Noul()


def _guarded(route: str, **policy: Any) -> tuple[RAGOrchestrator, MagicMock, _ThresholdGuardrails]:
    make = _ThresholdGuardrails()
    guardrails = build_route_guardrails(make, **policy)
    orchestrator, cache = _orchestrator(route, guardrail=guardrails)
    return orchestrator, cache, make


@pytest.mark.parametrize("route", ROUTES)
async def test_by_default_every_route_uses_the_global_noul_threshold(route):
    orchestrator, cache, make = _guarded(route)

    state, done = await _both(orchestrator)

    assert make.ran == [0.5, 0.5]
    assert state.unverified is False and done["unverified"] is False
    assert cache.put.call_count == 2


@pytest.mark.parametrize(("route", "threshold"), [("massive_context", 0.35), ("hybrid", 0.5)])
async def test_massive_context_threshold_applies_to_massive_context_only(route, threshold):
    orchestrator, cache, make = _guarded(route, massive_context_threshold=0.35)

    state, done = await _both(orchestrator)

    assert make.ran == [threshold, threshold]
    assert state.unverified is False and done["unverified"] is False
    assert cache.put.call_count == 2


async def test_skipping_noul_on_massive_context_leaves_the_answer_unverified_and_uncached():
    orchestrator, cache, make = _guarded("massive_context", skip_massive_context=True)

    state, done = await _both(orchestrator)

    assert make.ran == []
    assert state.unverified is True and done["unverified"] is True
    # The unchecked chunks still reach the answer
    assert [c["node_id"] for c in state.filtered_chunks] == [1]
    cache.put.assert_not_called()


@pytest.mark.parametrize("route", ["vector_exact", "hybrid", "graph_relational"])
async def test_skipping_noul_on_massive_context_keeps_it_on_other_routes(route):
    orchestrator, cache, make = _guarded(route, skip_massive_context=True)

    state, done = await _both(orchestrator)

    assert make.ran == [0.5, 0.5]
    assert state.unverified is False and done["unverified"] is False
    assert cache.put.call_count == 2


async def test_route_guardrails_close_every_guardrail_they_built():
    make = _ThresholdGuardrails()
    guardrails = build_route_guardrails(make, massive_context_threshold=0.35)

    await guardrails.aclose()

    assert make.built == [0.5, 0.35] and make.closed == 2


async def test_no_guardrail_factory_means_no_noul_on_any_route():
    """Degraded mode (no TypeSafe key): nothing is checked, every answer is Unverified."""
    orchestrator, cache = _orchestrator(
        "massive_context", guardrail=build_route_guardrails(None, massive_context_threshold=0.35)
    )

    state, done = await _both(orchestrator)

    assert state.unverified is True and done["unverified"] is True
    cache.put.assert_not_called()
