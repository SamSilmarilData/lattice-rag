"""RAG orchestrator: one pipeline core behind the buffered and streaming entry points."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lattice_rag.api.dtos import GenerationProvider
from lattice_rag.caching.semantic_cache import CacheVerificationError
from lattice_rag.config import DEFAULT_LLM_MAX_TOKENS
from lattice_rag.generation.prompt import graph_for_prompt
from lattice_rag.generation.resilient_synthesizer import (
    PoolMember,
    ResilientSynthesizer,
    StreamOutcome,
)
from lattice_rag.orchestration.state import PipelineState
from lattice_rag.vocabulary import GraphContextMode

if TYPE_CHECKING:
    from lattice_rag.caching.fallback_cache import FallbackCache
    from lattice_rag.caching.semantic_cache import SemanticCache
    from lattice_rag.generation.chitchat import ChitchatHandler
    from lattice_rag.retrieval.pipeline import RetrievalPipeline
    from lattice_rag.routing.router import QueryRouter, RouteDecision

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
# Cross-document synthesis goes to Gemini's large context window, so give it more evidence
# (bounded because every retrieved candidate passes through the CPU cross-encoder)
MASSIVE_CONTEXT_TOP_K = 12


def top_k_for_route(route: str) -> int:
    return MASSIVE_CONTEXT_TOP_K if route == "massive_context" else DEFAULT_TOP_K


def subgraph_as_dict(graph_context: Any) -> dict[str, Any] | None:
    """Convert a SubgraphResult (or an already-plain dict) into the dict stored on state."""
    if not graph_context:
        return None
    if isinstance(graph_context, dict):
        return graph_context
    return {"nodes": graph_context.nodes, "edges": graph_context.edges}


def pool_of(*, groq: Any = None, gemini: Any = None) -> tuple[PoolMember, ...]:
    """A Model Pool of one Groq adapter then one Gemini adapter, skipping absent ones.

    Each member's model is the adapter's ``model`` attribute when it is a string.
    """
    members = []
    for provider, client in ((GenerationProvider.GROQ, groq), (GenerationProvider.GEMINI, gemini)):
        if client is None:
            continue
        model = getattr(client, "model", None)
        members.append(
            PoolMember(provider, model if isinstance(model, str) else str(provider), client)
        )
    return tuple(members)


@dataclass
class _Triage:
    """What the front door learned: a Tier 1 hit, or the embedding and route for a miss."""

    cached: dict[str, Any] | None = None
    query_embedding: Any = None
    decision: RouteDecision | None = None
    # A near-duplicate existed but Noul cache verification could not run
    cache_unverified: bool = False


class RAGOrchestrator:
    """Runs the GraphRAG pipeline through a single core, :meth:`_pipeline`.

    Pipeline flow::

        query -> triage (exact cache | embed -> semantic cache | Jev route on a miss)
              -> chitchat | retrieval (stages 1-2-3 + guardrail)
              -> generation (circuit breaker cascade) -> Tier 1 write rule
              -> done

    ``stream_query()`` forwards the core's events; ``run()`` drains them into a
    :class:`PipelineState`. Caching, degradation and Unverified Answer rules live only
    in the core, so the two entry points cannot drift apart.
    """

    def __init__(
        self,
        *,
        router: QueryRouter,
        semantic_cache: SemanticCache,
        fallback_cache: FallbackCache,
        retrieval_pipeline: RetrievalPipeline,
        chitchat: ChitchatHandler,
        embedding_service: Any,
        pool: Sequence[PoolMember] | None = None,
        groq: Any = None,
        gemini: Any = None,
        answer_max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        graph_context_in_prompt: GraphContextMode = GraphContextMode.ON,
    ) -> None:
        """``pool`` is the Model Pool in cascade order (built by ``build_model_pool``).

        ``groq``/``gemini`` are a shorthand for a two-member pool of those adapters, in
        that order; they are ignored when ``pool`` is given.
        """
        self.router = router
        self.semantic_cache = semantic_cache
        self.fallback_cache = fallback_cache
        self.retrieval = retrieval_pipeline
        self.pool: tuple[PoolMember, ...] = (
            tuple(pool) if pool is not None else pool_of(groq=groq, gemini=gemini)
        )
        self.chitchat = chitchat
        self.embedding_service = embedding_service
        self.answer_max_tokens = answer_max_tokens
        self.graph_context_in_prompt = graph_context_in_prompt
        self.synthesizer = ResilientSynthesizer(
            pool=self.pool,
            fallback_cache=self.fallback_cache,
            default_max_tokens=answer_max_tokens,
        )

    def with_retrieval(self, retrieval_pipeline: RetrievalPipeline) -> RAGOrchestrator:
        """The same orchestrator over another retrieval pipeline (the eval temp DB).

        Every setting carries over, so a Live Tier arm measures the configuration it names.
        The synthesizer is shared, so a pool member whose breaker opened stays skipped.
        """
        clone = RAGOrchestrator(
            router=self.router,
            semantic_cache=self.semantic_cache,
            fallback_cache=self.fallback_cache,
            retrieval_pipeline=retrieval_pipeline,
            chitchat=self.chitchat,
            embedding_service=self.embedding_service,
            pool=self.pool,
            answer_max_tokens=self.answer_max_tokens,
            graph_context_in_prompt=self.graph_context_in_prompt,
        )
        clone.synthesizer = self.synthesizer
        return clone

    # ── Public Interfaces ────────────────────────────────────────────────

    async def run(self, query: str, use_cache: bool = True) -> PipelineState:
        """Execute the full pipeline and return the buffered final state.

        ``use_cache=False`` bypasses Tier 1 entirely (no lookup, no store), so evaluation
        measures the live pipeline without touching the cache users are served from.
        """
        done: dict[str, Any] = {}
        async for event in self._pipeline(query, use_cache=use_cache, streaming=False):
            if event["event"] == "done":
                done = event["data"]

        return PipelineState(
            query=query,
            use_cache=use_cache,
            route=done["route"],
            cache_hit=done["cached"],
            graph_context=done["graph_path"],
            prompt_graph_context=done["prompt_graph"],
            filtered_chunks=done["sources"],
            low_grounding=done["low_grounding"],
            unverified=done["unverified"],
            route_verified=done["route_verified"],
            answer=done["answer"],
            degraded=done["degraded"],
            provider=done["provider"],
            model=done["model"],
            timings=done["timings"],
            total_latency_ms=done["latency_ms"],
        )

    async def stream_query(self, query: str) -> AsyncGenerator[dict[str, Any], None]:
        """Stream typed SSE lifecycle events ('stage', 'token', 'error', 'done')."""
        async for event in self._pipeline(query, use_cache=True, streaming=True):
            yield event

    # ── Core ─────────────────────────────────────────────────────────────

    async def _pipeline(
        self, query: str, *, use_cache: bool, streaming: bool
    ) -> AsyncGenerator[dict[str, Any], None]:
        """The one pipeline. ``streaming`` only chooses how generation produces tokens."""
        t0 = time.perf_counter()
        timings: list[dict[str, Any]] = []

        def finish(**data: Any) -> dict[str, Any]:
            data["timings"] = timings
            data["latency_ms"] = (time.perf_counter() - t0) * 1000
            return {"event": "done", "data": data}

        # 1. Triage
        yield {"event": "stage", "data": {"stage": "triage", "status": "started"}}
        t_stage = time.perf_counter()
        triage = await self._triage(query, use_cache=use_cache)
        timings.append({"stage": "triage", "duration_ms": (time.perf_counter() - t_stage) * 1000})

        if triage.cached is not None:
            # A Tier 1 entry was verified, grounded and non-degraded when it was stored
            cached = triage.cached
            route = cached.get("route") or (
                triage.decision.route if triage.decision else "vector_exact"
            )
            answer = cached.get("answer", "")
            logger.info("Tier 1 cache HIT (route=%s)", route)
            yield {"event": "stage", "data": {"stage": "cache_hit", "route": route}}
            yield {"event": "token", "data": {"delta": answer}}
            yield finish(
                answer=answer,
                route=route,
                cached=True,
                sources=cached.get("sources", []),
                graph_path=cached.get("graph_path"),
                prompt_graph=cached.get("prompt_graph"),
                degraded=False,
                provider=None,
                model=None,
                low_grounding=False,
                unverified=False,
                route_verified=True,  # only verified answers are stored in Tier 1
            )
            return

        decision = triage.decision
        assert decision is not None  # a miss always carries a routing decision
        route = decision.route

        # 2a. Chitchat: deterministic, no LLM, never cached
        if route == "chitchat":
            yield {"event": "stage", "data": {"stage": "chitchat", "route": "chitchat"}}
            t_stage = time.perf_counter()
            answer = await self.chitchat.handle(query)
            timings.append(
                {"stage": "chitchat", "duration_ms": (time.perf_counter() - t_stage) * 1000}
            )
            yield {"event": "token", "data": {"delta": answer}}
            yield finish(
                answer=answer,
                route="chitchat",
                cached=False,
                sources=[],
                graph_path=None,
                prompt_graph=None,
                degraded=False,
                provider=None,
                model=None,
                low_grounding=False,
                unverified=False,
                route_verified=decision.verified,
            )
            return

        # 2b. Retrieval (massive_context gets a larger evidence budget)
        yield {"event": "stage", "data": {"stage": "retrieval", "route": route}}
        t_stage = time.perf_counter()
        retrieved = await self.retrieval.execute(
            query,
            route,
            top_k=top_k_for_route(route),
            query_embedding=triage.query_embedding,
        )
        timings.append(
            {"stage": "retrieval", "duration_ms": (time.perf_counter() - t_stage) * 1000}
        )
        chunks = retrieved.final_chunks
        graph_dict = subgraph_as_dict(retrieved.graph_context)
        prompt_graph = graph_for_prompt(graph_dict, route, self.graph_context_in_prompt)

        # 3. Generation
        yield {"event": "stage", "data": {"stage": "generation", "chunks_count": len(chunks)}}
        t_stage = time.perf_counter()
        outcome = StreamOutcome()
        tokens: list[str] = []
        async for tok in self._generate(query, chunks, prompt_graph, route, outcome, streaming):
            tokens.append(tok)
            yield {"event": "token", "data": {"delta": tok}}
        if outcome.error:
            yield {"event": "error", "data": {"detail": outcome.error}}
        timings.append(
            {"stage": "generation", "duration_ms": (time.perf_counter() - t_stage) * 1000}
        )
        answer = "".join(tokens)

        # 4. Tier 1 write rule: complete Model Pool answers (any member; not the FAQ, not an
        #    interrupted stream) grounded in verified evidence only
        unverified = not decision.verified or retrieved.unverified or triage.cache_unverified
        if (
            use_cache
            and not outcome.degraded
            and not retrieved.low_grounding
            and not unverified
            and triage.query_embedding is not None
        ):
            self.semantic_cache.put(
                query,
                triage.query_embedding,
                {
                    "answer": answer,
                    "route": route,
                    "sources": chunks,
                    "graph_path": graph_dict,
                    "prompt_graph": prompt_graph,
                },
            )

        yield finish(
            answer=answer,
            route=route,
            cached=False,
            sources=chunks,
            graph_path=graph_dict,
            prompt_graph=prompt_graph,
            degraded=outcome.degraded,
            provider=outcome.provider,
            model=outcome.model,
            low_grounding=retrieved.low_grounding,
            unverified=unverified,
            route_verified=decision.verified,
        )

    async def _triage(self, query: str, *, use_cache: bool) -> _Triage:
        """Front door: exact match, then embed + semantic match, and only on a miss the route.

        Tier 1 is consulted before Jev routing so a hit (exact or semantic) never pays for a
        routing call. Routing starts only after the embedding succeeded, so a failed
        embedding leaves no routing task behind.
        """
        if use_cache:
            exact_hit = self.semantic_cache.get_exact(query)
            if isinstance(exact_hit, dict):
                return _Triage(cached=exact_hit)

        query_embedding = await asyncio.to_thread(self.embedding_service.embed_query, query)
        cache_unverified = False
        if use_cache:
            try:
                cached = await self.semantic_cache.get(query, query_embedding)
            except CacheVerificationError:
                # A near-duplicate could not be checked: recompute, and flag the answer
                cached = None
                cache_unverified = True
            if cached is not None:
                return _Triage(cached=cached, query_embedding=query_embedding)

        decision = await self.router.route_query(query)
        return _Triage(
            query_embedding=query_embedding,
            decision=decision,
            cache_unverified=cache_unverified,
        )

    async def _generate(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None,
        route: str,
        outcome: StreamOutcome,
        streaming: bool,
    ) -> AsyncGenerator[str, None]:
        """Yield answer tokens and fill ``outcome``; buffered mode yields the whole answer."""
        if streaming:
            async for tok in self.synthesizer.stream(
                query=query,
                context_chunks=chunks,
                graph_context=graph_context,
                route=route,
                outcome=outcome,
            ):
                yield tok
            return

        synth = await self.synthesizer.synthesize(
            query=query, context_chunks=chunks, graph_context=graph_context, route=route
        )
        outcome.provider = synth.provider
        outcome.model = synth.model
        outcome.degraded = synth.degraded
        yield synth.answer
