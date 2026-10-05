"""LangGraph StateGraph assembling the full RAG pipeline."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

from langgraph.graph import END, StateGraph

from lattice_rag.orchestration.state import PipelineState

if TYPE_CHECKING:
    from lattice_rag.caching.fallback_cache import FallbackCache
    from lattice_rag.caching.semantic_cache import SemanticCache
    from lattice_rag.generation.chitchat import ChitchatHandler
    from lattice_rag.generation.gemini_fallback import GeminiFallback
    from lattice_rag.generation.groq_synthesizer import GroqSynthesizer
    from lattice_rag.retrieval.pipeline import RetrievalPipeline
    from lattice_rag.routing.router import QueryRouter

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Assembles and runs the full GraphRAG pipeline via LangGraph StateGraph.

    Pipeline flow::

        query -> [concurrent: cache_check + jev_route]
              -> route_dispatch
              -> retrieval_pipeline (stages 1-2-3 + guardrail)
              -> generation (with circuit breaker cascade)
              -> response
    """

    def __init__(
        self,
        *,
        router: QueryRouter,
        semantic_cache: SemanticCache,
        fallback_cache: FallbackCache,
        retrieval_pipeline: RetrievalPipeline,
        groq: GroqSynthesizer,
        gemini: GeminiFallback,
        chitchat: ChitchatHandler,
        embedding_service: Any,
    ) -> None:
        self.router = router
        self.semantic_cache = semantic_cache
        self.fallback_cache = fallback_cache
        self.retrieval = retrieval_pipeline
        self.retrieval_pipeline = retrieval_pipeline
        self.groq = groq
        self.gemini = gemini
        self.chitchat = chitchat
        self.embedding_service = embedding_service
        from lattice_rag.generation.resilient_synthesizer import ResilientSynthesizer
        self.synthesizer = ResilientSynthesizer(
            groq=self.groq,
            gemini=self.gemini,
            fallback_cache=self.fallback_cache,
        )
        self._graph = self._build_graph()

    # ── Graph Construction ──────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """Wire the LangGraph StateGraph nodes and edges."""
        graph = StateGraph(PipelineState)

        graph.add_node("triage", self._triage_node)
        graph.add_node("route_dispatch", self._route_dispatch_node)
        graph.add_node("chitchat", self._chitchat_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("massive_context", self._massive_context_node)

        graph.set_entry_point("triage")

        graph.add_conditional_edges(
            "triage",
            self._after_triage,
            {
                "cached": END,
                "route_dispatch": "route_dispatch",
            },
        )

        graph.add_conditional_edges(
            "route_dispatch",
            self._select_route,
            {
                "chitchat": "chitchat",
                "massive_context": "massive_context",
                "retrieve": "retrieve",
            },
        )

        graph.add_edge("chitchat", END)
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        graph.add_edge("massive_context", END)

        return graph.compile()

    # ── Node Implementations ────────────────────────────────────────────

    async def _triage_node(self, state: PipelineState) -> dict[str, Any]:
        """Two-stage front-door triage: O(1) exact match check, then concurrent Jev routing + CPU embedding."""
        t0 = time.perf_counter()

        # Step 1: O(1) Exact-match cache shortcut
        exact_hit = (
            self.semantic_cache.get_exact(state.query)
            if hasattr(type(self.semantic_cache), "get_exact")
            else None
        )
        if isinstance(exact_hit, dict):
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Exact cache HIT for query in O(1) time")
            return {
                "cache_hit": True,
                "cached_response": exact_hit,
                "answer": exact_hit.get("answer", ""),
                "filtered_chunks": exact_hit.get("sources", []),
                "graph_context": exact_hit.get("graph_path"),
                "route": exact_hit.get("route", "vector_exact"),
                "timings": state.timings + [{"stage": "triage", "duration_ms": elapsed}],
            }

        # Step 2: Concurrently execute Jev routing and CPU embedding (offloaded to thread)
        route_task = asyncio.create_task(self.router.route_query(state.query))
        query_embedding = await asyncio.to_thread(self.embedding_service.embed_query, state.query)
        cache_result = await self.semantic_cache.get(state.query, query_embedding)
        route_decision = await route_task

        elapsed = (time.perf_counter() - t0) * 1000

        updates: dict[str, Any] = {
            "query_embedding": query_embedding,
            "route": route_decision.route,
            "route_confidence": route_decision.confidence,
            "timings": state.timings + [{"stage": "triage", "duration_ms": elapsed}],
        }

        if cache_result is not None:
            updates["cache_hit"] = True
            updates["cached_response"] = cache_result
            updates["answer"] = cache_result.get("answer", "")
            updates["filtered_chunks"] = cache_result.get("sources", [])
            updates["graph_context"] = cache_result.get("graph_path")
            logger.info("Cache HIT for query (route=%s)", route_decision.route)

        return updates

    async def _chitchat_node(self, state: PipelineState) -> dict[str, Any]:
        """Handle chitchat queries with zero LLM cost."""
        t0 = time.perf_counter()
        answer = await self.chitchat.handle(state.query)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "answer": answer,
            "timings": state.timings + [{"stage": "chitchat", "duration_ms": elapsed}],
        }

    async def _retrieve_node(self, state: PipelineState) -> dict[str, Any]:
        """Execute 3-stage retrieval pipeline with guardrail."""
        t0 = time.perf_counter()
        result = await self.retrieval.execute(state.query, state.route)
        elapsed = (time.perf_counter() - t0) * 1000

        graph_dict = None
        if result.graph_context:
            if hasattr(result.graph_context, "nodes") and hasattr(result.graph_context, "edges"):
                graph_dict = {
                    "nodes": result.graph_context.nodes,
                    "edges": result.graph_context.edges,
                }
            elif isinstance(result.graph_context, dict):
                graph_dict = result.graph_context

        return {
            "filtered_chunks": result.final_chunks,
            "graph_context": graph_dict,
            "timings": state.timings + [{"stage": "retrieval", "duration_ms": elapsed}],
        }

    async def _generate_node(self, state: PipelineState) -> dict[str, Any]:
        """Generate answer with circuit breaker cascade: Groq -> Gemini -> Redis."""
        t0 = time.perf_counter()
        degraded = False

        try:
            # Primary: Groq under circuit breaker
            if hasattr(self.fallback_cache, "call_with_breaker") and callable(self.fallback_cache.call_with_breaker):
                res = self.fallback_cache.call_with_breaker(
                    self.groq.synthesize,
                    state.query,
                    state.filtered_chunks,
                    state.graph_context,
                )
                if hasattr(res, "__await__"):
                    answer = await res
                else:
                    answer = res
            else:
                synth = await self.synthesizer.synthesize(
                    state.query,
                    state.filtered_chunks,
                    state.graph_context,
                    state.route,
                )
                answer = synth.answer
                degraded = synth.degraded
        except Exception as e:
            logger.warning("Groq synthesis failed, cascading to Gemini: %s", e)
            try:
                answer = await self.gemini.synthesize(
                    state.query,
                    state.filtered_chunks,
                    state.graph_context,
                )
                degraded = True
            except Exception as e2:
                logger.warning("Gemini fallback failed, using Redis FAQ cache: %s", e2)
                cached = await self.fallback_cache.find_closest_fallback(state.query)
                if cached:
                    answer = cached
                else:
                    answer = (
                        "I'm sorry, all generation services are currently unavailable. "
                        "Please try again in a moment."
                    )
                degraded = True

        elapsed = (time.perf_counter() - t0) * 1000

        # Cache the verified synthesis in Tier 1 Semantic Vector Cache
        if not degraded and state.query_embedding is not None:
            self.semantic_cache.put(
                state.query,
                state.query_embedding,
                {
                    "answer": answer,
                    "route": state.route,
                    "sources": state.filtered_chunks,
                    "graph_path": state.graph_context,
                },
            )

        return {
            "answer": answer,
            "degraded": degraded,
            "timings": state.timings + [{"stage": "generation", "duration_ms": elapsed}],
        }

    async def _massive_context_node(self, state: PipelineState) -> dict[str, Any]:
        """Handle massive context queries via Gemini."""
        t0 = time.perf_counter()

        result = await self.retrieval.execute(state.query, state.route)
        graph_dict = None
        if result.graph_context:
            if hasattr(result.graph_context, "nodes") and hasattr(result.graph_context, "edges"):
                graph_dict = {
                    "nodes": result.graph_context.nodes,
                    "edges": result.graph_context.edges,
                }
            elif isinstance(result.graph_context, dict):
                graph_dict = result.graph_context

        try:
            answer = await self.gemini.synthesize(
                state.query,
                result.final_chunks,
                graph_dict,
            )
            degraded = False
        except Exception as e:
            logger.warning("Gemini massive context failed, using fallback FAQ: %s", e)
            cached = await self.fallback_cache.find_closest_fallback(state.query)
            if cached:
                answer = cached
            else:
                answer = "Unable to process massive context query at this time."
            degraded = True

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "answer": answer,
            "degraded": degraded,
            "filtered_chunks": result.final_chunks,
            "graph_context": graph_dict,
            "timings": state.timings + [{"stage": "massive_context", "duration_ms": elapsed}],
        }

    # ── Routing Logic ───────────────────────────────────────────────────

    @staticmethod
    def _after_triage(state: PipelineState) -> str:
        """Decide whether to return cached result or proceed to routing."""
        if state.cache_hit:
            return "cached"
        return "route_dispatch"

    @staticmethod
    def _route_dispatch_node(state: PipelineState) -> dict[str, Any]:
        """Pass-through node; routing happens in conditional edges."""
        return {}

    @staticmethod
    def _select_route(state: PipelineState) -> str:
        """Select the pipeline branch based on Jev route decision."""
        if state.route == "chitchat":
            return "chitchat"
        if state.route == "massive_context":
            return "massive_context"
        return "retrieve"

    # ── Public Interfaces ────────────────────────────────────────────────

    async def run(self, query: str, stream: bool = False) -> PipelineState:
        """Execute the full pipeline and return the final state."""
        t0 = time.perf_counter()
        initial_state = PipelineState(query=query, stream=stream)

        result = await self._graph.ainvoke(initial_state)

        if isinstance(result, dict):
            final = PipelineState(**{k: v for k, v in result.items() if k in PipelineState.__dataclass_fields__})
        else:
            final = result

        final.total_latency_ms = (time.perf_counter() - t0) * 1000
        return final

    async def stream_query(self, query: str) -> AsyncGenerator[dict[str, Any], None]:
        """Stream typed SSE lifecycle events ('stage', 'token', 'done')."""
        t0 = time.perf_counter()
        timings: list[dict[str, Any]] = []

        # 1. Triage Stage
        yield {"event": "stage", "data": {"stage": "triage", "status": "started"}}
        t_triage = time.perf_counter()

        # Step 1: O(1) Exact-match cache shortcut
        exact_hit = (
            self.semantic_cache.get_exact(query)
            if hasattr(type(self.semantic_cache), "get_exact")
            else None
        )
        if isinstance(exact_hit, dict):
            cached_answer = exact_hit.get("answer", "")
            route_val = exact_hit.get("route", "vector_exact")
            triage_elapsed = (time.perf_counter() - t_triage) * 1000
            timings.append({"stage": "triage", "duration_ms": triage_elapsed})
            yield {"event": "stage", "data": {"stage": "cache_hit", "route": route_val}}
            yield {"event": "token", "data": {"delta": cached_answer}}
            total_elapsed = (time.perf_counter() - t0) * 1000
            yield {
                "event": "done",
                "data": {
                    "answer": cached_answer,
                    "route": route_val,
                    "cached": True,
                    "sources": exact_hit.get("sources", []),
                    "graph_path": exact_hit.get("graph_path"),
                    "timings": timings,
                    "latency_ms": total_elapsed,
                    "degraded": False,
                },
            }
            return

        # Step 2: Concurrently execute Jev routing and CPU embedding (offloaded to thread)
        route_task = asyncio.create_task(self.router.route_query(query))
        query_embedding = await asyncio.to_thread(self.embedding_service.embed_query, query)
        cache_result = await self.semantic_cache.get(query, query_embedding)
        route_decision = await route_task
        triage_elapsed = (time.perf_counter() - t_triage) * 1000
        timings.append({"stage": "triage", "duration_ms": triage_elapsed})

        # Cache Hit short-circuit
        if cache_result is not None:
            cached_answer = cache_result.get("answer", "")
            yield {"event": "stage", "data": {"stage": "cache_hit", "route": route_decision.route}}
            yield {"event": "token", "data": {"delta": cached_answer}}
            total_elapsed = (time.perf_counter() - t0) * 1000
            yield {
                "event": "done",
                "data": {
                    "answer": cached_answer,
                    "route": route_decision.route,
                    "cached": True,
                    "sources": cache_result.get("sources", []),
                    "graph_path": cache_result.get("graph_path"),
                    "timings": timings,
                    "latency_ms": total_elapsed,
                    "degraded": False,
                },
            }
            return

        # Chitchat short-circuit
        if route_decision.route == "chitchat":
            yield {"event": "stage", "data": {"stage": "chitchat", "route": "chitchat"}}
            t_cc = time.perf_counter()
            chitchat_answer = await self.chitchat.handle(query)
            cc_elapsed = (time.perf_counter() - t_cc) * 1000
            timings.append({"stage": "chitchat", "duration_ms": cc_elapsed})

            yield {"event": "token", "data": {"delta": chitchat_answer}}
            total_elapsed = (time.perf_counter() - t0) * 1000
            yield {
                "event": "done",
                "data": {
                    "answer": chitchat_answer,
                    "route": "chitchat",
                    "cached": False,
                    "sources": [],
                    "graph_path": None,
                    "timings": timings,
                    "latency_ms": total_elapsed,
                    "degraded": False,
                },
            }
            return

        # 2. Retrieval Stage
        yield {"event": "stage", "data": {"stage": "retrieval", "route": route_decision.route}}
        t_ret = time.perf_counter()
        ret_result = await self.retrieval.execute(query, route_decision.route)
        ret_elapsed = (time.perf_counter() - t_ret) * 1000
        timings.append({"stage": "retrieval", "duration_ms": ret_elapsed})

        graph_dict = None
        if ret_result.graph_context:
            if hasattr(ret_result.graph_context, "nodes") and hasattr(ret_result.graph_context, "edges"):
                graph_dict = {
                    "nodes": ret_result.graph_context.nodes,
                    "edges": ret_result.graph_context.edges,
                }
            elif isinstance(ret_result.graph_context, dict):
                graph_dict = ret_result.graph_context

        # 3. Generation Stage
        yield {"event": "stage", "data": {"stage": "generation", "chunks_count": len(ret_result.final_chunks)}}
        t_gen = time.perf_counter()
        full_tokens: list[str] = []

        async for tok in self.synthesizer.stream(
            query=query,
            context_chunks=ret_result.final_chunks,
            graph_context=graph_dict,
            route=route_decision.route,
        ):
            full_tokens.append(tok)
            yield {"event": "token", "data": {"delta": tok}}

        degraded = self.synthesizer.circuit_status != "closed"
        gen_elapsed = (time.perf_counter() - t_gen) * 1000
        timings.append({"stage": "generation", "duration_ms": gen_elapsed})

        full_answer = "".join(full_tokens)

        # Cache answer in Tier 1 Semantic Cache
        if not degraded and query_embedding is not None:
            self.semantic_cache.put(
                query,
                query_embedding,
                {
                    "answer": full_answer,
                    "route": route_decision.route,
                    "sources": ret_result.final_chunks,
                    "graph_path": graph_dict,
                },
            )

        total_elapsed = (time.perf_counter() - t0) * 1000
        yield {
            "event": "done",
            "data": {
                "answer": full_answer,
                "route": route_decision.route,
                "cached": False,
                "sources": ret_result.final_chunks,
                "graph_path": graph_dict,
                "timings": timings,
                "latency_ms": total_elapsed,
                "degraded": degraded,
            },
        }
