from __future__ import annotations

import os
from dataclasses import dataclass

import structlog
from typesafe_sdk import AsyncTypeSafeClient, Choice

from lattice_rag.generation.chitchat import is_pure_chitchat
from lattice_rag.security import query_fingerprint, sanitize_error_detail

logger = structlog.get_logger(__name__)


_ROUTER_INSTRUCTIONS = (
    "Classify the user query into the single most appropriate routing category "
    "for a Hybrid GraphRAG system."
)
_ROUTER_CRITERIA = {
    "vector_exact": "Precise factual lookups, e.g., 'What is X?', 'Define Y'.",
    "graph_relational": "Multi-hop relational queries, e.g., 'How does X relate to Y?', 'What connects A to B?'.",
    "hybrid": "Complex queries needing both vector and graph context, e.g., 'Explain the relationship between X, Y, and Z in the context of W'.",
    "chitchat": "Greetings, small talk, non-technical conversation.",
    "massive_context": "Queries requiring cross-document synthesis, e.g., 'Summarize all findings about...', 'Compare everything we know about...'.",
}


@dataclass(frozen=True)
class RouteDecision:
    route: str
    confidence: float
    # False when the route is a fallback because the Jev check could not run
    verified: bool = True


# System 1 fallback when Jev could not choose a route (an Unverified Answer follows)
_FALLBACK = RouteDecision(route="hybrid", confidence=0.5, verified=False)


class QueryRouter:
    """Routes user queries to the appropriate RAG strategy using TypeSafe AI Jev Choice."""

    def __init__(
        self, ts_client: AsyncTypeSafeClient | None = None, *, offline: bool = False
    ) -> None:
        """Initializes the QueryRouter and its TypeSafe AI client.

        Args:
            ts_client: Optional injected AsyncTypeSafeClient (useful for testing).
            offline: Mock-key mode: no Jev client at all. Pure small talk still takes the
                deterministic chitchat path; every other query falls back to ``hybrid`` as
                an unverified route without any network call.

        Raises:
            ValueError: If ``offline`` is combined with a ``ts_client``.
            RuntimeError: If not offline, TYPESAFE_API_KEY is not set and ts_client is None.
        """
        self._client: AsyncTypeSafeClient | None
        if offline:
            if ts_client is not None:
                raise ValueError("An offline QueryRouter takes no ts_client.")
            self._client = None
            logger.debug("QueryRouter initialized offline")
            return
        if ts_client is not None:
            self._client = ts_client
        else:
            if not os.environ.get("TYPESAFE_API_KEY"):
                raise RuntimeError(
                    "TYPESAFE_API_KEY environment variable is not set. "
                    "Please set it to use the QueryRouter."
                )
            self._client = AsyncTypeSafeClient()
        logger.debug("QueryRouter initialized")

    @classmethod
    def offline(cls) -> QueryRouter:
        """A router with no Jev client (mock-key mode); see ``offline`` in ``__init__``."""
        return cls(offline=True)

    async def route_query(self, query: str) -> RouteDecision:
        """Routes a query into one of the supported strategies.

        Args:
            query: The user query to route.

        Returns:
            RouteDecision: The chosen route and confidence score.
        """
        if not query or not query.strip():
            logger.debug("empty_query_routed_to_chitchat")
            return RouteDecision(route="chitchat", confidence=1.0)

        # Fast-path: messages that are entirely small talk skip the Jev call
        if is_pure_chitchat(query):
            logger.debug("deterministic_chitchat_fast_path", query=query)
            return RouteDecision(route="chitchat", confidence=1.0)

        if self._client is None:
            return _FALLBACK

        logger.debug("Routing query", query=query)
        try:
            response = await self._client.system_one(
                state={"query": query},
                questions={
                    "route": Choice(
                        instructions=_ROUTER_INSTRUCTIONS,
                        criteria=_ROUTER_CRITERIA,
                    ),
                },
            )

            choice_result = response.choices.get("route")
            if choice_result is None:
                logger.warning("missing_route_choice", **query_fingerprint(query))
                return _FALLBACK

            chosen_route = str(choice_result.choice).strip().lower()
            if chosen_route not in _ROUTER_CRITERIA:
                logger.warning("unknown_route_choice", choice=chosen_route, fallback="hybrid")
                return _FALLBACK

            confidence = float(getattr(choice_result, "confidence", 1.0))
            decision = RouteDecision(
                route=chosen_route,
                confidence=confidence,
            )
            logger.info("Query routed", route=decision.route, confidence=decision.confidence)
            return decision
        except Exception as e:
            logger.warning(
                "query_routing_failed_falling_back_to_hybrid",
                **query_fingerprint(query),
                error=sanitize_error_detail(str(e)),
            )
            return _FALLBACK

    async def aclose(self) -> None:
        """Closes the underlying TypeSafe AI client."""
        if self._client is not None:
            await self._client.aclose()
        logger.debug("QueryRouter closed")

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()
