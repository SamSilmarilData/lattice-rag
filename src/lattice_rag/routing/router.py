from __future__ import annotations

import os
from dataclasses import dataclass

import structlog
from typesafe_sdk import AsyncTypeSafeClient, Choice

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


class QueryRouter:
    """Routes user queries to the appropriate RAG strategy using TypeSafe AI Jev Choice."""

    def __init__(self, ts_client: AsyncTypeSafeClient | None = None) -> None:
        """Initializes the QueryRouter and its TypeSafe AI client.

        Args:
            ts_client: Optional injected AsyncTypeSafeClient (useful for testing).

        Raises:
            RuntimeError: If TYPESAFE_API_KEY environment variable is not set and ts_client is None.
        """
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

    async def route_query(self, query: str) -> RouteDecision:
        """Routes a query into one of the supported strategies.

        Args:
            query: The user query to route.

        Returns:
            RouteDecision: The chosen route and confidence score.
        """
        logger.info("Routing query", query=query)
        response = await self._client.system_one(
            state={"query": query},
            questions={
                "route": Choice(
                    instructions=_ROUTER_INSTRUCTIONS,
                    criteria=_ROUTER_CRITERIA,
                ),
            },
        )

        choice_result = response.choices["route"]
        decision = RouteDecision(
            route=choice_result.choice,
            confidence=choice_result.confidence
        )
        logger.info("Query routed", route=decision.route, confidence=decision.confidence)
        return decision

    async def aclose(self) -> None:
        """Closes the underlying TypeSafe AI client."""
        await self._client.aclose()
        logger.debug("QueryRouter closed")

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()

