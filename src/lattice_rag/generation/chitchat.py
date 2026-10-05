from __future__ import annotations

import re
import structlog

logger = structlog.get_logger(__name__)


_CHITCHAT_AUTOMATON = re.compile(
    r"(?P<greetings>\b(?:hello|hi|hey|greetings|morning)\b)|"
    r"(?P<farewell>\b(?:bye|goodbye|farewell|cya)\b)|"
    r"(?P<thanks>\b(?:thanks|thank you|thx|appreciate)\b)|"
    r"(?P<help>\b(?:help|what can you do|assist)\b)",
    re.IGNORECASE,
)


class ChitchatHandler:
    """Instant deterministic handler for chitchat queries. Zero LLM cost."""

    responses: dict[str, str] = {
        "greetings": "Hello! I am lattice-rag, a Hybrid GraphRAG engine. How can I help you with your knowledge base today?",
        "farewell": "Goodbye! Feel free to come back anytime you need help exploring your knowledge graph.",
        "thanks": "You are welcome! Let me know if you need anything else.",
        "help": "I can help you search through documents, explore entity relationships, and answer complex multi-hop questions. Try asking me a specific question about your ingested documents!",
        "default": "I am a specialized knowledge engine. Could you please ask me a question about your documents or knowledge base?",
    }

    async def handle(self, query: str) -> str:
        """Pattern-matches query against common chitchat patterns in a single regex pass."""
        logger.debug("Handling potential chitchat query", query=query)
        match = _CHITCHAT_AUTOMATON.search(query)
        if match and match.lastgroup:
            return self.responses.get(match.lastgroup, self.responses["default"])
        return self.responses["default"]
