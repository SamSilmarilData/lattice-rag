from __future__ import annotations

import re
import structlog

logger = structlog.get_logger(__name__)


class ChitchatHandler:
    """Instant deterministic handler for chitchat queries. Zero LLM cost."""

    responses: dict[str, str] = {
        "greetings": "Hello! I am lattice-rag, a Hybrid GraphRAG engine. How can I help you with your knowledge base today?",
        "farewell": "Goodbye! Feel free to come back anytime you need help exploring your knowledge graph.",
        "thanks": "You are welcome! Let me know if you need anything else.",
        "help": "I can help you search through documents, explore entity relationships, and answer complex multi-hop questions. Try asking me a specific question about your ingested documents!",
        "default": "I am a specialized knowledge engine. Could you please ask me a question about your documents or knowledge base?",
    }

    _PATTERNS = {
        "greetings": [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreetings\b", r"\bmorning\b"],
        "farewell": [r"\bbye\b", r"\bgoodbye\b", r"\bfarewell\b", r"\bcya\b"],
        "thanks": [r"\bthanks\b", r"\bthank you\b", r"\bthx\b", r"\bappreciate\b"],
        "help": [r"\bhelp\b", r"\bwhat can you do\b", r"\bassist\b"],
    }

    async def handle(self, query: str) -> str:
        """Pattern-matches query against common chitchat patterns."""
        logger.debug("Handling potential chitchat query", query=query)
        q = query.lower()

        for category, patterns in self._PATTERNS.items():
            if any(re.search(p, q) for p in patterns):
                return self.responses[category]

        return self.responses["default"]
