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

_WORD = re.compile(r"[a-z']+")
_MAX_CHITCHAT_WORDS = 8
# Every word of a pure chitchat message must come from this vocabulary. Anything else
# (a topic, a technical term) means the user is asking a real question.
_CHITCHAT_VOCABULARY = frozenset(
    {
        "hello", "hi", "hey", "greetings", "good", "morning", "afternoon", "evening",
        "bye", "goodbye", "farewell", "cya", "see", "later", "thanks", "thank", "thx",
        "appreciate", "it", "help", "assist", "what", "can", "do", "you", "me", "there",
        "everyone", "team", "assistant", "so", "much", "a", "lot", "for", "now", "again",
        "really", "your", "the", "to", "ok", "okay", "cool", "great", "very", "all",
    }
)  # fmt: skip


def is_pure_chitchat(query: str) -> bool:
    """True only when the whole message is small talk, not a question that contains a greeting.

    "hi!" and "what can you do?" qualify; "How does HNSW help recall?" does not.
    """
    if not _CHITCHAT_AUTOMATON.search(query):
        return False
    words = _WORD.findall(query.lower())
    return 0 < len(words) <= _MAX_CHITCHAT_WORDS and all(w in _CHITCHAT_VOCABULARY for w in words)


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
