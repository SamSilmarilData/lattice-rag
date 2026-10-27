"""Closed domain vocabularies shared by the Gold Set, generation, the eval report and the
wire DTOs.

Kept dependency-free so ``api.dtos``, ``config``, ``generation`` and ``eval`` can all
import it without a cycle (and without generation importing the API layer).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

Slice = Literal["factoid", "multihop", "unanswerable", "chitchat", "long_context"]
# How a Gold Set query is posed, orthogonal to its Slice: ``direct`` asks in the corpus's
# own words; the others stress retrieval with rewording, negation, an ambiguous referent,
# Documents that disagree, or a question that sits just outside what the corpus says.
Case = Literal[
    "direct",
    "paraphrase",
    "negation",
    "ambiguity",
    "conflicting_sources",
    "unanswerable_near_miss",
]
# ADR-0003 collapsed vector_exact, graph_relational and hybrid into the one retrieval
# Route, ``hybrid``: they ran the same retrieval and differed only in graph hop count.
Route = Literal["hybrid", "massive_context", "chitchat"]


class GraphContextMode(StrEnum):
    """``GRAPH_CONTEXT_IN_PROMPT``: which graph context reaches the generation prompt (#47).

    ``on`` sends the whole traversed subgraph on ``hybrid`` and ``massive_context``, ``off`` none,
    and ``relational`` sends it only on the retrieval Route (``hybrid``), capped at ten
    edges (``generation.prompt.graph_for_prompt``). ``off`` is the default: #47 measured no
    gain from graph context, and the traversal still feeds the API's graph path.
    """

    ON = "on"
    OFF = "off"
    RELATIONAL = "relational"


class GenerationProvider(StrEnum):
    """Who generated an answer: a Groq or Gemini Model Pool member, the Tier 2 FAQ, or the
    offline message."""

    GROQ = "groq"
    GEMINI = "gemini"
    REDIS_FAQ = "redis_faq"
    OFFLINE = "offline"

    @classmethod
    def pool_providers(cls) -> tuple[GenerationProvider, ...]:
        """The providers a Model Pool member (``LLM_POOL``) can name, in display order."""
        return (cls.GROQ, cls.GEMINI)


def model_label(provider: GenerationProvider, model: str) -> str:
    """``provider:model``, the one label for a Model Pool member (config, breakers, logs)."""
    return f"{provider}:{model}"


def answered_by(provider: GenerationProvider | None, model: str | None) -> str | None:
    """Who answered: ``provider:model`` for a pool member, the provider alone for the
    FAQ/offline message, ``None`` when nothing generated (cache hit, chitchat)."""
    if provider is None:
        return None
    return model_label(provider, model) if model else str(provider)
