"""Closed domain vocabularies shared by the Gold Set, the eval report and its wire DTOs.

Kept dependency-free so ``api.dtos`` and ``eval`` can both import it without a cycle.
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
Route = Literal["vector_exact", "graph_relational", "hybrid", "chitchat", "massive_context"]


class GraphContextMode(StrEnum):
    """``GRAPH_CONTEXT_IN_PROMPT``: which graph context reaches the generation prompt (#47).

    ``on`` sends the whole traversed subgraph on every retrieval Route, ``off`` sends none,
    and ``relational`` sends it only on ``graph_relational`` and ``hybrid``, capped at ten
    edges (``generation.prompt.graph_for_prompt``).
    """

    ON = "on"
    OFF = "off"
    RELATIONAL = "relational"
