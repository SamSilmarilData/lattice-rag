"""Closed domain vocabularies shared by the Gold Set, the eval report and its wire DTOs.

Kept dependency-free so ``api.dtos`` and ``eval`` can both import it without a cycle.
"""

from __future__ import annotations

from typing import Literal

Slice = Literal["factoid", "multihop", "unanswerable", "chitchat", "long_context"]
Route = Literal["vector_exact", "graph_relational", "hybrid", "chitchat", "massive_context"]
