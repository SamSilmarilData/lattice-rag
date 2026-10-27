"""Test-only Model Pool builder for orchestrator tests that wire fake adapters by hand."""

from __future__ import annotations

from typing import Any

from lattice_rag.generation.resilient_synthesizer import PoolMember
from lattice_rag.vocabulary import GenerationProvider


def pool_of(*, groq: Any = None, gemini: Any = None) -> tuple[PoolMember, ...]:
    """A Model Pool of one Groq adapter then one Gemini adapter, skipping absent ones.

    Each member's model is the adapter's ``model`` attribute when it is a string, else the
    provider name.
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
