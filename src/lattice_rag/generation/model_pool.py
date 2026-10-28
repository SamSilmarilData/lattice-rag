"""Model Pool construction: the one place both composition roots turn config into members."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lattice_rag.config import AppConfig, ModelSpec

# Module access (not ``from ... import``) so tests can patch an adapter class
from lattice_rag.generation import gemini_synthesizer, groq_synthesizer
from lattice_rag.generation.resilient_synthesizer import PoolMember
from lattice_rag.vocabulary import GenerationProvider


def _groq(spec: ModelSpec, config: AppConfig) -> groq_synthesizer.GroqSynthesizer:
    return groq_synthesizer.GroqSynthesizer(
        api_key=config.groq_api_key.get_secret_value() if config.groq_api_key else None,
        model=spec.model,
        max_tokens=config.llm_max_tokens,
        timeout_seconds=config.llm_timeout_seconds,
    )


def _gemini(spec: ModelSpec, config: AppConfig) -> gemini_synthesizer.GeminiSynthesizer:
    return gemini_synthesizer.GeminiSynthesizer(
        api_key=config.gemini_api_key.get_secret_value() if config.gemini_api_key else None,
        model=spec.model,
        max_output_tokens=config.llm_max_tokens,
        timeout_seconds=config.llm_timeout_seconds,
        thinking=spec.thinking,
    )


# The provider adapter for each Model Pool provider (``GenerationProvider.pool_providers()``)
ADAPTERS: dict[GenerationProvider, Callable[[ModelSpec, AppConfig], Any]] = {
    GenerationProvider.GROQ: _groq,
    GenerationProvider.GEMINI: _gemini,
}


def build_model_pool(config: AppConfig) -> tuple[PoolMember, ...]:
    """One provider adapter per ``config.llm_pool`` member, in cascade order (ADR-0004).

    Every member gets the shared answer budget and per-call timeout. A provider without a
    key still gets its members: their calls raise, so the pool falls through to the next.
    """
    return tuple(
        PoolMember(spec.provider, spec.model, ADAPTERS[spec.provider](spec, config))
        for spec in config.llm_pool
    )
