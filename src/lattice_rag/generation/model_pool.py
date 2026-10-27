"""Model Pool construction: the one place both composition roots turn config into members."""

from __future__ import annotations

from lattice_rag.api.dtos import GenerationProvider
from lattice_rag.config import AppConfig
from lattice_rag.generation import gemini_fallback, groq_synthesizer
from lattice_rag.generation.resilient_synthesizer import PoolMember


def build_model_pool(config: AppConfig) -> tuple[PoolMember, ...]:
    """One provider adapter per ``config.llm_pool`` member, in cascade order (ADR-0004).

    Every member gets the shared answer budget and per-call timeout. A provider without a
    key still gets its members: their calls raise, so the pool falls through to the next.
    """
    groq_key = config.groq_api_key.get_secret_value() if config.groq_api_key else None
    gemini_key = config.gemini_api_key.get_secret_value() if config.gemini_api_key else None
    members: list[PoolMember] = []
    for spec in config.llm_pool:
        client: groq_synthesizer.GroqSynthesizer | gemini_fallback.GeminiFallback
        if spec.provider == GenerationProvider.GROQ:
            client = groq_synthesizer.GroqSynthesizer(
                api_key=groq_key,
                model=spec.model,
                max_tokens=config.llm_max_tokens,
                timeout_seconds=config.llm_timeout_seconds,
            )
        else:
            client = gemini_fallback.GeminiFallback(
                api_key=gemini_key,
                model=spec.model,
                max_output_tokens=config.llm_max_tokens,
                timeout_seconds=config.llm_timeout_seconds,
                thinking=spec.thinking,
            )
        members.append(PoolMember(GenerationProvider(spec.provider), spec.model, client))
    return tuple(members)
