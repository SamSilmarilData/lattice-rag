import functools
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from lattice_rag.security import SecretStr
from lattice_rag.vocabulary import GenerationProvider, GraphContextMode, model_label

# Load environment variables from .env file at import time
load_dotenv()

# Default answer budget (tokens) for Groq and Gemini; LLM_MAX_TOKENS overrides it.
DEFAULT_LLM_MAX_TOKENS = 768
# Per-call bound (seconds) on a Groq or Gemini request; LLM_TIMEOUT_SECONDS overrides it.
# For streaming it bounds the wait for each chunk, including the first.
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0
# onnx-community/bge-reranker-v2-m3-ONNX at its 2025-09-01 commit
DEFAULT_RERANKER_ONNX_REVISION = "6f5ff65298512715a1e669753bc754d2bc8f367b"
# Minimum Jev Noul probability for a chunk to pass the context guardrail
DEFAULT_GUARDRAIL_THRESHOLD = 0.5
# massive_context only (#48): a tie with 0.50 in the Live Tier keeps the per-route value
DEFAULT_MASSIVE_CONTEXT_GUARDRAIL_THRESHOLD = 0.35

# Model Pool defaults (ADR-0004). Free-tier quotas are per model, so the default pool
# spreads load over two Groq models before Gemini. gemini-3.5-flash-lite rather than
# gemini-3.8-flash: on 2026-10-27 3.8-flash returned 503 "high demand" on most free-tier
# calls (and once hung 6.5 minutes) while flash-lite answered normally.
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
DEFAULT_SECOND_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
# Gemini thinking value meaning "send no thinking config; use the model's own default"
THINKING_NONE = "none"
_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})


@dataclass(frozen=True)
class ModelSpec:
    """One Model Pool member as configured: a provider, a model and (Gemini only) thinking.

    ``thinking`` is a Gemini thinking setting: a token budget as digits (``"0"`` turns
    thinking off) or a level name (``"low"``). ``None`` sends no thinking config (written
    ``none`` in LLM_POOL and GEMINI_THINKING).
    """

    provider: GenerationProvider
    model: str
    thinking: str | None = None

    @property
    def label(self) -> str:
        return model_label(self.provider, self.model)


def default_gemini_thinking(model: str) -> str | None:
    """The lowest thinking setting a Gemini model accepts, so thinking spares the answer budget.

    Flash-Lite gets no thinking config: gemini-3.5-flash-lite rejects ``thinking_budget=0``
    with a 400 and, without a config, does not think (probed 2026-10-27). 2.5 Flash accepts
    a zero budget. Other Gemini 3 models take a level; ``low`` because gemini-3.8-flash
    rejects ``minimal`` with a 400 (``low`` itself is unverified: 3.8-flash was returning
    503s). Unknown models get no thinking config. Override per member with ``@<setting>`` in
    LLM_POOL, or with GEMINI_THINKING for the default pool, when the API rejects a value.
    """
    if "flash-lite" in model:
        return None
    if model.startswith("gemini-2.5-flash"):
        return "0"
    if model.startswith("gemini-3"):
        return "low"
    return None


def _thinking(raw: str, *, model: str, source: str) -> str | None:
    value = raw.strip().lower()
    if value == THINKING_NONE:
        return None
    if value.isdigit() or value in _THINKING_LEVELS:
        return value
    levels = " | ".join(sorted(_THINKING_LEVELS))
    raise ValueError(
        f"Invalid {source} thinking setting {raw!r} for {model}: use a token budget "
        f"(digits), {levels}, or {THINKING_NONE}."
    )


def _parse_pool(raw: str) -> tuple[ModelSpec, ...]:
    """Parse ``provider:model[@thinking]`` members, comma separated, in cascade order."""
    members: list[ModelSpec] = []
    for part in _split_csv(raw):
        provider_raw, sep, rest = part.partition(":")
        provider_raw = provider_raw.strip().lower()
        model, at, thinking_raw = rest.strip().partition("@")
        model = model.strip()
        if not sep or not model:
            raise ValueError(f"Invalid LLM_POOL member {part!r}: use provider:model.")
        pool_providers = GenerationProvider.pool_providers()
        provider = next((p for p in pool_providers if p == provider_raw), None)
        if provider is None:
            raise ValueError(
                f"Invalid LLM_POOL provider {provider_raw!r}: use {' | '.join(pool_providers)}."
            )
        is_gemini = provider is GenerationProvider.GEMINI
        if at and not is_gemini:
            raise ValueError(f"Invalid LLM_POOL member {part!r}: thinking is Gemini-only.")
        thinking = (
            _thinking(thinking_raw, model=model, source="LLM_POOL")
            if at
            else (default_gemini_thinking(model) if is_gemini else None)
        )
        spec = ModelSpec(provider, model, thinking)
        if any(m.label == spec.label for m in members):
            raise ValueError(f"Invalid LLM_POOL: {spec.label} is listed twice.")
        members.append(spec)
    if not members:
        raise ValueError("Invalid LLM_POOL: list at least one provider:model member.")
    return tuple(members)


def _default_pool(
    groq_model: str, gemini_model: str, gemini_thinking: str | None
) -> tuple[ModelSpec, ...]:
    thinking = (
        _thinking(gemini_thinking, model=gemini_model, source="GEMINI_THINKING")
        if gemini_thinking and gemini_thinking.strip()
        else default_gemini_thinking(gemini_model)
    )
    candidates = (
        ModelSpec(GenerationProvider.GROQ, groq_model),
        ModelSpec(GenerationProvider.GROQ, DEFAULT_SECOND_GROQ_MODEL),
        ModelSpec(GenerationProvider.GEMINI, gemini_model, thinking),
    )
    pool: list[ModelSpec] = []
    for spec in candidates:
        if all(m.label != spec.label for m in pool):
            pool.append(spec)
    return tuple(pool)


def _llm_pool(env: Mapping[str, str]) -> tuple[ModelSpec, ...]:
    raw = env.get("LLM_POOL", "")
    if raw.strip():
        return _parse_pool(raw)
    return _default_pool(
        env.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL,
        env.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        env.get("GEMINI_THINKING"),
    )


@dataclass(frozen=True)
class AppConfig:
    """
    Application configuration dataclass.
    Frozen so configuration is immutable after initialization.
    """

    typesafe_api_key: SecretStr
    groq_api_key: SecretStr
    gemini_api_key: SecretStr
    admin_api_key: SecretStr = SecretStr("")
    redis_url: str = "embedded"
    latticedb_path: Path = Path("data/lattice_rag.db")
    host: str = "0.0.0.0"
    port: int = 8000
    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # INT8 ONNX export used when reranker_model is bge-reranker-v2-m3, pinned to a commit so
    # the downloaded weights cannot drift under the Hermetic Tier floors.
    reranker_onnx_repo: str = "onnx-community/bge-reranker-v2-m3-ONNX"
    reranker_onnx_revision: str | None = DEFAULT_RERANKER_ONNX_REVISION
    # Model Pool (ADR-0004): every Route tries these members in order. LLM_POOL sets it;
    # without it the pool is GROQ_MODEL, openai/gpt-oss-120b, then GEMINI_MODEL.
    llm_pool: tuple[ModelSpec, ...] = (
        ModelSpec(GenerationProvider.GROQ, DEFAULT_GROQ_MODEL),
        ModelSpec(GenerationProvider.GROQ, DEFAULT_SECOND_GROQ_MODEL),
        ModelSpec(GenerationProvider.GEMINI, DEFAULT_GEMINI_MODEL),
    )
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    cors_allow_origins: tuple[str, ...] = ()
    query_rate_limit_per_minute: int = 60
    # Reverse proxies in front of the app whose X-Forwarded-For entries are trusted.
    # 0 identifies clients by the socket peer and ignores the header.
    trusted_proxy_hops: int = 0
    # Chunks whose cross-encoder rerank_score (raw logit) is at or above this value skip the
    # Jev Noul guardrail. None (the default) sends every chunk to Noul. Enable only with
    # Live Tier results; see docs/production-readiness.md.
    guardrail_skip_rerank_score: float | None = None
    # Experiment Arm switches (spec #22; docs/production-readiness.md §6). Defaults are the
    # kept configurations. Graph triples in the generation prompt: on | off | relational;
    # off since #47 found no gain (the traversal still runs for the graph path)
    graph_context_in_prompt: GraphContextMode = GraphContextMode.OFF
    # Noul threshold for massive_context retrieval only; the other Routes keep the global
    # one. 0.35 since #48 (a tie with 0.50 keeps the per-route threshold)
    guardrail_massive_context_threshold: float = DEFAULT_MASSIVE_CONTEXT_GUARDRAIL_THRESHOLD
    # Skip Noul on massive_context: its answers become Unverified and never reach Tier 1.
    # Wins over GUARDRAIL_MASSIVE_CONTEXT_THRESHOLD.
    guardrail_skip_massive_context: bool = False
    # Live Tier: seconds to wait before the single retry of a degraded answer (no Model Pool
    # member answered), so rate limits or open breakers can clear
    eval_degraded_retry_delay: float = 5.0
    # True when the app booted without TYPESAFE_API_KEY and runs in degraded hermetic mode
    using_mock_keys: bool = False

    def __post_init__(self) -> None:
        """Validate config after initialization."""
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Invalid port: {self.port}. Must be between 1 and 65535.")
        if self.trusted_proxy_hops < 0:
            raise ValueError(
                f"Invalid TRUSTED_PROXY_HOPS: {self.trusted_proxy_hops}. Must be 0 or greater."
            )
        if not 0.0 <= self.guardrail_massive_context_threshold <= 1.0:
            raise ValueError(
                "Invalid GUARDRAIL_MASSIVE_CONTEXT_THRESHOLD: "
                f"{self.guardrail_massive_context_threshold}. Must be between 0 and 1."
            )
        if self.llm_timeout_seconds <= 0:
            raise ValueError(
                f"Invalid LLM_TIMEOUT_SECONDS: {self.llm_timeout_seconds}. Must be greater than 0."
            )
        if self.eval_degraded_retry_delay < 0:
            raise ValueError(
                f"Invalid EVAL_DEGRADED_RETRY_DELAY: {self.eval_degraded_retry_delay}. "
                "Must be 0 or greater."
            )


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _optional_float(raw: str | None) -> float | None:
    return float(raw) if raw and raw.strip() else None


_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
# Unset or empty counts as false
_FALSY_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


def _bool(raw: str | None, *, name: str) -> bool:
    value = (raw or "").strip().lower()
    if value in _TRUTHY_ENV_VALUES:
        return True
    if value in _FALSY_ENV_VALUES:
        return False
    raise ValueError(f"Invalid {name}: {value!r}. Use true or false.")


def _graph_context_mode(raw: str | None) -> GraphContextMode:
    value = (raw or GraphContextMode.OFF).strip().lower()
    try:
        return GraphContextMode(value)
    except ValueError:
        choices = " | ".join(m.value for m in GraphContextMode)
        raise ValueError(f"Invalid GRAPH_CONTEXT_IN_PROMPT: {value!r}. Use {choices}.") from None


def load_config(require_typesafe: bool = True) -> AppConfig:
    """
    Reads os.environ and constructs AppConfig.
    Raises SystemExit if TYPESAFE_API_KEY is missing and require_typesafe is True.
    Storage-only commands (seed, ingest) pass require_typesafe=False.
    """
    typesafe_key = os.environ.get("TYPESAFE_API_KEY", "")
    if require_typesafe and not typesafe_key:
        raise SystemExit("Error: TYPESAFE_API_KEY environment variable is required.")

    return AppConfig(
        typesafe_api_key=SecretStr(typesafe_key),
        groq_api_key=SecretStr(os.environ.get("GROQ_API_KEY", "")),
        gemini_api_key=SecretStr(os.environ.get("GEMINI_API_KEY", "")),
        admin_api_key=SecretStr(os.environ.get("ADMIN_API_KEY", "")),
        redis_url=os.environ.get("REDIS_URL", "embedded"),
        latticedb_path=Path(os.environ.get("LATTICEDB_PATH", "data/lattice_rag.db")),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        embed_model=os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker_model=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        reranker_onnx_repo=os.environ.get(
            "RERANKER_ONNX_REPO", "onnx-community/bge-reranker-v2-m3-ONNX"
        ),
        reranker_onnx_revision=(
            os.environ.get("RERANKER_ONNX_REVISION") or DEFAULT_RERANKER_ONNX_REVISION
        ),
        llm_pool=_llm_pool(os.environ),
        llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", str(DEFAULT_LLM_MAX_TOKENS))),
        llm_timeout_seconds=float(
            os.environ.get("LLM_TIMEOUT_SECONDS") or DEFAULT_LLM_TIMEOUT_SECONDS
        ),
        cors_allow_origins=_split_csv(os.environ.get("CORS_ALLOW_ORIGINS", "")),
        query_rate_limit_per_minute=int(os.environ.get("QUERY_RATE_LIMIT_PER_MINUTE", "60")),
        trusted_proxy_hops=int(os.environ.get("TRUSTED_PROXY_HOPS", "0")),
        guardrail_skip_rerank_score=_optional_float(os.environ.get("GUARDRAIL_SKIP_RERANK_SCORE")),
        graph_context_in_prompt=_graph_context_mode(os.environ.get("GRAPH_CONTEXT_IN_PROMPT")),
        guardrail_massive_context_threshold=float(
            os.environ.get("GUARDRAIL_MASSIVE_CONTEXT_THRESHOLD")
            or DEFAULT_MASSIVE_CONTEXT_GUARDRAIL_THRESHOLD
        ),
        guardrail_skip_massive_context=_bool(
            os.environ.get("GUARDRAIL_SKIP_MASSIVE_CONTEXT"), name="GUARDRAIL_SKIP_MASSIVE_CONTEXT"
        ),
        eval_degraded_retry_delay=float(os.environ.get("EVAL_DEGRADED_RETRY_DELAY") or "5"),
    )


@functools.lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Module-level cached singleton for AppConfig.
    Constructs the config only once on first call.
    """
    return load_config()
