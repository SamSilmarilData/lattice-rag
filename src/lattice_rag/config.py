import functools
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from lattice_rag.security import SecretStr

# Load environment variables from .env file at import time
load_dotenv()


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
    # INT8 ONNX export used when reranker_model is bge-reranker-v2-m3. Set the revision to a
    # commit hash to pin the downloaded weights.
    reranker_onnx_repo: str = "onnx-community/bge-reranker-v2-m3-ONNX"
    reranker_onnx_revision: str | None = None
    groq_model: str = "qwen/qwen3.8-27b"
    gemini_model: str = "gemini-3.8-flash"
    llm_max_tokens: int = 256
    cors_allow_origins: tuple[str, ...] = ()
    query_rate_limit_per_minute: int = 60
    # Reverse proxies in front of the app whose X-Forwarded-For entries are trusted.
    # 0 identifies clients by the socket peer and ignores the header.
    trusted_proxy_hops: int = 0
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


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


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
        reranker_onnx_revision=os.environ.get("RERANKER_ONNX_REVISION") or None,
        groq_model=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
        llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
        cors_allow_origins=_split_csv(os.environ.get("CORS_ALLOW_ORIGINS", "")),
        query_rate_limit_per_minute=int(os.environ.get("QUERY_RATE_LIMIT_PER_MINUTE", "60")),
        trusted_proxy_hops=int(os.environ.get("TRUSTED_PROXY_HOPS", "0")),
    )


@functools.lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Module-level cached singleton for AppConfig.
    Constructs the config only once on first call.
    """
    return load_config()
