import os
import functools
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
    redis_url: str = "embedded"
    latticedb_path: Path = Path('data/lattice_rag.db')
    host: str = '0.0.0.0'
    port: int = 8000
    embed_model: str = 'BAAI/bge-small-en-v1.5'
    reranker_model: str = 'BAAI/bge-reranker-v2-m3'
    groq_model: str = 'qwen/qwen3.8-27b'
    gemini_model: str = 'gemini-3.8-flash'

    def __post_init__(self) -> None:
        """Validate config after initialization."""
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Invalid port: {self.port}. Must be between 1 and 65535.")



def load_config() -> AppConfig:
    """
    Reads os.environ and constructs AppConfig.
    Raises SystemExit if TYPESAFE_API_KEY is missing.
    """
    typesafe_key = os.environ.get("TYPESAFE_API_KEY")
    if not typesafe_key:
        raise SystemExit("Error: TYPESAFE_API_KEY environment variable is required.")

    return AppConfig(
        typesafe_api_key=SecretStr(typesafe_key),
        groq_api_key=SecretStr(os.environ.get("GROQ_API_KEY", "")),
        gemini_api_key=SecretStr(os.environ.get("GEMINI_API_KEY", "")),
        redis_url=os.environ.get("REDIS_URL", "embedded"),
        latticedb_path=Path(os.environ.get("LATTICEDB_PATH", "data/lattice_rag.db")),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        embed_model=os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker_model=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        groq_model=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
    )


@functools.lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Module-level cached singleton for AppConfig.
    Constructs the config only once on first call.
    """
    return load_config()
