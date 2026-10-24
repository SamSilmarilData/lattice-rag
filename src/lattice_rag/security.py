import hashlib
import logging
import os
import re
from pathlib import Path

import structlog

# Matches common API-key prefixes, bearer tokens, token query params, and URI credentials
# (including password-only forms such as redis://:secret@host).
REDACT_PATTERN = re.compile(
    r"(?:sk-[a-zA-Z0-9_-]+|gsk_[a-zA-Z0-9_-]+|AIza[a-zA-Z0-9_-]+|Bearer\s+[a-zA-Z0-9_.-]+"
    r"|token=[a-zA-Z0-9_.-]+|://[^:/@\s]*:[^@\s]+@)"
)

_ENV_KEY_PATTERN = re.compile(
    r"(TYPESAFE_API_KEY|GROQ_API_KEY|GEMINI_API_KEY|REDIS_URL|ADMIN_API_KEY)\s*[:=]\s*([^\s`]+)"
)


def query_fingerprint(query: str) -> dict[str, int | str]:
    """Correlation fields for logging a user query without its text.

    Returns the query's length and a short SHA-256 prefix, so log lines about the
    same query can be grouped while the query itself stays out of logs above DEBUG.
    """
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    return {"query_len": len(query), "query_hash": digest}


class SecretStr:
    """A wrapper around a string to mask secret values in logs and outputs."""

    def __init__(self, value: str):
        """
        Initialize the SecretStr. Works with msgspec as long as it handles
        single-argument string constructors.
        """
        self._value = value

    def __str__(self) -> str:
        return self._mask()

    def __repr__(self) -> str:
        return f"SecretStr('{self._mask()}')"

    def __bool__(self) -> bool:
        """Truthy only when a non-empty secret is present."""
        return bool(self._value)

    def get_secret_value(self) -> str:
        """Returns the actual unmasked secret string."""
        return self._value

    def _mask(self) -> str:
        """Masks the secret if it matches common prefixes, otherwise returns [REDACTED]."""
        if not self._value:
            return "[REDACTED]"

        prefixes = ("sk-", "gsk_", "AIza")
        for prefix in prefixes:
            if self._value.startswith(prefix):
                if len(self._value) > len(prefix) + 4:
                    return f"{prefix}****{self._value[-4:]}"
                return f"{prefix}****"
        return "[REDACTED]"


def sanitize_error_detail(detail: str) -> str:
    """
    Strips potential auth tokens/headers from error message strings.
    """
    if not detail:
        return detail
    return REDACT_PATTERN.sub("[REDACTED]", detail)


class RedactingFilter(logging.Filter):
    """
    A logging filter that scans log records and redacts patterns matching API
    key prefixes (sk-, gsk_, AIza, Bearer, token=) and URI credentials.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_error_detail(record.msg)

        if record.args and isinstance(record.args, tuple):
            record.args = tuple(
                sanitize_error_detail(arg) if isinstance(arg, str) else arg for arg in record.args
            )

        return True


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts secrets from the fully rendered line, including tracebacks."""

    def __init__(
        self,
        fmt: str | None = "%(levelname)s - %(asctime)s - %(name)s - %(module)s - %(message)s",
        datefmt: str | None = None,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        return sanitize_error_detail(super().format(record))


def configure_structlog() -> None:
    """Route structlog through stdlib logging so stdlib handlers and redaction apply to it."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def configure_logging(level: int = logging.INFO) -> None:
    """Install redacting stdlib logging plus structlog routing for CLI entry points.

    The web app configures stdlib handlers through Litestar's LoggingConfig instead
    (see ``app.py``) and only calls :func:`configure_structlog`.
    """
    root = logging.getLogger()
    if not any(isinstance(h.formatter, RedactingFormatter) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(RedactingFormatter())
        handler.addFilter(RedactingFilter())
        root.addHandler(handler)
    root.setLevel(level)
    configure_structlog()


def migrate_apikeys_to_dotenv(apikeys_path: Path, dotenv_path: Path) -> bool:
    """
    Reads apikeys.md, extracts lines matching known key patterns, appends any keys
    not already present to the .env file (created with 0o600 permissions), and
    deletes apikeys.md.

    Returns:
        bool: True if migration occurred, False if there was nothing to migrate.
        apikeys.md is left untouched when no keys are recognized.
    """
    if not apikeys_path.exists():
        return False

    matches = _ENV_KEY_PATTERN.findall(apikeys_path.read_text(encoding="utf-8"))
    if not matches:
        return False

    existing_keys: set[str] = set()
    if dotenv_path.exists():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            key, sep, _ = line.partition("=")
            if sep:
                existing_keys.add(key.strip())

    env_lines = [f"{k}={v}" for k, v in matches if k not in existing_keys]

    # Create with owner-only permissions from the start rather than chmod-after-write
    fd = os.open(dotenv_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        if env_lines:
            f.write("\n" + "\n".join(env_lines) + "\n")
    dotenv_path.chmod(0o600)

    apikeys_path.unlink()
    return True
