"""Unit tests for secret security, masking, and credential migration."""
from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from lattice_rag.security import (
    RedactingFilter,
    SecretStr,
    migrate_apikeys_to_dotenv,
    sanitize_error_detail,
)


def test_secret_str_masking():
    """Verify SecretStr masks raw tokens in str() and repr() while preserving value."""
    # Test OpenAI-style prefix
    s1 = SecretStr("sk-live_1234567890abcdef")
    assert str(s1) == "sk-****cdef"
    assert repr(s1) == "SecretStr('sk-****cdef')"
    assert s1.get_secret_value() == "sk-live_1234567890abcdef"

    # Test Groq-style prefix
    s2 = SecretStr("gsk_secret_token_value_9999")
    assert str(s2) == "gsk_****9999"
    assert s2.get_secret_value() == "gsk_secret_token_value_9999"

    # Test Google-style prefix
    s3 = SecretStr("AIzaSyD1234567890_test_key_0001")
    assert str(s3) == "AIza****0001"

    # Test non-standard secret
    s4 = SecretStr("custom_secret_password_here")
    assert str(s4) == "[REDACTED]"
    assert s4.get_secret_value() == "custom_secret_password_here"


def test_redacting_filter():
    """Verify logging filter redacts tokens from log messages."""
    filter_ = RedactingFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Connecting with Bearer sk-1234567890abcdef and gsk_token1234",
        args=(),
        exc_info=None,
    )
    filter_.filter(record)
    assert "sk-" not in record.msg
    assert "gsk_" not in record.msg
    assert "[REDACTED]" in record.msg


def test_sanitize_error_detail():
    """Verify auth headers and credentials are scrubbed from error details."""
    raw = "Request failed: Bearer sk-proj_987654321, token=super_secret_token"
    cleaned = sanitize_error_detail(raw)
    assert "sk-proj" not in cleaned
    assert "super_secret" not in cleaned
    assert "[REDACTED]" in cleaned


def test_migrate_apikeys_to_dotenv(tmp_path: Path):
    """Verify migration extracts keys, writes .env with 0600 permissions, and deletes plaintext apikeys.md."""
    apikeys_file = tmp_path / "apikeys.md"
    dotenv_file = tmp_path / ".env"

    apikeys_content = """# My API Keys
TYPESAFE_API_KEY=ts_live_key_1111
GROQ_API_KEY: gsk_groq_key_2222
GEMINI_API_KEY = AIzaSy_gemini_3333
REDIS_URL=redis://localhost:6379/0
"""
    apikeys_file.write_text(apikeys_content, encoding="utf-8")

    result = migrate_apikeys_to_dotenv(apikeys_path=apikeys_file, dotenv_path=dotenv_file)
    assert result is True

    # 1. Verify plaintext file is securely deleted
    assert not apikeys_file.exists()

    # 2. Verify .env exists and contains the keys
    assert dotenv_file.exists()
    dotenv_text = dotenv_file.read_text(encoding="utf-8")
    assert "TYPESAFE_API_KEY=ts_live_key_1111" in dotenv_text
    assert "GROQ_API_KEY=gsk_groq_key_2222" in dotenv_text
    assert "GEMINI_API_KEY=AIzaSy_gemini_3333" in dotenv_text

    # 3. Verify file permissions are 0o600 (owner read/write only)
    file_stat = os.stat(dotenv_file)
    file_mode = stat.S_IMODE(file_stat.st_mode)
    assert file_mode == 0o600


def test_secret_str_truthiness():
    """Empty secrets are falsy so `if config.groq_api_key` reflects configuration."""
    assert not SecretStr("")
    assert SecretStr("gsk_x")


def test_redaction_covers_password_only_uri():
    """redis://:password@host has no username and must still be redacted."""
    cleaned = sanitize_error_detail("connecting to redis://:hunter2pass@cache:6379/0")
    assert "hunter2pass" not in cleaned


def test_redacting_formatter_scrubs_structlog_and_tracebacks(caplog: pytest.LogCaptureFixture):
    """structlog output and exception tracebacks are redacted once rendered."""
    import structlog

    from lattice_rag.security import RedactingFormatter, configure_structlog

    configure_structlog()
    formatter = RedactingFormatter()
    with caplog.at_level(logging.INFO):
        structlog.get_logger("redaction-test").info("connect", url="redis://:pw123secret@h:1")
        try:
            raise RuntimeError("upstream rejected gsk_tracebacksecret")
        except RuntimeError:
            logging.getLogger("redaction-test").exception("failed")

    rendered = "\n".join(formatter.format(r) for r in caplog.records)
    assert "pw123secret" not in rendered
    assert "gsk_tracebacksecret" not in rendered
    assert "[REDACTED]" in rendered


def test_migrate_keeps_file_when_no_keys(tmp_path: Path):
    """An apikeys.md without recognizable keys must not be deleted."""
    apikeys_file = tmp_path / "apikeys.md"
    apikeys_file.write_text("my groq key is in my password manager\n", encoding="utf-8")

    assert migrate_apikeys_to_dotenv(apikeys_file, tmp_path / ".env") is False
    assert apikeys_file.exists()


def test_migrate_skips_keys_already_in_dotenv(tmp_path: Path):
    """Re-running the migration does not duplicate keys that .env already defines."""
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("GROQ_API_KEY=gsk_existing\n", encoding="utf-8")
    apikeys_file = tmp_path / "apikeys.md"
    apikeys_file.write_text("GROQ_API_KEY=gsk_new\nTYPESAFE_API_KEY=ts_new\n", encoding="utf-8")

    assert migrate_apikeys_to_dotenv(apikeys_file, dotenv_file) is True
    text = dotenv_file.read_text(encoding="utf-8")
    assert text.count("GROQ_API_KEY=") == 1
    assert "TYPESAFE_API_KEY=ts_new" in text
    assert stat.S_IMODE(os.stat(dotenv_file).st_mode) == 0o600
