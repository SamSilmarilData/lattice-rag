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
