"""Comprehensive Zero-Trust Security & Credential Leakage Audit Suite."""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
import pytest
from litestar import Litestar, get
from litestar.testing import AsyncTestClient

from lattice_rag.api.dtos import (
    BenchmarkRunResponse,
    CacheStatsResponse,
    EvalRunResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from lattice_rag.app import create_app, secret_sanitizing_exception_handler
from lattice_rag.security import RedactingFilter, SecretStr, sanitize_error_detail


@pytest.mark.asyncio
async def test_rfc9457_error_shielding_scrubs_all_credentials():
    """Verify RFC 9457 error handler completely strips API keys and Bearer tokens."""
    sensitive_error_msg = (
        "Upstream 401 Unauthorized calling https://api.groq.com with header "
        "Authorization: Bearer gsk_liveSecretKey998877665544332211 and token=AIzaSyA1234567890abcdef"
    )

    @get("/error-test")
    async def failing_endpoint() -> None:
        raise RuntimeError(sensitive_error_msg)

    app = Litestar(
        route_handlers=[failing_endpoint],
        exception_handlers={Exception: secret_sanitizing_exception_handler},
    )

    async with AsyncTestClient(app) as client:
        response = await client.get("/error-test")
        assert response.status_code == 500
        assert "application/problem+json" in response.headers["content-type"]
        data = response.json()
        assert "detail" in data

        # Assert no sensitive credentials survived in detail
        assert "gsk_liveSecretKey" not in data["detail"]
        assert "AIzaSy" not in data["detail"]
        assert "Bearer" not in data["detail"]
        assert "[REDACTED]" in data["detail"]


@pytest.mark.asyncio
async def test_openapi_schema_contains_no_secrets():
    """Verify generated OpenAPI schema contains zero secret field leaks."""
    app = create_app()
    async with AsyncTestClient(app) as client:
        response = await client.get("/schema/openapi.json")
        assert response.status_code == 200
        schema_text = response.text.lower()

        forbidden_tokens = [
            "typesafe_api_key",
            "groq_api_key",
            "gemini_api_key",
            "gsk_",
            "aizasy",
            "redis://",
        ]
        for token in forbidden_tokens:
            assert token not in schema_text, f"Forbidden token '{token}' discovered in OpenAPI schema"


def test_dto_boundary_isolation():
    """Verify that all public DTO structs exclude internal secret configuration fields."""
    dto_classes = [
        QueryRequest,
        QueryResponse,
        IngestRequest,
        IngestResponse,
        CacheStatsResponse,
        EvalRunResponse,
        HealthResponse,
        BenchmarkRunResponse,
    ]

    for dto in dto_classes:
        fields = [f.lower() for f in getattr(dto, "__struct_fields__", ())]
        for sensitive_term in ["secret", "key", "token", "password", "auth", "credential"]:
            # 'key' is permitted in cache_key or query_key if any, but not api_key
            for field_name in fields:
                assert "api_key" not in field_name, f"DTO {dto.__name__} exposes sensitive field {field_name}"
                assert "password" not in field_name, f"DTO {dto.__name__} exposes sensitive field {field_name}"
                assert "token" not in field_name, f"DTO {dto.__name__} exposes sensitive field {field_name}"


def test_log_filter_redaction_comprehensive():
    """Verify RedactingFilter redacts diverse secret prefixes across log records."""
    redacting_filter = RedactingFilter()

    test_cases = [
        ("Calling Groq with key gsk_abc12345678901234567890", "gsk_"),
        ("Using OpenAI sk-12345678901234567890abcdef", "sk-"),
        ("Gemini AIzaSyA1234567890abcdef12345678901234", "AIza"),
        ("Header Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6", "Bearer"),
        ("Connecting to redis://user:supersecretpass@localhost:6379/0", "supersecretpass"),
    ]

    for raw_message, prefix in test_cases:
        record = logging.LogRecord(
            name="security_test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg=raw_message,
            args=(),
            exc_info=None,
        )
        assert redacting_filter.filter(record) is True
        assert "[REDACTED]" in record.msg
        assert prefix not in record.msg or prefix in ["[REDACTED]"]


def test_git_tracked_files_zero_secrets():
    """Verify no live API keys or credentials are committed into tracked git files."""
    try:
        tracked_files = subprocess.check_output(
            ["git", "ls-files"], text=True
        ).splitlines()
    except Exception as e:
        pytest.skip(f"Git command failed: {e}")

    # Regex patterns for live production credentials
    secret_patterns = [
        re.compile(r"gsk_[a-zA-Z0-9]{30,}"),
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),
        re.compile(r"sk-[a-zA-Z0-9]{32,}"),
    ]

    for file_path in tracked_files:
        p = Path(file_path)
        if not p.is_file() or p.name in [".gitignore", ".dockerignore"]:
            continue

        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for pat in secret_patterns:
            matches = pat.findall(content)
            # Filter out mock strings in tests
            real_matches = [m for m in matches if not any(x in m for x in ["example", "abc1234", "dummy", "fake", "placeholder"])]
            assert len(real_matches) == 0, f"Potential real credential found in tracked file {file_path}: {real_matches}"
