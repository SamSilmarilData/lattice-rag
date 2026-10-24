"""Unit tests for the admin-key route guard."""

from __future__ import annotations

import pytest
from litestar import Litestar, post
from litestar.datastructures import State
from litestar.testing import AsyncTestClient

from lattice_rag.api.guards import ADMIN_KEY_HEADER, require_admin_key
from lattice_rag.config import AppConfig
from lattice_rag.security import SecretStr


@post("/protected", guards=[require_admin_key])
async def protected() -> str:
    return "ok"


def _app(admin_key: str) -> Litestar:
    config = AppConfig(
        typesafe_api_key=SecretStr("ts"),
        groq_api_key=SecretStr(""),
        gemini_api_key=SecretStr(""),
        admin_api_key=SecretStr(admin_key),
    )
    return Litestar(route_handlers=[protected], state=State({"config": config}))


@pytest.mark.asyncio
async def test_guard_rejects_missing_header():
    async with AsyncTestClient(app=_app("admin-secret")) as client:
        resp = await client.post("/protected")
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_guard_rejects_wrong_key():
    async with AsyncTestClient(app=_app("admin-secret")) as client:
        resp = await client.post("/protected", headers={ADMIN_KEY_HEADER: "nope"})
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_guard_accepts_correct_key():
    async with AsyncTestClient(app=_app("admin-secret")) as client:
        resp = await client.post("/protected", headers={ADMIN_KEY_HEADER: "admin-secret"})
        assert resp.status_code == 201


@pytest.mark.asyncio
async def test_guard_fails_closed_without_configured_key():
    async with AsyncTestClient(app=_app("")) as client:
        resp = await client.post("/protected", headers={ADMIN_KEY_HEADER: ""})
        assert resp.status_code == 401
