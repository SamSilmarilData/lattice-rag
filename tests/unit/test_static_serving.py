"""Unit tests for static asset mounting and fallback landing route in Litestar."""
from __future__ import annotations

from pathlib import Path
import pytest
from litestar.testing import AsyncTestClient

from lattice_rag.app import create_app


@pytest.mark.asyncio
async def test_fallback_landing_page(monkeypatch, tmp_path):
    orig_path = Path
    non_existent = tmp_path / "does_not_exist"

    def fake_path(p, *args, **kwargs):
        if str(p) == "frontend/dist":
            return non_existent
        return orig_path(p, *args, **kwargs)

    monkeypatch.setattr("lattice_rag.app.Path", fake_path)

    app = create_app()
    async with AsyncTestClient(app) as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "lattice-rag Engine" in response.text
        assert "Visual Playground" in response.text
        assert "/schema/scalar" in response.text


@pytest.mark.asyncio
async def test_static_spa_serving():
    # frontend/dist exists and is mounted at root
    app = create_app()
    async with AsyncTestClient(app) as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Hybrid GraphRAG Studio" in response.text
