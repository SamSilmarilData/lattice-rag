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
async def test_static_spa_serving(monkeypatch, tmp_path):
    """Verify static SPA router serves index.html when frontend/dist is present."""
    mock_dist = tmp_path / "mock_dist"
    mock_dist.mkdir()
    (mock_dist / "index.html").write_text(
        "<!DOCTYPE html><html><body>Hybrid GraphRAG Studio</body></html>",
        encoding="utf-8",
    )

    orig_path = Path

    def fake_path(p, *args, **kwargs):
        if str(p) == "frontend/dist":
            return mock_dist
        return orig_path(p, *args, **kwargs)

    monkeypatch.setattr("lattice_rag.app.Path", fake_path)

    app = create_app()
    async with AsyncTestClient(app) as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Hybrid GraphRAG Studio" in response.text
