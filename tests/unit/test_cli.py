"""CLI entry-point tests that do not load ML models."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from click.testing import CliRunner

from lattice_rag import cli


def test_seed_runs_without_typesafe_key(monkeypatch, tmp_path):
    """`seed` only touches storage, so it must not require TYPESAFE_API_KEY (Docker build)."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("LATTICEDB_PATH", str(tmp_path / "seed.db"))
    dataset = tmp_path / "ds.json"
    dataset.write_text(json.dumps({"corpus": [{"document_id": "d1", "text": "Hello."}]}))

    ingester = MagicMock()
    ingester.ingest = AsyncMock()
    monkeypatch.setattr("lattice_rag.ingestion.DocumentIngester", lambda **_: ingester)
    monkeypatch.setattr("lattice_rag.retrieval.embeddings.EmbeddingService", MagicMock())
    monkeypatch.setattr("lattice_rag.storage.extract.EntityExtractor", MagicMock())

    result = CliRunner().invoke(cli.main, ["seed", "--dataset", str(dataset)])
    assert result.exit_code == 0, result.output
    ingester.ingest.assert_awaited_once()


def test_serve_passes_host_and_port_through(monkeypatch):
    """`serve --host/--port` reach the server instead of being silently ignored."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "lattice_rag.server.main",
        lambda host=None, port=None: captured.update(host=host, port=port),
    )
    result = CliRunner().invoke(cli.main, ["serve", "--host", "127.0.0.1", "--port", "9123"])
    assert result.exit_code == 0, result.output
    assert captured == {"host": "127.0.0.1", "port": 9123}


def test_server_main_boots_without_typesafe_key(monkeypatch):
    """The server launcher must not exit when TYPESAFE_API_KEY is absent."""
    from lattice_rag import server

    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    started: dict[str, object] = {}

    class FakeGranian:
        def __init__(self, target: str, **kwargs: object) -> None:
            started.update(kwargs)

        def serve(self) -> None:
            started["served"] = True

    monkeypatch.setattr(server, "Granian", FakeGranian)
    server.main(port=9124)
    assert started["served"] is True
    assert started["port"] == 9124
