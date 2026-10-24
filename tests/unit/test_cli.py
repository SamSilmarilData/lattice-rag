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
    dataset.write_text(json.dumps(_gold_set_with_one_document()))

    ingester = MagicMock()
    ingester.ingest = AsyncMock()
    monkeypatch.setattr("lattice_rag.ingestion.DocumentIngester", lambda **_: ingester)
    monkeypatch.setattr("lattice_rag.retrieval.embeddings.EmbeddingService", MagicMock())
    monkeypatch.setattr("lattice_rag.storage.extract.EntityExtractor", MagicMock())

    result = CliRunner().invoke(cli.main, ["seed", "--dataset", str(dataset)])
    assert result.exit_code == 0, result.output
    ingester.ingest.assert_awaited_once()


def _gold_set_with_one_document() -> dict[str, object]:
    return {
        "version": "2.0.0",
        "domain": "test",
        "corpus": [{"document_id": "d1", "title": "D1", "text": "Hello world."}],
        "queries": [
            {
                "id": "q1",
                "query": "What is said?",
                "slice": "factoid",
                "expected_route": "vector_exact",
                "reference_answer": "Hello world.",
                "gold_passages": [{"document_id": "d1", "text": "Hello world."}],
            }
        ],
    }


def test_seed_refuses_an_invalid_gold_set(monkeypatch, tmp_path):
    """A Gold Passage that is not verbatim fails seeding (and so the Docker build) loudly."""
    monkeypatch.setenv("LATTICEDB_PATH", str(tmp_path / "seed.db"))
    data = _gold_set_with_one_document()
    data["queries"][0]["gold_passages"][0]["text"] = "Hello there."  # type: ignore[index]
    dataset = tmp_path / "ds.json"
    dataset.write_text(json.dumps(data))
    ingester = MagicMock()
    ingester.ingest = AsyncMock()
    monkeypatch.setattr("lattice_rag.ingestion.DocumentIngester", lambda **_: ingester)

    result = CliRunner().invoke(cli.main, ["seed", "--dataset", str(dataset)])

    assert result.exit_code != 0
    assert "q1: gold_passages[0] is not verbatim" in result.output
    ingester.ingest.assert_not_called()


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
