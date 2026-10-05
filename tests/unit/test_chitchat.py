"""Unit tests for deterministic ChitchatHandler."""
from __future__ import annotations

import pytest

from lattice_rag.generation.chitchat import ChitchatHandler


@pytest.fixture
def handler() -> ChitchatHandler:
    return ChitchatHandler()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "hello",
        "Hi there!",
        "Hey assistant",
        "Good morning!",
        "greetings to the team",
    ],
)
async def test_chitchat_greetings(handler: ChitchatHandler, query: str):
    """Verify greeting phrases trigger greeting response."""
    response = await handler.handle(query)
    assert response == handler.responses["greetings"]
    assert "Hello! I am lattice-rag" in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "bye",
        "Goodbye!",
        "farewell for now",
        "cya later",
    ],
)
async def test_chitchat_farewell(handler: ChitchatHandler, query: str):
    """Verify farewell phrases trigger farewell response."""
    response = await handler.handle(query)
    assert response == handler.responses["farewell"]
    assert "Goodbye!" in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "thanks",
        "Thank you so much!",
        "thx a lot",
        "really appreciate your help",
    ],
)
async def test_chitchat_thanks(handler: ChitchatHandler, query: str):
    """Verify gratitude phrases trigger thanks response."""
    response = await handler.handle(query)
    assert response == handler.responses["thanks"]
    assert "You are welcome!" in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "help",
        "can you assist me?",
        "what can you do?",
    ],
)
async def test_chitchat_help(handler: ChitchatHandler, query: str):
    """Verify help/assistance phrases trigger help response."""
    response = await handler.handle(query)
    assert response == handler.responses["help"]
    assert "I can help you search through documents" in response


@pytest.mark.asyncio
async def test_chitchat_default_fallback(handler: ChitchatHandler):
    """Verify unclassified conversational input returns the default guidance response."""
    response = await handler.handle("something completely random with no greeting")
    assert response == handler.responses["default"]
    assert "I am a specialized knowledge engine" in response
