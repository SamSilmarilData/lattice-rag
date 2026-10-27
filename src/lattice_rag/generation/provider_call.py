"""What every provider adapter does the same way: bound each call, and describe a failure."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable

from lattice_rag.security import sanitize_error_detail


def describe_failure(error: Exception, timeout_seconds: float) -> str:
    """A log- and response-safe description of a failed provider call."""
    if isinstance(error, TimeoutError):
        return f"timed out after {timeout_seconds:g}s"
    return sanitize_error_detail(str(error))


async def timed_chunks[T](
    stream: AsyncIterable[T], timeout_seconds: float
) -> AsyncGenerator[T, None]:
    """Yield ``stream``'s chunks, raising ``TimeoutError`` when any one (the first
    included) takes longer than ``timeout_seconds``."""
    chunks = aiter(stream)
    while True:
        try:
            chunk = await asyncio.wait_for(anext(chunks), timeout=timeout_seconds)
        except StopAsyncIteration:
            return
        yield chunk
