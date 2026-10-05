from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

_pool: ProcessPoolExecutor | None = None

def get_process_pool(max_workers: int = 2) -> ProcessPoolExecutor:
    """
    Returns a singleton ProcessPoolExecutor. Creates it on first call.
    
    Args:
        max_workers: Maximum number of worker processes.
        
    Returns:
        ProcessPoolExecutor: The singleton instance.
    """
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=max_workers)
    return _pool

def shutdown_pool() -> None:
    """
    Shuts down the pool gracefully if it exists.
    """
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True)
        _pool = None

def run_in_pool(fn: Callable[..., T], *args: Any) -> Awaitable[T]:
    """
    Async helper that runs a sync callable in the ProcessPoolExecutor using asyncio.get_running_loop().run_in_executor().
    
    Args:
        fn: The callable to run.
        args: Arguments to pass to the callable.
        
    Returns:
        Awaitable[T]: An awaitable that resolves to the result of the callable.
    """
    pool = get_process_pool()
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(pool, fn, *args)
