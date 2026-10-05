"""Unit tests for ProcessPoolExecutor isolation."""
from __future__ import annotations

import os
import pytest

from lattice_rag.orchestration.pool import get_process_pool, run_in_pool, shutdown_pool


def _cpu_worker_task(x: int) -> tuple[int, int]:
    """Helper function running in child process."""
    return x * x, os.getpid()


@pytest.mark.asyncio
async def test_run_in_pool_isolation():
    """Verify run_in_pool executes CPU task in an isolated worker process."""
    try:
        main_pid = os.getpid()
        res, worker_pid = await run_in_pool(_cpu_worker_task, 9)

        assert res == 81
        assert worker_pid != main_pid, "Worker PID should differ from main asyncio loop PID"
    finally:
        shutdown_pool()
