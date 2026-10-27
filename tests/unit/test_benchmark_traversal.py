"""Stage 2 of ``lattice-rag benchmark`` times the traversal queries actually run."""

from __future__ import annotations

from typing import Any

from lattice_rag.benchmark import BenchmarkEngine
from lattice_rag.storage.db import SubgraphResult


class _Store:
    def __init__(self) -> None:
        self.hops: list[int] = []

    def get_entities_for_chunks(self, chunk_ids: list[int], budget: int) -> list[int]:
        return [101]

    def traverse_from_entities(
        self, entity_ids: list[int], *, max_hops: int, budget: int
    ) -> SubgraphResult:
        self.hops.append(max_hops)
        return SubgraphResult(nodes=[], edges=[])


async def test_stage2_times_the_production_one_hop_walk() -> None:
    store = _Store()
    engine = BenchmarkEngine(store, embedding_service=None)  # type: ignore[arg-type]

    stats: Any = await engine.benchmark_stage2_traversal(anchor_ids=[1], iterations=2, warmup=1)

    assert set(store.hops) == {1}
    assert stats.stage == "Stage 2: Graph Traversal (1-hop)"
