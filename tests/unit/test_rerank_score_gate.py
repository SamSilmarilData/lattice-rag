"""The optional rerank-score gate in front of the Jev Noul context guardrail (#20)."""

from __future__ import annotations

from typing import Any

from lattice_rag.routing.guardrail import (
    ContextGuardrail,
    GuardrailResult,
    RerankScoreGate,
    with_rerank_skip,
)


class _RecordingGuardrail:
    """Stand-in for ContextGuardrail: keeps chunks whose text is not in ``reject``."""

    def __init__(self, reject: tuple[str, ...] = (), verified: bool = True) -> None:
        self.reject = reject
        self.verified = verified
        self.calls: list[list[str]] = []
        self.closed = False

    async def filter_chunks(self, query: str, chunks: list[dict[str, Any]]) -> GuardrailResult:
        self.calls.append([c["text"] for c in chunks])
        kept = [c for c in chunks if c["text"] not in self.reject]
        return GuardrailResult(
            chunks=kept, low_grounding=bool(chunks) and not kept, verified=self.verified
        )

    async def aclose(self) -> None:
        self.closed = True


def _chunks(*scored: tuple[str, float]) -> list[dict[str, Any]]:
    return [{"node_id": i, "text": t, "rerank_score": s} for i, (t, s) in enumerate(scored)]


async def test_clearly_relevant_chunks_skip_noul_and_the_rest_are_checked():
    inner = _RecordingGuardrail()
    gate = RerankScoreGate(inner, min_score=4.0)

    result = await gate.filter_chunks("q", _chunks(("a", 7.1), ("b", 4.0), ("c", 1.2), ("d", -3)))

    assert inner.calls == [["c", "d"]]
    assert [c["text"] for c in result.chunks] == ["a", "b", "c", "d"]
    assert result.verified is True
    assert result.low_grounding is False


async def test_noul_rejections_still_apply_below_the_threshold_and_order_is_kept():
    inner = _RecordingGuardrail(reject=("c",))
    gate = RerankScoreGate(inner, min_score=4.0)

    result = await gate.filter_chunks("q", _chunks(("c", 2.0), ("a", 5.0), ("d", 1.0)))

    assert [c["text"] for c in result.chunks] == ["a", "d"]


async def test_no_noul_call_when_every_chunk_clears_the_threshold():
    inner = _RecordingGuardrail()
    gate = RerankScoreGate(inner, min_score=4.0)

    result = await gate.filter_chunks("q", _chunks(("a", 9.0), ("b", 6.0)))

    assert inner.calls == []
    assert [c["text"] for c in result.chunks] == ["a", "b"]
    assert result == GuardrailResult(chunks=result.chunks, low_grounding=False, verified=True)


async def test_skipped_chunks_keep_grounding_when_noul_rejects_all_the_others():
    inner = _RecordingGuardrail(reject=("b", "c"))
    gate = RerankScoreGate(inner, min_score=4.0)

    result = await gate.filter_chunks("q", _chunks(("a", 5.0), ("b", 1.0), ("c", 0.0)))

    assert [c["text"] for c in result.chunks] == ["a"]
    assert result.low_grounding is False


async def test_unverified_noul_verdict_stays_unverified():
    inner = _RecordingGuardrail(verified=False)
    gate = RerankScoreGate(inner, min_score=4.0)

    result = await gate.filter_chunks("q", _chunks(("a", 5.0), ("b", 1.0)))

    assert result.verified is False


async def test_chunk_without_a_rerank_score_is_checked_by_noul():
    inner = _RecordingGuardrail()
    gate = RerankScoreGate(inner, min_score=4.0)

    await gate.filter_chunks("q", [{"node_id": 1, "text": "a", "score": 99.0}])

    assert inner.calls == [["a"]]


async def test_closing_the_gate_closes_the_guardrail():
    inner = _RecordingGuardrail()

    await RerankScoreGate(inner, min_score=4.0).aclose()

    assert inner.closed is True


def test_with_rerank_skip_off_returns_the_guardrail_unchanged():
    guardrail = ContextGuardrail(ts_client=object())  # type: ignore[arg-type]

    assert with_rerank_skip(guardrail, None) is guardrail
    assert with_rerank_skip(None, 4.0) is None


def test_with_rerank_skip_on_wraps_the_guardrail():
    guardrail = ContextGuardrail(ts_client=object())  # type: ignore[arg-type]

    gate = with_rerank_skip(guardrail, 4.0)

    assert isinstance(gate, RerankScoreGate)
    assert gate.min_score == 4.0
