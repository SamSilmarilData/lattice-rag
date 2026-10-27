# Plan: Gold Set, eval suite and agent remediation

Source: the 2026-10-25 audit (Hermetic Tier run, one Live Tier run, diagnostic run of the weakest queries) and the design interview that followed. Vocabulary follows `GLOSSARY.md`. The Live Tier stays advisory (ADR-0002).

## Why

The Live Tier run passed 23/30 queries and failed its own gate on per-query floors. Causes found:

- **Generation:** Groq `max_tokens=256` truncates broad answers (eval_20, eval_29). `massive_context` has no Groq fallback when Gemini fails (eval_28, eval_30).
- **Routing:** Route Accuracy is 17/27 (63%). Nothing measures it, and confidence is ignored.
- **Eval validity:** the judge does not see the graph triples the generator saw. Correct refusals are penalised on relevance. Provider fallbacks are scored as answers.
- **Gold Set:** 5 Documents and 10 Chunks, so Hermetic recall is 1.0 on every Slice. The unanswerable rerank ceiling separates nothing. `expected_entities` is never read. Some reference answers claim reranker behaviour the code does not have (eval_04, eval_21).

## Phases

Each phase is its own branch and PR off `main`. Phases 1 and 3 are independent and may run in parallel; 4 follows 1-3 for measurement purposes; 5 is last because every earlier phase moves the metrics.

### Phase 1: Generation reliability (`fix/generation-reliability`)

| Task | Detail | Acceptance |
|---|---|---|
| 1.1 Route-aware token cap | Default 768, `massive_context` 1536. `LLM_MAX_TOKENS` still overrides the default. Explicit cap for Gemini. Log when `finish_reason == "length"`. | A broad query's answer ends on a complete sentence. Unit test with a fake client returning `finish_reason="length"` asserts the log. |
| 1.2 `massive_context` fallback | Gemini fails -> Groq (behind the breaker, answer `degraded`) -> Redis FAQ. Same in `synthesize` and `stream`. | `test_orchestrator_unified.py` and `test_resilient_synthesizer.py` cover both entry points. Degraded answers are not written to Tier 1. |
| 1.3 Citation numbering | Prompt chunks are `index="1"`...; judge context is `[1]`...; both say `[Chunk N]`. | `test_prompt.py` asserts 1-based indices. Re-judge eval_03 before claiming this as its cause. |

### Phase 2: Route Accuracy (`feat/route-accuracy`)

| Task | Detail | Acceptance |
|---|---|---|
| 2.1 Metric | Live Tier reports Route Accuracy overall and per Slice, and lists each misroute by query id. Advisory floor 0.80 in `eval_baseline.json`. Hermetic Tier unchanged. | `EvalRunResponse` and `frontend/src/types.ts` carry the new fields together. |
| 2.2 Router prompt | Sharper `_ROUTER_CRITERIA`, worked examples per Route, especially `massive_context` versus multi-part "describe the full flow" questions. | Route Accuracy >= 0.80 on the current 30 queries (from 0.63). |
| 2.3 Glossary | `GLOSSARY.md` edits (Route, Expected Route, Route Accuracy). | Already merged with the plan PR; phase 2 only has to keep the code consistent with them. |

A confidence-based override is not built unless 2.2 misses the floor.

### Phase 3: Eval validity (`fix/eval-validity`)

| Task | Detail | Acceptance |
|---|---|---|
| 3.1 Judge context | Pass graph triples to the judge as part of the evaluated context. | `test_eval_triage.py` asserts triples reach the judge state. |
| 3.2 Unanswerable scoring | `_SLICE_METRICS["unanswerable"] = ("faithfulness", "answer_correctness")`. | eval_24 passes on a correct refusal. |
| 3.3 Provider noise | Retry a degraded answer once. If still degraded, fail the query with `provider_degraded`. Report lists the provider per query. | `EvalQueryError` and `types.ts` change together. An induced Gemini 503 shows as a failed query, not a low score. |

### Phase 4: Gold Set v3 (`feat/gold-set-v3`)

Order inside the phase matters: 4.1 first, because it moves Chunk boundaries and therefore Gold Passage resolution.

| Task | Detail | Acceptance |
|---|---|---|
| 4.1 Chunk overlap | `chunk_text` carries the last sentence of each Chunk into the next regardless of length. | `test_document_ingester.py` covers overlap. `_covered` in `eval/hermetic.py` still resolves passages across boundaries. |
| 4.2 Corpus | Keep the 5 existing Documents and add the six `docs/*.md` files, split into Documents. Snapshot into the Gold Set file, not read live. Target about 150 Chunks. | `load_gold_set` passes; every Gold Passage is verbatim. |
| 4.3 Truth fixes | Corpus text says graph triples go to the generation prompt, not the reranker. Rewrite eval_04, eval_19 (drop Neo4j/Pinecone), eval_20, eval_21, eval_30 and their Gold Passages. | No reference answer asserts behaviour absent from the code. |
| 4.4 Queries | Version 3.0.0, about 80 queries with a `case` tag: paraphrase, negation, ambiguity, conflicting sources, about 10 unanswerable near-misses. `find_problems` validates the tag. | `test_gold_set.py` covers the new validation. |
| 4.5 `expected_entities` | Becomes a graph-recall check (Live Tier unless it stays fast and key-free with GLiNER). Fix the 5 entries whose entities are not in the corpus. | No `expected_entities` value is missing from its corpus. |
| 4.6 Remove ceiling | Delete `unanswerable_max_rerank_score` from `hermetic_thresholds.json` and `hermetic.py`. Abstention is checked by the Live Tier. | `test_hermetic_tier.py` updated. |
| 4.7 Floors | Run the Hermetic Tier on the new set, set each Slice floor about 0.05 below the measured value, record the measured values in the thresholds file. Rebuild the Seeded Database. | The Hermetic Tier discriminates (recall below 1.0 on some Slice) and passes on the new floors. The canary in `tests/hermetic/` still fails a broken retriever. |

### Phase 5: Experiments and re-baseline (`chore/rebaseline-v3`)

One run per experiment arm; the full three-run set only for the final baseline.

| Task | Detail | Decision rule |
|---|---|---|
| 5.1 Route ablation | Two Live Tier arms on the three retrieval Routes (`vector_exact`, `hybrid`, `graph_relational`): force 1-hop, then force 2-hop (`lattice-rag eval --force-hops 1|2`). `vector_exact` and `hybrid` already run identically, so a per-Route arm would only measure judge noise (#46 Decision). `massive_context` keeps its own hop count. | See ADR-0003, written before the numbers. |
| 5.2 Graph context ablation | `GRAPH_CONTEXT_IN_PROMPT` on/off, per Slice. | If it helps only `multihop`, send it only on `graph_relational` and `hybrid`, capped at 10 edges. Bidirectional traversal only if `multihop` gains. |
| 5.3 Noul on broad queries | `long_context` Slice: Noul 0.50 (today), 0.35 for `massive_context`, guardrail skipped for `massive_context` (answer stays Unverified). | Keep the best on the `long_context` Slice. Prefer the per-route threshold on a tie. |
| 5.4 Baseline | Three `lattice-rag eval --output` runs, then `lattice-rag measure-baseline`. Update `docs/production-readiness.md` section 6 if the procedure changed. | The Live Tier gate is green on the code it was measured on. |

## ADR

The route-collapse rule is recorded in `docs/adr/0003-collapse-routes-on-evidence.md`.

## Cross-cutting

- `GLOSSARY.md` edits (Route, Expected Route, Route Accuracy) ship with the plan PR.
- ADR-0002 is unchanged: the Live Tier stays advisory. A green gate in 5.4 is a quality target, not a PR gate.
- Docs and diagrams affected (update the paired light/dark `.html` and `.svg` where they change): the query-flow diagram if Routes collapse, the CI diagram if the Hermetic Tier gains checks.
- Phase 5 spends real API budget; run the full set only once.

## Out of scope

Confidence-based route override (unless 2.2 misses its floor), bidirectional graph traversal (unless 5.2 shows a gain), and any change to the Live Tier's advisory status.

## Risks

| Risk | Mitigation |
|---|---|
| The Gold Set rebuild changes every metric at once | Keep phases 1-3 measured on v2 first, then re-baseline only after phase 4. |
| Docs edits break verbatim Gold Passages | Snapshot the docs into the Gold Set file. |
| Chunk overlap changes Chunk boundaries and the Seeded Database | Land 4.1 first, rebuild the Seeded Database in 4.7. |
| Judge noise (faithfulness sd 0.029) hides small effects | Treat experiment deltas under the measured tolerance as ties. |
