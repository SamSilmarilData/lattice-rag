---
status: accepted
---

# Collapse the retrieval Routes only on measured evidence

`vector_exact`, `hybrid` and `graph_relational` run the same retrieval and differ only in graph hop count: `GraphTraverser` walks 2 hops for `graph_relational` and 1 hop for the other two, so `vector_exact` and `hybrid` are already identical. Every query still pays a Jev Choice call that picks among them. We keep all five Routes until the phase 5 ablation (#46) has run. The ablation forces 1-hop onto every query the router sends to one of these three Routes in one Live Tier run and 2-hop in another, on Gold Set v3. `massive_context` is not part of the ablation and keeps its own hop count (1) in both arms. **If 2-hop does not beat 1-hop by more than the baseline tolerance (`max_regression_delta` in `eval_baseline.json`) on any metric of any Slice, the three collapse into one retrieval Route**, leaving retrieval, `massive_context` and `chitchat`. Otherwise all five stay. We write this rule before the numbers exist so the result cannot be argued afterwards.

If the Routes collapse, `expected_route` is remapped in the same commit, and Route Accuracy is recorded before and after the change: the jump comes from the smaller Route set, not from a better router.

## Considered Options

- Collapse now: a guess, with no evidence that the Routes are redundant on the harder Gold Set.
- Keep five Routes and give each retrieval Route its own retrieval behaviour: more code, and no evidence it would help.
- Run all three Routes as separate arms: `vector_exact` and `hybrid` execute identically, so a third arm only measures judge noise.

## Outcome

Collapsed (2026-10-27), after amending the rule once the results were in. The amendment is recorded here so it can be judged, not hidden.

Live Tier on Gold Set 3.0.0, one run per arm, 83 queries, judged means (faithfulness / context precision / answer relevance / answer correctness). The 1-hop arm had 8 `provider_degraded` queries and the 2-hop arm 7, all on free-tier provider limits; those are not judged.

| Slice | 1-hop | 2-hop | 2-hop − 1-hop |
| :--- | :--- | :--- | :--- |
| factoid (41) | 0.739 / 0.733 / 0.820 / 0.746 | 0.728 / 0.742 / 0.831 / 0.742 | −0.011 / +0.009 / +0.011 / −0.004 |
| multihop (15) | 0.651 / 0.718 / 0.817 / 0.737 | 0.759 / 0.709 / 0.792 / 0.707 | **+0.108** / −0.009 / −0.025 / −0.030 |
| long_context (10) | 0.430 / 0.629 / 0.767 / 0.637 | 0.769 / 0.572 / 0.678 / 0.577 | not comparable: 2 and 3 judged queries |
| unanswerable (14) | 0.877 / – / – / 0.837 | 0.935 / – / – / 0.891 | +0.058 / – / – / +0.054 |
| chitchat (3) | – / – / 0.971 / 0.993 | – / – / 0.975 / 0.993 | +0.004 / – / – / 0.000 |

Graph Recall was identical for every query in both arms (0.19 overall), as a structural probe of the seeded graph predicted: a 1-hop walk reaches about 6.8 nodes and a 2-hop walk about 7.0, with identical subgraphs in 18 of 24 samples. `long_context` is mostly routed to `massive_context`, which kept 1 hop in both arms.

**Read literally, the rule keeps all five Routes**: 2-hop beats 1-hop by more than the tolerance (0.06) on multihop faithfulness. The whole gap is two queries, eval_53 and eval_54, whose faithfulness was zeroed by a Noul contradiction veto in the 1-hop arm while their answer correctness was 0.80 and 0.86. Without them the arms differ by 0.007. The veto is not caused by the hop count: eval_53 ran the identical 1-hop configuration in the default run and scored 0.72.

**Amendment:** a faithfulness score zeroed by a Noul veto is excluded from the per-Slice comparison, because a veto flips between runs of the same configuration and so measures judge noise, not the arm. Under the amended rule no metric of any Slice clears the tolerance, so `vector_exact`, `hybrid` and `graph_relational` collapse into `hybrid`, which walks 1 hop. The user chose to amend rather than replicate the multihop arm.

**Route Accuracy** on the 2026-10-27 default run: 0.7375 with five Routes (21 misroutes, 15 of them between `vector_exact` and `hybrid`, which already ran identically), and 0.975 when the same choices are scored against three Routes. The jump comes from the smaller Route set, not from a better router. The 0.80 advisory floor is now met, so the confidence-based route override from spec #22 is not built.

**Scope:** the result holds for this corpus only. The Gold Set 3.0.0 graph is sparse (140 entities over 169 Chunks), so it cannot show a benefit from a second hop. #56 tracks an entity-dense corpus; re-run this ablation on it before treating the graph as useless in general.
