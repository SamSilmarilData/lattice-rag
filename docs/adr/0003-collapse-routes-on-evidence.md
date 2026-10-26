---
status: proposed
---

# Collapse the retrieval Routes only on measured evidence

`vector_exact`, `hybrid` and `graph_relational` run the same retrieval and differ only in graph hop count: `GraphTraverser` walks 2 hops for `graph_relational` and 1 hop for the other two, so `vector_exact` and `hybrid` are already identical. Every query still pays a Jev Choice call that picks among them. We keep all five Routes until the phase 5 ablation (#46) has run. The ablation forces 1-hop onto every non-`chitchat` query in one Live Tier run and 2-hop in another, on Gold Set v3. **If 2-hop does not beat 1-hop by more than the baseline tolerance (`max_regression_delta` in `eval_baseline.json`) on any metric of any Slice, the three collapse into one retrieval Route**, leaving retrieval, `massive_context` and `chitchat`. Otherwise all five stay. We write this rule before the numbers exist so the result cannot be argued afterwards.

If the Routes collapse, `expected_route` is remapped in the same commit, and Route Accuracy is recorded before and after the change: the jump comes from the smaller Route set, not from a better router.

## Considered Options

- Collapse now: a guess, with no evidence that the Routes are redundant on the harder Gold Set.
- Keep five Routes and give each retrieval Route its own retrieval behaviour: more code, and no evidence it would help.
- Run all three Routes as separate arms: `vector_exact` and `hybrid` execute identically, so a third arm only measures judge noise.

## Outcome

Pending the #46 ablation. The per-Slice results and the decision are recorded here when it runs.
