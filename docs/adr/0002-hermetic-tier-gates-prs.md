---
status: accepted
---

# The Hermetic Tier gates PRs; the Live Tier is advisory

The PR gate is the key-free Hermetic Tier: retrieval checked against Gold Passages plus validation of the Gold Set itself. The Live Tier (Jev-judged faithfulness, context precision, answer relevance, answer correctness) runs when the secret is available and reports trends, but does not block merges. We moved away from a live-judged gate because with 20 queries the judge's run-to-run noise is larger than the regression tolerance, and a gate that is skipped for fork PRs or missing secrets can read as green without having run.

## Considered Options

- Keep the live gate and widen the tolerance: still skippable, still costs API money on every PR.
- Make the live gate mandatory: blocks fork PRs and external contributors.
