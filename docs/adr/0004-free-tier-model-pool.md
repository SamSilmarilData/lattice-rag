---
status: accepted
---

# Generate from a free-tier Model Pool, attributed rather than degraded

The project must run entirely on free tiers, and free quotas are per model: on 2026-10-27 Groq gave about 200k tokens a day, with separate request counters, to each of `qwen/qwen3.8-27b`, `openai/gpt-oss-120b` and `openai/gpt-oss-20b`. The same day `gemini-3.8-flash` returned 503 "high demand" on most free-tier calls and once hung for 6.5 minutes, while `gemini-3.5-flash-lite` answered normally. One primary model with every other answer marked degraded used a third of the free capacity, and failed Live Tier queries as `provider_degraded` whenever the primary was throttled.

So generation goes through a **Model Pool**: an ordered list of (provider, model) members set by `LLM_POOL` (for example `groq:qwen/qwen3.8-27b,groq:openai/gpt-oss-120b,gemini:gemini-3.5-flash-lite`). Without `LLM_POOL` the pool is `GROQ_MODEL`, `openai/gpt-oss-120b`, then `GEMINI_MODEL` (default `gemini-3.5-flash-lite`, because of the 3.8-flash outage above). Each member has its own circuit breaker. A rate limit, quota exhaustion, 5xx or timeout moves to the next member at once: neither the Groq SDK nor our adapter retries, since waiting out a 429 stalls a query that another member could answer now. Every Route uses the same order, including `massive_context`, which keeps only its larger answer budget; this replaces the earlier Gemini-first order for that Route.

**An answer from any pool member is a normal answer**: judged by the Live Tier, eligible for Tier 1, and attributed with its provider and model on `QueryResponse`, the stream `done` event, the studio and the eval report. `degraded` now means no pool member gave a complete answer (the Redis FAQ, the offline message, or a stream cut off after its first token), and `provider_degraded` in the Live Tier means the same. Each Live Tier report counts answers per model (the Model Mix), and `measure-baseline` writes the mix into the baseline notes, so a baseline says which models produced it.

Gemini members send the lowest thinking setting each model accepts, so thinking does not eat the answer budget: a zero `thinking_budget` for Flash-Lite and 2.5 Flash, `thinking_level=low` for other Gemini 3 models (`gemini-3.8-flash` rejects `minimal` with a 400). A rejected value is fixed without code, with `@<budget|level|default>` on the member in `LLM_POOL` or `GEMINI_THINKING` for the default pool.

## Considered Options

- Keep one primary and mark everything else degraded: uses a third of the free capacity, and turns ordinary throttling into failed eval queries.
- Pool, but keep later members degraded: the same eval failures. A judged, attributed answer from a smaller model is more honest than an unjudged one.
- Retry a throttled member with backoff before moving on: per-day quotas do not clear in seconds, and the wait adds latency for every query in the window.
- Round-robin across members: spreads load, but makes a run's model mix depend on timing; a fixed order is reproducible, and the Model Mix still shows how much each member answered.

## Consequences

A Live Tier run may judge answers from several models. Compare runs only when their Model Mix is similar, and re-measure the baseline (`docs/production-readiness.md` §6) when the pool changes. The phase 5 Experiment Arms that depend on generation (#47, #48, #49) are re-run on the pool configuration.
