# System Architecture & Technical Design

This document details the architectural principles, subsystems, and execution lifecycle of the **`lattice-rag`** engine.

---

## 1. Design Philosophy: The In-Process Paradigm

Traditional RAG and GraphRAG architectures introduce severe performance and cost bottlenecks:
- **Network Boundaries:** Hitting remote vector databases (Pinecone, Qdrant) over HTTP incurs 30–100ms per search.
- **Heavy Infrastructure Footprint:** JVM-based graph servers (Neo4j) consume gigabytes of memory and require dedicated cloud clusters.
- **Uncontrolled Context Windows:** Shoveling raw retrieved text into frontier LLMs inflates generation costs and increases hallucination surface.

`lattice-rag` adopts an **in-process, single-binary architecture**:
1. All graph nodes, edges, HNSW vector indices, and BM25 inverted indices reside inside a **single local file** (`data/lattice_rag.db`) managed by **LatticeDB**.
2. All embeddings and entity extractions run locally on **CPU** using quantized ONNX models via **FastEmbed** and **GLiNER2.5-Decide**.
3. Small, calibrated "System One" decision models (**TypeSafe AI Jev**) act as front-door traffic controllers and context guardrails, reducing token expenditures.

---

## 2. End-to-End Query Execution Lifecycle

The following sequence diagram illustrates the lifecycle of a user query through the system:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/d2-query-lifecycle-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="diagrams/d2-query-lifecycle-light.svg">
  <img alt="Lattice RAG Query Execution Lifecycle" src="diagrams/d2-query-lifecycle-dark.svg" width="100%">
</picture>

<sub>Interactive &amp; High-Resolution Inspection: <a href="diagrams/d2-query-lifecycle-dark.html">Dark Specimen</a> | <a href="diagrams/d2-query-lifecycle-light.html">Light Specimen</a></sub>

---

## 3. Subsystem Breakdown

### 3.1 Web Server & API Layer (Litestar + Granian + msgspec)
- **Granian ASGI Server:** Multi-threaded Rust HTTP engine running on Python 3.12 without the overhead of standard Python web servers.
- **Litestar Framework:** Employs explicit parameter markers (`JSONBody`, `FromQuery`, `FromPath`) and DTO architectures.
- **msgspec Structs:** All request/response schemas use strict `msgspec.Struct` with `rename="camel"`, outperforming Pydantic by 2–4x in serialization speed.
- **Interactive Documentation:** Native **Scalar** OpenAPI interface rendered directly at `/schema/scalar`.

### 3.2 System 1 Decision Routing (TypeSafe AI Jev)
Instead of invoking large generative models for intent recognition, `lattice-rag` uses **TypeSafe AI Jev**:
- **Front-Door Router (`routing/router.py`):**
  Uses the `Choice` primitive to classify queries into:
  - `vector_exact`: Direct factual lookups and code definitions.
  - `graph_relational`: Questions requiring multi-step relational hops across entities.
  - `hybrid`: Broad semantic queries needing both text context and relational graph structure.
  - `chitchat`: Greetings and pleasantries. Messages made up entirely of small talk ("hi!", "thanks a lot", "what can you do?") skip the Jev call through a deterministic fast path; a question that merely contains such a word ("How does HNSW help recall?") is always routed by Jev.
  - `massive_context`: Cross-corpus summarizations requiring 1M token analysis.
- **Context Guardrail (`routing/guardrail.py`):**
  Uses the boolean `Noul` primitive to batch-score retrieved chunks against the user query, pruning irrelevant or tangential passages. It returns a per-request `GuardrailResult(chunks, low_grounding)`; when no chunk passes, the answer is generated without evidence, flagged `lowGrounding: true` in the response, and never cached. If Noul itself cannot run, the candidates pass through unchecked and the answer is an Unverified Answer (`unverified: true`), also never cached; the same applies when routing falls back to `hybrid`, and when a Tier 1 near-duplicate is found but its Noul equivalence check cannot run (the answer is recomputed rather than served). In mock-key mode the router and guardrail are not called at all. An optional `RerankScoreGate` (`GUARDRAIL_SKIP_RERANK_SCORE`, off by default) keeps chunks whose cross-encoder score clears the value without asking Noul; it is enabled only with Live Tier evidence (production-readiness §6).

### 3.3 Storage Engine (LatticeDB)
- Embedded single-file property-graph engine (`data/lattice_rag.db`).
- **Unified Querying:**
  - HNSW vector similarity search (`<=>` operator) across 384-dimensional dense vectors.
  - BM25 full-text inverted index (`@@` operator) on text content.
  - Sub-millisecond graph traversals using native C bindings.

### 3.4 3-Stage Hybrid Retrieval Pipeline
1. **Stage 1 (Vector Candidate Recall):** The query embedding computed once during triage is reused; LatticeDB executes HNSW vector search and BM25 search, combining results via Reciprocal Rank Fusion (RRF) into `top_k + 3` candidates (`RERANK_HEADROOM`). Each candidate keeps its `doc_id` and `position` for citations.
2. **Stage 2 (Relational Traversal):** Runs after Stage 3 and the context guardrail. It takes entity anchors only from the final chunks (those that survived reranking and the guardrail), so the LLM never sees relations reachable only from rejected candidates, and walks LatticeDB edges in-process for 1 hop (2 for `graph_relational`). The budget (25) caps nodes and edges separately; reaching either cap ends the whole traversal, and an edge is kept only with both endpoints. This is a pointer walk through the LatticeDB transaction API, not a Cypher query.
3. **Stage 3 (Precision Reranking):** The INT8 ONNX cross-encoder (`bge-reranker-v2-m3`) scores **every** Stage 1 candidate on its own chunk text and keeps the top `top_k` (5, or 12 for `massive_context`). Graph triples are not appended to candidates: they are identical for every candidate, so they could not change the ordering. They go to the LLM prompt instead. Each candidate costs about 100 ms on CPU, which is why the headroom is small.

### 3.5 Multi-Tier Caching & Circuit Breaker

`ResilientSynthesizer` (`generation/resilient_synthesizer.py`) is the single generation path for both `RAGOrchestrator.run()` and `stream_query()`. It owns one `pybreaker` circuit breaker around Groq, paired with a 3-tier cascade:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/d6-circuit-breaker-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="diagrams/d6-circuit-breaker-light.svg">
  <img alt="3-Tier Resilient Circuit Breaker Cascade" src="diagrams/d6-circuit-breaker-dark.svg" width="100%">
</picture>

<sub>Interactive &amp; High-Resolution Inspection: <a href="diagrams/d6-circuit-breaker-dark.html">Dark Specimen</a> | <a href="diagrams/d6-circuit-breaker-light.html">Light Specimen</a></sub>

- **Tier 1 (Semantic Vector Cache & Primary Groq):** Evaluates incoming query against in-memory query embeddings. If cosine similarity $\ge 0.90$, Jev's `Noul` primitive verifies semantic equivalence before the cached answer is returned. Standard non-cached queries route to primary **GroqCloud** (`GROQ_MODEL`). Only answers from the primary provider that were grounded in guardrail-approved evidence are cached.
- **Tier 2 (Circuit Breaker & Cascade 1 Fallback):** The breaker records every Groq outcome from both the buffered and streaming paths (`fail_max=3`, `reset_timeout=30s`). A Groq call that ends in an error counts as one failure. The Groq SDK and `GroqSynthesizer` retry HTTP 429 internally first, so a rate-limited call only counts once its retries are exhausted. Client disconnects (`asyncio.CancelledError`) are not counted. After 3 consecutive failures the circuit opens and Groq is skipped in favour of **Gemini** (`GEMINI_MODEL`) until the reset timeout lets a probe through. Answers produced by Gemini or the FAQ are marked `degraded: true`.
- **Tier 3 (Cascade 2 Offline FAQ Hash):** If both LLM providers fail, the system returns the closest pre-computed answer from the Redis hash (`cache:fallback:queries`, seeded from `fallback_queries.json`). This answer is for the closest *seeded* question, not necessarily the one asked, so it is always marked degraded.
- **Streaming:** The cascade only moves to the next provider while nothing has been emitted. If a provider fails after its first token, the stream sends an `error` event and ends with `done` (`degraded: true`) rather than appending a second provider's answer.

### 3.6 Keeping CPU Work Off the Event Loop
Embedding, reranking, GLiNER extraction and LatticeDB reads and writes run in worker threads via `asyncio.to_thread()`, so the Litestar event loop keeps serving other requests. ONNX Runtime releases the GIL during inference, so the heaviest work (embedding and reranking) runs in parallel with request handling, without the pickling and model-reload cost of a process pool.
