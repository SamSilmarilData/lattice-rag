# Domain Glossary

This document defines the ubiquitous language and domain concepts for the `lattice-rag` project. Architectural proposals and module seams must align with these terms.

---

### Core Domain Entities

- **Document**: A cohesive body of text ingested into the knowledge base, uniquely identified by a `doc_id` and human-readable `title`.
- **Chunk**: A contiguous segment of text (typically ~512 characters with sentence-aware overlap) extracted from a Document. Each chunk possesses a dense vector embedding for HNSW similarity and raw text for BM25 search.
- **Entity**: A named entity identified within a chunk by GLiNER (e.g., person, organization, technology, concept, database). Each entity node carries a name and entity type; entities have no embedding (only chunks are vector-indexed).
- **Relation**: A typed, directed relationship between two entities co-occurring in the same chunk context (e.g., `[:USES]`, `[:IMPLEMENTS]`, `[:RELATES_TO]`).
- **Subgraph**: A traversed neighborhood of Entity and Chunk nodes connected by Relation edges, providing structured multi-hop relational context.

---

### Ingestion Subsystem

- **Document Ingestion**: The end-to-end lifecycle of accepting raw document text, chunking it into sentence-bounded segments, generating dense embeddings, extracting entity-relation triples, persisting nodes and edges to LatticeDB, and invalidating stale caches.
- **Document Ingester**: The deep module responsible for executing document ingestion behind a single, high-leverage interface.

---

### Retrieval & Synthesis Subsystem

- **Hybrid Retrieval**: The 3-stage retrieval pipeline combining Stage 1 (HNSW vector + BM25 keyword recall fused via RRF into `top_k + 3` candidates), Stage 3 (cross-encoder reranking of every candidate on its chunk text), and Stage 2 (1-hop graph traversal from the entities of the final chunks, run after reranking and the guardrail, with a budget that ends the whole traversal).
- **Hybrid Retriever**: The deep module (`RetrievalPipeline`) orchestrating the 3-stage retrieval process along with guardrail filtering and low-grounding signaling behind a clean, unified async interface.
- **Low Grounding**: The state where retrieval found candidates but none passed the Noul relevance guardrail. The answer is generated without evidence, reported as `lowGrounding: true`, and never cached.
- **Route**: The strategy TypeSafe Jev Choice selects for a query: `hybrid` (the one retrieval Route: Hybrid Retrieval with a 1-hop traversal), `massive_context` (wide retrieval, `top_k=12`, a larger answer budget; the same Model Pool order as every Route, ADR-0004) or `chitchat` (no retrieval). ADR-0003 collapsed the former `vector_exact` and `graph_relational` Routes into `hybrid`, since all three ran the same retrieval and a second graph hop added no evidence. The choice is probabilistic; when Jev cannot decide, the Route falls back to `hybrid` and the answer is Unverified.
- **Synthesis**: Grounded generation producing a verifiable natural language answer with inline citations from retrieved chunks. The traversed Subgraph is not part of the prompt by default (`GRAPH_CONTEXT_IN_PROMPT=off`, #47); it is shown as the graph path and scored by Graph Recall.
- **Model Pool**: The ordered list of (provider, model) members that may generate an answer, the same order for every Route (ADR-0004). Each member has its own circuit breaker; a throttled, failing or timed-out member hands the query to the next. An answer from any member is a normal answer, attributed to the provider and model that gave it. _Avoid_: primary/fallback model.
- **Resilient Synthesizer**: The single generation path for both buffered and token-streaming modes. It tries the Model Pool in order, then the Redis FAQ.
- **Degraded Answer**: An answer no Model Pool member gave: the closest Redis FAQ answer or the offline message (or a stream cut off mid-answer). It is flagged to the caller, never stored in Tier 1, and fails a Live Tier query as `provider_degraded`. _Avoid_: fallback answer (for a later pool member's answer).
- **Tiered Cache**: Tier 1 is the in-memory exact + semantic answer cache (only complete Model Pool answers that are grounded and verified are stored). Tier 2 is the Redis hash of pre-computed FAQ answers, served when no Model Pool member answers.
- **Exact Cache Shortcut**: An $O(1)$ fast-path in Tier 1 cache that bypasses dense embedding generation and Jev Noul verification on exact query string matches, returning in $< 0.5$ms at zero API cost.
- **Document Bundle**: An atomic single-transaction ingestion payload committing Document, Chunk, Entity, and Relation graph nodes and edges into LatticeDB in a single WAL disk sync.


---

### Evaluation Subsystem

- **Gold Set**: The curated evaluation dataset: a small corpus of Documents plus queries, each with a reference answer and the verbatim passages that justify it.
- **Gold Passage**: A verbatim quote from a Gold Set Document that a correct retrieval must surface. It is resolved to the Chunk(s) containing it when the corpus is seeded.
- **Slice**: A tag grouping Gold Set queries by kind (`factoid`, `multihop`, `unanswerable`, `chitchat`, `long_context`) so results can be broken down. Every Slice but `chitchat` must hold at least 10 queries (`load_gold_set` rejects a smaller one), and the Hermetic Tier gates a Slice on its own floor only when it meets that minimum.
- **Case**: A tag recording how a Gold Set query is posed, orthogonal to its Slice: `direct` (in the corpus's own words), `paraphrase` (avoiding its Gold Passages' wording), `negation`, `ambiguity`, `conflicting_sources` or `unanswerable_near_miss`. The Hermetic Tier reports recall and MRR per Case but gates only per Slice.
- **Hermetic Tier**: The evaluation tier that needs no API keys and, once the local embedding models are cached, no network. It checks retrieval against Gold Passages and validates the Gold Set itself, and gates every PR.
- **Kernel Class**: The group of CPUs whose onnxruntime kernel scores the INT8 reranker identically: AVX2, AVX-512 VNNI or ARM. Rerank scores are bit-identical within a Kernel Class and can differ between them, so each Hermetic floor is the lowest value measured across Kernel Classes.
- **Live Tier**: The evaluation tier that runs the full pipeline and has Jev judge faithfulness, context precision, answer relevance and answer correctness. It needs API keys and is advisory, not a PR gate.
- **Experiment Arm**: One Live Tier run with a single switch changed from its default (`--force-hops`, `GRAPH_CONTEXT_IN_PROMPT` or the `massive_context` Noul settings), compared per Slice with the other arms of the same experiment. The **Route ablation** is the pair of arms that force 1-hop and then 2-hop traversal on the retrieval Route (`hybrid`; before ADR-0003 collapsed them, on `vector_exact`, `hybrid` and `graph_relational`); `massive_context` keeps its own hop count.
- **Eval Job**: One evaluation run started over the HTTP API, identified by a handle and polled for its status and report. Only one runs at a time per server; a start while one is running is rejected, not queued.
- **Answer Correctness**: How well an answer agrees with the Gold Set query's reference answer. Unlike faithfulness, which compares an answer to the evidence it was given, it catches answers that are well-grounded but wrong.
- **Expected Route**: The Route a Gold Set query should take, recorded as `expected_route`: `hybrid`, `massive_context` or `chitchat`. The Gold Set loader rejects the Route names ADR-0003 retired.
- **Route Accuracy**: The share of scored Gold Set queries whose chosen Route equals their Expected Route, reported by the Live Tier overall and per Slice. A query is scored when it is outside the `chitchat` Slice, its pipeline ran, and Jev (not the System 1 fallback) chose the Route; unverified and errored queries are counted, not scored. Advisory, like the rest of the Live Tier. It is only comparable across runs with the same Route set: the ADR-0003 collapse from five Routes to three raised it from 0.7375 to an expected ~0.975 on the 2026-10-27 default run without any change to the router.
- **Graph Recall**: The share of a Gold Set query's expected entities found among the Entities of its graph context, matched by name ignoring case and whitespace. Reported by the Live Tier per query, overall and per Slice; computed, not judged by Jev, and advisory. It is not in the Hermetic Tier, which seeds without entity extraction.
- **Misroute**: A scored Gold Set query whose chosen Route differs from its Expected Route. The Live Tier report lists each one with its Slice and both Routes.
- **Unverified Answer**: An answer produced while a Jev check (routing, guardrail, or cache verification) was unavailable, so its grounding was not confirmed. It is flagged to the caller and never stored in Tier 1, like a Degraded Answer.
- **Model Mix**: How many of a Live Tier run's answers each Model Pool member (and the FAQ) gave. Reported with every run and recorded in the baseline notes, so a baseline says which models it measured.
- **Seeded Database**: The knowledge base baked into a deployment at build time. A deployed instance treats it as read-only; documents ingested at runtime are not expected to survive a restart.
