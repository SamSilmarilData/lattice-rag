# Domain Glossary

This document defines the ubiquitous language and domain concepts for the `lattice-rag` project. Architectural proposals and module seams must align with these terms.

---

### Core Domain Entities

- **Document**: A cohesive body of text ingested into the knowledge base, uniquely identified by a `doc_id` and human-readable `title`.
- **Chunk**: A contiguous segment of text (typically ~512 characters with sentence-aware overlap) extracted from a Document. Each chunk possesses a dense vector embedding for HNSW similarity and raw text for BM25 search.
- **Entity**: A named entity identified within a chunk by GLiNER (e.g., person, organization, technology, concept, database). Each entity node carries a name, entity type, and dense embedding.
- **Relation**: A typed, directed relationship between two entities co-occurring in the same chunk context (e.g., `[:USES]`, `[:IMPLEMENTS]`, `[:RELATES_TO]`).
- **Subgraph**: A traversed neighborhood of Entity and Chunk nodes connected by Relation edges, providing structured multi-hop relational context.

---

### Ingestion Subsystem

- **Document Ingestion**: The end-to-end lifecycle of accepting raw document text, chunking it into sentence-bounded segments, generating dense embeddings, extracting entity-relation triples, persisting nodes and edges to LatticeDB, and invalidating stale caches.
- **Document Ingester**: The deep module responsible for executing document ingestion behind a single, high-leverage interface.

---

### Retrieval & Synthesis Subsystem

- **Hybrid Retrieval**: The 3-stage retrieval pipeline combining Stage 1 (HNSW vector + BM25 keyword recall fused via RRF), Stage 2 (dynamic 1-to-2 hop Cypher traversal), and Stage 3 (cross-encoder precision reranking with linearized triples).
- **Hybrid Retriever**: The deep module orchestrating the 3-stage retrieval process along with guardrail filtering and low-grounding signaling behind a clean, unified async interface.
- **Route**: A deterministic query classification produced by TypeSafe Jev (`vector_exact`, `graph_relational`, `hybrid`, `chitchat`, `massive_context`).
- **Synthesis**: Grounded generation producing a verifiable natural language answer with inline citations from retrieved chunks and subgraphs.
- **Resilient Synthesizer**: The deep module encapsulating multi-tier generative resilience, managing circuit breaker state across GroqCloud, Gemini Flash, and local Redis FAQ fallbacks in both buffered and token-streaming modes.
- **Tiered Cache**: Multi-level query caching combining Tier 1 in-memory semantic vector similarity with Tier 2 circuit-broken Redis FAQ fallback.
- **Exact Cache Shortcut**: An $O(1)$ fast-path in Tier 1 cache that bypasses dense embedding generation and Jev Noul verification on exact query string matches, returning in $< 0.5$ms at zero API cost.
- **Document Bundle**: An atomic single-transaction ingestion payload committing Document, Chunk, Entity, and Relation graph nodes and edges into LatticeDB in a single WAL disk sync.

