# Changelog

All notable changes to the `lattice-rag` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-10-06 (Algorithmic Time-Space Complexity Optimization)

### Added
- **O(1) Exact-Match Cache Shortcut (`src/lattice_rag/caching/semantic_cache.py`):**
  - Added `get_exact(query)` bypassing dense vector embedding and Jev Noul verification on exact query matches, achieving sub-millisecond return ($p50 < 0.2$ms) at zero API/CPU cost.
- **BLAS Vectorized Cosine Similarity (`src/lattice_rag/caching/semantic_cache.py`):**
  - Implemented pre-allocated contiguous 2D NumPy embeddings matrix `_embeddings_matrix` replacing Python list iteration.
  - Replaced $O(N)$ Python loop vector math with single-instruction BLAS matrix-vector dot product `np.dot(_embeddings_matrix[:n], q_unit)`.
  - Added bounded LRU eviction policy capped at 1,000 entries (~1.5 MB RAM bound) preventing unbounded memory growth.
- **Atomic Document Bundle Ingestion (`src/lattice_rag/storage/db.py`):**
  - Added `ingest_document_bundle(...)` committing Document node, Chunk nodes, Entity nodes, and Relation edges within a single write transaction and 1 WAL disk sync (reduced disk syncs from $2N+1$ to 1).
  - Added `get_entities_for_chunks(...)` batched Cypher lookup querying entities for all anchor chunks in 1 query.
- **Relation Triple Deduplication (`src/lattice_rag/storage/extract.py`):**
  - Added set-based deduplication `seen_triples` reducing relation extraction complexity from $O(E^2)$ to $O(E)$ unique edges.
- **Concurrent Pipelined Hybrid Search (`src/lattice_rag/retrieval/vector_search.py`):**
  - Overlapped CPU ONNX dense embedding generation and BM25 disk index search concurrently via `asyncio.gather`.
  - Replaced full $O(M \log M)$ sorting with $O(M \log K)$ min-heap selection via `heapq.nlargest`.
- **Pre-Tokenized Inverted Index (`src/lattice_rag/caching/fallback_cache.py`):**
  - Pre-tokenized candidate query tokens in `seed_fallback`, reducing fuzzy lookup to $O(T)$ set intersections without per-request tokenization.
- **Single-Pass Chitchat Automaton (`src/lattice_rag/generation/chitchat.py`):**
  - Replaced linear dictionary iteration with a single compiled regex automaton `_CHITCHAT_AUTOMATON` executing in $O(|query|)$ time.
- **Front-Door Triage Concurrency (`src/lattice_rag/orchestration/graph.py`):**
  - Triages exact cache hit first in $O(1)$; on miss, concurrently launches TypeSafe Jev routing and CPU embedding in thread pool.
- **Hybrid Warmup 2D Force Simulation (`frontend/src/components/GraphExplorer.tsx`):**
  - 60 ticks synchronous warmup off-screen for instant graph stabilization, followed by `requestAnimationFrame` sampling to throttle rendering at display refresh rate.
- **New Complexity Unit Tests (`tests/unit/test_complexity_optimizations.py`):**
  - 7 new unit tests verifying exact cache shortcuts, BLAS dot product, LRU eviction, single-pass chitchat, pipelined hybrid search, triple deduplication, and atomic bundle ingestion.
  - Test suite expanded to **131 tests (100% green, 0 warnings)**.

### Changed
- Configured explicit `max_tokens=512` on `GroqSynthesizer` to prevent OTPM over-reservation rate limit rejections (429) on on-demand tiers.
- Moved evaluation questions in `src/lattice_rag/eval/triage_gate.py` to module-level constant `_EVAL_QUESTIONS` eliminating per-query object allocations.
- Replaced $O(K)$ chunk popping in `DocumentIngester` with $O(1)$ list slicing `current_chunk[-keep_count:]`.

---

## [1.1.0] - 2026-10-06 (Architecture Deepening & Zero-Seam Refactoring)

### Added
- **Deep Document Ingestion Subsystem (`src/lattice_rag/ingestion/ingester.py`):**
  - Created `DocumentIngester` deep module encapsulating sentence-aware chunking, batch ONNX embedding generation, GLiNER entity-relation extraction, LatticeDB property-graph persistence, and Tier 1 semantic cache invalidation.
  - Reduced ONNX embedding cross-process invocations from $N$ chunks to a single batched call per document for all unique entities, eliminating repetitive model round-trips.
  - Re-exported `DocumentIngester` cleanly via `src/lattice_rag/ingestion/__init__.py`.
  - Registered `provide_ingester` provider in `app.py` for Litestar dependency injection.
- **Deep Hybrid Retriever Subsystem (`src/lattice_rag/retrieval/retriever.py`):**
  - Built `HybridRetriever` unifying 3-stage search (HNSW+BM25 with RRF, dynamic Cypher graph traversal, cross-encoder precision reranking) and Jev Noul context guardrails behind a single high-leverage interface.
  - Streamlined `retrieval/pipeline.py` and `benchmark.py` to delegate to `HybridRetriever`, eliminating leaked stage coordination and fragmented timing calculation logic.
- **Deep Resilient Synthesizer Subsystem (`src/lattice_rag/generation/resilient_synthesizer.py`):**
  - Created `ResilientSynthesizer` encapsulating full multi-tier failover logic across GroqCloud (under `pybreaker.CircuitBreaker`), Google AI Studio Gemini 2.5 Flash, and local Redis FAQ fallback.
  - Unified buffered (`synthesize`) and token-by-token streaming (`stream`) generation pipelines with exact parity in circuit breaker tripping and fallback triggers.
- **Unified Orchestration Parity (`src/lattice_rag/orchestration/graph.py`):**
  - Refactored `RAGOrchestrator` to delegate generative synthesis and failover to `ResilientSynthesizer` in both `_generate_node` and `stream_query`.
  - Eliminated duplicate circuit-breaker cascade code between buffered and SSE streaming query paths.
- **Comprehensive TDD Test Suite (16 New Unit Tests):**
  - Added `tests/unit/test_document_ingester.py` (4 tests) verifying edge cases, single/multi-chunk flows, and batch entity embedding efficiency.
  - Added `tests/unit/test_hybrid_retriever.py` (4 tests) verifying end-to-end 3-stage coordination, empty results, and guardrail integration.
  - Added `tests/unit/test_resilient_synthesizer.py` (5 tests) verifying Groq normal execution, circuit breaker tripping, Gemini fallback, and Redis FAQ cascades for both buffered and streaming paths.
  - Added `tests/unit/test_orchestrator_unified.py` (3 tests) verifying buffered and streaming synthesis parity.
  - Expanded test suite to **124 tests (100% green, 0 warnings)**.

### Changed
- Refactored `IngestController` in `src/lattice_rag/api/controllers/ingest.py` from 89 lines down to a lean controller delegating entirely to `DocumentIngester`.
- Refactored CLI commands `ingest` and `seed` in `src/lattice_rag/cli.py` to use `DocumentIngester`, deleting ~80 lines of duplicate orchestration.
- Sanitized mock test tokens in `tests/unit/test_security_audit.py` to prevent false positive triggers during automated git secret scans.

---

## [1.0.0] - 2026-10-06 (Phase 7: Production Verification, Benchmarking, Performance Profiling & Release Hardening)

### Added
- **Production Benchmark Engine & Dual-SLA Validation (`src/lattice_rag/benchmark.py`):**
  - Built comprehensive micro-benchmarking engine profiling Tier 1 Semantic Cache, Stage 1 Hybrid Retrieval (HNSW+BM25), Stage 2 Dynamic Cypher Traversal, Stage 3 Cross-Encoder Reranking, and End-to-End pipeline.
  - Implemented exact statistical percentiles ($p50, p90, p95, p99$), arithmetic mean, standard deviation, min/max bounds, and throughput (QPS).
  - Enforced dual production SLAs:
    - **Sub-Second Multi-Hop SLA**: End-to-End $p95 < 1,000$ms (achieved **$545.4$ms**).
    - **Tier 1 Cache Latency SLA**: In-memory cache hit $p95 < 25$ms (achieved **$1.2$ms** with $> 4,000$ QPS).
  - Rendered clean ASCII tabular output and structured JSON export formats.
- **Sequential Multi-Threaded INT8 ONNX Optimization:**
  - Configured `onnxruntime.SessionOptions` with `intra_op_num_threads = min(4, cpu_count)`, `ORT_SEQUENTIAL` execution, and `ORT_ENABLE_ALL` graph optimizations for `BAAI/bge-reranker-v2-m3-ONNX`.
  - Added exact-match and near-identical fast-path to `SemanticCache.get()` eliminating redundant remote LLM calls for identical queries.
  - Optimized candidate reranking and linearized graph context capping to deliver a 15x CPU inference speedup.
- **REST API & Visual Studio Benchmark Runner:**
  - Created `BenchmarkController` exposing `POST /api/v1/benchmark/run` with configurable iterations and warmup runs returning `BenchmarkRunResponse` DTO.
  - Integrated "Run Performance Benchmark" button into `EvalMatrix.tsx` in the web studio, displaying live SLA pass/fail badges and stage-by-stage latency percentiles table.
- **Comprehensive Security Audit Suite (`tests/unit/test_security_audit.py`):**
  - Verified RFC 9457 Problem Details error shielding prevents internal stack traces and credential exposure.
  - Validated OpenAPI schema inspection to guarantee secret keys are excluded from public API specs.
  - Enforced msgspec DTO isolation separating wire transfer models from internal storage representations.
  - Tested `RedactingFilter` regex sanitization across log records and error message details.
  - Added automated git tree scanner verifying zero plaintext keys or `.env` files are tracked.
- **Production Operations Documentation:**
  - Authored `docs/performance-benchmark.md` documenting empirical latency percentiles, memory footprint, and architectural comparison against cloud vector DBs.
  - Authored `docs/production-readiness.md` operator runbook covering lifecycle, hot backups, health probes, disaster recovery, and scaling strategies.

---

## [0.6.0] - 2026-10-06 (Phase 6: Embedded Interactive Visual Playground & Evaluation UI)

### Added
- **Embedded Visual Engineering Playground & Studio (`frontend/`):**
  - Built single-page React 18 + TypeScript + Vite + Tailwind CSS studio mounted into Litestar at root `/` with zero external frontend hosting dependencies.
  - Split-screen layout uniting real-time SSE streaming answer generation on the left with an interactive 2D knowledge graph on the right.
  - Live dynamic token velocity meter (`tok/s`) measuring synthesis throughput continuously on every SSE token chunk event alongside elapsed time and total tokens.
  - Interactive Horizontal Gantt-style execution trace waterfall visualizing exact durations (`ms`) and percentage proportions across all 6 pipeline stages (Front-Door Triage, Stage 1 Hybrid, Stage 2 Traversal, Stage 3 Cross-Encoder Rerank, Jev Noul Guardrail, Generative Synthesis).
  - Grounded source chunk inspector drawer displaying retrieved text chunks, positions, doc IDs, and similarity scores.
- **Interactive 2D Knowledge Graph Visualizer (`GraphExplorer.tsx`):**
  - Integrated `d3-force` physics simulation with SVG rendering for crisp typography, custom styled entity nodes (`Database`, `Technology`, `Algorithm`, `Protocol`, `Concept`), and directed relation edges.
  - Added mode toggle between **"Query Traversed Path"** and **"Full Knowledge Graph"** with dynamic neighborhood highlighting and slide-out node inspector panel.
  - Implemented `GET /api/v1/graph/subgraph?limit=50` controller endpoint and `LatticeStore.get_subgraph_snapshot()` method.
- **CI/CD Evaluation Matrix & Benchmark Dashboard (`EvalMatrix.tsx`):**
  - Visualized the 20-query golden benchmark suite (`eval_dataset.json`) with summary KPI cards for Mean Faithfulness, Context Precision, Answer Relevance, and CI/CD Gate Status.
  - Integrated "Trigger Live Evaluation Gate" button calling `POST /api/v1/eval/run` with real-time runner status.
  - Added filterable data grid by query text or pass/fail status, expandable side-by-side metric comparison, and "Test in Studio" click-to-test navigation.
- **Document Ingestion Drawer (`IngestModal.tsx`):**
  - Slide-out ingestion UI with 5 pre-built corpus presets (LatticeDB, FastEmbed, GLiNER, TypeSafe, Litestar), live ingestion progress, and returned chunk/entity/relation counts.
- **Zero-Cost Deployment & Containerization:**
  - Multi-stage `Dockerfile` (Node 22 build -> Python 3.12-slim) pre-downloading ONNX embedding/reranker models and pre-seeding LatticeDB with the golden corpus.
  - Cloud Run deployment automation script `scripts/deploy_cloud_run.sh` configured for 2GB RAM container scaling to zero ($0.00/month for ~225,000 queries).
  - Comprehensive zero-cost hosting guide `docs/deployment.md` comparing Google Cloud Run, Cloudflare Tunnel, and OCI Always Free.
  - CLI `lattice-rag seed --dataset eval_dataset.json` command for pre-seeding knowledge graphs.
- **Graceful Static Hosting in Litestar:**
  - Configured `create_static_files_router` at `/` with `html_mode=True` when `frontend/dist` is present, with an automated fallback HTML landing page pointing to OpenAPI Scalar docs when unbuilt.

---

## [0.5.0] - 2026-10-05 (Phase 5: Golden Benchmark Dataset & Automated CI/CD Evaluation Gate)

### Added
- **Jev-Native Continuous Evaluation Engine (`JevEvaluator`):**
  - Engineered continuous expected-value semantic scoring using TypeSafe AI Jev `Score` primitive across 5-level descriptive criteria (0.0 to 1.0) for Faithfulness, Context Precision, and Answer Relevance.
  - Replaced non-deterministic LLM-as-a-judge temperature drift and fragile JSON output parsers with mathematically calibrated discrete probability distributions ($E[S] = \sum p_i \cdot s_i / 4.0$).
  - Implemented hard-zero contradiction veto using Jev `Noul` ($P(\text{contradiction}) \ge 0.40 \implies \text{faithfulness} = 0.0$) preventing ungrounded hallucinations from slipping past the gate.
  - Executed all 3 RAG Triad metrics plus the contradiction check in a single atomic TypeSafe System 1 request (~300ms) with zero generative token costs.
- **Hermetic Ephemeral Evaluation Runner (`EvalRunner`):**
  - Isolated evaluation runs into ephemeral temporary directories (`tempfile.TemporaryDirectory()`), indexing the evaluation corpus into an isolated SQLite-backed LatticeDB without mutating or locking production `data/lattice_rag.db`.
  - Implemented concurrent batch query execution through the complete `RAGOrchestrator` pipeline.
  - Engineered independent regression detection checking $\Delta = \text{Metric}_{\text{run}} - \text{Metric}_{\text{baseline}} \ge -0.03$ across all metrics, plus a hard per-query floor check ($\ge 0.50$).
- **Version-Controlled 20-Query Golden Dataset & Baseline (`eval_dataset.json`, `eval_baseline.json`):**
  - Authored a 5-document technical corpus spanning LatticeDB, FastEmbed, GLiNER, TypeSafe AI, and Litestar.
  - Formulated 20 multi-hop queries across `vector_exact`, `graph_relational`, and `hybrid` routes with expected entity anchors and ground-truth references.
  - Tracked main branch baseline metrics: Faithfulness (0.88), Context Precision (0.85), Answer Relevance (0.90), Max Allowed Regression Delta (0.03), and Floor (0.50).
- **Web API & CLI Evaluation Surface:**
  - Added `EvalController` (`POST /api/v1/eval/run`) with `FromQuery` parameters for dataset and baseline paths, returning camelCase `EvalRunResponse` DTOs.
  - Wired `lattice-rag eval --dataset ... --baseline ... --fail-on-regression` CLI command rendering ASCII tabular per-query telemetry, metric means, regression deltas, and CI/CD exit codes.
- **GitHub Actions CI/CD Quality Gate (`.github/workflows/eval-gate.yml`):**
  - Built two-tier workflow executing 97 hermetic unit and integration tests across Ubuntu and macOS runners, running live TypeSafe Jev quality gate checks on pull requests.
- **Automated Test Suite Expansion:**
  - Added 10 new tests across `test_eval_triage.py`, `test_eval_runner.py`, `test_eval_controller.py`, and `test_phase5_eval_live.py`.
  - Test suite now stands at 97 tests, 100% green with 0 warnings.

---

## [0.4.0] - 2026-10-05 (Phase 4: Generation Model Orchestration & Web API Layer)

### Added
- **Primary Synthesizer (`GroqSynthesizer` with `qwen/qwen3.8-27b`):**
  - Integrated GroqCloud's ultra-fast Qwen 3.8 27B model (131k context window, thinking and instruct capabilities) delivering 200+ tok/s.
  - Implemented structured citation grounding (`[Chunk i (Doc: ...)]` and `[Knowledge Graph Context]`) instructing the model to ground all assertions strictly in evidence.
  - Added secret sanitization wrapping API exceptions via `sanitize_error_detail`.
- **Context Fallback Synthesizer (`GeminiFallback` with `gemini-3.8-flash`):**
  - Integrated Google AI Studio `gemini-3.8-flash` for massive cross-document context (>100k tokens) and multi-level circuit breaker cascade.
  - Implemented dual-mode async execution: synchronous awaiting and token-by-token async generator streaming.
- **LangGraph Agentic Orchestrator (`RAGOrchestrator`):**
  - Built unified stateful pipeline coordinating query routing, Tier 1 semantic vector cache verification, 3-stage hybrid retrieval, generation, and multi-tier failover.
  - Implemented typed Server-Sent Events (SSE) streaming yielding `event: stage`, `event: token`, and `event: done` for real-time frontend waterfalls.
  - Implemented full cascading failover: Groq (under `pybreaker.CircuitBreaker`) $\to$ Gemini 3.8 Flash $\to$ Tier 2 Embedded Redis FAQ hash with fuzzy and token-overlap matching.
  - Integrated automatic post-synthesis Tier 1 cache insertion for verified responses.
- **Rust-Optimized Web Stack (Litestar 2.24+ ASGI & Granian):**
  - `QueryController` (`POST /api/v1/query`, `POST /api/v1/query/stream`) with `NamedDependency[SkipValidation[...]]` pattern preventing msgspec DI type inspection conflicts.
  - `IngestController` (`POST /api/v1/ingest`) featuring sentence-aware chunking (~512 chars, 64-char overlap), dense vector embedding, GLiNER named entity recognition, LatticeDB graph storage, and Tier 1 cache invalidation.
  - `CacheController` (`GET /api/v1/cache/stats`) returning real-time hits, misses, and circuit breaker status.
  - `HealthController` (`GET /health`) verifying connectivity to LatticeDB, Redis, and validation of all API keys.
  - RFC 9457 Problem Details error shielding with zero-trust token redaction (`sanitize_error_detail`).
  - Production-ready Granian server runner in `lattice_rag.server` configured for high-concurrency ASGI serving.
- **Automated Test Suite & Live E2E Integration:**
  - Added 21 new tests across `test_groq_synthesizer.py`, `test_gemini_fallback.py`, `test_orchestration_graph.py`, `test_api_controllers.py`, and `test_phase4_e2e.py`.
  - Full end-to-end integration test (`test_phase4_e2e.py`) validating the entire 7-step user journey against real LatticeDB, real FastEmbed ONNX embeddings, real TypeSafe AI, and live GroqCloud inference.
  - All 87 unit and integration tests passing 100% green with 0 warnings.

### Fixed
- **Retrieval Pipeline Logging Contract:** Replaced standard Python `logging.Logger` with `structlog.get_logger` across `pipeline.py`, `graph_traversal.py`, `reranker.py`, and `vector_search.py`, resolving `TypeError: Logger._log() got an unexpected keyword argument`.
- **ProcessPool Teardown on macOS:** Added explicit `shutdown_pool()` cleanup in `test_pool.py` and session fixture in `conftest.py`, eliminating `libc++abi: recursive_mutex lock failed` on interpreter exit.
- **Third-Party Warning Filter:** Configured pytest filterwarnings in `pyproject.toml` to silence upstream PyTorch JIT and Hugging Face Hub deprecation warnings.
- **GLiNER Model Availability Fallback:** Added graceful automatic fallback in `EntityExtractor` to public `urchade/gliner_small-v2.1` when gated repositories return 401 Unauthorized.

---

## [0.3.0] - 2026-10-05 (Phase 3: 3-Stage Hybrid Retrieval Pipeline & Context Guardrails)

### Added
- **State-of-the-Art Cross-Encoder Reranking (`bge-reranker-v2-m3` INT8 ONNX):**
  - Integrated `onnx-community/bge-reranker-v2-m3-ONNX` with an 8,192 token context window, 568M parameter quality, and a compact 544MB INT8 footprint.
  - Benchmarked sub-100ms CPU inference using dynamic token batch padding in `onnxruntime`.
  - Built unified `RerankerAdapter` in `EmbeddingService` supporting both `bge-reranker-v2-m3` (default) and FastEmbed `bge-reranker-base`.
- **Stage 1 Hybrid Searcher (HNSW Dense + BM25 Lexical via RRF):**
  - Implemented concurrent HNSW vector and BM25 index search merged via Reciprocal Rank Fusion ($k=60$) with node deduplication and non-blocking `asyncio.to_thread` execution.
- **Stage 2 Dynamic Graph Traverser:**
  - Implemented rank-ordered entity anchor extraction via `get_entities_for_chunk()`, dynamic 1-hop or 2-hop traversal (`route == "graph_relational"`), and a 25-node budget cap.
- **Stage 3 Precision Cross-Encoder Reranking with Graph Linearization:**
  - Linearized `SubgraphResult` into formatted relational triples (`Entity1 --[RELATION]--> Entity2`, capped at 15 triples / 1000 chars) appended to candidate chunks.
- **Guardrail Integration & Low Grounding Telemetry:**
  - Added `low_grounding: bool = False` to `RetrievalResult`, short-circuiting empty search queries and capturing Jev Noul relevance rejections.
- **Live TypeSafe AI & Phase 3 Test Suite:**
  - Added 17 new automated tests in `test_embeddings.py`, `test_vector_search.py`, `test_graph_traversal.py`, `test_reranker.py`, `test_pipeline.py`, and `test_retrieval_integration.py`.
  - Verified live TypeSafe AI Jev Noul guardrail filtering with sub-second total pipeline execution (~598ms).
  - All 66 test cases passing 100% green.

### Fixed
- **LatticeStore Parameter Mismatch:** Corrected `limit=top_k * 2` to `top_k=top_k * 2` in `vector_search.py`.
- **Missing Query Method in Graph Traversal:** Replaced non-existent `store.query()` with `store.get_entities_for_chunk()` and `store.traverse_from_entities()`.
- **Reranker Output Contract:** Corrected float score unpacking from cross-encoders without assuming missing `.corpus_id` attributes.
- **Silent Graph Discarding:** Fixed `reranker.py` attribute checks to properly inspect `SubgraphResult.nodes` and `edges`.

---

## [0.2.0] - 2026-10-05 (Phase 2 Verification & Hardening)


### Added
- **Embedded In-Process Redis (`fakeredis`):**
  - Integrated `fakeredis.aioredis` directly into `FallbackCache`, delivering microsecond in-memory key-value lookups with zero external Redis server dependencies and zero Redis API keys.
  - Implemented `seed_from_file()` with `fallback_queries.json` containing 50 curated technical Q&A pairs covering LatticeDB, Litestar, FastEmbed, GLiNER, and TypeSafe AI.
  - Added multi-tier query matching in `find_closest_fallback` combining exact matching, sequence similarity, and token overlap with stop-word pruning.
- **Asynchronous Circuit Breaker (`pybreaker`):**
  - Configured `CircuitBreaker(fail_max=3, reset_timeout=30s)` wrapping async generative calls via `call_with_breaker()`.
  - Trips from `closed` to `open` on 3 consecutive LLM failures, blocking further remote calls and directing traffic to the embedded fallback cache.
- **Strict Context Guardrail with Insufficient Evidence Injection:**
  - Configured batched TypeSafe AI Jev `Noul` evaluations ($\ge 0.50$ threshold) with automatic `low_grounding_flag` tracking.
  - Injected an `INSUFFICIENT_EVIDENCE` sentinel chunk when all retrieved chunks fail grounding, preventing LLM hallucination.
- **Phase 2 Automated Test Suite:**
  - Added 37 new unit tests across `test_routing.py`, `test_guardrail.py`, `test_caching.py`, and `test_chitchat.py`, bringing the test suite to 49 passing tests (100% green).

### Fixed
- **TypeSafe SDK Async Cleanup:** Fixed invalid `.close()` calls to `await client.aclose()` across `QueryRouter`, `ContextGuardrail`, and `SemanticCache`.
- **TypeSafe Noul Semantic Cache Contract:** Replaced invalid `.noul(...)` method with `client.system_one(state=..., questions={"equivalent": Noul(...)})`.
- **Chitchat Substring Collision:** Fixed naive substring matching in `ChitchatHandler` by applying regular expression word boundaries (`\b`) to prevent false triggers (e.g. "hi" inside "something").

---

## [0.1.0] - 2026-10-05


### Added
- **Rust-Optimized Web Stack:**
  - Integrated **Litestar 2.24+** ASGI framework served by **Granian** for 2–4x throughput over Uvicorn.
  - Built typed API contracts using **msgspec.Struct** with `rename="camel"` for sub-millisecond JSON serialization.
  - Implemented OpenAPI documentation powered by **Scalar** accessible at `/schema/scalar`.
  - Added `QueryController`, `IngestController`, `CacheController`, and `HealthController`.

- **In-Process Graph & Vector Engine (LatticeDB):**
  - Integrated **LatticeDB 0.15** embedded single-file property-graph engine (`data/lattice_rag.db`).
  - Hierarchical schema: `(:Document)-[:HAS_CHUNK]->(:Chunk)`, `(:Chunk)-[:CONTAINS]->(:Entity)`, and `(:Entity)-[:RELATION]->(:Entity)`.
  - Native HNSW vector index on chunk and entity embeddings (`<=>` operator) with 384 dimensions.
  - Native BM25 inverted full-text index on chunk text and entity names (`@@` operator).
  - Sub-millisecond multi-hop graph traversal expanding up to 2 hops with an entity budget cap of 25 nodes.

- **Local Quantized SLMs (FastEmbed & GLiNER):**
  - `EmbeddingService` wrapping **FastEmbed** ONNX models (`BAAI/bge-small-en-v1.5`) for local CPU dense and sparse vector generation.
  - Cross-Encoder precision reranker using `BAAI/bge-reranker-v2-m3`.
  - `EntityExtractor` wrapping Fastino Labs' **GLiNER2.5-Decide** (340M parameters) on CPU for local entity recognition and sentence-level triple extraction.

- **System 1 Decision Routing & Guardrails (TypeSafe AI Jev):**
  - Front-door traffic cop using Jev `Choice` primitive (70–500ms) to classify queries into `vector_exact`, `graph_relational`, `hybrid`, `chitchat`, or `massive_context`.
  - Context guardrail using Jev `Noul` primitive to batch-filter candidate chunks and prune tangential noise before generative synthesis.

- **Multi-Tier Latency & Fallback Caching:**
  - **Tier 1:** In-memory semantic vector cache comparing query cosine similarity (>= 0.90) and verifying true semantic identity via Jev Noul (< 20ms response time).
  - **Tier 2:** `pybreaker.CircuitBreaker` wrapping generative calls with cascading failover: Groq $\to$ Gemini 2.5 Flash $\to$ Redis Hash (`cache:fallback:queries`) storing pre-computed top 50 domain queries.

- **Generation Model Orchestration:**
  - Primary synthesis using **GroqCloud** (Llama 3.3 70B / DeepSeek-R1) streaming at 200+ tokens/s via Server-Sent Events (SSE) and JSON.
  - Context fallback using **Google AI Studio** (Gemini 2.5 Flash) for queries requiring massive context analysis (> 100k tokens).
  - Deterministic `ChitchatHandler` returning immediate structured conversational responses at zero cost.

- **ProcessPool CPU Isolation:**
  - Dedicated `ProcessPoolExecutor` with `run_in_pool()` to offload heavy ONNX embeddings and GLiNER extraction, preventing event loop starvation.

- **Command-Line Interface (CLI):**
  - CLI commands implemented via Click: `lattice-rag setup-keys`, `serve`, `query`, `ingest`, `benchmark`, and `eval`.

### Changed
- Standardized package build backend to `setuptools.build_meta` with `where = ["src"]` package discovery.
- Pinned `vector_dimensions=384` on LatticeDB initialization to align with FastEmbed BGE models.

### Fixed
- **LatticeDB Transaction Commit:** Fixed critical bug where `latticedb` write transactions silently rolled back on context exit unless `txn.commit()` was explicitly invoked.
- **LatticeDB Database Lifecycle:** Ensured explicit `self.db.open()` is called during `LatticeStore` initialization.
- **Editable Install Discovery:** Configured `.pth` linking for local package imports across Python environments.

### Security
- **Zero Git Leaks:** Hardened `.gitignore` to strictly exclude `.env`, `apikeys.md`, `*.key`, `*.pem`, and `data/*.db`.
- **Secret Masking:** Implemented `SecretStr` masking tokens as `sk-****{last_4}` in `str()` and `repr()`.
- **Log Sanitizer:** Added `RedactingFilter` to redact Authorization headers and Bearer tokens from structured logs.
- **RFC 9457 Error Shielding:** Sanitized outbound error responses to prevent credential leakage.
- **Safe Key Migration:** Implemented `setup-keys` CLI command to migrate `apikeys.md` into `chmod 600 .env` and securely shred the plaintext file.
