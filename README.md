# lattice-rag

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Litestar 2.24](https://img.shields.io/badge/Litestar-2.24-211A44?logo=fastapi&logoColor=white)](https://litestar.dev/)
[![Granian](https://img.shields.io/badge/Granian-Rust_ASGI-DEA584?logo=rust&logoColor=white)](https://github.com/emmett-framework/granian)
[![LatticeDB](https://img.shields.io/badge/LatticeDB-In--Process_Graph-00ADD8)](https://github.com/latticedb/latticedb)
[![TypeSafe AI](https://img.shields.io/badge/TypeSafe_AI-Jev_System_1-4B32C3)](https://typesafe.ai/)
[![Release](https://img.shields.io/badge/Release-v1.2.0-success.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-131%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **In-process, zero-cloud-cost Hybrid GraphRAG engine and CI/CD evaluation suite.**  
> Built on **LatticeDB**, **Litestar (Rust/Granian)**, **FastEmbed**, **GLiNER2.5-Decide**, and **TypeSafe AI Jev**. Features sub-second multi-hop relational traversal, deterministic System 1 routing, multi-tiered fallback caching, and automated CI/CD regression gates.

---

## 🚀 Overview & 2026 Zero-Cost Architecture

Traditional GraphRAG architectures suffer from heavy JVM database footprints (Neo4j), expensive remote vector stores (Pinecone, Qdrant), and sequential LLM network hops that push query latencies into multiple seconds and incur steep cloud hosting costs.

`lattice-rag` eliminates these bottlenecks through an **entirely in-process, single-binary architecture**:

```mermaid
flowchart TD
    UserQuery([User Query / Ingestion]) --> LitestarServer[Litestar 2.24 + Granian Rust ASGI]
    
    subgraph FrontDoorTriage [Concurrent Front-Door Triage]
        LitestarServer --> Gather[asyncio.gather]
        Gather --> CacheCheck[Tier 1: Semantic Vector Cache]
        Gather --> JevRouter[TypeSafe AI Jev: Choice Primitive Router]
    end
    
    CacheCheck -- "Match >= 0.90 & Jev Noul Verified" --> InstantReturn[Instant Return < 20ms]
    
    JevRouter -- "'chitchat'" --> StaticResponse[Instant Static Response]
    JevRouter -- "'massive_context'" --> GeminiFlag[Flag Gemini 2.5 Flash]
    JevRouter -- "'vector_exact' | 'graph_relational' | 'hybrid'" --> LangGraphLoop[LangGraph Agentic StateGraph]
    
    subgraph ProcessPoolIsolation [ProcessPoolExecutor CPU Isolation]
        FastEmbedEmbed[FastEmbed: ONNX Dense/Sparse Embeddings]
        GLiNERWorker[GLiNER2.5-Decide: Local Triple Extraction]
        CrossEncoder[FastEmbed: BGE-Reranker-v2-m3 Cross-Encoder]
    end
    
    LangGraphLoop --> Stage1[Stage 1: LatticeDB HNSW Vector + BM25 Recall]
    Stage1 --> Stage2[Stage 2: LatticeDB Cypher 1-2 Hop Traversal max 25 nodes]
    Stage2 --> Stage3[Stage 3: Cross-Encoder Precision Reranking]
    
    Stage3 --> Guardrail[TypeSafe AI Jev: Noul Guardrail Noise Pruning]
    
    Guardrail --> CircuitBreaker{pybreaker Circuit Breaker}
    CircuitBreaker -- "Normal" --> GroqLLM[GroqCloud: Llama 3.3 / DeepSeek-R1 @ 200+ tok/s]
    CircuitBreaker -- "Groq Outage / 429" --> GeminiFallback[Cascade 1: Gemini 2.5 Flash]
    GeminiFallback -- "Gemini Outage / Timeout" --> RedisFallback[Cascade 2: Tier 2 Redis Hash Top 50 FAQ]
    GeminiFlag --> GeminiLLM[Google AI Studio: Gemini 2.5 Flash]
    
    GroqLLM --> SSEStream[Litestar SSE Stream / JSON Response]
    GeminiFallback --> SSEStream
    RedisFallback --> SSEStream
    GeminiLLM --> SSEStream
    StaticResponse --> SSEStream
```

---

## ✨ Key Architectural Highlights

| Subsystem | Technology | Capability |
|---|---|---|
| **Web Server** | **Granian** | Rust-based HTTP server delivering 2–4x the throughput of standard Uvicorn. |
| **API Framework** | **Litestar 2.24+** | High-performance ASGI framework with native DTOs, dependency injection, and Scalar OpenAPI. |
| **Serialization** | **msgspec** | Rust-backed struct serialization doubling JSON encode/decode performance. |
| **Storage Engine** | **LatticeDB 0.15+** | Embedded SQLite-like property-graph file combining native HNSW vector index (`<=>`) and BM25 text index (`@@`). |
| **Local SLMs** | **FastEmbed + GLiNER** | Quantized ONNX dense/sparse embeddings and local entity-relationship triple extraction on CPU. |
| **Cross-Encoder** | **bge-reranker-v2-m3** | High-precision INT8 ONNX reranker with 8,192 token context window. |
| **System 1 Routing** | **TypeSafe AI Jev** | 70–500ms `Choice` primitive routing + batched `Noul` boolean context noise pruning. |
| **Multi-Tier Cache** | **Vector + Redis Hash** | Tier 1 local semantic vector cache (<20ms); Tier 2 `pybreaker` circuit breaker cascading to Redis Top 50 FAQ. |
| **Primary Synthesis** | **GroqCloud** | Ultra-fast token streaming (200+ tok/s) via `qwen/qwen3.8-27b` with structured citation grounding. |
| **Context Fallback** | **Gemini 3.8 Flash** | Massive context window (>100k tokens) reserved for cross-document synthesis and circuit breaker failover. |
| **CI/CD Quality Gate** | **TypeSafe Jev Score + Noul** | Automated 20-query evaluation gate blocking PRs on regression ($\Delta \ge -0.03$), floor drop ($< 0.50$), and contradiction veto. |
| **Secret Hygiene** | **Zero-Trust Security** | `SecretStr` masking, log redaction, RFC 9457 error shielding, and automated `apikeys.md` migration. |

---

## 📦 Directory Structure

```
lattice-rag/
├── pyproject.toml               # Python 3.12 dependencies and package configuration
├── README.md                    # Project documentation and quickstart
├── CHANGELOG.md                 # Version history and architectural log
├── LICENSE                      # MIT License
├── .gitignore                   # Strict boundary blocking .env, apikeys.md, and databases
├── .env.example                 # Sanitized configuration template
├── eval_dataset.json            # Version-controlled 20-query golden benchmark dataset
├── eval_baseline.json           # Tracked main-branch baseline metrics & gate thresholds
├── .github/workflows/           # GitHub Actions CI/CD workflows
│   └── eval-gate.yml            # Two-tier PR regression gate (hermetic tests + live Jev evaluation)
├── docs/                        # In-depth architectural guides
│   ├── architecture.md          # Multi-layer system architecture
│   ├── security.md              # Zero-trust secret management
│   ├── storage-and-schema.md    # LatticeDB graph and index modeling
│   ├── api-reference.md         # Litestar endpoints & DTO contracts
│   ├── deployment.md            # Zero-cost production hosting runbook
│   ├── performance-benchmark.md # Empirical latency profiles & dual SLA verification
│   └── production-readiness.md  # Operator runbooks, health probes & disaster recovery
├── src/lattice_rag/
│   ├── config.py                # Environment configuration with SecretStr validation
│   ├── security.py              # Secret masking, log redaction, apikeys migration
│   ├── app.py                   # Litestar application factory & ApplicationCore
│   ├── server.py                # Granian ASGI runner entrypoint
│   ├── cli.py                   # Command-line interface (Click)
│   ├── benchmark.py             # Empirical benchmark engine & SLA verification
│   ├── api/
│   │   ├── dtos.py              # msgspec Structs (camelCase wire contracts)
│   │   └── controllers/         # Query, Ingestion, Cache, Health, Eval & Benchmark controllers
│   ├── ingestion/
│   │   └── ingester.py          # Unified document ingester: chunking, batch ONNX embeddings, GLiNER, LatticeDB
│   ├── storage/
│   │   ├── db.py                # LatticeStore embedded database manager
│   │   └── extract.py           # GLiNER2.5-Decide local triple extraction
│   ├── routing/
│   │   ├── router.py            # TypeSafe AI Jev Choice front-door router
│   │   └── guardrail.py         # TypeSafe AI Jev Noul context pruning
│   ├── caching/
│   │   ├── semantic_cache.py    # Tier 1 Semantic Vector Cache (<20ms)
│   │   └── fallback_cache.py    # Tier 2 pybreaker + Redis Hash top 50 FAQ
│   ├── retrieval/
│   │   ├── retriever.py         # Deep HybridRetriever: unifies 3-stage search, guardrail, timings
│   │   ├── embeddings.py        # FastEmbed dense/sparse ONNX models
│   │   ├── vector_search.py     # Stage 1: HNSW vector + BM25 RRF recall
│   │   ├── graph_traversal.py   # Stage 2: Dynamic 1-to-2 hop Cypher traversal
│   │   ├── reranker.py          # Stage 3: Cross-encoder precision reranking
│   │   └── pipeline.py          # Legacy/internal 3-stage retrieval pipeline
│   ├── generation/
│   │   ├── resilient_synthesizer.py # Deep ResilientSynthesizer: multi-tier failovers (Groq -> Gemini -> Redis)
│   │   ├── groq_synthesizer.py  # Primary Groq streaming generator (qwen/qwen3.8-27b @ 200+ tok/s)
│   │   ├── gemini_fallback.py   # Context fallback generator (gemini-3.8-flash)
│   │   └── chitchat.py          # Instant deterministic conversational handler
│   ├── orchestration/
│   │   ├── pool.py              # Centralized ProcessPoolExecutor for CPU ML tasks
│   │   ├── state.py             # LangGraph typed PipelineState
│   │   └── graph.py             # RAGOrchestrator StateGraph pipeline
│   └── eval/
│       ├── triage_gate.py       # TypeSafe AI Jev Score & Noul continuous evaluation engine
│       └── runner.py            # Ephemeral isolated database evaluation runner & regression gate
└── tests/
    ├── unit/                    # Hermetic unit tests (eval, storage, security, config, routing, generation, caching, ingestion, retrieval)
    └── integration/             # Live E2E tests (LatticeDB + FastEmbed + Groq + TypeSafe + Phase 7 Benchmarks)
```

---

## ⚡ Quickstart

### 1. Prerequisites
- **macOS** or **Linux**
- **Python 3.12** (`/opt/homebrew/bin/python3.12` or system python)
- **Redis** (optional, for Tier 2 fallback cache: `brew install redis && brew services start redis`)

### 2. Environment Setup

Clone the repository and create a Python 3.12 virtual environment:

```bash
git clone https://github.com/samyakmeshram/lattice-rag.git
cd lattice-rag

# Create and activate Python 3.12 venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e ".[dev]"
```

### 3. Configure API Credentials

Copy the environment template:

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` and set your credentials:
```bash
# TypeSafe AI Jev (Required in production)
TYPESAFE_API_KEY=your_typesafe_key_here

# GroqCloud (Primary generation at 200+ tok/s)
GROQ_API_KEY=your_groq_key_here

# Google AI Studio (Massive context fallback)
GEMINI_API_KEY=your_gemini_key_here

# Redis (Tier 2 fallback cache)
REDIS_URL=redis://localhost:6379/0
```

> [!TIP]
> **Safe Key Migration Utility:**
> If you have a temporary `apikeys.md` file, run:
> ```bash
> lattice-rag setup-keys
> ```
> This extracts all keys into `.env` with strict `0600` permissions and automatically shreds `apikeys.md` so plaintext secrets never linger.

---

## 💻 CLI Commands

The `lattice-rag` CLI provides end-to-end operational controls:

```bash
# Inspect all available commands
lattice-rag --help

# Migrate temporary apikeys.md into secure .env
lattice-rag setup-keys

# Start the Granian Rust ASGI server on port 8000
lattice-rag serve --host 0.0.0.0 --port 8000

# Execute a query directly via CLI
lattice-rag query "How does LatticeDB combine HNSW and BM25 search?"

# Pre-seed the embedded knowledge graph from the golden corpus
lattice-rag seed --dataset eval_dataset.json

# Ingest a text document into the embedded knowledge graph
lattice-rag ingest path/to/document.txt

# Run latency benchmarks on the retrieval pipeline
lattice-rag benchmark

# Run the 20-query CI/CD evaluation gate against baseline
lattice-rag eval --dataset eval_dataset.json --baseline eval_baseline.json
```

---

## 🎨 Embedded Visual Engineering Studio & Playground

When running `lattice-rag serve`, navigate to `http://localhost:8000/` to open the built-in single-origin studio:

- **Split-Screen Studio**: Left pane features query input, SSE streaming token synthesis, real-time dynamic velocity meter (`tok/s`), horizontal Gantt-style execution trace waterfall, and Jev-verified grounded sources.
- **Interactive 2D Knowledge Graph Visualizer**: Right pane renders force-directed SVG graphs (`d3-force`) with zoom/pan, node inspection drawers, and a toggle between **Query Traversed Path** and **Full Knowledge Graph**.
- **CI/CD Evaluation Matrix**: Interactive dashboard displaying per-query Faithfulness, Context Precision, and Relevance scores across the 20-query golden dataset, with a one-click **"Run Live Evaluation Gate"** runner and click-to-test navigation.
- **Document Ingestion Drawer**: Live document text chunking, embedding, entity extraction, and LatticeDB linking with immediate graph visualizer refresh.

---

## ☁️ Zero-Cost Production Deployment

`lattice-rag` can be hosted for **$0.00/month** in production without downgrading our state-of-the-art `bge-reranker-v2-m3` cross-encoder.

See [`docs/deployment.md`](file:///Users/samyakmeshram/Documents/GitHub/lattice-rag/docs/deployment.md) for full guides covering:
1. **Google Cloud Run (Always Free Tier - Recommended)**: Serverless container with 2GB RAM scaling to zero, consuming $0.00 within Google's monthly 360,000 GiB-seconds free tier (~225,000 queries/month free). Deploy via `./scripts/deploy_cloud_run.sh`.
2. **Cloudflare Tunnel (`cloudflared`)**: Run locally on Apple Silicon (sub-100ms rerank) and expose globally with free HTTPS and DDoS protection via `cloudflared tunnel --url http://localhost:8000`.
3. **Oracle Cloud Infrastructure (OCI Always Free)**: 4 ARM vCPUs + 24GB RAM persistent Ubuntu VM running 24/7 with zero cold starts.

---

## 🌐 API & Web Service

Launch the Granian server:

```bash
lattice-rag serve
```

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root Embedded Visual Playground & Studio SPA (or graceful fallback). |
| `POST` | `/api/v1/query` | Standard JSON query endpoint returning answer, sources, graph path, and latency breakdown. |
| `POST` | `/api/v1/query/stream` | Server-Sent Events (SSE) streaming real-time tokens and stage progress. |
| `POST` | `/api/v1/ingest` | Ingests document text: chunks, computes ONNX embeddings, extracts entities, and commits to LatticeDB. |
| `GET` | `/api/v1/graph/subgraph` | Subgraph snapshot (nodes & edges) for the interactive 2D knowledge graph. |
| `GET` | `/api/v1/cache/stats` | Telemetry on Tier 1 semantic vector cache hits and Tier 2 circuit breaker status. |
| `POST` | `/api/v1/eval/run` | Triggers the CI/CD evaluation gate with custom dataset/baseline paths and regression tolerance. |
| `POST` | `/api/v1/benchmark/run` | Executes statistical performance benchmarking with dual SLA verification ($p50, p90, p95, p99$, QPS). |
| `GET` | `/health` | Service health, LatticeDB connection state, and API configuration flags. |
| `GET` | `/schema/scalar` | Interactive OpenAPI documentation powered by Scalar. |

---

## ⚡ Performance Profiling & Dual-SLA Verification

`lattice-rag` provides strict sub-second performance guarantees verified by an empirical benchmarking harness:

```bash
# Run CLI benchmark harness with dual SLA verification
lattice-rag benchmark --iterations 5 --warmup 2 --sla-check
```

### Empirical Production Benchmark Results

| Pipeline Stage | p50 (Median) | p90 | p95 (SLA Target) | p99 | Mean | Throughput |
|---|---|---|---|---|---|---|
| **Tier 1 Semantic Cache** | **0.2ms** | 1.1ms | **1.1ms** (<25ms SLA) | 1.1ms | 0.3ms | **3,346 QPS** |
| **Stage 1: Hybrid Retrieval (HNSW+BM25)** | 11.4ms | 11.9ms | 11.9ms | 11.9ms | 11.0ms | 91.1 QPS |
| **Stage 2: Dynamic Cypher Traversal** | 1.4ms | 13.8ms | 13.8ms | 13.8ms | 5.4ms | 186.9 QPS |
| **Stage 3: Cross-Encoder Rerank (v2-m3)** | 308.6ms | 338.5ms | 338.5ms | 338.5ms | 314.4ms | 3.2 QPS |
| **End-to-End Pipeline (Full StateGraph)** | **547.1ms** | 557.0ms | **557.0ms** (<1000ms SLA) | 557.0ms | 549.4ms | **1.8 QPS** |

- **Sub-Second Multi-Hop SLA**: $p95 = 557.0\text{ms} < 1,000\text{ms}$ **[PASSED]**
- **Tier 1 Cache Latency SLA**: $p95 = 1.1\text{ms} < 25\text{ms}$ **[PASSED]**

For detailed architectural benchmarks, memory consumption profiles (stable at 340 MB RSS), and comparison against cloud vector architectures, see [`docs/performance-benchmark.md`](file:///Users/samyakmeshram/Documents/GitHub/lattice-rag/docs/performance-benchmark.md). For operational runbooks and health probes, see [`docs/production-readiness.md`](file:///Users/samyakmeshram/Documents/GitHub/lattice-rag/docs/production-readiness.md).

---

## 🧪 Testing & Verification

Run the automated test suite across unit and integration suites:

```bash
# Run full test suite (124 tests, 100% green, 0 warnings)
pytest tests/ -v
```

All 124 tests execute cleanly with 0 warnings:
- **Hermetic Unit Tests (112 tests)**: Security secret masking, RFC 9457 error shielding, log redaction, config post-init validation, fast embeddings, dynamic Cypher traversal, Stage 3 triples capping, Jev triage & contradiction veto, semantic cache cosine similarity, Redis circuit breaker transitions, API controllers, unified document ingestion with batched embeddings, deep hybrid retrieval, resilient synthesis cascading failover, and unified orchestrator streaming/buffered parity.
- **Integration Tests (12 tests)**: Live end-to-end RAG pipeline, live CI/CD regression gate, benchmark engine SLA verification, and benchmark API endpoints.

---

## 🔒 Security & Secret Hygiene

- **Zero Git Leaks:** `.env`, `apikeys.md`, `*.key`, and `data/*.db` are hard-blocked in `.gitignore`.
- **Masked Runtime Logs:** Keys wrapped in `SecretStr` are masked as `sk-****{last_4}` in all `str()` and `repr()` representations.
- **Log Sanitizer:** `RedactingFilter` regex-scrubs Bearer tokens and API keys from logging outputs.
- **RFC 9457 Error Shielding:** Exception handlers sanitize outbound error responses to ensure auth headers never leak to clients.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
