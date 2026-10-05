# **Project Specification: 2026 Production GraphRAG & Eval Suite (V4 Zero-Cost Architecture)**

# **1\. Executive Summary & Core Objective**

The objective is to engineer a mathematically defensible, hyper-optimized Hybrid GraphRAG pipeline that executes multi-hop relational reasoning in under a second. By abandoning heavy JVM databases and cloud vector stores in favor of an entirely in-process, single-binary architecture, this system achieves zero-latency data traversal and absolute zero cloud hosting costs. This specification defines a production-ready stack optimized for 2026 hardware and software realities.

# **2\. API Framework & Orchestration (The Rust-Optimized Web Stack)**

Standard ASGI configurations (FastAPI/Uvicorn) introduce Python-level Global Interpreter Lock (GIL) bottlenecks when handling heavy concurrent ML workloads. We bypass this entirely by using a Rust-optimized stack.

* **Web Server (Granian):** The application is served by Granian, a Rust-based HTTP server that delivers 2–4x the throughput of Uvicorn.  
* **API Framework (Litestar):** We replace FastAPI with Litestar, the highest-performance ASGI framework in 2026, which natively integrates with Data Transfer Objects (DTOs) for zero-boilerplate validation.  
* **Serialization (msgspec):** Pydantic is replaced by msgspec, a Rust-backed serialization library that doubles JSON parsing speeds.  
* **Orchestration (LangGraph & ProcessPoolExecutor):** LangGraph's StateGraph manages the agentic loop. To prevent the single-threaded asynchronous event loop from freezing, all CPU-bound machine learning tasks (embeddings, graph extraction) are strictly isolated in a ProcessPoolExecutor.

# **3\. High-Speed Decision Routing (TypeSafe AI Jev)**

Instead of feeding every query into an expensive generative LLM for evaluation, we employ TypeSafe AI Jev, a "System One" decision model, as a deterministic traffic cop.

* **Concurrent Execution:** Jev routing and cache verification run concurrently via asyncio.gather() to eliminate sequential network hops.  
* **The Choice Primitive (Front-Door Router):** Jev evaluates the incoming prompt in 70–500ms, routing exact technical lookups to vector search, relational logic to graph traversal, and conversational noise to a static chitchat handler.  
* **The Noul Primitive (Context Guardrail):** Before the retrieved chunks are sent to the final generative LLM, Jev processes a batched payload of the chunks. Using the boolean Noul primitive, it ruthlessly filters out tangential noise, drastically reducing the LLM's context window and generation costs.

# **4\. Ingestion & Storage (The In-Process Paradigm)**

We eliminate the network boundary between the application and the database. The system uses a single, embedded file for all search types.

* **Storage Engine (LatticeDB):** Operating like SQLite for connected data, LatticeDB handles Cypher graph traversals, BM25 full-text search, and HNSW vector similarity search in a single local file, achieving sub-millisecond vector recall.  
* **CPU-Optimized Embeddings (FastEmbed):** Dense and sparse vectors are generated locally using quantized ONNX models via FastEmbed, requiring zero GPU provisioning.  
* **Local Graph Extraction (GLiNER2.5-Decide):** An asynchronous local worker runs Fastino Labs' 340M-parameter GLiNER2.5-Decide model on the CPU. It performs lightning-fast triple (entity-relationship) extraction, streaming the nodes directly into LatticeDB.

# **5\. The Retrieval Pipeline (Single-Binary Hybrid Search)**

* **Stage 1 (Vector Candidate Recall):** FastEmbed queries LatticeDB's HNSW vector index to rapidly isolate initial entity candidates based on cosine similarity.  
* **Stage 2 (Relational Traversal):** LatticeDB immediately pivots from the vector results to traverse the explicit relationships expanding around those entities using standard Cypher queries, capturing the multi-hop context.  
* **Stage 3 (Precision Reranking):** FastEmbed runs a CPU-quantized Cross-Encoder (e.g., BGE-Reranker-v2) to evaluate and compress the expanded graph paths into the definitive top 5 chunks.

# **Tier 1 (Semantic Vector Cache): Evaluates query intent locally. If a semantic match is found, Jev's Noul primitive verifies the semantic alignment and instantly returns the cached context, bypassing the database layer.6. Multi-Tiered Latency & Fallback Caching**

*   
* **Tier 2 (Circuit Breaker Hash Map):** A local pybreaker instance monitors the generative LLM endpoint. If the API fails or throttles, it trips the breaker and falls back to a pre-computed Redis hash map of the top 50 domain queries.

# **7\. Generation Model Orchestration**

* **Primary Synthesizer:** GroqCloud free tier (Llama 3.3 or DeepSeek-R1). It receives the rigorously filtered LatticeDB graph context and streams the final synthesized response at 200+ tokens per second.  
* **Context Fallback:** Google AI Studio (Gemini 2.5 Flash) is reserved solely for queries that Jev flags as requiring massive cross-document context window analysis (up to 1M tokens).

# **8\. CI/CD Evaluation Gate**

* **Triage & Quality Gate:** Jev uses the Score primitive to run a high-speed, low-cost evaluation pass in GitHub Actions. DeepEval invokes an LLM-as-a-judge for edge cases. Pull requests automatically fail if the regression delta for Faithfulness or Context Precision drops below the main branch baseline (e.g., Score\_PR \>= Score\_main \- 0.03).Golden Dataset: A strictly version-controlled eval\_dataset.json containing 20 complex multi-hop queries with ground-truth answers.