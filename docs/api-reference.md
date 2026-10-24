# API & DTO Contract Reference

`lattice-rag` provides a high-throughput REST and Server-Sent Events (SSE) streaming API built on **Litestar 2.24+** and **Granian**.

---

## 1. Interactive Documentation

An interactive Scalar OpenAPI documentation portal is automatically served at:
```
http://localhost:8000/schema/scalar
```

---

## 2. Endpoints Overview

| Method | Path | Auth | Request | Response Body | Description |
|---|---|---|---|---|---|
| `GET` | `/` | — | — | `text/html` | Studio SPA (or fallback landing page). |
| `POST` | `/api/v1/query` | rate-limited | `QueryRequest` | `QueryResponse` | Synchronous RAG query. |
| `POST` | `/api/v1/query/stream` | rate-limited | `QueryRequest` | `text/event-stream` | SSE stream of stage, token, error and done events. |
| `POST` | `/api/v1/ingest` | **admin** | `IngestRequest` | `IngestResponse` | Ingest a document. `409` if `documentId` already exists. |
| `GET` | `/api/v1/graph/subgraph` | — | query `limit` (1–500, default 50) | `SubgraphDTO` | Knowledge graph snapshot for the visualizer. |
| `GET` | `/api/v1/cache/stats` | — | — | `CacheStatsResponse` | Tier 1 hit/miss counts and generation circuit-breaker state. |
| `POST` | `/api/v1/eval/run` | **admin** | query `dataset`, `baseline` (bare `*.json` names in the project root), `limit`, `use_temp_db` (default `true`) | `EvalRunResponse` | Run the evaluation gate. |
| `POST` | `/api/v1/benchmark/run` | **admin** | — | `BenchmarkRunResponse` | Retrieval-only latency benchmark (6 iterations, 2 warmup) with SLA checks. |
| `GET` | `/health` | — | — | `HealthResponse` | Health and configuration status. |

### 2.1 Authentication & Limits
- **Admin endpoints** require `X-Admin-Key: <ADMIN_API_KEY>`. A missing or wrong key returns `401`; if `ADMIN_API_KEY` is unset on the server, these endpoints are disabled. After 5 wrong keys within 5 minutes a client gets `429` for the rest of that window.
- **Query endpoints** are rate-limited per client (`QUERY_RATE_LIMIT_PER_MINUTE`, default 60) and return `429` when exceeded. Clients are identified by the socket peer, or by `X-Forwarded-For` when `TRUSTED_PROXY_HOPS` is set (see `docs/deployment.md`).
- **Validation** failures return `400` problem details: `query` must be 1–2,000 characters, `text` 1–200,000, `documentId` must match `[A-Za-z0-9_.-]{1,128}`, and request bodies are capped at 1 MB.
- **Unexpected errors** return `500` with a generic `detail` and a `requestId` to correlate with server logs; exception text is never returned.

---

## 3. Data Transfer Objects (DTOs)

All DTOs are declared as `msgspec.Struct` with `rename="camel"` for zero-copy JSON parsing.

### 3.1 `QueryRequest`
```typescript
interface QueryRequest {
  query: string;           // The natural language question (1–2,000 chars)
}
```

### 3.2 `QueryResponse`
```typescript
interface QueryResponse {
  answer: string;                  // Synthesized LLM response
  route: string;                   // 'vector_exact' | 'graph_relational' | 'hybrid' | 'chitchat' | 'massive_context'
  cached: boolean;                 // Whether Tier 1 semantic cache was hit
  latencyMs: number;               // Total execution latency in milliseconds
  sources: SourceChunk[];          // Attributed source text chunks
  graphPath?: SubgraphDTO;         // Traversed entity graph nodes and edges
  timings: StageTiming[];          // Stage latency breakdown (triage, retrieval, generation)
  degraded: boolean;               // True if the answer came from Gemini fallback or the FAQ cache
  lowGrounding: boolean;           // True if no retrieved chunk passed the relevance guardrail
}
```

### 3.3 `SourceChunk` & `SubgraphDTO`
```typescript
interface SourceChunk {
  chunkId: number;         // LatticeDB node id
  text: string;
  score: number;           // Cross-encoder score (fused RRF score if not reranked)
  documentId: string;
  position: number;        // Chunk position within its document
}
// The answer cites sources as [Chunk N], where N is the index in `sources`.

interface SubgraphDTO {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface GraphNode {
  nodeId: number;
  label: string;
  name: string;
  properties?: Record<string, unknown>;
}

interface GraphEdge {
  sourceId: number;
  targetId: number;
  relationType: string;
}
```

### 3.4 `IngestRequest` & `IngestResponse`
```typescript
interface IngestRequest {
  documentId: string;      // Unique identifier for the document
  title: string;           // Document title
  text: string;            // Raw document text content
  tags?: string[];         // Up to 32 tags (1–64 chars), stored on the Document node
}

interface IngestResponse {
  documentId: string;
  chunkCount: number;      // Number of chunks stored in LatticeDB
  entityCount: number;     // Number of entities extracted by GLiNER
  relationCount: number;   // Number of relationships established
}
```

### 3.5 `CacheStatsResponse` & `HealthResponse`
```typescript
interface CacheStatsResponse {
  tier1Hits: number;
  tier1Misses: number;
  tier1CachedQueries: number;
  tier2CircuitStatus: "closed" | "open" | "half-open";  // Groq circuit breaker (shared by /query and /query/stream)
  tier2CachedQueries: number;
}

interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  latticedbConnected: boolean;
  redisConnected: boolean;
  typesafeConfigured: boolean;
  groqConfigured: boolean;
  geminiConfigured: boolean;
}
```

### 3.6 `EvalRunResponse` & `EvalQueryResult`
```typescript
interface EvalQueryResult {
  query: string;               // Evaluated benchmark query
  faithfulness: number;        // Jev Score normalized expected value [0.0, 1.0]
  contextPrecision: number;    // Jev Score normalized expected value [0.0, 1.0]
  answerRelevance: number;     // Jev Score normalized expected value [0.0, 1.0]
  passed: boolean;             // faithfulness >= 0.70, contextPrecision >= 0.65, answerRelevance >= 0.70, no contradiction veto
}

interface EvalRunResponse {
  totalQueries: number;         // Total evaluated queries
  meanFaithfulness: number;     // Average faithfulness across all queries
  meanContextPrecision: number; // Average context precision across all queries
  meanAnswerRelevance: number;  // Average answer relevance across all queries
  passedGate: boolean;          // No mean metric dropped > 0.03 vs baseline and every query's faithfulness >= 0.50
  regressionDelta: number;      // Actual worst-case delta compared to baseline
  results: EvalQueryResult[];   // Granular query evaluation results
}
```

### 3.7 `StageBenchmarkDTO` & `BenchmarkRunResponse`
```typescript
interface StageBenchmarkDTO {
  stage: string;          // Name of the pipeline stage profiled
  samplesCount: number;   // Number of iterations executed
  meanMs: number;         // Arithmetic mean latency (ms)
  minMs: number;          // Minimum latency recorded (ms)
  p50Ms: number;          // 50th percentile (median) latency (ms)
  p90Ms: number;          // 90th percentile latency (ms)
  p95Ms: number;          // 95th percentile latency (ms)
  p99Ms: number;          // 99th percentile latency (ms)
  maxMs: number;          // Maximum latency recorded (ms)
  qps: number;            // Queries / Operations per second throughput
}

interface BenchmarkRunResponse {
  totalDurationSec: number;     // Total benchmark suite execution time
  subSecondSlaMet: boolean;     // Retrieval-only E2E p95 < 1,000ms (no routing, guardrail or LLM)
  tier1CacheSlaMet: boolean;    // Tier 1 exact-hit p95 < 25ms
  stages: StageBenchmarkDTO[];  // Stage-by-stage latency percentiles
}
```

---

## 4. Server-Sent Events (SSE) Protocol

When querying `/api/v1/query/stream`, events are streamed in standard SSE format:

```http
POST /api/v1/query/stream HTTP/1.1
Content-Type: application/json

{"query": "What is the relation between LatticeDB and FastEmbed?"}
```

**Stream Event Structure:** every `data:` line is a camelCase JSON object matching one of the `Stream*` DTOs in `api/dtos.py` (mirrored in `frontend/src/types.ts`).

```
event: stage
data: {"stage":"triage","status":"started"}

event: stage
data: {"stage":"retrieval","route":"hybrid"}

event: stage
data: {"stage":"generation","chunksCount":5}

event: token
data: {"delta":"LatticeDB"}

event: token
data: {"delta":" unites HNSW and BM25 in a single binary [Chunk 0]."}

event: done
data: {"answer":"LatticeDB unites ...","route":"hybrid","cached":false,"latencyMs":528.2,"sources":[...],"graphPath":{...},"timings":[...],"degraded":false,"lowGrounding":false}
```

| Event | Payload | When |
|---|---|---|
| `stage` | `{stage, status?, route?, chunksCount?}` | A pipeline stage starts (`triage`, `cache_hit`, `chitchat`, `retrieval`, `generation`). |
| `token` | `{delta}` | Next piece of answer text. |
| `error` | `{detail}` | Generation stopped early (the provider failed after its first token), or the request failed (`detail` carries a reference id). |
| `done` | Same fields as `QueryResponse` | Final event of a completed stream. |
