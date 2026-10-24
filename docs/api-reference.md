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
| `GET` | `/api/v1/graph/subgraph` | **admin** | query `limit` (1–500, default 50) | `SubgraphDTO` | Knowledge graph snapshot for the visualizer. |
| `GET` | `/api/v1/cache/stats` | **admin** | — | `CacheStatsResponse` | Tier 1 hit/miss counts and generation circuit-breaker state. |
| `POST` | `/api/v1/eval/jobs` | **admin** | query `dataset`, `baseline` (bare `*.json` names in the project root), `limit`, `use_temp_db` (default `true`) | `202` `EvalJobResponse` | Start an evaluation run in the background and return its handle immediately. `409` if a run is already in progress. |
| `GET` | `/api/v1/eval/jobs/{jobId}` | **admin** | — | `EvalJobResponse` | Status of an Eval Job and, once it has succeeded, its `EvalRunResponse` report. `404` for an unknown or expired handle. |
| `POST` | `/api/v1/benchmark/run` | **admin** | — | `BenchmarkRunResponse` | Retrieval-only latency benchmark (6 iterations, 2 warmup) with SLA checks. |
| `GET` | `/health` | — | — | `HealthResponse` | Health and configuration status. |

### 2.1 Authentication & Limits
- **Public endpoints** (no key): `/`, `/health`, `POST /api/v1/query` and `POST /api/v1/query/stream`.
- **Admin-only endpoints**: `POST /api/v1/ingest`, `GET /api/v1/graph/subgraph`, `GET /api/v1/cache/stats`, `POST /api/v1/eval/jobs`, `GET /api/v1/eval/jobs/{jobId}` and `POST /api/v1/benchmark/run`.
- **Admin endpoints** require `X-Admin-Key: <ADMIN_API_KEY>`. A missing or wrong key returns `401`; if `ADMIN_API_KEY` is unset on the server, these endpoints are disabled. After 5 wrong keys within 5 minutes a client gets `429` for the rest of that window.
- **Query endpoints** are rate-limited per client (`QUERY_RATE_LIMIT_PER_MINUTE`, default 60) and return `429` when exceeded. Clients are identified by the socket peer, or by `X-Forwarded-For` when `TRUSTED_PROXY_HOPS` is set (see `docs/deployment.md`).
- **Validation** failures return `400` problem details: `query` must be 1–2,000 characters, `text` 1–200,000, `documentId` must match `[A-Za-z0-9_.-]{1,128}`, and request bodies are capped at 1 MB.
- **Eval runs are serialized.** Only one Eval Job runs at a time per server process; a start request while one is running is rejected with `409` (it is not queued), so parallel requests cannot multiply judge and LLM API cost. Poll `GET /api/v1/eval/jobs/{jobId}` until `status` leaves `running`. The 20 most recent jobs are kept in memory and are lost on restart. A failed job's `error` carries only a reference id for the server logs. The `lattice-rag eval` CLI runs in its own process and does not go through this queue.
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

### 3.6 `EvalJobResponse`, `EvalRunResponse` & `EvalQueryResult`
```typescript
interface EvalJobResponse {
  jobId: string;                           // Handle for GET /api/v1/eval/jobs/{jobId}
  status: "running" | "succeeded" | "failed";
  startedAt: string;                       // ISO 8601, UTC
  finishedAt: string | null;               // Set once the job is no longer running
  report: EvalRunResponse | null;          // Set when status is "succeeded"
  error: string | null;                    // Generic message with a log reference when status is "failed"
}

// Each metric is a Jev Score normalized to [0.0, 1.0], or null when it was not judged:
// it does not apply to the query's Slice (chitchat: relevance + correctness only;
// unanswerable: no context precision), or the query failed and was never judged.
interface EvalQueryResult {
  query: string;                     // Evaluated benchmark query
  passed: boolean;                   // Judged, and every judged metric >= its pass threshold
  faithfulness: number | null;       // Answer vs retrieved context; 0.0 on a contradiction veto
  contextPrecision: number | null;   // Retrieved context vs what the reference answer needs
  answerRelevance: number | null;    // Answer vs the query
  answerCorrectness: number | null;  // Answer vs the Gold Set reference answer
  error: "pipeline_error" | "judge_error" | null;  // Set when the query failed; it is then not judged
}

interface EvalRunResponse {
  totalQueries: number;                // Total evaluated queries
  passedGate: boolean;                 // See the gate rules below
  regressionDelta: number;             // Worst mean delta vs the baseline's measured metrics
  meanFaithfulness: number | null;     // Means are over queries where the metric was judged
  meanContextPrecision: number | null;
  meanAnswerRelevance: number | null;
  meanAnswerCorrectness: number | null;
  failedQueries: number;               // Queries with an error (not judged, excluded from means)
  results: EvalQueryResult[];          // Granular query evaluation results
}
```

The gate rules live in `eval_baseline.json` under `gate_thresholds` and all are required:
`passedGate` is false when any query failed, when any judged metric of any query is below
`min_query_floor`, or when a mean drops more than `max_regression_delta` below its value in
`metrics`. A metric absent from `metrics` is reported but not regression-checked.
`pass_thresholds` sets the per-query, per-metric threshold behind `passed`.

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
