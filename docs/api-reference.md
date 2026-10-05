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

| Method | Path | Request Body | Response Body | Description |
|---|---|---|---|---|
| `GET` | `/` | — | `text/html` | Root Embedded Visual Playground & Studio SPA (or fallback landing). |
| `POST` | `/api/v1/query` | `QueryRequest` | `QueryResponse` | Synchronous RAG query endpoint. |
| `POST` | `/api/v1/query/stream` | `QueryRequest` | `text/event-stream` | Real-time SSE token & event stream. |
| `POST` | `/api/v1/ingest` | `IngestRequest` | `IngestResponse` | Document ingestion & graph construction. |
| `GET` | `/api/v1/graph/subgraph` | — (Query param: `limit`) | `SubgraphDTO` | Knowledge graph snapshot (nodes & edges) for 2D visualizer. |
| `GET` | `/api/v1/cache/stats` | — | `CacheStatsResponse` | Hit/miss metrics for Tier 1 & 2 caches. |
| `POST` | `/api/v1/eval/run` | — (Query params) | `EvalRunResponse` | Run CI/CD evaluation gate against benchmark dataset. |
| `GET` | `/health` | — | `HealthResponse` | System health and connectivity telemetry. |

---

## 3. Data Transfer Objects (DTOs)

All DTOs are declared as `msgspec.Struct` with `rename="camel"` for zero-copy JSON parsing.

### 3.1 `QueryRequest`
```typescript
interface QueryRequest {
  query: string;           // The natural language question
  stream?: boolean;        // Whether to stream tokens (default: false)
  filterTags?: string[];   // Optional document tag filters
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
  degraded: boolean;               // True if circuit breaker failover was activated
}
```

### 3.3 `SourceChunk` & `SubgraphDTO`
```typescript
interface SourceChunk {
  chunkId: number;
  text: string;
  score: number;
  documentId: string;
  position: number;
}

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
  tags?: string[];         // Metadata tags
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
  tier2CircuitStatus: "closed" | "open" | "half-open";
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
  passed: boolean;             // True if all metric floors (>= 0.50) are met
}

interface EvalRunResponse {
  totalQueries: number;         // Total evaluated queries
  meanFaithfulness: number;     // Average faithfulness across all queries
  meanContextPrecision: number; // Average context precision across all queries
  meanAnswerRelevance: number;  // Average answer relevance across all queries
  passedGate: boolean;          // True if max regression delta >= -0.03 and all floors met
  regressionDelta: number;      // Actual worst-case delta compared to baseline
  results: EvalQueryResult[];   // Granular query evaluation results
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

**Stream Event Structure:**
```
event: stage
data: {"stage": "routing", "route": "vector_exact", "confidence": 0.99}

event: stage
data: {"stage": "retrieval", "chunks": 4, "nodes": 3, "edges": 1}

event: stage
data: {"stage": "guardrail", "retained_chunks": 1}

event: token
data: {"token": "LatticeDB"}

event: token
data: {"token": " unites"}

event: token
data: {"token": " Cypher, HNSW, and BM25 in a single binary [Chunk 0]."}

event: done
data: {"latencyMs": 528.2, "cached": false, "route": "vector_exact"}
```
