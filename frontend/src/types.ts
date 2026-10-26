export interface SourceChunk {
  chunkId: number;
  text: string;
  score: number;
  documentId: string;
  position: number;
}

export interface GraphNode {
  nodeId: number;
  label: string;
  name: string;
  properties?: Record<string, unknown>;
  // UI Simulation properties
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphEdge {
  sourceId: number;
  targetId: number;
  relationType: string;
  // UI Simulation references (can be populated by d3-force with node references)
  source?: GraphNode | number;
  target?: GraphNode | number;
}

export interface SubgraphDTO {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface StageTiming {
  stage: string;
  durationMs: number;
}

export interface QueryRequest {
  query: string;
}

export interface QueryResponse {
  answer: string;
  route: string;
  cached: boolean;
  latencyMs: number;
  sources: SourceChunk[];
  graphPath?: SubgraphDTO | null;
  timings: StageTiming[];
  degraded: boolean;
  lowGrounding: boolean;
  // Unverified Answer: a Jev check (routing or guardrail) could not run; never cached
  unverified: boolean;
}

export interface IngestRequest {
  documentId: string;
  title: string;
  text: string;
  tags?: string[];
  replace?: boolean;
}

export interface IngestResponse {
  documentId: string;
  chunkCount: number;
  entityCount: number;
  relationCount: number;
  persistence: string;
}

export interface CacheStatsResponse {
  tier1Hits: number;
  tier1Misses: number;
  tier1CachedQueries: number;
  tier2CircuitStatus: string;
  tier2CachedQueries: number;
}

export interface HealthResponse {
  status: string;
  latticedbConnected: boolean;
  redisConnected: boolean;
  typesafeConfigured: boolean;
  groqConfigured: boolean;
  geminiConfigured: boolean;
}

// A metric is null when it was not judged: it does not apply to the query's
// Slice, or the query failed (error is set) and was never sent to the judge.
export type EvalQueryError = 'pipeline_error' | 'judge_error' | 'provider_degraded';

export type GenerationProvider = 'groq' | 'gemini' | 'redis_faq' | 'offline';

// Mirror the Route and Slice Literals in src/lattice_rag/vocabulary.py
export type Route = 'vector_exact' | 'graph_relational' | 'hybrid' | 'chitchat' | 'massive_context';
export type Slice = 'factoid' | 'multihop' | 'unanswerable' | 'chitchat' | 'long_context';

export interface EvalQueryResult {
  query: string;
  passed: boolean;
  faithfulness: number | null;
  contextPrecision: number | null;
  answerRelevance: number | null;
  answerCorrectness: number | null;
  error: EvalQueryError | null;
  // null when the pipeline failed; routeVerified is false for the System 1 fallback route
  chosenRoute: Route | null;
  routeVerified: boolean;
  // Provider that generated the answer; null when nothing was generated
  // (pipeline error, chitchat, cache hit)
  provider: GenerationProvider | null;
  // Graph Recall: share of the query's expected entities among the graph context's Entities;
  // null when the query lists none or the pipeline failed. Unjudged and advisory.
  graphRecall: number | null;
}

// A query whose chosen Route differs from its Expected Route
export interface Misroute {
  queryId: string;
  slice: Slice;
  chosenRoute: Route;
  expectedRoute: Route;
}

// Means are over the queries where the metric was judged (null if none were).
// routeAccuracy is over routeAccuracyScored queries: outside the chitchat Slice, pipeline ran,
// and Jev chose the Route. Unverified fallbacks and pipeline errors are counted, not scored.
// Its floor is advisory and never fails the gate.
// graphRecall is the mean per-query Graph Recall over graphRecallScored queries (those with
// expected entities whose pipeline ran). Unjudged and advisory: it never fails the gate.
export interface EvalRunResponse {
  totalQueries: number;
  passedGate: boolean;
  regressionDelta: number;
  meanFaithfulness: number | null;
  meanContextPrecision: number | null;
  meanAnswerRelevance: number | null;
  meanAnswerCorrectness: number | null;
  failedQueries: number;
  results: EvalQueryResult[];
  routeAccuracy: number | null;
  routeAccuracyBySlice: Record<string, number>;
  routeAccuracyFloor: number | null;
  routeAccuracyScored: number;
  routeAccuracyUnverified: number;
  routeAccuracyErrored: number;
  misroutes: Misroute[];
  graphRecall: number | null;
  graphRecallBySlice: Record<string, number>;
  graphRecallScored: number;
}

export type EvalJobStatus = 'running' | 'succeeded' | 'failed';

// POST /api/v1/eval/jobs (202) and GET /api/v1/eval/jobs/{jobId}
export interface EvalJobResponse {
  jobId: string;
  status: EvalJobStatus;
  startedAt: string;
  finishedAt: string | null;
  report: EvalRunResponse | null;
  error: string | null;
}

// SSE payloads of POST /api/v1/query/stream (mirror the Stream* DTOs in api/dtos.py)
export interface StreamStage {
  stage: string;
  status?: string;
  route?: string;
  chunksCount?: number;
}

export interface StreamToken {
  delta: string;
}

export type StreamDone = QueryResponse;

export interface StreamError {
  detail: string;
}

export type StreamEvent =
  | { event: 'stage'; data: StreamStage }
  | { event: 'token'; data: StreamToken }
  | { event: 'done'; data: StreamDone }
  | { event: 'error'; data: StreamError };

export interface StageBenchmarkDTO {
  stage: string;
  samplesCount: number;
  meanMs: number;
  minMs: number;
  p50Ms: number;
  p90Ms: number;
  p95Ms: number;
  p99Ms: number;
  maxMs: number;
  qps: number;
}

export interface BenchmarkRunResponse {
  totalDurationSec: number;
  subSecondSlaMet: boolean;
  tier1CacheSlaMet: boolean;
  stages: StageBenchmarkDTO[];
}
