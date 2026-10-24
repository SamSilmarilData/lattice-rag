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
}

export interface IngestRequest {
  documentId: string;
  title: string;
  text: string;
  tags?: string[];
}

export interface IngestResponse {
  documentId: string;
  chunkCount: number;
  entityCount: number;
  relationCount: number;
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
export type EvalQueryError = 'pipeline_error' | 'judge_error';

export interface EvalQueryResult {
  query: string;
  passed: boolean;
  faithfulness: number | null;
  contextPrecision: number | null;
  answerRelevance: number | null;
  answerCorrectness: number | null;
  error: EvalQueryError | null;
}

// Means are over the queries where the metric was judged (null if none were)
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
