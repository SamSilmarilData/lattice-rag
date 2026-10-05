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
  stream?: boolean;
  filterTags?: string[];
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

export interface EvalQueryResult {
  query: string;
  faithfulness: number;
  contextPrecision: number;
  answerRelevance: number;
  passed: boolean;
}

export interface EvalRunResponse {
  totalQueries: number;
  meanFaithfulness: number;
  meanContextPrecision: number;
  meanAnswerRelevance: number;
  passedGate: boolean;
  regressionDelta: number;
  results: EvalQueryResult[];
}

export type StreamEvent =
  | { event: 'stage'; data: { stage: string; status: string; durationMs?: number } }
  | { event: 'token'; data: { delta: string } }
  | {
      event: 'done';
      data: {
        answer: string;
        route: string;
        latencyMs: number;
        sources?: SourceChunk[];
        graphPath?: SubgraphDTO;
        timings?: StageTiming[];
        cached?: boolean;
      };
    }
  | { event: 'error'; data: { detail: string } };

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
