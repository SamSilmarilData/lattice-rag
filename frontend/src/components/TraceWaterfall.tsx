import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Clock } from 'lucide-react';
import { StageTiming } from '../types';

interface TraceWaterfallProps {
  timings: StageTiming[];
  totalLatencyMs: number;
}

const STAGE_CONFIG: Record<
  string,
  { label: string; color: string; bg: string; border: string; description: string }
> = {
  triage: {
    label: 'Front-Door Triage',
    color: 'text-indigo-400',
    bg: 'bg-indigo-500',
    border: 'border-indigo-500/30',
    description: 'TypeSafe Jev Choice classification and Tier 1 semantic cache probe.',
  },
  stage_1_hybrid: {
    label: 'Stage 1: Hybrid Retrieval',
    color: 'text-cyan-400',
    bg: 'bg-cyan-500',
    border: 'border-cyan-500/30',
    description: 'Concurrent HNSW dense vector search + BM25 keyword recall merged via RRF (k=60).',
  },
  stage_2_traversal: {
    label: 'Stage 2: Graph Traversal',
    color: 'text-amber-400',
    bg: 'bg-amber-500',
    border: 'border-amber-500/30',
    description: 'Dynamic 1-to-2 hop LatticeDB Cypher traversal expanding from retrieved entity anchors.',
  },
  stage_3_rerank: {
    label: 'Stage 3: Cross-Encoder Rerank',
    color: 'text-emerald-400',
    bg: 'bg-emerald-500',
    border: 'border-emerald-500/30',
    description: 'INT8 ONNX bge-reranker-v2-m3 scoring chunk texts + linearized graph triples.',
  },
  guardrail: {
    label: 'Context Guardrail',
    color: 'text-purple-400',
    bg: 'bg-purple-500',
    border: 'border-purple-500/30',
    description: 'TypeSafe Jev Noul factual necessity check pruning irrelevant context.',
  },
  synthesis: {
    label: 'Generative Synthesis',
    color: 'text-orange-400',
    bg: 'bg-orange-500',
    border: 'border-orange-500/30',
    description: 'Streaming token generation with grounded source citations.',
  },
};

export const TraceWaterfall: React.FC<TraceWaterfallProps> = ({ timings, totalLatencyMs }) => {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!timings || timings.length === 0) {
    return null;
  }

  const effectiveTotal = Math.max(
    totalLatencyMs,
    timings.reduce((sum, t) => sum + t.durationMs, 0),
    0.1
  );

  return (
    <div className="bg-card border border-border rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between pb-2 border-b border-border/50">
        <div className="flex items-center space-x-2">
          <Clock className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-semibold text-gray-200">Execution Trace Waterfall</h3>
        </div>
        <div className="text-[11px] font-mono text-gray-400">
          Total Latency:{' '}
          <span className="text-cyan-400 font-semibold">{totalLatencyMs.toFixed(1)} ms</span>
        </div>
      </div>

      {/* Waterfall Rows */}
      <div className="space-y-2">
        {timings.map((timing) => {
          const cfg = STAGE_CONFIG[timing.stage] || {
            label: timing.stage,
            color: 'text-gray-300',
            bg: 'bg-gray-500',
            border: 'border-gray-500/30',
            description: 'Pipeline stage execution.',
          };

          const pct = Math.min(100, Math.max(2, (timing.durationMs / effectiveTotal) * 100));
          const isSelected = expanded === timing.stage;

          return (
            <div
              key={timing.stage}
              className="group cursor-pointer rounded-lg hover:bg-surface/40 p-1.5 transition"
              onClick={() => setExpanded(isSelected ? null : timing.stage)}
            >
              <div className="flex items-center justify-between text-xs mb-1">
                <div className="flex items-center space-x-1.5">
                  {isSelected ? (
                    <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
                  ) : (
                    <ChevronRight className="w-3.5 h-3.5 text-gray-500 group-hover:text-gray-300" />
                  )}
                  <span className={`font-medium ${cfg.color}`}>{cfg.label}</span>
                </div>
                <div className="flex items-center space-x-2 font-mono text-[11px]">
                  <span className="text-gray-300">{timing.durationMs.toFixed(1)} ms</span>
                  <span className="text-gray-500">({pct.toFixed(0)}%)</span>
                </div>
              </div>

              {/* Progress Bar Track */}
              <div className="w-full h-2 bg-background/80 rounded-full overflow-hidden border border-border/40">
                <div
                  className={`h-full rounded-full ${cfg.bg} transition-all duration-300`}
                  style={{ width: `${pct}%` }}
                />
              </div>

              {/* Expandable Stage Details */}
              {isSelected && (
                <div className="mt-2 text-[11px] text-gray-400 bg-background/90 p-2.5 rounded border border-border/80 space-y-1 animate-fadeIn">
                  <p className="text-gray-300">{cfg.description}</p>
                  <div className="font-mono text-[10px] text-gray-500">
                    Stage ID: {timing.stage} • Duration: {timing.durationMs.toFixed(2)} ms
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
