import React, { useState, useRef, useEffect } from 'react';
import { Send, Zap, GitBranch, Layers, ShieldCheck, ChevronDown, ChevronRight, FileText } from 'lucide-react';
import { QueryResponse, SubgraphDTO, SourceChunk } from '../types';
import { TraceWaterfall } from './TraceWaterfall';

interface QueryStudioProps {
  onGraphUpdate: (graph: SubgraphDTO | null) => void;
  presetQuery?: string | null;
  onClearPreset?: () => void;
}

const PRESET_QUERIES = [
  'Compare the caching layers in Lattice RAG.',
  'How does LatticeDB handle vector search and graph relations together?',
  'What is the role of TypeSafe Jev Choice in query routing?',
  'Explain the multi-hop Cypher traversal process from entity anchors.',
  'What fallback mechanisms ensure zero-downtime during API outages?',
];

export const QueryStudio: React.FC<QueryStudioProps> = ({
  onGraphUpdate,
  presetQuery,
  onClearPreset,
}) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState('');
  const [route, setRoute] = useState<string | null>(null);
  const [cached, setCached] = useState<boolean | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [sources, setSources] = useState<SourceChunk[]>([]);
  const [timings, setTimings] = useState<QueryResponse['timings']>([]);
  const [expandedSource, setExpandedSource] = useState<number | null>(null);

  // Dynamic Token Velocity Meter State
  const [tokenCount, setTokenCount] = useState(0);
  const [tokenVelocity, setTokenVelocity] = useState<number | null>(null);
  const synthesisStartTimeRef = useRef<number | null>(null);
  const tokenCountRef = useRef(0);

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // When a preset query is passed from EvalMatrix
  useEffect(() => {
    if (presetQuery) {
      setQuery(presetQuery);
      if (onClearPreset) onClearPreset();
    }
  }, [presetQuery, onClearPreset]);

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setAnswer('');
    setRoute(null);
    setCached(null);
    setLatencyMs(null);
    setSources([]);
    setTimings([]);
    setTokenCount(0);
    setTokenVelocity(null);
    tokenCountRef.current = 0;
    synthesisStartTimeRef.current = null;

    try {
      const response = await fetch('/api/v1/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed, stream: true }),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine.startsWith('data:')) {
            const jsonStr = trimmedLine.replace(/^data:\s*/, '');
            if (!jsonStr) continue;

            try {
              const event = JSON.parse(jsonStr);
              if (event.event === 'token' && event.data?.delta) {
                if (!synthesisStartTimeRef.current) {
                  synthesisStartTimeRef.current = performance.now();
                }
                setAnswer((prev) => prev + event.data.delta);
                tokenCountRef.current += 1;
                setTokenCount(tokenCountRef.current);

                const elapsedSec = (performance.now() - synthesisStartTimeRef.current) / 1000;
                if (elapsedSec > 0.05) {
                  setTokenVelocity(parseFloat((tokenCountRef.current / elapsedSec).toFixed(1)));
                }
              } else if (event.event === 'stage') {
                if (event.data?.stage === 'triage' && event.data?.route) {
                  setRoute(event.data.route);
                }
              } else if (event.event === 'done') {
                const d = event.data;
                if (d.route) setRoute(d.route);
                if (d.cached !== undefined) setCached(d.cached);
                if (d.latencyMs !== undefined) setLatencyMs(d.latencyMs);
                if (d.sources) setSources(d.sources);
                if (d.timings) setTimings(d.timings);
                if (d.graphPath) onGraphUpdate(d.graphPath);
              }
            } catch (parseErr) {
              console.warn('Error parsing SSE event chunk:', parseErr, jsonStr);
            }
          }
        }
      }
    } catch (err: unknown) {
      console.error('Query execution error:', err);
      const msg = err instanceof Error ? err.message : String(err);
      setAnswer(`[Error]: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Query Input Section */}
      <div className="bg-card border border-border rounded-xl p-4 shadow-sm">
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="relative">
            <textarea
              ref={textareaRef}
              rows={3}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a technical question across your knowledge graph... (Press Enter to send)"
              className="w-full bg-background/80 border border-border rounded-lg p-3 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/40 resize-none font-sans"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-2.5 bottom-3.5 px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-800 disabled:text-gray-600 text-slate-950 font-semibold text-xs rounded-md flex items-center space-x-1.5 transition"
            >
              <span>{loading ? 'Synthesizing...' : 'Execute'}</span>
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Quick-Fill Presets */}
          <div className="flex flex-wrap gap-1.5 pt-1">
            <span className="text-[11px] text-gray-500 self-center mr-1">Presets:</span>
            {PRESET_QUERIES.map((q, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setQuery(q)}
                className="text-[11px] px-2 py-0.5 rounded bg-background/70 hover:bg-surface border border-border/80 text-gray-400 hover:text-gray-200 transition truncate max-w-[260px]"
                title={q}
              >
                {q}
              </button>
            ))}
          </div>
        </form>
      </div>

      {/* Real-time Telemetry & Dynamic Velocity Meter */}
      {(route || cached !== null || tokenVelocity !== null || loading) && (
        <div className="flex flex-wrap items-center justify-between gap-2 bg-card/60 border border-border px-3.5 py-2 rounded-lg text-xs">
          <div className="flex items-center space-x-3">
            {/* Jev Route Badge */}
            {route && (
              <div className="flex items-center space-x-1 text-cyan-400 font-mono text-[11px]">
                <GitBranch className="w-3 h-3 text-cyan-400" />
                <span className="font-semibold">{route}</span>
              </div>
            )}

            {/* Cache Status Badge */}
            {cached !== null && (
              <div
                className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium ${
                  cached
                    ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/40'
                    : 'bg-gray-800/60 text-gray-400 border border-gray-700/40'
                }`}
              >
                {cached ? '⚡ TIER-1 CACHE HIT' : 'CACHE MISS'}
              </div>
            )}
          </div>

          {/* Dynamic Token Velocity Speedometer */}
          <div className="flex items-center space-x-2 font-mono text-[11px]">
            <div className="flex items-center space-x-1 bg-background/80 px-2 py-0.5 rounded border border-border text-gray-300">
              <Zap className={`w-3 h-3 ${loading ? 'text-amber-400 animate-pulse' : 'text-cyan-400'}`} />
              <span className="text-gray-400">Velocity:</span>
              <span className="text-cyan-300 font-semibold">
                {tokenVelocity !== null ? `${tokenVelocity} tok/s` : loading ? 'measuring...' : 'idle'}
              </span>
            </div>
            {tokenCount > 0 && (
              <span className="text-gray-500 text-[10px]">({tokenCount} tokens)</span>
            )}
          </div>
        </div>
      )}

      {/* Response Card */}
      <div className="flex-1 bg-card border border-border rounded-xl p-4 overflow-y-auto space-y-3 min-h-[220px]">
        <div className="flex items-center justify-between pb-2 border-b border-border/50">
          <div className="flex items-center space-x-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-semibold text-gray-200">Grounded Synthesis</h3>
          </div>
          {latencyMs !== null && (
            <span className="text-[11px] font-mono text-gray-400">
              Total: <span className="text-cyan-400 font-semibold">{latencyMs.toFixed(1)}ms</span>
            </span>
          )}
        </div>

        {answer ? (
          <div className="prose prose-invert max-w-none text-sm leading-relaxed text-gray-200 font-sans whitespace-pre-wrap">
            {answer}
            {loading && <span className="inline-block w-2 h-4 bg-cyan-400 ml-1 animate-pulse" />}
          </div>
        ) : (
          <div className="h-40 flex flex-col items-center justify-center text-center text-gray-500 text-xs space-y-1">
            <p>Select a preset or ask a question to test the GraphRAG engine.</p>
            <p className="text-[11px] text-gray-600">
              Streams token-by-token with live velocity measurement and source chunk citations.
            </p>
          </div>
        )}
      </div>

      {/* Execution Trace Waterfall */}
      {timings && timings.length > 0 && (
        <TraceWaterfall timings={timings} totalLatencyMs={latencyMs || 0} />
      )}

      {/* Grounded Source Chunks Inspector */}
      {sources && sources.length > 0 && (
        <div className="bg-card border border-border rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between pb-1.5 border-b border-border/50">
            <div className="flex items-center space-x-1.5 text-xs font-semibold text-gray-200">
              <FileText className="w-3.5 h-3.5 text-cyan-400" />
              <span>Grounded Evidence ({sources.length} chunks)</span>
            </div>
            <span className="text-[10px] text-emerald-400 font-mono flex items-center space-x-1">
              <ShieldCheck className="w-3 h-3" />
              <span>Jev Guardrail Verified</span>
            </span>
          </div>

          <div className="space-y-1.5 max-h-48 overflow-y-auto">
            {sources.map((chunk, idx) => {
              const isExpanded = expandedSource === idx;
              return (
                <div
                  key={idx}
                  className="bg-background/70 border border-border/70 rounded-md p-2 hover:border-cyan-500/40 transition cursor-pointer text-xs"
                  onClick={() => setExpandedSource(isExpanded ? null : idx)}
                >
                  <div className="flex items-center justify-between text-[11px] font-mono text-gray-400 mb-1">
                    <div className="flex items-center space-x-1">
                      {isExpanded ? (
                        <ChevronDown className="w-3 h-3 text-cyan-400" />
                      ) : (
                        <ChevronRight className="w-3 h-3 text-gray-500" />
                      )}
                      <span className="text-cyan-400 font-medium">[Chunk {chunk.position}]</span>
                      <span className="text-gray-500">Doc: {chunk.documentId}</span>
                    </div>
                    <span className="text-emerald-400">Score: {chunk.score.toFixed(3)}</span>
                  </div>
                  <p
                    className={`text-gray-300 font-sans text-xs ${
                      isExpanded ? 'whitespace-pre-wrap' : 'line-clamp-2'
                    }`}
                  >
                    {chunk.text}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
